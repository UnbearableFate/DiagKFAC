from ..layers.eigen import *
from ..layers.modules import (
    get_cov,
    append_bias_ones,
    LinearModuleHelper,
    Conv2dModuleHelper,
)
from .norm_modules import LayerNormModuleHelper, BN2dModuleHelper
from .common import *


class DiaEigenLayer(KFACEigenLayer):
    """
    DiaEigenLayer is a class that implements the KFAC algorithm for diagonal
    approximation of the Fisher information matrix. It inherits from the
    KFACEigenLayer class and provides methods for computing the Fisher
    information matrix and its inverse using the diagonal approximation.
    """

    def __init__(
        self,
        module: ModuleHelper,
        name: str,
        tensor_parallel_dist_group: dist.ProcessGroup = None,
        chunk_rank: int = -1,
        group_size: int = -1,
        split_end: SplitEnd = SplitEnd.BOTH,
        *,
        tdc: TorchDistributedCommunicator,
        allreduce_method: AllreduceMethod = AllreduceMethod.ALLREDUCE,
        a_inv_gathered=None,
        g_inv_gathered=None,
        factor_dtype: torch.dtype | None = None,
        grad_scaler: torch.cuda.amp.GradScaler | Callable[[], float] | None = None,
        inv_dtype: torch.dtype = torch.float32,
        symmetry_aware: bool = False,
        prediv_eigenvalues: bool = False,
    ) -> None:
        """
        Initializes the MiniEigen class.

        Args:
            module (ModuleHelper): The module to be optimized.
            tdc (TorchDistributedCommunicator): The communicator for distributed training.
            allreduce_method (AllreduceMethod, optional): The method for all-reduce operation. Defaults to AllreduceMethod.ALLREDUCE.
            factor_dtype (torch.dtype, optional): The data type for factors. Defaults to None.
            grad_scaler (torch.cuda.amp.GradScaler | Callable[[], float] | None, optional): The gradient scaler. Defaults to None.
            inv_dtype (torch.dtype, optional): The data type for inverse computation. Defaults to torch.float32.
            symmetry_aware (bool, optional): Whether to use symmetry-aware optimization. Defaults to False.
            prediv_eigenvalues (bool, optional): Whether to pre-divide eigenvalues. Defaults to False.
        """
        super().__init__(
            module=module,
            factor_dtype=factor_dtype,
            grad_scaler=grad_scaler,
            inv_dtype=inv_dtype,
            symmetry_aware=symmetry_aware,
            prediv_eigenvalues=False,
            tdc=tdc,
            allreduce_method=allreduce_method,
            name=name,
        )

        self.g_inv_gathered = g_inv_gathered
        self.a_inv_gathered = a_inv_gathered

        self.a_inv_local = None
        self.g_inv_local = None

        self.tp_group = tensor_parallel_dist_group
        if self.tp_group is not None and chunk_rank == -1:
            chunk_rank = dist.get_rank(self.tp_group)
            group_size = dist.get_world_size(self.tp_group)

        self.name = name + f"_chunk{chunk_rank}"
        self.chunk_rank = chunk_rank
        self.grp_size = group_size

        self.split_in = True if split_end in [SplitEnd.IN, SplitEnd.BOTH] else False
        self.split_out = True if split_end in [SplitEnd.OUT, SplitEnd.BOTH] else False
        self.a_factor_split_start_end = []
        self.g_factor_split_start_end = []
        self.compute_chunk_setting(chunk_rank=chunk_rank, group_size=group_size)

        """
        print(
            f"DiaEigen {self.name} initialized with split_end={split_end}, A shape {self.module.a_factor_shape}, G shape {self.module.g_factor_shape}",
            f"input chunk start={self.input_chunk_start}, end={self.input_chunk_end}",
            f"output chunk start={self.output_chunk_start}, end={self.output_chunk_end}",
            f"a_factor_width={self.a_factor_width}, g_factor_width={self.g_factor_width}",
            f"a factor start end {self.a_factor_split_start_end}, g factor start end {self.g_factor_split_start_end}",
        )
        """

    @property
    def a_inv_local(self) -> torch.Tensor | None:
        """Get eigen vectors of A."""
        if isinstance(self._a_inv_local, Future):
            self._a_inv_local = cast(torch.Tensor, self._a_inv_local.wait())
        return self._a_inv_local

    @a_inv_local.setter
    def a_inv_local(self, value: torch.Tensor | FutureType | None) -> None:
        """Set eigen vectors of A."""
        self._a_inv_local = value

    @property
    def g_inv_local(self) -> torch.Tensor | None:
        """Get eigen vectors of G."""
        if isinstance(self._g_inv_local, Future):
            self._g_inv_local = cast(torch.Tensor, self._g_inv_local.wait())
        return self._g_inv_local

    @g_inv_local.setter
    def g_inv_local(self, value: torch.Tensor | FutureType | None) -> None:
        """Set eigen vectors of G."""
        self._g_inv_local = value

    def has_a_block(self) -> bool:
        """Check if the layer has an A block."""
        return (
            self.split_in
            and self.a_factor_width > 0
            and self.input_chunk_end - self.input_chunk_start > 0
        )

    def has_g_block(self) -> bool:
        """Check if the layer has a G block."""
        return (
            self.split_out
            and self.g_factor_width > 0
            and self.output_chunk_end - self.output_chunk_start > 0
        )

    def compute_chunk_setting(self, chunk_rank: int, group_size: int = 2) -> None:
        """
        Compute the chunk start/end indices and allocate eigendecomposition buffers
        according to the split_end configuration.
        """
        # Special-case for Norm layers (LayerNorm / BatchNorm2d):
        # For these, the activation-side factor A is always 2x2 ([xhat, 1]),
        # so there is no point (and no correctness) in splitting the input (A).
        # We therefore force no input split, while still allowing output (G) split
        # across channels/parameter dimension.
        if isinstance(self.module, (LayerNormModuleHelper, BN2dModuleHelper)):
            device = self.module.module.weight.device
            dtype = self.inv_dtype

            # A factor: fixed width = 2
            self.in_features = 2
            self.input_chunk_start, self.input_chunk_end = 0, 2
            self.a_factor_width = 2

            # Force-disable input splitting for Norm layers
            self.split_in = False

            # G factor: width equals number of parameters (channels) of the affine part
            out_features = int(self.module.module.weight.numel())
            self.out_features = out_features

            if self.split_out:
                self._setup_output_chunk(
                    chunk_rank, group_size, out_features, device, dtype
                )
            else:
                self.output_chunk_start, self.output_chunk_end = 0, out_features
                self.g_factor_width = self.module.g_factor_shape[0]
            return

        in_features = self.module.module.weight.shape[1]
        out_features = self.module.module.weight.shape[0]
        self.in_features = in_features
        self.out_features = out_features
        device = self.module.module.weight.device
        dtype = self.inv_dtype

        if self.split_in:
            self._setup_input_chunk(chunk_rank, group_size, in_features, device, dtype)
            # assert self.in_features == self.a_factor_width * self.grp_size , \
            #     f"TODO: Input features {self.in_features} must be divisible by a_factor_width {self.a_factor_width}."
        else:
            self.input_chunk_start, self.input_chunk_end = 0, in_features
            self.a_factor_width = self.module.a_factor_shape[0]

        if self.split_out:
            self._setup_output_chunk(
                chunk_rank, group_size, out_features, device, dtype
            )
            # assert self.out_features == self.g_factor_width * self.grp_size, \
            #     f"TODO: Output features {self.out_features} must be divisible by g_factor_width {self.g_factor_width}."
        else:
            self.output_chunk_start, self.output_chunk_end = 0, out_features
            self.g_factor_width = self.module.g_factor_shape[0]

    def _setup_input_chunk(self, chunk_rank, group_size, in_features, device, dtype):
        # 计算所有chunk的a_factor_width
        all_a_factor_widths = []
        a_factor_split_start_end = []
        for rank in range(group_size):
            start, end = self.get_chunk_start_end(rank, group_size, in_features)

            if isinstance(self.module, LinearModuleHelper):
                if rank == group_size - 1 and self.module.has_bias():
                    a_factor_width = end - start + 1
                else:
                    a_factor_width = end - start
            elif isinstance(self.module, Conv2dModuleHelper):
                ksize0: int = self.module.module.kernel_size[0]  # kernel height
                ksize1: int = self.module.module.kernel_size[1]  # kernel width
                in_ch: int = end - start
                a_factor_width = in_ch * ksize0 * ksize1
                if rank == group_size - 1 and self.module.has_bias():
                    a_factor_width += 1
            else:
                raise ValueError(f"Unsupported module type: {type(self.module)}")

            all_a_factor_widths.append(a_factor_width)
            if rank == chunk_rank:
                self.input_chunk_start, self.input_chunk_end = start, end
                self.a_factor_width = a_factor_width

            if rank == 0:
                a_factor_split_start_end.append((0, a_factor_width))
            else:
                a_factor_split_start_end.append(
                    (
                        a_factor_split_start_end[-1][1],
                        a_factor_split_start_end[-1][1] + a_factor_width,
                    )
                )

        self.a_factor_split_start_end = a_factor_split_start_end
        assert (
            self.a_factor_split_start_end[-1][1] == self.module.a_factor_shape[0]
        ), f"Expected {self.module.a_factor_shape[0]}, but got {self.a_factor_split_start_end[-1][1]}"

        # 初始化a_inv_gathered
        if self.a_inv_gathered is None:
            self.a_inv_gathered = []
            for width in all_a_factor_widths:
                self.a_inv_gathered.append(
                    torch.empty((width, width), dtype=dtype, device=device)
                )

    def _setup_output_chunk(self, chunk_rank, group_size, out_features, device, dtype):
        # 计算所有chunk的g_factor_width
        all_g_factor_widths = []
        g_factor_split_start_end = []
        for rank in range(group_size):
            start, end = self.get_chunk_start_end(rank, group_size, out_features)
            g_factor_width = end - start
            all_g_factor_widths.append(g_factor_width)
            if rank == chunk_rank:
                self.output_chunk_start, self.output_chunk_end = start, end
                self.g_factor_width = g_factor_width

            if rank == 0:
                g_factor_split_start_end.append((0, g_factor_width))
            else:
                g_factor_split_start_end.append(
                    (
                        g_factor_split_start_end[-1][1],
                        g_factor_split_start_end[-1][1] + g_factor_width,
                    )
                )

        self.g_factor_split_start_end = g_factor_split_start_end
        assert (
            self.g_factor_split_start_end[-1][1] == self.module.g_factor_shape[0]
        ), f"Expected {self.module.g_factor_shape[0]}, but got {self.g_factor_split_start_end[-1][1]}"

        # 初始化g_inv_gathered
        if self.g_inv_gathered is None:
            self.g_inv_gathered = []
            for width in all_g_factor_widths:
                self.g_inv_gathered.append(
                    torch.empty((width, width), dtype=dtype, device=device)
                )

    def get_chunk_start_end(
        self, chunk_rank, group_size, feature_num: int
    ) -> tuple[int, int]:
        """Get the start and end indices for the current rank's chunk of features."""
        """
        compute the start and end index of the chunk for the current rank
        Args:
            chunk_rank (int): The rank of the current chunk.
            group_size (int): The total number of chunks.
            feature_num (int): The total number of features.
        Returns:
            tuple[int, int]: The start and end indices of the chunk.
        """
        # 计算每个 rank 分得的特征宽度
        chunk_size = round(feature_num / group_size)
        start = chunk_rank * chunk_size
        # 最后一个 rank 把多出来的维度也一起拿进去
        end = (
            (chunk_rank + 1) * chunk_size
            if chunk_rank < group_size - 1
            else feature_num
        )
        return start, end

    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        if not self.split_in:
            super().save_layer_input(input_)
            return
        """Save input for layer, but only use the slice corresponding to this rank."""
        if self.input_chunk_start == self.input_chunk_end:
            self._a_batch = None
            return  # No input to save for this rank

        if isinstance(self.module, LinearModuleHelper):
            a = input_[0][..., self.input_chunk_start : self.input_chunk_end]
            # 避免不必要的拷贝：只有在 dtype 不匹配时才转换
            if a.dtype != self.factor_dtype:
                a = a.to(self.factor_dtype)
            # 使用 reshape 更鲁棒，必要时内部才会拷贝；若可视图则零拷贝
            a = a.reshape(-1, a.size(-1))
            if self.module.has_bias() and self.chunk_rank == self.grp_size - 1:
                a = append_bias_ones(a)
            a = get_cov(a)
        elif isinstance(self.module, Conv2dModuleHelper):
            x = input_[0][:, self.input_chunk_start : self.input_chunk_end, ...]
            # 仅在需要时转换 dtype
            if x.dtype != self.factor_dtype:
                x = x.to(self.factor_dtype)
            # 若提取 kernel 需要连续内存，则按需 contiguous
            if not x.is_contiguous():
                x = x.contiguous()
            # 提取patches
            a = self.module._extract_patches(x)
            spatial_size = a.size(1) * a.size(2)
            # 使用 reshape 以避免强制要求连续
            a = a.reshape(-1, a.size(-1))
            if self.module.has_bias() and self.chunk_rank == self.grp_size - 1:
                a = append_bias_ones(a)
            a = a / spatial_size
            a = get_cov(a)
        else:
            raise ValueError(f"Unsupported module type: {type(self.module)}")

        # 累加到全局统计里
        if self._a_batch is None:
            self._a_batch = a
            self._a_count = 1
        else:
            self._a_batch = self._a_batch + a
            self._a_count += 1

    def save_layer_grad_output(self, grad_output: tuple[torch.Tensor, ...]) -> None:
        if not self.split_out:
            super().save_layer_grad_output(grad_output)
            return
        if self.output_chunk_start == self.output_chunk_end:
            self._g_batch = None
            return

        if isinstance(self.module, LinearModuleHelper):
            # 线性层分块
            g = grad_output[0][..., self.output_chunk_start : self.output_chunk_end].to(
                self.factor_dtype
            )
            g = g.view(-1, g.size(-1))
            if self.grad_scaler is not None:
                g = g / self.grad_scaler()
            g = get_cov(g)
        elif isinstance(self.module, Conv2dModuleHelper):
            # 卷积层分块（输出通道分块）
            g = grad_output[0][
                :, self.output_chunk_start : self.output_chunk_end, ...
            ].to(self.factor_dtype)
            spatial_size = g.size(2) * g.size(3)
            g = g.transpose(1, 2).transpose(2, 3)
            g = g.reshape(-1, g.size(-1))
            if self.grad_scaler is not None:
                g = g / self.grad_scaler()
            g = g / spatial_size
            g = get_cov(g)
        elif isinstance(self.module, LayerNormModuleHelper):
            # LayerNorm output-chunked G factor
            gy = grad_output[0]
            # Keep batch dimension as samples; sum over non-parameter, non-batch axes
            norm_shape = getattr(self.module.module, "normalized_shape", None)
            if isinstance(norm_shape, int):
                norm_ndims = 1
            else:
                norm_ndims = len(norm_shape)
            param_dims = tuple(range(gy.ndim - norm_ndims, gy.ndim))
            reduce_dims = tuple(
                d for d in range(gy.ndim) if d not in param_dims and d != 0
            )
            # Aggregate over reduce dims -> [N, *normalized_shape]
            gparam = gy.sum(dim=reduce_dims)
            N = gparam.shape[0]
            D = (
                int(torch.tensor(norm_shape).prod().item())
                if not isinstance(norm_shape, int)
                else norm_shape
            )
            gparam = gparam.reshape(N, D)
            # Slice parameter (column) chunk
            g = gparam[:, self.output_chunk_start : self.output_chunk_end].to(
                self.factor_dtype
            )
            if self.grad_scaler is not None:
                g = g / self.grad_scaler()
            g = get_cov(g)
        elif isinstance(self.module, BN2dModuleHelper):
            # BatchNorm2d output-chunked G factor: sum over spatial, keep batch as samples
            gy = grad_output[0]
            gparam = gy[:, self.output_chunk_start : self.output_chunk_end, ...].to(
                self.factor_dtype
            )
            gparam = gparam.sum(dim=(2, 3))  # [N, C_chunk]
            if self.grad_scaler is not None:
                gparam = gparam / self.grad_scaler()
            g = get_cov(gparam)
        else:
            raise ValueError(f"Unsupported module type: {type(self.module)}")

        if self._g_batch is None:
            self._g_batch = g
            self._g_count = 1
        else:
            self._g_batch = self._g_batch + g
            self._g_count += 1

    def all_gather_a_inv_tensors(self) -> None:
        """
        Gather per-rank computed qa and da tensors across tensor-parallel group
        and assemble into self.qa_gathered and self.da_gathered.
        """
        assert self.a_inv_gathered is not None, "a_inv_gathered must be initialized."
        assert self.a_inv_local is not None, "a_inv_local must be initialized."
        if self.module.has_bias():
            dist.all_gather(self.a_inv_gathered, self.a_inv_local, group=self.tp_group)
        else:
            dist.all_gather_into_tensor(
                self.a_inv_gathered, self.a_inv_local, group=self.tp_group
            )

    def all_gather_g_inv_tensors(self) -> None:
        """
        Gather per-rank computed qg and dg tensors across tensor-parallel group
        and assemble into self.qg_gathered and self.dg_gathered.
        """
        assert self.g_inv_gathered is not None, "g_inv_gathered must be initialized."
        assert self.g_inv_local is not None, "g_inv_local must be initialized."
        assert (
            self.g_inv_gathered.shape[0] == self.g_inv_local.shape[0] * self.grp_size
        ), f"g_inv_gathered.shape[0] {self.g_inv_gathered.shape[0]} must equal to g_inv_local.shape[0] {self.g_inv_local.shape[0]} * grp_size {self.grp_size}."

        dist.all_gather_into_tensor(
            self.g_inv_gathered, self.g_inv_local, group=self.tp_group
        )

    def compute_a_inv(self, damping: float = 0.001) -> None:
        if self.a_factor_width == 0:
            if self.a_inv_local is None:
                self.a_inv_local = torch.empty((0, 0)).to(
                    device=self.module.module.weight.device, dtype=self.inv_dtype
                )
            return
        super().compute_a_inv(damping=damping)
        # Gather each block's qa and da into self.qa_gathered and self.da_gathered
        inv_vals = 1.0 / (self.da + damping)  # (k,)
        F = self.qa.clone().mul_(
            inv_vals.unsqueeze(0)
        )  # 先克隆，再对每列 in-place 缩放
        self.a_inv_local = torch.mm(F, self.qa.t())  # 整体乘一次
        self.qa = None
        self.da = None
        # self.a_inv_local = MinMaxNormalization(self.a_inv_local)
        if self.split_in and self.tp_group is not None:
            self.all_gather_a_inv_tensors()

    def compute_g_inv(self, damping: float = 0.001) -> None:
        if self.g_factor_width == 0:
            if self.g_inv_local is None:
                self.g_inv_local = torch.empty((0, 0)).to(
                    device=self.module.module.weight.device, dtype=self.inv_dtype
                )
            return
        super().compute_g_inv(damping=damping)
        # Gather each block's qg and dg into self.qg_gathered and self.dg_gathered
        inv_vals_g = 1.0 / (self.dg + damping)  # dg 已经>0
        Qg_scaled = self.qg * inv_vals_g.unsqueeze(0)
        self.g_inv_local = Qg_scaled @ self.qg.t()
        self.qg = None
        self.dg = None
        # self.g_inv_local = MinMaxNormalization(self.g_inv_local)
        if self.split_out and self.tp_group is not None:
            self.all_gather_g_inv_tensors()

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        raise NotImplementedError("preconditioned_grad is not implemented")
        """Compute precondition gradient of each weight in module.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        if self.a_inv_local is None or self.g_inv_local is None:
            return

        '''
        def get_grad(modu) -> torch.Tensor:
            """Get formatted gradients (weight and bias) of module.

            Returns:
                gradient of shape If bias != None,
                concats bias.
            """
            g = cast(torch.Tensor, modu.module.weight.main_grad)
            if g is None:
                raise RuntimeError(
                    f"MiniEigen {self.name} get_grad, module weight grad is None, "
                    "make sure to call backward() before this method."
                )
            if modu.has_bias():
                g = torch.cat([g, modu.module.bias.main_grad.view(-1, 1)], 1)  # type: ignore
            return g
        '''

        grad = self.module.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.a_inv_local.dtype)
        # self.grad = (self.g_inv @ grad @ self.a_inv).to(grad_type)
        if self.split_in:
            if self.split_out:
                # Block-diagonal left and right multiplication
                if not self.module.has_bias():
                    self.grad = block_diag_left_and_right(
                        A_blocks=self.g_inv_gathered,
                        B_blocks=grad,
                        C_blocks=self.a_inv_gathered,
                        is_a_transpose=False,
                        is_c_transpose=False,
                    )
                else:
                    self.grad = block_diag_right_mm_with_bias(
                        block_diag_left_mm(self.g_inv_gathered, grad),
                        self.a_inv_gathered,
                    )
            else:
                # Block-diagonal left multiplication
                self.grad = block_diag_right_mm(
                    self.g_inv_local @ grad, self.a_inv_gathered
                )
        elif self.split_out:
            # Block-diagonal right multiplication
            self.grad = block_diag_left_mm(self.g_inv_gathered, grad) @ self.a_inv_local
        else:
            # Full matrix multiplication
            self.grad = self.g_inv_local @ grad @ self.a_inv_local
            self.grad = self.grad.to(grad_type)
            return
        # self.grad.mul_(0.5).add_(grad, alpha=0.5).to(dtype=grad_type)  # Add damping to the gradient
        self.grad.add_(grad).mul_(0.5).to(
            dtype=grad_type
        )  # Add damping to the gradient

    def broadcast_a_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        if not self.split_in:
            raise RuntimeError(
                f"Don't use broadcast A inv from dia_eigen when split_in is False",
            )

        if self.a_inv_local is None:
            if get_rank() == src:
                raise RuntimeError(
                    f"Attempt to broadcast A inv from src={src} but this rank "
                    "has not computed A inv yet.",
                )
            self.a_inv_local = torch.empty(
                (self.a_factor_width, self.a_factor_width),
                device=self.module.device,
                dtype=self.inv_dtype,
            )

        self.a_inv_local = self.tdc.broadcast(  # type: ignore
            self.a_inv_local,
            src=src,
            group=group,
        )

    def broadcast_g_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """Initiate G inv broadcast and store future to result.

        Note:
            all ranks must enter this function even if the rank is not
            a part of the inverse broadcast group.

        Args:
            src (int): src rank that computed G inverse.
            group (ProcessGroup): process group to which src should broadcast
                G inv. All ranks in group should enter this function.
                Defaults to None, the default process group.
        """
        if not self.split_out:
            raise RuntimeError(
                f"Don't use broadcast G inv from dia_eigen when split_out is False",
            )
        if self.g_inv_local is None:
            if get_rank() == src:
                raise RuntimeError(
                    f"Attempt to broadcast G inv from src={src} but this rank "
                    "has not computed G inv yet.",
                )
            self.g_inv_local = torch.empty(
                (self.g_factor_width, self.g_factor_width),
                device=self.module.device,
                dtype=self.inv_dtype,
            )

        self.g_inv_local = self.tdc.broadcast(  # type: ignore
            self.g_inv_local,
            src=src,
            group=group,
        )
