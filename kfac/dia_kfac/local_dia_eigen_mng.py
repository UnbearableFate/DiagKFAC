from ..layers.eigen import *
from .dia_eigen import DiaEigenLayer
from typing import List
from .common import *


class LocalDiaEigenLayerManager(KFACEigenLayer):
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
        split_num: int,
        split_end: SplitEnd = SplitEnd.BOTH,
        *,
        tdc: TorchDistributedCommunicator,
        allreduce_method: AllreduceMethod = AllreduceMethod.ALLREDUCE,
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

        self.sub_layers: List[DiaEigenLayer] = []
        self.split_in = split_end in [SplitEnd.IN, SplitEnd.BOTH]
        self.split_out = split_end in [SplitEnd.OUT, SplitEnd.BOTH]

        if self.split_in and self.module.a_factor_shape[0] // split_num < 8:
            print(
                f"Disabling split_in at layer {self.name} because a factor shape {self.module.a_factor_shape[0]} // {split_num} < 8"
            )
            self.split_in = False
        if self.split_out and self.module.g_factor_shape[0] // split_num < 8:
            print(
                f"Disabling split_out at layer {self.name} because g factor shape {self.module.g_factor_shape[0]} // {split_num} < 8"
            )
            self.split_out = False

        sub_split_end = split_end
        if self.split_in:
            if self.split_out:
                sub_split_end = SplitEnd.BOTH
            else:
                sub_split_end = SplitEnd.IN
        elif self.split_out:
            sub_split_end = SplitEnd.OUT
        else:
            sub_split_end = SplitEnd.NONE

        self._a_inv_local: torch.Tensor | FutureType | None = None
        self._g_inv_local: torch.Tensor | FutureType | None = None

        self._a_inv_local_flattened: torch.Tensor | FutureType | None = None
        self._g_inv_local_flattened: torch.Tensor | FutureType | None = None

        self.sub_a_inv_width_list = []
        self.sub_g_inv_width_list = []

        if sub_split_end == SplitEnd.NONE:
            return

        a_inv_flatten_len = 0
        g_inv_flatten_len = 0

        for i in range(split_num):
            sub_layer = DiaEigenLayer(
                module=module,
                name=name,
                chunk_rank=i,
                group_size=split_num,
                split_end=sub_split_end,
                factor_dtype=factor_dtype,
                grad_scaler=grad_scaler,
                inv_dtype=inv_dtype,
                symmetry_aware=symmetry_aware,
                prediv_eigenvalues=prediv_eigenvalues,
                a_inv_gathered=1,
                g_inv_gathered=1,
                tdc=tdc,
                allreduce_method=allreduce_method,
            )
            self.sub_layers.append(sub_layer)
            if self.split_in:
                a_inv_flatten_len += (
                    sub_layer.a_factor_width * (sub_layer.a_factor_width + 1) / 2
                )
            if self.split_out:
                g_inv_flatten_len += (
                    sub_layer.g_factor_width * (sub_layer.g_factor_width + 1) / 2
                )
        if self.split_in:
            self.a_inv_local_flattened = torch.empty(
                int(a_inv_flatten_len), dtype=self.inv_dtype, device=self.module.device
            )
            print(
                f"{self.name} a_inv_local_flattened shape: {self.a_inv_local_flattened.shape}"
            )
        if self.split_out:
            self.g_inv_local_flattened = torch.empty(
                int(g_inv_flatten_len), dtype=self.inv_dtype, device=self.module.device
            )
            print(
                f"{self.name} g_inv_local_flattened shape: {self.g_inv_local_flattened.shape}"
            )

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

    @property
    def a_inv_local_flattened(self) -> torch.Tensor | None:
        """Get flattened eigen vectors of A."""
        if isinstance(self._a_inv_local_flattened, Future):
            val = self._a_inv_local_flattened.wait()
            # torch.distributed.Work.get_future().wait() may return [Tensor]
            if isinstance(val, list):
                if len(val) == 1 and isinstance(val[0], torch.Tensor):
                    val = val[0]
                else:
                    raise TypeError(
                        f"Expected Future to resolve to a single Tensor or [Tensor], got list of len {len(val)}"
                    )
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"a_inv_local_flattened Future resolved to {type(val)}, expected Tensor"
                )
            if val.ndim != 1:
                # 保守处理：拉平成 1D
                val = val.reshape(-1)
            self._a_inv_local_flattened = val
        return self._a_inv_local_flattened

    @a_inv_local_flattened.setter
    def a_inv_local_flattened(self, value: torch.Tensor | FutureType | None) -> None:
        # 类型与形状检查：仅允许 None / Tensor(1D) / Future
        if value is None:
            self._a_inv_local_flattened = None
            return
        # 允许 torch Tensor（必须为 1D）
        if isinstance(value, torch.Tensor):
            if value.ndim != 1:
                raise ValueError(
                    f"a_inv_local_flattened must be 1D Tensor, got shape {tuple(value.shape)}"
                )
            self._a_inv_local_flattened = value
            return
        # 允许 Future（延迟张量）
        if isinstance(value, Future):
            self._a_inv_local_flattened = value
            return
        # 其它类型一律拒绝（例如 list/np.ndarray 等）
        raise TypeError(
            f"a_inv_local_flattened must be Tensor | Future | None, got {type(value)}"
        )

    @property
    def g_inv_local_flattened(self) -> torch.Tensor | None:
        """Get flattened eigen vectors of G."""
        if isinstance(self._g_inv_local_flattened, Future):
            val = self._g_inv_local_flattened.wait()
            if isinstance(val, list):
                if len(val) == 1 and isinstance(val[0], torch.Tensor):
                    val = val[0]
                else:
                    raise TypeError(
                        f"Expected Future to resolve to a single Tensor or [Tensor], got list of len {len(val)}"
                    )
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"g_inv_local_flattened Future resolved to {type(val)}, expected Tensor"
                )
            if val.ndim != 1:
                val = val.reshape(-1)
            self._g_inv_local_flattened = val
        return self._g_inv_local_flattened

    @g_inv_local_flattened.setter
    def g_inv_local_flattened(self, value: torch.Tensor | FutureType | None) -> None:
        # 类型与形状检查：仅允许 None / Tensor(1D) / Future
        if value is None:
            self._g_inv_local_flattened = None
            return
        if isinstance(value, torch.Tensor):
            if value.ndim != 1:
                raise ValueError(
                    f"g_inv_local_flattened must be 1D Tensor, got shape {tuple(value.shape)}"
                )
            self._g_inv_local_flattened = value
            return
        if isinstance(value, Future):
            self._g_inv_local_flattened = value
            return
        raise TypeError(
            f"g_inv_local_flattened must be Tensor | Future | None, got {type(value)}"
        )

    """
    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        if not self.split_in:
            super().save_layer_input(input_)
            return
        '''Save input for layer, but only use the slice corresponding to this rank.'''

        for sub_layer in self.sub_layers:
            sub_layer.save_layer_input(input_)

    def save_layer_grad_output(self, grad_output: tuple[torch.Tensor, ...]) -> None:
        if not self.split_out:
            super().save_layer_grad_output(grad_output)
            return

        for sub_layer in self.sub_layers:
            sub_layer.save_layer_grad_output(grad_output)

    def update_a_factor(self, alpha: float = 0.95) -> None:
        if not self.split_in:
            super().update_a_factor(alpha=alpha)
            return
        for sub_layer in self.sub_layers:
            sub_layer.update_a_factor(alpha=alpha)
    

    def update_g_factor(self, alpha: float = 0.95) -> None:
        if not self.split_out:
            super().update_g_factor(alpha=alpha)
            return
        for sub_layer in self.sub_layers:
            sub_layer.update_g_factor(alpha=alpha)

    def reduce_a_factor(self, group=None):
        if not self.split_in:
            super().reduce_a_factor(group=group)
            return
        for sub_layer in self.sub_layers:
            if sub_layer.a_factor_width > 0:
                sub_layer.reduce_a_factor(group=group)

    def reduce_g_factor(self, group=None):
        if not self.split_out:
            super().reduce_g_factor(group=group)
            return
        for sub_layer in self.sub_layers:
            if sub_layer.g_factor_width > 0:
                sub_layer.reduce_g_factor(group=group)
    """

    def broadcast_a_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        if not self.split_in:
            if self.a_inv_local is None:
                if get_rank() == src:
                    raise RuntimeError(
                        f"Attempt to broadcast A inv from src={src} but this rank "
                        "has not computed A inv yet.",
                    )
                self.a_inv_local = torch.empty(
                    self.module.a_factor_shape,
                    device=self.module.device,
                    dtype=self.inv_dtype,
                )

            self.a_inv_local = self.tdc.broadcast(  # type: ignore
                self.a_inv_local,
                src=src,
                group=group,
            )
            return

        # split_in == True: flatten per-sub-layer A^{-1} (upper-tri only) and broadcast
        if get_rank() == src:
            a_inv_local_flattened, _ = flatten_sym_blocks_upper(
                [
                    sub_layer.a_inv_local
                    for sub_layer in self.sub_layers
                    if sub_layer.has_a_block()
                ]
            )
            # debug/shape guard
            assert self.a_inv_local_flattened.shape == a_inv_local_flattened.shape, (
                f"{self.name} a_inv_local_flattened shape mismatch, expected "
                f"{a_inv_local_flattened.shape}, got {self.a_inv_local_flattened.shape}"
            )
            self.a_inv_local_flattened = a_inv_local_flattened

        future = dist.broadcast(
            self.a_inv_local_flattened,
            src=src,
            group=group,
            async_op=True,
        ).get_future()

        self.a_inv_local_flattened = future
        return

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
            if self.g_inv_local is None:
                if get_rank() == src:
                    raise RuntimeError(
                        f"Attempt to broadcast G inv from src={src} but this rank "
                        "has not computed G inv yet.",
                    )
                self.g_inv_local = torch.empty(
                    self.module.g_factor_shape,
                    device=self.module.device,
                    dtype=self.inv_dtype,
                )

            self.g_inv_local = self.tdc.broadcast(  # type: ignore
                self.g_inv_local,
                src=src,
                group=group,
            )
            return

        if get_rank() == src:
            g_inv_local_flattened, _ = flatten_sym_blocks_upper(
                [
                    sub_layer.g_inv_local
                    for sub_layer in self.sub_layers
                    if sub_layer.has_g_block()
                ]
            )
            ## debug
            assert (
                self.g_inv_local_flattened.shape == g_inv_local_flattened.shape
            ), f"{self.name} g_inv_local_flattened shape mismatch, expected {g_inv_local_flattened.shape}, got {self.g_inv_local_flattened.shape}"
            self.g_inv_local_flattened = g_inv_local_flattened

        future = dist.broadcast(
            self.g_inv_local_flattened,
            src=src,
            group=group,
            async_op=True,
        ).get_future()

        self.g_inv_local_flattened = future

    def assemble_a_inv_from_flattened(self, damping) -> None:
        """反解 `a_inv_local_flattened` 为各子块并 block_diag 回 `self.a_inv_local`。

        仅在 split_in 为 True 时有意义。若 `a_inv_local_flattened` 仍是 Future，将在此处等待。
        """
        if not self.split_in:
            return

        sizes = [
            sub_layer.a_factor_width
            for sub_layer in self.sub_layers
            if sub_layer.has_a_block()
        ]
        if len(sizes) == 0:
            raise ValueError("No valid sizes found for unflattening.")
        # 反解各块（保持 inv_dtype / module.device）
        blocks = unflatten_sym_blocks_upper(
            self.a_inv_local_flattened,
            sizes,
            dtype=self.inv_dtype,
            device=self.module.device,
        )

        self.a_inv_local = torch.block_diag(*blocks).add_(damping)

    def assemble_g_inv_from_flattened(self, damping) -> None:
        """反解 `g_inv_local_flattened` 为各子块并 block_diag 回 `self.g_inv_local`。

        仅在 split_out 为 True 时有意义。若 `g_inv_local_flattened` 仍是 Future，将在此处等待。
        """
        if not self.split_out:
            return

        sizes = [
            sub_layer.g_factor_width
            for sub_layer in self.sub_layers
            if sub_layer.has_g_block()
        ]
        if len(sizes) == 0:
            raise ValueError("No valid sizes found for unflattening.")

        blocks = unflatten_sym_blocks_upper(
            self.g_inv_local_flattened,
            sizes,
            dtype=self.inv_dtype,
            device=self.module.device,
        )

        self.g_inv_local = torch.block_diag(*blocks).add_(damping)

    def compute_a_inv(self, damping: float = 0.001) -> None:
        if not self.split_in:
            super().compute_a_inv(damping=damping)
            # Gather each block's qa and da into self.qa_gathered and self.da_gathered
            inv_vals = 1.0 / (self.da + damping)  # (k,)
            F = self.qa.clone().mul_(
                inv_vals.unsqueeze(0)
            )  # 先克隆，再对每列 in-place 缩放
            self.a_inv_local = torch.mm(F, self.qa.t())  # 整体乘一次
            self.qa = None
            self.da = None
        else:
            for sub_layer in self.sub_layers:
                start = sub_layer.a_factor_split_start_end[sub_layer.chunk_rank][0]
                end = sub_layer.a_factor_split_start_end[sub_layer.chunk_rank][1]
                sub_layer.a_factor = self.a_factor[start:end, start:end]
                assert sub_layer.a_factor.shape == (
                    sub_layer.a_factor_width,
                    sub_layer.a_factor_width,
                ), f"a_factor must be square {sub_layer.a_factor.shape}, {sub_layer.a_factor_width}"
                sub_layer.compute_a_inv(damping=damping)

            """
            self.a_inv_local = torch.block_diag(
                *[sub_layer.a_inv_local for sub_layer in self.sub_layers]
            )
            self.a_inv_local.add_(damping)
            assert (
                self.a_inv_local.shape[0] == self.module.get_grad().shape[1]
            ), "a_inv_local must match module output size"
            assert (
                self.a_inv_local.shape == self.module.a_factor_shape
            ), "a_inv_local must match module weights size"
            """

    def compute_g_inv(self, damping: float = 0.001) -> None:
        if not self.split_out:
            super().compute_g_inv(damping=damping)
            # Gather each block's qg and dg into self.qg_gathered and self.dg_gathered
            inv_vals_g = 1.0 / (self.dg + damping)  # dg 已经>0
            Qg_scaled = self.qg * inv_vals_g.unsqueeze(0)
            self.g_inv_local = Qg_scaled @ self.qg.t()
            self.qg = None
            self.dg = None
        else:
            for sub_layer in self.sub_layers:
                start = sub_layer.g_factor_split_start_end[sub_layer.chunk_rank][0]
                end = sub_layer.g_factor_split_start_end[sub_layer.chunk_rank][1]
                sub_layer.g_factor = self.g_factor[start:end, start:end]
                assert sub_layer.g_factor.shape == (
                    sub_layer.g_factor_width,
                    sub_layer.g_factor_width,
                ), f"g_factor must be square {sub_layer.g_factor.shape}, {sub_layer.g_factor_width}"
                sub_layer.compute_g_inv(damping=damping)

            """
            self.g_inv_local = torch.block_diag(
                *[sub_layer.g_inv_local for sub_layer in self.sub_layers]
            )
            self.g_inv_local.add_(damping)

            assert (
                self.g_inv_local.shape[0] == self.g_inv_local.shape[1]
            ), "g_inv_local must be square"
            assert (
                self.g_inv_local.shape == self.module.g_factor_shape
            ), "g_inv_local must match module weights size"
            """

    def preconditioned_grad(self, damping: float = 0.001) -> None:
        """Compute precondition gradient of each weight in module.

        Preconditioned gradients can be applied to the actual gradients with
        `update_gradient()`. Note the steps are separate in the event that
        intermediate steps will be applied to the preconditioned gradient.

        Args:
            damping (float, optional): damping to use if preconditioning using
                the eigendecomposition method (default: 0.001).
        """
        grad = self.module.get_grad()
        grad_type = grad.dtype
        grad = grad.to(self.inv_dtype)
        # self.grad = (self.g_inv @ grad @ self.a_inv).to(grad_type)
        self.grad = self.g_inv_local @ grad @ self.a_inv_local
        self.grad = self.grad.to(grad_type)
        # self.grad.add_(grad).mul_(0.5).to(dtype=grad_type)
        return
