from torch import Tensor, inf
from ..layers.eigen import *
from ..layers.modules import get_cov, append_bias_ones
from .dia_eigen import DiaEigenLayer
from typing import List
from enum import Enum
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

        if sub_split_end == SplitEnd.NONE:
            return

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

    def save_layer_input(self, input_: list[torch.Tensor]) -> None:
        if not self.split_in:
            super().save_layer_input(input_)
            return
        """Save input for layer, but only use the slice corresponding to this rank."""

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

    def broadcast_a_inv(
        self,
        src: int,
        group: dist.ProcessGroup | None = None,
        ) -> None:
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
                sub_layer.compute_a_inv(damping=damping)
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
                sub_layer.compute_g_inv(damping=damping)
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
