from typing import cast
import torch
from kfac.layers.modules import ModuleHelper
from kfac.layers.utils import get_cov

class LayerNormModuleHelper(ModuleHelper):
    """ModuleHelper for torch.nn.LayerNorm modules (classic K-FAC form).

    视 LayerNorm 的仿射部分为一个从两维特征 [x_hat, 1] 到 D=prod(normalized_shape) 输出
    的线性映射，其中 x_hat 为按 LN 归一化轴标准化后的激活。
    - A 因子：共享 2x2，来自样本级别的 [x_hat, 1] 协方差。
    - G 因子：D x D，来自把上游梯度在“非参数轴”上聚合后的样本协方差。
    """

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        # A 是共享的 2x2
        return (2, 2)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        # G 的维度等于参数维 D = prod(normalized_shape)
        # 对应 module.weight.shape 与 module.bias.shape
        D = int(self.module.weight.numel())  # type: ignore[attr-defined]
        return (D, D)

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass.

        a: 与 LN 输入同形状。按 normalized_shape 的最后 k 个维度做标准化，
        构造 [x_hat, 1] 的样本矩阵，返回 2x2 协方差。
        """
        norm_shape = getattr(self.module, 'normalized_shape', None)
        assert norm_shape is not None, "LayerNorm must have normalized_shape"
        eps = getattr(self.module, 'eps', 1e-5)

        # LN 在最后 norm_ndims 个维度上标准化
        norm_ndims = len(norm_shape) if not isinstance(norm_shape, int) else 1
        assert a.ndim >= norm_ndims, "Input rank smaller than normalized axes"

        reduce_dims = tuple(range(a.ndim - norm_ndims, a.ndim))
        mean = a.mean(dim=reduce_dims, keepdim=True)
        var = a.var(dim=reduce_dims, unbiased=False, keepdim=True) + eps
        xhat = (a - mean) / var.sqrt()

        # 样本矩阵：每个标量位置视为一个样本的 2 维特征 [x_hat, 1]
        x = xhat.reshape(-1, 1)
        ones = torch.ones_like(x)
        phi = torch.cat([x, ones], dim=1)  # [num_samples, 2]

        return get_cov(phi)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output.

        将上游梯度在所有“非参数轴”上求和，保留参数轴（即 normalized_shape 对应的最后 k 维），
        得到每个样本的长度为 D 的通道向量；然后对这些样本做协方差。
        """
        norm_shape = getattr(self.module, 'normalized_shape', None)
        assert norm_shape is not None, "LayerNorm must have normalized_shape"
        norm_ndims = len(norm_shape) if not isinstance(norm_shape, int) else 1

        # 参数轴 = 最后 norm_ndims 个维度；其余轴（含 batch 等）需要聚合
        param_dims = tuple(range(g.ndim - norm_ndims, g.ndim))
        reduce_dims = tuple(d for d in range(g.ndim) if d not in param_dims)

        gparam = g.sum(dim=reduce_dims)        # 形状 == normalized_shape
        samples = gparam.reshape(gparam.size(0), -1) if gparam.ndim > 1 else gparam.unsqueeze(0)
        # 上面确保 samples 形状是 [N, D]：N=批大小（或聚合后的样本数），D=参数维

        return get_cov(samples)

    def get_diag_block_g_factor(self, gy: torch.Tensor, output_chunk_start, output_chunk_end) -> torch.Tensor:
        # LayerNorm output-chunked G factor
        # Keep batch dimension as samples; sum over non-parameter, non-batch axes
        norm_shape = getattr(self.module.module, 'normalized_shape', None)
        if isinstance(norm_shape, int):
            norm_ndims = 1
        else:
            norm_ndims = len(norm_shape)
        param_dims = tuple(range(gy.ndim - norm_ndims, gy.ndim))
        reduce_dims = tuple(d for d in range(gy.ndim) if d not in param_dims and d != 0)
        # Aggregate over reduce dims -> [N, *normalized_shape]
        gparam = gy.sum(dim=reduce_dims)
        N = gparam.shape[0]
        D = int(torch.tensor(norm_shape).prod().item()) if not isinstance(norm_shape, int) else norm_shape
        gparam = gparam.reshape(N, D)
        # Slice parameter (column) chunk
        g = gparam[:, output_chunk_start : output_chunk_end].to(self.factor_dtype)
        if self.grad_scaler is not None:
            g = g / self.grad_scaler()
        return get_cov(g)


    def get_grad(self) -> torch.Tensor:
        """Return gradients stacked as [D, 2]: [grad_w (gamma), grad_b (beta)]."""
        w = cast(torch.Tensor, self.module.weight.grad)
        if self.has_bias():
            b = cast(torch.Tensor, self.module.bias.grad)
        else:
            # 理论上 LN 一般有 bias，这里兜底：若无 bias，则用 0 列占位
            b = torch.zeros_like(w)
        return torch.stack([w, b], dim=1)  # [D,2]

    def set_grad(self, grad: torch.Tensor) -> None:
        """Accept preconditioned grad in shape [D, 2] and write back."""
        w_grad = grad[:, 0].view(self.module.weight.grad.size())          # type: ignore[attr-defined]
        if self.has_bias():
            b_grad = grad[:, 1].view(self.module.bias.grad.size())        # type: ignore[attr-defined]
            self.module.bias.grad = b_grad.contiguous()                   # type: ignore[attr-defined]
        self.module.weight.grad = w_grad.contiguous()                     # type: ignore[attr-defined]

class BN2dModuleHelper(ModuleHelper):
    """ModuleHelper for torch.nn.BatchNorm2d modules (classic K-FAC form).

    视 BN2d 的仿射部分为一个从两维特征 [x_hat, 1] 到 C=num_features 输出的线性映射，
    其中 x_hat 为按 (N,H,W) 统计标准化后的激活。
    - A 因子：共享 2x2，来自样本级别的 [x_hat, 1] 协方差。
    - G 因子：C x C，来自把上游梯度在 (H,W) 聚合后的每样本通道向量的协方差。
    """

    @property
    def a_factor_shape(self) -> tuple[int, int]:
        return (2, 2)

    @property
    def g_factor_shape(self) -> tuple[int, int]:
        C = int(self.module.num_features)  # type: ignore[attr-defined]
        return (C, C)

    def get_a_factor(self, a: torch.Tensor) -> torch.Tensor:
        """Compute A factor with the input from the forward pass.

        a: 形状 [N, C, H, W]。对每个通道用 (N,H,W) 的 batch 统计做标准化得到 x_hat，
        再把所有位置视为样本，构造 [x_hat, 1] 的样本矩阵，返回 2x2 协方差。
        """
        assert a.ndim == 4, "BatchNorm2d expects input as [N,C,H,W]"
        eps = getattr(self.module, 'eps', 1e-5)

        mean = a.mean(dim=(0, 2, 3), keepdim=True)
        var  = a.var(dim=(0, 2, 3), unbiased=False, keepdim=True) + eps
        xhat = (a - mean) / var.sqrt()          # [N,C,H,W]

        x = xhat.reshape(-1, 1)
        ones = torch.ones_like(x)
        phi = torch.cat([x, ones], dim=1)       # [N*C*H*W, 2]

        return get_cov(phi)

    def get_g_factor(self, g: torch.Tensor) -> torch.Tensor:
        """Compute G factor with the gradient w.r.t. the output.

        对上游梯度在 (H,W) 聚合，得到每个样本的通道向量 [C]，再对这些样本做协方差。
        """
        assert g.ndim == 4, "BatchNorm2d grad expects shape [N,C,H,W]"
        # sum over spatial, keep batch to形成样本维
        gs = g.sum(dim=(2, 3))                  # [N, C]

        return get_cov(gs)

    def get_grad(self) -> torch.Tensor:
        """Return gradients stacked as [C, 2]: [grad_gamma, grad_beta]."""
        w = cast(torch.Tensor, self.module.weight.grad)
        if self.has_bias():
            b = cast(torch.Tensor, self.module.bias.grad)
        else:
            b = torch.zeros_like(w)
        return torch.stack([w, b], dim=1)  # [C,2]

    def set_grad(self, grad: torch.Tensor) -> None:
        """Accept preconditioned grad in shape [C, 2] and write back."""
        w_grad = grad[:, 0].view(self.module.weight.grad.size())          # type: ignore[attr-defined]
        if self.has_bias():
            b_grad = grad[:, 1].view(self.module.bias.grad.size())        # type: ignore[attr-defined]
            self.module.bias.grad = b_grad.contiguous()                   # type: ignore[attr-defined]
        self.module.weight.grad = w_grad.contiguous()                     # type: ignore[attr-defined]