from typing import List
import torch
from enum import Enum
from torch import Tensor, inf
from typing import List, Sequence, Tuple

print_ct = {}


def print_once(message: str) -> None:
    """Print a message only once."""
    if message not in print_ct:
        print(message)
        print_ct[message] = 1
    else:
        print_ct[message] += 1


class SplitEnd(Enum):
    IN = 1
    OUT = 2
    BOTH = 3
    NONE = 0


def block_diag_left_and_right(
    A_blocks: torch.Tensor,
    B_blocks: torch.Tensor,
    C_blocks: torch.Tensor,
    is_a_transpose=False,
    is_c_transpose=False,
) -> torch.Tensor:
    """
    Multiply block-diagonal matrix (represented by stacked blocks) with block-structured matrix B.

    A_blocks: (kn, k) → n blocks of (k x k)
    B_blocks: (kn, km) → n×m blocks of (k x k)
    c_blocks: (km, k) → m blocks of (k x k)
    Returns: (kn, km)
    """
    kn, k = A_blocks.shape
    kn2, km = B_blocks.shape
    km2, k2 = C_blocks.shape
    assert kn == kn2, "Row count mismatch"
    assert kn % k == 0 and km % k == 0, "Dimensions must be divisible by k"
    assert km == km2 and k == k2, "C_blocks shape must match B_blocks"

    n = kn // k
    m = km // k

    # Reshape A to (n, k, k)
    A_reshaped = A_blocks.view(n, k, k)
    if is_a_transpose:
        A_reshaped = A_reshaped.transpose(1, 2)
    # Reshape B to (n, m, k, k)
    B_reshaped = B_blocks.view(n, k, m, k).permute(0, 2, 1, 3)  # (n, m, k, k)

    # Reshape C to (m, k, k)
    C_reshaped = C_blocks.view(m, k, k)
    if is_c_transpose:
        C_reshaped = C_reshaped.transpose(1, 2)

    # Block-wise multiplication
    D_blocks = torch.matmul(A_reshaped[:, None], B_reshaped)  # (n, m, k, k)
    D_blocks = torch.einsum("imjk,mkl->imjl", D_blocks, C_reshaped)

    # Reshape back to (kn, km)
    D = D_blocks.permute(0, 2, 1, 3).contiguous().reshape(n * k, m * k)
    return D


def block_diag_left_mm(A_blocks: torch.Tensor, B_blocks: torch.Tensor) -> torch.Tensor:
    kn, k = A_blocks.shape
    kn2, km_total = B_blocks.shape
    assert kn == kn2, "Row count mismatch"
    assert kn % k == 0, "kn must be divisible by k"
    n = kn // k

    if km_total % k == 0:
        m = km_total // k
        bias = False
    else:
        m = (km_total - 1) // k
        assert (km_total - 1) % k == 0, "Bias dimension must align with k"
        bias = True

    A_reshaped = A_blocks.view(n, k, k)
    B_main = B_blocks[:, : m * k].view(n, k, m, k).permute(0, 2, 1, 3)  # (n, m, k, k)
    C_main = torch.matmul(A_reshaped[:, None], B_main)  # (n, m, k, k)
    C_main = C_main.permute(0, 2, 1, 3).contiguous().reshape(n * k, m * k)

    if bias:
        # extract and reshape the last column (bias)
        B_bias = B_blocks[:, -1:].view(n, k, 1)  # (n, k, 1)
        C_bias = torch.matmul(A_reshaped, B_bias).reshape(n * k, 1)
        C = torch.cat([C_main, C_bias], dim=1)
    else:
        C = C_main
    return C


def block_diag_right_mm(
    B_flat: torch.Tensor, A_flat: torch.Tensor | List[torch.Tensor]
) -> torch.Tensor:
    if isinstance(A_flat, list):
        return block_diag_right_mm_with_bias(B_flat, A_flat)
    elif isinstance(A_flat, torch.Tensor):
        return block_diag_right_mm_no_bias(B_flat, A_flat)
    else:
        print_once(f"block_diag_right_mm: {A_flat}")
        raise TypeError("A_flat must be a torch.Tensor or a list of torch.Tensor")


def block_diag_right_mm_with_bias(
    B_flat: torch.Tensor, A_flat: List[torch.Tensor]
) -> torch.Tensor:
    sizes = [a.shape[0] for a in A_flat]
    cum_sizes = [0]
    for s in sizes:
        cum_sizes.append(cum_sizes[-1] + s)
    C = B_flat.clone()
    for i in range(len(A_flat)):
        start = cum_sizes[i]
        end = cum_sizes[i + 1]
        C[:, start:end] = B_flat[:, start:end] @ A_flat[i]
    return C


def block_diag_right_mm_no_bias(
    B_flat: torch.Tensor, A_flat: torch.Tensor, is_a_transpose: bool = False
) -> torch.Tensor:
    """
    计算 C = B × A，其中 A 是带对角块的块对角矩阵，B 是一个按块分割的矩阵。

    参数:
    - B_flat: 形状 (n*k, m*k)，代表一个由 n×m 个 k×k 子块构成的大矩阵 B。
    - A_flat: 形状 (m*k, k)，代表一个由 m 个 k×k 对角块组成的块对角矩阵 A（按行堆叠）。
    - k:     每个块的大小 (k×k)。
    - is_a_transpose: 如果为 True，则在乘法前先对每个 k×k 的对角块做转置，
                      等价于 A_block = A_block.T。

    返回:
    - C: 形状 (n*k, m*k)，等于 B_flat × (block-diag(A_flat))，其中 block-diag(A_flat) 是一个 (m*k)×(m*k) 的对角块矩阵。
    """

    # --- 1. 检查 shape 约束 ---
    kn, km = B_flat.shape
    mk, k = A_flat.shape
    assert kn % k == 0 and km % k == 0 and mk % k == 0, "kn, km, mk 必须能被 k 整除"
    n = kn // k
    m = mk // k
    assert km == m * k, "B_flat 的列数应该等于 m*k"

    # --- 2. 将 B_flat reshape 成 (n, m, k, k) 格式的 sub-blocks ---
    # 首先把 B_flat 看作 (n*k, m*k)，reshape 为 (n, k, m, k)，再 permute 到 (n, m, k, k)
    B_reshaped = B_flat.view(n, k, m, k).permute(0, 2, 1, 3)
    # 此时 B_reshaped[i,j] 是一个 k×k 的子块，对应原来 B_flat 中行块 i、列块 j

    # --- 3. 将 A_flat reshape 成 (m, k, k) 格式的对角块 ---
    A_reshaped = A_flat.view(m, k, k)
    if is_a_transpose:
        # 如果需要先转置每个对角块
        A_reshaped = A_reshaped.transpose(1, 2)  # 变成 (m, k, k)，每个块内部转置

    # --- 4. 执行分块右乘：对每个 (i,j) 对做 B_ij @ A_j ---
    # B_reshaped 的 shape: (n, m, k, k)
    # A_reshaped 的 shape: (m, k, k)
    # 我们需要得到 C_blocks[i,j] = B_reshaped[i,j] @ A_reshaped[j]
    #
    # 方法 1：显式把 A_reshaped expand 成 (n, m, k, k)：
    #   A_expand = A_reshaped.unsqueeze(0).expand(n, m, k, k)
    #   C_blocks = torch.matmul(B_reshaped, A_expand)
    #
    # 方法 2：用 einsum 表达更直观：
    C_blocks = torch.einsum("imjk,mkl->imjl", B_reshaped, A_reshaped)
    # 解释：i: 0..n-1, m: 0..m-1, j: 0..k-1, k: 0..k-1, l: 0..k-1

    # 此时 C_blocks 形状为 (n, m, k, k)，其中 C_blocks[i,j] = B_reshaped[i,j] @ A_reshaped[j]

    # --- 5. 把 C_blocks 拼回 (n*k, m*k) 的格式 ---
    # 先 permute 到 (n, k, m, k)，再 reshape 成 (n*k, m*k)
    C = C_blocks.permute(0, 2, 1, 3).contiguous().view(n * k, m * k)
    return C


def smart_detect_inf(tensor: Tensor) -> Tensor:
    """
    Replaces positive infinity in the tensor with 1. and negative infinity with 0..

    Parameters:
    tensor (torch.Tensor): Input tensor that can have any dimension.

    Returns:
    torch.Tensor: A tensor with the same shape, dtype, and device as the input, where
                  positive infinities are replaced by 1. and negative infinities by 0..
    """
    result_tensor = tensor.clone()
    result_tensor[tensor == torch.inf] = 1.0
    result_tensor[tensor == -inf] = 0.0
    return result_tensor


def MinMaxNormalization(tensor: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Scales tensor values to range [0,1] using min-max normalization.

    Args:
        tensor: Input tensor
        epsilon: Small value to prevent division by zero (default: 1e-6)

    Returns:
        Normalized tensor with values in [0,1]
    """
    tensor = smart_detect_inf(tensor)
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    range_tensor = max_tensor - min_tensor
    return tensor.add_(-min_tensor).div_(range_tensor + epsilon)


def MinMaxNormalizationForList(
    tensors: List[Tensor], epsilon: float = 1e-6
) -> List[Tensor]:
    """
    Scales tensor values to range [0,1] using min-max normalization.

    Args:
        tensor: Input tensor
        epsilon: Small value to prevent division by zero (default: 1e-6)
    """
    tensors = [smart_detect_inf(t) for t in tensors]
    all_values = torch.cat([t.flatten() for t in tensors])
    min_val = all_values.min()
    max_val = all_values.max()
    denom = (max_val - min_val).clamp(min=epsilon)
    normed = [(t - min_val) / denom for t in tensors]
    return normed


def _upper_triangle_length(n: int) -> int:
    """Return number of elements in the upper triangle (including diagonal) of an n×n matrix."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n + 1) // 2


def flatten_sym_blocks_upper(
    blocks: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, List[int]]:
    """
    将一组对称方阵的对角块（大小各不相同）按“仅上三角含对角线”的顺序拉平成一个一维张量，
    并返回每个方阵的阶数列表以便随后反解。

    Args:
        blocks: 可迭代的方阵张量序列，每个张量形状为 (n_i, n_i)。默认假定它们为对称矩阵。
                所有张量将被视为在同一 device、同一 dtype（以 blocks[0] 为准）。

    Returns:
        flat:   连接后的 1D 张量，包含各块上三角（含对角线）元素，按块顺序依次拼接。
        sizes:  每个块的阶数列表 [n_1, n_2, ... , n_k]，用于后续反解。

    Notes:
        - 为节省显存/通信，仅抽取上三角元素（包含对角线）。
        - 若需要严格校验对称性，可在下方打开断言。不过这会有额外开销。
    """
    blocks = list(blocks)
    if len(blocks) == 0:
        # 返回空向量和空尺寸列表
        return torch.empty(0), []

    # 统一 device/dtype 以第一块为准
    device = blocks[0].device
    dtype = blocks[0].dtype

    sizes: List[int] = []
    lengths: List[int] = []
    total_len = 0
    for B in blocks:
        if B.ndim != 2 or B.size(0) != B.size(1):
            raise ValueError("All blocks must be 2D square matrices.")
        n = B.size(0)
        sizes.append(n)
        l = _upper_triangle_length(n)
        lengths.append(l)
        total_len += l
        # 可选对称性检查（关闭以节约时间）
        # if not torch.allclose(B, B.T, atol=1e-6, rtol=1e-6):
        #     raise ValueError("Block is not symmetric within tolerance.")

    flat = torch.empty(total_len, device=device, dtype=dtype)

    offset = 0
    for B, n, l in zip(blocks, sizes, lengths):
        # 确保位于正确 device/dtype
        if B.device != device or B.dtype != dtype:
            B = B.to(device=device, dtype=dtype)
        iu = torch.triu_indices(n, n, offset=0, device=B.device)
        vals = B[iu[0], iu[1]]
        # 保障连续性
        flat[offset : offset + l].copy_(vals.reshape(-1))
        offset += l

    return flat, sizes


def unflatten_sym_blocks_upper(
    flat: torch.Tensor,
    sizes: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> List[torch.Tensor]:
    """
    将由 `flatten_sym_blocks_upper` 生成的一维张量反解回对称方阵块列表。

    Args:
        flat:   1D 张量，按块依次存放各块上三角（含对角线）元素。
        sizes:  每个块的阶数列表 [n_1, n_2, ... , n_k]。
        dtype:  结果张量的数据类型（可选，默认沿用 flat 的 dtype）。
        device: 结果张量的 device（可选，默认沿用 flat 的 device）。

    Returns:
        blocks: 复原后的对称方阵块列表。

    Raises:
        ValueError: 若 flat 的长度与 sizes 推导的总上三角元素数量不匹配。
    """
    if flat.ndim != 1:
        raise ValueError(
            f"flat must be a 1D tensor. {flat.__class__.__name__} {flat} found."
        )

    if dtype is None:
        dtype = flat.dtype
    if device is None:
        device = flat.device

    sizes = list(sizes)
    expected = sum(_upper_triangle_length(n) for n in sizes)
    if flat.numel() != expected:
        raise ValueError(
            f"flat length {flat.numel()} does not match expected total {expected} from sizes."
        )

    blocks: List[torch.Tensor] = []
    offset = 0
    for n in sizes:
        l = _upper_triangle_length(n)
        tri_vals = flat[offset : offset + l].to(dtype=dtype, device=device)
        offset += l

        # 构造上三角
        M = torch.zeros((n, n), dtype=dtype, device=device)
        iu = torch.triu_indices(n, n, offset=0, device=device)
        M[iu[0], iu[1]] = tri_vals
        # 反射到下三角，确保对称（对角线不重复加）
        M = M + M.T - torch.diag(torch.diagonal(M))
        blocks.append(M)

    return blocks


# ------------------------------
# Tests (unittest-style)
# ------------------------------
import unittest


def _make_sym(n: int, *, device=None, dtype=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    A = torch.randn((n, n), device=device, dtype=dtype)
    M = (A + A.T) / 2.0
    return M


class TestDiagBlockFlattening(unittest.TestCase):
    def test_empty(self):
        flat, sizes = flatten_sym_blocks_upper([])
        self.assertEqual(sizes, [])
        self.assertTrue(isinstance(flat, torch.Tensor))
        self.assertEqual(flat.numel(), 0)
        # unflatten of empty
        blocks = unflatten_sym_blocks_upper(flat, sizes)
        self.assertEqual(blocks, [])

    def test_roundtrip_cpu(self):
        torch.manual_seed(42)
        sizes = [1, 2, 5, 3]
        blocks_in = [
            _make_sym(n, device=torch.device("cpu"), dtype=torch.float32) for n in sizes
        ]
        flat, got_sizes = flatten_sym_blocks_upper(blocks_in)
        self.assertEqual(got_sizes, sizes)
        # 长度校验
        expected_len = sum(n * (n + 1) // 2 for n in sizes)
        self.assertEqual(flat.numel(), expected_len)
        # 反解
        blocks_out = unflatten_sym_blocks_upper(flat, sizes)
        self.assertEqual(len(blocks_out), len(blocks_in))
        for a, b in zip(blocks_in, blocks_out):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))
            # 对称性
            self.assertTrue(torch.allclose(b, b.T, atol=0, rtol=0))

    def test_dtype_device_override(self):
        torch.manual_seed(0)
        sizes = [3, 4]
        blocks_in = [
            _make_sym(n, device=torch.device("cpu"), dtype=torch.float64) for n in sizes
        ]
        flat, got_sizes = flatten_sym_blocks_upper(blocks_in)
        self.assertEqual(got_sizes, sizes)
        # 指定不同 dtype/device 输出
        out_dtype = torch.float32
        out_device = torch.device("cpu")
        blocks_out = unflatten_sym_blocks_upper(
            flat, sizes, dtype=out_dtype, device=out_device
        )
        for b in blocks_out:
            self.assertEqual(b.dtype, out_dtype)
            self.assertEqual(b.device, out_device)
        # 数值上仍一致（float64->float32 容忍更大的误差）
        for a, b in zip(blocks_in, blocks_out):
            self.assertTrue(
                torch.allclose(a.to(dtype=out_dtype), b, atol=1e-5, rtol=1e-5)
            )

    def test_error_mismatch_length(self):
        torch.manual_seed(123)
        sizes = [2, 3]
        blocks_in = [_make_sym(n) for n in sizes]
        flat, _ = flatten_sym_blocks_upper(blocks_in)
        # 少取一个元素，触发长度不匹配
        with self.assertRaises(ValueError):
            _ = unflatten_sym_blocks_upper(flat[:-1], sizes)

    def test_non_square_raises(self):
        # 构造非方阵
        B = torch.randn(2, 3)
        with self.assertRaises(ValueError):
            _ = flatten_sym_blocks_upper([B])

    def test_cuda_roundtrip_if_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(7)
        device = torch.device("cuda")
        sizes = [2, 3]
        blocks_in = [_make_sym(n, device=device) for n in sizes]
        flat, got_sizes = flatten_sym_blocks_upper(blocks_in)
        self.assertEqual(got_sizes, sizes)
        self.assertEqual(flat.device.type, "cuda")
        blocks_out = unflatten_sym_blocks_upper(flat, sizes)
        for a, b in zip(blocks_in, blocks_out):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))
            self.assertEqual(b.device.type, "cuda")


if __name__ == "__main__":
    # unittest.main()
    device = torch.device("cuda")
    sizes = [2, 3, 2]
    blocks_in = [_make_sym(n, device=device) for n in sizes]
    print(f"Blocks in: {blocks_in}")
    flat, got_sizes = flatten_sym_blocks_upper(blocks_in)
    print(f"Flattened: {flat}, Shape {flat.shape}, Sizes: {got_sizes}")
    blocks_out = unflatten_sym_blocks_upper(flat, sizes)
    print(f"Blocks out: {blocks_out}")
