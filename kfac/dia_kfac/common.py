from typing import List
import torch
from enum import Enum
from torch import Tensor, inf

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


def block_diag_right_mm_with_bias(B_flat: torch.Tensor, A_flat: List[torch.Tensor]) -> torch.Tensor:
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
    result_tensor[tensor == torch.inf] = 1.
    result_tensor[tensor == -inf] = 0.
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

def MinMaxNormalizationForList(tensors: List[Tensor], epsilon: float = 1e-6) -> List[Tensor]:
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