import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flagtensor import runtime


@triton.autotune(
    configs=runtime.get_tuned_config("CUTENSOR_OP_MISH"),
    key=["n_elements"],
)
@triton.jit
def _mish_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    softplus = tl.log(1 + tl.exp(-tl.abs(xf))) + tl.where(xf > 0, xf, 0)
    log2e: tl.constexpr = 1.4426950408889634
    tanh_softplus = 2 / (1 + tl.exp2(-2 * softplus * log2e)) - 1
    y = xf * tanh_softplus
    tl.store(y_ptr + offsets, y, mask=mask)


def mish(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("input tensor must be on CUDA")
    if x.dtype == torch.float64:
        return F.mish(x)
    y = torch.empty_like(x)
    n_elements = y.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _mish_kernel[grid](x, y, n_elements)
    return y
