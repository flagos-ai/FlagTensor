import torch
import triton
import triton.language as tl

from flagtensor import runtime
from flagtensor.utils import libtuner


@libtuner(
    configs=runtime.get_tuned_config("elementwise_binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_binary"))
@triton.jit
def _max_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    if KERNEL_ID == 0:
        offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        z = tl.maximum(x, y)
        tl.store(z_ptr + offsets, z, mask=mask)
    else:
        for block_idx in tl.static_range(0, BLOCKS_PER_PROGRAM):
            offsets = block_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            z = tl.where(x > y, x, y)
            tl.store(z_ptr + offsets, z, mask=mask)


def max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("input tensors must be on CUDA")
    if x.dtype != y.dtype:
        raise TypeError("input tensors must have the same dtype")
    if x.shape != y.shape:
        raise ValueError("input tensors must have the same shape")
    z = torch.empty_like(x)
    n_elements = z.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _max_kernel[grid](x, y, z, n_elements)
    return z
