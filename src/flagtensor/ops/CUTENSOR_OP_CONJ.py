import torch
import triton
import triton.language as tl

from flagtensor import runtime


@triton.autotune(
    configs=runtime.get_tuned_config("CUTENSOR_OP_CONJ"),
    key=["n_elements"],
)
@triton.jit
def _conj_kernel(
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
    real = tl.load(x_ptr + 2 * offsets, mask=mask)
    imag = tl.load(x_ptr + 2 * offsets + 1, mask=mask)
    tl.store(y_ptr + 2 * offsets, real, mask=mask)
    tl.store(y_ptr + 2 * offsets + 1, -imag, mask=mask)


def conj(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("input tensor must be on CUDA")
    if not x.is_complex():
        return x.clone()
    y = torch.empty_like(x)
    if not x.is_contiguous():
        x = x.contiguous()
    real_dtype = torch.float32 if x.dtype == torch.complex64 else torch.float64
    x_view = x.view(real_dtype)
    y_view = y.view(real_dtype)
    n_elements = x.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _conj_kernel[grid](x_view, y_view, n_elements)
    return y
