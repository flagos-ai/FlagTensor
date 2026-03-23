import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _abs_scalar(x):
    return tl.abs(x)


_abs_kernel, abs = make_unary_pointwise_from_family(
    "abs",
    "abs_like",
    _abs_scalar,
    fallback_float64=torch.abs,
)
