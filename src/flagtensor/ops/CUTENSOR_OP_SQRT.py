import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _sqrt_scalar(x):
    return tl.sqrt(x)


_sqrt_kernel, sqrt = make_unary_pointwise_from_family(
    "sqrt",
    "sqrt_like",
    _sqrt_scalar,
    fallback_float64=torch.sqrt,
)
