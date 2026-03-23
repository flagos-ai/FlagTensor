import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _sin_scalar(x):
    return tl.sin(x)


_sin_kernel, sin = make_unary_pointwise_from_family(
    "sin",
    "sin_like",
    _sin_scalar,
    fallback_float64=torch.sin,
)
