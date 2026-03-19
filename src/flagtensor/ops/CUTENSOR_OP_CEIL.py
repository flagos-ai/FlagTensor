import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _ceil_scalar(x):
    return tl.ceil(x)


_ceil_kernel, ceil = make_unary_pointwise_from_family(
    "ceil",
    "ceil_like",
    _ceil_scalar,
    fallback_float64=torch.ceil,
)
