import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _sinh_scalar(x):
    return 0.5 * (tl.exp(x) - tl.exp(-x))


_sinh_kernel, sinh = make_unary_pointwise_from_family(
    "sinh",
    "sinh_like",
    _sinh_scalar,
    fallback_float64=torch.sinh,
)
