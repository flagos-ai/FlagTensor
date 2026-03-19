import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _cosh_scalar(x):
    return 0.5 * (tl.exp(x) + tl.exp(-x))


_cosh_kernel, cosh = make_unary_pointwise_from_family(
    "cosh",
    "cosh_like",
    _cosh_scalar,
    fallback_float64=torch.cosh,
)
