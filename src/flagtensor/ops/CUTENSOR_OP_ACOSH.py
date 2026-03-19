import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _acosh_scalar(x):
    return tl.log(x + tl.sqrt(x * x - 1))


_acosh_kernel, acosh = make_unary_pointwise_from_family(
    "acosh",
    "acosh_like",
    _acosh_scalar,
    fallback_float64=torch.acosh,
)
