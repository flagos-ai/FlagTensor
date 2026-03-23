import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _atan_scalar(x):
    return x


_atan_kernel, atan = make_unary_pointwise_from_family(
    "atan",
    "atan_like",
    _atan_scalar,
    fallback_float64=torch.atan,
)
