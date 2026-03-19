import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _acos_scalar(x):
    return x


_acos_kernel, acos = make_unary_pointwise_from_family(
    "acos",
    "acos_like",
    _acos_scalar,
    fallback_float64=torch.acos,
)
