import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _cos_scalar(x):
    return tl.cos(x)


_cos_kernel, cos = make_unary_pointwise_from_family(
    "cos",
    "cos_like",
    _cos_scalar,
    fallback_float64=torch.cos,
)
