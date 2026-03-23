import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _tan_scalar(x):
    return tl.sin(x) / tl.cos(x)


_tan_kernel, tan = make_unary_pointwise_from_family(
    "tan",
    "tan_like",
    _tan_scalar,
    fallback_float64=torch.tan,
)
