import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _asinh_scalar(x):
    abs_x = tl.abs(x)
    inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
    return tl.where(x >= 0, tl.log(inner), -tl.log(inner))


_asinh_kernel, asinh = make_unary_pointwise_from_family(
    "asinh",
    "asinh_like",
    _asinh_scalar,
    fallback_float64=torch.asinh,
)
