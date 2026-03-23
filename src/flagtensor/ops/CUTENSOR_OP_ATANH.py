import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _atanh_scalar(x):
    return 0.5 * tl.log((1 + x) / (1 - x))


_atanh_kernel, atanh = make_unary_pointwise_from_family(
    "atanh",
    "atanh_like",
    _atanh_scalar,
    fallback_float64=torch.atanh,
)
