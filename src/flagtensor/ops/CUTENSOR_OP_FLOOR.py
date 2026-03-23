import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _floor_scalar(x):
    return tl.floor(x)


_floor_kernel, floor = make_unary_pointwise_from_family(
    "floor",
    "floor_like",
    _floor_scalar,
    fallback_float64=torch.floor,
)
