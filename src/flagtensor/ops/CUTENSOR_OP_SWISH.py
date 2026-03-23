import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _swish_scalar(x):
    return x / (1 + tl.exp(-x))


_swish_kernel, swish = make_unary_pointwise_from_family(
    "swish",
    "swish_like",
    _swish_scalar,
    fallback_float64=lambda x: x * torch.sigmoid(x),
)
