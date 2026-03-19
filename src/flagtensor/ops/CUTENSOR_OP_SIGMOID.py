import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _sigmoid_scalar(x):
    return 1 / (1 + tl.exp(-x))


_sigmoid_kernel, sigmoid = make_unary_pointwise_from_family(
    "sigmoid",
    "sigmoid_like",
    _sigmoid_scalar,
    fallback_float64=torch.sigmoid,
)
