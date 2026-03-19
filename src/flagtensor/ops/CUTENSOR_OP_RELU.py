import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _relu_scalar(x):
    return tl.maximum(x, 0)


_relu_kernel, relu = make_unary_pointwise_from_family(
    "relu",
    "relu_like",
    _relu_scalar,
)
