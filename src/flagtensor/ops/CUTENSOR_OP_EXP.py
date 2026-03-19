import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _exp_scalar(x):
    return tl.exp(x)


_exp_kernel, exp = make_unary_pointwise_from_family(
    "exp",
    "exp_like",
    _exp_scalar,
)
