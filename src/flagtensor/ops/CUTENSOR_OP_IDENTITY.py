import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _identity_scalar(x):
    return x


_identity_kernel, identity = make_unary_pointwise_from_family(
    "identity",
    "identity_like",
    _identity_scalar,
)
