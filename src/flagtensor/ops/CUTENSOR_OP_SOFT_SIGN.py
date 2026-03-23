import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _soft_sign_scalar(x):
    return x / (tl.abs(x) + 1)


_soft_sign_kernel, soft_sign = make_unary_pointwise_from_family(
    "soft_sign",
    "softsign_like",
    _soft_sign_scalar,
    fallback_float64=lambda x: x / (torch.abs(x) + 1),
)
