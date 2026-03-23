import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _soft_plus_scalar(x):
    return tl.log(1 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0)


_soft_plus_kernel, soft_plus = make_unary_pointwise_from_family(
    "soft_plus",
    "softplus_like",
    _soft_plus_scalar,
    fallback_float64=F.softplus,
)
