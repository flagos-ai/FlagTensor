import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _neg_scalar(x):
    return -x


_neg_kernel, neg = make_unary_pointwise_from_family(
    "neg",
    "neg_like",
    _neg_scalar,
    fallback_float64=torch.neg,
)
