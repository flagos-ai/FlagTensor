import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family
from flagtensor.utils.unary_pointwise import _NEG_UNARY_EXTRA, _DEFAULT_UNARY_DTYPES


@triton.jit
def _neg_scalar(x):
    return -x


_neg_kernel, neg = make_unary_pointwise_from_family(
    "neg",
    "neg_like",
    _neg_scalar,
    supported_dtypes=_DEFAULT_UNARY_DTYPES | _NEG_UNARY_EXTRA,
)
