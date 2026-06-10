import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family
from flagtensor.utils.unary_pointwise import _NEG_UNARY_EXTRA, _DEFAULT_UNARY_DTYPES


@triton.jit
def _abs_scalar(x):
    return tl.abs(x)


_abs_kernel, abs = make_unary_pointwise_from_family(
    "abs",
    "abs_like",
    _abs_scalar,
    supported_dtypes=_DEFAULT_UNARY_DTYPES | _NEG_UNARY_EXTRA,  # INT8 ok; FP8_E5M2 fails because abs_where uses -x
)
