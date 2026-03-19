import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _asin_scalar(x):
    return x


_asin_kernel, asin = make_unary_pointwise_from_family(
    "asin",
    "asin_like",
    _asin_scalar,
    fallback_float64=torch.asin,
)
