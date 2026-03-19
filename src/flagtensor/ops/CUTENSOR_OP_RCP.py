import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _rcp_scalar(x):
    return 1.0 / x


_rcp_kernel, rcp = make_unary_pointwise_from_family(
    "rcp",
    "rcp_like",
    _rcp_scalar,
    fallback_float64=torch.reciprocal,
)
