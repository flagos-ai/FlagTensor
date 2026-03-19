import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _log_scalar(x):
    return tl.log(x)


_log_kernel, log = make_unary_pointwise_from_family(
    "log",
    "log_like",
    _log_scalar,
)
