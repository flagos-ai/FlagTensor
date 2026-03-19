import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _tanh_scalar(x):
    exp_neg_twice = tl.exp(-2 * x)
    return (1 - exp_neg_twice) / (1 + exp_neg_twice)


_tanh_kernel, tanh = make_unary_pointwise_from_family(
    "tanh",
    "tanh_like",
    _tanh_scalar,
    fallback_float64=torch.tanh,
)
