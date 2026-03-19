import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _mish_scalar(x):
    softplus = tl.log(1 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0)
    exp_neg_twice = tl.exp(-2 * softplus)
    tanh_softplus = (1 - exp_neg_twice) / (1 + exp_neg_twice)
    return x * tanh_softplus


_mish_kernel, mish = make_unary_pointwise_from_family(
    "mish",
    "mish_like",
    _mish_scalar,
    fallback_float64=F.mish,
)
