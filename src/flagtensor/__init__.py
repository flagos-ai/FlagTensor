import torch

from flagtensor import runtime
from flagtensor.cutensor import BlockSparseTensor
from flagtensor.cutensor import BlockSparseTensorContraction
from flagtensor.cutensor import BlockSparseTensorDescriptor
from flagtensor.ops.CUTENSOR_OP_ADD import add
from flagtensor.ops.CUTENSOR_OP_ABS import abs
from flagtensor.ops.CUTENSOR_OP_ACOSH import acosh
from flagtensor.ops.CUTENSOR_OP_ACOS import acos
from flagtensor.ops.CUTENSOR_OP_ASIN import asin
from flagtensor.ops.CUTENSOR_OP_ASINH import asinh
from flagtensor.ops.CUTENSOR_OP_ATAN import atan
from flagtensor.ops.CUTENSOR_OP_ATANH import atanh
from flagtensor.ops.CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION import block_sparse_tensor_contraction
from flagtensor.ops.CUTENSOR_OP_CEIL import ceil
from flagtensor.ops.CUTENSOR_OP_CONJ import conj
from flagtensor.ops.CUTENSOR_OP_COS import cos
from flagtensor.ops.CUTENSOR_OP_COSH import cosh
from flagtensor.ops.CUTENSOR_OP_EXP import exp
from flagtensor.ops.CUTENSOR_OP_FLOOR import floor
from flagtensor.ops.CUTENSOR_OP_GETT import gett
from flagtensor.ops.CUTENSOR_OP_IDENTITY import identity
from flagtensor.ops.CUTENSOR_OP_LOG import log
from flagtensor.ops.CUTENSOR_OP_MISH import mish
from flagtensor.ops.CUTENSOR_OP_MIN import min
from flagtensor.ops.CUTENSOR_OP_MAX import max
from flagtensor.ops.CUTENSOR_OP_MUL import mul
from flagtensor.ops.CUTENSOR_OP_NEG import neg
from flagtensor.ops.CUTENSOR_OP_RCP import rcp
from flagtensor.ops.CUTENSOR_OP_RELU import relu
from flagtensor.ops.CUTENSOR_OP_SIGMOID import sigmoid
from flagtensor.ops.CUTENSOR_OP_SIN import sin
from flagtensor.ops.CUTENSOR_OP_SINH import sinh
from flagtensor.ops.CUTENSOR_OP_SOFT_PLUS import soft_plus
from flagtensor.ops.CUTENSOR_OP_SOFT_SIGN import soft_sign
from flagtensor.ops.CUTENSOR_OP_SQRT import sqrt
from flagtensor.ops.CUTENSOR_OP_SWISH import swish
from flagtensor.ops.CUTENSOR_OP_TAN import tan
from flagtensor.ops.CUTENSOR_OP_TANH import tanh
from flagtensor.ops.CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY import tensor_contraction_trinary
from flagtensor.ops.CUTENSOR_OP_TGETT import tgett
try:
    from flagtensor.ops.CUTENSOR_OP_TRINARY_GENERIC import trinary
except Exception:
    trinary = None
from flagtensor.ops.CUTENSOR_OP_TTGT import ttgt

__all__ = ["BlockSparseTensor", "BlockSparseTensorContraction", "BlockSparseTensorDescriptor", "add", "abs", "acosh", "acos", "asin", "asinh", "atan", "atanh", "block_sparse_tensor_contraction", "ceil", "exp", "floor", "gett", "identity", "log", "mish", "min", "max", "mul", "soft_plus", "soft_sign", "sqrt", "relu", "conj", "cos", "cosh", "neg", "rcp", "sigmoid", "sin", "sinh", "swish", "tan", "tanh", "tensor_contraction_trinary", "tgett", "trinary", "ttgt",
"aten_lib", "enable", "only_enable", "all_registered_ops", "all_registered_keys"]

runtime.replace_customized_ops(globals())

# ---------------------------------------------------------------------------
# FlagOS plugin registration — torch.library dispatch
# ---------------------------------------------------------------------------
from flagtensor.runtime.op_registrar import GeneralOpRegistrar  # noqa: E402

aten_lib = torch.library.Library("aten", "IMPL")

_FULL_CONFIG = (
    # Unary operators — 28
    ("abs", abs),
    ("acos", acos),
    ("acosh", acosh),
    ("asin", asin),
    ("asinh", asinh),
    ("atan", atan),
    ("atanh", atanh),
    ("ceil", ceil),
    ("conj", conj),
    ("cos", cos),
    ("cosh", cosh),
    ("exp", exp),
    ("floor", floor),
    ("identity", identity),
    ("log", log),
    ("mish", mish),
    ("neg", neg),
    ("reciprocal", rcp),       # FlagTensor "rcp" → aten "reciprocal"
    ("relu", relu),
    ("sigmoid", sigmoid),
    ("sin", sin),
    ("sinh", sinh),
    ("softplus", soft_plus),   # FlagTensor "soft_plus" → aten "softplus"
    ("sqrt", sqrt),
    ("silu", swish),           # FlagTensor "swish" → aten "silu"
    ("tan", tan),
    ("tanh", tanh),
    # soft_sign — no aten equivalent, skipped for now
    # Binary operators — to be added after verification
    # ("add.Tensor", add),
    # ("mul.Tensor", mul),
    # ("max", max),
    # ("min", min),
    # Contraction operators — to be added after verification
    # Sparse operators — to be added after verification
)

_current_registrar = None


def enable(lib=None, exclude=None):
    """Register all FlagTensor ops into the PyTorch aten dispatch table.

    After calling this, ``torch.abs(x)`` (on a CUDA tensor) will
    automatically use FlagTensor's implementation.

    Parameters
    ----------
    lib : torch.library.Library, optional
        Library instance. Defaults to the global ``aten_lib``.
    exclude : list of str, optional
        List of function names to skip.
    """
    global _current_registrar
    _current_registrar = GeneralOpRegistrar(
        _FULL_CONFIG,
        lib=lib or aten_lib,
        exclude_ops=list(exclude or []),
    )


def only_enable(lib=None, include=None):
    """Register *only* the specified FlagTensor ops.

    Parameters
    ----------
    lib : torch.library.Library, optional
        Library instance. Defaults to the global ``aten_lib``.
    include : list of str
        List of function names to register. All other ops are skipped.
    """
    global _current_registrar
    if not include:
        import warnings
        warnings.warn("only_enable: 'include' is empty — no ops registered.")
        return
    _current_registrar = GeneralOpRegistrar(
        _FULL_CONFIG,
        lib=lib or aten_lib,
        include_ops=list(include),
    )


def all_registered_ops():
    """Return the list of function names registered by the last enable/only_enable call."""
    if _current_registrar is None:
        return []
    return _current_registrar.get_all_ops()


def all_registered_keys():
    """Return the list of aten operator keys registered by the last enable/only_enable call."""
    if _current_registrar is None:
        return []
    return _current_registrar.get_all_keys()
