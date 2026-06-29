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
from flagtensor.ops.CUTENSOR_OP_SOFT_SIGN import soft_sign  # PyTorch decomposes F.softsign→abs+add+div; no aten::softsign dispatch key
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

# ---------------------------------------------------------------------------
# Aten-compatible wrappers — bridge dispatcher arg order to FlagTensor signatures
# ---------------------------------------------------------------------------


def _aten_add_tensor(*args, **kwargs):
    """Wrapper for aten::add.Tensor → flagtensor.add.

    Metax PyTorch 2.8 may pass ``alpha`` as a positional arg rather than keyword-only,
    so we accept ``*args, **kwargs`` and resolve the three-argument layout heuristically.
    """
    alpha = kwargs.pop("alpha", 1)

    if len(args) == 2:
        x, y = args
    elif len(args) == 3:
        if torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
            x, y, alpha = args
        elif torch.is_tensor(args[1]) and torch.is_tensor(args[2]):
            alpha, x, y = args
        else:
            raise TypeError(
                f"unsupported aten add args: {[type(a).__name__ for a in args]}"
            )
    else:
        raise TypeError(f"unsupported aten add arg count: {len(args)}")

    if not torch.is_tensor(x):
        x = _ensure_tensor(x, y)
    if not torch.is_tensor(y):
        y = _ensure_tensor(y, x)

    if alpha != 1:
        y = mul(y, _ensure_tensor(alpha, y))

    return add(x, y)


def _ensure_tensor(v, ref):
    """Promote scalar to tensor on the same device/dtype as *ref*."""
    if torch.is_tensor(v):
        return v
    return torch.as_tensor(v, device=ref.device, dtype=ref.dtype)


def _aten_mul_tensor(*args, **kwargs):
    """Wrapper for aten::mul.Tensor → flagtensor.mul.

    Handles tensor×scalar / scalar×tensor (MetaX PyTorch may pass raw scalars).
    """
    x, y = args[:2]
    if not torch.is_tensor(x):
        x = _ensure_tensor(x, y)
    if not torch.is_tensor(y):
        y = _ensure_tensor(y, x)
    return mul(x, y)


def _aten_maximum(*args, **kwargs):
    """Wrapper for aten::maximum → flagtensor.max (element-wise broadcasting)."""
    x, y = args[:2]
    if not torch.is_tensor(x):
        x = _ensure_tensor(x, y)
    if not torch.is_tensor(y):
        y = _ensure_tensor(y, x)
    return max(x, y)


def _aten_minimum(*args, **kwargs):
    """Wrapper for aten::minimum → flagtensor.min (element-wise broadcasting)."""
    x, y = args[:2]
    if not torch.is_tensor(x):
        x = _ensure_tensor(x, y)
    if not torch.is_tensor(y):
        y = _ensure_tensor(y, x)
    return min(x, y)


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
    # soft_sign — no aten::softsign dispatch key (PyTorch decomposes to abs+add+div); use flagtensor.soft_sign() directly
    # Binary operators — 6 (element-wise)
    ("add.Tensor", _aten_add_tensor),
    ("mul.Tensor", _aten_mul_tensor),
    ("max", max),             # aten::max.other (element-wise)
    ("maximum", _aten_maximum),  # aten::maximum (broadcasting)
    ("min", min),             # aten::min.other (element-wise)
    ("minimum", _aten_minimum),  # aten::minimum (broadcasting)
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
