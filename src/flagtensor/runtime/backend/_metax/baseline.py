# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX-native baseline classes for FlagTensor benchmarks.

On the MetaX (C500/C550) accelerator (MACA SDK at ``/opt/maca-3.7.1``,
device exposed through ``torch.cuda`` by the torch-maca plugin):

* **No cuTensor equivalent exists.** No ``libcutensor`` ships with the
  MACA SDK, and no other MetaX library (libmcblas / libmcdnn / libmcfft
  / libmccl ...) exposes cuTensor's generalized tensor-contraction /
  elementwise-trinary API. ``flagtensor.cutensor`` therefore marks
  cuTensor unavailable on this platform.

* **PyTorch native ops DO use vendor-optimised kernels.** ``torch.matmul``
  / ``torch.addmm`` / ``torch.einsum`` and the elementwise ``aten`` ops
  dispatch through the MACA stack to MetaX's vendor libraries — the same
  position cuTensor occupies on NVIDIA.

* **This mirrors the FlagGems convention** and the PPU / Iluvatar
  backends: for platforms without a cuTensor-equivalent vendor library,
  ``torch.*`` ops are the native baseline. This is NOT a "fallback" — it
  is the vendor-supplied optimised path on MetaX.

Unlike PPU / Iluvatar (whose baseline module is prepared but not yet
loaded by the benchmark harness), this module is actively wired into
``benchmark_core.Benchmark._baseline_module`` via the
``BASELINE_MODULE_NAME`` sentinel on ``_metax/__init__.py``. To make that
resolution work, the classes here are exposed under the ``CuTensor*``
naming convention that ``Benchmark._get_baseline_instance`` looks up
(``CuTensorAbs``, ``CuTensorSin``, ``CuTensorContraction`` ...), exactly
like ``flagtensor.torch_npu_baseline`` does on Ascend. The actual
implementation lives in :mod:`flagtensor.torch_baseline`.
"""

import torch

from flagtensor.torch_baseline import (
    TorchBinaryBaseline,
    TorchBlockSparseContractionBaseline,
    TorchContractionBaseline,
    TorchContractionTrinaryBaseline,
    TorchTrinaryBaseline,
    TorchUnaryBaseline,
)
from flagtensor.cutensor import (
    CUTENSOR_OP_ABS,
    CUTENSOR_OP_ACOS,
    CUTENSOR_OP_ACOSH,
    CUTENSOR_OP_ADD,
    CUTENSOR_OP_ASIN,
    CUTENSOR_OP_ASINH,
    CUTENSOR_OP_ATAN,
    CUTENSOR_OP_ATANH,
    CUTENSOR_OP_CEIL,
    CUTENSOR_OP_CONJ,
    CUTENSOR_OP_COS,
    CUTENSOR_OP_COSH,
    CUTENSOR_OP_EXP,
    CUTENSOR_OP_FLOOR,
    CUTENSOR_OP_IDENTITY,
    CUTENSOR_OP_LOG,
    CUTENSOR_OP_MAX,
    CUTENSOR_OP_MIN,
    CUTENSOR_OP_MISH,
    CUTENSOR_OP_MUL,
    CUTENSOR_OP_NEG,
    CUTENSOR_OP_RCP,
    CUTENSOR_OP_RELU,
    CUTENSOR_OP_SIGMOID,
    CUTENSOR_OP_SIN,
    CUTENSOR_OP_SINH,
    CUTENSOR_OP_SOFT_PLUS,
    CUTENSOR_OP_SOFT_SIGN,
    CUTENSOR_OP_SQRT,
    CUTENSOR_OP_SWISH,
    CUTENSOR_OP_TAN,
    CUTENSOR_OP_TANH,
)


# ---------------------------------------------------------------------------
# Unary / binary class factories.
#
# Each generated class is named ``CuTensor{CamelSlug}`` so that
# ``Benchmark._get_baseline_instance`` (which looks up
# ``f"CuTensor{suffix}"`` on the baseline module) resolves it exactly as
# it resolves cuTensor classes on NVIDIA / torch_npu_baseline classes on
# Ascend. The ``dtype`` kwarg matches the call site
# ``baseline_cls(dtype=dtype)``.
# ---------------------------------------------------------------------------
def _camel(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


def _unary(op, slug):
    cls = type(
        f"CuTensor{_camel(slug)}",
        (TorchUnaryBaseline,),
        {
            "__init__": lambda self, dtype=torch.float32, _op=op: TorchUnaryBaseline.__init__(
                self, op=_op, dtype=dtype
            ),
        },
    )
    return cls


def _binary(op, slug):
    cls = type(
        f"CuTensor{_camel(slug)}",
        (TorchBinaryBaseline,),
        {
            "__init__": lambda self, dtype=torch.float32, _op=op: TorchBinaryBaseline.__init__(
                self, op=_op, dtype=dtype
            ),
        },
    )
    return cls


# ---------------------------------------------------------------------------
# Unary baselines (one CuTensor{Op} per operator)
# ---------------------------------------------------------------------------
CuTensorAbs = _unary(CUTENSOR_OP_ABS, "abs")
CuTensorAcos = _unary(CUTENSOR_OP_ACOS, "acos")
CuTensorAcosh = _unary(CUTENSOR_OP_ACOSH, "acosh")
CuTensorAsin = _unary(CUTENSOR_OP_ASIN, "asin")
CuTensorAsinh = _unary(CUTENSOR_OP_ASINH, "asinh")
CuTensorAtan = _unary(CUTENSOR_OP_ATAN, "atan")
CuTensorAtanh = _unary(CUTENSOR_OP_ATANH, "atanh")
CuTensorCeil = _unary(CUTENSOR_OP_CEIL, "ceil")
CuTensorConj = _unary(CUTENSOR_OP_CONJ, "conj")
CuTensorCos = _unary(CUTENSOR_OP_COS, "cos")
CuTensorCosh = _unary(CUTENSOR_OP_COSH, "cosh")
CuTensorExp = _unary(CUTENSOR_OP_EXP, "exp")
CuTensorFloor = _unary(CUTENSOR_OP_FLOOR, "floor")
CuTensorIdentity = _unary(CUTENSOR_OP_IDENTITY, "identity")
CuTensorLog = _unary(CUTENSOR_OP_LOG, "log")
CuTensorMish = _unary(CUTENSOR_OP_MISH, "mish")
CuTensorNeg = _unary(CUTENSOR_OP_NEG, "neg")
CuTensorRcp = _unary(CUTENSOR_OP_RCP, "rcp")
CuTensorRelu = _unary(CUTENSOR_OP_RELU, "relu")
CuTensorSigmoid = _unary(CUTENSOR_OP_SIGMOID, "sigmoid")
CuTensorSin = _unary(CUTENSOR_OP_SIN, "sin")
CuTensorSinh = _unary(CUTENSOR_OP_SINH, "sinh")
CuTensorSoftPlus = _unary(CUTENSOR_OP_SOFT_PLUS, "soft_plus")
CuTensorSoftSign = _unary(CUTENSOR_OP_SOFT_SIGN, "soft_sign")
CuTensorSqrt = _unary(CUTENSOR_OP_SQRT, "sqrt")
CuTensorSwish = _unary(CUTENSOR_OP_SWISH, "swish")
CuTensorTan = _unary(CUTENSOR_OP_TAN, "tan")
CuTensorTanh = _unary(CUTENSOR_OP_TANH, "tanh")


# ---------------------------------------------------------------------------
# Binary baselines
# ---------------------------------------------------------------------------
CuTensorAdd = _binary(CUTENSOR_OP_ADD, "add")
CuTensorMul = _binary(CUTENSOR_OP_MUL, "mul")
CuTensorMax = _binary(CUTENSOR_OP_MAX, "max")
CuTensorMin = _binary(CUTENSOR_OP_MIN, "min")


# ---------------------------------------------------------------------------
# Contraction / trinary / block-sparse baselines.
#
# Subclass the Torch* counterparts without overriding anything — the
# benchmark harness instantiates them as ``cls(dtype=dtype)`` and calls
# ``prepare`` / ``__call__`` / ``build_kernel_callable``, all of which
# are inherited unchanged.
# ---------------------------------------------------------------------------
class CuTensorContraction(TorchContractionBaseline):
    pass


class CuTensorContractionTrinary(TorchContractionTrinaryBaseline):
    pass


class CuTensorTrinary(TorchTrinaryBaseline):
    def __init__(self, op_ab, op_abc, op_a='identity', op_b='identity',
                 op_c='identity', dtype=torch.float32):
        super().__init__(
            op_ab=op_ab, op_abc=op_abc,
            op_a=op_a, op_b=op_b, op_c=op_c,
            dtype=dtype,
        )


# Alias matching the operators.yaml name for the trinary elementwise op,
# so both the slug-derived and the explicit-suffix lookup paths resolve.
CuTensorElementwiseTrinary = CuTensorTrinary


class CuTensorBlockSparseContraction(TorchBlockSparseContractionBaseline):
    pass


# ---------------------------------------------------------------------------
# Function-style entry points (mirror flagtensor.cutensor API)
# ---------------------------------------------------------------------------
# ``benchmark/test_ElementwiseTrinary_perf.py`` calls the baseline through
# the function-style API: ``elementwise_trinary(a, b, c, **kwargs)`` and
# ``_get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, dtype)``. We
# expose the same names here so the benchmark harness can load them
# uniformly. (ElementwiseTrinary is not in the MetaX pilot op set, but the
# wiring is here for completeness when running the full suite via --ops.)
_TRINARY_EXECUTOR_CACHE = {}


def _get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, dtype):
    """Return a cached TorchTrinaryBaseline instance for the given op combo."""
    from flagtensor.cutensor import _resolve_operator, BINARY_OPERATOR_MAP, UNARY_OPERATOR_MAP
    key = (
        _resolve_operator(op_ab, BINARY_OPERATOR_MAP, "binary"),
        _resolve_operator(op_abc, BINARY_OPERATOR_MAP, "binary"),
        _resolve_operator(op_a, UNARY_OPERATOR_MAP, "unary"),
        _resolve_operator(op_b, UNARY_OPERATOR_MAP, "unary"),
        _resolve_operator(op_c, UNARY_OPERATOR_MAP, "unary"),
        dtype,
    )
    executor = _TRINARY_EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = TorchTrinaryBaseline(
            op_ab=op_ab, op_abc=op_abc,
            op_a=op_a, op_b=op_b, op_c=op_c,
            dtype=dtype,
        )
        _TRINARY_EXECUTOR_CACHE[key] = executor
    return executor


def elementwise_trinary(
    a, b, c, *,
    op_a="identity", op_b="identity", op_c="identity",
    op_ab="add", op_abc="add",
    alpha=1.0, beta=1.0, gamma=1.0,
    mode_a=None, mode_b=None, mode_c=None, mode_d=None,
    out=None,
):
    """MetaX-native elementwise trinary function (mirror of cutensor.elementwise_trinary)."""
    executor = _get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, a.dtype)
    return executor(
        a, b, c, alpha=alpha, beta=beta, gamma=gamma,
        mode_a=mode_a, mode_b=mode_b, mode_c=mode_c,
        mode_d=mode_d, out=out,
    )


# ---------------------------------------------------------------------------
# Registry: op_slug → baseline class
# ---------------------------------------------------------------------------
# Keys must match Benchmark._get_op_slug() output (lowercased op name with
# the CUTENSOR_OP_ prefix stripped, plus the special-case slugs used for
# contraction / trinary / block-sparse operators in conf/operators.yaml).
# Exposed both as BASELINE_CLASSES (vendor registry convention) and as
# module-level attributes (so getattr(module, "CuTensorSin") works).
BASELINE_CLASSES = {
    # unary
    'abs': CuTensorAbs,
    'acos': CuTensorAcos,
    'acosh': CuTensorAcosh,
    'asin': CuTensorAsin,
    'asinh': CuTensorAsinh,
    'atan': CuTensorAtan,
    'atanh': CuTensorAtanh,
    'ceil': CuTensorCeil,
    'conj': CuTensorConj,
    'cos': CuTensorCos,
    'cosh': CuTensorCosh,
    'exp': CuTensorExp,
    'floor': CuTensorFloor,
    'identity': CuTensorIdentity,
    'log': CuTensorLog,
    'mish': CuTensorMish,
    'neg': CuTensorNeg,
    'rcp': CuTensorRcp,
    'relu': CuTensorRelu,
    'sigmoid': CuTensorSigmoid,
    'sin': CuTensorSin,
    'sinh': CuTensorSinh,
    'soft_plus': CuTensorSoftPlus,
    'soft_sign': CuTensorSoftSign,
    'sqrt': CuTensorSqrt,
    'swish': CuTensorSwish,
    'tan': CuTensorTan,
    'tanh': CuTensorTanh,
    # binary
    'add': CuTensorAdd,
    'mul': CuTensorMul,
    'max': CuTensorMax,
    'min': CuTensorMin,
    # contraction / trinary / block-sparse
    'contraction': CuTensorContraction,
    'contraction_trinary': CuTensorContractionTrinary,
    'elementwise_trinary': CuTensorElementwiseTrinary,
    'block_sparse_contraction': CuTensorBlockSparseContraction,
}


__all__ = [
    'BASELINE_CLASSES',
    'elementwise_trinary',
    '_get_trinary_executor',
] + [name for name in dir() if name.startswith('CuTensor')]
