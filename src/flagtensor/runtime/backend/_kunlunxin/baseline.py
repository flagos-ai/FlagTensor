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

"""Kunlunxin-native baseline classes for FlagTensor benchmarks.

On the Kunlunxin XPU (P800 / R200, CUDA-compatible via ``torch_xmlir``):

* **No usable cuTensor exists.**  The SDK does not ship ``libcutensor.so``
  (or any cuTensor-equivalent generalized tensor-contraction /
  elementwise-trinary library), so ``flagtensor.cutensor`` marks cuTensor
  unavailable on this platform.

* **PyTorch native ops DO use vendor-optimised kernels.**  ``torch.matmul``
  / ``torch.addmm`` / ``torch.einsum`` and the elementwise aten ops
  dispatch through the ``torch_xmlir`` symbol-rewrite hook to Kunlunxin's
  vendor libraries (``libxpu_blas.so``, ``libxpu_dnn.so``, ...) — the same
  position cuTensor occupies on NVIDIA.

* **This mirrors the FlagGems convention** and the PPU / Iluvatar
  backends: for platforms without a cuTensor-equivalent vendor library,
  ``torch.*`` ops are the native baseline.  This is NOT a "fallback" — it
  is the vendor-supplied optimised path on Kunlunxin.

The classes here mirror the public interface of the ``CuTensor*``
baseline classes in :mod:`flagtensor.cutensor` (``prepare`` /
``__call__`` / ``build_kernel_callable``) so the benchmark harness can
treat them uniformly.  The actual implementation lives in
:mod:`flagtensor.torch_baseline`.

In addition to the ``BASELINE_CLASSES`` registry (keyed by op-slug), this
module also exposes ``CuTensor{SlugCamelCase}`` aliases so that
``Benchmark._get_baseline_instance()`` — which resolves classes via
``getattr(baseline_module, "CuTensorAbs")`` — works transparently on
Kunlunxin, exactly as it does on Ascend via ``torch_npu_baseline``.
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
# Unary baselines — one subclass per operator so the benchmark harness can
# instantiate them by op slug (matching the CuTensorXxx naming convention).
# ---------------------------------------------------------------------------
def _unary(op):
    class _B(TorchUnaryBaseline):
        def __init__(self, dtype=torch.float32):
            super().__init__(op=op, dtype=dtype)
    return _B


# ---------------------------------------------------------------------------
# Vendor-unsupported op overrides
# ---------------------------------------------------------------------------
# ``torch_xmlir`` on Kunlunxin XPU does NOT implement ``torch.cosh``,
# ``torch.acosh``, ``torch.asinh``, ``torch.atanh``, and ``torch.conj``
# (they raise ``[NOT IMPLEMENTED]: error code= 4``). The benchmark
# baseline calls these torch ops, so we provide custom baseline classes
# that compute the same result using only supported torch ops (exp, log,
# sqrt, sin, cos — which torch_xmlir DOES implement).
class _UnsupportedUnaryBaseline(TorchUnaryBaseline):
    """Base class for baselines that need a custom torch expression.

    Subclasses override ``_compute(x)`` to return the result tensor.
    """
    def __init__(self, op, dtype=torch.float32):
        super().__init__(op=op, dtype=dtype)

    def _compute(self, x):
        raise NotImplementedError

    def build_kernel_callable(self, x, alpha=1.0):
        self.prepare(x)
        y = torch.empty_like(x)

        def run_kernel():
            torch.mul(self._compute(x), alpha, out=y)
            return y

        return run_kernel

    def __call__(self, x, alpha=1.0):
        self.prepare(x)
        return alpha * self._compute(x)


class BaselineCosh(_UnsupportedUnaryBaseline):
    def __init__(self, dtype=torch.float32):
        super().__init__(op=CUTENSOR_OP_COSH, dtype=dtype)
    def _compute(self, x):
        xf = x.float()
        return (0.5 * (torch.exp(xf) + torch.exp(-xf))).to(x.dtype)


class BaselineAcosh(_UnsupportedUnaryBaseline):
    def __init__(self, dtype=torch.float32):
        super().__init__(op=CUTENSOR_OP_ACOSH, dtype=dtype)
    def _compute(self, x):
        xf = x.float()
        return torch.log(xf + torch.sqrt(xf * xf - 1.0)).to(x.dtype)


class BaselineAsinh(_UnsupportedUnaryBaseline):
    def __init__(self, dtype=torch.float32):
        super().__init__(op=CUTENSOR_OP_ASINH, dtype=dtype)
    def _compute(self, x):
        xf = x.float()
        return (torch.sign(xf) * torch.log(torch.abs(xf) + torch.sqrt(xf * xf + 1.0))).to(x.dtype)


class BaselineAtanh(_UnsupportedUnaryBaseline):
    def __init__(self, dtype=torch.float32):
        super().__init__(op=CUTENSOR_OP_ATANH, dtype=dtype)
    def _compute(self, x):
        xf = x.float()
        return (0.5 * torch.log((1.0 + xf) / (1.0 - xf))).to(x.dtype)


class BaselineConj(_UnsupportedUnaryBaseline):
    def __init__(self, dtype=torch.float32):
        super().__init__(op=CUTENSOR_OP_CONJ, dtype=dtype)
    def _compute(self, x):
        if x.dtype == torch.complex64:
            real_dtype = torch.float32
            x_view = x.view(real_dtype)
            y = torch.empty_like(x)
            y_view = y.view(real_dtype)
            y_view.copy_(x_view)
            y_view[..., 1].neg_()
            return y
        return x.clone()


BaselineAbs = _unary(CUTENSOR_OP_ABS)
BaselineAcos = _unary(CUTENSOR_OP_ACOS)
BaselineAsin = _unary(CUTENSOR_OP_ASIN)
BaselineAtan = _unary(CUTENSOR_OP_ATAN)
BaselineCeil = _unary(CUTENSOR_OP_CEIL)
BaselineCos = _unary(CUTENSOR_OP_COS)
BaselineExp = _unary(CUTENSOR_OP_EXP)
BaselineFloor = _unary(CUTENSOR_OP_FLOOR)
BaselineIdentity = _unary(CUTENSOR_OP_IDENTITY)
BaselineLog = _unary(CUTENSOR_OP_LOG)
BaselineMish = _unary(CUTENSOR_OP_MISH)
BaselineNeg = _unary(CUTENSOR_OP_NEG)
BaselineRcp = _unary(CUTENSOR_OP_RCP)
BaselineRelu = _unary(CUTENSOR_OP_RELU)
BaselineSigmoid = _unary(CUTENSOR_OP_SIGMOID)
BaselineSin = _unary(CUTENSOR_OP_SIN)
class BaselineSinh(_UnsupportedUnaryBaseline):
    def __init__(self, dtype=torch.float32):
        super().__init__(op=CUTENSOR_OP_SINH, dtype=dtype)
    def _compute(self, x):
        xf = x.float()
        return (0.5 * (torch.exp(xf) - torch.exp(-xf))).to(x.dtype)
BaselineSoftPlus = _unary(CUTENSOR_OP_SOFT_PLUS)
BaselineSoftSign = _unary(CUTENSOR_OP_SOFT_SIGN)
BaselineSqrt = _unary(CUTENSOR_OP_SQRT)
BaselineSwish = _unary(CUTENSOR_OP_SWISH)
BaselineTan = _unary(CUTENSOR_OP_TAN)
BaselineTanh = _unary(CUTENSOR_OP_TANH)


# ---------------------------------------------------------------------------
# Binary baselines
# ---------------------------------------------------------------------------
def _binary(op):
    class _B(TorchBinaryBaseline):
        def __init__(self, dtype=torch.float32):
            super().__init__(op=op, dtype=dtype)
    return _B


BaselineAdd = _binary(CUTENSOR_OP_ADD)
BaselineMul = _binary(CUTENSOR_OP_MUL)
BaselineMax = _binary(CUTENSOR_OP_MAX)
BaselineMin = _binary(CUTENSOR_OP_MIN)


# ---------------------------------------------------------------------------
# Contraction / trinary / block-sparse baselines
# ---------------------------------------------------------------------------
class BaselineContraction(TorchContractionBaseline):
    pass


class BaselineContractionTrinary(TorchContractionTrinaryBaseline):
    pass


class BaselineElementwiseTrinary(TorchTrinaryBaseline):
    def __init__(self, op_ab, op_abc, op_a='identity', op_b='identity',
                 op_c='identity', dtype=torch.float32):
        super().__init__(
            op_ab=op_ab, op_abc=op_abc,
            op_a=op_a, op_b=op_b, op_c=op_c,
            dtype=dtype,
        )


class BaselineBlockSparseContraction(TorchBlockSparseContractionBaseline):
    pass


# ---------------------------------------------------------------------------
# Function-style entry points (mirror flagtensor.cutensor API)
# ---------------------------------------------------------------------------
# Some benchmark tests (test_ElementwiseTrinary_perf.py) call the baseline
# through the function-style API: `elementwise_trinary(a, b, c, **kwargs)`
# and `_get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, dtype)`. We
# expose the same names here so the benchmark harness can load them
# uniformly via benchmark_core.get_baseline_module().
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
    """Kunlunxin-native elementwise trinary function (mirror of cutensor.elementwise_trinary)."""
    executor = _get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, a.dtype)
    return executor(
        a, b, c, alpha=alpha, beta=beta, gamma=gamma,
        mode_a=mode_a, mode_b=mode_b, mode_c=mode_c,
        mode_d=mode_d, out=out,
    )


# ---------------------------------------------------------------------------
# CuTensor* aliases
# ---------------------------------------------------------------------------
# Benchmark._get_baseline_instance() resolves baseline classes via
# ``getattr(baseline_module, "CuTensor{SlugCamelCase}")``.  We expose
# identically-named aliases pointing at the Baseline* classes above so the
# same resolution logic works on Kunlunxin without modifying benchmark_core.
def _camel(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


_CUTENSOR_ALIASES = {
    f"CuTensor{_camel(slug)}": cls
    for slug, cls in (
        ('abs', BaselineAbs), ('acos', BaselineAcos), ('acosh', BaselineAcosh),
        ('asin', BaselineAsin), ('asinh', BaselineAsinh), ('atan', BaselineAtan),
        ('atanh', BaselineAtanh), ('ceil', BaselineCeil), ('conj', BaselineConj),
        ('cos', BaselineCos), ('cosh', BaselineCosh), ('exp', BaselineExp),
        ('floor', BaselineFloor), ('identity', BaselineIdentity), ('log', BaselineLog),
        ('mish', BaselineMish), ('neg', BaselineNeg), ('rcp', BaselineRcp),
        ('relu', BaselineRelu), ('sigmoid', BaselineSigmoid), ('sin', BaselineSin),
        ('sinh', BaselineSinh), ('soft_plus', BaselineSoftPlus),
        ('soft_sign', BaselineSoftSign), ('sqrt', BaselineSqrt),
        ('swish', BaselineSwish), ('tan', BaselineTan), ('tanh', BaselineTanh),
        ('add', BaselineAdd), ('mul', BaselineMul),
        ('max', BaselineMax), ('min', BaselineMin),
    )
}
# Contraction / trinary / block-sparse use specific class-name suffixes
# (see Benchmark._OP_NAME_TO_BASELINE_SUFFIX).
_CUTENSOR_ALIASES["CuTensorContraction"] = BaselineContraction
_CUTENSOR_ALIASES["CuTensorContractionTrinary"] = BaselineContractionTrinary
_CUTENSOR_ALIASES["CuTensorTrinary"] = BaselineElementwiseTrinary
_CUTENSOR_ALIASES["CuTensorElementwiseTrinary"] = BaselineElementwiseTrinary
_CUTENSOR_ALIASES["CuTensorBlockSparseContraction"] = BaselineBlockSparseContraction

globals().update(_CUTENSOR_ALIASES)


# ---------------------------------------------------------------------------
# Registry: op_slug → baseline class
# ---------------------------------------------------------------------------
# Keys must match Benchmark._get_op_slug() output (lowercased op name with
# the CUTENSOR_OP_ prefix stripped, plus the special-case slugs used for
# contraction / trinary / block-sparse operators in conf/operators.yaml).
BASELINE_CLASSES = {
    # unary
    'abs': BaselineAbs,
    'acos': BaselineAcos,
    'acosh': BaselineAcosh,
    'asin': BaselineAsin,
    'asinh': BaselineAsinh,
    'atan': BaselineAtan,
    'atanh': BaselineAtanh,
    'ceil': BaselineCeil,
    'conj': BaselineConj,
    'cos': BaselineCos,
    'cosh': BaselineCosh,
    'exp': BaselineExp,
    'floor': BaselineFloor,
    'identity': BaselineIdentity,
    'log': BaselineLog,
    'mish': BaselineMish,
    'neg': BaselineNeg,
    'rcp': BaselineRcp,
    'relu': BaselineRelu,
    'sigmoid': BaselineSigmoid,
    'sin': BaselineSin,
    'sinh': BaselineSinh,
    'soft_plus': BaselineSoftPlus,
    'soft_sign': BaselineSoftSign,
    'sqrt': BaselineSqrt,
    'swish': BaselineSwish,
    'tan': BaselineTan,
    'tanh': BaselineTanh,
    # binary
    'add': BaselineAdd,
    'mul': BaselineMul,
    'max': BaselineMax,
    'min': BaselineMin,
    # contraction / trinary / block-sparse
    'contraction': BaselineContraction,
    'contraction_trinary': BaselineContractionTrinary,
    'elementwise_trinary': BaselineElementwiseTrinary,
    'block_sparse_contraction': BaselineBlockSparseContraction,
}


__all__ = [
    'BASELINE_CLASSES',
    'elementwise_trinary',
    '_get_trinary_executor',
] + [name for name in dir() if name.startswith('Baseline')]
