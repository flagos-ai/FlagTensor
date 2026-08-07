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

"""NVIDIA-native baseline classes for FlagTensor benchmarks.

On NVIDIA GPUs the vendor-supplied baseline for FlagTensor is cuTensor
(``libcutensor.so``), which exposes the same tensor-contraction /
elementwise-trinary API that FlagTensor's operators mirror. The cuTensor
implementation lives in :mod:`flagtensor.cutensor` (``CuTensorAbs``,
``CuTensorAdd``, ``CuTensorContraction`` etc.).

This module exposes a uniform ``BASELINE_CLASSES`` registry — keyed by
the same op-slug convention used by :mod:`flagtensor.torch_baseline` —
so that ``benchmark_core`` can load the right baseline per vendor without
hardcoding the ``CuTensor`` prefix anywhere outside this module.

When cuTensor is not installed on the host (``CUTENSOR_AVAILABLE = False``),
all entries resolve to ``None`` and the benchmark tests are skipped with
a clear "cuTensor unavailable" message — preserving the original NVIDIA
behaviour exactly.
"""

from flagtensor.cutensor import (
    CUTENSOR_AVAILABLE,
    CuTensorAbs,
    CuTensorAcosh,
    CuTensorAcos,
    CuTensorAdd,
    CuTensorAsinh,
    CuTensorAsin,
    CuTensorAtanh,
    CuTensorAtan,
    CuTensorBlockSparseContraction,
    CuTensorCeil,
    CuTensorConj,
    CuTensorContraction,
    CuTensorContractionTrinary,
    CuTensorCosh,
    CuTensorCos,
    CuTensorExp,
    CuTensorFloor,
    CuTensorIdentity,
    CuTensorLog,
    CuTensorMax,
    CuTensorMin,
    CuTensorMish,
    CuTensorMul,
    CuTensorNeg,
    CuTensorRcp,
    CuTensorRelu,
    CuTensorSigmoid,
    CuTensorSin,
    CuTensorSinh,
    CuTensorSoftPlus,
    CuTensorSoftSign,
    CuTensorSqrt,
    CuTensorSwish,
    CuTensorTan,
    CuTensorTanh,
    CuTensorTrinary,
    _get_trinary_executor,
    elementwise_trinary,
)


# When cuTensor is unavailable, every entry resolves to None — the
# benchmark harness will then skip the test with "cuTensor unavailable",
# exactly as it did before the vendor-aware baseline refactor.
_C = (lambda slug, cls: cls) if CUTENSOR_AVAILABLE else (lambda slug, cls: None)


BASELINE_CLASSES = {
    # unary
    'abs': _C('abs', CuTensorAbs),
    'acos': _C('acos', CuTensorAcos),
    'acosh': _C('acosh', CuTensorAcosh),
    'asin': _C('asin', CuTensorAsin),
    'asinh': _C('asinh', CuTensorAsinh),
    'atan': _C('atan', CuTensorAtan),
    'atanh': _C('atanh', CuTensorAtanh),
    'ceil': _C('ceil', CuTensorCeil),
    'conj': _C('conj', CuTensorConj),
    'cos': _C('cos', CuTensorCos),
    'cosh': _C('cosh', CuTensorCosh),
    'exp': _C('exp', CuTensorExp),
    'floor': _C('floor', CuTensorFloor),
    'identity': _C('identity', CuTensorIdentity),
    'log': _C('log', CuTensorLog),
    'mish': _C('mish', CuTensorMish),
    'neg': _C('neg', CuTensorNeg),
    'rcp': _C('rcp', CuTensorRcp),
    'relu': _C('relu', CuTensorRelu),
    'sigmoid': _C('sigmoid', CuTensorSigmoid),
    'sin': _C('sin', CuTensorSin),
    'sinh': _C('sinh', CuTensorSinh),
    'soft_plus': _C('soft_plus', CuTensorSoftPlus),
    'soft_sign': _C('soft_sign', CuTensorSoftSign),
    'sqrt': _C('sqrt', CuTensorSqrt),
    'swish': _C('swish', CuTensorSwish),
    'tan': _C('tan', CuTensorTan),
    'tanh': _C('tanh', CuTensorTanh),
    # binary
    'add': _C('add', CuTensorAdd),
    'mul': _C('mul', CuTensorMul),
    'max': _C('max', CuTensorMax),
    'min': _C('min', CuTensorMin),
    # contraction / trinary / block-sparse
    'contraction': _C('contraction', CuTensorContraction),
    'contraction_trinary': _C('contraction_trinary', CuTensorContractionTrinary),
    'elementwise_trinary': _C('elementwise_trinary', CuTensorTrinary),
    'block_sparse_contraction': _C('block_sparse_contraction', CuTensorBlockSparseContraction),
}


__all__ = [
    'BASELINE_CLASSES',
    'elementwise_trinary',
    '_get_trinary_executor',
]
