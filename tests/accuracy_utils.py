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

"""Accuracy testing utilities for FlagTensor.

Provides FlagGems-compatible assertion wrappers and shared test constants
for per-operator correctness tests.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flagtensor.testing import assert_close as _assert_close
from flagtensor.testing import assert_equal as _assert_equal
from flagtensor.testing import correctness_dtypes
from flagtensor.testing import DEFAULT_CONTRACTION_TEST_SHAPES
from flagtensor.testing import DEFAULT_CORRECTNESS_TOLERANCES
from flagtensor.testing import DEFAULT_POINTWISE_TEST_SHAPES
from flagtensor.testing import default_binary_shapes
from flagtensor.testing import default_contraction_shapes
from flagtensor.testing import default_pointwise_shapes
from flagtensor.testing import get_tolerance
from flagtensor.runtime import device as runtime_device

# ---------------------------------------------------------------------------
# Device capability flags (matching FlagGems pattern)
# ---------------------------------------------------------------------------
fp64_is_supported = runtime_device.support_fp64
bf16_is_supported = runtime_device.support_bf16
int64_is_supported = runtime_device.support_int64

# ---------------------------------------------------------------------------
# Shared dtype constants (matching FlagGems pattern)
# ---------------------------------------------------------------------------
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
FLOAT_DTYPES_NO_BF16 = [torch.float16, torch.float32]
ALL_FLOAT_DTYPES = FLOAT_DTYPES + ([torch.float64] if fp64_is_supported else [])
INT_DTYPES = [torch.int32] + ([torch.int64] if int64_is_supported else [])
BOOL_TYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex64]

# ---------------------------------------------------------------------------
# Shared shape constants (matching FlagGems pattern)
# ---------------------------------------------------------------------------
POINTWISE_SHAPES = list(DEFAULT_POINTWISE_TEST_SHAPES)
CONTRACTION_SHAPES = list(DEFAULT_CONTRACTION_TEST_SHAPES)


# ---------------------------------------------------------------------------
# Reference utilities
# ---------------------------------------------------------------------------
def to_reference(inp, upcast=False):
    """Move input to CPU and optionally upcast to float64 for golden reference.

    Args:
        inp: Input tensor (may be None).
        upcast: If True, upcast floating types to float64 and complex to complex128.

    Returns:
        CPU tensor suitable as golden reference, or None if inp is None.
    """
    if inp is None:
        return None
    ref = inp.detach().cpu()
    if not upcast:
        return ref
    if ref.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return ref.to(torch.float64)
    if ref.dtype in (torch.complex32, torch.complex64):
        return ref.to(torch.complex128)
    return ref


# ---------------------------------------------------------------------------
# FlagGems-compatible assertion wrappers
# ---------------------------------------------------------------------------
def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=None):
    """Assert that ``res`` is close to ``ref`` for the given ``dtype``.

    Moves both tensors to CPU before comparing, using the tolerance table
    defined in ``flagtensor.testing.assertions``.

    Args:
        res:  GPU result tensor (or already on CPU).
        ref:  Reference tensor (typically CPU-FP64).
        dtype: Torch dtype used to look up tolerances.
        equal_nan: If True, treat two NaNs as equal.
        reduce_dim: Unused (kept for FlagGems API compatibility).
        atol: Optional absolute tolerance override.
    """
    res_cpu = res.detach().cpu() if res.is_cuda else res.detach()
    ref_cpu = ref.detach().cpu() if ref.is_cuda else ref.detach()

    if dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
        if ref_cpu.dtype != dtype:
            ref_cpu = ref_cpu.to(dtype)
    _assert_close(res_cpu, ref_cpu, dtype=dtype, atol=atol, equal_nan=equal_nan)


def gems_assert_equal(res, ref, equal_nan=False):
    """Assert that ``res`` is bit-exact equal to ``ref``.

    Args:
        res:  GPU result tensor (or already on CPU).
        ref:  Reference tensor.
        equal_nan: If True, treat two NaNs as equal.
    """
    res_cpu = res.detach().cpu() if res.is_cuda else res.detach()
    ref_cpu = ref.detach().cpu() if ref.is_cuda else ref.detach()
    _assert_equal(res_cpu, ref_cpu, equal_nan=equal_nan)


# Legacy re-exports (backward compatibility)
__all__ = [
    # Device capability flags (FlagGems style)
    "fp64_is_supported",
    "bf16_is_supported",
    "int64_is_supported",
    # Shared constants (FlagGems style)
    "FLOAT_DTYPES",
    "FLOAT_DTYPES_NO_BF16",
    "ALL_FLOAT_DTYPES",
    "INT_DTYPES",
    "BOOL_TYPES",
    "COMPLEX_DTYPES",
    "POINTWISE_SHAPES",
    "CONTRACTION_SHAPES",
    # Reference utilities
    "to_reference",
    # Assertion wrappers (FlagGems style)
    "gems_assert_close",
    "gems_assert_equal",
    # Legacy re-exports
    "assert_close",
    "assert_equal",
    "correctness_dtypes",
    "DEFAULT_CONTRACTION_TEST_SHAPES",
    "DEFAULT_CORRECTNESS_TOLERANCES",
    "DEFAULT_POINTWISE_TEST_SHAPES",
    "default_binary_shapes",
    "default_contraction_shapes",
    "default_pointwise_shapes",
    "get_tolerance",
]
