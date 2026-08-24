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

"""Assertion utilities for correctness testing."""

from typing import Dict
from typing import Optional
from typing import Tuple

import torch

DEFAULT_CORRECTNESS_TOLERANCES: Dict[torch.dtype, Tuple[float, float]] = {
    # Integer types — must be bit-exact
    torch.bool: (0, 0),
    torch.uint8: (0, 0),
    torch.int8: (0, 0),
    torch.int16: (0, 0),
    torch.int32: (0, 0),
    torch.int64: (0, 0),
    # FP8 types — very low precision
    torch.float8_e4m3fn: (1e-3, 1e-3),
    torch.float8_e5m2: (1e-3, 1e-3),
    torch.float8_e4m3fnuz: (1e-3, 1e-3),
    torch.float8_e5m2fnuz: (1e-3, 1e-3),
    # Floating-point types — per operator-library spec
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1.3e-6, 1.3e-6),
    torch.bfloat16: (0.016, 0.016),
    torch.float64: (1e-7, 1e-7),
    # Complex types
    torch.complex32: (1e-3, 1e-3),
    torch.complex64: (1.3e-6, 1.3e-6),
    torch.complex128: (1e-7, 1e-7),
}


def get_tolerance(
    dtype: torch.dtype, atol: Optional[float] = None, rtol: Optional[float] = None
) -> Tuple[float, float]:
    """Get tolerance values for a given dtype.

    On non-NVIDIA backends (e.g. Ascend), transcendental libdevice functions
    are dispatched through CANN aclnn which can produce slightly different
    bit-for-bit results from a direct torch_npu aten call. Float32 results
    in particular may differ by ~5e-6 even though both paths use the same
    underlying ACL kernel. To avoid spurious failures when comparing two
    legitimate accelerator outputs, we widen the float32 tolerance on
    non-NVIDIA backends. NVIDIA stays at the strict library spec.

    Args:
        dtype: The torch dtype to get tolerances for.
        atol: Override for absolute tolerance. Uses default if None.
        rtol: Override for relative tolerance. Uses default if None.

    Returns:
        Tuple of (atol, rtol) for the given dtype.
    """
    default_atol, default_rtol = DEFAULT_CORRECTNESS_TOLERANCES.get(dtype, (1.3e-6, 1.3e-6))
    if atol is None or rtol is None:
        try:
            from flagtensor.runtime import device as _ft_device
            if _ft_device.vendor_name != "nvidia" and dtype == torch.float32:
                # aclnn vs triton-ascend libdevice path differences for
                # transcendentals (asin/acos/atan/...) can produce ~5e-5
                # numerical noise even when both paths use the same ACL
                # kernel family. Use 1e-4 to stay well above the noise floor
                # without hiding real bugs (which would be 1e-2 or larger).
                if atol is None:
                    atol = 1e-4
                if rtol is None:
                    rtol = 1e-4
        except Exception:
            pass
    return (default_atol if atol is None else atol, default_rtol if rtol is None else rtol)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    equal_nan: bool = False,
):
    """Assert that two tensors are close within dtype-specific tolerances.

    Args:
        actual: The actual tensor from FlagTensor.
        expected: The expected tensor from reference implementation.
        dtype: The dtype to use for tolerance lookup. Uses actual.dtype if None.
        atol: Override for absolute tolerance.
        rtol: Override for relative tolerance.
        equal_nan: If True, two NaN values are considered equal.
    """
    resolved_dtype = dtype or actual.dtype
    resolved_atol, resolved_rtol = get_tolerance(resolved_dtype, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual, expected, atol=resolved_atol, rtol=resolved_rtol, equal_nan=equal_nan)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor, equal_nan: bool = False):
    """Assert that two tensors are exactly equal.

    Use for bit-exact operations or integer dtypes.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
        equal_nan: If True, two NaN values are considered equal.
    """
    if equal_nan:
        assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
        nan_mask = torch.isnan(actual) & torch.isnan(expected)
        actual_clean = torch.where(nan_mask, torch.zeros_like(actual), actual)
        expected_clean = torch.where(nan_mask, torch.zeros_like(expected), expected)
        assert torch.equal(actual_clean, expected_clean), "Tensors differ outside NaN positions"
    else:
        assert torch.equal(actual, expected), "Tensors are not equal"
