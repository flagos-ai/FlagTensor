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

import os
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import yaml

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

    Args:
        dtype: The torch dtype to get tolerances for.
        atol: Override for absolute tolerance. Uses default if None.
        rtol: Override for relative tolerance. Uses default if None.

    Returns:
        Tuple of (atol, rtol) for the given dtype.
    """
    default_atol, default_rtol = DEFAULT_CORRECTNESS_TOLERANCES.get(dtype, (1.3e-6, 1.3e-6))
    return (default_atol if atol is None else atol, default_rtol if rtol is None else rtol)


# ---------------------------------------------------------------------------
# Vendor-specific benchmark-verify floor tolerance
# ---------------------------------------------------------------------------
# Cached per-vendor floor to avoid re-reading the yaml on every benchmark
# iteration. Keyed by vendor_name (e.g. "nvidia", "ppu").
_VENDOR_FLOOR_CACHE: Dict[str, Tuple[float, float]] = {}
_VENDOR_FLOOR_BY_OP_CACHE: Dict[Tuple[str, str], Tuple[float, float]] = {}
_VENDOR_YAML_CACHE: Dict[str, dict] = {}


def _load_vendor_tolerances_yaml(vendor_name: str) -> dict:
    """Load and cache the raw yaml dict for a vendor's tolerances.yaml."""
    if vendor_name in _VENDOR_YAML_CACHE:
        return _VENDOR_YAML_CACHE[vendor_name]
    data = {}
    try:
        backend_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        yaml_path = os.path.join(
            backend_dir, "runtime", "backend",
            f"_{vendor_name}", "tolerances.yaml",
        )
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass
    _VENDOR_YAML_CACHE[vendor_name] = data
    return data


def get_vendor_benchmark_floor(vendor_name: str, op_slug: Optional[str] = None) -> Tuple[float, float]:
    """Return the (atol, rtol) floor used by Benchmark.verify() for a vendor.

    The floor is loaded from ``_<vendor>/tolerances.yaml``. Two layers:

    1. ``benchmark_verify_floor`` (top-level) — vendor-wide default floor.
    2. ``benchmark_verify_floor_by_op.<op_slug>`` — per-op override that
       takes precedence when ``op_slug`` is provided. Used to give
       contraction-family ops a larger atol on PPU where acblas's GEMM
       summation order differs from the Triton kernel, producing
       ~3e-3 absolute differences on near-zero outputs.

    NVIDIA keeps 1e-4 (historical value) at both layers. PPU uses 1e-3
    vendor-wide and 5e-3 atol for contraction-family ops.

    If the vendor's yaml is missing or unreadable, falls back to
    (1e-4, 1e-4) to preserve the historical NVIDIA behaviour for any
    vendor that has not yet shipped a tolerances.yaml.

    Args:
        vendor_name: Active vendor name (e.g. "nvidia", "ppu").
        op_slug: Optional operator slug (e.g. "contraction"). When
            provided and the yaml declares a per-op override, that
            override is returned instead of the vendor-wide default.
    """
    if op_slug is not None:
        cache_key = (vendor_name, op_slug)
        if cache_key in _VENDOR_FLOOR_BY_OP_CACHE:
            return _VENDOR_FLOOR_BY_OP_CACHE[cache_key]
    else:
        if vendor_name in _VENDOR_FLOOR_CACHE:
            return _VENDOR_FLOOR_CACHE[vendor_name]

    data = _load_vendor_tolerances_yaml(vendor_name)
    floor_cfg = data.get("benchmark_verify_floor", {}) or {}
    floor = (
        float(floor_cfg.get("atol", 1e-4)),
        float(floor_cfg.get("rtol", 1e-4)),
    )

    # Per-op override takes precedence when present
    if op_slug is not None:
        by_op = data.get("benchmark_verify_floor_by_op", {}) or {}
        op_cfg = by_op.get(op_slug, {}) or {}
        if op_cfg:
            floor = (
                float(op_cfg.get("atol", floor[0])),
                float(op_cfg.get("rtol", floor[1])),
            )
        _VENDOR_FLOOR_BY_OP_CACHE[(vendor_name, op_slug)] = floor
    else:
        _VENDOR_FLOOR_CACHE[vendor_name] = floor

    return floor


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
