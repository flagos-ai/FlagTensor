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

"""Dtype capability registry for FlagTensor.

Central authority for determining which dtypes are supported per vendor/arch.
Follows the same pattern as ``runtime.common.UNSUPPORT_*`` in FlagGems.

Usage::

    from flagtensor.runtime.dtype_capability import dtype_capability

    supported = dtype_capability.supported_dtypes  # set of torch dtypes
    print(dtype_capability.supported_fp8)          # set of torch fp8 dtypes
    print(dtype_capability.supported_int)           # set of torch int dtypes
"""

import torch

from flagtensor.runtime.backend.device import DeviceDetector

# ── Per-vendor dtype definitions ──────────────────────────────────────────
# Each vendor maps arch → set of supported torch dtypes.
# Triton 3.3 + flagtree limitations are factored into these tables.

_VENDOR_DTYPE_SUPPORT: dict[str, dict[str, set[torch.dtype]]] = {
    "nvidia": {
        "ampere": {
            # A100 (SM80)
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.int8,
            torch.float8_e5m2,       # tl.float8e5  ✅ load/store + abs
        },
        "hopper": {
            # H100/H200 (SM90) — not tested but logically should support more
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.int8,
            torch.float8_e5m2,
            torch.float8_e4m3fn,     # tl.float8e4nv only available on Hopper
        },
    },
    # Future vendors:
    # "aipu": {"aipu": {torch.float16, torch.float32, torch.int8}},
}

# ── Default fallback (any vendor not listed) ──────────────────────────────
_DEFAULT_SUPPORTED = {torch.float16, torch.float32, torch.bfloat16}

# ── cuTensor baseline dtype support ───────────────────────────────────────
_CUTENSOR_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float16: "CUDA_R_16F",
    torch.float32: "CUDA_R_32F",
    torch.float64: "CUDA_R_64F",
    torch.bfloat16: "CUDA_R_16BF",
    torch.complex64: "CUDA_C_32F",
    torch.complex128: "CUDA_C_64F",
    # cuTensor FP8 support available via CUTENSOR_COMPUTE_DESC_* constants
    torch.float8_e5m2: "CUDA_R_8E5M2",     # defined in cuTensor 2.x
    torch.float8_e4m3fn: "CUDA_R_8E4M3",   # defined in cuTensor 2.x
    torch.int8: "CUDA_R_8I",
}

_ACCUMULATOR_DTYPE_MAP: dict[torch.dtype, torch.dtype] = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.float8_e5m2: torch.float32,
}


class DtypeCapability:
    """Singleton that caches the detected device → supported dtypes."""

    _instance: "DtypeCapability | None" = None

    def __new__(cls) -> "DtypeCapability":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self._device = DeviceDetector()
        self._vendor = self._device.vendor_name
        self._arch = self._detect_arch()

        self._supported = self._resolve_supported()
        self._fp8 = {d for d in self._supported if "float8" in str(d)}
        self._int = {d for d in self._supported if "int" in str(d) and d is not torch.int64}
        self._float = {d for d in self._supported if d.is_floating_point}

    # ── Public properties ──────────────────────────────────────────────

    @property
    def vendor(self) -> str:
        return self._vendor

    @property
    def arch(self) -> str | None:
        return self._arch

    @property
    def supported_dtypes(self) -> set[torch.dtype]:
        """All dtypes supported by the current vendor/arch combination."""
        return self._supported

    @property
    def supported_fp8(self) -> set[torch.dtype]:
        """FP8 dtypes supported (subset of supported_dtypes)."""
        return self._fp8

    @property
    def supported_int(self) -> set[torch.dtype]:
        """Integer dtypes supported (subset of supported_dtypes)."""
        return self._int

    @property
    def supported_float(self) -> set[torch.dtype]:
        """Floating-point dtypes supported."""
        return self._float

    def is_supported(self, dtype: torch.dtype) -> bool:
        """Check if a dtype is supported on the current device."""
        return dtype in self._supported

    def accumulate_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Return the accumulation dtype (e.g. f32 for f16 compute)."""
        return _ACCUMULATOR_DTYPE_MAP.get(dtype, dtype)

    def cutensor_type(self, dtype: torch.dtype) -> str | None:
        """Return cuTensor C enum name for a dtype, or None if unmapped."""
        return _CUTENSOR_DTYPE_MAP.get(dtype)

    # ── Internal helpers ───────────────────────────────────────────────

    def _detect_arch(self) -> str | None:
        try:
            props = torch.cuda.get_device_properties(0)
            sm = props.major
            # Match against vendor's ARCH_MAP
            from flagtensor.runtime import backend

            mod = backend.get_vendor_module(self._vendor)
            arch_map = getattr(mod, "ARCH_MAP", {})
            return arch_map.get(str(sm))
        except Exception:
            return None

    def _resolve_supported(self) -> set[torch.dtype]:
        vendor_map = _VENDOR_DTYPE_SUPPORT.get(self._vendor, {})
        if not vendor_map:
            return _DEFAULT_SUPPORTED

        if self._arch and self._arch in vendor_map:
            return vendor_map[self._arch]

        # Try parent arch (e.g. fallback "ampere" for unknown sm80 variants)
        if self._arch is None:
            # No arch detected — return the most conservative set
            return _DEFAULT_SUPPORTED | vendor_map.get("ampere", set())

        return _DEFAULT_SUPPORTED


# Module-level singleton
dtype_capability = DtypeCapability()
