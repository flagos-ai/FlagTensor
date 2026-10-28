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

"""Ascend-specific persistent-grid helpers.

On Ascend, a Triton launch is a *block* launch handled by the device-side
scheduler.  Launching one program per tile (the conventional CUDA mapping)
means the runtime has to schedule hundreds/thousands of blocks.  For pure
vector pointwise kernels the recommended mapping is the opposite: launch one
persistent program per vector core and let each program walk a strided range
of tiles.  This removes almost all block-scheduling overhead and is what the
official triton-ascend programming guide recommends for elementwise ops.

Everything in this module is a no-op unless the active vendor is Ascend, so
importing it from vendor-neutral code does not change other backends.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from flagtensor import runtime

# Fallback used only when the runtime device properties cannot be queried.
# 40 is the 910B/910C vector-core count reported by current triton-ascend
# builds; the value is refreshed from the driver on first successful query.
_FALLBACK_VECTOR_CORE_NUM = 40

_vector_core_num: Optional[int] = None


def is_ascend() -> bool:
    """Return True when FlagTensor is running on the Ascend backend."""
    try:
        return getattr(runtime.device, "vendor_name", None) == "ascend"
    except Exception:
        return False


def vector_core_num() -> int:
    """Return the number of Ascend vector cores (AIV) for the active device.

    The result is cached after the first successful query.  If the query
    fails (e.g. during import in a CPU-only environment) we return a safe
    fallback so callers never have to special-case it.
    """
    global _vector_core_num
    if _vector_core_num is not None:
        return _vector_core_num
    if not is_ascend():
        _vector_core_num = 1
        return _vector_core_num
    try:
        import torch
        import triton.runtime.driver as driver

        try:
            device_index = torch.npu.current_device()
        except Exception:
            device_index = "npu"
        properties = driver.active.utils.get_device_properties(device_index)
        value = int(properties.get("num_vectorcore") or 0)
        if value <= 0:
            raise ValueError("invalid num_vectorcore")
        _vector_core_num = value
    except Exception:
        _vector_core_num = _FALLBACK_VECTOR_CORE_NUM
    return _vector_core_num


def persistent_grid(n_elements: int, tile_size: int) -> int:
    """Grid size for a persistent elementwise kernel.

    We launch one program per vector core, capped by the number of tiles so
    tiny tensors do not pay for idle programs.  At least one program is
    always launched, which keeps zero-element tensors valid.
    """
    if tile_size <= 0:
        return 1
    if n_elements <= 0:
        return 1
    num_tiles = (int(n_elements) + int(tile_size) - 1) // int(tile_size)
    return max(1, min(num_tiles, vector_core_num()))


def persistent_grid_from_meta(n_elements: int, meta: Mapping[str, Any]) -> int:
    """``persistent_grid`` using the ``BLOCK_SIZE``/``BLOCKS_PER_PROGRAM`` meta."""
    tile_size = int(meta["BLOCK_SIZE"]) * int(meta["BLOCKS_PER_PROGRAM"])
    return persistent_grid(n_elements, tile_size)


__all__ = [
    "is_ascend",
    "vector_core_num",
    "persistent_grid",
    "persistent_grid_from_meta",
]
