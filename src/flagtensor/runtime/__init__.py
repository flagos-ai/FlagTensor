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

"""FlagTensor runtime — device detection, config loading, backend abstraction."""

import torch

from . import backend
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()

backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()
torch_backend_device = backend.get_torch_backend_device_fn()


# ---------------------------------------------------------------------------
# Device abstraction — vendor-neutral accelerator access
# ---------------------------------------------------------------------------
# Canonical accelerator device string for the current backend. Tests and
# benchmark code should use this instead of hard-coded "cuda".
#   NVIDIA -> "cuda"
#   Ascend -> "npu"
device_str = device.name if isinstance(device.name, str) else "cuda"


def is_accelerator_available() -> bool:
    """Return True when the active vendor's accelerator is usable."""
    try:
        return bool(torch_device_fn.is_available())
    except Exception:
        return False


def synchronize():
    """Synchronize the active accelerator device (no-op on CPU)."""
    if not is_accelerator_available():
        return
    try:
        torch_device_fn.synchronize()
    except Exception:
        pass


def empty_cache():
    """Release the accelerator's caching allocator hold (no-op on CPU)."""
    if not is_accelerator_available():
        return
    try:
        torch_device_fn.empty_cache()
    except Exception:
        pass


def is_on_accelerator(tensor: torch.Tensor) -> bool:
    """Return True if ``tensor`` lives on the active accelerator.

    On NVIDIA this is equivalent to ``tensor.is_cuda``; on Ascend it covers
    ``tensor.is_npu``. Falls back to comparing ``tensor.device.type``.
    """
    try:
        if device_str == "cuda" and getattr(tensor, "is_cuda", False):
            return True
        if device_str == "npu" and getattr(tensor, "is_npu", False):
            return True
    except Exception:
        pass
    try:
        return tensor.device.type == device_str
    except Exception:
        return False


def is_ascend() -> bool:
    """Return True when FlagTensor is running on the Huawei Ascend backend."""
    try:
        return device.vendor_name == "ascend"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Persistent-grid launch mapping
# ---------------------------------------------------------------------------
# On Ascend a Triton launch is a *block* launch handled by the device-side
# scheduler, so one program per tile (the conventional CUDA mapping) makes the
# runtime schedule hundreds/thousands of blocks for what is a pure vector
# pointwise kernel.  The mapping recommended there is the opposite: launch one
# persistent program per vector core and let each program walk a strided range
# of tiles.  Other vendors keep one-program-per-tile, which ``triton.cdiv``
# computes directly.
#
# 40 is the vector-core count (AIV) reported by current 910B/910C builds; it is
# only used when the driver query fails.
_FALLBACK_VECTOR_CORE_NUM = 40
_vector_core_num = None


def vector_core_num() -> int:
    """Return the number of vector cores on the active device.

    Cached after the first successful query.  Returns 1 off Ascend and a safe
    fallback when the driver query fails, so callers never special-case it.
    """
    global _vector_core_num
    if _vector_core_num is not None:
        return _vector_core_num
    if not is_ascend():
        _vector_core_num = 1
        return _vector_core_num
    try:
        import triton.runtime.driver as driver

        try:
            device_index = torch_device_fn.current_device()
        except Exception:
            device_index = device_str
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

    One program per vector core, capped by the number of tiles so tiny tensors
    do not pay for idle programs.  Always at least one program, which keeps
    zero-element tensors valid.
    """
    if tile_size <= 0 or n_elements <= 0:
        return 1
    num_tiles = (int(n_elements) + int(tile_size) - 1) // int(tile_size)
    return max(1, min(num_tiles, vector_core_num()))


def persistent_grid_from_meta(n_elements, meta) -> int:
    """``persistent_grid`` using the ``BLOCK_SIZE``/``BLOCKS_PER_PROGRAM`` meta."""
    tile_size = int(meta["BLOCK_SIZE"]) * int(meta["BLOCKS_PER_PROGRAM"])
    return persistent_grid(n_elements, tile_size)


def get_tuned_config(op_name):
    return config_loader.get_tuned_config(op_name)


def get_heuristic_config(op_name):
    return config_loader.get_heuristics_config(op_name)


def replace_customized_ops(_globals):
    event = backend.BackendArchEvent()
    arch_specialization_operators = event.get_arch_ops() if event.has_arch else None
    backend_customization_operators = backend.get_current_device_extend_op(
        device.vendor_name
    )
    if backend_customization_operators:
        for fn_name, fn in backend_customization_operators:
            _globals[fn_name] = fn
    if arch_specialization_operators:
        for fn_name, fn in arch_specialization_operators:
            _globals[fn_name] = fn


__all__ = [
    "backend",
    "config_loader",
    "device",
    "torch_device_fn",
    "torch_backend_device",
    "device_str",
    "is_accelerator_available",
    "synchronize",
    "empty_cache",
    "is_on_accelerator",
    "is_ascend",
    "vector_core_num",
    "persistent_grid",
    "persistent_grid_from_meta",
    "get_tuned_config",
    "get_heuristic_config",
    "replace_customized_ops",
]
