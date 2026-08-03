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
    "get_tuned_config",
    "get_heuristic_config",
    "replace_customized_ops",
]
