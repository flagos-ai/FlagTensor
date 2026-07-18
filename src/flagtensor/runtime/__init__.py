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

from . import backend
from .backend.device import DeviceDetector
from .configloader import ConfigLoader

config_loader = ConfigLoader()
device = DeviceDetector()

backend.set_torch_backend_device_fn(device.vendor_name)
torch_device_fn = backend.gen_torch_device_object()
torch_backend_device = backend.get_torch_backend_device_fn()


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
    "get_tuned_config",
    "get_heuristic_config",
    "replace_customized_ops",
]
