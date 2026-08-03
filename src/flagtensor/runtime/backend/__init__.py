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

"""Backend abstraction — vendor modules, device detection, arch specialization.

Following FlagGems runtime/backend/__init__.py pattern.
"""
import importlib
import inspect
import os
import sys
from pathlib import Path

from . import backend_utils
from ..common import vendors

# ---------------------------------------------------------------------------
# BackendState — singleton holding cached vendor/device state
# ---------------------------------------------------------------------------


class BackendState:
    """Singleton managing backend state variables."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.vendor_module = None
        self.device_name = None
        self.torch_device_object = None
        self.torch_device_fn_device = None
        self.tl_extra_backend_module = None
        self.ops_module = None
        self.fused_module = None
        self.heuristic_config_module = None
        self.vendor_extra_lib_imported = False
        self.customized_ops = None


_state = BackendState()

# ---------------------------------------------------------------------------
# BackendArchEvent — GPU architecture specialization (ampere/hopper/blackwell)
# ---------------------------------------------------------------------------


class BackendArchEvent:
    has_arch = False
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend=None):
        if BackendArchEvent._initialized:
            return
        BackendArchEvent._initialized = True
        self.backend = backend
        self.error_msgs = []
        self.arch = self.get_arch()
        if self.has_arch:
            self.supported_archs = self._get_supported_archs()
            self.current_arch_path = self.supported_archs.get(self.arch)
            if self.current_arch_path is not None:
                self.arch_module = self.get_arch_module()
                self.autotune_configs = self.get_autotune_configs()
                self.heuristics_configs = self.get_heuristics_configs()
            else:
                self.has_arch = False

    def get_functions_from_module(self, module):
        return inspect.getmembers(module, inspect.isfunction) if module else []

    def get_heuristics_configs(self):
        heuristic_module = None
        try:
            heuristic_module = self.arch_module
        except Exception:
            sys.path.insert(0, str(self.current_arch_path))
            heuristic_module = importlib.import_module("heuristics_config_utils")
            sys.path.remove(str(self.current_arch_path))
        if hasattr(heuristic_module, "HEURISTICS_CONFIGS"):
            return heuristic_module.HEURISTICS_CONFIGS
        return None

    def get_autotune_configs(self):
        return backend_utils.get_tune_config(file_path=self.current_arch_path)

    def get_arch(self, device=0):
        if not hasattr(_state.vendor_module, "ARCH_MAP"):
            return
        arch_map = _state.vendor_module.ARCH_MAP
        arch_string = os.environ.get("ARCH", "")
        arch_string_num = (
            arch_string.split("_")[-1][0] if arch_string else arch_string
        )
        if not arch_string_num:
            try:
                if not _state.torch_device_object.is_available():
                    return False
                props = _state.torch_device_object.get_device_properties(device)
                arch_string_num = str(props.major)
            except Exception:
                self.has_arch = False
        if arch_string_num in arch_map:
            self.has_arch = True
            return arch_map[arch_string_num]
        return None

    def _get_supported_archs(self, path=None):
        path = path or _state.vendor_module.__path__[0]
        excluded = ("ops", "fused")
        path = Path(path)
        path = path.parent if path.is_file() else path
        return {
            p.name: str(p)
            for p in path.iterdir()
            if p.is_dir() and p.name not in excluded and not p.name.startswith("_")
        }

    def get_arch_module(self):
        path_dir = os.path.dirname(self.current_arch_path)
        sys.path.insert(0, str(path_dir))
        current_arch_module = importlib.import_module(self.arch)
        sys.path.remove(str(path_dir))
        return current_arch_module

    def get_arch_ops(self):
        arch_specialized_ops = []
        modules = []
        try:
            modules.append(self.arch_module.ops)
        except Exception:
            try:
                sys.path.append(self.current_arch_path)
                ops_module = importlib.import_module(f"{self.arch}.ops")
                modules.append(ops_module)
            except Exception as err_msg:
                self.error_msgs.append(err_msg)
        for mod in modules:
            arch_specialized_ops.extend(self.get_functions_from_module(mod))
        return arch_specialized_ops


# ---------------------------------------------------------------------------
# Vendor module helpers
# ---------------------------------------------------------------------------


def get_vendor_module(vendor_name, query=False):
    def load(name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vendor_dir = os.path.join(current_dir, name)
        if not os.path.isdir(vendor_dir):
            raise ModuleNotFoundError(f"No vendor module: {name}")
        sys.path.append(current_dir)
        try:
            return importlib.import_module(name)
        finally:
            sys.path.remove(current_dir)

    if query:
        return load(vendor_name)

    if _state.vendor_module is None:
        _state.vendor_module = load("_" + vendor_name)
    return _state.vendor_module


def get_vendor_info(vendor_name=None, query=False):
    if query:
        try:
            return get_vendor_module(vendor_name, query).vendor_info
        except ModuleNotFoundError:
            return None
    try:
        get_vendor_module(vendor_name)
    except ModuleNotFoundError:
        # Fallback to nvidia for unknown vendor
        get_vendor_module("nvidia")
    return _state.vendor_module.vendor_info


def get_vendor_infos():
    """Return all available vendor_info objects by scanning backend/ directory."""
    infos = []
    for vendor_name in vendors.get_all_vendors():
        try:
            infos.append(get_vendor_info(f"_{vendor_name}", query=True))
        except Exception:
            continue
    return infos


def import_vendor_extra_lib(vendor_name=None):
    if _state.vendor_extra_lib_imported:
        return
    try:
        _state.ops_module = importlib.import_module(f"_{vendor_name}.ops")
    except ModuleNotFoundError:
        _state.ops_module = None
    _state.vendor_extra_lib_imported = True


def set_torch_backend_device_fn(vendor_name=None):
    _state.device_name = _state.device_name or get_vendor_info(vendor_name).device_name
    module_str = f"torch.backends.{_state.device_name}"
    try:
        _state.torch_device_fn_device = importlib.import_module(module_str)
    except ImportError:
        # Some backends (e.g. Ascend/torch_npu) do not expose a
        # torch.backends.<device> module. Fall back to the vendor's top-level
        # torch extension module so callers can still query device caps.
        try:
            _state.torch_device_fn_device = importlib.import_module(
                f"torch.{_state.device_name}"
            )
        except ImportError:
            _state.torch_device_fn_device = None


def get_torch_backend_device_fn():
    return _state.torch_device_fn_device


def gen_torch_device_object(vendor_name=None):
    if _state.torch_device_object is not None:
        return _state.torch_device_object
    _state.device_name = _state.device_name or get_vendor_info(vendor_name).device_name
    namespace = {}
    exec(f"import torch\nfn = torch.{_state.device_name}", namespace)
    _state.torch_device_object = namespace["fn"]
    return _state.torch_device_object


def get_current_device_extend_op(vendor_name=None):
    import_vendor_extra_lib(vendor_name)
    if _state.customized_ops is not None:
        return _state.customized_ops
    _state.customized_ops = []
    if _state.ops_module is not None:
        _state.customized_ops.extend(
            inspect.getmembers(_state.ops_module, inspect.isfunction)
        )
    return _state.customized_ops


def get_heuristic_config(vendor_name=None):
    if _state.heuristic_config_module is None:
        try:
            _state.heuristic_config_module = importlib.import_module(
                f"_{vendor_name}.heuristics_config_utils"
            )
        except Exception:
            _state.heuristic_config_module = importlib.import_module(
                "_nvidia.heuristics_config_utils"
            )
    if hasattr(_state.heuristic_config_module, "HEURISTICS_CONFIGS"):
        return _state.heuristic_config_module.HEURISTICS_CONFIGS
    return None


def get_tune_config(vendor_name=None):
    get_vendor_module(vendor_name)
    return backend_utils.get_tune_config(vendor_name)


def get_backend_state():
    return _state


__all__ = ["*"]
