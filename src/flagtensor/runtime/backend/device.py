"""Device detection — auto-discovers GPU vendor through multiple fallback layers.

Following FlagGems runtime/backend/device.py pattern.
"""

import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch  # noqa: F401

from . import backend_utils
from . import (
    get_vendor_info,
    get_vendor_infos,
    gen_torch_device_object,
    get_vendor_module,
)
from ..common import (
    _VENDOR_TORCH_ATTR,
    UNSUPPORT_BF16,
    UNSUPPORT_FP64,
    UNSUPPORT_INT64,
    vendors,
)


class DeviceDetector:
    """Singleton detector that auto-discovers the active GPU vendor.

    Detection priority (first match wins):
    1. Environment variables: GEMS_VENDOR, FLAGGEMS_VENDOR, GEMS_BACKEND, FLAGGEMS_BACKEND, PPU_SDK
    2. Quick PyTorch attribute check: torch.npu, torch.mlu, torch.musa, etc.
       → Falls back to torch.cuda.get_device_properties() for NVIDIA
    3. System query: runs each vendor's device_query_cmd in parallel
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, vendor_name=None):
        if hasattr(self, "initialized"):
            return
        self.initialized = True

        self.vendor_list = vendors.get_all_vendors()
        self.info = self.get_vendor(vendor_name)
        self.vendor_name = self.info.vendor_name
        self.name = self.info.device_name
        self.dispatch_key = (
            self.name.upper()
            if self.info.dispatch_key is None
            else self.info.dispatch_key
        )

        self.vendor = vendors.get_all_vendors().get(self.vendor_name)
        self.device_count = self._safe_device_count()

        # dtype capability flags
        self.support_fp64 = self.vendor not in UNSUPPORT_FP64
        self.support_bf16 = self.vendor not in UNSUPPORT_BF16
        self.support_int64 = self.vendor not in UNSUPPORT_INT64

    def _safe_device_count(self) -> int:
        try:
            return gen_torch_device_object(self.vendor_name).device_count()
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # Layer 1: environment variables
    # ------------------------------------------------------------------
    def _get_vendor_from_env(self):
        # PPU_SDK → thead (T-Head XuanTie)
        if "PPU_SDK" in os.environ:
            return "thead"

        env_keys = (
            "GEMS_VENDOR",
            "FLAGGEMS_VENDOR",
            "GEMS_BACKEND",
            "FLAGGEMS_BACKEND",
        )
        for key in env_keys:
            if key in os.environ:
                return str(os.environ.get(key).lower())

        return False

    # ------------------------------------------------------------------
    # Layer 2: PyTorch quick check
    # ------------------------------------------------------------------
    def _get_vendor_from_quick_cmd(self):
        # Check vendor-specific torch attributes first
        try:
            import torch_npu

            torch_module = torch_npu
        except ImportError:
            torch_module = torch

        for vendor_name, attr in _VENDOR_TORCH_ATTR.items():
            if hasattr(torch_module, attr):
                return str(vendor_name)

        # Fallback: check torch.cuda for NVIDIA
        if hasattr(torch_module, "cuda") and hasattr(
            torch_module.cuda, "get_device_properties"
        ):
            try:
                prop = torch_module.cuda.get_device_properties(0)
                if "NVIDIA" in prop.name.upper():
                    return "nvidia"
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------
    # Layer 3: system device query (parallel)
    # ------------------------------------------------------------------
    def _get_vendor_from_sys(self):
        vendor_infos = get_vendor_infos()

        def check_vendor(info):
            try:
                cmd_args = shlex.split(info.device_query_cmd)
                result = subprocess.run(
                    cmd_args, capture_output=True, text=True, timeout=10
                )
                return info if result.returncode == 0 else None
            except Exception:
                return None

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(check_vendor, info): info for info in vendor_infos
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    return result

        return False

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------
    def get_vendor(self, vendor_name=None):
        """Determine the active vendor.

        If vendor_name is explicitly given, use it directly.
        Otherwise run the detection chain: env → quick_cmd → sys.
        """
        if vendor_name is not None:
            return get_vendor_info(vendor_name)

        # Layer 1
        vendor_from_env = self._get_vendor_from_env()
        if vendor_from_env:
            return get_vendor_info(vendor_from_env)

        # Layer 2
        vendor_name = self._get_vendor_from_quick_cmd()
        if vendor_name:
            return get_vendor_info(vendor_name)

        # Layer 3
        info = self._get_vendor_from_sys()
        if info:
            return info

        raise RuntimeError(
            "No supported device found for FlagTensor runtime. "
            "Set GEMS_VENDOR=nvidia to force NVIDIA backend."
        )
