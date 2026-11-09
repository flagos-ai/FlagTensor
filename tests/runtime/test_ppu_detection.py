"""PPU detection and benchmark baseline regressions without PPU hardware."""
import os
import subprocess
import sys


def _check(code, **env_changes):
    env = os.environ.copy()
    for name in ("GEMS_VENDOR", "FLAGGEMS_VENDOR", "GEMS_BACKEND", "FLAGGEMS_BACKEND", "PPU_SDK"):
        env.pop(name, None)
    env.update(env_changes)
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_ppu_sdk_uses_ppu_baseline():
    _check(
        """
import torch
from flagtensor.runtime import device
import flagtensor.benchmark_core as core
assert device.vendor_name == 'ppu'
assert core.vendor_baseline_available()
assert core.get_vendor_baseline_class('ceil') is not None
# A loadable cuTensor must not change the PPU baseline selection.
core.CUTENSOR_AVAILABLE = True
bench = core.Benchmark(op_name='CUTENSOR_OP_CEIL')
assert '_ppu' in bench._baseline_module().__name__
assert bench._get_baseline_instance(torch.float32) is not None
""",
        PPU_SDK="/tmp/ppu-sdk",
    )


def test_explicit_vendor_overrides_ppu_sdk():
    _check(
        "from flagtensor.runtime import device; assert device.vendor_name == 'nvidia'",
        PPU_SDK="/tmp/ppu-sdk", GEMS_VENDOR="nvidia",
    )
