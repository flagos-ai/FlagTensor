"""Regression coverage for PPU SDK detection and baseline setup."""
import os
import subprocess
import sys


def test_ppu_sdk_selects_ppu_baseline():
    env = os.environ.copy()
    for key in ("GEMS_VENDOR", "FLAGGEMS_VENDOR", "GEMS_BACKEND", "FLAGGEMS_BACKEND"):
        env.pop(key, None)
    env["PPU_SDK"] = "/tmp/ppu-sdk"
    code = """
from flagtensor.runtime import device
from flagtensor.benchmark_core import Benchmark, vendor_baseline_available, get_vendor_baseline_class
assert device.vendor_name == 'ppu', device.vendor_name
assert vendor_baseline_available()
assert get_vendor_baseline_class('swish') is not None
bench = Benchmark(op_name='CUTENSOR_OP_SWISH')
assert 'ppu' in bench._baseline_module().__name__
assert bench._get_baseline_instance(__import__('torch').float32) is not None
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
