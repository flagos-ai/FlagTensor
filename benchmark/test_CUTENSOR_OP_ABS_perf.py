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

import os

import pytest
import torch
import triton

from flagtensor import abs
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig, vendor_baseline_available
from flagtensor.config import DEFAULT_ABS_BENCHMARK_SHAPES, DEFAULT_BENCHMARK_DTYPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.ops.CUTENSOR_OP_ABS import _abs_kernel
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
try:
    from flagtensor.cutensor import CuTensorAbs as _BaselineClass
except ImportError:
    _BaselineClass = None
try:
    from flagtensor.torch_npu_baseline import torch_npu_available as _TORCH_NPU_AVAILABLE
except ImportError:
    _TORCH_NPU_AVAILABLE = lambda: False
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_ABS"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")

# A vendor-optimized baseline is available on NVIDIA (cuTensor), Ascend
BASELINE_AVAILABLE = (
    CUTENSOR_AVAILABLE
    or _TORCH_NPU_AVAILABLE()
    or _BaselineClass is not None
    or vendor_baseline_available()
)


class AbsBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(DEFAULT_BENCHMARK_DTYPES),
                shapes=tuple(DEFAULT_ABS_BENCHMARK_SHAPES),
            ),
        )
        self.baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        for shape in self.config.shapes:
            yield (torch.randn(shape, device=self.device, dtype=dtype),)

    def baseline_impl(self, x):
        baseline = self.baselines.get(x.dtype)
        if baseline is None:
            baseline = self._get_baseline_instance(x.dtype)
            if baseline is None:
                raise RuntimeError(
                    "No vendor baseline available for ABS on this device"
                )
            self.baselines[x.dtype] = baseline
        baseline.prepare(x)
        return baseline(x)

    def triton_impl(self, x):
        return abs(x)

    def reference_impl(self, x):
        return torch.abs(x)

    def build_triton_kernel_callable(self, x):
        y = torch.empty_like(x)
        n_elements = y.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )

        def run_kernel():
            _abs_kernel[grid](x, y, n_elements)
            return y

        return run_kernel

    def build_baseline_kernel_callable(self, x):
        baseline = self.baselines.get(x.dtype)
        if baseline is None:
            baseline = self._get_baseline_instance(x.dtype)
            if baseline is None:
                return None
            self.baselines[x.dtype] = baseline
        return baseline.build_kernel_callable(x)


@pytest.mark.performance
def test_abs_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("No vendor baseline available (cuTensor/torch_npu)")

    bench = AbsBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
