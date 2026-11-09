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

from flagtensor import conj
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig, vendor_baseline_available
from flagtensor.config import DEFAULT_CONJ_BENCHMARK_DTYPES, DEFAULT_CONJ_BENCHMARK_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
try:
    from flagtensor.cutensor import CuTensorConj as _BaselineClass
except ImportError:
    _BaselineClass = None
try:
    from flagtensor.torch_npu_baseline import torch_npu_available as _TORCH_NPU_AVAILABLE
except ImportError:
    _TORCH_NPU_AVAILABLE = lambda: False
BASELINE_AVAILABLE = vendor_baseline_available()
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_CONJ"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


def x_real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


def _supported_complex_dtypes(dtypes):
    """Keep only dtypes this backend really allocates.

    ``torch_xmlir`` rejects complex arithmetic (``real + 1j * imag`` raises
    ``NOT IMPLEMENTED``) and silently downcasts ``complex128`` to
    ``complex64``; probing the allocated dtype keeps the benchmark honest
    instead of recording the same measurement twice.
    """
    kept = []
    for dtype in dtypes:
        try:
            probe = torch.empty((2,), device=_device_str, dtype=dtype)
            if probe.dtype == dtype:
                kept.append(dtype)
        except Exception:
            pass
    return tuple(kept)


class ConjBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=_supported_complex_dtypes(DEFAULT_CONJ_BENCHMARK_DTYPES),
                shapes=tuple(DEFAULT_CONJ_BENCHMARK_SHAPES),
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
            self.baselines[x.dtype] = baseline
        baseline.prepare(x)
        return baseline(x)

    def triton_impl(self, x):
        return conj(x)

    def reference_impl(self, x):
        # ``torch.conj`` is not implemented by torch_xmlir, and a strided
        # in-place ``neg_`` on ``x.view(float)[..., 1]`` only updates the
        # first lane there, so build the conjugated pair explicitly.
        real_dtype = torch.float32 if x.dtype == torch.complex64 else torch.float64
        rv = x.view(real_dtype).view(*x.shape, 2)
        pair = torch.stack([rv[..., 0], -rv[..., 1]], dim=-1).contiguous()
        return torch.view_as_complex(pair)

    def verify(self, reference, test, dtype):
        # ``torch.allclose`` has no complex kernel on torch_xmlir; comparing
        # the interleaved real view is equivalent and uses the same
        # tolerance policy as every other operator.
        if reference.is_complex():
            real_dtype = torch.float32 if reference.dtype == torch.complex64 else torch.float64
            return super().verify(reference.view(real_dtype), test.view(real_dtype), real_dtype)
        return super().verify(reference, test, dtype)


@pytest.mark.performance
def test_conj_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("Vendor baseline unavailable")

    bench = ConjBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
