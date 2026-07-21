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

from flagtensor import exp
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig, get_baseline_class, vendor_baseline_available
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES, DEFAULT_EXP_BENCHMARK_SHAPES
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_EXP"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


class ExpBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(DEFAULT_BENCHMARK_DTYPES),
                shapes=tuple(DEFAULT_EXP_BENCHMARK_SHAPES),
            ),
        )
        self.baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        for shape in self.config.shapes:
            yield (torch.empty(shape, device=self.device, dtype=dtype).uniform_(-8.0, 8.0),)

    def baseline_impl(self, x):
        baseline = self.baselines.get(x.dtype)
        if baseline is None:
            baseline = get_baseline_class(OP_NAME)(dtype=x.dtype)
            self.baselines[x.dtype] = baseline
        baseline.prepare(x)
        return baseline(x)

    def triton_impl(self, x):
        return exp(x)

    def reference_impl(self, x):
        return torch.exp(x)


@pytest.mark.performance
def test_exp_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not vendor_baseline_available():
        pytest.skip("baseline unavailable")

    bench = ExpBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
