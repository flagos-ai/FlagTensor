import os

import pytest
import torch

from flagtensor import sigmoid
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES, DEFAULT_SIGMOID_BENCHMARK_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorSigmoid
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_SIGMOID"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


class SigmoidBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(DEFAULT_BENCHMARK_DTYPES),
                shapes=tuple(DEFAULT_SIGMOID_BENCHMARK_SHAPES),
            ),
        )
        self.baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        for shape in self.config.shapes:
            yield (torch.randn(shape, device=self.device, dtype=dtype),)

    def baseline_impl(self, x):
        baseline = self.baselines.get(x.dtype)
        if baseline is None:
            baseline = CuTensorSigmoid(dtype=x.dtype)
            self.baselines[x.dtype] = baseline
        baseline.prepare(x)
        return baseline(x)

    def triton_impl(self, x):
        return sigmoid(x)

    def reference_impl(self, x):
        return torch.sigmoid(x)


@pytest.mark.performance
def test_sigmoid_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not CUTENSOR_AVAILABLE:
        pytest.skip("cuTensor unavailable")

    bench = SigmoidBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
