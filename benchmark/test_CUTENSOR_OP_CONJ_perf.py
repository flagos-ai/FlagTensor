import os

import pytest
import torch

from flagtensor import conj
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig
from flagtensor.config import DEFAULT_CONJ_BENCHMARK_DTYPES, DEFAULT_CONJ_BENCHMARK_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorConj
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_CONJ"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


def x_real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


class ConjBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(DEFAULT_CONJ_BENCHMARK_DTYPES),
                shapes=tuple(DEFAULT_CONJ_BENCHMARK_SHAPES),
            ),
        )
        self.baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        real_dtype = x_real_dtype(dtype)
        for shape in self.config.shapes:
            real = torch.randn(shape, device=self.device, dtype=real_dtype)
            imag = torch.randn(shape, device=self.device, dtype=real_dtype)
            yield ((real + 1j * imag).to(dtype),)

    def baseline_impl(self, x):
        baseline = self.baselines.get(x.dtype)
        if baseline is None:
            baseline = CuTensorConj(dtype=x.dtype)
            self.baselines[x.dtype] = baseline
        baseline.prepare(x)
        return baseline(x)

    def triton_impl(self, x):
        return conj(x)

    def reference_impl(self, x):
        return torch.conj(x)


@pytest.mark.performance
def test_conj_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not CUTENSOR_AVAILABLE:
        pytest.skip("cuTensor unavailable")

    bench = ConjBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
