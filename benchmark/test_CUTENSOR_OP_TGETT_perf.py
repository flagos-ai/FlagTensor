import os

import pytest
import torch

from flagtensor import tgett
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES, DEFAULT_TGETT_BENCHMARK_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorContraction
from flagtensor.ops.CUTENSOR_OP_GETT import _launch_gett_like_kernel
from flagtensor.ops.CUTENSOR_OP_TGETT import _is_default_2d_tgett_case
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_TGETT"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


def _tgett_case(shape_a, shape_b):
    if len(shape_a) == 2 and len(shape_b) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[1], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b)
    if len(shape_a) == 3 and len(shape_b) == 2:
        return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[2], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b)
    raise ValueError("unsupported TGETT benchmark shape combination")


def _tgett_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


class TgettBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(dtype for dtype in DEFAULT_BENCHMARK_DTYPES if dtype in (torch.float16, torch.float32)),
                shapes=tuple(DEFAULT_TGETT_BENCHMARK_SHAPES),
                mode="kernel",
            ),
        )
        self.baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        for shape_a, shape_b in self.config.shapes:
            _, _, _, c_shape, _ = _tgett_case(shape_a, shape_b)
            yield (
                torch.empty(shape_a, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(shape_b, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(c_shape, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
            )

    def baseline_impl(self, a, b, c):
        baseline = self.baselines.get(a.dtype)
        if baseline is None:
            baseline = CuTensorContraction(dtype=a.dtype)
            self.baselines[a.dtype] = baseline
        mode_a, mode_b, mode_d, _, _ = _tgett_case(tuple(a.shape), tuple(b.shape))
        a_t = a.transpose(-1, -2)
        return baseline(a_t, b, c=c, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_d, mode_d=mode_d)

    def triton_impl(self, a, b, c):
        mode_a, mode_b, mode_d, _, _ = _tgett_case(tuple(a.shape), tuple(b.shape))
        return tgett(a, b, c=c, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_d, mode_d=mode_d)

    def reference_impl(self, a, b, c):
        _, _, _, _, reference = _tgett_case(tuple(a.shape), tuple(b.shape))
        return _tgett_reference(a, b, c, reference)

    def build_triton_kernel_callable(self, *args):
        a, b, c = args
        mode_a, mode_b, mode_d, _, _ = _tgett_case(tuple(a.shape), tuple(b.shape))
        if not _is_default_2d_tgett_case(a, b, c, mode_a, mode_b, mode_d, mode_d):
            return None
        out = torch.empty((a.shape[1], b.shape[1]), device=a.device, dtype=a.dtype)

        def run_kernel():
            return _launch_gett_like_kernel(a, b, c, out, 1.25, 0.5, trans_a=True, trans_b=False)

        return run_kernel

    def build_baseline_kernel_callable(self, *args):
        a, b, c = args
        mode_a, mode_b, mode_d, _, _ = _tgett_case(tuple(a.shape), tuple(b.shape))
        if not _is_default_2d_tgett_case(a, b, c, mode_a, mode_b, mode_d, mode_d):
            return None
        baseline = self.baselines.get(a.dtype)
        if baseline is None:
            baseline = CuTensorContraction(dtype=a.dtype)
            self.baselines[a.dtype] = baseline
        a_t = a.transpose(-1, -2)

        def run_kernel():
            return baseline(a_t, b, c=c, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_d, mode_d=mode_d)

        return run_kernel


@pytest.mark.performance
@pytest.mark.tgett
def test_tgett_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not CUTENSOR_AVAILABLE:
        pytest.skip("cuTensor unavailable")

    bench = TgettBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
