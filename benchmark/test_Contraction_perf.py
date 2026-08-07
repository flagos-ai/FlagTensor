import os

import pytest
import torch

from flagtensor import contraction
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES, DEFAULT_GETT_BENCHMARK_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
try:
    from flagtensor.cutensor import CuTensorContraction as _BaselineClass
except ImportError:
    _BaselineClass = None
try:
    from flagtensor.torch_npu_baseline import torch_npu_available as _TORCH_NPU_AVAILABLE
except ImportError:
    _TORCH_NPU_AVAILABLE = lambda: False
BASELINE_AVAILABLE = CUTENSOR_AVAILABLE or _BaselineClass is not None or _TORCH_NPU_AVAILABLE()
from flagtensor.ops.CUTENSOR_OP_GETT import _is_default_2d_gett_case, _launch_gett_kernel
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_GETT"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


def _Contraction_case(shape_a, shape_b):
    if len(shape_a) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[0], shape_b[1]), lambda a, b: torch.matmul(a, b)
    return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[1], shape_b[1]), lambda a, b: torch.einsum("abc,cd->abd", a, b)


def _Contraction_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


class GettBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=(torch.float32,),
                shapes=tuple(DEFAULT_GETT_BENCHMARK_SHAPES),
                mode="kernel",
            ),
        )
        self.baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        for shape_a, shape_b in self.config.shapes:
            _, _, _, c_shape, _ = _Contraction_case(shape_a, shape_b)
            yield (
                torch.empty(shape_a, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(shape_b, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(c_shape, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
            )

    def baseline_impl(self, a, b, c):
        baseline = self.baselines.get(a.dtype)
        if baseline is None:
            baseline = self._get_baseline_instance(a.dtype)
            self.baselines[a.dtype] = baseline
        mode_a, mode_b, mode_d, _, _ = _Contraction_case(tuple(a.shape), tuple(b.shape))
        return baseline(a, b, c=c, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_d, mode_d=mode_d)

    def triton_impl(self, a, b, c):
        mode_a, mode_b, mode_d, _, _ = _Contraction_case(tuple(a.shape), tuple(b.shape))
        return contraction(a, b, c=c, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_d, mode_d=mode_d)

    def reference_impl(self, a, b, c):
        _, _, _, _, reference = _Contraction_case(tuple(a.shape), tuple(b.shape))
        return _Contraction_reference(a, b, c, reference)

    def build_triton_kernel_callable(self, *args):
        a, b, c = args
        mode_a, mode_b, mode_d, _, _ = _Contraction_case(tuple(a.shape), tuple(b.shape))
        if not _is_default_2d_gett_case(a, b, c, mode_a, mode_b, mode_d, mode_d):
            return None
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

        def run_kernel():
            return _launch_gett_kernel(a, b, c, out, 1.25, 0.5)

        return run_kernel

    def build_baseline_kernel_callable(self, *args):
        a, b, c = args
        mode_a, mode_b, mode_d, _, _ = _Contraction_case(tuple(a.shape), tuple(b.shape))
        if not _is_default_2d_gett_case(a, b, c, mode_a, mode_b, mode_d, mode_d):
            return None
        baseline = self.baselines.get(a.dtype)
        if baseline is None:
            baseline = self._get_baseline_instance(a.dtype)
            self.baselines[a.dtype] = baseline

        def run_kernel():
            return baseline(a, b, c=c, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_d, mode_d=mode_d)

        return run_kernel


@pytest.mark.performance
@pytest.mark.Contraction
def test_Contraction_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("Vendor baseline unavailable")

    bench = GettBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
