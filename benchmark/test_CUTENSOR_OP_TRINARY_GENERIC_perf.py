import os

import pytest
import torch

from flagtensor import trinary
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, _get_trinary_executor, trinary as cutensor_trinary
from flagtensor.ops.CUTENSOR_OP_TRINARY_GENERIC import _get_triton_trinary_executor
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_TRINARY_GENERIC"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")
SMALL_SHAPES = ((1024,), (4096,))
LARGE_SHAPES = ((16384,), (65536,), (262144,), (1048576,))
INDEXED_SHAPES = ((2, 3, 32, 64, 16),)

_TRINARY_OP_KWARGS = {
    "op_a": "log",
    "op_b": "neg",
    "op_c": "sqrt",
    "op_ab": "add",
    "op_abc": "max",
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
}

_INDEXED_MODE_KWARGS = {
    "mode_a": (0, 1, 2, 3, 4),
    "mode_b": (1, 2, 4, 0, 3),
    "mode_c": (2, 3, 4, 0, 1),
    "mode_d": (2, 3, 4, 0, 1),
}


def trinary_reference(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ref_a = torch.log(a.float())
    ref_b = -b.float()
    ref_c = torch.sqrt(c.float())
    out = torch.maximum(ref_a + ref_b, ref_c)
    return out.to(a.dtype) if a.dtype in (torch.float16, torch.bfloat16) else out.to(a.dtype)


class TrinaryGenericBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(dtype for dtype in DEFAULT_BENCHMARK_DTYPES if dtype in (torch.float16, torch.float32)),
                shapes=SMALL_SHAPES,
                mode="operator",
            ),
        )

    def get_input_iter(self, dtype: torch.dtype):
        for shape in self.config.shapes:
            yield (
                torch.empty(shape, device=self.device, dtype=dtype).uniform_(0.5, 4.0),
                torch.empty(shape, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(shape, device=self.device, dtype=dtype).uniform_(0.5, 4.0),
            )

    def baseline_impl(self, a, b, c):
        return cutensor_trinary(a, b, c, **_TRINARY_OP_KWARGS)

    def triton_impl(self, a, b, c):
        return trinary(a, b, c, **_TRINARY_OP_KWARGS)

    def reference_impl(self, a, b, c):
        return trinary_reference(a, b, c)

    def build_triton_kernel_callable(self, *args):
        a, b, c = args
        out = torch.empty_like(c)
        executor = _get_triton_trinary_executor(**_TRINARY_OP_KWARGS)

        def _run():
            return executor(a, b, c, out=out)

        return _run

    def build_baseline_kernel_callable(self, *args):
        a, b, c = args
        out = torch.empty_like(c)
        executor = _get_trinary_executor(
            _TRINARY_OP_KWARGS["op_ab"],
            _TRINARY_OP_KWARGS["op_abc"],
            _TRINARY_OP_KWARGS["op_a"],
            _TRINARY_OP_KWARGS["op_b"],
            _TRINARY_OP_KWARGS["op_c"],
            a.dtype,
        )

        def _run():
            return executor(
                a,
                b,
                c,
                alpha=_TRINARY_OP_KWARGS["alpha"],
                beta=_TRINARY_OP_KWARGS["beta"],
                gamma=_TRINARY_OP_KWARGS["gamma"],
                out=out,
            )

        return _run


class TrinaryGenericKernelBenchmark(TrinaryGenericBenchmark):
    def __init__(self):
        super().__init__()
        self.config = BenchmarkConfig(
            warmup=self.config.warmup,
            repetitions=self.config.repetitions,
            dtypes=self.config.dtypes,
            shapes=LARGE_SHAPES,
            metrics=self.config.metrics,
            mode="kernel",
        )


class TrinaryGenericHighRankIndexedBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(dtype for dtype in DEFAULT_BENCHMARK_DTYPES if dtype in (torch.float16, torch.float32)),
                shapes=INDEXED_SHAPES,
                mode="kernel",
            ),
        )

    def get_input_iter(self, dtype: torch.dtype):
        for _ in self.config.shapes:
            yield (
                torch.empty((2, 3, 32, 64, 16), device=self.device, dtype=dtype).uniform_(0.5, 4.0),
                torch.empty((3, 32, 16, 2, 64), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty((32, 64, 16, 2, 3), device=self.device, dtype=dtype).uniform_(0.5, 4.0),
            )

    def baseline_impl(self, a, b, c):
        return cutensor_trinary(a, b, c, **_TRINARY_OP_KWARGS, **_INDEXED_MODE_KWARGS)

    def triton_impl(self, a, b, c):
        return trinary(a, b, c, **_TRINARY_OP_KWARGS, **_INDEXED_MODE_KWARGS)

    def reference_impl(self, a, b, c):
        ref_a = torch.log(a.float()).permute(2, 3, 4, 0, 1)
        ref_b = -b.float().permute(1, 4, 2, 3, 0)
        ref_c = torch.sqrt(c.float())
        out = torch.maximum(ref_a + ref_b, ref_c)
        return out.to(a.dtype)

    def build_triton_kernel_callable(self, *args):
        a, b, c = args
        out = torch.empty_like(c)
        executor = _get_triton_trinary_executor(**_TRINARY_OP_KWARGS)

        def _run():
            return executor(a, b, c, out=out, **_INDEXED_MODE_KWARGS)

        return _run

    def build_baseline_kernel_callable(self, *args):
        a, b, c = args
        out = torch.empty_like(c)
        executor = _get_trinary_executor(
            _TRINARY_OP_KWARGS["op_ab"],
            _TRINARY_OP_KWARGS["op_abc"],
            _TRINARY_OP_KWARGS["op_a"],
            _TRINARY_OP_KWARGS["op_b"],
            _TRINARY_OP_KWARGS["op_c"],
            a.dtype,
        )

        def _run():
            return executor(
                a,
                b,
                c,
                alpha=_TRINARY_OP_KWARGS["alpha"],
                beta=_TRINARY_OP_KWARGS["beta"],
                gamma=_TRINARY_OP_KWARGS["gamma"],
                out=out,
                **_INDEXED_MODE_KWARGS,
            )

        return _run


@pytest.mark.performance
@pytest.mark.trinary_generic
def test_trinary_generic_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not CUTENSOR_AVAILABLE:
        pytest.skip("cuTensor unavailable")

    bench = TrinaryGenericBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} mode={result.mode} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )


@pytest.mark.performance
@pytest.mark.trinary_generic
def test_trinary_generic_kernel_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not CUTENSOR_AVAILABLE:
        pytest.skip("cuTensor unavailable")

    bench = TrinaryGenericKernelBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, f"{OP_NAME}_KERNEL")
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} mode={result.mode} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )


@pytest.mark.performance
@pytest.mark.trinary_generic
def test_trinary_generic_high_rank_indexed_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not CUTENSOR_AVAILABLE:
        pytest.skip("cuTensor unavailable")

    bench = TrinaryGenericHighRankIndexedBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, f"{OP_NAME}_HIGH_RANK_INDEXED")
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} cutensor_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
