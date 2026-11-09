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

from flagtensor import elementwise_trinary
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig, vendor_baseline_available
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
try:
    from flagtensor.torch_npu_baseline import torch_npu_available as _TORCH_NPU_AVAILABLE
except ImportError:
    _TORCH_NPU_AVAILABLE = lambda: False

    
# A trinary baseline is available on NVIDIA (cuTensor), Ascend
# (torch_npu-aten / CANN aclnn), vendors exposing a trinary executor
# through BASELINE_MODULE_NAME (e.g. MetaX), or vendor backends that
# declare BASELINE_AVAILABLE (e.g. Iluvatar CoreX).
def _vendor_baseline_has_trinary() -> bool:
    try:
        mod = Benchmark.__new__(Benchmark)._baseline_module()
        return mod is not None and hasattr(mod, "_get_trinary_executor")
    except Exception:
        return False


BASELINE_AVAILABLE = (
    CUTENSOR_AVAILABLE
    or _TORCH_NPU_AVAILABLE()
    or _vendor_baseline_has_trinary()
    or vendor_baseline_available()
)

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


def _resolve_baseline_executor(dtype):
    """Return a callable ``executor(a, b, c, alpha=, beta=, gamma=, out=, ...)``.

    On NVIDIA this is the cuTensor trinary executor; on Ascend it is the
    torch_npu-aten based ``CuTensorTrinary`` instance; on any vendor that
    opts in via ``BASELINE_MODULE_NAME`` (e.g. MetaX) it is the vendor
    baseline module's ``_get_trinary_executor``. NVIDIA/Ascend are checked
    first (short-circuit), so their resolution path is unchanged.
    """
    if CUTENSOR_AVAILABLE:
        from flagtensor.cutensor import _get_trinary_executor as _cu_get
        op_a = _TRINARY_OP_KWARGS["op_a"]
        op_b = _TRINARY_OP_KWARGS["op_b"]
        op_c = _TRINARY_OP_KWARGS["op_c"]
        op_ab = _TRINARY_OP_KWARGS["op_ab"]
        op_abc = _TRINARY_OP_KWARGS["op_abc"]
        return _cu_get(op_ab, op_abc, op_a, op_b, op_c, dtype)
    if _TORCH_NPU_AVAILABLE():
        try:
            from flagtensor.torch_npu_baseline import CuTensorTrinary
            from flagtensor.cutensor import (
                UNARY_OPERATOR_MAP, BINARY_OPERATOR_MAP,
            )
            return CuTensorTrinary(
                op_ab=BINARY_OPERATOR_MAP[_TRINARY_OP_KWARGS["op_ab"]],
                op_abc=BINARY_OPERATOR_MAP[_TRINARY_OP_KWARGS["op_abc"]],
                op_a=UNARY_OPERATOR_MAP[_TRINARY_OP_KWARGS["op_a"]],
                op_b=UNARY_OPERATOR_MAP[_TRINARY_OP_KWARGS["op_b"]],
                op_c=UNARY_OPERATOR_MAP[_TRINARY_OP_KWARGS["op_c"]],
                dtype=dtype,
            )
        except Exception:
            pass
    # Vendor-neutral fallback (MetaX and any BASELINE_MODULE_NAME vendor).
    try:
        mod = Benchmark.__new__(Benchmark)._baseline_module()
        if mod is not None and hasattr(mod, "_get_trinary_executor"):
            return mod._get_trinary_executor(
                _TRINARY_OP_KWARGS["op_ab"], _TRINARY_OP_KWARGS["op_abc"],
                _TRINARY_OP_KWARGS["op_a"], _TRINARY_OP_KWARGS["op_b"],
                _TRINARY_OP_KWARGS["op_c"], dtype,
            )
    except Exception:
        pass
    return None


class TrinaryGenericBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=tuple(DEFAULT_BENCHMARK_DTYPES),
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
        executor = _resolve_baseline_executor(a.dtype)
        if executor is None:
            raise RuntimeError("No vendor baseline available for trinary on this device")
        return executor(
            a, b, c,
            alpha=_TRINARY_OP_KWARGS["alpha"],
            beta=_TRINARY_OP_KWARGS["beta"],
            gamma=_TRINARY_OP_KWARGS["gamma"],
        )

    def triton_impl(self, a, b, c):
        return elementwise_trinary(a, b, c, **_TRINARY_OP_KWARGS)

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
        executor = _resolve_baseline_executor(a.dtype)
        if executor is None:
            return None

        def _run():
            return executor(
                a, b, c,
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
                dtypes=tuple(DEFAULT_BENCHMARK_DTYPES),
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
        executor = _resolve_baseline_executor(a.dtype)
        if executor is None:
            raise RuntimeError("No vendor baseline available for trinary on this device")
        return executor(
            a, b, c,
            alpha=_TRINARY_OP_KWARGS["alpha"],
            beta=_TRINARY_OP_KWARGS["beta"],
            gamma=_TRINARY_OP_KWARGS["gamma"],
            **_INDEXED_MODE_KWARGS,
        )

    def triton_impl(self, a, b, c):
        return elementwise_trinary(a, b, c, **_TRINARY_OP_KWARGS, **_INDEXED_MODE_KWARGS)

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
        executor = _resolve_baseline_executor(a.dtype)
        if executor is None:
            return None

        def _run():
            return executor(
                a, b, c,
                alpha=_TRINARY_OP_KWARGS["alpha"],
                beta=_TRINARY_OP_KWARGS["beta"],
                gamma=_TRINARY_OP_KWARGS["gamma"],
                out=out,
                **_INDEXED_MODE_KWARGS,
            )

        return _run


@pytest.mark.performance
@pytest.mark.ElementwiseTrinary
def test_trinary_generic_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("Vendor baseline unavailable")

    bench = TrinaryGenericBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} mode={result.mode} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )


@pytest.mark.performance
@pytest.mark.ElementwiseTrinary
def test_trinary_generic_kernel_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("Vendor baseline unavailable")

    bench = TrinaryGenericKernelBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, f"{OP_NAME}_KERNEL")
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} mode={result.mode} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )


@pytest.mark.performance
@pytest.mark.ElementwiseTrinary
def test_trinary_generic_high_rank_indexed_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("Vendor baseline unavailable")

    bench = TrinaryGenericHighRankIndexedBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, f"{OP_NAME}_HIGH_RANK_INDEXED")
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
