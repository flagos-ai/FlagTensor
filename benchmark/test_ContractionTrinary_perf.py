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

from flagtensor import contraction_trinary
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig, get_vendor_baseline_class, vendor_baseline_available
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES, DEFAULT_TENSOR_CONTRACTION_TRINARY_BENCHMARK_SHAPES
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
# Operator-mode baseline: the full A@B@C chain. NVIDIA -> cuTensor chain
# executor; Ascend -> torch_npu chain; other vendors -> vendor-native chain
# baseline (e.g. Iluvatar CoreX PyTorch-native).
if CUTENSOR_AVAILABLE:
    from flagtensor.cutensor import CuTensorContractionTrinary as _ContractionTrinaryBaseline
elif _TORCH_NPU_AVAILABLE():
    from flagtensor.torch_npu_baseline import CuTensorContractionTrinary as _ContractionTrinaryBaseline
else:
    _ContractionTrinaryBaseline = get_vendor_baseline_class("contraction_trinary")
from flagtensor.ops.CUTENSOR_OP_GETT import _launch_gett_kernel
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


def _tensor_contraction_trinary_case(shape_a, shape_b, shape_c):
    return (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
        (0, 3),
        (shape_a[0], shape_c[1]),
        lambda a, b, c: torch.matmul(torch.matmul(a, b), c),
    )


def _tensor_contraction_trinary_reference(a, b, c, d, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        temp = torch.matmul(a.float(), b.float()).to(a.dtype)
        return (1.25 * torch.matmul(temp.float(), c.float()) + 0.5 * d.float()).to(a.dtype)
    return 1.25 * reference(a, b, c) + 0.5 * d


class TensorContractionTrinaryBenchmark(Benchmark):
    def __init__(self, mode="kernel"):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=(torch.float32,),
                shapes=tuple(DEFAULT_TENSOR_CONTRACTION_TRINARY_BENCHMARK_SHAPES),
                mode=mode,
            ),
        )
        self.operator_baselines = {}
        self.kernel_baselines = {}

    def get_input_iter(self, dtype: torch.dtype):
        for shape_a, shape_b, shape_c in self.config.shapes:
            _, _, _, _, _, d_shape, _ = _tensor_contraction_trinary_case(shape_a, shape_b, shape_c)
            yield (
                torch.empty(shape_a, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(shape_b, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(shape_c, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                torch.empty(d_shape, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
            )

    def baseline_impl(self, a, b, c, d):
        baseline = self.operator_baselines.get(a.dtype)
        if baseline is None:
            baseline = self._get_baseline_instance(a.dtype)
            self.operator_baselines[a.dtype] = baseline
        mode_a, mode_b, mode_c, mode_d, mode_e, _, _ = _tensor_contraction_trinary_case(tuple(a.shape), tuple(b.shape), tuple(c.shape))
        return baseline(a, b, c, d=d, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_c, mode_d=mode_d, mode_e=mode_e)

    def triton_impl(self, a, b, c, d):
        mode_a, mode_b, mode_c, mode_d, mode_e, _, _ = _tensor_contraction_trinary_case(tuple(a.shape), tuple(b.shape), tuple(c.shape))
        return contraction_trinary(a, b, c, d=d, alpha=1.25, beta=0.5, mode_a=mode_a, mode_b=mode_b, mode_c=mode_c, mode_d=mode_d, mode_e=mode_e)

    def reference_impl(self, a, b, c, d):
        _, _, _, _, _, _, reference = _tensor_contraction_trinary_case(tuple(a.shape), tuple(b.shape), tuple(c.shape))
        return _tensor_contraction_trinary_reference(a, b, c, d, reference)

    def build_triton_kernel_callable(self, *args):
        a, b, c, d = args
        mode_a, mode_b, mode_c, mode_d, mode_e, _, _ = _tensor_contraction_trinary_case(tuple(a.shape), tuple(b.shape), tuple(c.shape))
        # Only support default 2D chain case for kernel mode
        if not (mode_a == (0, 1) and mode_b == (1, 2) and mode_c == (2, 3) and mode_d == (0, 3) and mode_e == (0, 3)):
            return None
        if not (a.ndim == 2 and b.ndim == 2 and c.ndim == 2):
            return None
        if a.dtype not in (torch.float16, torch.float32):
            return None
        intermediate = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        out = torch.empty((a.shape[0], c.shape[1]), device=a.device, dtype=a.dtype)

        def run_kernel():
            # Time the two-step GETT path — the same path contraction_trinary()
            # actually dispatches to for all dtypes (the fused kernel is
            # disabled in the op: _supports_fused_triton_trinary() is False).
            _launch_gett_kernel(a, b, None, intermediate, 1.0, 0.0)
            _launch_gett_kernel(intermediate, c, d, out, 1.25, 0.5)
            return out

        return run_kernel

    def build_baseline_kernel_callable(self, *args):
        # The kernel-mode two-step GETT path drives the trinary baseline
        # with c=None for the first contraction plus intricate mode
        # remapping for the second — only validated against the cuTensor
        # baseline. On vendors without cuTensor (e.g. MetaX), return None
        # so the harness falls back to operator-mode timing (baseline_impl,
        # which is vendor-neutral and correct). NVIDIA keeps the kernel path.
        if not CUTENSOR_AVAILABLE:
            return None
        a, b, c, d = args
        mode_a, mode_b, mode_c, mode_d, mode_e, _, _ = _tensor_contraction_trinary_case(tuple(a.shape), tuple(b.shape), tuple(c.shape))
        # Only support default 2D chain case for kernel mode
        if not (mode_a == (0, 1) and mode_b == (1, 2) and mode_c == (2, 3) and mode_d == (0, 3) and mode_e == (0, 3)):
            return None
        if not (a.ndim == 2 and b.ndim == 2 and c.ndim == 2):
            return None
        baseline = self.kernel_baselines.get(a.dtype)
        if baseline is None:
            baseline = self._get_baseline_instance(a.dtype)
            self.kernel_baselines[a.dtype] = baseline
        # Pre-allocate intermediate buffer
        intermediate = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)

        def run_kernel():
            # First contraction: a @ b -> intermediate
            baseline(a, b, c=None, alpha=1.0, beta=0.0, mode_a=mode_a, mode_b=mode_b, mode_c=(0, 2), mode_d=(0, 2), out=intermediate)
            # Second contraction: intermediate @ c + d -> out
            return baseline(intermediate, c, c=d, alpha=1.25, beta=0.5, mode_a=(0, 2), mode_b=mode_c, mode_c=mode_d, mode_d=mode_e)

        return run_kernel


@pytest.mark.performance
@pytest.mark.ContractionTrinary
def test_tensor_contraction_trinary_perf():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    if not BASELINE_AVAILABLE:
        pytest.skip("Vendor baseline unavailable")

    kernel_bench = TensorContractionTrinaryBenchmark(mode="kernel")
    operator_bench = TensorContractionTrinaryBenchmark(mode="operator")
    results = kernel_bench.run() + operator_bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"mode={result.mode} shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
