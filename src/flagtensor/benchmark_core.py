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

from dataclasses import asdict, dataclass
import importlib
import os
import time
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import torch
import triton

from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.runtime import (
    device_str as _default_device_str,
    empty_cache as _device_empty_cache,
    is_accelerator_available as _is_accelerator_available,
    is_on_accelerator as _is_on_accelerator,
    synchronize as _device_synchronize,
)

DEFAULT_DTYPES = [torch.float16, torch.float32]
DEFAULT_SHAPES = [(2**i,) for i in range(10, 24)]
DEFAULT_WARMUP = 50
DEFAULT_REPETITIONS = 100
DEFAULT_METRICS = ["latency", "latency_base", "speedup"]
DEFAULT_MODE = "kernel"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_mode(default: str) -> str:
    value = os.getenv("FLAGTENSOR_BENCHMARK_MODE")
    if value is None or value == "":
        return default
    mode = value.strip().lower()
    if mode not in {"kernel", "operator", "wrapper"}:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    return mode


def _env_shape_limit(default_shapes: Sequence[Tuple[int, ...]]) -> Sequence[Tuple[int, ...]]:
    value = os.getenv("FLAGTENSOR_BENCHMARK_MAX_SHAPES")
    if value is None or value == "":
        return tuple(default_shapes)
    limit = max(1, int(value))
    return tuple(default_shapes[:limit])


def _env_dtype_filter(default_dtypes: Sequence[torch.dtype]) -> Sequence[torch.dtype]:
    value = os.getenv("FLAGTENSOR_BENCHMARK_DTYPES")
    if value is None or value == "":
        return tuple(default_dtypes)
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "complex64": torch.complex64,
        "cfloat": torch.complex64,
        "complex128": torch.complex128,
        "cdouble": torch.complex128,
    }
    resolved = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        dtype = aliases.get(key)
        if dtype is not None and dtype in default_dtypes:
            resolved.append(dtype)
    return tuple(resolved or default_dtypes)


@dataclass
class BenchmarkConfig:
    warmup: int = DEFAULT_WARMUP
    repetitions: int = DEFAULT_REPETITIONS
    dtypes: Sequence[torch.dtype] = tuple(DEFAULT_DTYPES)
    shapes: Sequence[Tuple[int, ...]] = tuple(DEFAULT_SHAPES)
    metrics: Sequence[str] = tuple(DEFAULT_METRICS)
    mode: str = DEFAULT_MODE


@dataclass
class BenchmarkMetrics:
    shape: Tuple[int, ...]
    dtype: str
    mode: str
    latency: Optional[float] = None
    latency_base: Optional[float] = None
    speedup: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class Benchmark:
    def __init__(self, op_name: str, config: Optional[BenchmarkConfig] = None):
        self.op_name = op_name
        self.config = config or BenchmarkConfig()
        self.config = BenchmarkConfig(
            warmup=_env_int("FLAGTENSOR_BENCHMARK_WARMUP", self.config.warmup),
            repetitions=_env_int("FLAGTENSOR_BENCHMARK_REPETITIONS", self.config.repetitions),
            dtypes=_env_dtype_filter(self.config.dtypes),
            shapes=_env_shape_limit(self.config.shapes),
            metrics=tuple(self.config.metrics),
            mode=_env_mode(self.config.mode),
        )
        self.device = _default_device_str if _is_accelerator_available() else "cpu"
        self.cutensor_available = CUTENSOR_AVAILABLE
        # On non-NVIDIA backends we fall back to a vendor-native baseline
        # (torch_npu-aten on Ascend). ``baseline_available`` is the
        # vendor-neutral flag the run() loop checks against.
        self.baseline_available = CUTENSOR_AVAILABLE or self._baseline_module() is not None
        self._auto_kernel_cache = {}

    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        raise NotImplementedError

    def baseline_impl(self, *args):
        raise NotImplementedError

    def triton_impl(self, *args):
        raise NotImplementedError

    def reference_impl(self, *args):
        return args[0]

    def verify(self, reference: torch.Tensor, test: torch.Tensor, dtype: torch.dtype):
        """Verify correctness between reference and test tensors.

        Uses relaxed tolerances compared to accuracy tests: benchmark verification
        only needs to catch major errors, not ulp-level differences caused by
        Tensor Core rounding differences across GPU architectures.
        Accuracy correctness is validated separately by per-operator tests.
        """
        from flagtensor.testing.assertions import get_tolerance as _get_tol

        atol, rtol = _get_tol(dtype)
        # Relax tolerance for benchmark comparison (10x looser than accuracy tests)
        # because different GPU archs (Ampere vs Hopper) have slightly different
        # Tensor Core rounding behavior for contractions
        atol = max(atol, 1e-4)
        rtol = max(rtol, 1e-4)
        return torch.allclose(reference, test, atol=atol, rtol=rtol)

    def _get_op_slug(self) -> str:
        _OP_PREFIX = "CUTENSOR_OP_"
        return self.op_name[len(_OP_PREFIX) :].lower() if self.op_name.startswith(_OP_PREFIX) else self.op_name.lower()

    # Mapping for ops whose FlagTensor name does not match the cuTensor /
    # torch_npu_baseline class name. Keys are FlagTensor op_name values,
    # values are the class-name suffix (without the ``CuTensor`` prefix).
    _OP_NAME_TO_BASELINE_SUFFIX = {
        "Contraction": "Contraction",
        "ContractionTrinary": "ContractionTrinary",
        "ElementwiseTrinary": "Trinary",  # matches CuTensorTrinary
        "BlockSparseContraction": "BlockSparseContraction",
        # Some perf files use the legacy CUTENSOR_OP_* op_name even when the
        # underlying op is a Contraction/Trinary variant.
        "CUTENSOR_OP_GETT": "Contraction",
        "CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY": "ContractionTrinary",
        "CUTENSOR_OP_TRINARY_GENERIC": "Trinary",
        "CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION": "BlockSparseContraction",
    }

    def _get_baseline_store(self):
        baselines = getattr(self, "baselines", None)
        if baselines is None:
            baselines = {}
            self.baselines = baselines
        return baselines

    def _baseline_module(self):
        """Return the vendor-native baseline module, or None.

        On NVIDIA this is ``flagtensor.cutensor`` (cuTensor ctypes bindings).
        On Ascend this is ``flagtensor.torch_npu_baseline`` (CANN aclnn-backed
        torch_npu aten ops). The two modules expose identically-named
        ``CuTensor{Op}`` classes so the rest of the code is vendor-agnostic.
        """
        if CUTENSOR_AVAILABLE:
            return importlib.import_module("flagtensor.cutensor")
        try:
            mod = importlib.import_module("flagtensor.torch_npu_baseline")
            if mod.torch_npu_available():
                return mod
        except Exception:
            pass
        return None

    def _get_baseline_instance(self, dtype: torch.dtype):
        baselines = self._get_baseline_store()
        baseline = baselines.get(dtype)
        if baseline is not None:
            return baseline
        baseline_module = self._baseline_module()
        if baseline_module is None:
            return None
        # Resolve the baseline class name. Most ops follow the
        # ``CuTensor{SlugCamelCase}`` convention derived from op_name, but a
        # few ops (Contraction/Trinary/BlockSparse) have a class name that
        # does not match the op slug, so consult the explicit map first.
        suffix = self._OP_NAME_TO_BASELINE_SUFFIX.get(self.op_name)
        if suffix is None:
            slug = self._get_op_slug()
            suffix = "".join(part.capitalize() for part in slug.split("_"))
        class_name = f"CuTensor{suffix}"
        baseline_cls = getattr(baseline_module, class_name, None)
        if baseline_cls is None:
            return None
        baseline = baseline_cls(dtype=dtype)
        baselines[dtype] = baseline
        return baseline

    def _resolve_triton_kernel(self):
        kernel = self._auto_kernel_cache.get("triton_kernel")
        if kernel is not None:
            return kernel
        slug = self._get_op_slug()
        module = importlib.import_module(f"flagtensor.ops.CUTENSOR_OP_{slug.upper()}")
        kernel = getattr(module, f"_{slug}_kernel", None)
        self._auto_kernel_cache["triton_kernel"] = kernel
        return kernel

    def _build_unary_triton_kernel_callable(self, kernel, x):
        if kernel is None:
            return None
        if self._get_op_slug() == "conj":
            y = torch.empty_like(x)
            input_x = x.contiguous() if not x.is_contiguous() else x
            real_dtype = torch.float32 if input_x.dtype == torch.complex64 else torch.float64
            x_view = input_x.view(real_dtype)
            y_view = y.view(real_dtype)
            n_elements = input_x.numel()
            grid = lambda meta: (
                triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
            )

            def run_kernel():
                kernel[grid](x_view, y_view, n_elements)
                return y

            return run_kernel

        y = torch.empty_like(x)
        n_elements = y.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )

        def run_kernel():
            kernel[grid](x, y, n_elements)
            return y

        return run_kernel

    def _build_trinary_triton_kernel_callable(self, kernel, x, y, z):
        if kernel is None:
            return None
        out = torch.empty_like(z)
        n_elements = out.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )

        def run_kernel():
            kernel[grid](x, y, z, out, n_elements)
            return out

        return run_kernel

    def _build_binary_triton_kernel_callable(self, kernel, x, y):
        if kernel is None:
            return None
        z = torch.empty_like(x)
        n_elements = z.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )

        def run_kernel():
            kernel[grid](x, y, z, n_elements)
            return z

        return run_kernel

    def build_triton_kernel_callable(self, *args) -> Optional[Callable[[], torch.Tensor]]:
        kernel = self._resolve_triton_kernel()
        if len(args) == 1:
            return self._build_unary_triton_kernel_callable(kernel, args[0])
        if len(args) == 2:
            return self._build_binary_triton_kernel_callable(kernel, args[0], args[1])
        if len(args) == 3:
            return self._build_trinary_triton_kernel_callable(kernel, args[0], args[1], args[2])
        return None

    def build_baseline_kernel_callable(self, *args) -> Optional[Callable[[], torch.Tensor]]:
        if not args:
            return None
        baseline = self._get_baseline_instance(args[0].dtype)
        if baseline is None or not hasattr(baseline, "build_kernel_callable"):
            return None
        return baseline.build_kernel_callable(*args)

    def build_triton_wrapper_callable(self, *args) -> Optional[Callable[[], torch.Tensor]]:
        return lambda: self.triton_impl(*args)

    def build_baseline_wrapper_callable(self, *args) -> Optional[Callable[[], torch.Tensor]]:
        if not self.baseline_available:
            return None
        return lambda: self.baseline_impl(*args)

    def _synchronize_device(self):
        _device_synchronize()

    def _time_host_loop(self, fn: Callable[[], torch.Tensor], *, synchronize_before_end: bool) -> Tuple[float, torch.Tensor]:
        result = None
        for _ in range(self.config.warmup):
            result = fn()
        self._synchronize_device()
        start = time.time()
        for _ in range(self.config.repetitions):
            result = fn()
        if synchronize_before_end:
            self._synchronize_device()
        end = time.time()
        return (end - start) / self.config.repetitions * 1000.0, result

    def time_operator(self, fn: Callable, *args) -> Tuple[float, torch.Tensor]:
        return self._time_host_loop(lambda: fn(*args), synchronize_before_end=True)

    def time_kernel(self, fn: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
        result = None
        for _ in range(self.config.warmup):
            result = fn()
        self._synchronize_device()

        if self._use_npu_graph:
            latency = self._do_bench_npu_graph(fn)
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=self.config.warmup,
                rep=self.config.repetitions,
                return_mode="median",
            )

        self._synchronize_device()
        result = fn()
        self._synchronize_device()
        return latency, result

    @property
    def _use_npu_graph(self) -> bool:
        """On Ascend, triton-ascend's Python launch path is ~130us/call
        (libtuner + MLIR runtime), vs ~10us/call for aten baseline. This
        makes ``triton.testing.do_bench`` measure Python submission rate
        rather than GPU kernel execution time.

        When running on Ascend, use NPU graph capture to eliminate
        Python launch overhead: capture the triton kernel into a graph,
        then replay the graph (C++ replay path, ~1us overhead).
        """
        try:
            from flagtensor.runtime import device as _ft_device
            if _ft_device.vendor_name != "ascend":
                return False
            import torch
            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False

    def _do_bench_npu_graph(self, fn: Callable[[], torch.Tensor]) -> float:
        """Benchmark fn() using NPU graph capture (eliminates Python launch overhead).

        Captures the callable into a NPU graph during warmup, then uses
        ``triton.testing.do_bench`` on the graph replay (which only goes
        through the C++ replay path, ~1us overhead per call).

        Falls back to plain ``do_bench`` if graph capture fails (e.g. the
        callable uses dynamic shapes or unsupported ops).
        """
        import torch

        # Try to capture the callable into a NPU graph.
        # We need sample args; the fn signature is `() -> Tensor`,
        # so we call it once to get the result tensor, then capture.
        try:
            sample = fn()
            torch.npu.synchronize()

            # make_graphed_callables expects a callable (args) -> result
            # We wrap fn (no args) to match.
            def wrapper(*args):
                return fn()

            graphed_fn = torch.npu.make_graphed_callables(wrapper, (sample,))
            replay_fn = lambda: graphed_fn(sample)

            # warmup the graph replay
            for _ in range(self.config.warmup):
                replay_fn()
            self._synchronize_device()

            latency = triton.testing.do_bench(
                replay_fn,
                warmup=self.config.warmup,
                rep=self.config.repetitions,
                return_mode="median",
            )
            return latency
        except Exception:
            # Fallback: plain do_bench (includes Python launch overhead,
            # but at least produces a result instead of timing out).
            return triton.testing.do_bench(
                fn,
                warmup=self.config.warmup,
                rep=self.config.repetitions,
                return_mode="median",
            )

    def time_wrapper(self, fn: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
        return self._time_host_loop(fn, synchronize_before_end=False)

    def time_function(
        self,
        operator_fn: Callable,
        kernel_fn: Optional[Callable],
        wrapper_fn: Optional[Callable],
        *args,
    ) -> Tuple[float, torch.Tensor]:
        if self.config.mode == "kernel" and kernel_fn is not None:
            return self.time_kernel(kernel_fn)
        if self.config.mode == "wrapper" and wrapper_fn is not None:
            return self.time_wrapper(wrapper_fn)
        return self.time_operator(operator_fn, *args)

    def run(self) -> List[BenchmarkMetrics]:
        results: List[BenchmarkMetrics] = []
        for dtype in self.config.dtypes:
            for input_args in self.get_input_iter(dtype):
                args = input_args if isinstance(input_args, tuple) else (input_args,)
                shape = tuple(args[0].shape)
                reference = self.reference_impl(*args)
                triton_kernel = self.build_triton_kernel_callable(*args)
                triton_wrapper = self.build_triton_wrapper_callable(*args)
                baseline_latency = None
                baseline_kernel = None
                baseline_wrapper = None
                selected_mode = "operator"
                use_kernel_mode = self.config.mode == "kernel"
                if self.baseline_available:
                    baseline_kernel = self.build_baseline_kernel_callable(*args)
                    baseline_wrapper = self.build_baseline_wrapper_callable(*args)
                    use_kernel_mode = use_kernel_mode and triton_kernel is not None and baseline_kernel is not None
                else:
                    use_kernel_mode = use_kernel_mode and triton_kernel is not None

                if use_kernel_mode:
                    selected_mode = "kernel"
                elif self.config.mode == "wrapper":
                    if self.baseline_available:
                        if triton_wrapper is not None and baseline_wrapper is not None:
                            selected_mode = "wrapper"
                    elif triton_wrapper is not None:
                        selected_mode = "wrapper"

                latency, triton_out = self.time_function(
                    self.triton_impl,
                    triton_kernel if selected_mode == "kernel" else None,
                    triton_wrapper if selected_mode == "wrapper" else None,
                    *args,
                )

                if self.baseline_available:
                    baseline_latency, baseline_out = self.time_function(
                        self.baseline_impl,
                        baseline_kernel if selected_mode == "kernel" else None,
                        baseline_wrapper if selected_mode == "wrapper" else None,
                        *args,
                    )
                    if not self.verify(reference, baseline_out, dtype):
                        raise AssertionError(f"baseline correctness failed for {shape} {dtype}")
                    if not self.verify(baseline_out, triton_out, dtype):
                        raise AssertionError(f"triton correctness failed for {shape} {dtype}")
                else:
                    if not self.verify(reference, triton_out, dtype):
                        raise AssertionError(f"triton correctness failed for {shape} {dtype}")
                metric = BenchmarkMetrics(
                    shape=shape,
                    dtype=str(dtype),
                    mode=selected_mode,
                    latency=latency,
                    latency_base=baseline_latency,
                    speedup=(baseline_latency / latency) if baseline_latency else None,
                )
                results.append(metric)
                # Inject into benchmark conftest recording (if running under pytest --record json)
                _try_record_benchmark(self.op_name, asdict(metric))
        return results


# ---------------------------------------------------------------------------
# Bridge to benchmark/conftest.py recording system
# ---------------------------------------------------------------------------
def _try_record_benchmark(op_name: str, metric: dict) -> None:
    """If benchmark/conftest.py is loaded with --record json, push metric data."""
    import sys
    bm = sys.modules.get("conftest")
    if bm is None:
        return
    Config = getattr(bm, "Config", None)
    if Config is None or not Config.record_json:
        return
    update_result = getattr(bm, "update_result", None)
    if update_result is None:
        return

    # Derive level from bench conftest Config (mirrors flag_gems default: "core")
    bench_level = getattr(Config, "bench_level", None)
    if bench_level is not None:
        level = bench_level.value if hasattr(bench_level, "value") else str(bench_level)
    else:
        level = "core"

    shape_tuple = metric.get("shape", ())
    mode = metric.get("mode", "kernel")
    dtype = metric.get("dtype", "")

    data = {
        "op_name": op_name,
        "dtype": dtype,
        "mode": mode,
        "level": level,
        "result": [{
            "legacy_shape": None,
            "shape_detail": [list(shape_tuple)],
            "latency_base": metric.get("latency_base", 0) or 0,
            "latency": metric.get("latency", 0) or 0,
            "gbps_base": None,
            "gbps": None,
            "speedup": metric.get("speedup", 0) or 0,
            "accuracy": None,
            "tflops": None,
            "utilization": None,
            "compared_speedup": None,
            "error_msg": None,
        }],
    }
    update_result(op_name, data)


# Set by benchmark/conftest.py pytest_configure for --record json
_record_callback = None
