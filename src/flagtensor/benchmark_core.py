from dataclasses import asdict, dataclass
import importlib
import os
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import torch
import triton

from flagtensor.cutensor import CUTENSOR_AVAILABLE

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
    if mode not in {"kernel", "operator"}:
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cutensor_available = CUTENSOR_AVAILABLE
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
        atol = 1e-5 if dtype != torch.float16 else 1e-3
        rtol = 1e-5 if dtype != torch.float16 else 1e-3
        return torch.allclose(reference, test, atol=atol, rtol=rtol)

    def _get_op_slug(self) -> str:
        prefix = "CUTENSOR_OP_"
        return self.op_name[len(prefix) :].lower() if self.op_name.startswith(prefix) else self.op_name.lower()

    def _get_baseline_store(self):
        baselines = getattr(self, "baselines", None)
        if baselines is None:
            baselines = {}
            self.baselines = baselines
        return baselines

    def _get_baseline_instance(self, dtype: torch.dtype):
        baselines = self._get_baseline_store()
        baseline = baselines.get(dtype)
        if baseline is not None:
            return baseline
        slug = self._get_op_slug()
        cutensor_module = importlib.import_module("flagtensor.cutensor")
        class_name = f"CuTensor{''.join(part.capitalize() for part in slug.split('_'))}"
        baseline_cls = getattr(cutensor_module, class_name, None)
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
        return None

    def build_baseline_kernel_callable(self, *args) -> Optional[Callable[[], torch.Tensor]]:
        if not args:
            return None
        baseline = self._get_baseline_instance(args[0].dtype)
        if baseline is None or not hasattr(baseline, "build_kernel_callable"):
            return None
        return baseline.build_kernel_callable(*args)

    def time_operator(self, fn: Callable, *args) -> Tuple[float, torch.Tensor]:
        for _ in range(self.config.warmup):
            result = fn(*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.config.repetitions):
            result = fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / self.config.repetitions, result

    def time_kernel(self, fn: Callable[[], torch.Tensor]) -> Tuple[float, torch.Tensor]:
        result = None
        for _ in range(self.config.warmup):
            result = fn()
        torch.cuda.synchronize()
        latency = triton.testing.do_bench(
            fn,
            warmup=self.config.warmup,
            rep=self.config.repetitions,
            return_mode="median",
        )
        torch.cuda.synchronize()
        result = fn()
        torch.cuda.synchronize()
        return latency, result

    def time_function(self, operator_fn: Callable, kernel_fn: Optional[Callable], *args) -> Tuple[float, torch.Tensor]:
        if self.config.mode == "kernel" and kernel_fn is not None:
            return self.time_kernel(kernel_fn)
        return self.time_operator(operator_fn, *args)

    def run(self) -> List[BenchmarkMetrics]:
        results: List[BenchmarkMetrics] = []
        for dtype in self.config.dtypes:
            for input_args in self.get_input_iter(dtype):
                args = input_args if isinstance(input_args, tuple) else (input_args,)
                shape = tuple(args[0].shape)
                reference = self.reference_impl(*args)
                triton_kernel = self.build_triton_kernel_callable(*args)
                baseline_latency = None
                baseline_kernel = None
                use_kernel_mode = self.config.mode == "kernel"
                if self.cutensor_available:
                    baseline_kernel = self.build_baseline_kernel_callable(*args)
                    use_kernel_mode = use_kernel_mode and triton_kernel is not None and baseline_kernel is not None
                else:
                    use_kernel_mode = use_kernel_mode and triton_kernel is not None

                triton_fn = triton_kernel if use_kernel_mode else None
                latency, triton_out = self.time_function(self.triton_impl, triton_fn, *args)

                if self.cutensor_available:
                    baseline_latency, baseline_out = self.time_function(
                        self.baseline_impl,
                        baseline_kernel if use_kernel_mode else None,
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
                    mode="kernel" if use_kernel_mode else "operator",
                    latency=latency,
                    latency_base=baseline_latency,
                    speedup=(baseline_latency / latency) if baseline_latency else None,
                )
                results.append(metric)
        return results
