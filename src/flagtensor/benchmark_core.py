from dataclasses import asdict, dataclass
import os
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import torch

from flagtensor.cutensor import CUTENSOR_AVAILABLE

DEFAULT_DTYPES = [torch.float16, torch.float32]
DEFAULT_SHAPES = [(2**i,) for i in range(10, 24)]
DEFAULT_WARMUP = 50
DEFAULT_REPETITIONS = 100
DEFAULT_METRICS = ["latency", "latency_base", "speedup"]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


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


@dataclass
class BenchmarkMetrics:
    shape: Tuple[int, ...]
    dtype: str
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
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cutensor_available = CUTENSOR_AVAILABLE

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

    def time_function(self, fn: Callable, *args) -> Tuple[float, torch.Tensor]:
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

    def run(self) -> List[BenchmarkMetrics]:
        results: List[BenchmarkMetrics] = []
        for dtype in self.config.dtypes:
            for input_args in self.get_input_iter(dtype):
                args = input_args if isinstance(input_args, tuple) else (input_args,)
                shape = tuple(args[0].shape)
                reference = self.reference_impl(*args)
                latency, triton_out = self.time_function(self.triton_impl, *args)
                baseline_latency = None
                if self.cutensor_available:
                    baseline_latency, baseline_out = self.time_function(self.baseline_impl, *args)
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
                    latency=latency,
                    latency_base=baseline_latency,
                    speedup=(baseline_latency / latency) if baseline_latency else None,
                )
                results.append(metric)
        return results
