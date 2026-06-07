"""Benchmark constants for FlagTensor.

Provides shared dtype lists, shape definitions, and benchmark configuration
following the FlagGems benchmark/consts.py pattern.
"""
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# FP8 detection — A100 (Ampere SM80) does NOT support float8, returns None
# ---------------------------------------------------------------------------


def get_fp8_dtype() -> Optional[torch.dtype]:
    """Return the appropriate FP8 dtype for the current GPU, or None if unsupported.

    Hopper+ (SM90): float8_e4m3fn
    Ampere (SM80): float8_e5m2 — NOT supported on A100, so returns None
    Older GPUs:   None
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        return torch.float8_e4m3fn
    # A100 (SM80) advertises float8_e5m2 but does not actually support it
    # for any kernel operations. Return None to skip FP8 benchmarks on Ampere.
    return None


# ---------------------------------------------------------------------------
# Shared dtype lists
# ---------------------------------------------------------------------------
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int32, torch.int64]
BOOL_DTYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex64]
FP8_DTYPE = get_fp8_dtype()
FP8_DTYPES = [FP8_DTYPE] if FP8_DTYPE is not None else []
ALL_BENCHMARK_DTYPES = FLOAT_DTYPES + FP8_DTYPES + INT_DTYPES + BOOL_DTYPES

# ---------------------------------------------------------------------------
# Benchmark timing defaults
# ---------------------------------------------------------------------------
DEFAULT_WARMUP_COUNT = 1000
DEFAULT_ITER_COUNT = 100

# ---------------------------------------------------------------------------
# Default shapes (aligned with FlagGems)
# ---------------------------------------------------------------------------
DEFAULT_SHAPES = [
    (1024 * 1024 * 1024,),  # 1D large
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),  # 3D large
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BenchMode(Enum):
    KERNEL = "kernel"
    OPERATOR = "operator"
    WRAPPER = "wrapper"


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


# ---------------------------------------------------------------------------
# Default metrics
# ---------------------------------------------------------------------------
DEFAULT_METRICS = ["latency_base", "latency", "speedup"]


# ---------------------------------------------------------------------------
# Benchmark metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetrics:
    """Per-shape benchmark metrics."""

    shape: tuple = ()
    shape_detail: str = ""
    dtype: str = ""
    latency_base: float = 0.0  # reference baseline latency (ms)
    latency: float = 0.0  # FlagTensor latency (ms)
    speedup: float = 0.0  # latency_base / latency
    gbps: float = 0.0
    tflops: float = 0.0
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "shape_detail": self.shape_detail,
            "dtype": self.dtype,
            "latency_base": self.latency_base,
            "latency": self.latency,
            "speedup": self.speedup,
            "gbps": self.gbps,
            "tflops": self.tflops,
        }


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result for one operator."""

    op_name: str
    dtype: str
    mode: str = "operator"
    level: str = "core"
    result: List[BenchmarkMetrics] = field(default_factory=list)

    @property
    def avg_speedup(self) -> float:
        if not self.result:
            return 0.0
        return sum(m.speedup for m in self.result) / len(self.result)

    def to_dict(self) -> dict:
        return {
            "op_name": self.op_name,
            "dtype": self.dtype,
            "mode": self.mode,
            "level": self.level,
            "avg_speedup": self.avg_speedup,
            "details": [m.to_dict() for m in self.result],
        }


# ---------------------------------------------------------------------------
# All available metric names (derived from BenchmarkMetrics fields)
# ---------------------------------------------------------------------------
ALL_AVAILABLE_METRICS = set(f.name for f in fields(BenchmarkMetrics)) - {
    "shape", "error_msg", "dtype",
}


def check_metric_dependencies(requested: List[str]) -> List[str]:
    """Ensure requested metrics are valid; return sorted deduped list."""
    valid = [m for m in requested if m in ALL_AVAILABLE_METRICS]
    return sorted(set(valid))


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def get_recommended_shapes(
    op_name: str, user_shapes: Optional[List] = None
) -> Optional[List]:
    """Return recommended core shapes for an operator, or user-provided shapes."""
    if user_shapes:
        return user_shapes
    return None  # use DEFAULT_SHAPES


@dataclass
class OperationAttribute:
    """Metadata about an operator's benchmark configuration."""

    op_name: str = ""
    recommended_core_shapes: Optional[List] = None
    shape_desc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_name": self.op_name,
            "recommended_core_shapes": self.recommended_core_shapes,
            "shape_desc": self.shape_desc,
        }
