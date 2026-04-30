from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
LEGACY_BENCHMARKS = ROOT
UNARY_FILES = [
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ABS_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ACOS_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ACOSH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ASIN_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ASINH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ATAN_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ATANH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_CEIL_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_CONJ_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_COS_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_COSH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_EXP_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_FLOOR_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_IDENTITY_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_LOG_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_MISH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_NEG_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_RCP_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_RELU_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SIGMOID_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SIN_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SINH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SOFT_PLUS_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SOFT_SIGN_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SQRT_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_SWISH_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_TAN_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_TANH_perf.py",
]


def _load_legacy_module(path: Path):
    spec = spec_from_file_location(f"legacy_benchmark_{path.stem}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for legacy_path in UNARY_FILES:
    if not legacy_path.exists():
        continue
    legacy_module = _load_legacy_module(legacy_path)
    marker_name = legacy_path.stem.replace("test_CUTENSOR_OP_", "").replace("_perf", "").lower()
    marker = getattr(pytest.mark, marker_name)
    for name, value in vars(legacy_module).items():
        if name.startswith("test_"):
            globals()[f"{legacy_path.stem}__{name}"] = marker(value)
