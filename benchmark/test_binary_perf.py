from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
LEGACY_BENCHMARKS = ROOT
BINARY_FILES = [
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_ADD_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_MUL_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_MAX_perf.py",
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_MIN_perf.py",
]


def _load_legacy_module(path: Path):
    spec = spec_from_file_location(f"legacy_benchmark_{path.stem}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for legacy_path in BINARY_FILES:
    if not legacy_path.exists():
        continue
    legacy_module = _load_legacy_module(legacy_path)
    marker_name = legacy_path.stem.removeprefix("test_").removesuffix("_perf")
    marker = getattr(pytest.mark, marker_name)
    for name, value in vars(legacy_module).items():
        if name.startswith("test_"):
            globals()[f"{legacy_path.stem}__{name}"] = marker(value)
