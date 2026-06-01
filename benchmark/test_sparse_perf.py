from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
LEGACY_BENCHMARKS = ROOT
SPARSE_FILES = [
    LEGACY_BENCHMARKS / "test_CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION_perf.py",
]


def _load_legacy_module(path: Path):
    spec = spec_from_file_location(f"legacy_benchmark_{path.stem}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for legacy_path in SPARSE_FILES:
    if not legacy_path.exists():
        continue
    legacy_module = _load_legacy_module(legacy_path)
    marker_name = legacy_path.stem.replace("test_CUTENSOR_OP_", "").replace("_perf", "").lower()
    marker = getattr(pytest.mark, marker_name)
    for name, value in vars(legacy_module).items():
        if name.startswith("test_"):
            globals()[f"{legacy_path.stem}__{name}"] = marker(value)
