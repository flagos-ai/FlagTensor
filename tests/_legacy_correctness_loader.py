from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path
from typing import Iterable

import pytest

from flagtensor_registry import load_operator_registry

ROOT = Path(__file__).resolve().parent.parent


def _load_legacy_module(path: Path):
    spec = spec_from_file_location(f"legacy_{path.stem}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _marker_name(path: Path) -> str:
    stem = path.stem
    prefix = "test_CUTENSOR_OP_"
    if stem.startswith(prefix):
        return stem[len(prefix) :].lower()
    return stem.lower()


def iter_legacy_correctness_files(category: str, skipped_names: Iterable[str] = ()):
    skipped = {name.strip().lower() for name in skipped_names if name and name.strip()}
    for spec in load_operator_registry():
        if spec.category != category:
            continue
        if spec.name.lower() in skipped:
            continue
        legacy_path = ROOT / spec.correctness_test
        if legacy_path.exists():
            yield legacy_path


def populate_category_proxy(namespace, category: str, skipped_names: Iterable[str] = ()):
    for legacy_path in sorted(iter_legacy_correctness_files(category, skipped_names=skipped_names)):
        legacy_module = _load_legacy_module(legacy_path)
        marker = getattr(pytest.mark, _marker_name(legacy_path))
        for name, value in vars(legacy_module).items():
            if name.startswith("test_"):
                namespace[f"{legacy_path.stem}__{name}"] = marker(value)
