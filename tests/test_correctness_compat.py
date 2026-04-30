from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEGACY_CTESTS = ROOT / "ctests"


def _legacy_correctness_files():
    return sorted(path for path in LEGACY_CTESTS.glob("test_CUTENSOR_OP_*.py") if path.is_file())


def pytest_generate_tests(metafunc):
    if "legacy_test_file" in metafunc.fixturenames:
        metafunc.parametrize("legacy_test_file", _legacy_correctness_files())


def test_legacy_correctness_files_exist(legacy_test_file):
    assert legacy_test_file.exists()


def test_legacy_correctness_layout_is_populated():
    assert _legacy_correctness_files(), "No correctness tests found under ctests/"
