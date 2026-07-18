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
