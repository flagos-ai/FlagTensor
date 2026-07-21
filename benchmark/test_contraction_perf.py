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

from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
LEGACY_BENCHMARKS = ROOT
LEGACY_FILES = [
    LEGACY_BENCHMARKS / "test_Contraction_perf.py",
    LEGACY_BENCHMARKS / "test_ContractionTrinary_perf.py",
    LEGACY_BENCHMARKS / "test_ElementwiseTrinary_perf.py",
]


def _load_legacy_module(path: Path):
    spec = spec_from_file_location(f"legacy_benchmark_{path.stem}", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


for legacy_path in LEGACY_FILES:
    if not legacy_path.exists():
        continue
    legacy_module = _load_legacy_module(legacy_path)
    marker_name = legacy_path.stem.removeprefix("test_").removesuffix("_perf")
    marker = getattr(pytest.mark, marker_name)
    for name, value in vars(legacy_module).items():
        if name.startswith("test_"):
            globals()[f"{legacy_path.stem}__{name}"] = marker(value)
