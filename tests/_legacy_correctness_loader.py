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

"""Backward-compat loader stub.

Older correctness modules referenced a ``populate_category_proxy`` helper
that imported category-level correctness tests into a per-op test module's
namespace. The current layout uses per-op test files directly, so the
proxy is a no-op. This stub lets legacy modules import without errors.
"""

def populate_category_proxy(globals_dict, category, skipped_names=()):
    """No-op stub kept for backwards compatibility."""
    return None
