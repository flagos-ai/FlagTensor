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

import pytest
import torch

from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
from flagtensor import block_sparse_contraction

@pytest.mark.BlockSparseContraction
def test_block_sparse_contraction_smoke():
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    assert callable(block_sparse_contraction), "block_sparse_contraction should be callable"
