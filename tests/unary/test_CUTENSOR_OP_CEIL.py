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

from tests.accuracy_utils import gems_assert_equal, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
from flagtensor import ceil


@pytest.mark.CUTENSOR_OP_CEIL
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_ceil_correctness(shape, dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")
    x = torch.randn(shape, device=_device_str, dtype=dtype)
    ref = to_reference(x, upcast=(dtype in (torch.float16, torch.float32, torch.bfloat16)))
    ref_out = torch.ceil(x)
    y = ceil(x)
    gems_assert_equal(y, ref_out)
