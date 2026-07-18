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

from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import acosh


@pytest.mark.CUTENSOR_OP_ACOSH
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_acosh_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(1, 3)
    ref = to_reference(x, upcast=True)
    ref_out = torch.acosh(x)
    y = acosh(x)
    gems_assert_close(y, ref_out, dtype)
