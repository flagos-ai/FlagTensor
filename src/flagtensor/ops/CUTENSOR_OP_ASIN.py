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

import torch
import triton
import triton.language as tl

from flagtensor.utils import make_unary_pointwise_from_family
from flagtensor.runtime import is_ascend as _is_ascend


@triton.jit
def _asin_scalar(x):
    return x


_asin_kernel, asin = make_unary_pointwise_from_family(
    "asin",
    "asin_like",
    _asin_scalar,
    # The asin/acos polynomial needs more UB temporaries than the other unary
    # kernels; on Ascend it gets a dedicated (smaller-max-tile) tuning table.
    config_key=(
        "elementwise_unary_persistent_heavy" if _is_ascend() else None
    ),
)
