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
from flagtensor.utils.ascend_runtime import is_ascend as _is_ascend


@triton.jit
def _atan_scalar(x):
    return x


_atan_kernel, atan = make_unary_pointwise_from_family(
    "atan",
    "atan_like",
    _atan_scalar,
    # The fp32 atan lowering keeps several UB temporaries live and overflows
    # the unified buffer at tile 8192 on Ascend ("ub overflow, requires
    # 2097152 bits while 1572864 bits available"), so use the heavy table.
    config_key=(
        "elementwise_unary_persistent_heavy" if _is_ascend() else None
    ),
)
