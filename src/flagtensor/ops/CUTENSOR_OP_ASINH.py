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
def _asinh_scalar(x):
    abs_x = tl.abs(x)
    inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
    return tl.where(x >= 0, tl.log(inner), -tl.log(inner))


_asinh_kernel, asinh = make_unary_pointwise_from_family(
    "asinh",
    "asinh_like",
    _asinh_scalar,
    # asinh evaluates abs + sqrt + log with ~11 live fp32 temporaries, which
    # overflows the 192 KiB unified buffer at a 8192-element tile.
    config_key=(
        "elementwise_unary_persistent_heavy" if _is_ascend() else None
    ),
)
