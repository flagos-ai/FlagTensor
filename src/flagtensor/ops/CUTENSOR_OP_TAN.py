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

from flagtensor import runtime
from flagtensor.utils import make_unary_pointwise_from_family


@triton.jit
def _tan_scalar(x):
    return tl.sin(x) / tl.cos(x)


_MTHREADS_REWRITE_RULES = (
    ("tan_divide", "tan_recip_divide")
    if runtime.device.vendor_name == "mthreads"
    else None
)


_tan_kernel, tan = make_unary_pointwise_from_family(
    "tan",
    "tan_like",
    _tan_scalar,
    rewrite_rules=_MTHREADS_REWRITE_RULES,
)
