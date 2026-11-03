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
def _atanh_scalar(x):
    return 0.5 * tl.log((1 + x) / (1 - x))


_MTHREADS_REWRITE_RULES = (
    ("atanh_log_ratio", "atanh_log_ratio")
    if runtime.device.vendor_name == "mthreads"
    else None
)


_atanh_kernel, atanh = make_unary_pointwise_from_family(
    "atanh",
    "atanh_like",
    _atanh_scalar,
    rewrite_rules=_MTHREADS_REWRITE_RULES,
)
