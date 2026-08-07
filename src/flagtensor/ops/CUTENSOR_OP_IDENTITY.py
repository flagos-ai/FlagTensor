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
from flagtensor.utils.unary_pointwise import _TRIVIAL_UNARY_EXTRA, _DEFAULT_UNARY_DTYPES


@triton.jit
def _identity_scalar(x):
    return x


_identity_kernel, identity = make_unary_pointwise_from_family(
    "identity",
    "identity_like",
    _identity_scalar,
    supported_dtypes=_DEFAULT_UNARY_DTYPES | _TRIVIAL_UNARY_EXTRA,
)
