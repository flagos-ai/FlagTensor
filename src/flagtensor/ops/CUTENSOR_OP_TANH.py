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


@triton.jit
def _tanh_scalar(x):
    exp_neg_twice = tl.exp(-2 * x)
    return (1 - exp_neg_twice) / (1 + exp_neg_twice)


_tanh_kernel, tanh = make_unary_pointwise_from_family(
    "tanh",
    "tanh_like",
    _tanh_scalar,
)
