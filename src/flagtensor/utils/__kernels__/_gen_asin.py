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

import triton
import triton.language as tl
from flagtensor import runtime
from flagtensor.utils.libtuner import libtuner

@libtuner(
    configs=runtime.get_tuned_config("elementwise_unary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_unary"))
@triton.jit
def _asin_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if KERNEL_ID == 0:
        y = _variant0(x)
    else:
        y = _variant1(x)
    tl.store(y_ptr + offsets, y, mask=mask)

result = _asin_kernel
result.__name__ = "_asin_kernel"
