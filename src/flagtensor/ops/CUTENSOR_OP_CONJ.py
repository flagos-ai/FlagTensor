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
from flagtensor.utils import libtuner
from flagtensor.runtime import is_on_accelerator as _is_on_accelerator


@libtuner(
    configs=runtime.get_tuned_config("elementwise_unary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_unary"))
@triton.jit
def _conj_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    if KERNEL_ID == 0:
        offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
        mask = offsets < n_elements
        real = tl.load(x_ptr + 2 * offsets, mask=mask)
        imag = tl.load(x_ptr + 2 * offsets + 1, mask=mask)
        tl.store(y_ptr + 2 * offsets, real, mask=mask)
        tl.store(y_ptr + 2 * offsets + 1, -imag, mask=mask)
    else:
        for block_idx in tl.static_range(0, BLOCKS_PER_PROGRAM):
            offsets = block_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            real = tl.load(x_ptr + 2 * offsets, mask=mask)
            imag = tl.load(x_ptr + 2 * offsets + 1, mask=mask)
            tl.store(y_ptr + 2 * offsets, real, mask=mask)
            tl.store(y_ptr + 2 * offsets + 1, 0 - imag, mask=mask)


def conj(x: torch.Tensor) -> torch.Tensor:
    if not _is_on_accelerator(x):
        raise ValueError("input tensor must be on CUDA")
    if not x.is_complex():
        return x.clone()
    y = torch.empty_like(x)
    if not x.is_contiguous():
        x = x.contiguous()
    real_dtype = torch.float32 if x.dtype == torch.complex64 else torch.float64
    x_view = x.view(real_dtype)
    y_view = y.view(real_dtype)
    n_elements = x.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _conj_kernel[grid](x_view, y_view, n_elements)
    return y
