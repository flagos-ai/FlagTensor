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
from flagtensor.runtime import is_ascend as _is_ascend


def _build_conj_kernel():
    @libtuner(
        configs=runtime.get_tuned_config("elementwise_unary"),
        key=["n_elements"],
        strategy=["align32"],
        warmup=5,
        rep=10,
    )
    @triton.heuristics(runtime.get_heuristic_config("elementwise_conj"))
    @triton.jit
    def _conj_kernel(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_PROGRAM: tl.constexpr,
        KERNEL_ID: tl.constexpr,
    ):
        # Work on the *flat* interleaved real/imag buffer: a stride-2 access
        # pattern (``2*offset`` / ``2*offset + 1``) is dramatically slower than a
        # contiguous one on XPU, so load a contiguous run of 2*BLOCK floats and
        # negate the odd (imaginary) lanes.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
        n_lanes = n_elements * 2
        if KERNEL_ID == 0:
            offsets = 2 * block_start + tl.arange(0, 2 * BLOCK_SIZE * BLOCKS_PER_PROGRAM)
            mask = offsets < n_lanes
            value = tl.load(x_ptr + offsets, mask=mask)
            y = tl.where((offsets % 2) == 0, value, -value)
            tl.store(y_ptr + offsets, y, mask=mask)
        else:
            for block_idx in tl.static_range(0, BLOCKS_PER_PROGRAM):
                base = 2 * (block_start + block_idx * BLOCK_SIZE)
                offsets = base + tl.arange(0, 2 * BLOCK_SIZE)
                mask = offsets < n_lanes
                value = tl.load(x_ptr + offsets, mask=mask)
                y = tl.where((offsets % 2) == 0, value, -value)
                tl.store(y_ptr + offsets, y, mask=mask)

    return _conj_kernel


# On Ascend ``torch.conj`` is a metadata-only view (it flips the tensor's
# conjugate bit; no kernel runs), which is also the documented PyTorch
# semantics.  Materialising the result costs a full read+write of the tensor
# and can therefore never be competitive, so the Ascend path returns the view
# as well and builds no kernel (fp64, which complex128 would need, is
# unsupported there too).
if _is_ascend():
    _conj_kernel = None
else:
    _conj_kernel = _build_conj_kernel()


def conj(x: torch.Tensor) -> torch.Tensor:
    if not _is_on_accelerator(x):
        raise ValueError("input tensor must be CUDA")
    if not x.is_complex():
        return x.clone()
    if _is_ascend():
        # Lazy conjugation: identical numerics, no memory traffic.  Consumers
        # that need a materialised tensor call ``resolve_conj()`` themselves.
        return x.conj()
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
