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

import pytest
import torch

from flagtensor import BlockSparseTensor
from flagtensor import BlockSparseTensorContraction
from flagtensor import BlockSparseTensorDescriptor
from flagtensor import block_sparse_contraction
from flagtensor.config import DEFAULT_BLOCK_SPARSE_TENSOR_CONTRACTION_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
from flagtensor.testing import assert_close
from tests._legacy_correctness_loader import populate_category_proxy

# On Ascend, BlockSparseTensorContraction (from flagtensor.cutensor) already
# falls back to a dense contraction when cuTensor is unavailable, so we can
# re-use it as the vendor baseline there too.
_BASELINE_AVAILABLE = True


def _make_block_sparse(shape, block_shape, dtype, device):
    block_h, block_w = block_shape
    blocks = {}
    num_block_rows = shape[0] // block_h
    num_block_cols = shape[1] // block_w
    for i in range(num_block_rows):
        for j in range(num_block_cols):
            if (i + j) % 2 == 0:
                blocks[(i, j)] = torch.empty(block_shape, device=device, dtype=dtype).uniform_(-2.0, 2.0)
    desc = BlockSparseTensorDescriptor(
        shape=shape,
        block_shape=block_shape,
        nonzero_coordinates=tuple(sorted(blocks.keys())),
    )
    return BlockSparseTensor(desc, blocks)


def _make_block_sparse_from_sections(shape, section_extents, coords, dtype, device):
    desc = BlockSparseTensorDescriptor(
        shape=shape,
        num_sections_per_mode=tuple(len(mode_extents) for mode_extents in section_extents),
        section_extents=section_extents,
        nonzero_coordinates=coords,
    )
    blocks = {}
    for coord in coords:
        block_shape = tuple(section_extents[mode][index] for mode, index in enumerate(coord))
        real = torch.empty(block_shape, device=device, dtype=torch.float32).uniform_(-2.0, 2.0)
        if dtype in (torch.complex64, torch.complex128):
            imag = torch.empty(block_shape, device=device, dtype=torch.float32).uniform_(-2.0, 2.0)
            block = torch.complex(real, imag).to(dtype)
        else:
            block = real.to(dtype)
        blocks[coord] = block
    return BlockSparseTensor(desc, blocks)


def _output_mask(tensor: BlockSparseTensor):
    mask = torch.zeros(tensor.shape, device=tensor.device, dtype=torch.float32)
    offsets_per_mode = []
    for mode_extents in tensor.descriptor.section_extents:
        offsets = [0]
        for extent in mode_extents:
            offsets.append(offsets[-1] + extent)
        offsets_per_mode.append(offsets)
    for coord in tensor.descriptor.canonical_nonzero_coordinates:
        slices = tuple(
            slice(offsets_per_mode[mode][index], offsets_per_mode[mode][index + 1])
            for mode, index in enumerate(coord)
        )
        mask[slices] = 1.0
    return mask


def _block_sparse_reference(a, b, c, alpha=1.25, beta=0.5):
    out = alpha * torch.matmul(a.to_dense(), b.to_dense()) + beta * c.to_dense()
    mask = _output_mask(c)
    if out.dtype.is_complex:
        mask = mask.to(out.real.dtype)
    out = out * mask.to(out.dtype)
    return out


@pytest.mark.BlockSparseContraction
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_BLOCK_SPARSE_TENSOR_CONTRACTION_TEST_SHAPES)
def test_block_sparse_contraction_correctness(dtype, shape_a, shape_b):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    block_k = 4 if shape_a[1] % 4 == 0 and shape_b[0] % 4 == 0 else shape_a[1]
    block_shape_a = (shape_a[0] // 2, block_k)
    block_shape_b = (block_k, shape_b[1] // 2)
    a = _make_block_sparse(shape_a, block_shape_a, dtype, _device_str)
    b = _make_block_sparse(shape_b, block_shape_b, dtype, _device_str)
    c_coords = ((0, 0), (1, 1))
    c_desc = BlockSparseTensorDescriptor(
        shape=(shape_a[0], shape_b[1]),
        block_shape=(block_shape_a[0], block_shape_b[1]),
        nonzero_coordinates=c_coords,
    )
    c = BlockSparseTensor(
        c_desc,
        {
            (0, 0): torch.empty(c_desc.block_shape, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0),
            (1, 1): torch.empty(c_desc.block_shape, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0),
        },
    )

    out = block_sparse_contraction(a, b, c=c, alpha=1.25, beta=0.5)
    expected = _block_sparse_reference(a, b, c, alpha=1.25, beta=0.5).to(dtype)
    assert_close(out, expected, dtype)

    baseline = BlockSparseTensorContraction()
    out_base = baseline(a, b, c=c, alpha=1.25, beta=0.5)
    assert_close(out_base, expected, dtype)
    assert_close(out, out_base, dtype)


populate_category_proxy(globals(), "sparse", skipped_names=("block_sparse_contraction",))
