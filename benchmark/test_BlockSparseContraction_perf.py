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

import os

import pytest
import torch

from flagtensor import BlockSparseTensor
from flagtensor import BlockSparseTensorContraction
from flagtensor import BlockSparseTensorDescriptor
from flagtensor import block_sparse_contraction
from flagtensor.benchmark_core import Benchmark, BenchmarkConfig, get_baseline_class, vendor_baseline_available
from flagtensor.config import DEFAULT_BENCHMARK_DTYPES, DEFAULT_BLOCK_SPARSE_TENSOR_CONTRACTION_BENCHMARK_SHAPES
from flagtensor.ops.CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION import _build_block_contraction_plan
from flagtensor.ops.CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION import _get_section_extents_for_coord
from flagtensor.ops.CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION import _launch_block_sparse_gemm
from flagtensor.visualization import plot_latency_and_speedup, write_benchmark_csv

OP_NAME = "CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION"
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
RESULTS_DIR = os.path.join(RESULTS_ROOT, OP_NAME)
CSV_PATH = os.path.join(RESULTS_DIR, "benchmark.csv")


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


def _sparsity_mask(tensor: BlockSparseTensor):
    mask = torch.zeros(tensor.shape, device=tensor.device, dtype=torch.float32)
    row_offsets = [0]
    for extent in tensor.descriptor.section_extents[0]:
        row_offsets.append(row_offsets[-1] + extent)
    col_offsets = [0]
    for extent in tensor.descriptor.section_extents[1]:
        col_offsets.append(col_offsets[-1] + extent)
    for block_row, block_col in tensor.descriptor.canonical_nonzero_coordinates:
        row_start = row_offsets[block_row]
        row_end = row_offsets[block_row + 1]
        col_start = col_offsets[block_col]
        col_end = col_offsets[block_col + 1]
        mask[row_start:row_end, col_start:col_end] = 1.0
    return mask


class BlockSparseTensorContractionBenchmark(Benchmark):
    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            config=BenchmarkConfig(
                dtypes=(torch.float32,),
                shapes=tuple(DEFAULT_BLOCK_SPARSE_TENSOR_CONTRACTION_BENCHMARK_SHAPES),
                mode="operator",
            ),
        )
        self.baseline = BlockSparseTensorContraction()

    def get_input_iter(self, dtype: torch.dtype):
        for shape_a, shape_b in self.config.shapes:
            block_k = 4 if shape_a[1] % 4 == 0 and shape_b[0] % 4 == 0 else shape_a[1]
            block_shape_a = (shape_a[0] // 2, block_k)
            block_shape_b = (block_k, shape_b[1] // 2)
            a = _make_block_sparse(shape_a, block_shape_a, dtype, self.device)
            b = _make_block_sparse(shape_b, block_shape_b, dtype, self.device)
            c_desc = BlockSparseTensorDescriptor(
                shape=(shape_a[0], shape_b[1]),
                block_shape=(block_shape_a[0], block_shape_b[1]),
                nonzero_coordinates=((0, 0),),
            )
            c = BlockSparseTensor(
                c_desc,
                {(0, 0): torch.empty(c_desc.block_shape, device=self.device, dtype=dtype).uniform_(-2.0, 2.0)},
            )
            yield a, b, c

    def baseline_impl(self, a, b, c):
        result = self.baseline(a, b, c=c, alpha=1.25, beta=0.5)
        # Convert BlockSparseTensor to dense if needed
        if hasattr(result, 'to_dense'):
            return result.to_dense()
        return result

    def triton_impl(self, a, b, c):
        result = block_sparse_contraction(a, b, c=c, alpha=1.25, beta=0.5)
        # Result is already dense from Triton path, but handle BlockSparseTensor just in case
        if hasattr(result, 'to_dense'):
            return result.to_dense()
        return result

    def reference_impl(self, a, b, c):
        out = 1.25 * torch.matmul(a.to_dense().float(), b.to_dense().float()) + 0.5 * c.to_dense().float()
        out = out * _sparsity_mask(c)
        return out.to(a.dtype)

    def build_triton_kernel_callable(self, *args):
        a, b, c = args
        # Only support default 2D block-sparse case with specific mode
        if a.ndim != 2 or b.ndim != 2:
            return None
        if a.dtype not in (torch.float16, torch.float32):
            return None

        # Build contraction plan
        plan = _build_block_contraction_plan(a.descriptor, b.descriptor, (0, 1), (1, 2), (0, 2))
        if plan is None:
            return None

        # Pre-allocate output blocks
        out_blocks = {}
        for out_coord in plan.keys():
            out_block_shape = _get_section_extents_for_coord(a.descriptor, (out_coord[0], 0))
            out_block_shape = (out_block_shape[0], _get_section_extents_for_coord(b.descriptor, (0, out_coord[1]))[1])
            out_blocks[out_coord] = torch.empty(out_block_shape, device=a.device, dtype=a.dtype)

        # Pre-allocate intermediate buffers for multi-pair accumulations
        temp_buffers = {}
        for out_coord, pairs in plan.items():
            if len(pairs) > 1:
                temp_buffers[out_coord] = torch.empty_like(out_blocks[out_coord])

        # Pre-build output dense tensor and slice mapping
        output_shape = (a.shape[0], b.shape[1])
        out_dense = torch.empty(output_shape, device=a.device, dtype=a.dtype)

        # Calculate block offsets for dense tensor slicing
        row_offsets = [0]
        for extent in a.descriptor.section_extents[0]:
            row_offsets.append(row_offsets[-1] + extent)
        col_offsets = [0]
        for extent in b.descriptor.section_extents[1]:
            col_offsets.append(col_offsets[-1] + extent)

        def run_kernel():
            for out_coord, pairs in plan.items():
                out_block = out_blocks[out_coord]

                # Initialize with beta * C
                addend_block = None
                if c is not None and out_coord in c.blocks:
                    addend_block = c.blocks[out_coord]
                    out_block.copy_(addend_block * 0.5)
                else:
                    out_block.zero_()

                # Accumulate contributions
                for i, (a_coord, b_coord) in enumerate(pairs):
                    a_block = a.blocks[a_coord]
                    b_block = b.blocks[b_coord]

                    # Use temp buffer for accumulation, or direct to output for single pair
                    if len(pairs) > 1:
                        temp = temp_buffers[out_coord]
                        _launch_block_sparse_gemm(a_block, b_block, None, temp, 1.25, 0.0)
                        out_block.add_(temp)
                    else:
                        _launch_block_sparse_gemm(a_block, b_block, addend_block, out_block, 1.25, 0.5 if addend_block is not None else 0.0)

                # Write result to dense output tensor
                r, c_idx = out_coord
                row_start, row_end = row_offsets[r], row_offsets[r + 1]
                col_start, col_end = col_offsets[c_idx], col_offsets[c_idx + 1]
                out_dense[row_start:row_end, col_start:col_end] = out_block

            return out_dense

        return run_kernel

    def build_baseline_kernel_callable(self, *args):
        a, b, c = args
        # Only support default 2D block-sparse case with cuTensor
        if a.dtype not in (torch.float32, torch.complex64, torch.complex128):
            return None

        baseline = get_baseline_class("BlockSparseContraction")(dtype=a.dtype)

        def run_kernel():
            result = baseline(a, (0, 1), b, (1, 2), c, (0, 2), (0, 2), alpha=1.25, beta=0.5)
            # Convert BlockSparseTensor to dense
            if hasattr(result, 'to_dense'):
                return result.to_dense()
            return result

        return run_kernel


@pytest.mark.performance
@pytest.mark.BlockSparseContraction
def test_block_sparse_tensor_contraction_perf():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    bench = BlockSparseTensorContractionBenchmark()
    results = bench.run()
    write_benchmark_csv(results, CSV_PATH)
    plot_latency_and_speedup(results, RESULTS_DIR, OP_NAME)
    for result in results:
        print(
            f"shape={result.shape} dtype={result.dtype} "
            f"triton_ms={result.latency:.6f} baseline_ms={result.latency_base:.6f} "
            f"speedup={result.speedup:.3f}x"
        )
