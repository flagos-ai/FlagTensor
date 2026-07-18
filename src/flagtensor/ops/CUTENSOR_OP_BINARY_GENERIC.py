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
from flagtensor.cutensor import BINARY_OPERATOR_MAP, _normalize_modes, _resolve_operator
from flagtensor.ops.elementwise_common import (
    align_tensors_to_output,
    prepare_indexed_launch_tensors,
    infer_elementwise_output_modes,
    infer_elementwise_output_shape,
    allocate_output_tensor,
)
from flagtensor.utils import libtuner


_SUPPORTED_TRITON_BINARY_OPS = {
    BINARY_OPERATOR_MAP["add"],
    BINARY_OPERATOR_MAP["mul"],
    BINARY_OPERATOR_MAP["max"],
    BINARY_OPERATOR_MAP["min"],
}

_BINARY_EXECUTOR_CACHE = {}


@libtuner(
    configs=runtime.get_tuned_config("elementwise_binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_binary"))
@triton.jit
def _binary_generic_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if OP == 3:
        out = x + y
    elif OP == 5:
        out = x * y
    elif OP == 6:
        out = tl.maximum(x, y)
    else:
        out = tl.minimum(x, y)
    tl.store(out_ptr + offsets, out, mask=mask)


@libtuner(
    configs=runtime.get_tuned_config("elementwise_binary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_binary"))
@triton.jit
def _binary_generic_indexed_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    out_dim0,
    out_dim1,
    out_dim2,
    out_dim3,
    x_axis0: tl.constexpr,
    x_axis1: tl.constexpr,
    x_axis2: tl.constexpr,
    x_axis3: tl.constexpr,
    y_axis0: tl.constexpr,
    y_axis1: tl.constexpr,
    y_axis2: tl.constexpr,
    y_axis3: tl.constexpr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    x_stride2: tl.constexpr,
    x_stride3: tl.constexpr,
    y_stride0: tl.constexpr,
    y_stride1: tl.constexpr,
    y_stride2: tl.constexpr,
    y_stride3: tl.constexpr,
    x_shape0: tl.constexpr,
    x_shape1: tl.constexpr,
    x_shape2: tl.constexpr,
    x_shape3: tl.constexpr,
    y_shape0: tl.constexpr,
    y_shape1: tl.constexpr,
    y_shape2: tl.constexpr,
    y_shape3: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements

    dim3 = offsets % out_dim3
    tmp = offsets // out_dim3
    dim2 = tmp % out_dim2
    tmp = tmp // out_dim2
    dim1 = tmp % out_dim1
    dim0 = tmp // out_dim1

    x_offset = (
        (0 if x_axis0 == -1 or x_shape0 == 1 else dim0) * x_stride0
        + (0 if x_axis1 == -1 or x_shape1 == 1 else dim1) * x_stride1
        + (0 if x_axis2 == -1 or x_shape2 == 1 else dim2) * x_stride2
        + (0 if x_axis3 == -1 or x_shape3 == 1 else dim3) * x_stride3
    )
    y_offset = (
        (0 if y_axis0 == -1 or y_shape0 == 1 else dim0) * y_stride0
        + (0 if y_axis1 == -1 or y_shape1 == 1 else dim1) * y_stride1
        + (0 if y_axis2 == -1 or y_shape2 == 1 else dim2) * y_stride2
        + (0 if y_axis3 == -1 or y_shape3 == 1 else dim3) * y_stride3
    )

    x = tl.load(x_ptr + x_offset, mask=mask)
    y = tl.load(y_ptr + y_offset, mask=mask)
    if OP == 3:
        out = x + y
    elif OP == 5:
        out = x * y
    elif OP == 6:
        out = tl.maximum(x, y)
    else:
        out = tl.minimum(x, y)
    tl.store(out_ptr + offsets, out, mask=mask)


def _supports_triton_binary(op):
    return op in _SUPPORTED_TRITON_BINARY_OPS


def _supports_direct_binary(x, y, mode_x, mode_y, mode_out, op_code):
    if not _supports_triton_binary(op_code):
        return False
    if not (x.is_contiguous() and y.is_contiguous() and x.shape == y.shape):
        return False
    canonical = tuple(range(x.ndim))
    norm_x = _normalize_modes(mode_x, x.ndim)
    norm_y = _normalize_modes(mode_y, y.ndim)
    norm_out = infer_elementwise_output_modes((tuple(norm_x), tuple(norm_y)), mode_out)
    return tuple(norm_x) == canonical and tuple(norm_y) == canonical and tuple(norm_out) == canonical


def _launch_contiguous_binary(x, y, output, op_code):
    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _binary_generic_kernel[grid](x, y, output, n_elements, OP=op_code)
    return output


def _launch_indexed_binary(x, y, output, output_shape, mode_x, mode_y, mode_out, op_code):
    launch_data = prepare_indexed_launch_tensors(output_shape, ((x, mode_x), (y, mode_y)), mode_out)
    if launch_data is None:
        return None
    padded_shape, metadata = launch_data
    (x_axes, x_strides, x_shapes), (y_axes, y_strides, y_shapes) = metadata

    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _binary_generic_indexed_kernel[grid](
        x,
        y,
        output,
        n_elements,
        padded_shape[0],
        padded_shape[1],
        padded_shape[2],
        padded_shape[3],
        x_axis0=x_axes[0],
        x_axis1=x_axes[1],
        x_axis2=x_axes[2],
        x_axis3=x_axes[3],
        y_axis0=y_axes[0],
        y_axis1=y_axes[1],
        y_axis2=y_axes[2],
        y_axis3=y_axes[3],
        x_stride0=x_strides[0],
        x_stride1=x_strides[1],
        x_stride2=x_strides[2],
        x_stride3=x_strides[3],
        y_stride0=y_strides[0],
        y_stride1=y_strides[1],
        y_stride2=y_strides[2],
        y_stride3=y_strides[3],
        x_shape0=x_shapes[0],
        x_shape1=x_shapes[1],
        x_shape2=x_shapes[2],
        x_shape3=x_shapes[3],
        y_shape0=y_shapes[0],
        y_shape1=y_shapes[1],
        y_shape2=y_shapes[2],
        y_shape3=y_shapes[3],
        OP=op_code,
    )
    return output


def _try_indexed_binary(context, op_code):
    (mode_x, mode_y) = context["mode_groups"]
    return _launch_indexed_binary(
        context["tensors_with_modes"][0][0],
        context["tensors_with_modes"][1][0],
        context["output"],
        context["output_shape"],
        mode_x,
        mode_y,
        context["output_modes"],
        op_code,
    )


def _execute_binary_aligned(context, aligned_tensors, op_code):
    x_aligned, y_aligned = aligned_tensors
    (mode_x, mode_y) = context["mode_groups"]
    mode_out = context["output_modes"]
    output_shape = context["output_shape"]
    output = context["output"]
    x = context["tensors_with_modes"][0][0]
    y = context["tensors_with_modes"][1][0]

    if tuple(mode_x) == tuple(mode_out) and tuple(mode_y) == tuple(mode_out) and tuple(x.shape) == tuple(output_shape) and tuple(y.shape) == tuple(output_shape) and x.is_contiguous() and y.is_contiguous() and output.is_contiguous():
        return _launch_contiguous_binary(x, y, output, op_code)
    return _launch_contiguous_binary(x_aligned, y_aligned, output, op_code)


class _TritonBinaryExecutor:
    def __init__(self, op):
        self.op_code = _resolve_operator(op, BINARY_OPERATOR_MAP, "binary")
        if not _supports_triton_binary(self.op_code):
            raise ValueError(f"unsupported binary operator: {op}")
        self.layout_cache = {}

    def _layout_key(self, x, y, mode_x, mode_y, mode_out):
        return (
            x.dtype,
            tuple(x.shape),
            x.stride(),
            mode_x if mode_x is None else tuple(mode_x),
            tuple(y.shape),
            y.stride(),
            mode_y if mode_y is None else tuple(mode_y),
            mode_out if mode_out is None else tuple(mode_out),
        )

    def _build_plan(self, x, y, mode_x, mode_y, mode_out):
        norm_x = _normalize_modes(mode_x, x.ndim)
        norm_y = _normalize_modes(mode_y, y.ndim)
        mode_groups = (tuple(norm_x), tuple(norm_y))
        output_modes = infer_elementwise_output_modes(mode_groups, mode_out)
        output_shape = infer_elementwise_output_shape(((x, mode_groups[0]), (y, mode_groups[1])), output_modes)

        if _supports_direct_binary(x, y, norm_x, norm_y, output_modes, self.op_code):
            return {
                "kind": "direct",
                "output_shape": output_shape,
            }

        launch_data = prepare_indexed_launch_tensors(output_shape, ((x, mode_groups[0]), (y, mode_groups[1])), output_modes)
        if launch_data is not None:
            padded_shape, metadata = launch_data
            return {
                "kind": "indexed",
                "output_shape": output_shape,
                "padded_shape": padded_shape,
                "x_meta": metadata[0],
                "y_meta": metadata[1],
            }

        return {
            "kind": "general",
            "mode_groups": mode_groups,
            "output_modes": output_modes,
            "output_shape": output_shape,
        }

    def __call__(self, x, y, *, mode_x=None, mode_y=None, mode_out=None, out=None):
        if not x.is_cuda or not y.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if x.dtype != y.dtype:
            raise TypeError("input tensors must have the same dtype")

        layout_key = self._layout_key(x, y, mode_x, mode_y, mode_out)
        plan = self.layout_cache.get(layout_key)
        if plan is None:
            plan = self._build_plan(x, y, mode_x, mode_y, mode_out)
            self.layout_cache[layout_key] = plan

        output = allocate_output_tensor(plan["output_shape"], out=out, device=x.device, dtype=x.dtype)
        if plan["kind"] == "direct":
            return _launch_contiguous_binary(x, y, output, self.op_code)
        if plan["kind"] == "indexed":
            padded_shape = plan["padded_shape"]
            x_axes, x_strides, x_shapes = plan["x_meta"]
            y_axes, y_strides, y_shapes = plan["y_meta"]
            n_elements = output.numel()
            grid = lambda meta: (
                triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
            )
            _binary_generic_indexed_kernel[grid](
                x,
                y,
                output,
                n_elements,
                padded_shape[0],
                padded_shape[1],
                padded_shape[2],
                padded_shape[3],
                x_axis0=x_axes[0],
                x_axis1=x_axes[1],
                x_axis2=x_axes[2],
                x_axis3=x_axes[3],
                y_axis0=y_axes[0],
                y_axis1=y_axes[1],
                y_axis2=y_axes[2],
                y_axis3=y_axes[3],
                x_stride0=x_strides[0],
                x_stride1=x_strides[1],
                x_stride2=x_strides[2],
                x_stride3=x_strides[3],
                y_stride0=y_strides[0],
                y_stride1=y_strides[1],
                y_stride2=y_strides[2],
                y_stride3=y_strides[3],
                x_shape0=x_shapes[0],
                x_shape1=x_shapes[1],
                x_shape2=x_shapes[2],
                x_shape3=x_shapes[3],
                y_shape0=y_shapes[0],
                y_shape1=y_shapes[1],
                y_shape2=y_shapes[2],
                y_shape3=y_shapes[3],
                OP=self.op_code,
            )
            return output

        aligned = align_tensors_to_output(
            ((x, plan["mode_groups"][0]), (y, plan["mode_groups"][1])),
            plan["output_modes"],
            plan["output_shape"],
        )
        return _launch_contiguous_binary(aligned[0], aligned[1], output, self.op_code)


def _get_triton_binary_executor(op):
    key = _resolve_operator(op, BINARY_OPERATOR_MAP, "binary")
    executor = _BINARY_EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = _TritonBinaryExecutor(op)
        _BINARY_EXECUTOR_CACHE[key] = executor
    return executor


def binary_generic(x: torch.Tensor, y: torch.Tensor, *, op="add", mode_x=None, mode_y=None, mode_out=None, out=None):
    executor = _get_triton_binary_executor(op)
    return executor(x, y, mode_x=mode_x, mode_y=mode_y, mode_out=mode_out, out=out)


__all__ = ["binary_generic"]
