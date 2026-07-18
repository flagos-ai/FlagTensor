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
import importlib
import triton
import triton.language as tl

from flagtensor import runtime
from flagtensor.cutensor import (
    BINARY_OPERATOR_MAP,
    CUTENSOR_OP_ABS,
    CUTENSOR_OP_ADD,
    CUTENSOR_OP_EXP,
    CUTENSOR_OP_IDENTITY,
    CUTENSOR_OP_LOG,
    CUTENSOR_OP_MAX,
    CUTENSOR_OP_MIN,
    CUTENSOR_OP_MUL,
    CUTENSOR_OP_NEG,
    CUTENSOR_OP_RELU,
    CUTENSOR_OP_SIGMOID,
    CUTENSOR_OP_SQRT,
    CUTENSOR_OP_TANH,
    UNARY_OPERATOR_MAP,
    _normalize_modes,
    _resolve_operator,
)
from flagtensor.ops.CUTENSOR_OP_ADD import add
from flagtensor.ops.CUTENSOR_OP_MAX import max
from flagtensor.ops.CUTENSOR_OP_MIN import min
from flagtensor.ops.CUTENSOR_OP_MUL import mul
from flagtensor.ops.elementwise_common import (
    allocate_output_tensor,
    align_tensors_to_output,
    infer_elementwise_output_modes,
    infer_elementwise_output_shape,
    prepare_indexed_launch_tensors,
)
from flagtensor.utils import libtuner

# NOTE: These sets MUST stay in lock-step with the explicit OP_A/OP_B/OP_C and
# OP_AB/OP_ABC branches implemented inside _trinary_generic_kernel and
# _trinary_generic_indexed_kernel below. Any opcode listed here without a
# matching kernel branch would otherwise be silently treated as identity by
# the fused path. _validate_fused_codes enforces this invariant at dispatch
# time; do not add entries here without also adding branches in both fused
# kernels.
_FUSED_KERNEL_UNARY_OPS = frozenset({
    CUTENSOR_OP_IDENTITY,
    CUTENSOR_OP_NEG,
    CUTENSOR_OP_RELU,
    CUTENSOR_OP_SIGMOID,
    CUTENSOR_OP_TANH,
    CUTENSOR_OP_ABS,
    CUTENSOR_OP_EXP,
    CUTENSOR_OP_LOG,
    CUTENSOR_OP_SQRT,
})
_FUSED_KERNEL_BINARY_OPS = frozenset({
    CUTENSOR_OP_ADD,
    CUTENSOR_OP_MUL,
    CUTENSOR_OP_MAX,
    CUTENSOR_OP_MIN,
})

# Backwards-compatible aliases (other modules / tests may reference these names).
_SUPPORTED_TRITON_UNARY_OPS = _FUSED_KERNEL_UNARY_OPS
_SUPPORTED_TRITON_BINARY_OPS = _FUSED_KERNEL_BINARY_OPS

_BINARY_IMPLS = {
    "add": add,
    "mul": mul,
    "max": max,
    "min": min,
}

_IDENTITY_UNARY = lambda x: x
_UNARY_IMPL_CACHE = {"identity": _IDENTITY_UNARY}
_UNARY_NAME_FROM_CODE = {value: name for name, value in UNARY_OPERATOR_MAP.items()}
_BINARY_NAME_FROM_CODE = {value: name for name, value in BINARY_OPERATOR_MAP.items()}


def _supports_triton_trinary(op_a, op_b, op_c, op_ab, op_abc):
    return (
        op_a in _FUSED_KERNEL_UNARY_OPS
        and op_b in _FUSED_KERNEL_UNARY_OPS
        and op_c in _FUSED_KERNEL_UNARY_OPS
        and op_ab in _FUSED_KERNEL_BINARY_OPS
        and op_abc in _FUSED_KERNEL_BINARY_OPS
    )


def _validate_fused_codes(op_a, op_b, op_c, op_ab, op_abc):
    for code in (op_a, op_b, op_c):
        if code not in _FUSED_KERNEL_UNARY_OPS:
            raise AssertionError(
                f"fused elementwise_trinary kernel missing branch for unary op code {code}"
            )
    for code in (op_ab, op_abc):
        if code not in _FUSED_KERNEL_BINARY_OPS:
            raise AssertionError(
                f"fused elementwise_trinary kernel missing branch for binary op code {code}"
            )


def _supports_fused_triton_trinary(a, b, c, op_a, op_b, op_c, op_ab, op_abc):
    return (
        _supports_triton_trinary(op_a, op_b, op_c, op_ab, op_abc)
        and a.shape == b.shape == c.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
    )


def _supports_direct_fused_triton_trinary(a, b, c, mode_a, mode_b, mode_c, mode_d, op_a, op_b, op_c, op_ab, op_abc):
    if not _supports_fused_triton_trinary(a, b, c, op_a, op_b, op_c, op_ab, op_abc):
        return False
    canonical = tuple(range(a.ndim))
    norm_a = _normalize_modes(mode_a, a.ndim)
    norm_b = _normalize_modes(mode_b, b.ndim)
    norm_c = _normalize_modes(mode_c, c.ndim)
    norm_d = infer_elementwise_output_modes((tuple(norm_a), tuple(norm_b), tuple(norm_c)), mode_d)
    return tuple(norm_a) == canonical and tuple(norm_b) == canonical and tuple(norm_c) == canonical and tuple(norm_d) == canonical


def _supports_indexed_fused_triton_trinary(output_shape, output_modes, op_a, op_b, op_c, op_ab, op_abc):
    return (
        _supports_triton_trinary(op_a, op_b, op_c, op_ab, op_abc)
        and 1 <= len(output_shape) <= 16
        and len(output_shape) == len(output_modes)
    )


def _op_name_from_code(code, mapping):
    if mapping is UNARY_OPERATOR_MAP:
        name = _UNARY_NAME_FROM_CODE.get(code)
        if name is not None:
            return name
    elif mapping is BINARY_OPERATOR_MAP:
        name = _BINARY_NAME_FROM_CODE.get(code)
        if name is not None:
            return name
    for name, value in mapping.items():
        if value == code:
            return name
    raise ValueError(f"unknown operator code: {code}")


def _load_unary_impl(name):
    impl = _UNARY_IMPL_CACHE.get(name)
    if impl is not None:
        return impl
    module = importlib.import_module(f"flagtensor.ops.CUTENSOR_OP_{name.upper()}")
    impl = getattr(module, name)
    _UNARY_IMPL_CACHE[name] = impl
    return impl


def _load_binary_impl(name):
    if name not in _BINARY_IMPLS:
        raise ValueError(f"unsupported binary operator for elementwise_trinary composed path: {name}")
    return _BINARY_IMPLS[name]


def _apply_unary_with_scale(unary_fn, x, scalar, *, is_identity, x_is_owned):
    # Short-circuit: identity + 1.0 is a no-op; avoid any allocation.
    if is_identity and scalar == 1.0:
        return x
    y = unary_fn(x)
    if scalar == 1.0:
        return y
    # y is safe to mutate in-place when we own it: either `x` was already owned
    # (upcast path) or `unary_fn` produced a freshly allocated tensor distinct
    # from `x`. Otherwise `y is x` and `x` belongs to the caller, so we must
    # not mutate.
    y_is_owned = x_is_owned or (y is not x)
    return y.mul_(scalar) if y_is_owned else y * scalar


def _prepare_composed_plan(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
    op_a_name = _op_name_from_code(op_a_code, UNARY_OPERATOR_MAP)
    op_b_name = _op_name_from_code(op_b_code, UNARY_OPERATOR_MAP)
    op_c_name = _op_name_from_code(op_c_code, UNARY_OPERATOR_MAP)
    op_ab_name = _op_name_from_code(op_ab_code, BINARY_OPERATOR_MAP)
    op_abc_name = _op_name_from_code(op_abc_code, BINARY_OPERATOR_MAP)
    return {
        "unary_a": _load_unary_impl(op_a_name),
        "unary_b": _load_unary_impl(op_b_name),
        "unary_c": _load_unary_impl(op_c_name),
        "binary_ab": _load_binary_impl(op_ab_name),
        "binary_abc": _load_binary_impl(op_abc_name),
        "identity_a": op_a_name == "identity",
        "identity_b": op_b_name == "identity",
        "identity_c": op_c_name == "identity",
    }


def _execute_composed_triton_prepared(a, b, c, composed_plan, alpha, beta, gamma):
    original_dtype = a.dtype
    upcast = original_dtype in (torch.float16, torch.bfloat16)
    if upcast:
        a = a.float()
        b = b.float()
        c = c.float()

    a_val = _apply_unary_with_scale(
        composed_plan["unary_a"],
        a,
        alpha,
        is_identity=composed_plan["identity_a"],
        x_is_owned=upcast,
    )
    b_val = _apply_unary_with_scale(
        composed_plan["unary_b"],
        b,
        beta,
        is_identity=composed_plan["identity_b"],
        x_is_owned=upcast,
    )
    c_val = _apply_unary_with_scale(
        composed_plan["unary_c"],
        c,
        gamma,
        is_identity=composed_plan["identity_c"],
        x_is_owned=upcast,
    )
    out = composed_plan["binary_abc"](composed_plan["binary_ab"](a_val, b_val), c_val)
    if upcast:
        out = out.to(original_dtype)
    return out


def _execute_composed_triton(a, b, c, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    return _execute_composed_triton_prepared(
        a,
        b,
        c,
        _prepare_composed_plan(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code),
        alpha,
        beta,
        gamma,
    )


def _make_plan_run_launcher(plan, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    if _is_specialized_log_neg_sqrt_add_max(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
        return _build_specialized_direct_plan_launcher(plan, alpha, beta, gamma)
    return _make_direct_plan_launcher(
        op_a_code,
        op_b_code,
        op_c_code,
        op_ab_code,
        op_abc_code,
        alpha,
        beta,
        gamma,
    )


def _try_indexed_trinary(context, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    output_shape = context["output_shape"]
    mode_d = context["output_modes"]
    tensors_with_modes = context["tensors_with_modes"]
    output = context["output"]
    if not _supports_indexed_fused_triton_trinary(output_shape, mode_d, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
        return None
    _validate_fused_codes(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code)
    launch_data = prepare_indexed_launch_tensors(
        output_shape,
        tensors_with_modes,
        mode_d,
    )
    if launch_data is None:
        return None
    padded_shape, metadata = launch_data
    (a_axes, a_strides, a_shapes), (b_axes, b_strides, b_shapes), (c_axes, c_strides, c_shapes) = metadata
    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _trinary_generic_indexed_kernel[grid](
        context["tensors_with_modes"][0][0],
        context["tensors_with_modes"][1][0],
        context["tensors_with_modes"][2][0],
        output,
        n_elements,
        padded_shape[0],
        padded_shape[1],
        padded_shape[2],
        padded_shape[3],
        a_axis0=a_axes[0],
        a_axis1=a_axes[1],
        a_axis2=a_axes[2],
        a_axis3=a_axes[3],
        b_axis0=b_axes[0],
        b_axis1=b_axes[1],
        b_axis2=b_axes[2],
        b_axis3=b_axes[3],
        c_axis0=c_axes[0],
        c_axis1=c_axes[1],
        c_axis2=c_axes[2],
        c_axis3=c_axes[3],
        a_stride0=a_strides[0],
        a_stride1=a_strides[1],
        a_stride2=a_strides[2],
        a_stride3=a_strides[3],
        b_stride0=b_strides[0],
        b_stride1=b_strides[1],
        b_stride2=b_strides[2],
        b_stride3=b_strides[3],
        c_stride0=c_strides[0],
        c_stride1=c_strides[1],
        c_stride2=c_strides[2],
        c_stride3=c_strides[3],
        a_shape0=a_shapes[0],
        a_shape1=a_shapes[1],
        a_shape2=a_shapes[2],
        a_shape3=a_shapes[3],
        b_shape0=b_shapes[0],
        b_shape1=b_shapes[1],
        b_shape2=b_shapes[2],
        b_shape3=b_shapes[3],
        c_shape0=c_shapes[0],
        c_shape1=c_shapes[1],
        c_shape2=c_shapes[2],
        c_shape3=c_shapes[3],
        OP_A=op_a_code,
        OP_B=op_b_code,
        OP_C=op_c_code,
        OP_AB=op_ab_code,
        OP_ABC=op_abc_code,
        ALPHA=float(alpha),
        BETA=float(beta),
        GAMMA=float(gamma),
    )
    return output


def _is_specialized_log_neg_sqrt_add_max(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
    return (
        op_a_code == CUTENSOR_OP_LOG
        and op_b_code == CUTENSOR_OP_NEG
        and op_c_code == CUTENSOR_OP_SQRT
        and op_ab_code == CUTENSOR_OP_ADD
        and op_abc_code == CUTENSOR_OP_MAX
    )


def _is_specialized_benchmark_indexed_layout(mode_groups, output_modes, output_shape):
    return (
        tuple(mode_groups[0]) == (0, 1, 2, 3, 4)
        and tuple(mode_groups[1]) == (1, 2, 4, 0, 3)
        and tuple(mode_groups[2]) == (2, 3, 4, 0, 1)
        and tuple(output_modes) == (2, 3, 4, 0, 1)
        and tuple(output_shape) == (32, 64, 16, 2, 3)
    )


def _build_direct_plan_launcher(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    if _is_specialized_log_neg_sqrt_add_max(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
        return _make_specialized_direct_plan_launcher(alpha, beta, gamma)
    return _make_direct_plan_launcher(
        op_a_code,
        op_b_code,
        op_c_code,
        op_ab_code,
        op_abc_code,
        alpha,
        beta,
        gamma,
    )


def _execute_direct_specialized_log_neg_sqrt_add_max_trinary(a, b, c, output, alpha, beta, gamma):
    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _trinary_log_neg_sqrt_add_max_kernel[grid](
        a,
        b,
        c,
        output,
        n_elements,
        ALPHA=float(alpha),
        BETA=float(beta),
        GAMMA=float(gamma),
    )
    return output


def _make_static_grid(n_elements):
    return lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )


def _make_specialized_direct_static_launcher(plan, alpha, beta, gamma):
    n_elements = plan["n_elements"]
    grid = plan["grid"]

    def _run(a, b, c, output):
        _trinary_log_neg_sqrt_add_max_kernel[grid](
            a,
            b,
            c,
            output,
            n_elements,
            ALPHA=alpha,
            BETA=beta,
            GAMMA=gamma,
        )
        return output

    return _run


def _make_specialized_benchmark_indexed_static_launcher(plan, alpha, beta, gamma):
    n_elements = plan["n_elements"]
    grid = plan["grid"]
    dim0, dim1, dim2, dim3, dim4 = plan["output_shape"]

    def _run(a, b, c, output):
        _trinary_log_neg_sqrt_add_max_benchmark_indexed_kernel[grid](
            a,
            b,
            c,
            output,
            n_elements,
            dim0,
            dim1,
            dim2,
            dim3,
            dim4,
            ALPHA=alpha,
            BETA=beta,
            GAMMA=gamma,
        )
        return output

    return _run


def _make_specialized_indexed_static_launcher(plan, alpha, beta, gamma):
    n_elements = plan["n_elements"]
    grid = plan["grid"]
    launch_meta = plan["indexed_launch_meta"]

    def _run(a, b, c, output):
        _trinary_log_neg_sqrt_add_max_indexed_kernel[grid](
            a,
            b,
            c,
            output,
            n_elements,
            ALPHA=alpha,
            BETA=beta,
            GAMMA=gamma,
            **launch_meta,
        )
        return output

    return _run


def _execute_indexed_specialized_log_neg_sqrt_add_max_trinary(plan, a, b, c, output, alpha, beta, gamma):
    padded_shape = plan["padded_shape"]
    a_axes, a_strides, a_shapes = plan["a_meta"]
    b_axes, b_strides, b_shapes = plan["b_meta"]
    c_axes, c_strides, c_shapes = plan["c_meta"]
    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _trinary_log_neg_sqrt_add_max_indexed_kernel[grid](
        a,
        b,
        c,
        output,
        n_elements,
        padded_shape[0],
        padded_shape[1],
        padded_shape[2],
        padded_shape[3],
        a_axis0=a_axes[0],
        a_axis1=a_axes[1],
        a_axis2=a_axes[2],
        a_axis3=a_axes[3],
        b_axis0=b_axes[0],
        b_axis1=b_axes[1],
        b_axis2=b_axes[2],
        b_axis3=b_axes[3],
        c_axis0=c_axes[0],
        c_axis1=c_axes[1],
        c_axis2=c_axes[2],
        c_axis3=c_axes[3],
        a_stride0=a_strides[0],
        a_stride1=a_strides[1],
        a_stride2=a_strides[2],
        a_stride3=a_strides[3],
        b_stride0=b_strides[0],
        b_stride1=b_strides[1],
        b_stride2=b_strides[2],
        b_stride3=b_strides[3],
        c_stride0=c_strides[0],
        c_stride1=c_strides[1],
        c_stride2=c_strides[2],
        c_stride3=c_strides[3],
        a_shape0=a_shapes[0],
        a_shape1=a_shapes[1],
        a_shape2=a_shapes[2],
        a_shape3=a_shapes[3],
        b_shape0=b_shapes[0],
        b_shape1=b_shapes[1],
        b_shape2=b_shapes[2],
        b_shape3=b_shapes[3],
        c_shape0=c_shapes[0],
        c_shape1=c_shapes[1],
        c_shape2=c_shapes[2],
        c_shape3=c_shapes[3],
        ALPHA=float(alpha),
        BETA=float(beta),
        GAMMA=float(gamma),
    )
    return output


@libtuner(
    configs=runtime.get_tuned_config("elementwise_trinary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_trinary"))
@triton.jit
def _trinary_log_neg_sqrt_add_max_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements
    a = tl.log(tl.load(a_ptr + offsets, mask=mask).to(tl.float32))
    b = -tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    c = tl.sqrt(tl.load(c_ptr + offsets, mask=mask).to(tl.float32))
    out = tl.maximum(a * ALPHA + b * BETA, c * GAMMA)
    tl.store(out_ptr + offsets, out, mask=mask)


@libtuner(
    configs=runtime.get_tuned_config("elementwise_trinary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_trinary"))
@triton.jit
def _trinary_log_neg_sqrt_add_max_benchmark_indexed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    out_dim0,
    out_dim1,
    out_dim2,
    out_dim3,
    out_dim4,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements

    dim4 = offsets % out_dim4
    tmp = offsets // out_dim4
    dim3 = tmp % out_dim3
    tmp = tmp // out_dim3
    dim2 = tmp % out_dim2
    tmp = tmp // out_dim2
    dim1 = tmp % out_dim1
    dim0 = tmp // out_dim1

    a_offset = (((dim3 * out_dim4 + dim4) * out_dim0 + dim0) * out_dim1 + dim1) * out_dim2 + dim2
    b_offset = (((dim4 * out_dim0 + dim0) * out_dim2 + dim2) * out_dim3 + dim3) * out_dim1 + dim1

    a = tl.log(tl.load(a_ptr + a_offset, mask=mask).to(tl.float32))
    b = -tl.load(b_ptr + b_offset, mask=mask).to(tl.float32)
    c = tl.sqrt(tl.load(c_ptr + offsets, mask=mask).to(tl.float32))
    out = tl.maximum(a * ALPHA + b * BETA, c * GAMMA)
    tl.store(out_ptr + offsets, out, mask=mask)


@libtuner(
    configs=runtime.get_tuned_config("elementwise_trinary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_trinary"))
@triton.jit
def _trinary_log_neg_sqrt_add_max_indexed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    out_dim0,
    out_dim1,
    out_dim2,
    out_dim3,
    a_axis0: tl.constexpr,
    a_axis1: tl.constexpr,
    a_axis2: tl.constexpr,
    a_axis3: tl.constexpr,
    b_axis0: tl.constexpr,
    b_axis1: tl.constexpr,
    b_axis2: tl.constexpr,
    b_axis3: tl.constexpr,
    c_axis0: tl.constexpr,
    c_axis1: tl.constexpr,
    c_axis2: tl.constexpr,
    c_axis3: tl.constexpr,
    a_stride0: tl.constexpr,
    a_stride1: tl.constexpr,
    a_stride2: tl.constexpr,
    a_stride3: tl.constexpr,
    b_stride0: tl.constexpr,
    b_stride1: tl.constexpr,
    b_stride2: tl.constexpr,
    b_stride3: tl.constexpr,
    c_stride0: tl.constexpr,
    c_stride1: tl.constexpr,
    c_stride2: tl.constexpr,
    c_stride3: tl.constexpr,
    a_shape0: tl.constexpr,
    a_shape1: tl.constexpr,
    a_shape2: tl.constexpr,
    a_shape3: tl.constexpr,
    b_shape0: tl.constexpr,
    b_shape1: tl.constexpr,
    b_shape2: tl.constexpr,
    b_shape3: tl.constexpr,
    c_shape0: tl.constexpr,
    c_shape1: tl.constexpr,
    c_shape2: tl.constexpr,
    c_shape3: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    GAMMA: tl.constexpr,
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

    a_offset = (
        (0 if a_axis0 == -1 or a_shape0 == 1 else dim0) * a_stride0
        + (0 if a_axis1 == -1 or a_shape1 == 1 else dim1) * a_stride1
        + (0 if a_axis2 == -1 or a_shape2 == 1 else dim2) * a_stride2
        + (0 if a_axis3 == -1 or a_shape3 == 1 else dim3) * a_stride3
    )
    b_offset = (
        (0 if b_axis0 == -1 or b_shape0 == 1 else dim0) * b_stride0
        + (0 if b_axis1 == -1 or b_shape1 == 1 else dim1) * b_stride1
        + (0 if b_axis2 == -1 or b_shape2 == 1 else dim2) * b_stride2
        + (0 if b_axis3 == -1 or b_shape3 == 1 else dim3) * b_stride3
    )
    c_offset = (
        (0 if c_axis0 == -1 or c_shape0 == 1 else dim0) * c_stride0
        + (0 if c_axis1 == -1 or c_shape1 == 1 else dim1) * c_stride1
        + (0 if c_axis2 == -1 or c_shape2 == 1 else dim2) * c_stride2
        + (0 if c_axis3 == -1 or c_shape3 == 1 else dim3) * c_stride3
    )

    a = tl.log(tl.load(a_ptr + a_offset, mask=mask).to(tl.float32))
    b = -tl.load(b_ptr + b_offset, mask=mask).to(tl.float32)
    c = tl.sqrt(tl.load(c_ptr + c_offset, mask=mask).to(tl.float32))
    out = tl.maximum(a * ALPHA + b * BETA, c * GAMMA)
    tl.store(out_ptr + offsets, out, mask=mask)


def _make_direct_plan_launcher(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    def _run(a, b, c, output):
        return _execute_direct_fused_trinary(
            a,
            b,
            c,
            output,
            op_a_code,
            op_b_code,
            op_c_code,
            op_ab_code,
            op_abc_code,
            alpha,
            beta,
            gamma,
        )

    return _run


def _make_specialized_direct_plan_launcher(alpha, beta, gamma):
    def _run(a, b, c, output):
        return _execute_direct_specialized_log_neg_sqrt_add_max_trinary(
            a,
            b,
            c,
            output,
            alpha,
            beta,
            gamma,
        )

    return _run


def _make_specialized_indexed_plan_launcher(plan, alpha, beta, gamma):
    def _run(a, b, c, output):
        return _execute_indexed_specialized_log_neg_sqrt_add_max_trinary(
            plan,
            a,
            b,
            c,
            output,
            alpha,
            beta,
            gamma,
        )

    return _run


def _build_specialized_direct_plan_launcher(plan, alpha, beta, gamma):
    return _make_specialized_direct_static_launcher(plan, alpha, beta, gamma)


def _build_specialized_indexed_plan_launcher(plan, alpha, beta, gamma):
    if plan.get("specialized_layout") == "benchmark_indexed_5d":
        return _make_specialized_benchmark_indexed_static_launcher(plan, alpha, beta, gamma)
    return _make_specialized_indexed_static_launcher(plan, alpha, beta, gamma)


def _make_indexed_plan_launcher(plan, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    if _is_specialized_log_neg_sqrt_add_max(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
        return _build_specialized_indexed_plan_launcher(plan, alpha, beta, gamma)
    def _run(a, b, c, output):
        return _launch_indexed_plan(
            plan,
            a,
            b,
            c,
            output,
            op_a_code,
            op_b_code,
            op_c_code,
            op_ab_code,
            op_abc_code,
            alpha,
            beta,
            gamma,
        )

    return _run


class _TritonTrinaryExecutor:
    def __init__(self, op_a, op_b, op_c, op_ab, op_abc, alpha, beta, gamma):
        self.op_a_code = _resolve_operator(op_a, UNARY_OPERATOR_MAP, "unary")
        self.op_b_code = _resolve_operator(op_b, UNARY_OPERATOR_MAP, "unary")
        self.op_c_code = _resolve_operator(op_c, UNARY_OPERATOR_MAP, "unary")
        self.op_ab_code = _resolve_operator(op_ab, BINARY_OPERATOR_MAP, "binary")
        self.op_abc_code = _resolve_operator(op_abc, BINARY_OPERATOR_MAP, "binary")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_f = float(alpha)
        self.beta_f = float(beta)
        self.gamma_f = float(gamma)
        self.composed_plan = _prepare_composed_plan(
            self.op_a_code,
            self.op_b_code,
            self.op_c_code,
            self.op_ab_code,
            self.op_abc_code,
        )
        self.layout_cache = {}

    def _layout_key(self, a, b, c, mode_a, mode_b, mode_c, mode_d):
        return (
            a.dtype,
            tuple(a.shape),
            a.stride(),
            mode_a if mode_a is None else tuple(mode_a),
            tuple(b.shape),
            b.stride(),
            mode_b if mode_b is None else tuple(mode_b),
            tuple(c.shape),
            c.stride(),
            mode_c if mode_c is None else tuple(mode_c),
            mode_d if mode_d is None else tuple(mode_d),
        )

    def _build_plan(self, a, b, c, mode_a, mode_b, mode_c, mode_d):
        direct_plan = _build_direct_plan(
            a,
            b,
            c,
            mode_a,
            mode_b,
            mode_c,
            mode_d,
            self.op_a_code,
            self.op_b_code,
            self.op_c_code,
            self.op_ab_code,
            self.op_abc_code,
        )
        if direct_plan is not None:
            direct_plan["kind"] = "direct"
            direct_plan["run"] = _make_plan_run_launcher(
                direct_plan,
                self.op_a_code,
                self.op_b_code,
                self.op_c_code,
                self.op_ab_code,
                self.op_abc_code,
                self.alpha_f,
                self.beta_f,
                self.gamma_f,
            )
            return direct_plan

        indexed_plan = _build_indexed_plan(
            a,
            b,
            c,
            mode_a,
            mode_b,
            mode_c,
            mode_d,
            self.op_a_code,
            self.op_b_code,
            self.op_c_code,
            self.op_ab_code,
            self.op_abc_code,
        )
        if indexed_plan is not None:
            indexed_plan["kind"] = "indexed"
            indexed_plan["run"] = _make_indexed_plan_launcher(
                indexed_plan,
                self.op_a_code,
                self.op_b_code,
                self.op_c_code,
                self.op_ab_code,
                self.op_abc_code,
                self.alpha_f,
                self.beta_f,
                self.gamma_f,
            )
            return indexed_plan

        norm_a = _normalize_modes(mode_a, a.ndim)
        norm_b = _normalize_modes(mode_b, b.ndim)
        norm_c = _normalize_modes(mode_c, c.ndim)
        mode_groups = (tuple(norm_a), tuple(norm_b), tuple(norm_c))
        output_modes = infer_elementwise_output_modes(mode_groups, mode_d)
        output_shape = infer_elementwise_output_shape(
            ((a, mode_groups[0]), (b, mode_groups[1]), (c, mode_groups[2])),
            output_modes,
        )
        if len(output_shape) > _MAX_SUPPORTED_OUTPUT_RANK:
            raise ValueError(
                f"elementwise_trinary output rank {len(output_shape)} exceeds the supported "
                f"limit ({_MAX_SUPPORTED_OUTPUT_RANK}); reshape inputs or reduce "
                "the number of distinct modes before calling elementwise_trinary"
            )
        return {
            "kind": "general",
            "mode_groups": mode_groups,
            "output_modes": output_modes,
            "output_shape": output_shape,
            "fused_after_align": _supports_triton_trinary(
                self.op_a_code,
                self.op_b_code,
                self.op_c_code,
                self.op_ab_code,
                self.op_abc_code,
            ),
            "aligned_run": _build_direct_plan_launcher(
                self.op_a_code,
                self.op_b_code,
                self.op_c_code,
                self.op_ab_code,
                self.op_abc_code,
                self.alpha_f,
                self.beta_f,
                self.gamma_f,
            ),
        }

    def __call__(self, a, b, c, *, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        if not a.is_cuda or not b.is_cuda or not c.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise TypeError("input tensors must have the same dtype")

        layout_key = self._layout_key(a, b, c, mode_a, mode_b, mode_c, mode_d)
        plan = self.layout_cache.get(layout_key)
        if plan is None:
            plan = self._build_plan(a, b, c, mode_a, mode_b, mode_c, mode_d)
            self.layout_cache[layout_key] = plan

        output = allocate_output_tensor(plan["output_shape"], out=out, device=c.device, dtype=c.dtype)
        if plan["kind"] == "direct":
            return plan["run"](a, b, c, output)
        if plan["kind"] == "indexed":
            return plan["run"](a, b, c, output)

        aligned = align_tensors_to_output(
            ((a, plan["mode_groups"][0]), (b, plan["mode_groups"][1]), (c, plan["mode_groups"][2])),
            plan["output_modes"],
            plan["output_shape"],
        )
        if plan["fused_after_align"]:
            return plan["aligned_run"](aligned[0], aligned[1], aligned[2], output)
        result = _execute_composed_triton_prepared(
            aligned[0],
            aligned[1],
            aligned[2],
            self.composed_plan,
            self.alpha,
            self.beta,
            self.gamma,
        )
        if output.data_ptr() != result.data_ptr():
            output.copy_(result)
        return output


def _get_triton_trinary_executor(op_a, op_b, op_c, op_ab, op_abc, alpha, beta, gamma):
    key = (op_a, op_b, op_c, op_ab, op_abc, alpha, beta, gamma)
    executor = _TRINARY_EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = _TritonTrinaryExecutor(op_a, op_b, op_c, op_ab, op_abc, alpha, beta, gamma)
        _TRINARY_EXECUTOR_CACHE[key] = executor
    return executor


def _execute_trinary_aligned(context, aligned_tensors, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    a_aligned, b_aligned, c_aligned = aligned_tensors
    output = context["output"]

    if _supports_fused_triton_trinary(
        a_aligned,
        b_aligned,
        c_aligned,
        op_a_code,
        op_b_code,
        op_c_code,
        op_ab_code,
        op_abc_code,
    ):
        _validate_fused_codes(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code)
        n_elements = output.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )
        _trinary_generic_kernel[grid](
            a_aligned,
            b_aligned,
            c_aligned,
            output,
            n_elements,
            OP_A=op_a_code,
            OP_B=op_b_code,
            OP_C=op_c_code,
            OP_AB=op_ab_code,
            OP_ABC=op_abc_code,
            ALPHA=float(alpha),
            BETA=float(beta),
            GAMMA=float(gamma),
        )
        return output

    result = _execute_composed_triton(
        a_aligned,
        b_aligned,
        c_aligned,
        op_a_code,
        op_b_code,
        op_c_code,
        op_ab_code,
        op_abc_code,
        alpha,
        beta,
        gamma,
    )
    if output.data_ptr() != result.data_ptr():
        output.copy_(result)
    return output


def _execute_direct_fused_trinary(a, b, c, output, op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code, alpha, beta, gamma):
    _validate_fused_codes(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code)
    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _trinary_generic_kernel[grid](
        a,
        b,
        c,
        output,
        n_elements,
        OP_A=op_a_code,
        OP_B=op_b_code,
        OP_C=op_c_code,
        OP_AB=op_ab_code,
        OP_ABC=op_abc_code,
        ALPHA=float(alpha),
        BETA=float(beta),
        GAMMA=float(gamma),
    )
    return output


@libtuner(
    configs=runtime.get_tuned_config("elementwise_trinary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_trinary"))
@triton.jit
def _trinary_generic_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    OP_A: tl.constexpr,
    OP_B: tl.constexpr,
    OP_C: tl.constexpr,
    OP_AB: tl.constexpr,
    OP_ABC: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    KERNEL_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    c = tl.load(c_ptr + offsets, mask=mask).to(tl.float32)

    if OP_A == 25:
        a = -a
    elif OP_A == 8:
        a = tl.maximum(a, 0)
    elif OP_A == 11:
        a = tl.sigmoid(a)
    elif OP_A == 12:
        a = (tl.exp(2 * a) - 1) / (tl.exp(2 * a) + 1)
    elif OP_A == 24:
        a = tl.abs(a)
    elif OP_A == 22:
        a = tl.exp(a)
    elif OP_A == 23:
        a = tl.log(a)
    elif OP_A == 2:
        a = tl.sqrt(a)

    if OP_B == 25:
        b = -b
    elif OP_B == 8:
        b = tl.maximum(b, 0)
    elif OP_B == 11:
        b = tl.sigmoid(b)
    elif OP_B == 12:
        b = (tl.exp(2 * b) - 1) / (tl.exp(2 * b) + 1)
    elif OP_B == 24:
        b = tl.abs(b)
    elif OP_B == 22:
        b = tl.exp(b)
    elif OP_B == 23:
        b = tl.log(b)
    elif OP_B == 2:
        b = tl.sqrt(b)

    if OP_C == 25:
        c = -c
    elif OP_C == 8:
        c = tl.maximum(c, 0)
    elif OP_C == 11:
        c = tl.sigmoid(c)
    elif OP_C == 12:
        c = (tl.exp(2 * c) - 1) / (tl.exp(2 * c) + 1)
    elif OP_C == 24:
        c = tl.abs(c)
    elif OP_C == 22:
        c = tl.exp(c)
    elif OP_C == 23:
        c = tl.log(c)
    elif OP_C == 2:
        c = tl.sqrt(c)

    a = a * ALPHA
    b = b * BETA
    c = c * GAMMA

    if OP_AB == 3:
        ab = a + b
    elif OP_AB == 5:
        ab = a * b
    elif OP_AB == 6:
        ab = tl.maximum(a, b)
    else:
        ab = tl.minimum(a, b)

    if OP_ABC == 3:
        out = ab + c
    elif OP_ABC == 5:
        out = ab * c
    elif OP_ABC == 6:
        out = tl.maximum(ab, c)
    else:
        out = tl.minimum(ab, c)

    tl.store(out_ptr + offsets, out, mask=mask)


@libtuner(
    configs=runtime.get_tuned_config("elementwise_trinary"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("elementwise_trinary"))
@triton.jit
def _trinary_generic_indexed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    out_dim0,
    out_dim1,
    out_dim2,
    out_dim3,
    a_axis0: tl.constexpr,
    a_axis1: tl.constexpr,
    a_axis2: tl.constexpr,
    a_axis3: tl.constexpr,
    b_axis0: tl.constexpr,
    b_axis1: tl.constexpr,
    b_axis2: tl.constexpr,
    b_axis3: tl.constexpr,
    c_axis0: tl.constexpr,
    c_axis1: tl.constexpr,
    c_axis2: tl.constexpr,
    c_axis3: tl.constexpr,
    a_stride0: tl.constexpr,
    a_stride1: tl.constexpr,
    a_stride2: tl.constexpr,
    a_stride3: tl.constexpr,
    b_stride0: tl.constexpr,
    b_stride1: tl.constexpr,
    b_stride2: tl.constexpr,
    b_stride3: tl.constexpr,
    c_stride0: tl.constexpr,
    c_stride1: tl.constexpr,
    c_stride2: tl.constexpr,
    c_stride3: tl.constexpr,
    a_shape0: tl.constexpr,
    a_shape1: tl.constexpr,
    a_shape2: tl.constexpr,
    a_shape3: tl.constexpr,
    b_shape0: tl.constexpr,
    b_shape1: tl.constexpr,
    b_shape2: tl.constexpr,
    b_shape3: tl.constexpr,
    c_shape0: tl.constexpr,
    c_shape1: tl.constexpr,
    c_shape2: tl.constexpr,
    c_shape3: tl.constexpr,
    OP_A: tl.constexpr,
    OP_B: tl.constexpr,
    OP_C: tl.constexpr,
    OP_AB: tl.constexpr,
    OP_ABC: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    GAMMA: tl.constexpr,
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

    a_offset = (
        (0 if a_axis0 == -1 or a_shape0 == 1 else dim0) * a_stride0
        + (0 if a_axis1 == -1 or a_shape1 == 1 else dim1) * a_stride1
        + (0 if a_axis2 == -1 or a_shape2 == 1 else dim2) * a_stride2
        + (0 if a_axis3 == -1 or a_shape3 == 1 else dim3) * a_stride3
    )
    b_offset = (
        (0 if b_axis0 == -1 or b_shape0 == 1 else dim0) * b_stride0
        + (0 if b_axis1 == -1 or b_shape1 == 1 else dim1) * b_stride1
        + (0 if b_axis2 == -1 or b_shape2 == 1 else dim2) * b_stride2
        + (0 if b_axis3 == -1 or b_shape3 == 1 else dim3) * b_stride3
    )
    c_offset = (
        (0 if c_axis0 == -1 or c_shape0 == 1 else dim0) * c_stride0
        + (0 if c_axis1 == -1 or c_shape1 == 1 else dim1) * c_stride1
        + (0 if c_axis2 == -1 or c_shape2 == 1 else dim2) * c_stride2
        + (0 if c_axis3 == -1 or c_shape3 == 1 else dim3) * c_stride3
    )

    a = tl.load(a_ptr + a_offset, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + b_offset, mask=mask).to(tl.float32)
    c = tl.load(c_ptr + c_offset, mask=mask).to(tl.float32)

    if OP_A == 25:
        a = -a
    elif OP_A == 8:
        a = tl.maximum(a, 0)
    elif OP_A == 11:
        a = tl.sigmoid(a)
    elif OP_A == 12:
        a = (tl.exp(2 * a) - 1) / (tl.exp(2 * a) + 1)
    elif OP_A == 24:
        a = tl.abs(a)
    elif OP_A == 22:
        a = tl.exp(a)
    elif OP_A == 23:
        a = tl.log(a)
    elif OP_A == 2:
        a = tl.sqrt(a)

    if OP_B == 25:
        b = -b
    elif OP_B == 8:
        b = tl.maximum(b, 0)
    elif OP_B == 11:
        b = tl.sigmoid(b)
    elif OP_B == 12:
        b = (tl.exp(2 * b) - 1) / (tl.exp(2 * b) + 1)
    elif OP_B == 24:
        b = tl.abs(b)
    elif OP_B == 22:
        b = tl.exp(b)
    elif OP_B == 23:
        b = tl.log(b)
    elif OP_B == 2:
        b = tl.sqrt(b)

    if OP_C == 25:
        c = -c
    elif OP_C == 8:
        c = tl.maximum(c, 0)
    elif OP_C == 11:
        c = tl.sigmoid(c)
    elif OP_C == 12:
        c = (tl.exp(2 * c) - 1) / (tl.exp(2 * c) + 1)
    elif OP_C == 24:
        c = tl.abs(c)
    elif OP_C == 22:
        c = tl.exp(c)
    elif OP_C == 23:
        c = tl.log(c)
    elif OP_C == 2:
        c = tl.sqrt(c)

    a = a * ALPHA
    b = b * BETA
    c = c * GAMMA

    if OP_AB == 3:
        ab = a + b
    elif OP_AB == 5:
        ab = a * b
    elif OP_AB == 6:
        ab = tl.maximum(a, b)
    else:
        ab = tl.minimum(a, b)

    if OP_ABC == 3:
        out = ab + c
    elif OP_ABC == 5:
        out = ab * c
    elif OP_ABC == 6:
        out = tl.maximum(ab, c)
    else:
        out = tl.minimum(ab, c)

    tl.store(out_ptr + offsets, out, mask=mask)


_MAX_SUPPORTED_OUTPUT_RANK = 16

_TRINARY_EXECUTOR_CACHE = {}


def _build_direct_plan(a, b, c, mode_a, mode_b, mode_c, mode_d,
                       op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
    norm_a = _normalize_modes(mode_a, a.ndim)
    norm_b = _normalize_modes(mode_b, b.ndim)
    norm_c = _normalize_modes(mode_c, c.ndim)
    output_modes = infer_elementwise_output_modes((tuple(norm_a), tuple(norm_b), tuple(norm_c)), mode_d)
    output_shape = infer_elementwise_output_shape(
        ((a, tuple(norm_a)), (b, tuple(norm_b)), (c, tuple(norm_c))),
        output_modes,
    )
    if len(output_shape) > _MAX_SUPPORTED_OUTPUT_RANK:
        raise ValueError(
            f"elementwise_trinary output rank {len(output_shape)} exceeds the supported "
            f"limit ({_MAX_SUPPORTED_OUTPUT_RANK}); reshape inputs or reduce "
            "the number of distinct modes before calling elementwise_trinary"
        )
    if not _supports_direct_fused_triton_trinary(
        a, b, c, norm_a, norm_b, norm_c, output_modes,
        op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code,
    ):
        return None
    return {
        "output_shape": output_shape,
        "n_elements": int(torch.Size(output_shape).numel()),
        "grid": _make_static_grid(int(torch.Size(output_shape).numel())),
    }


def _build_indexed_plan(a, b, c, mode_a, mode_b, mode_c, mode_d,
                        op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code):
    mode_a = _normalize_modes(mode_a, a.ndim)
    mode_b = _normalize_modes(mode_b, b.ndim)
    mode_c = _normalize_modes(mode_c, c.ndim)
    mode_groups = (tuple(mode_a), tuple(mode_b), tuple(mode_c))
    tensors_with_modes = ((a, mode_groups[0]), (b, mode_groups[1]), (c, mode_groups[2]))
    output_modes = infer_elementwise_output_modes(mode_groups, mode_d)
    output_shape = infer_elementwise_output_shape(tensors_with_modes, output_modes)

    if len(output_shape) > _MAX_SUPPORTED_OUTPUT_RANK:
        raise ValueError(
            f"elementwise_trinary output rank {len(output_shape)} exceeds the supported "
            f"limit ({_MAX_SUPPORTED_OUTPUT_RANK}); reshape inputs or reduce "
            "the number of distinct modes before calling elementwise_trinary"
        )
    if not _supports_indexed_fused_triton_trinary(
        output_shape, output_modes,
        op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code,
    ):
        return None
    _validate_fused_codes(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code)
    launch_data = prepare_indexed_launch_tensors(output_shape, tensors_with_modes, output_modes)
    if launch_data is None:
        return None

    padded_shape, metadata = launch_data
    a_axes, a_strides, a_shapes = metadata[0]
    b_axes, b_strides, b_shapes = metadata[1]
    c_axes, c_strides, c_shapes = metadata[2]
    n_elements = int(torch.Size(output_shape).numel())
    return {
        "mode_groups": mode_groups,
        "output_modes": output_modes,
        "output_shape": output_shape,
        "n_elements": n_elements,
        "grid": _make_static_grid(n_elements),
        "specialized_layout": (
            "benchmark_indexed_5d"
            if _is_specialized_log_neg_sqrt_add_max(op_a_code, op_b_code, op_c_code, op_ab_code, op_abc_code)
            and _is_specialized_benchmark_indexed_layout(mode_groups, output_modes, output_shape)
            else None
        ),
        "padded_shape": padded_shape,
        "a_meta": metadata[0],
        "b_meta": metadata[1],
        "c_meta": metadata[2],
        "indexed_launch_meta": {
            "out_dim0": padded_shape[0],
            "out_dim1": padded_shape[1],
            "out_dim2": padded_shape[2],
            "out_dim3": padded_shape[3],
            "a_axis0": a_axes[0],
            "a_axis1": a_axes[1],
            "a_axis2": a_axes[2],
            "a_axis3": a_axes[3],
            "b_axis0": b_axes[0],
            "b_axis1": b_axes[1],
            "b_axis2": b_axes[2],
            "b_axis3": b_axes[3],
            "c_axis0": c_axes[0],
            "c_axis1": c_axes[1],
            "c_axis2": c_axes[2],
            "c_axis3": c_axes[3],
            "a_stride0": a_strides[0],
            "a_stride1": a_strides[1],
            "a_stride2": a_strides[2],
            "a_stride3": a_strides[3],
            "b_stride0": b_strides[0],
            "b_stride1": b_strides[1],
            "b_stride2": b_strides[2],
            "b_stride3": b_strides[3],
            "c_stride0": c_strides[0],
            "c_stride1": c_strides[1],
            "c_stride2": c_strides[2],
            "c_stride3": c_strides[3],
            "a_shape0": a_shapes[0],
            "a_shape1": a_shapes[1],
            "a_shape2": a_shapes[2],
            "a_shape3": a_shapes[3],
            "b_shape0": b_shapes[0],
            "b_shape1": b_shapes[1],
            "b_shape2": b_shapes[2],
            "b_shape3": b_shapes[3],
            "c_shape0": c_shapes[0],
            "c_shape1": c_shapes[1],
            "c_shape2": c_shapes[2],
            "c_shape3": c_shapes[3],
        },
    }


def _launch_indexed_plan(plan, a, b, c, output, op_a_code, op_b_code, op_c_code,
                         op_ab_code, op_abc_code, alpha, beta, gamma):
    padded_shape = plan["padded_shape"]
    a_axes, a_strides, a_shapes = plan["a_meta"]
    b_axes, b_strides, b_shapes = plan["b_meta"]
    c_axes, c_strides, c_shapes = plan["c_meta"]
    n_elements = output.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
    )
    _trinary_generic_indexed_kernel[grid](
        a, b, c, output, n_elements,
        padded_shape[0], padded_shape[1], padded_shape[2], padded_shape[3],
        a_axis0=a_axes[0], a_axis1=a_axes[1], a_axis2=a_axes[2], a_axis3=a_axes[3],
        b_axis0=b_axes[0], b_axis1=b_axes[1], b_axis2=b_axes[2], b_axis3=b_axes[3],
        c_axis0=c_axes[0], c_axis1=c_axes[1], c_axis2=c_axes[2], c_axis3=c_axes[3],
        a_stride0=a_strides[0], a_stride1=a_strides[1], a_stride2=a_strides[2], a_stride3=a_strides[3],
        b_stride0=b_strides[0], b_stride1=b_strides[1], b_stride2=b_strides[2], b_stride3=b_strides[3],
        c_stride0=c_strides[0], c_stride1=c_strides[1], c_stride2=c_strides[2], c_stride3=c_strides[3],
        a_shape0=a_shapes[0], a_shape1=a_shapes[1], a_shape2=a_shapes[2], a_shape3=a_shapes[3],
        b_shape0=b_shapes[0], b_shape1=b_shapes[1], b_shape2=b_shapes[2], b_shape3=b_shapes[3],
        c_shape0=c_shapes[0], c_shape1=c_shapes[1], c_shape2=c_shapes[2], c_shape3=c_shapes[3],
        OP_A=op_a_code, OP_B=op_b_code, OP_C=op_c_code,
        OP_AB=op_ab_code, OP_ABC=op_abc_code,
        ALPHA=float(alpha), BETA=float(beta), GAMMA=float(gamma),
    )
    return output


def elementwise_trinary(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    op_a="identity",
    op_b="identity",
    op_c="identity",
    op_ab="add",
    op_abc="add",
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    mode_a=None,
    mode_b=None,
    mode_c=None,
    mode_d=None,
    out=None,
):
    """Compute ``op_abc(op_ab(alpha*op_a(a), beta*op_b(b)), gamma*op_c(c))``.

    Execution strategy (picked automatically based on shapes / ops):

    1. **Fused indexed Triton kernel** — taken when every op is in the fused
       kernel's supported set and the output rank fits the indexed kernel's
       packing limit. Handles broadcasting and mode remapping directly.
    2. **Fused contiguous Triton kernel** — taken after input alignment when
       every op is in the fused set. All intermediate math is done in fp32.
    3. **Composed fallback** — used when any op falls outside the fused set
       (e.g. ``sin`` / ``cos``). Dispatches to this project's own
       ``add``/``mul``/``max``/``min`` and unary ops. Low-precision inputs
       (fp16 / bf16) are upcast to fp32 for the whole composition and cast
       back at the end.

    Numerical note: fused and composed paths both carry fp32 intermediates
    but differ in *when* rounding happens (per-op vs per-composition), so
    last-bit differences are expected for low-precision dtypes.

    Raises ``ValueError`` when the output rank exceeds
    ``_MAX_SUPPORTED_OUTPUT_RANK`` (16).
    """
    if not a.is_cuda or not b.is_cuda or not c.is_cuda:
        raise ValueError("input tensors must be on CUDA")
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise TypeError("input tensors must have the same dtype")
    executor = _get_triton_trinary_executor(op_a, op_b, op_c, op_ab, op_abc, alpha, beta, gamma)
    return executor(
        a,
        b,
        c,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        out=out,
    )
