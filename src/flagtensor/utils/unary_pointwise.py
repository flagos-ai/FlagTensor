from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from flagtensor import runtime
from flagtensor.utils.libtuner import libtuner


_UNARY_FAMILY_RULES = {
    "acos_like": ("acos_libdevice", "acos_asin_shift"),
    "acosh_like": ("acosh_libdevice", "acosh_log_sqrt"),
    "asin_like": ("asin_libdevice", "asin_atan2"),
    "asinh_like": ("asinh_libdevice", "asinh_log_sqrt"),
    "atan_like": ("atan_libdevice", "atan_atan2"),
    "atanh_like": ("atanh_libdevice", "atanh_log_ratio"),
    "abs_like": ("abs_intrinsic", "abs_where"),
    "ceil_like": ("ceil_intrinsic", "ceil_floor_adjust"),
    "cos_like": ("cos_intrinsic", "cos_phase_shift"),
    "cosh_like": ("cosh_exp_pair", "cosh_exp_recip"),
    "exp_like": ("exp_intrinsic", "exp2_scaled"),
    "floor_like": ("floor_intrinsic", "floor_ceil_adjust"),
    "identity_like": ("identity_direct", "identity_f32"),
    "log_like": ("log_intrinsic", "log2_scaled"),
    "neg_like": ("neg_direct", "neg_sub"),
    "rcp_like": ("rcp_direct", "rcp_exp_log"),
    "relu_like": ("relu_where", "relu_max"),
    "sigmoid_like": ("sigmoid_exp2", "sigmoid_exp"),
    "sin_like": ("sin_intrinsic", "sin_phase_shift"),
    "sinh_like": ("sinh_exp_pair", "sinh_exp_recip"),
    "softsign_like": ("softsign_abs", "softsign_piecewise"),
    "sqrt_like": ("sqrt_intrinsic", "sqrt_rsqrt"),
    "tan_like": ("tan_divide", "tan_recip_divide"),
    "tanh_like": ("tanh_exp2", "tanh_exp"),
    "softplus_like": ("softplus_where", "softplus_max"),
    "swish_like": ("swish_exp2", "swish_exp"),
    "mish_like": ("mish_exp2", "mish_exp"),
}


_UNARY_REWRITE_BUILDERS = {}


def _register_unary_rewrite(rewrite_name: str):
    def _decorator(builder):
        _UNARY_REWRITE_BUILDERS[rewrite_name] = builder
        return builder

    return _decorator


@_register_unary_rewrite("scalar_f32")
def _build_scalar_f32_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return scalar_fn(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("abs_intrinsic")
def _build_abs_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.abs(x)

    return _variant


@_register_unary_rewrite("abs_where")
def _build_abs_where_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.where(x >= 0, x, -x)

    return _variant


@_register_unary_rewrite("acos_libdevice")
def _build_acos_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.acos(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("acos_asin_shift")
def _build_acos_asin_shift_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        pi_over_2: tl.constexpr = 1.5707963267948966
        xf = x.to(tl.float32)
        return pi_over_2 - libdevice.asin(xf)

    return _variant


@_register_unary_rewrite("acosh_libdevice")
def _build_acosh_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.acosh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("acosh_log_sqrt")
def _build_acosh_log_sqrt_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(xf + tl.sqrt(xf * xf - 1))

    return _variant


@_register_unary_rewrite("asin_libdevice")
def _build_asin_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.asin(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("asin_atan2")
def _build_asin_atan2_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 2 * libdevice.atan2(xf, 1 + tl.sqrt(1 - xf * xf))

    return _variant


@_register_unary_rewrite("asinh_libdevice")
def _build_asinh_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.asinh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("asinh_log_sqrt")
def _build_asinh_log_sqrt_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        abs_x = tl.abs(xf)
        inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
        return tl.where(xf >= 0, tl.log(inner), -tl.log(inner))

    return _variant


@_register_unary_rewrite("atan_libdevice")
def _build_atan_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.atan(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("atan_atan2")
def _build_atan_atan2_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return libdevice.atan2(xf, 1.0)

    return _variant


@_register_unary_rewrite("atanh_libdevice")
def _build_atanh_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.atanh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("atanh_log_ratio")
def _build_atanh_log_ratio_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 0.5 * tl.log((1 + xf) / (1 - xf))

    return _variant


@_register_unary_rewrite("ceil_intrinsic")
def _build_ceil_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.ceil(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("ceil_floor_adjust")
def _build_ceil_floor_adjust_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        floored = tl.floor(xf)
        return floored + tl.where(xf > floored, 1.0, 0.0)

    return _variant


@_register_unary_rewrite("cos_intrinsic")
def _build_cos_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.cos(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("cos_phase_shift")
def _build_cos_phase_shift_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.sin(0.5 * 3.141592653589793 - xf)

    return _variant


@_register_unary_rewrite("cosh_exp_pair")
def _build_cosh_exp_pair_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 0.5 * (tl.exp(xf) + tl.exp(-xf))

    return _variant


@_register_unary_rewrite("cosh_exp_recip")
def _build_cosh_exp_recip_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_pos = tl.exp(xf)
        return 0.5 * (exp_pos + 1.0 / exp_pos)

    return _variant


@_register_unary_rewrite("exp_intrinsic")
def _build_exp_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.exp(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("exp2_scaled")
def _build_exp2_scaled_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return tl.exp2(xf * log2e)

    return _variant


@_register_unary_rewrite("floor_intrinsic")
def _build_floor_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.floor(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("floor_ceil_adjust")
def _build_floor_ceil_adjust_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        ceiled = tl.ceil(xf)
        return ceiled - tl.where(xf < ceiled, 1.0, 0.0)

    return _variant


@_register_unary_rewrite("identity_direct")
def _build_identity_direct_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return x

    return _variant


@_register_unary_rewrite("identity_f32")
def _build_identity_f32_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return scalar_fn(x.to(tl.float32)).to(x.dtype)

    return _variant


@_register_unary_rewrite("log_intrinsic")
def _build_log_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.log(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("log2_scaled")
def _build_log2_scaled_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        ln2: tl.constexpr = 0.6931471805599453
        xf = x.to(tl.float32)
        return tl.log2(xf) * ln2

    return _variant


@_register_unary_rewrite("neg_direct")
def _build_neg_direct_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return -x

    return _variant


@_register_unary_rewrite("neg_sub")
def _build_neg_sub_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return 0 - x

    return _variant


@_register_unary_rewrite("rcp_direct")
def _build_rcp_direct_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return 1.0 / x.to(tl.float32)

    return _variant


@_register_unary_rewrite("rcp_exp_log")
def _build_rcp_exp_log_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        ax = tl.abs(xf)
        recip_abs = tl.exp(-tl.log(ax))
        return tl.where(xf >= 0, recip_abs, -recip_abs)

    return _variant


@_register_unary_rewrite("relu_where")
def _build_relu_where_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.where(x > 0, x, 0)

    return _variant


@_register_unary_rewrite("relu_max")
def _build_relu_max_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.maximum(x, 0)

    return _variant


@_register_unary_rewrite("sin_intrinsic")
def _build_sin_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.sin(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("sin_phase_shift")
def _build_sin_phase_shift_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.cos(0.5 * 3.141592653589793 - xf)

    return _variant


@_register_unary_rewrite("sinh_exp_pair")
def _build_sinh_exp_pair_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 0.5 * (tl.exp(xf) - tl.exp(-xf))

    return _variant


@_register_unary_rewrite("sinh_exp_recip")
def _build_sinh_exp_recip_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_pos = tl.exp(xf)
        return 0.5 * (exp_pos - 1.0 / exp_pos)

    return _variant


@_register_unary_rewrite("softsign_abs")
def _build_softsign_abs_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return xf / (tl.abs(xf) + 1)

    return _variant


@_register_unary_rewrite("softsign_piecewise")
def _build_softsign_piecewise_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        denom = tl.where(xf >= 0, xf + 1, 1 - xf)
        return xf / denom

    return _variant


@_register_unary_rewrite("sqrt_intrinsic")
def _build_sqrt_intrinsic_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return tl.sqrt(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("sqrt_rsqrt")
def _build_sqrt_rsqrt_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return xf * tl.rsqrt(xf)

    return _variant


@_register_unary_rewrite("tan_divide")
def _build_tan_divide_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.sin(xf) / tl.cos(xf)

    return _variant


@_register_unary_rewrite("tan_recip_divide")
def _build_tan_recip_divide_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 1.0 / (tl.cos(xf) / tl.sin(xf))

    return _variant


@_register_unary_rewrite("sigmoid_exp2")
def _build_sigmoid_exp2_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return 1 / (1 + tl.exp2(-xf * log2e))

    return _variant


@_register_unary_rewrite("sigmoid_exp")
def _build_sigmoid_exp_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_neg = tl.exp(-xf)
        return 1 / (1 + exp_neg)

    return _variant


@_register_unary_rewrite("tanh_exp2")
def _build_tanh_exp2_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return 2 / (1 + tl.exp2(-2 * xf * log2e)) - 1

    return _variant


@_register_unary_rewrite("tanh_exp")
def _build_tanh_exp_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_neg_twice = tl.exp(-2 * xf)
        return (1 - exp_neg_twice) / (1 + exp_neg_twice)

    return _variant


@_register_unary_rewrite("softplus_where")
def _build_softplus_where_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(1 + tl.exp(-tl.abs(xf))) + tl.where(xf > 0, xf, 0)

    return _variant


@_register_unary_rewrite("softplus_max")
def _build_softplus_max_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(1 + tl.exp(-tl.abs(xf))) + tl.maximum(xf, 0)

    return _variant


@_register_unary_rewrite("swish_exp2")
def _build_swish_exp2_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        sigmoid = 1 / (1 + tl.exp2(-xf * log2e))
        return xf * sigmoid

    return _variant


@_register_unary_rewrite("swish_exp")
def _build_swish_exp_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_neg = tl.exp(-xf)
        return xf / (1 + exp_neg)

    return _variant


@_register_unary_rewrite("mish_exp2")
def _build_mish_exp2_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        softplus = tl.log(1 + tl.exp(-tl.abs(xf))) + tl.where(xf > 0, xf, 0)
        tanh_softplus = 2 / (1 + tl.exp2(-2 * softplus * log2e)) - 1
        return xf * tanh_softplus

    return _variant


@_register_unary_rewrite("mish_exp")
def _build_mish_exp_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        softplus = tl.log(1 + tl.exp(-tl.abs(xf))) + tl.maximum(xf, 0)
        exp_neg_twice = tl.exp(-2 * softplus)
        tanh_softplus = (1 - exp_neg_twice) / (1 + exp_neg_twice)
        return xf * tanh_softplus

    return _variant


def _make_variant_from_rewrite(rewrite_name: str, scalar_fn):
    builder = _UNARY_REWRITE_BUILDERS.get(rewrite_name)
    if builder is None:
        raise ValueError(f"unsupported unary rewrite rule: {rewrite_name}")
    return builder(scalar_fn)


def _resolve_family_variants(
    family: str,
    scalar_fn,
    rewrite_rules: Optional[Tuple[str, str]] = None,
):
    resolved_rules = rewrite_rules or _UNARY_FAMILY_RULES[family]
    return (
        _make_variant_from_rewrite(resolved_rules[0], scalar_fn),
        _make_variant_from_rewrite(resolved_rules[1], scalar_fn),
    )


def _build_unary_kernel(op_name: str, variant0, variant1):
    @libtuner(
        configs=runtime.get_tuned_config("elementwise_unary"),
        key=["n_elements"],
        strategy=["align32"],
        warmup=5,
        rep=10,
    )
    @triton.heuristics(runtime.get_heuristic_config("elementwise_unary"))
    @triton.jit
    def _kernel(
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
            x = tl.load(x_ptr + offsets, mask=mask)
            y = variant0(x)
            tl.store(y_ptr + offsets, y, mask=mask)
        else:
            for block_idx in tl.static_range(0, BLOCKS_PER_PROGRAM):
                offsets = block_start + block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = variant1(x)
                tl.store(y_ptr + offsets, y, mask=mask)

    _kernel.__name__ = f"_{op_name}_kernel"
    return _kernel


def _default_prepare(x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    return None, x


def make_unary_pointwise(
    op_name: str,
    variant0,
    variant1,
    *,
    fallback_float64: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    prepare_input: Optional[
        Callable[[torch.Tensor], Tuple[Optional[torch.Tensor], torch.Tensor]]
    ] = None,
):
    kernel = _build_unary_kernel(op_name, variant0, variant1)
    prepare = prepare_input or _default_prepare

    def op(x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("input tensor must be on CUDA")
        handled, prepared_x = prepare(x)
        if handled is not None:
            return handled
        if prepared_x.dtype == torch.float64 and fallback_float64 is not None:
            return fallback_float64(prepared_x)
        y = torch.empty_like(prepared_x)
        n_elements = y.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )
        kernel[grid](prepared_x, y, n_elements)
        return y

    op.__name__ = op_name
    return kernel, op


def make_unary_pointwise_from_family(
    op_name: str,
    family: str,
    scalar_fn,
    *,
    fallback_float64: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    prepare_input: Optional[
        Callable[[torch.Tensor], Tuple[Optional[torch.Tensor], torch.Tensor]]
    ] = None,
    rewrite_rules: Optional[Tuple[str, str]] = None,
):
    if family not in _UNARY_FAMILY_RULES:
        raise ValueError(f"unsupported unary family: {family}")
    variant0, variant1 = _resolve_family_variants(family, scalar_fn, rewrite_rules)
    return make_unary_pointwise(
        op_name,
        variant0,
        variant1,
        fallback_float64=fallback_float64,
        prepare_input=prepare_input,
    )
