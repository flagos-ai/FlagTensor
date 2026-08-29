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

from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl

try:
    # Triton <= 3.3 (e.g. flagtree 0.4.0+3.3 / iluvatar3.1): the direct
    # extern libdevice lives under the cuda namespace. CoreX 3.1 also ships
    # a top‑level triton/language/extra/libdevice.py, but that one is a
    # dispatch variant that produces wrong results on this backend — so the
    # cuda path must be tried first.
    from triton.language.extra.cuda import libdevice
except ImportError:
    try:
        # CoreX Triton >= 3.6 (e.g. flagtree 0.6.1+iluvatar3.6): the cuda
        # namespace is gone; the real extern libdevice moved to
        # triton.language.extra.corex (the top‑level extra/libdevice.py
        # there is a signature stub with no implementations).
        from triton.language.extra.corex import libdevice
    except ImportError:
        try:
            # Upstream Triton >= 3.4: real libdevice at the top level.
            from triton.language.extra import libdevice
        except ImportError:
            # Non‑NVIDIA backends (e.g. triton‑ascend) ship their own libdevice under
            # a vendor‑specific subpackage. Fall back to the active vendor's module.
            try:
                from triton.language.extra.ascend import libdevice  # type: ignore
            except ImportError:  # pragma: no cover — keeps test collection working
                libdevice = None  # type: ignore


from flagtensor import runtime
from flagtensor.runtime import is_on_accelerator as _is_on_accelerator
from flagtensor.utils.libtuner import libtuner


# ---------------------------------------------------------------------------
# Vendor flag — triton-ascend's libdevice has precision bugs in asin/acos
# and a JIT bug in atan2. On NVIDIA we keep the original libdevice calls
# (faster, more precise, autotuner has two distinct variants to compare).
# On Ascend we fall back to atan-based mathematically-equivalent forms.
# ---------------------------------------------------------------------------
try:
    _IS_ASCEND = runtime.device.vendor_name == "ascend"
except Exception:
    _IS_ASCEND = False


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
    "identity_like": ("identity_direct", "identity_direct"),
    "log_like": ("log_intrinsic", "log2_scaled"),
    "neg_like": ("neg_direct", "neg_sub"),
    "rcp_like": ("rcp_direct", "rcp_exp_log"),
    "relu_like": ("relu_where", "relu_max"),
    "sigmoid_like": ("sigmoid_exp2", "sigmoid_exp"),
    "sin_like": ("sin_intrinsic", "sin_phase_shift"),
    "sinh_like": ("sinh_exp_pair", "sinh_exp_recip"),
    "softsign_like": ("softsign_abs", "softsign_piecewise"),
    "sqrt_like": ("sqrt_intrinsic", "sqrt_rsqrt"),
    "tan_like": ("tan_libdevice", "tan_divide"),
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
    if _IS_ASCEND:
        # triton-ascend's libdevice.acos has a precision bug (~3e-4 error).
        # Fall back to ``pi/2 - atan(x / sqrt(1-x*x))`` which uses only the
        # well-behaved libdevice.atan path.
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            return pi_over_2 - libdevice.atan(xf / tl.sqrt(1.0 - xf * xf))
        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.acos(x.to(tl.float32))
    return _variant


@_register_unary_rewrite("acos_asin_shift")
def _build_acos_asin_shift_variant(scalar_fn):
    if _IS_ASCEND:
        # Same atan-based fallback as acos_libdevice above.
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            return pi_over_2 - libdevice.atan(xf / tl.sqrt(1.0 - xf * xf))
        return _variant

    @triton.jit
    def _variant(x):
        pi_over_2: tl.constexpr = 1.5707963267948966
        xf = x.to(tl.float32)
        return pi_over_2 - libdevice.asin(xf)
    return _variant


@_register_unary_rewrite("acos_atan_poly")
def _build_acos_atan_poly_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        pi_over_2: tl.constexpr = 1.5707963267948966
        xf = x.to(tl.float32)
        atan_x = xf / tl.sqrt(1.0 - xf * xf)
        ax = tl.abs(atan_x)
        use_recip = ax > 1.0
        z = tl.where(use_recip, 1.0 / ax, ax)
        t = z * z
        p = 0.003049968053146109
        p = p * t + -0.01682744845907338
        p = p * t + 0.04385559893427329
        p = p * t + -0.07596809856142807
        p = p * t + 0.10681421027256047
        p = p * t + -0.1421319619537072
        p = p * t + 0.19993716142481666
        p = p * t + -0.33333120780994563
        p = p * t + 0.9999999880828081
        result = z * p
        result = tl.where(use_recip, pi_over_2 - result, result)
        result = tl.where(atan_x < 0, -result, result)
        return pi_over_2 - result

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
    if _IS_ASCEND:
        # triton-ascend's libdevice.asin has a precision bug (~3e-4 error).
        # Fall back to ``atan(x / sqrt(1-x*x))`` which uses only the
        # well-behaved libdevice.atan path.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return libdevice.atan(xf / tl.sqrt(1.0 - xf * xf))
        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.asin(x.to(tl.float32))
    return _variant


@_register_unary_rewrite("asin_atan2")
def _build_asin_atan2_variant(scalar_fn):
    if _IS_ASCEND:
        # triton-ascend's libdevice.atan2 is unusable inside JIT functions.
        # Reuse the atan-based form from asin_libdevice.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return libdevice.atan(xf / tl.sqrt(1.0 - xf * xf))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 2 * libdevice.atan2(xf, 1 + tl.sqrt(1 - xf * xf))
    return _variant


@_register_unary_rewrite("asin_atan_poly")
def _build_asin_atan_poly_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        pi_over_2: tl.constexpr = 1.5707963267948966
        xf = x.to(tl.float32)
        atan_x = xf / tl.sqrt(1.0 - xf * xf)
        ax = tl.abs(atan_x)
        use_recip = ax > 1.0
        z = tl.where(use_recip, 1.0 / ax, ax)
        t = z * z
        p = 0.003049968053146109
        p = p * t + -0.01682744845907338
        p = p * t + 0.04385559893427329
        p = p * t + -0.07596809856142807
        p = p * t + 0.10681421027256047
        p = p * t + -0.1421319619537072
        p = p * t + 0.19993716142481666
        p = p * t + -0.33333120780994563
        p = p * t + 0.9999999880828081
        result = z * p
        result = tl.where(use_recip, pi_over_2 - result, result)
        return tl.where(atan_x < 0, -result, result)

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
    if _IS_ASCEND:
        # triton-ascend's libdevice.atan2 is unusable inside JIT functions.
        # atan2(x, 1.0) == atan(x), so use libdevice.atan directly.
        @triton.jit
        def _variant(x):
            return libdevice.atan(x.to(tl.float32))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return libdevice.atan2(xf, 1.0)
    return _variant


def _make_atan_poly_variant():
    @triton.jit
    def _variant(x):
        pi_over_2: tl.constexpr = 1.5707963267948966
        ax = tl.abs(x.to(tl.float32))
        use_recip = ax > 1.0
        z = tl.where(use_recip, 1.0 / ax, ax)
        t = z * z
        p = 0.003049968053146109
        p = p * t + -0.01682744845907338
        p = p * t + 0.04385559893427329
        p = p * t + -0.07596809856142807
        p = p * t + 0.10681421027256047
        p = p * t + -0.1421319619537072
        p = p * t + 0.19993716142481666
        p = p * t + -0.33333120780994563
        p = p * t + 0.9999999880828081
        result = z * p
        result = tl.where(use_recip, pi_over_2 - result, result)
        return tl.where(x < 0, -result, result)

    return _variant


@_register_unary_rewrite("atan_poly")
def _build_atan_poly_variant(scalar_fn):
    return _make_atan_poly_variant()


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


@_register_unary_rewrite("tan_libdevice")
def _build_tan_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.tan(x.to(tl.float32))

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
    # Triton 3.3 JIT 编译器的 JITFunction.__init__ 依赖
    # inspect.getsourcelines(fn) 获取源码，且 AST→TTIR 阶段无法解析
    # Python 闭包变量 (free variable)。将 kernel 写入 .py 文件后通过
    # importlib 加载，使 inspect 能定位源码，同时将 variant 函数
    # 及其闭包依赖全部注入为模块全局变量来绕开限制。
    import os
    import sys
    import importlib.util

    cache_dir = os.path.join(os.path.dirname(__file__), "__kernels__")
    os.makedirs(cache_dir, exist_ok=True)

    module_name = f"_gen_{op_name}"
    file_path = os.path.join(cache_dir, f"{module_name}.py")

    kernel_src = f"""import triton
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
def _{op_name}_kernel(
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

result = _{op_name}_kernel
result.__name__ = "_{op_name}_kernel"
"""

    # 多 GPU worker 并发导入同一算子时（如 tools/run_tests.py 每 GPU 一个
    # 进程），直接覆写共享的 _gen_*.py 会让其他进程读到写了一半的内容，
    # 导致 triton.jit 的 inspect.getsourcelines 抛 "could not get source
    # code"。内容不变时跳过写入；需要写入时先写临时文件再 os.replace
    # 原子替换，保证并发读者始终看到完整文件（生成的内容是确定性的）。
    need_write = True
    try:
        with open(file_path, "r") as f:
            need_write = f.read() != kernel_src
    except OSError:
        need_write = True
    if need_write:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(kernel_src)
            os.replace(tmp_path, file_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)

    mod._variant0 = variant0
    mod._variant1 = variant1

    # variant 函数本身可能也有闭包依赖（如 scalar_fn），一并注入
    for variant in (variant0, variant1):
        # JITFunction 包装了一层，闭包在 .fn 上
        inner = getattr(variant, "fn", variant)
        closure = getattr(inner, "__closure__", None)
        if closure:
            freevars = getattr(inner, "__code__", None)
            if freevars:
                for cell, name in zip(closure, freevars.co_freevars):
                    if not hasattr(mod, name):
                        setattr(mod, name, cell.cell_contents)

    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    sys.modules.pop(module_name, None)
    return mod.result


def _default_prepare(x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    return None, x


# Default dtypes supported by all Triton unary kernels.
# Each operator can expand this via the supported_dtypes kwarg.
_DEFAULT_UNARY_DTYPES = {torch.float16, torch.float32, torch.bfloat16}

# Additional dtype groups for operators that support them.
_TRIVIAL_UNARY_EXTRA = {torch.int8, torch.float8_e5m2}  # identity, abs
_NEG_UNARY_EXTRA = {torch.int8}  # neg works for int8 (fp8_e5m2 fails on triton 3.3)

class _UnaryPointwiseExecutor:
    def __init__(self, kernel, prepare_input, supported_dtypes=None):
        self.kernel = kernel
        self.prepare_input = prepare_input
        self.layout_cache = {}
        self._supported = supported_dtypes or _DEFAULT_UNARY_DTYPES

    def _layout_key(self, x: torch.Tensor):
        return (
            x.dtype,
            tuple(x.shape),
            x.stride(),
            x.is_contiguous(),
        )

    def _build_plan(self, prepared_x: torch.Tensor):
        return {
            "contiguous": prepared_x.is_contiguous(),
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not _is_on_accelerator(x):
            raise ValueError("input tensor must be on the active accelerator device")
        if x.dtype not in self._supported:
            raise ValueError(
                f"unsupported dtype {x.dtype} for unary operator; "
                f"supported: {sorted(str(d) for d in self._supported)}"
            )
        handled, prepared_x = self.prepare_input(x)
        if handled is not None:
            return handled

        layout_key = self._layout_key(prepared_x)
        plan = self.layout_cache.get(layout_key)
        if plan is None:
            plan = self._build_plan(prepared_x)
            self.layout_cache[layout_key] = plan

        y = torch.empty_like(prepared_x)
        n_elements = y.numel()
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )
        self.kernel[grid](prepared_x, y, n_elements)
        return y


def make_unary_pointwise(
    op_name: str,
    variant0,
    variant1,
    *,
    prepare_input: Optional[
        Callable[[torch.Tensor], Tuple[Optional[torch.Tensor], torch.Tensor]]
    ] = None,
    supported_dtypes: Optional[set[torch.dtype]] = None,
):
    kernel = _build_unary_kernel(op_name, variant0, variant1)
    prepare = prepare_input or _default_prepare
    executor = _UnaryPointwiseExecutor(kernel, prepare, supported_dtypes=supported_dtypes)

    def op(x: torch.Tensor) -> torch.Tensor:
        return executor(x)

    op.__name__ = op_name
    return kernel, op


def make_unary_pointwise_from_family(
    op_name: str,
    family: str,
    scalar_fn,
    *,
    prepare_input: Optional[
        Callable[[torch.Tensor], Tuple[Optional[torch.Tensor], torch.Tensor]]
    ] = None,
    rewrite_rules: Optional[Tuple[str, str]] = None,
    supported_dtypes: Optional[set[torch.dtype]] = None,
):
    if family not in _UNARY_FAMILY_RULES:
        raise ValueError(f"unsupported unary family: {family}")
    variant0, variant1 = _resolve_family_variants(family, scalar_fn, rewrite_rules)
    return make_unary_pointwise(
        op_name,
        variant0,
        variant1,
        prepare_input=prepare_input,
        supported_dtypes=supported_dtypes,
    )
