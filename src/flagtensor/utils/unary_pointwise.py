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
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl

from flagtensor import runtime
from flagtensor.runtime import is_on_accelerator as _is_on_accelerator
from flagtensor.runtime import persistent_grid_from_meta
from flagtensor.utils.libtuner import libtuner


# ---------------------------------------------------------------------------
# Vendor flag — detect the active vendor once at import time.
#   * triton-ascend's libdevice has precision bugs in asin/acos and a JIT
#     bug in atan2. On Ascend we fall back to atan-based forms.
#   * On Kunlunxin XPU, ``triton.language.extra.cuda.libdevice`` is
#     importable but its ``__nv_*`` symbols are rejected by the XPU
#     elfconv, so every ``libdevice.*`` call fails to compile. We must
#     use ``triton.language.extra.xpu.libdevice`` instead. Additionally,
#     ``tl.exp`` / ``tl.log`` on XPU use low-precision polynomial
#     approximations; we redirect them to the hardware-backed
#     ``tl.exp2`` / ``tl.log2`` paths.
#   * On NVIDIA we keep the original libdevice calls (faster, more
#     precise, autotuner has two distinct variants to compare).
# ---------------------------------------------------------------------------
try:
    _VENDOR_NAME = runtime.device.vendor_name
except Exception:
    _VENDOR_NAME = None
_IS_ASCEND = _VENDOR_NAME == "ascend"
_IS_KUNLUNXIN = _VENDOR_NAME == "kunlunxin"


# ---------------------------------------------------------------------------
# libdevice selection — vendor-appropriate to avoid symbol mismatch.
# The XPU triton package ships ``triton.language.extra.cuda`` (inherited
# from the upstream tree), so the naive ``try: from ...cuda import
# libdevice`` succeeds on Kunlunxin but binds ``__nv_*`` symbols that the
# XPU elfconv cannot resolve, causing every ``libdevice.*`` kernel to
# fail at compile time. Select the XPU libdevice explicitly on Kunlunxin;
# every other vendor keeps the original import order
# (cuda → corex → top-level extra → ascend).
# ---------------------------------------------------------------------------
if _IS_KUNLUNXIN:
    try:
        from triton.language.extra.xpu import libdevice  # type: ignore
    except ImportError:  # pragma: no cover — keeps test collection working
        libdevice = None  # type: ignore
else:
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

# On Ascend the top-level ``triton.language.extra.libdevice`` above resolves to
# a signature-only stub, so the real extern library must be imported from the
# vendor namespace explicitly.  It exposes the vector intrinsics the kernels
# below rely on (``reciprocal`` in particular).
_ascend_libdevice = None
if _IS_ASCEND:
    try:
        from triton.language.extra.ascend import libdevice as _ascend_libdevice  # type: ignore
    except ImportError:  # pragma: no cover
        _ascend_libdevice = None


# ---------------------------------------------------------------------------
# Fast inverse-trig helpers for Ascend.
#
# ``triton.language.extra.libdevice`` is a signature-only stub on the
# triton-ascend build used here (every body is ``...``, so any call returns
# ``None`` and compilation fails), and the patched ``tl.math.atan`` lowers to a
# scalar libm call that costs more than the whole memory traffic of the op.
# asin/acos are therefore evaluated with the standard half-angle reduction
# ``acos(|x|) = 2 * atan(sqrt((1-|x|) / (1+|x|)))`` plus a degree-5 minimax
# polynomial for ``atan`` on [0, tan(pi/8)] (max error 1.4e-9; 6e-5
# end-to-end, well inside the 1e-4 fp32 tolerance).
# ---------------------------------------------------------------------------
@triton.jit
def _atan_reduced(t):
    """``atan`` for ``t`` in [0, 1] (reduced to |z| <= tan(pi/8))."""
    big = t > 0.41421356237309503
    z = tl.where(big, (t - 1.0) / (t + 1.0), t)
    u = z * z
    p = -0.06029505937492857
    p = p * u + 0.10566481414358579
    p = p * u - 0.14238113424472507
    p = p * u + 0.199980132839913
    p = p * u - 0.33333299856917903
    p = p * u + 0.9999999986499089
    a = z * p
    return tl.where(big, 0.7853981633974483 + a, a)


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
    if _IS_ASCEND:
        # ``tl.where(x >= 0, x, -x)`` lowers to compare + select + negate
        # (three vector passes over the tile) on triton-ascend, which makes the
        # large shapes 3-5x slower than the vendor baseline.  ``tl.abs`` is a
        # single ``fabs`` instruction, so both autotuner variants use it and a
        # wrong pick cannot cost anything.
        @triton.jit
        def _variant(x):
            return tl.abs(x)

        return _variant

    @triton.jit
    def _variant(x):
        return tl.where(x >= 0, x, -x)

    return _variant


@_register_unary_rewrite("acos_libdevice")
def _build_acos_libdevice_variant(scalar_fn):
    if _IS_ASCEND:
        # ``libdevice`` is a stub on triton-ascend and ``tl.math.atan`` is a
        # slow scalar call; use the polynomial helper (see _atan_reduced).
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            s = tl.abs(xf)
            # acos(|x|) = 2 * atan(sqrt((1-|x|) / (1+|x|)))
            acos_abs = 2.0 * _atan_reduced(tl.sqrt((1.0 - s) / (1.0 + s)))
            acos_x = tl.where(xf < 0, 3.141592653589793 - acos_abs, acos_abs)
            # ``pi_over_2`` keeps the odd/even symmetry explicit for the
            # shapes where acos(x) is exactly pi/2.
            return tl.where(xf == 0, pi_over_2, acos_x)
        return _variant

    if _IS_KUNLUNXIN:
        # XPU's libdevice.acos has a ~3e-3 precision error. Fall back to
        # ``pi/2 - atan(x / sqrt(1-x*x))`` which uses only the well-behaved
        # libdevice.atan path.
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
        # Alternate form for the autotuner: single reduced atan without the
        # half-angle square root (slightly cheaper, same accuracy class).
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            s = tl.abs(xf)
            acos_abs = 2.0 * _atan_reduced(tl.sqrt((1.0 - s) / (1.0 + s)))
            return tl.where(xf < 0, 3.141592653589793 - acos_abs, acos_abs)
        return _variant

    if _IS_KUNLUNXIN:
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


@_register_unary_rewrite("acosh_libdevice")
def _build_acosh_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.acosh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("acosh_log_sqrt")
def _build_acosh_log_sqrt_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(xf + tl.sqrt(xf * xf - 1)) * ln2
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(xf + tl.sqrt(xf * xf - 1))

    return _variant


@_register_unary_rewrite("asin_libdevice")
def _build_asin_libdevice_variant(scalar_fn):
    if _IS_ASCEND:
        # See _atan_reduced: asin(x) = sign(x) * (pi/2 - acos(|x|)).
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            s = tl.abs(xf)
            acos_abs = 2.0 * _atan_reduced(tl.sqrt((1.0 - s) / (1.0 + s)))
            asin_abs = pi_over_2 - acos_abs
            return tl.where(xf < 0, -asin_abs, asin_abs)
        return _variant

    if _IS_KUNLUNXIN:
        # XPU's libdevice.asin has a precision issue. Fall back to
        # ``atan(x / sqrt(1-x*x))`` which uses only the well-behaved
        # libdevice.atan path.
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
        # Alternate form for the autotuner (see asin_libdevice above): same
        # identity, but the half-angle ratio is evaluated with a reciprocal
        # square root instead of a division plus sqrt.
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            s = tl.abs(xf)
            acos_abs = 2.0 * _atan_reduced(tl.sqrt(1.0 - s) * tl.rsqrt(1.0 + s))
            asin_abs = pi_over_2 - acos_abs
            return tl.where(xf < 0, -asin_abs, asin_abs)
        return _variant

    if _IS_KUNLUNXIN:
        # XPU's libdevice.asin has precision issues. Reuse the atan-based form
        # from asin_libdevice.
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


@_register_unary_rewrite("asinh_libdevice")
def _build_asinh_libdevice_variant(scalar_fn):
    @triton.jit
    def _variant(x):
        return libdevice.asinh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("asinh_log_sqrt")
def _build_asinh_log_sqrt_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            abs_x = tl.abs(xf)
            inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
            return tl.where(xf >= 0, libdevice.log2(inner) * ln2, -libdevice.log2(inner) * ln2)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        abs_x = tl.abs(xf)
        inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
        return tl.where(xf >= 0, tl.log(inner), -tl.log(inner))

    return _variant


@_register_unary_rewrite("atan_libdevice")
def _build_atan_libdevice_variant(scalar_fn):
    if _IS_ASCEND:
        # ``triton.language.extra.libdevice`` is a signature-only stub on
        # triton-ascend (``atan`` returns ``None`` inside JIT) and the real
        # vendor extern (``__hmf_atanf``) is ~10% slower than aten for the
        # whole op.  The polynomial below is 2x faster and accurate to 2e-7:
        # atan(x) = sign(x) * (pi/2 - atan(1/|x|)) for |x| > 1, with
        # ``_atan_reduced`` covering [0, 1].
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            ax = tl.abs(xf)
            inv = ax > 1.0
            a = _atan_reduced(tl.where(inv, 1.0 / ax, ax))
            a = tl.where(inv, pi_over_2 - a, a)
            return tl.where(xf < 0, -a, a)

        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.atan(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("atan_atan2")
def _build_atan_atan2_variant(scalar_fn):
    if _IS_ASCEND:
        # triton-ascend's libdevice.atan2 is unusable inside JIT functions
        # (it lowers to a scalar ``math.atan2`` call), and atan2(x, 1.0) ==
        # atan(x); reuse the polynomial from ``atan_libdevice``.
        @triton.jit
        def _variant(x):
            pi_over_2: tl.constexpr = 1.5707963267948966
            xf = x.to(tl.float32)
            ax = tl.abs(xf)
            inv = ax > 1.0
            a = _atan_reduced(tl.where(inv, 1.0 / ax, ax))
            a = tl.where(inv, pi_over_2 - a, a)
            return tl.where(xf < 0, -a, a)

        return _variant

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
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return 0.5 * libdevice.log2((1 + xf) / (1 - xf)) * ln2
        return _variant

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
    if _IS_ASCEND:
        # ``tl.ceil`` already lowers to a single vector instruction on
        # triton-ascend, so building it out of ``floor`` + compare + select +
        # add is strictly more work.  The autotuner benchmarks the two
        # alternatives with a single output buffer, and at n <= 2048 (where
        # the whole kernel is ~0.3us on top of the 1.2us graph-replay floor)
        # the measurement is noise, so it sometimes picks this branch and
        # loses ~10%.  Making the alternative identical means a mis-pick
        # cannot cost anything.
        @triton.jit
        def _variant(x):
            return tl.ceil(x.to(tl.float32))

        return _variant

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
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            return 0.5 * (libdevice.exp2(xf * log2e) + libdevice.exp2(-xf * log2e))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 0.5 * (tl.exp(xf) + tl.exp(-xf))

    return _variant


@_register_unary_rewrite("cosh_exp_recip")
def _build_cosh_exp_recip_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            exp_pos = libdevice.exp2(xf * log2e)
            return 0.5 * (exp_pos + 1.0 / exp_pos)
        return _variant

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
    if _IS_ASCEND:
        # Alternate for the autotuner: evaluate ``exp`` in the *input* dtype
        # instead of staging through fp32.  For fp16 tensors the whole tensor
        # then stays 2 bytes wide (vexp in fp16 + no up/down conversion) which
        # is ~15% faster at HBM-bound sizes and equally correct: the input is
        # already fp16, so overflow to +inf is exactly what fp16 ``exp`` must
        # return.  bf16/fp32 fall through to the same code the primary variant
        # uses, so picking either cannot hurt.
        @triton.jit
        def _variant(x):
            return tl.exp(x)
        return _variant

    if _IS_KUNLUNXIN:
        # XPU's ``tl.exp2`` (core IR op lowered by the XPU backend) produces
        # large precision errors, but ``libdevice.exp2`` dispatches to the
        # hardware-backed ``_ZN3xpu5exp2fEf`` symbol with full precision.
        # Use libdevice.exp2 so the autotuner cannot pick a broken variant.
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            return libdevice.exp2(xf * log2e)
        return _variant

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
    if _IS_ASCEND:
        # ``tl.floor`` is a single vector instruction on triton-ascend; the
        # ceil-based fallback costs three extra vector ops.  Keep the two
        # autotuner variants identical on Ascend so a noisy tiny-shape
        # measurement cannot select the slower one (see ceil_floor_adjust).
        @triton.jit
        def _variant(x):
            return tl.floor(x.to(tl.float32))

        return _variant

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
    if _IS_ASCEND:
        # ``tl.log`` is a native vector instruction for fp32 *and* the 2-byte
        # dtypes on triton-ascend; the fp32 staging pass it used to do was
        # pure overhead for fp16/bf16 (and 4x of the traffic at 4M/8M).
        @triton.jit
        def _variant(x):
            return tl.log(x)

        return _variant

    if _IS_KUNLUNXIN:
        # XPU's ``tl.log`` (core IR op) produces large precision errors.
        # Use ``libdevice.log2`` (hardware-backed ``_ZN3xpu5log2fEf``)
        # via ``log(x) = log2(x) * ln2`` so the autotuner cannot pick a
        # broken variant.
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(xf) * ln2
        return _variant

    @triton.jit
    def _variant(x):
        return tl.log(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("log2_scaled")
def _build_log2_scaled_variant(scalar_fn):
    if _IS_ASCEND:
        # Same native path as ``log_intrinsic`` so a wrong autotuner pick is
        # harmless (the log2*ln2 form was ~5% slower at small shapes and
        # identical at large ones).
        @triton.jit
        def _variant(x):
            return tl.log(x)

        return _variant

    if _IS_KUNLUNXIN:
        # XPU's ``tl.log2`` (core IR op) produces large precision errors,
        # but ``libdevice.log2`` dispatches to the hardware-backed
        # ``_ZN3xpu5log2fEf`` symbol with full precision.
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(xf) * ln2
        return _variant

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
    if _IS_ASCEND and _ascend_libdevice is not None:
        # ``1.0 / x.to(fp32)`` stages fp16/bf16 through fp32 and pays for a
        # correctly-rounded divide; the vector reciprocal is a single
        # instruction (and stays 2 bytes wide for fp16).  bf16 has no extern,
        # so it is promoted to fp32 first.
        # NOTE: keep a single ``return`` -- an early return inside the
        # ``if`` makes triton-ascend trace both branches, and the fp16 extern
        # then rejects the bf16 value.
        @triton.jit
        def _variant(x):
            if x.dtype == tl.bfloat16:
                y = _ascend_libdevice.reciprocal(x.to(tl.float32))
            else:
                y = _ascend_libdevice.reciprocal(x)
            return y

        return _variant

    @triton.jit
    def _variant(x):
        return 1.0 / x.to(tl.float32)

    return _variant


@_register_unary_rewrite("rcp_exp_log")
def _build_rcp_exp_log_variant(scalar_fn):
    if _IS_ASCEND and _ascend_libdevice is not None:
        # Keep both autotuner variants on the one-instruction path; the
        # exp/log round trip is never competitive and only risks a bad pick.
        @triton.jit
        def _variant(x):
            if x.dtype == tl.bfloat16:
                y = _ascend_libdevice.reciprocal(x.to(tl.float32))
            else:
                y = _ascend_libdevice.reciprocal(x)
            return y

        return _variant

    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            ax = tl.abs(xf)
            recip_abs = libdevice.exp2(-libdevice.log2(ax) * ln2 * log2e)
            return tl.where(xf >= 0, recip_abs, -recip_abs)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        ax = tl.abs(xf)
        recip_abs = tl.exp(-tl.log(ax))
        return tl.where(xf >= 0, recip_abs, -recip_abs)

    return _variant


@_register_unary_rewrite("relu_where")
def _build_relu_where_variant(scalar_fn):
    if _IS_ASCEND:
        # Same story as ``abs_where``: compare + select is two passes over the
        # tile while the vector ``max`` is one.  ``propagate_nan=ALL`` restores
        # torch semantics (``relu(nan) == nan``) that the select form lost.
        @triton.jit
        def _variant(x):
            return tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)

        return _variant

    @triton.jit
    def _variant(x):
        return tl.where(x > 0, x, 0)

    return _variant


@_register_unary_rewrite("relu_max")
def _build_relu_max_variant(scalar_fn):
    if _IS_ASCEND:
        @triton.jit
        def _variant(x):
            return tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)

        return _variant

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
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            return 0.5 * (libdevice.exp2(xf * log2e) - libdevice.exp2(-xf * log2e))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 0.5 * (tl.exp(xf) - tl.exp(-xf))

    return _variant


@_register_unary_rewrite("sinh_exp_recip")
def _build_sinh_exp_recip_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            exp_pos = libdevice.exp2(xf * log2e)
            return 0.5 * (exp_pos - 1.0 / exp_pos)
        return _variant

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
    if _IS_ASCEND:
        # ``tl.rsqrt`` is *not* a vector intrinsic on triton-ascend (it lowers
        # to a slow scalar sequence: ~0.45x of aten on its own), so the usual
        # ``x * rsqrt(x)`` form must never be selected there.  Provide the
        # native-dtype ``tl.sqrt`` instead: for fp16 it removes the fp32
        # staging passes (~15% faster), for bf16/fp32 it matches the primary
        # variant.
        @triton.jit
        def _variant(x):
            return tl.sqrt(x)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return xf * tl.rsqrt(xf)

    return _variant


@_register_unary_rewrite("tan_libdevice")
def _build_tan_libdevice_variant(scalar_fn):
    if _IS_ASCEND and _ascend_libdevice is not None:
        # The top-level ``libdevice`` is a signature-only stub on
        # triton-ascend (``tan`` returns ``None`` inside JIT); the vendor
        # module exposes the real ``__hmf_tanf`` extern.  Single ``return``:
        # an early return makes triton-ascend trace both branches and then
        # the fp16/fp32 return types clash.
        @triton.jit
        def _variant(x):
            if x.dtype == tl.float16:
                y = _ascend_libdevice.tan(x)
            else:
                y = _ascend_libdevice.tan(x.to(tl.float32))
            return y

        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.tan(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("tan_divide")
def _build_tan_divide_variant(scalar_fn):
    if _IS_ASCEND and _ascend_libdevice is not None:
        # sin/cos + divide is 3 passes unless the backend fuses it.  Use the
        # one-instruction vector tan for both variants so the autotuner cannot
        # pick the slow (and UB-hungry) form.
        @triton.jit
        def _variant(x):
            if x.dtype == tl.float16:
                y = _ascend_libdevice.tan(x)
            else:
                y = _ascend_libdevice.tan(x.to(tl.float32))
            return y

        return _variant

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
    if _IS_ASCEND and _ascend_libdevice is not None:
        # ``tl.exp`` lowers to one vector instruction on triton-ascend while
        # ``tl.exp2`` needs an extra scaling pass (≈15% slower for the whole
        # op).  The algebraically equivalent ``1 - 1/(1+exp(x))`` is used
        # instead of ``1/(1+exp(-x))`` for two reasons:
        #   * it drops the negation of the input, and
        #   * it stays correct for large ``x``: ``exp(x)`` overflows to +inf,
        #     ``1/(1+inf)`` is 0 and the result is exactly 1.0 (the
        #     ``exp(x)/(1+exp(x))`` variant would give inf/inf = NaN).
        # fp16 keeps the whole tensor 2 bytes wide and uses the vector
        # reciprocal instead of the correctly-rounded divide (~+20%); bf16
        # and fp32 evaluate in fp32 because triton-ascend has no bf16
        # reciprocal and the extra fp16 conversion costs more than the divide
        # it saves.  ``tl.fdiv(..., False)`` selects the vector unit's
        # approximate division (1 ulp), one pass cheaper than ``/``.
        @triton.jit
        def _variant(x):
            if x.dtype == tl.float16:
                y = 1.0 - _ascend_libdevice.reciprocal(1.0 + tl.exp(x))
            else:
                xf = x.to(tl.float32)
                y = 1.0 - tl.fdiv(1.0, 1.0 + tl.exp(xf), False)
            return y
        return _variant

    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            return 1 / (1 + libdevice.exp2(-xf * log2e))
        return _variant

    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return 1 / (1 + tl.exp2(-xf * log2e))

    return _variant


@_register_unary_rewrite("sigmoid_exp")
def _build_sigmoid_exp_variant(scalar_fn):
    if _IS_ASCEND and _ascend_libdevice is not None:
        # Same identity as ``sigmoid_exp2``; the only difference is that bf16
        # is evaluated with native bf16 vector ops instead of fp32 staging, so
        # the autotuner has a real alternative for bf16 (the two forms are
        # within noise of each other, so a wrong pick is harmless).
        @triton.jit
        def _variant(x):
            if x.dtype == tl.float16:
                y = 1.0 - _ascend_libdevice.reciprocal(1.0 + tl.exp(x))
            elif x.dtype == tl.bfloat16:
                y = 1.0 - 1.0 / (1.0 + tl.exp(x))
            else:
                xf = x.to(tl.float32)
                y = 1.0 - tl.fdiv(1.0, 1.0 + tl.exp(xf), False)
            return y
        return _variant

    if _IS_ASCEND:
        @triton.jit
        def _variant(x):
            return 1.0 - 1.0 / (1.0 + tl.exp(x))
        return _variant

    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            exp_neg = libdevice.exp2(-xf * log2e)
            return 1 / (1 + exp_neg)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_neg = tl.exp(-xf)
        return 1 / (1 + exp_neg)

    return _variant


@_register_unary_rewrite("tanh_exp2")
def _build_tanh_exp2_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            return 2 / (1 + libdevice.exp2(-2 * xf * log2e)) - 1
        return _variant

    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return 2 / (1 + tl.exp2(-2 * xf * log2e)) - 1

    return _variant


@_register_unary_rewrite("tanh_exp")
def _build_tanh_exp_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            exp_neg_twice = libdevice.exp2(-2 * xf * log2e)
            return (1 - exp_neg_twice) / (1 + exp_neg_twice)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_neg_twice = tl.exp(-2 * xf)
        return (1 - exp_neg_twice) / (1 + exp_neg_twice)

    return _variant


@_register_unary_rewrite("softplus_where")
def _build_softplus_where_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(1 + libdevice.exp2(-tl.abs(xf) * log2e)) * ln2 + tl.where(xf > 0, xf, 0)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(1 + tl.exp(-tl.abs(xf))) + tl.where(xf > 0, xf, 0)

    return _variant


@_register_unary_rewrite("softplus_max")
def _build_softplus_max_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(1 + libdevice.exp2(-tl.abs(xf) * log2e)) * ln2 + tl.maximum(xf, 0)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(1 + tl.exp(-tl.abs(xf))) + tl.maximum(xf, 0)

    return _variant


@_register_unary_rewrite("swish_exp2")
def _build_swish_exp2_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            sigmoid = 1 / (1 + libdevice.exp2(-xf * log2e))
            return xf * sigmoid
        return _variant

    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        sigmoid = 1 / (1 + tl.exp2(-xf * log2e))
        return xf * sigmoid

    return _variant


@_register_unary_rewrite("swish_exp")
def _build_swish_exp_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            xf = x.to(tl.float32)
            exp_neg = libdevice.exp2(-xf * log2e)
            return xf / (1 + exp_neg)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        exp_neg = tl.exp(-xf)
        return xf / (1 + exp_neg)

    return _variant


@_register_unary_rewrite("mish_exp2")
def _build_mish_exp2_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            softplus = libdevice.log2(1 + libdevice.exp2(-tl.abs(xf) * log2e)) * ln2 + tl.where(xf > 0, xf, 0)
            tanh_softplus = 2 / (1 + libdevice.exp2(-2 * softplus * log2e)) - 1
            return xf * tanh_softplus
        return _variant

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
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            log2e: tl.constexpr = 1.4426950408889634
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            softplus = libdevice.log2(1 + libdevice.exp2(-tl.abs(xf) * log2e)) * ln2 + tl.maximum(xf, 0)
            exp_neg_twice = libdevice.exp2(-2 * softplus * log2e)
            tanh_softplus = (1 - exp_neg_twice) / (1 + exp_neg_twice)
            return xf * tanh_softplus
        return _variant

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


def _build_unary_kernel(op_name: str, variant0, variant1, config_key=None):
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

    # Ascend: one persistent program per vector core, each program walks a
    # strided range of tiles.  The conventional "one program per tile" mapping
    # generates thousands of blocks for the benchmark shapes and is dominated
    # by device-side block scheduling.
    #
    # When ``n_elements`` is an exact multiple of the tile (the common case for
    # the benchmark shapes) the bounds mask is dropped: predicated loads/stores
    # cost ~10-20% at HBM-bound sizes.  The check is a single uniform branch
    # outside the loop, so it does not inhibit loop pipelining.
    if _IS_ASCEND:
        # ``EVEN`` is a constexpr supplied by the heuristic: when the tensor
        # length is an exact multiple of the tile (all benchmark shapes) the
        # bounds mask is compiled away.  Predicated loads/stores cost 10-20%
        # at HBM-bound sizes.
        kernel_body = """    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    TILE: tl.constexpr = BLOCK_SIZE * BLOCKS_PER_PROGRAM
    num_tiles = tl.cdiv(n_elements, TILE)
    if EVEN:
        for tile_id in tl.range(pid, num_tiles, num_programs, num_stages=2):
            offsets = tile_id * TILE + tl.arange(0, TILE)
            x = tl.load(x_ptr + offsets)
            if KERNEL_ID == 0:
                y = _variant0(x)
            else:
                y = _variant1(x)
            tl.store(y_ptr + offsets, y)
    else:
        for tile_id in tl.range(pid, num_tiles, num_programs, num_stages=2):
            offsets = tile_id * TILE + tl.arange(0, TILE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            if KERNEL_ID == 0:
                y = _variant0(x)
            else:
                y = _variant1(x)
            tl.store(y_ptr + offsets, y, mask=mask)
"""
    else:
        kernel_body = """    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCKS_PER_PROGRAM
    offsets = block_start + tl.arange(0, BLOCK_SIZE * BLOCKS_PER_PROGRAM)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    if KERNEL_ID == 0:
        y = _variant0(x)
    else:
        y = _variant1(x)
    tl.store(y_ptr + offsets, y, mask=mask)
"""

    # The Ascend persistent kernel gets its own tuning table / tile heuristics:
    # it can use much larger tiles than the generic one-program-per-tile
    # kernels that share the "elementwise_unary" entry.
    if config_key is None:
        config_key = (
            "elementwise_unary_persistent" if _IS_ASCEND else "elementwise_unary"
        )
    even_param = "\n    EVEN: tl.constexpr," if _IS_ASCEND else ""

    kernel_src = f"""import triton
import triton.language as tl
from flagtensor import runtime
from flagtensor.utils.libtuner import libtuner

@libtuner(
    configs=runtime.get_tuned_config("{config_key}"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("{config_key}"))
@triton.jit
def _{op_name}_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,{even_param}
    KERNEL_ID: tl.constexpr,
):
{kernel_body}
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
# torch.float8_e5m2 / float8_e4m3fn were added in torch 2.1; guard with
# getattr so older torch (e.g. 2.0.1 used by the Kunlunxin XPU plugin)
# doesn't crash at import time.
_TRIVIAL_UNARY_EXTRA = {torch.int8}
if hasattr(torch, "float8_e5m2"):
    _TRIVIAL_UNARY_EXTRA.add(torch.float8_e5m2)
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
        if _IS_ASCEND:
            grid = lambda meta: (persistent_grid_from_meta(n_elements, meta),)
        else:
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
    config_key: Optional[str] = None,
):
    kernel = _build_unary_kernel(op_name, variant0, variant1, config_key=config_key)
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
    config_key: Optional[str] = None,
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
        config_key=config_key,
    )
