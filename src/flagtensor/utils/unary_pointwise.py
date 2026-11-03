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
from flagtensor.utils.direct_launch import execution_key as _execution_key
from flagtensor.utils.direct_launch import stream_fns as _stream_fns
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
    if _IS_ASCEND or _IS_KUNLUNXIN:
        # triton-ascend's libdevice.acos has a precision bug (~3e-4 error),
        # and XPU's libdevice.acos has a ~3e-3 precision error. Fall back to
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
    if _IS_ASCEND or _IS_KUNLUNXIN:
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
    if _IS_KUNLUNXIN:
        # See ``asinh_libdevice``: ``libdevice.acosh`` cannot be lowered for
        # fp16 with large tiles on P800, so use the log/sqrt identity.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return tl.log(xf + tl.sqrt(xf * xf - 1))
        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.acosh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("acosh_log_sqrt")
def _build_acosh_log_sqrt_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # ``libdevice.log2`` combined with a large ``BLOCK_SIZE *
        # BLOCKS_PER_PROGRAM`` tile trips the XPU lowering pass
        # ("triton_xpu.unpack reached lowering without a bufPtr").  The
        # core ``tl.log`` path compiles for every tile on P800 and is
        # accurate to ~1e-6 (float32).
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return tl.log(xf + tl.sqrt(xf * xf - 1))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(xf + tl.sqrt(xf * xf - 1))

    return _variant


@_register_unary_rewrite("asin_libdevice")
def _build_asin_libdevice_variant(scalar_fn):
    if _IS_ASCEND or _IS_KUNLUNXIN:
        # triton-ascend's libdevice.asin has a precision bug (~3e-4 error),
        # and XPU's libdevice.asin has a similar precision issue. Fall back
        # to ``atan(x / sqrt(1-x*x))`` which uses only the well-behaved
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
    if _IS_ASCEND or _IS_KUNLUNXIN:
        # triton-ascend's libdevice.atan2 is unusable inside JIT functions,
        # and XPU's libdevice.asin has precision issues. Reuse the
        # atan-based form from asin_libdevice.
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
    if _IS_KUNLUNXIN:
        # ``libdevice.asinh`` fails to legalize for fp16 outputs with tiles
        # >= 4096 ("triton_xpu.unpack ... bufPtr"); the log/sqrt form
        # compiles for every tile size.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            abs_x = tl.abs(xf)
            inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
            return tl.where(xf >= 0, tl.log(inner), -tl.log(inner))
        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.asinh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("asinh_log_sqrt")
def _build_asinh_log_sqrt_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            abs_x = tl.abs(xf)
            inner = abs_x + tl.sqrt(abs_x * abs_x + 1)
            return tl.where(xf >= 0, tl.log(inner), -tl.log(inner))
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

    if _IS_KUNLUNXIN:
        # ``libdevice.atan2`` hits the fp16 "triton_xpu.unpack" legalization
        # failure for tiles >= 4096 (same lowering hazard as
        # ``libdevice.asinh``/``atanh``), so use ``libdevice.atan``, which
        # compiles for every tile.
        @triton.jit
        def _variant(x):
            return libdevice.atan(x.to(tl.float32))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return libdevice.atan2(xf, 1.0)
    return _variant


@_register_unary_rewrite("atanh_libdevice")
def _build_atanh_libdevice_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # ``libdevice.atanh`` cannot be lowered for fp16 with large tiles;
        # use the log-ratio identity through the core ``tl.log``, which
        # compiles for every tile.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return 0.5 * tl.log((1 + xf) / (1 - xf))
        return _variant

    @triton.jit
    def _variant(x):
        return libdevice.atanh(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("atanh_log_ratio")
def _build_atanh_log_ratio_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return 0.5 * tl.log((1 + xf) / (1 - xf))
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 0.5 * tl.log((1 + xf) / (1 - xf))

    return _variant


@_register_unary_rewrite("ceil_intrinsic")
def _build_ceil_intrinsic_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # A bare ``tl.ceil`` on a masked fp32->fp16 tile of >= 4096 elements
        # hits the "triton_xpu.unpack ... nor a bufPtr" legalization failure.
        # Spelling it as floor + correction keeps the same result and
        # compiles for every tile.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            floored = tl.floor(xf)
            return tl.where(floored == xf, xf, floored + 1)
        return _variant

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
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return 0.5 * (tl.exp(xf) + tl.exp(-xf))
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
            xf = x.to(tl.float32)
            exp_pos = tl.exp(xf)
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
    if _IS_KUNLUNXIN:
        # XPU's ``tl.exp2`` (core IR op lowered by the XPU backend) produces
        # large precision errors, while ``libdevice.exp2`` is a software
        # emulation that is ~1000x slower than the hardware ``tl.exp``
        # (34 ms vs 0.2 ms for 8M elements).  ``tl.exp`` is both accurate
        # and fast on this backend.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return tl.exp(xf)
        return _variant

    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return tl.exp2(xf * log2e)

    return _variant


@_register_unary_rewrite("floor_intrinsic")
def _build_floor_intrinsic_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # See ``ceil_intrinsic``: a bare ``tl.floor`` cannot be lowered for
        # masked fp16 tiles >= 4096 on P800; the ceil + correction spelling
        # is equivalent and compiles everywhere.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            ceiled = tl.ceil(xf)
            return tl.where(ceiled == xf, xf, ceiled - 1)
        return _variant

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
    if _IS_KUNLUNXIN:
        # ``libdevice.log2`` (used below for accuracy on XPU) trips the
        # "triton_xpu.unpack reached lowering without a bufPtr" legalization
        # failure for fp16 outputs once the tile reaches 4096 elements, which
        # would remove the large-tile configs from the tuner.  ``tl.log``
        # compiles for every tile and measures the same accuracy on P800
        # (max rel. error 4.9e-4 on fp16, 1.6e-4 on fp32).
        @triton.jit
        def _variant(x):
            return tl.log(x.to(tl.float32))
        return _variant

    @triton.jit
    def _variant(x):
        return tl.log(x.to(tl.float32))

    return _variant


@_register_unary_rewrite("log2_scaled")
def _build_log2_scaled_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # See ``log_intrinsic``: prefer the core ``tl.log`` so fp16 works
        # with large tiles.
        @triton.jit
        def _variant(x):
            return tl.log(x.to(tl.float32))
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
    @triton.jit
    def _variant(x):
        return 1.0 / x.to(tl.float32)

    return _variant


@_register_unary_rewrite("rcp_exp_log")
def _build_rcp_exp_log_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            ax = tl.abs(xf)
            recip_abs = tl.exp(-libdevice.log2(ax) * ln2)
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
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return 0.5 * (tl.exp(xf) - tl.exp(-xf))
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
            xf = x.to(tl.float32)
            exp_pos = tl.exp(xf)
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
    if _IS_KUNLUNXIN:
        # Do *not* write this as ``tl.sin(x) / tl.cos(x)`` on XPU: for tiles
        # of one core (512 lanes) the compiler emits an approximate
        # range-reduction sequence (~1% error, and ``inf`` for values next to
        # odd multiples of pi/2) instead of the accurate polynomial used for
        # wider tiles.  ``libdevice.tan`` is accurate for every tile width.
        @triton.jit
        def _variant(x):
            return libdevice.tan(x.to(tl.float32))

        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.sin(xf) / tl.cos(xf)

    return _variant


@_register_unary_rewrite("tan_recip_divide")
def _build_tan_recip_divide_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # Same single-core fast-math hazard as ``tan_divide`` above.
        @triton.jit
        def _variant(x):
            return libdevice.tan(x.to(tl.float32))

        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return 1.0 / (tl.cos(xf) / tl.sin(xf))

    return _variant


@_register_unary_rewrite("sigmoid_exp2")
def _build_sigmoid_exp2_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # Avoid writing this as ``1 / (1 + exp(-x))``: the XPU compiler
        # pattern-matches that form into ``triton_xpu.vsigmoidf``, which
        # fails to legalize for tiles larger than one core (~512 lanes).
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            y = tl.where(
                xf >= 0,
                1.0 / (1.0 + tl.exp(-xf)),
                tl.exp(xf) / (1.0 + tl.exp(xf)),
            )
            return y
        return _variant

    @triton.jit
    def _variant(x):
        log2e: tl.constexpr = 1.4426950408889634
        xf = x.to(tl.float32)
        return 1 / (1 + tl.exp2(-xf * log2e))

    return _variant


@_register_unary_rewrite("sigmoid_exp")
def _build_sigmoid_exp_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            exp_pos = tl.exp(xf)
            return exp_pos / (1.0 + exp_pos)
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
        # ``libdevice.exp2`` is correct on XPU but compiles to a software
        # emulation that is ~1000x slower than the hardware ``tl.exp``
        # (34 ms vs 0.2 ms for 8M elements), and it also trips the
        # ``triton_xpu.unpack`` legalization bug for large tiles.  Use
        # ``tl.exp`` with the natural-exponent argument instead.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            return 2 / (1 + tl.exp(-2 * xf)) - 1
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
            xf = x.to(tl.float32)
            exp_neg_twice = tl.exp(-2 * xf)
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
        # ``libdevice.log2`` is the accurate XPU log (``tl.log`` drifts),
        # but ``libdevice.exp2`` is a ~1000x-slower software emulation, so
        # the exponent must go through the hardware ``tl.exp``.
        @triton.jit
        def _variant(x):
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(1 + tl.exp(-tl.abs(xf))) * ln2 + tl.where(xf > 0, xf, 0)
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
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            return libdevice.log2(1 + tl.exp(-tl.abs(xf))) * ln2 + tl.maximum(xf, 0)
        return _variant

    @triton.jit
    def _variant(x):
        xf = x.to(tl.float32)
        return tl.log(1 + tl.exp(-tl.abs(xf))) + tl.maximum(xf, 0)

    return _variant


@_register_unary_rewrite("swish_exp2")
def _build_swish_exp2_variant(scalar_fn):
    if _IS_KUNLUNXIN:
        # Same ``triton_xpu.vsigmoidf`` legalization hazard as sigmoid: the
        # literal ``1 / (1 + exp(-x))`` form only compiles for small tiles,
        # so use the branch-selected equivalent instead.
        @triton.jit
        def _variant(x):
            xf = x.to(tl.float32)
            sigmoid = tl.where(
                xf >= 0,
                1.0 / (1.0 + tl.exp(-xf)),
                tl.exp(xf) / (1.0 + tl.exp(xf)),
            )
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
            xf = x.to(tl.float32)
            exp_neg = tl.exp(-xf)
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
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            softplus = libdevice.log2(1 + tl.exp(-tl.abs(xf))) * ln2 + tl.where(xf > 0, xf, 0)
            tanh_softplus = 2 / (1 + tl.exp(-2 * softplus)) - 1
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
            ln2: tl.constexpr = 0.6931471805599453
            xf = x.to(tl.float32)
            softplus = libdevice.log2(1 + tl.exp(-tl.abs(xf))) * ln2 + tl.maximum(xf, 0)
            exp_neg_twice = tl.exp(-2 * softplus)
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
# torch.float8_e5m2 / float8_e4m3fn were added in torch 2.1; guard with
# getattr so older torch (e.g. 2.0.1 used by the Kunlunxin XPU plugin)
# doesn't crash at import time.
_TRIVIAL_UNARY_EXTRA = {torch.int8}
if hasattr(torch, "float8_e5m2"):
    _TRIVIAL_UNARY_EXTRA.add(torch.float8_e5m2)
_NEG_UNARY_EXTRA = {torch.int8}  # neg works for int8 (fp8_e5m2 fails on triton 3.3)


def _build_fast_launch(launcher, x: torch.Tensor, n_elements: int):
    """Compile the cached plan into a self-validating closure.

    On XPU the Python-side dispatch can cost more than the kernel itself, so
    the win is in *how few Python frames* sit between the caller and the C
    launcher.  Everything that is constant for a plan (launch arguments,
    allocator, expected dtype/device/stream) is captured here, leaving one
    guard, one allocation, two ``data_ptr`` calls and one C call.

    Returns ``None`` when the incoming tensor does not match the plan, which
    tells the caller to take the general dispatch path.
    """

    launch, fixed = launcher.launch_parts
    new_empty = torch.Tensor.new_empty
    get_stream = _stream_fns()[1]
    dtype = x.dtype
    align = x.data_ptr() & 15
    device = x.get_device()
    stream = get_stream(device)

    def fast(x: torch.Tensor):
        if not (
            x.dtype is dtype
            and x.numel() == n_elements
            and (x.data_ptr() & 15) == align
            and x.is_contiguous()
            and x.get_device() == device
            and get_stream(device) == stream
        ):
            return None
        y = new_empty(x, x.shape)
        launch(*fixed, x.data_ptr(), y.data_ptr(), n_elements)
        return y

    return fast


class _UnaryPointwiseExecutor:
    def __init__(self, kernel, prepare_input, supported_dtypes=None):
        self.kernel = kernel
        self.prepare_input = prepare_input
        self._supported = supported_dtypes or _DEFAULT_UNARY_DTYPES
        # Fast path: one cached launcher per (dtype, element count, pointer
        # alignment, device, stream).  Resolving it costs a tuple lookup
        # instead of the whole tuner dispatch, which matters on backends
        # where Python-side launch overhead dominates the kernel time.
        self._launchers = {}
        #: Self-validating closure for the most recent plan, or ``None``.
        #: Kept separate from :attr:`_launchers` so the hot call can skip the
        #: dict lookup and key construction entirely.
        self.fast = None
        #: The default ``prepare_input`` is a no-op, which is what makes the
        #: single-slot fast path applicable.
        self._plain_prepare = prepare_input is None or prepare_input is _default_prepare

    def _launch_plan_key(self, x: torch.Tensor, n_elements: int):
        return (x.dtype, n_elements, x.data_ptr() & 15, _execution_key(x.get_device()))

    def _remember(self, launcher, x: torch.Tensor, n_elements: int):
        """Refresh the single-slot fast plan so the next identical call is cheap."""
        self.fast = _build_fast_launch(launcher, x, n_elements)

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

        y = torch.empty_like(prepared_x)
        n_elements = y.numel()
        plan_key = self._launch_plan_key(prepared_x, n_elements)
        launcher = self._launchers.get(plan_key)
        if launcher is not None:
            if self._plain_prepare:
                self._remember(launcher, prepared_x, n_elements)
            launcher(prepared_x, y, n_elements)
            return y

        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["BLOCKS_PER_PROGRAM"]),
        )
        self.kernel[grid](prepared_x, y, n_elements)
        launcher = getattr(self.kernel, "last_launcher", None)
        if launcher is not None:
            if len(self._launchers) >= 1024:
                self._launchers.clear()
            self._launchers[plan_key] = launcher
            if self._plain_prepare:
                self._remember(launcher, prepared_x, n_elements)
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
        # Hot path: the cached plan validates itself and returns ``None`` as
        # soon as the tensor no longer matches, so nothing is dispatched
        # twice and the general path below stays authoritative.  The guard is
        # wrapped because it runs untrusted input through device queries
        # (``get_device`` raises on host tensors): any failure simply means
        # "not this plan", and the general path re-raises the real error with
        # the usual validation messages.
        f = executor.fast
        if f is not None:
            try:
                out = f(x)
            except Exception:
                out = None
            if out is not None:
                return out
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
