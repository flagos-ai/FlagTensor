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

"""torch_npu aten baseline — Ascend-native replacement for cuTensor.

cuTensor is NVIDIA-only. On Ascend we use ``torch_npu`` aten operators as
the vendor-optimized baseline. These operators are backed by CANN's aclnn
library, which is the Huawei-shipped equivalent of cuTensor: a curated
collection of highly tuned compute kernels for the Ascend AI Core. On
MThreads, the same torch-backed baseline is reused for the MUSA device so
the existing benchmark harness can run without a separate vendor fork.

This module exposes baseline classes that mirror the cuTensor class
hierarchy (``CuTensorAbs``, ``CuTensorBinary``...) so that
``flagtensor.benchmark_core.Benchmark._get_baseline_instance`` can resolve
them by class name on Ascend exactly as it resolves cuTensor classes on
NVIDIA. The classes implement the same minimal surface used by the
benchmark runner:

    - ``__init__(self, dtype=torch.float32)``
    - ``prepare(self, *args)``                          # no-op cache priming
    - ``__call__(self, *args) -> torch.Tensor``         # operator-mode call
    - ``build_kernel_callable(self, *args) -> Callable``# kernel-mode callable

The implementation deliberately wraps the corresponding ``torch`` /
``torch_npu`` aten functions (rather than calling aclnn directly through
ctypes) so the baseline tracks whatever kernel CANN dispatches for the
current Ascend chip. This matches the role cuTensor plays in the NVIDIA
pipeline: a vendor-supplied reference implementation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

try:
    import torch_npu  # noqa: F401  — registers `npu` device / aten kernels
    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False


def torch_npu_available() -> bool:
    """Return True when the vendor baseline can run on the active accelerator."""
    try:
        if _TORCH_NPU_AVAILABLE and torch.npu.is_available():
            return True
    except Exception:
        pass
    try:
        from flagtensor.runtime import device as _ft_device
        from flagtensor.runtime import is_accelerator_available as _is_accelerator_available

        return (
            _is_accelerator_available()
            and _ft_device.vendor_name in {"ascend", "mthreads"}
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Op name -> torch aten function (elementwise unary / binary / trinary)
# ---------------------------------------------------------------------------
# Each entry maps the lower-case FlagTensor op slug (the part of the
# CUTENSOR_OP_* name after the prefix, lower-cased) to a callable that
# implements the same math using torch aten ops. torch_npu registers
# Ascend-optimized kernels for these aten ops via CANN aclnn.
_UNARY_TORCH_MAP = {
    "identity": lambda x: x.clone(),
    "sqrt": torch.sqrt,
    "relu": torch.relu,
    "conj": torch.conj,
    "rcp": torch.reciprocal,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "neg": torch.neg,
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "asinh": torch.asinh,
    "acosh": torch.acosh,
    "atanh": torch.atanh,
    "ceil": torch.ceil,
    "floor": torch.floor,
    # mish(x) = x * tanh(softplus(x)) = x * tanh(log1p(exp(x)))
    "mish": lambda x: x * torch.tanh(torch.nn.functional.softplus(x)),
    # swish(x) = x * sigmoid(x)
    "swish": lambda x: x * torch.sigmoid(x),
    "soft_plus": torch.nn.functional.softplus,
    # soft_sign(x) = x / (1 + |x|)
    "soft_sign": lambda x: x / (1.0 + torch.abs(x)),
}

_BINARY_TORCH_MAP = {
    "add": torch.add,
    "mul": torch.mul,
    "max": torch.maximum,
    "min": torch.minimum,
}


# ---------------------------------------------------------------------------
# Mode helper — replicate the cuTensor mode normalization so that contraction
# baselines accept the same call signature as ``CuTensorContraction`` etc.
# ---------------------------------------------------------------------------
def _normalize_modes(modes, ndim):
    if modes is None:
        return tuple(range(ndim))
    return tuple(modes)


def _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d):
    extents = {}
    for tensor, modes in ((a, mode_a), (b, mode_b)):
        for extent, mode in zip(tensor.shape, modes):
            previous = extents.get(mode)
            if previous is not None and previous != extent:
                raise ValueError(f"incompatible extent for mode {mode}: {previous} vs {extent}")
            extents[mode] = extent
    return tuple(extents[mode] for mode in mode_d)


def _einsum_subscript(mode_a, mode_b, mode_d):
    """Build an einsum subscript for ``out = a @ b`` with arbitrary modes.

    Modes are integers; map them to lowercase letters so einsum can parse them.
    The contracted modes are those present in both mode_a and mode_b but not
    in mode_d.
    """
    all_modes = sorted(set(mode_a) | set(mode_b) | set(mode_d))
    if len(all_modes) > 26:
        raise ValueError("too many modes for einsum subscript (max 26)")
    mode_to_letter = {m: chr(ord("a") + i) for i, m in enumerate(all_modes)}
    a_sub = "".join(mode_to_letter[m] for m in mode_a)
    b_sub = "".join(mode_to_letter[m] for m in mode_b)
    d_sub = "".join(mode_to_letter[m] for m in mode_d)
    return f"{a_sub},{b_sub}->{d_sub}"


# ---------------------------------------------------------------------------
# Baseline classes
# ---------------------------------------------------------------------------
class TorchNpuUnary:
    """torch_npu-aten baseline for a single unary pointwise op."""

    def __init__(self, op_slug: str, dtype=torch.float32):
        self.op_slug = op_slug
        self.dtype = dtype
        self.fn = _UNARY_TORCH_MAP.get(op_slug)
        if self.fn is None:
            raise ValueError(f"unsupported unary op slug for torch_npu baseline: {op_slug}")
        self.signature = None

    def prepare(self, x):
        # No plan/descriptor caching needed — torch_npu dispatches the aclnn
        # kernel on each call. We just remember the input signature so the
        # benchmark framework's cache-check pattern stays satisfied.
        self.signature = (x.dtype, tuple(x.shape), tuple(x.stride()))

    def build_kernel_callable(self, x, alpha=1.0) -> Callable[[], torch.Tensor]:
        self.prepare(x)

        def run_kernel():
            return self.fn(x)

        return run_kernel

    def __call__(self, x, alpha=1.0):
        self.prepare(x)
        return self.fn(x)


class TorchNpuBinary:
    """torch_npu-aten baseline for a single binary pointwise op."""

    def __init__(self, op_slug: str, dtype=torch.float32):
        self.op_slug = op_slug
        self.dtype = dtype
        self.fn = _BINARY_TORCH_MAP.get(op_slug)
        if self.fn is None:
            raise ValueError(f"unsupported binary op slug for torch_npu baseline: {op_slug}")
        self.signature = None

    def prepare(self, x, y):
        self.signature = (
            x.dtype, tuple(x.shape), tuple(x.stride()),
            y.dtype, tuple(y.shape), tuple(y.stride()),
        )

    def build_kernel_callable(self, x, y, alpha=1.0, gamma=1.0) -> Callable[[], torch.Tensor]:
        self.prepare(x, y)

        def run_kernel():
            return self.fn(x, y)

        return run_kernel

    def __call__(self, x, y, alpha=1.0, gamma=1.0):
        self.prepare(x, y)
        return self.fn(x, y)


# ---------------------------------------------------------------------------
# Contraction baselines
# ---------------------------------------------------------------------------
class CuTensorContraction:
    """torch_npu-aten baseline for binary tensor contraction.

    Implements the same call surface as ``flagtensor.cutensor.CuTensorContraction``
    using ``torch.einsum`` + ``torch.matmul`` (both backed by CANN aclnn on
    Ascend). Modes are arbitrary integer tuples.
    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.signature = None

    def prepare(self, a, b, c=None, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        if mode_d is None:
            mode_d = tuple(m for m in mode_a + mode_b if m not in set(mode_a).intersection(mode_b))
        mode_d = tuple(mode_d)
        output_shape = _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d)
        if c is None:
            c = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
        if mode_c is None:
            mode_c = mode_d
        mode_c = tuple(mode_c)
        if out is None:
            out = torch.empty(output_shape, device=a.device, dtype=a.dtype)
        self.signature = (
            a.dtype, tuple(a.shape), mode_a,
            tuple(b.shape), mode_b,
            tuple(c.shape), mode_c,
            tuple(out.shape), mode_d,
        )
        return a, b, c, out, mode_a, mode_b, mode_c, mode_d

    def __call__(self, a, b, c=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        a, b, c, output, mode_a, mode_b, mode_c, mode_d = self.prepare(
            a, b, c=c, mode_a=mode_a, mode_b=mode_b, mode_c=mode_c, mode_d=mode_d, out=out,
        )
        subscript = _einsum_subscript(mode_a, mode_b, mode_d)
        # Use float32 accumulation for low-precision dtypes to mirror cuTensor
        # compute descriptors. The result is then cast back to the input dtype.
        compute_dtype = a.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            result = torch.einsum(subscript, a.float(), b.float()).to(a.dtype)
        else:
            result = torch.einsum(subscript, a, b)
        result = alpha * result + beta * c
        output.copy_(result)
        return output

    def __del__(self):
        pass


class CuTensorContractionTrinary:
    """torch_npu-aten baseline for trinary tensor contraction.

    Decomposes the trinary contraction into two binary contractions, exactly
    like ``flagtensor.cutensor.CuTensorContractionTrinary`` does on top of
    cuTensor.
    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.first = CuTensorContraction(dtype=dtype)
        self.second = CuTensorContraction(dtype=dtype)

    def __call__(self, a, b, c, d=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, mode_e=None, out=None):
        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        mode_c = _normalize_modes(mode_c, c.ndim)
        contracted_modes = (set(mode_a) & set(mode_b)) | (set(mode_a) & set(mode_c)) | (set(mode_b) & set(mode_c))
        if mode_e is None:
            mode_e = tuple(m for m in mode_a + mode_b + mode_c if m not in contracted_modes)
        mode_e = tuple(mode_e)
        # Infer output shape for the addend d.
        extents = {}
        for tensor, modes in ((a, mode_a), (b, mode_b), (c, mode_c)):
            for extent, mode in zip(tensor.shape, modes):
                extents[mode] = extent
        output_shape = tuple(extents[m] for m in mode_e)
        if d is None:
            d = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
        if mode_d is None:
            mode_d = mode_e
        shared_modes = tuple(m for m in mode_a if m in set(mode_b) and m not in mode_e)
        intermediate_modes = tuple(m for m in mode_a + mode_b if m not in shared_modes)
        temp = self.first(
            a, b, c=None, alpha=1.0, beta=0.0,
            mode_a=mode_a, mode_b=mode_b,
            mode_c=intermediate_modes, mode_d=intermediate_modes,
        )
        return self.second(
            temp, c, c=d, alpha=alpha, beta=beta,
            mode_a=intermediate_modes, mode_b=mode_c,
            mode_c=mode_d, mode_d=mode_e, out=out,
        )

    def __del__(self):
        pass


# ---------------------------------------------------------------------------
# Elementwise trinary baseline (cuTensor's CuTensorTrinary equivalent)
# ---------------------------------------------------------------------------
class CuTensorTrinary:
    """torch_npu-aten baseline for elementwise trinary ops.

    cuTensor's trinary elementwise computes ``D = f_abc(f_ab(f_a(A), f_b(B)), f_c(C))``
    on the GPU. We mirror that with chained torch aten ops, all dispatched to
    CANN aclnn on Ascend.

    The constructor accepts op codes either as integers (cuTensor constants
    like ``CUTENSOR_OP_ADD``) or as strings (e.g. ``"add"``) for symmetry
    with the ``elementwise_trinary`` python API.
    """

    def __init__(self, op_ab, op_abc, op_a=1, op_b=1, op_c=1, dtype=torch.float32):
        from flagtensor.cutensor import (
            CUTENSOR_OP_ADD, CUTENSOR_OP_MUL, CUTENSOR_OP_MAX, CUTENSOR_OP_MIN,
            CUTENSOR_OP_IDENTITY,
            BINARY_OPERATOR_MAP,
            UNARY_OPERATOR_MAP,
        )
        _BIN_CODE_TO_FN = {
            CUTENSOR_OP_ADD: torch.add,
            CUTENSOR_OP_MUL: torch.mul,
            CUTENSOR_OP_MAX: torch.maximum,
            CUTENSOR_OP_MIN: torch.minimum,
        }
        _UNARY_CODE_TO_FN = {
            CUTENSOR_OP_IDENTITY: lambda t: t,
        }
        # String op names → integer codes via cuTensor's existing tables.
        if isinstance(op_ab, str):
            op_ab = BINARY_OPERATOR_MAP.get(op_ab, CUTENSOR_OP_ADD)
        if isinstance(op_abc, str):
            op_abc = BINARY_OPERATOR_MAP.get(op_abc, CUTENSOR_OP_ADD)
        if isinstance(op_a, str):
            op_a = UNARY_OPERATOR_MAP.get(op_a, CUTENSOR_OP_IDENTITY)
        if isinstance(op_b, str):
            op_b = UNARY_OPERATOR_MAP.get(op_b, CUTENSOR_OP_IDENTITY)
        if isinstance(op_c, str):
            op_c = UNARY_OPERATOR_MAP.get(op_c, CUTENSOR_OP_IDENTITY)
        self.fn_ab = _BIN_CODE_TO_FN.get(op_ab, torch.add)
        self.fn_abc = _BIN_CODE_TO_FN.get(op_abc, torch.add)
        # Per-input unary ops. Resolve through the same _UNARY_TORCH_MAP the
        # unary baseline uses so we cover log/neg/sqrt/etc. consistently.
        self.fn_a = _resolve_unary_code(op_a)
        self.fn_b = _resolve_unary_code(op_b)
        self.fn_c = _resolve_unary_code(op_c)
        self.dtype = dtype

    def __call__(
        self, x, y, z, alpha=1.0, beta=1.0, gamma=1.0, out=None,
        mode_a=None, mode_b=None, mode_c=None, mode_d=None,
    ):
        # When modes are provided, permute each input so its modes align with
        # the output mode_d. Modes that exist in an input but not in mode_d
        # are treated as broadcast dimensions and kept in their natural order.
        def _align(t, mode_t):
            if mode_t is None or mode_d is None or t is None:
                return t
            mode_t = tuple(mode_t)
            # Build the permutation that reorders mode_t to match mode_d for
            # the modes that exist in both, keeping extra modes at the end.
            common = [m for m in mode_d if m in mode_t]
            extra = [m for m in mode_t if m not in set(mode_d)]
            target_order = common + extra
            if not target_order or tuple(target_order) == mode_t:
                # Already aligned (or nothing to do); still need to broadcast
                # to the output shape later.
                pass
            else:
                perm = [mode_t.index(m) for m in target_order]
                t = t.permute(*perm).contiguous()
            # Broadcast to full output extent layout: build shape that has 1
            # for modes not present in this input.
            return t
        xa = _align(x, mode_a)
        yb = _align(y, mode_b)
        zc = _align(z, mode_c)
        xa = self.fn_a(xa) if self.fn_a else xa
        yb = self.fn_b(yb) if self.fn_b else yb
        zc = self.fn_c(zc) if self.fn_c else zc
        # torch aten broadcasting handles the rest.
        result = self.fn_abc(self.fn_ab(alpha * xa, beta * yb), gamma * zc)
        if out is None:
            return result
        out.copy_(result)
        return out

    def __del__(self):
        pass


def _resolve_unary_code(code):
    """Map a cuTensor unary op code (int) to a torch callable."""
    from flagtensor.cutensor import UNARY_OPERATOR_MAP
    # Reverse-lookup: int code → string slug → torch fn
    for slug, value in UNARY_OPERATOR_MAP.items():
        if value == code:
            return _UNARY_TORCH_MAP.get(slug)
    return None


# ElementwiseTrinary wrapper used by benchmark_core resolution.
class CuTensorElementwiseTrinary(CuTensorTrinary):
    pass


# Block-sparse contraction baseline. The cuTensor version falls back to a
# dense contraction when cuTensor's block-sparse API is unavailable, so on
# Ascend we simply re-use the existing ``flagtensor.cutensor.BlockSparseTensorContraction``
# class which already does dense fallback when cuTensor is missing.
def _import_block_sparse_baseline():
    from flagtensor.cutensor import BlockSparseTensorContraction as _BSPC
    return _BSPC


# Lazy proxy so attribute access on the module resolves to the real class.
class _BlockSparseTensorContractionProxy:
    def __call__(self, *args, **kwargs):
        return _import_block_sparse_baseline()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_import_block_sparse_baseline(), name)


BlockSparseTensorContraction = _BlockSparseTensorContractionProxy()


# ---------------------------------------------------------------------------
# Auto-generated class aliases mirroring the cuTensor naming convention
# ---------------------------------------------------------------------------
# cuTensor exposes ``CuTensorAbs``, ``CuTensorSqrt`` ... so the benchmark
# core resolves baselines by ``CuTensor{SlugCamelCase}``. We expose
# identically-named subclasses here so the same resolution logic works on
# Ascend. The benchmark core imports whichever module is loaded — when
# running on Ascend the test fixture monkey-patches the cutensor module's
# class lookup to fall back to this module (see ``benchmark_core.py``).

def _camel(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


_UNARY_CLASSES = {}
_BINARY_CLASSES = {}


def _make_unary_class(slug: str):
    cls = type(f"CuTensor{_camel(slug)}", (TorchNpuUnary,), {
        "__init__": lambda self, dtype=torch.float32, _slug=slug: TorchNpuUnary.__init__(self, _slug, dtype=dtype),
    })
    return cls


def _make_binary_class(slug: str):
    cls = type(f"CuTensor{_camel(slug)}", (TorchNpuBinary,), {
        "__init__": lambda self, dtype=torch.float32, _slug=slug: TorchNpuBinary.__init__(self, _slug, dtype=dtype),
    })
    return cls


for _slug in _UNARY_TORCH_MAP:
    _UNARY_CLASSES[f"CuTensor{_camel(_slug)}"] = _make_unary_class(_slug)
for _slug in _BINARY_TORCH_MAP:
    _BINARY_CLASSES[f"CuTensor{_camel(_slug)}"] = _make_binary_class(_slug)


# Expose them as module-level attributes so ``getattr(module, class_name)``
# works exactly like it does for ``flagtensor.cutensor``.
globals().update(_UNARY_CLASSES)
globals().update(_BINARY_CLASSES)


__all__ = (
    [
        "TorchNpuUnary", "TorchNpuBinary", "torch_npu_available",
        "CuTensorContraction", "CuTensorContractionTrinary",
        "CuTensorTrinary", "CuTensorElementwiseTrinary",
        "BlockSparseTensorContraction",
    ]
    + list(_UNARY_CLASSES.keys())
    + list(_BINARY_CLASSES.keys())
)
