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

"""PyTorch-native baseline implementations for FlagTensor benchmarks.

These classes mirror the public interface of the ``CuTensor*`` baseline
classes in :mod:`flagtensor.cutensor`. They are used as a fallback when
``libcutensor`` is unavailable (e.g. on non-NVIDIA accelerators such as
the Alibaba PPU), so that benchmarks can still produce a meaningful
speedup number against a vendor-supplied PyTorch native kernel path.

Baseline selection rationale (PPU)
----------------------------------
On the Alibaba PPU (PPU-ZW810E, sm80, CUDA-compatible SDK at
``/usr/local/PPU_SDK``) the following was verified:

* **No cuTensor equivalent exists.** Neither ``libcutensor`` nor a PPU-native
  ``libacTensor`` ships with the SDK. The PPU-native math libraries are
  ``libacblas`` (BLAS), ``libacdnn`` (DNN), ``libacsparse`` (sparse),
  ``libacfft`` (FFT), ``libacsolver`` (solver) and ``libacrand`` (RNG) —
  none of them exposes cuTensor's generalized tensor-contraction /
  elementwise-trinary API.

* **Direct cuBLAS ctypes is not viable on PPU.** ``libcublas.so`` provided
  by the PPU SDK is a thin shim that forwards every ``cublasXxx`` symbol
  to ``libacblas.so``. Direct ``ctypes.CDLL('libcublas.so')`` calls fail
  inside acblas with ``HGGC_ERROR_ILLEGAL_ADDRESS`` because the shim
  depends on device-context setup that PyTorch performs internally but
  that is not trivially reproducible from Python.

* **PyTorch native ops DO use vendor-optimized kernels.** Profiling
  ``torch.matmul`` / ``torch.addmm`` on PPU shows acblas generating
  RTC-compiled GEMM kernels (e.g. ``gemm_*_tile128x256x64*.bin`` cached
  under ``/usr/local/PPU_SDK/rtccache/``). Elementwise ops likewise
  dispatch to acdnn/acsfu kernels via the standard aten dispatcher.

* **This mirrors the FlagGems convention.** FlagGems (the FlagOS sister
  project) uses ``torch.*`` as the baseline for pointwise/unary ops that
  lack a dedicated vendor-library equivalent; the same pattern applies
  here for the entire FlagTensor op surface.

Therefore ``torch.*`` ops, accessed through this module, are the
correct, vendor-backed baseline on PPU and any other CUDA-compatible
accelerator without cuTensor.

Every class implements the same minimal protocol that the benchmark
harness relies on:

    * ``prepare(*inputs)``            — validate/cache inputs
    * ``__call__(*inputs)``           — operator-level execution
    * ``build_kernel_callable(*inputs)`` — returns a zero-arg callable
      suitable for ``triton.testing.do_bench`` (kernel-mode timing)

The cuTensor semantics that are preserved by this module:

    * Unary  :  ``y = alpha * op(x)``
    * Binary :  ``y = op_AB(alpha * op_A(x), gamma * op_C(y))``
                (with ``op_A = op_C = IDENTITY`` for the binary family)
    * Trinary:  ``y = op_ABC(op_AB(alpha * op_A(x), beta * op_B(y)),
                             gamma * op_C(z))``
    * Contraction (GETT):
                ``d = alpha * einsum(a, b, modes) + beta * c``
    * ContractionTrinary:
                ``e = alpha * (contraction(contraction(a, b), c)) + beta * d``
    * BlockSparseContraction:
                dense ``alpha * matmul(a_dense, b_dense) + beta * c_dense``
                masked by the output block-sparsity pattern.
"""

from typing import Callable, Optional

import torch

from .cutensor import (
    BINARY_OPERATOR_MAP,
    CUTENSOR_OP_ABS,
    CUTENSOR_OP_ACOS,
    CUTENSOR_OP_ACOSH,
    CUTENSOR_OP_ADD,
    CUTENSOR_OP_ASIN,
    CUTENSOR_OP_ASINH,
    CUTENSOR_OP_ATAN,
    CUTENSOR_OP_ATANH,
    CUTENSOR_OP_CEIL,
    CUTENSOR_OP_CONJ,
    CUTENSOR_OP_COS,
    CUTENSOR_OP_COSH,
    CUTENSOR_OP_EXP,
    CUTENSOR_OP_FLOOR,
    CUTENSOR_OP_IDENTITY,
    CUTENSOR_OP_LOG,
    CUTENSOR_OP_MAX,
    CUTENSOR_OP_MIN,
    CUTENSOR_OP_MISH,
    CUTENSOR_OP_MUL,
    CUTENSOR_OP_NEG,
    CUTENSOR_OP_RCP,
    CUTENSOR_OP_RELU,
    CUTENSOR_OP_SIGMOID,
    CUTENSOR_OP_SIN,
    CUTENSOR_OP_SINH,
    CUTENSOR_OP_SOFT_PLUS,
    CUTENSOR_OP_SOFT_SIGN,
    CUTENSOR_OP_SQRT,
    CUTENSOR_OP_SWISH,
    CUTENSOR_OP_TAN,
    CUTENSOR_OP_TANH,
    UNARY_OPERATOR_MAP,
    BlockSparseTensor,
    _infer_contraction_output_shape,
    _infer_output_modes,
    _infer_output_shape,
    _infer_trinary_contraction_output_shape,
    _normalize_modes,
    _resolve_operator,
    _validate_contraction_addend,
    _validate_trinary_contraction_addend,
)


# ---------------------------------------------------------------------------
# CUTENSOR_OP_* → torch function mapping
# ---------------------------------------------------------------------------
def _torch_mish(x):
    # mish(x) = x * tanh(softplus(x))
    return x * torch.tanh(torch.nn.functional.softplus(x))


def _torch_swish(x):
    # swish(x) = x * sigmoid(x)  (a.k.a. SiLU)
    return torch.nn.functional.silu(x)


def _torch_soft_plus(x):
    return torch.nn.functional.softplus(x)


def _torch_soft_sign(x):
    return x / (1.0 + torch.abs(x))


def _torch_rcp(x):
    return torch.reciprocal(x)


def _torch_conj(x):
    return torch.conj(x)


_UNARY_TORCH_FN = {
    CUTENSOR_OP_IDENTITY: lambda x: x,
    CUTENSOR_OP_SQRT: torch.sqrt,
    CUTENSOR_OP_RELU: torch.relu,
    CUTENSOR_OP_CONJ: _torch_conj,
    CUTENSOR_OP_RCP: _torch_rcp,
    CUTENSOR_OP_SIGMOID: torch.sigmoid,
    CUTENSOR_OP_TANH: torch.tanh,
    CUTENSOR_OP_ABS: torch.abs,
    CUTENSOR_OP_EXP: torch.exp,
    CUTENSOR_OP_LOG: torch.log,
    CUTENSOR_OP_NEG: torch.neg,
    CUTENSOR_OP_SIN: torch.sin,
    CUTENSOR_OP_COS: torch.cos,
    CUTENSOR_OP_TAN: torch.tan,
    CUTENSOR_OP_SINH: torch.sinh,
    CUTENSOR_OP_COSH: torch.cosh,
    CUTENSOR_OP_ASIN: torch.asin,
    CUTENSOR_OP_ACOS: torch.acos,
    CUTENSOR_OP_ATAN: torch.atan,
    CUTENSOR_OP_ASINH: torch.asinh,
    CUTENSOR_OP_ACOSH: torch.acosh,
    CUTENSOR_OP_ATANH: torch.atanh,
    CUTENSOR_OP_CEIL: torch.ceil,
    CUTENSOR_OP_FLOOR: torch.floor,
    CUTENSOR_OP_MISH: _torch_mish,
    CUTENSOR_OP_SWISH: _torch_swish,
    CUTENSOR_OP_SOFT_PLUS: _torch_soft_plus,
    CUTENSOR_OP_SOFT_SIGN: _torch_soft_sign,
}


def _binary_apply(op, x, y):
    if op == CUTENSOR_OP_ADD:
        return x + y
    if op == CUTENSOR_OP_MUL:
        return x * y
    if op == CUTENSOR_OP_MAX:
        return torch.maximum(x, y)
    if op == CUTENSOR_OP_MIN:
        return torch.minimum(x, y)
    raise ValueError(f"unsupported binary op: {op}")


def _unary_apply(op, x):
    fn = _UNARY_TORCH_FN.get(op)
    if fn is None:
        raise ValueError(f"unsupported unary op: {op}")
    return fn(x)


def _permute_to_mode(tensor, from_mode, to_mode):
    """Permute ``tensor`` from ``from_mode`` layout to ``to_mode`` layout.

    ``from_mode[i]`` is the canonical label of the i-th dim of ``tensor``.
    The returned tensor has its i-th dim labelled ``to_mode[i]``.
    """
    if from_mode is None or to_mode is None:
        return tensor
    if tuple(from_mode) == tuple(to_mode):
        return tensor
    if len(from_mode) != len(to_mode):
        # Different ranks: broadcast by repeating the missing modes (rare).
        return tensor
    perm = [from_mode.index(m) for m in to_mode]
    return tensor.permute(*perm)


# ---------------------------------------------------------------------------
# Unary baseline (mirrors CuTensorUnary)
# ---------------------------------------------------------------------------
class TorchUnaryBaseline:
    def __init__(self, op, dtype=torch.float32):
        self.op = op
        self.dtype = dtype
        self.signature = None

    def prepare(self, x):
        if not x.is_cuda:
            raise ValueError("input tensor must be on CUDA")
        self.signature = (x.dtype, tuple(x.shape), tuple(x.stride()))

    def build_kernel_callable(self, x, alpha=1.0):
        self.prepare(x)
        y = torch.empty_like(x)

        def run_kernel():
            torch.mul(_unary_apply(self.op, x), alpha, out=y)
            return y

        return run_kernel

    def __call__(self, x, alpha=1.0):
        self.prepare(x)
        return alpha * _unary_apply(self.op, x)


# ---------------------------------------------------------------------------
# Binary baseline (mirrors CuTensorBinary)
# ---------------------------------------------------------------------------
class TorchBinaryBaseline:
    def __init__(self, op, dtype=torch.float32):
        self.op = op
        self.dtype = dtype
        self.signature = None

    def prepare(self, x, y):
        if not x.is_cuda or not y.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if x.dtype != y.dtype:
            raise TypeError("input tensors must have the same dtype")
        if x.shape != y.shape:
            raise ValueError("input tensors must have the same shape")
        if x.stride() != y.stride():
            raise ValueError("input tensors must have the same stride")
        self.signature = (
            x.dtype, tuple(x.shape), tuple(x.stride()),
            y.dtype, tuple(y.shape), tuple(y.stride()),
        )

    def _compute(self, x, y, alpha, gamma):
        # D = op_AB(alpha * op_A(x), gamma * op_C(y))   with op_A=op_C=IDENTITY
        return _binary_apply(self.op, alpha * x, gamma * y)

    def build_kernel_callable(self, x, y, alpha=1.0, gamma=1.0):
        self.prepare(x, y)

        def run_kernel():
            return self._compute(x, y, alpha, gamma)

        return run_kernel

    def __call__(self, x, y, alpha=1.0, gamma=1.0):
        self.prepare(x, y)
        return self._compute(x, y, alpha, gamma)


# ---------------------------------------------------------------------------
# Trinary elementwise baseline (mirrors CuTensorTrinary)
# ---------------------------------------------------------------------------
class TorchTrinaryBaseline:
    def __init__(self, op_ab, op_abc, op_a=CUTENSOR_OP_IDENTITY,
                 op_b=CUTENSOR_OP_IDENTITY, op_c=CUTENSOR_OP_IDENTITY,
                 dtype=torch.float32):
        self.op_ab = _resolve_operator(op_ab, BINARY_OPERATOR_MAP, "binary")
        self.op_abc = _resolve_operator(op_abc, BINARY_OPERATOR_MAP, "binary")
        self.op_a = _resolve_operator(op_a, UNARY_OPERATOR_MAP, "unary")
        self.op_b = _resolve_operator(op_b, UNARY_OPERATOR_MAP, "unary")
        self.op_c = _resolve_operator(op_c, UNARY_OPERATOR_MAP, "unary")
        self.dtype = dtype
        self.signature = None

    def _compute(self, x, y, z, alpha, beta, gamma,
                 mode_a, mode_b, mode_c, mode_d):
        mode_a = _normalize_modes(mode_a, x.ndim)
        mode_b = _normalize_modes(mode_b, y.ndim)
        mode_c = _normalize_modes(mode_c, z.ndim)
        inferred_mode_d = _infer_output_modes(mode_a, mode_b, mode_c, mode_d)
        if tuple(inferred_mode_d) != tuple(mode_c):
            raise ValueError(
                "trinary elementwise currently requires mode_d to be identical to mode_c"
            )
        mode_d = tuple(mode_c)

        # Bring x, y into the output (mode_d) layout; z is already in mode_c == mode_d
        x_perm = _permute_to_mode(x, mode_a, mode_d)
        y_perm = _permute_to_mode(y, mode_b, mode_d)

        ax = alpha * _unary_apply(self.op_a, x_perm)
        by = beta * _unary_apply(self.op_b, y_perm)
        gz = gamma * _unary_apply(self.op_c, z)

        ab = _binary_apply(self.op_ab, ax, by)
        return _binary_apply(self.op_abc, ab, gz)

    def prepare(self, x, y, z, mode_a=None, mode_b=None, mode_c=None,
                mode_d=None, out=None):
        if not x.is_cuda or not y.is_cuda or not z.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if x.dtype != y.dtype or x.dtype != z.dtype:
            raise TypeError("input tensors must have the same dtype")
        self.signature = (
            x.dtype, tuple(x.shape), tuple(x.stride()),
            tuple(mode_a) if mode_a else None,
            y.dtype, tuple(y.shape), tuple(y.stride()),
            tuple(mode_b) if mode_b else None,
            z.dtype, tuple(z.shape), tuple(z.stride()),
            tuple(mode_c) if mode_c else None,
            tuple(mode_d) if mode_d else None,
        )
        return mode_a, mode_b, mode_c, mode_d

    def build_kernel_callable(self, x, y, z, alpha=1.0, beta=1.0, gamma=1.0,
                              mode_a=None, mode_b=None, mode_c=None,
                              mode_d=None, out=None):
        self.prepare(x, y, z, mode_a, mode_b, mode_c, mode_d, out)
        output = out if out is not None else torch.empty_like(z)

        def run_kernel():
            result = self._compute(x, y, z, alpha, beta, gamma,
                                   mode_a, mode_b, mode_c, mode_d)
            output.copy_(result)
            return output

        return run_kernel

    def __call__(self, x, y, z, alpha=1.0, beta=1.0, gamma=1.0,
                 mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        self.prepare(x, y, z, mode_a, mode_b, mode_c, mode_d, out)
        result = self._compute(x, y, z, alpha, beta, gamma,
                               mode_a, mode_b, mode_c, mode_d)
        if out is not None:
            out.copy_(result)
            return out
        return result


# ---------------------------------------------------------------------------
# Contraction baseline (mirrors CuTensorContraction)
# ---------------------------------------------------------------------------
class TorchContractionBaseline:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.signature = None
        self._equation = None
        self._mode_a = None
        self._mode_b = None
        self._mode_c = None
        self._mode_d = None

    @staticmethod
    def _einsum_equation(mode_a, mode_b, mode_d):
        unique_modes = []
        for m in list(mode_a) + list(mode_b) + list(mode_d):
            if m not in unique_modes:
                unique_modes.append(m)
        if len(unique_modes) > 52:
            raise ValueError("too many unique modes for torch.einsum")
        letters = [
            chr(ord("a") + i) if i < 26 else chr(ord("A") + i - 26)
            for i in range(len(unique_modes))
        ]
        m2l = {m: l for m, l in zip(unique_modes, letters)}
        a_sub = "".join(m2l[m] for m in mode_a)
        b_sub = "".join(m2l[m] for m in mode_b)
        d_sub = "".join(m2l[m] for m in mode_d)
        return f"{a_sub},{b_sub}->{d_sub}"

    def prepare(self, a, b, c=None, mode_a=None, mode_b=None,
                mode_c=None, mode_d=None, out=None):
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if a.dtype != b.dtype:
            raise TypeError("input tensors must have the same dtype")

        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        mode_d = (
            tuple(mode_d) if mode_d is not None
            else tuple(
                m for m in mode_a + mode_b
                if m not in set(mode_a).intersection(mode_b)
            )
        )
        if len(set(mode_d)) != len(mode_d):
            raise ValueError("each output mode may appear at most once")
        output_shape = _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d)

        if c is None:
            c = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
        else:
            if not c.is_cuda:
                raise ValueError("addend tensor must be on CUDA")
            if c.dtype != a.dtype:
                raise TypeError("addend tensor must have the same dtype as inputs")
        mode_c = _validate_contraction_addend(
            c, mode_c if mode_c is not None else mode_d, mode_d, output_shape
        )

        if out is not None:
            if not out.is_cuda:
                raise ValueError("output tensor must be on CUDA")
            if out.dtype != a.dtype:
                raise TypeError("output tensor must have the same dtype as inputs")
            if tuple(out.shape) != tuple(output_shape):
                raise ValueError(
                    f"output tensor shape mismatch: expected {output_shape}, "
                    f"got {tuple(out.shape)}"
                )
        else:
            out = torch.empty(output_shape, device=a.device, dtype=a.dtype)

        self._mode_a = mode_a
        self._mode_b = mode_b
        self._mode_c = mode_c
        self._mode_d = mode_d
        self._equation = self._einsum_equation(mode_a, mode_b, mode_d)
        self.signature = (
            a.dtype, tuple(a.shape), tuple(a.stride()), tuple(mode_a),
            tuple(b.shape), tuple(b.stride()), tuple(mode_b),
            tuple(c.shape), tuple(c.stride()), tuple(mode_c),
            tuple(mode_d),
        )
        return a, b, c, out, mode_a, mode_b, mode_c, mode_d

    def _compute(self, a, b, c, alpha, beta):
        temp = torch.einsum(self._equation, a, b)
        if beta == 0.0:
            return alpha * temp
        return alpha * temp + beta * c

    def __call__(self, a, b, c=None, alpha=1.0, beta=0.0,
                 mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        a, b, c, out, mode_a, mode_b, mode_c, mode_d = self.prepare(
            a, b, c=c, mode_a=mode_a, mode_b=mode_b,
            mode_c=mode_c, mode_d=mode_d, out=out,
        )
        result = self._compute(a, b, c, alpha, beta)
        if out is not None:
            out.copy_(result)
            return out
        return result


# ---------------------------------------------------------------------------
# Trinary contraction baseline (mirrors CuTensorContractionTrinary)
# ---------------------------------------------------------------------------
class TorchContractionTrinaryBaseline:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.first = TorchContractionBaseline(dtype=dtype)
        self.second = TorchContractionBaseline(dtype=dtype)

    def __call__(self, a, b, c, d=None, alpha=1.0, beta=0.0,
                 mode_a=None, mode_b=None, mode_c=None,
                 mode_d=None, mode_e=None, out=None):
        if not a.is_cuda or not b.is_cuda or not c.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise TypeError("input tensors must have the same dtype")

        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        mode_c = _normalize_modes(mode_c, c.ndim)
        contracted_modes = (
            (set(mode_a) & set(mode_b))
            | (set(mode_a) & set(mode_c))
            | (set(mode_b) & set(mode_c))
        )
        mode_e = (
            tuple(mode_e) if mode_e is not None
            else tuple(m for m in mode_a + mode_b + mode_c
                       if m not in contracted_modes)
        )
        if len(set(mode_e)) != len(mode_e):
            raise ValueError("each output mode may appear at most once")
        output_shape = _infer_trinary_contraction_output_shape(
            a, mode_a, b, mode_b, c, mode_c, mode_e
        )

        if d is None:
            d = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
        else:
            if not d.is_cuda:
                raise ValueError("addend tensor must be on CUDA")
            if d.dtype != a.dtype:
                raise TypeError("addend tensor must have the same dtype as inputs")
        mode_d = _validate_trinary_contraction_addend(
            d, mode_d if mode_d is not None else mode_e, mode_e, output_shape
        )

        shared_modes = tuple(
            m for m in mode_a if m in set(mode_b) and m not in mode_e
        )
        intermediate_modes = tuple(
            m for m in mode_a + mode_b if m not in shared_modes
        )
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


# ---------------------------------------------------------------------------
# Block-sparse contraction baseline (mirrors CuTensorBlockSparseContraction)
# ---------------------------------------------------------------------------
class TorchBlockSparseContractionBaseline:
    """Dense matmul baseline masked by the output block-sparsity pattern.

    This matches the cuTensor block-sparse contraction semantics: only the
    output blocks listed in ``c.descriptor.nonzero_coordinates`` are
    materialised; everywhere else is zero.
    """

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    @staticmethod
    def _sparsity_mask(tensor: BlockSparseTensor) -> torch.Tensor:
        device = tensor.device
        mask = torch.zeros(tensor.shape, device=device, dtype=torch.float32)
        ndim = tensor.ndim
        offsets = []
        for mode in range(ndim):
            mode_offsets = [0]
            for extent in tensor.descriptor.section_extents[mode]:
                mode_offsets.append(mode_offsets[-1] + extent)
            offsets.append(mode_offsets)
        for coord in tensor.descriptor.canonical_nonzero_coordinates:
            slices = tuple(
                slice(offsets[mode][idx], offsets[mode][idx + 1])
                for mode, idx in enumerate(coord)
            )
            mask[slices] = 1.0
        return mask

    def __call__(self, a, mode_a, b, mode_b, c, mode_c, mode_d,
                 alpha=1.0, beta=0.0):
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise TypeError("block-sparse tensors must share the same dtype")

        dense_a = a.to_dense()
        dense_b = b.to_dense()
        dense_c = c.to_dense()

        # 2D contraction is the only supported case in the cuTensor path;
        # use torch.matmul directly. mode layout is (0,1),(1,2),(0,2).
        out = alpha * torch.matmul(dense_a, dense_b) + beta * dense_c
        mask = self._sparsity_mask(c)
        out = out * mask.to(out.dtype)
        return out


__all__ = [
    "TorchUnaryBaseline",
    "TorchBinaryBaseline",
    "TorchTrinaryBaseline",
    "TorchContractionBaseline",
    "TorchContractionTrinaryBaseline",
    "TorchBlockSparseContractionBaseline",
]
