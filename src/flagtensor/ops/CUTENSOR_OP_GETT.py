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

from flagtensor.cutensor import _normalize_modes
from flagtensor.runtime import is_on_accelerator as _is_on_accelerator

_GETT_PREPARED_LAUNCHER_CACHE = {}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "GROUP_M": 8}, num_warps=2, num_stages=5),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 4}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K", "TRANS_A", "TRANS_B"],
)
@triton.jit
def _gett_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_dm,
    stride_dn,
    alpha,
    beta,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    HAS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS_A:
        a_ptrs = a_ptr + offs_k[:, None] * stride_am + offs_m[None, :] * stride_ak
    else:
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    if TRANS_B:
        b_ptrs = b_ptr + offs_n[:, None] * stride_bk + offs_k[None, :] * stride_bn
    else:
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        if TRANS_A:
            a = tl.load(a_ptrs, mask=k_mask[:, None] & (offs_m[None, :] < M), other=0.0)
            a = tl.trans(a)
        else:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        if TRANS_B:
            b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
            b = tl.trans(b)
        else:
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        if TRANS_A:
            a_ptrs += BLOCK_K * stride_am
        else:
            a_ptrs += BLOCK_K * stride_ak
        if TRANS_B:
            b_ptrs += BLOCK_K * stride_bn
        else:
            b_ptrs += BLOCK_K * stride_bk

    acc = alpha * acc
    if HAS_C:
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc = acc + beta * c.to(tl.float32)

    d_ptrs = d_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    d_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(d_ptrs, acc, mask=d_mask)


def _is_default_2d_gett_case(a, b, c, mode_a, mode_b, mode_c, mode_d):
    if a.ndim != 2 or b.ndim != 2:
        return False
    if c is not None and c.ndim != 2:
        return False
    mode_a = tuple(mode_a) if mode_a is not None else (0, 1)
    mode_b = tuple(mode_b) if mode_b is not None else (1, 2)
    mode_d = tuple(mode_d) if mode_d is not None else (0, 2)
    mode_c = tuple(mode_c) if mode_c is not None else mode_d
    return mode_a == (0, 1) and mode_b == (1, 2) and mode_c == (0, 2) and mode_d == (0, 2)


def _supports_triton_gett(a, b, c, mode_a, mode_b, mode_c, mode_d):
    if not _is_on_accelerator(a) or not _is_on_accelerator(b):
        return False
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if c is not None and (not _is_on_accelerator(c) or c.dtype != a.dtype):
        return False
    return _is_default_2d_gett_case(a, b, c, mode_a, mode_b, mode_c, mode_d)


def _launch_gett_kernel(a, b, c, out, alpha, beta):
    return _launch_gett_like_kernel(a, b, c, out, alpha, beta, trans_a=False, trans_b=False)


def _gett_launcher_signature(a, b, c, out, *, trans_a, trans_b):
    return (
        a.device,
        a.dtype,
        tuple(a.shape),
        tuple(a.stride()),
        tuple(b.shape),
        tuple(b.stride()),
        tuple(c.shape) if c is not None else None,
        tuple(c.stride()) if c is not None else None,
        tuple(out.shape),
        tuple(out.stride()),
        trans_a,
        trans_b,
        c is not None,
    )


def _make_prepared_gett_launcher(a, b, c, out, *, trans_a=False, trans_b=False):
    if trans_a:
        M, K = a.shape[1], a.shape[0]
    else:
        M, K = a.shape
    if trans_b:
        K_b, N = b.shape[1], b.shape[0]
    else:
        K_b, N = b.shape
    if K != K_b:
        raise ValueError("inner dimensions must match for GETT-like Triton kernel")

    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm = c.stride(0) if c is not None else 0
    stride_cn = c.stride(1) if c is not None else 0
    stride_dm, stride_dn = out.stride(0), out.stride(1)
    has_c = c is not None
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    def _run(a_tensor, b_tensor, c_tensor, out_tensor, alpha, beta):
        _gett_kernel[grid](
            a_tensor,
            b_tensor,
            c_tensor,
            out_tensor,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_dm,
            stride_dn,
            alpha,
            beta,
            TRANS_A=trans_a,
            TRANS_B=trans_b,
            HAS_C=has_c,
        )
        return out_tensor

    return _run


def _get_prepared_gett_launcher(a, b, c, out, *, trans_a=False, trans_b=False):
    key = _gett_launcher_signature(a, b, c, out, trans_a=trans_a, trans_b=trans_b)
    launcher = _GETT_PREPARED_LAUNCHER_CACHE.get(key)
    if launcher is None:
        launcher = _make_prepared_gett_launcher(a, b, c, out, trans_a=trans_a, trans_b=trans_b)
        _GETT_PREPARED_LAUNCHER_CACHE[key] = launcher
    return launcher


def _launch_gett_like_kernel(a, b, c, out, alpha, beta, *, trans_a=False, trans_b=False):
    launcher = _get_prepared_gett_launcher(a, b, c, out, trans_a=trans_a, trans_b=trans_b)
    return launcher(a, b, c, out, alpha, beta)


def _validate_triton_gett_inputs(a, b, c, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("2D Triton GETT requires rank-2 inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions must match for GETT")
    if c is not None and tuple(c.shape) != (a.shape[0], b.shape[1]):
        raise ValueError(f"addend tensor shape mismatch: expected {(a.shape[0], b.shape[1])}, got {tuple(c.shape)}")
    if out is not None:
        if not _is_on_accelerator(out):
            raise ValueError("output tensor must be on CUDA")
        if out.dtype != a.dtype:
            raise TypeError("output tensor must have the same dtype as inputs")
        if tuple(out.shape) != (a.shape[0], b.shape[1]):
            raise ValueError(f"output tensor shape mismatch: expected {(a.shape[0], b.shape[1])}, got {tuple(out.shape)}")


# ── General contraction path (replaces cuTensor fallback) ────────────────────

def _contract_via_triton_gett(a, b, c, mode_a, mode_b, mode_c, mode_d, alpha, beta, out):
    """
    General tensor contraction implemented via our Triton GEMM kernel.

    Strategy:
      - 2D mode permutations → detect trans_a/trans_b flags, use kernel directly
      - ND contraction → permute + reshape to 2D → Triton GEMM → reshape back
      - bfloat16 → upcast to float32 → GEMM → cast back
      - Complex dtypes → NotImplementedError
    """
    if a.dtype.is_complex or b.dtype.is_complex:
        raise NotImplementedError(
            f"GETT: complex dtypes ({a.dtype}) are not yet supported"
        )

    cast_back = None
    if a.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        cast_back = a.dtype
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        if c is not None:
            c = c.to(torch.float32)

    # Resolve default modes
    if mode_a is None:
        mode_a = tuple(range(a.ndim))
    mode_a = _normalize_modes(mode_a, a.ndim)
    if mode_b is None:
        mode_b = tuple(range(mode_a[-1], mode_a[-1] + b.ndim))
    mode_b = _normalize_modes(mode_b, b.ndim)

    # Find shared modes between A and B.
    # Shared modes that also appear in mode_d are batch dimensions (preserved).
    # Shared modes absent from mode_d are contracted (summed over).
    shared_modes = [m for m in mode_a if m in mode_b]

    if mode_d is not None:
        mode_d_set = set(mode_d)
        contracted = [m for m in shared_modes if m not in mode_d_set]
        if not contracted:
            raise ValueError("no contracted modes between A and B (all shared modes appear in output)")
    else:
        contracted = list(shared_modes)
    free_a = [m for m in mode_a if m not in contracted]
    free_b = [m for m in mode_b if m not in contracted]

    if mode_d is None:
        mode_d = tuple(free_a + [m for m in free_b if m not in free_a])
    else:
        expected_output_modes = set(free_a + free_b)
        if set(mode_d) != expected_output_modes or len(mode_d) != len(expected_output_modes):
            raise ValueError(
                f"mode_d {mode_d} inconsistent with free modes "
                f"free_a={free_a} free_b={free_b}"
            )

    if mode_c is None:
        mode_c = mode_d
    elif c is not None:
        mode_c = _normalize_modes(mode_c, c.ndim)

    # Build mode → size map
    mode_sizes = {}
    for i, m in enumerate(mode_a):
        mode_sizes[m] = a.shape[i]
    for i, m in enumerate(mode_b):
        if m in mode_sizes and mode_sizes[m] != b.shape[i]:
            raise ValueError(
                f"contraction mode {m} has inconsistent sizes: "
                f"{mode_sizes[m]} in A vs {b.shape[i]} in B"
            )
        mode_sizes[m] = b.shape[i]

    output_shape = tuple(mode_sizes[m] for m in mode_d)
    K = 1
    for m in contracted:
        K *= mode_sizes[m]

    if a.ndim == 2 and b.ndim == 2 and len(contracted) == 1:
        # ── 2D path with mode detection ──────────────────────────────────
        trans_a = mode_a[0] in contracted
        trans_b = mode_b[1] in contracted

        a = a.contiguous()
        b = b.contiguous()

        if c is not None:
            if mode_c != mode_d:
                raise NotImplementedError(
                    "GETT: C tensor mode remapping is not yet supported"
                )
            c = c.contiguous()

        if out is None:
            out = torch.empty(output_shape, device=a.device, dtype=a.dtype)

        result = _launch_gett_like_kernel(a, b, c, out, alpha, beta,
                                          trans_a=trans_a, trans_b=trans_b)
        if cast_back is not None:
            result = result.to(cast_back)
        return result

    # ── ND path: permute + reshape to 2D ─────────────────────────────
    a_cont_dims = [mode_a.index(m) for m in contracted]
    a_free_dims = [mode_a.index(m) for m in free_a]
    b_cont_dims = [mode_b.index(m) for m in contracted]
    b_free_dims = [mode_b.index(m) for m in free_b]

    # Detect batch modes: free modes present in both A and B
    batch_modes = [m for m in free_a if m in free_b]
    free_a_unique = [m for m in free_a if m not in batch_modes]
    free_b_unique = [m for m in free_b if m not in batch_modes]

    if batch_modes and not (free_a_unique or free_b_unique):
        # Pure batch-mode contraction (all free modes are shared).
        # A and B have no free non-batch modes → output is just the batch shape.
        # Example: A=(B,K), B=(B,K) with mode_d=(B) → contraction along K only.
        # This requires a dot-product-like operation, which our GEMM does not
        # handle via the simple ND reshape path. Fall through to error.

        # Compute batch product for the 2D reshape
        a_batch_dims = [mode_a.index(m) for m in batch_modes]
        a_perm = a.permute(*a_batch_dims, *a_cont_dims).contiguous()
        batch_size = 1
        for i in range(len(batch_modes)):
            batch_size *= a_perm.shape[i]
        M = batch_size
        a_2d = a_perm.reshape(M, K)
        N = 1  # No unique free modes in B

        b_batch_dims = [mode_b.index(m) for m in batch_modes]
        b_perm = b.permute(*b_batch_dims, *b_cont_dims).contiguous()
        b_2d = b_perm.reshape(batch_size, K)
    else:
        if batch_modes:
            # Batched contraction: unroll batch dimension as a loop over 2D GEMMs
            batch_sizes = [mode_sizes[m] for m in batch_modes]
            total_batch = 1
            for s in batch_sizes:
                total_batch *= s
            batch_shape = tuple(batch_sizes)

            # A: permute to (batch_modes, free_a_unique, contracted)
            a_batch_dims = [mode_a.index(m) for m in batch_modes]
            a_free_u_dims = [mode_a.index(m) for m in free_a_unique]
            a_perm = a.permute(*(a_batch_dims + a_free_u_dims + a_cont_dims)).contiguous()
            M = 1
            for i in range(len(free_a_unique)):
                M *= a_perm.shape[len(batch_modes) + i]
            a_view = a_perm.reshape(total_batch, M, K)

            # B: permute to (batch_modes, contracted, free_b_unique)
            b_batch_dims = [mode_b.index(m) for m in batch_modes]
            b_free_u_dims = [mode_b.index(m) for m in free_b_unique]
            b_perm = b.permute(*(b_batch_dims + b_cont_dims + b_free_u_dims)).contiguous()
            N = 1
            for i in range(len(free_b_unique)):
                N *= b_perm.shape[len(batch_modes) + len(contracted) + i]
            b_view = b_perm.reshape(total_batch, K, N)

            # C: permute to (batch_modes, free_a_unique, free_b_unique) = mode_d order
            if c is not None:
                if mode_c != mode_d:
                    raise NotImplementedError(
                        "GETT: C tensor mode remapping is not yet supported"
                    )
                c_dims = [mode_c.index(m) for m in batch_modes + free_a_unique + free_b_unique]
                c_perm = c.permute(*c_dims).contiguous()
                c_view = c_perm.reshape(total_batch, M, N)
            else:
                c_view = None

            # Allocate output in batch-aware layout and loop
            out_perm_shape = batch_shape + (M, N)
            result_perm = torch.empty(out_perm_shape, device=a.device, dtype=a.dtype)
            for b in range(total_batch):
                c_slice = c_view[b] if c_view is not None else None
                _launch_gett_like_kernel(a_view[b], b_view[b], c_slice,
                                         result_perm[b], alpha, beta,
                                         trans_a=False, trans_b=False)

            # Reshape result to output shape (mode_d order)
            result = result_perm.reshape(output_shape)
            if cast_back is not None:
                result = result.to(cast_back)
            if out is not None:
                out.copy_(result)
                return out
            return result
        else:
            # No batch modes: simple permute + reshape to 2D
            a_perm = a.permute(*a_free_dims, *a_cont_dims).contiguous()
            M = 1
            for i in range(len(free_a)):
                M *= a_perm.shape[i]
            a_2d = a_perm.reshape(M, K)

            b_perm = b.permute(*b_cont_dims, *b_free_dims).contiguous()
            N = 1
            for i in range(len(free_b)):
                N *= b_perm.shape[len(contracted) + i]
            b_2d = b_perm.reshape(K, N)

            if c is not None:
                if mode_c != mode_d:
                    raise NotImplementedError(
                        "GETT: C tensor mode remapping is not yet supported"
                    )
                c_2d = c.contiguous().reshape(M, N)
            else:
                c_2d = None

            out_2d = torch.empty(M, N, device=a.device, dtype=a.dtype)
            result_2d = _launch_gett_like_kernel(a_2d, b_2d, c_2d, out_2d,
                                                 alpha, beta, trans_a=False, trans_b=False)

    result = result_2d.reshape(output_shape)
    if cast_back is not None:
        result = result.to(cast_back)

    if out is not None:
        out.copy_(result)
        return out
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def contraction(a, b, *, c=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
    """General tensor contraction: ``alpha * A @ B + beta * C``.

    Fast path: 2D contiguous float16/float32 with default modes uses a
    specialised autotuned Triton GEMM kernel.

    General path: arbitrary ND tensors, mode permutations, and non-standard
    dtypes are handled by reshaping to 2D and dispatching through the same
    Triton GEMM kernel.
    """
    if not _is_on_accelerator(a) or not _is_on_accelerator(b):
        raise ValueError("input tensors must be on CUDA")
    if a.dtype != b.dtype:
        raise TypeError("input tensors must have the same dtype")
    if c is not None and not _is_on_accelerator(c):
        raise ValueError("addend tensor must be on CUDA")
    if c is not None and c.dtype != a.dtype:
        raise TypeError("addend tensor must have the same dtype as inputs")
    if out is not None and not _is_on_accelerator(out):
        raise ValueError("output tensor must be on CUDA")
    if out is not None and out.dtype != a.dtype:
        raise TypeError("output tensor must have the same dtype as inputs")

    # Fast path: default 2D case with supported dtypes
    if _supports_triton_gett(a, b, c, mode_a, mode_b, mode_c, mode_d):
        _validate_triton_gett_inputs(a, b, c, out)
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        return _launch_gett_kernel(a, b, c, out, alpha, beta)

    return _contract_via_triton_gett(a, b, c, mode_a, mode_b, mode_c, mode_d, alpha, beta, out)
