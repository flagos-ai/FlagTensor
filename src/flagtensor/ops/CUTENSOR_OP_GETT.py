import torch
import triton
import triton.language as tl

from flagtensor.cutensor import gett as cutensor_gett


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
    if not a.is_cuda or not b.is_cuda:
        return False
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.float32):
        return False
    if c is not None and (not c.is_cuda or c.dtype != a.dtype):
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
        if not out.is_cuda:
            raise ValueError("output tensor must be on CUDA")
        if out.dtype != a.dtype:
            raise TypeError("output tensor must have the same dtype as inputs")
        if tuple(out.shape) != (a.shape[0], b.shape[1]):
            raise ValueError(f"output tensor shape mismatch: expected {(a.shape[0], b.shape[1])}, got {tuple(out.shape)}")


def gett(a, b, *, c=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
    if _supports_triton_gett(a, b, c, mode_a, mode_b, mode_c, mode_d):
        _validate_triton_gett_inputs(a, b, c, out)
        if out is None:
            out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        return _launch_gett_kernel(a, b, c, out, alpha, beta)
    return cutensor_gett(
        a,
        b,
        c=c,
        alpha=alpha,
        beta=beta,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        out=out,
    )
