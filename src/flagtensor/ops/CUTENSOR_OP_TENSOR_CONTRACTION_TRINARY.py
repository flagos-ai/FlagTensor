import triton
import triton.language as tl
import torch

from flagtensor.cutensor import _infer_trinary_contraction_output_shape
from flagtensor.cutensor import _infer_contraction_output_shape
from flagtensor.cutensor import _normalize_modes
from flagtensor.cutensor import _validate_trinary_contraction_addend
from flagtensor.ops.CUTENSOR_OP_GETT import contraction
from flagtensor.ops.CUTENSOR_OP_GETT import _launch_gett_kernel
from flagtensor.ops.CUTENSOR_OP_GETT import _get_prepared_gett_launcher


_TRINARY_INTERMEDIATE_CACHE = {}
_TRINARY_PREPARED_PLAN_CACHE = {}
_TRINARY_OUTPUT_CACHE = {}
_TRINARY_PREPARED_LAUNCHER_CACHE = {}
_FUSED_TRINARY_PREPARED_LAUNCHER_CACHE = {}


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_L": 16, "GROUP_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16, "BLOCK_L": 16, "GROUP_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16, "BLOCK_L": 16, "GROUP_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16, "BLOCK_L": 16, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_L": 16, "GROUP_M": 4}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K", "L"],
)
@triton.jit
def _fused_trinary_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out_ptr,
    M,
    N,
    K,
    L,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bl,
    stride_cl,
    stride_cn,
    stride_dm,
    stride_dn,
    stride_om,
    stride_on,
    alpha,
    beta,
    HAS_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
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
    offs_l = tl.arange(0, BLOCK_L)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for l_start in range(0, L, BLOCK_L):
        l_mask = l_start + offs_l < L
        ab_acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_mask = k_start + offs_k < K
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
            b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + (l_start + offs_l)[None, :] * stride_bl
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & l_mask[None, :], other=0.0)
            ab_acc = tl.dot(a, b, ab_acc, input_precision="ieee")

        c_ptrs = c_ptr + (l_start + offs_l)[:, None] * stride_cl + offs_n[None, :] * stride_cn
        c = tl.load(c_ptrs, mask=l_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(ab_acc, c.to(tl.float32), acc, input_precision="ieee")

    acc = alpha * acc
    if HAS_D:
        d_ptrs = d_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
        d = tl.load(d_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc = acc + beta * d.to(tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _get_intermediate_tensor(shape, *, device, dtype):
    key = (torch.device(device), tuple(shape), dtype)
    tensor = _TRINARY_INTERMEDIATE_CACHE.get(key)
    if tensor is None or tuple(tensor.shape) != tuple(shape) or tensor.device != torch.device(device) or tensor.dtype != dtype:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        _TRINARY_INTERMEDIATE_CACHE[key] = tensor
    return tensor


def _get_output_tensor(shape, *, device, dtype):
    key = (torch.device(device), tuple(shape), dtype)
    tensor = _TRINARY_OUTPUT_CACHE.get(key)
    if tensor is None or tuple(tensor.shape) != tuple(shape) or tensor.device != torch.device(device) or tensor.dtype != dtype:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        _TRINARY_OUTPUT_CACHE[key] = tensor
    return tensor


def _trinary_launcher_signature(a, b, c, d, intermediate, out):
    return (
        a.device,
        a.dtype,
        tuple(a.shape),
        tuple(a.stride()),
        tuple(b.shape),
        tuple(b.stride()),
        tuple(c.shape),
        tuple(c.stride()),
        tuple(d.shape) if d is not None else None,
        tuple(d.stride()) if d is not None else None,
        tuple(intermediate.shape),
        tuple(intermediate.stride()),
        tuple(out.shape),
        tuple(out.stride()),
        d is not None,
    )


def _make_prepared_trinary_launcher(a, b, c, d, intermediate, out):
    first_launcher = _get_prepared_gett_launcher(a, b, None, intermediate, trans_a=False, trans_b=False)
    second_launcher = _get_prepared_gett_launcher(intermediate, c, d, out, trans_a=False, trans_b=False)

    def _run(a_tensor, b_tensor, c_tensor, d_tensor, intermediate_tensor, out_tensor, alpha, beta):
        first_launcher(a_tensor, b_tensor, None, intermediate_tensor, 1.0, 0.0)
        return second_launcher(intermediate_tensor, c_tensor, d_tensor, out_tensor, alpha, beta)

    return _run


def _get_prepared_trinary_launcher(a, b, c, d, intermediate, out):
    key = _trinary_launcher_signature(a, b, c, d, intermediate, out)
    launcher = _TRINARY_PREPARED_LAUNCHER_CACHE.get(key)
    if launcher is None:
        launcher = _make_prepared_trinary_launcher(a, b, c, d, intermediate, out)
        _TRINARY_PREPARED_LAUNCHER_CACHE[key] = launcher
    return launcher


def _fused_trinary_launcher_signature(a, b, c, d, out):
    return (
        a.device,
        a.dtype,
        tuple(a.shape),
        tuple(a.stride()),
        tuple(b.shape),
        tuple(b.stride()),
        tuple(c.shape),
        tuple(c.stride()),
        tuple(d.shape) if d is not None else None,
        tuple(d.stride()) if d is not None else None,
        tuple(out.shape),
        tuple(out.stride()),
        d is not None,
    )


def _make_prepared_fused_trinary_launcher(a, b, c, d, out):
    M, K = a.shape
    K_b, L = b.shape
    L_c, N = c.shape
    if K != K_b:
        raise ValueError("inner dimensions must match for the first contraction")
    if L != L_c:
        raise ValueError("inner dimensions must match for the second contraction")
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bl = b.stride(0), b.stride(1)
    stride_cl, stride_cn = c.stride(0), c.stride(1)
    stride_dm = d.stride(0) if d is not None else 0
    stride_dn = d.stride(1) if d is not None else 0
    stride_om, stride_on = out.stride(0), out.stride(1)
    has_d = d is not None
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    def _run(a_tensor, b_tensor, c_tensor, d_tensor, out_tensor, alpha, beta):
        _fused_trinary_kernel[grid](
            a_tensor,
            b_tensor,
            c_tensor,
            d_tensor,
            out_tensor,
            M,
            N,
            K,
            L,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bl,
            stride_cl,
            stride_cn,
            stride_dm,
            stride_dn,
            stride_om,
            stride_on,
            alpha,
            beta,
            HAS_D=has_d,
        )
        return out_tensor

    return _run


def _get_prepared_fused_trinary_launcher(a, b, c, d, out):
    key = _fused_trinary_launcher_signature(a, b, c, d, out)
    launcher = _FUSED_TRINARY_PREPARED_LAUNCHER_CACHE.get(key)
    if launcher is None:
        launcher = _make_prepared_fused_trinary_launcher(a, b, c, d, out)
        _FUSED_TRINARY_PREPARED_LAUNCHER_CACHE[key] = launcher
    return launcher


def _launch_fused_trinary_kernel(a, b, c, d, out, alpha, beta):
    launcher = _get_prepared_fused_trinary_launcher(a, b, c, d, out)
    return launcher(a, b, c, d, out, alpha, beta)


def _trinary_plan_key(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e):
    return (
        a.device,
        a.dtype,
        tuple(a.shape),
        tuple(a.stride()),
        tuple(b.shape),
        tuple(b.stride()),
        tuple(c.shape),
        tuple(c.stride()),
        tuple(d.shape) if d is not None else None,
        tuple(d.stride()) if d is not None else None,
        tuple(mode_a) if mode_a is not None else None,
        tuple(mode_b) if mode_b is not None else None,
        tuple(mode_c) if mode_c is not None else None,
        tuple(mode_d) if mode_d is not None else None,
        tuple(mode_e) if mode_e is not None else None,
    )


def _build_trinary_prepared_plan(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e):
    norm_mode_a = _normalize_modes(mode_a, a.ndim)
    norm_mode_b = _normalize_modes(mode_b, b.ndim)
    norm_mode_c = _normalize_modes(mode_c, c.ndim)
    contracted_modes = (set(norm_mode_a) & set(norm_mode_b)) | (set(norm_mode_a) & set(norm_mode_c)) | (set(norm_mode_b) & set(norm_mode_c))
    resolved_mode_e = tuple(mode_e) if mode_e is not None else tuple(mode for mode in norm_mode_a + norm_mode_b + norm_mode_c if mode not in contracted_modes)
    if len(set(resolved_mode_e)) != len(resolved_mode_e):
        raise ValueError("each output mode may appear at most once")
    output_shape = _infer_trinary_contraction_output_shape(a, norm_mode_a, b, norm_mode_b, c, norm_mode_c, resolved_mode_e)
    shared_modes = tuple(mode for mode in norm_mode_a if mode in set(norm_mode_b) and mode not in resolved_mode_e)
    intermediate_modes = tuple(mode for mode in norm_mode_a + norm_mode_b if mode not in shared_modes)
    intermediate_shape = _infer_contraction_output_shape(a, norm_mode_a, b, norm_mode_b, intermediate_modes)
    resolved_mode_d = tuple(mode_d) if mode_d is not None else resolved_mode_e
    if d is not None:
        resolved_mode_d = _validate_trinary_contraction_addend(d, resolved_mode_d, resolved_mode_e, output_shape)
    return {
        "mode_a": norm_mode_a,
        "mode_b": norm_mode_b,
        "mode_c": norm_mode_c,
        "mode_d": resolved_mode_d,
        "mode_e": resolved_mode_e,
        "output_shape": output_shape,
        "intermediate_modes": intermediate_modes,
        "intermediate_shape": intermediate_shape,
        "is_default_2d": _is_default_2d_trinary_case(a, b, c, d, norm_mode_a, norm_mode_b, norm_mode_c, resolved_mode_d, resolved_mode_e),
    }


def _get_trinary_prepared_plan(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e):
    key = _trinary_plan_key(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e)
    plan = _TRINARY_PREPARED_PLAN_CACHE.get(key)
    if plan is None:
        plan = _build_trinary_prepared_plan(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e)
        _TRINARY_PREPARED_PLAN_CACHE[key] = plan
    return plan


def _is_default_2d_trinary_case(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e):
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        return False
    if d is not None and d.ndim != 2:
        return False
    mode_a = tuple(mode_a) if mode_a is not None else (0, 1)
    mode_b = tuple(mode_b) if mode_b is not None else (1, 2)
    mode_c = tuple(mode_c) if mode_c is not None else (2, 3)
    mode_e = tuple(mode_e) if mode_e is not None else (0, 3)
    mode_d = tuple(mode_d) if mode_d is not None else mode_e
    return mode_a == (0, 1) and mode_b == (1, 2) and mode_c == (2, 3) and mode_d == (0, 3) and mode_e == (0, 3)


def _supports_triton_trinary(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e):
    if not a.is_cuda or not b.is_cuda or not c.is_cuda:
        return False
    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if d is not None and (not d.is_cuda or d.dtype != a.dtype):
        return False
    return _is_default_2d_trinary_case(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e)


def _supports_fused_triton_trinary(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e):
    # Fused kernel is currently disabled due to severe performance regression
    # at medium-to-large problem sizes (BLOCK_L=16 causes excessive iterations).
    # The two-step GETT launcher path achieves much better performance across all sizes.
    return False


def _validate_triton_trinary_inputs(a, b, c, d, out):
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("default Triton contraction_trinary requires rank-2 inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions must match for the first contraction")
    if b.shape[1] != c.shape[0]:
        raise ValueError("inner dimensions must match for the second contraction")
    expected_shape = (a.shape[0], c.shape[1])
    if d is not None and tuple(d.shape) != expected_shape:
        raise ValueError(f"addend tensor shape mismatch: expected {expected_shape}, got {tuple(d.shape)}")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("output tensor must be on CUDA")
        if out.dtype != a.dtype:
            raise TypeError("output tensor must have the same dtype as inputs")
        if tuple(out.shape) != expected_shape:
            raise ValueError(f"output tensor shape mismatch: expected {expected_shape}, got {tuple(out.shape)}")


def contraction_trinary(a, b, c, *, d=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, mode_e=None, out=None):
    if not a.is_cuda or not b.is_cuda or not c.is_cuda:
        raise ValueError("input tensors must be on CUDA")
    if a.dtype != b.dtype or a.dtype != c.dtype:
        raise TypeError("input tensors must have the same dtype")

    plan = _get_trinary_prepared_plan(a, b, c, d, mode_a, mode_b, mode_c, mode_d, mode_e)
    mode_a = plan["mode_a"]
    mode_b = plan["mode_b"]
    mode_c = plan["mode_c"]
    mode_e = plan["mode_e"]
    output_shape = plan["output_shape"]

    addend = d
    if addend is None:
        if beta != 0.0:
            addend = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
            mode_d = plan["mode_d"]
        else:
            mode_d = plan["mode_d"]
    else:
        if not addend.is_cuda:
            raise ValueError("addend tensor must be on CUDA")
        if addend.dtype != a.dtype:
            raise TypeError("addend tensor must have the same dtype as inputs")
        mode_d = plan["mode_d"]

    if out is not None:
        if not out.is_cuda:
            raise ValueError("output tensor must be on CUDA")
        if out.dtype != a.dtype:
            raise TypeError("output tensor must have the same dtype as inputs")
        if tuple(out.shape) != tuple(output_shape):
            raise ValueError(f"output tensor shape mismatch: expected {output_shape}, got {tuple(out.shape)}")

    intermediate_modes = plan["intermediate_modes"]
    intermediate_shape = plan["intermediate_shape"]
    intermediate_out = _get_intermediate_tensor(intermediate_shape, device=a.device, dtype=a.dtype)
    if plan["is_default_2d"]:
        output = out if out is not None else _get_output_tensor(output_shape, device=a.device, dtype=a.dtype)
        if _supports_fused_triton_trinary(a, b, c, addend, mode_a, mode_b, mode_c, mode_d, mode_e):
            return _launch_fused_trinary_kernel(a, b, c, addend, output, alpha, beta)
        # Use the two-step GETT launcher only for dtypes the raw GETT kernel supports
        if a.dtype in (torch.float16, torch.float32, torch.bfloat16):
            launcher = _get_prepared_trinary_launcher(a, b, c, addend, intermediate_out, output)
            return launcher(a, b, c, addend, intermediate_out, output, alpha, beta)
        # Fall through to the generic contraction()-based path for unsupported dtypes (e.g. bfloat16)

    intermediate = contraction(
        a,
        b,
        alpha=1.0,
        beta=0.0,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=intermediate_modes,
        mode_d=intermediate_modes,
        out=intermediate_out,
    )
    return contraction(
        intermediate,
        c,
        c=addend,
        alpha=alpha,
        beta=beta,
        mode_a=intermediate_modes,
        mode_b=mode_c,
        mode_c=mode_d,
        mode_d=mode_e,
        out=out,
    )
