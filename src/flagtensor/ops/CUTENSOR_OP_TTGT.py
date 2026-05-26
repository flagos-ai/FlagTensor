import torch

from flagtensor.cutensor import _normalize_modes
from flagtensor.ops.CUTENSOR_OP_GETT import _contract_via_triton_gett, _launch_gett_like_kernel


def _is_default_2d_ttgt_case(a, b, c, mode_a, mode_b, mode_c, mode_d):
    if a.ndim != 2 or b.ndim != 2:
        return False
    if c is not None and c.ndim != 2:
        return False
    mode_a = tuple(mode_a) if mode_a is not None else (0, 1)
    mode_b = tuple(mode_b) if mode_b is not None else (1, 2)
    mode_d = tuple(mode_d) if mode_d is not None else (0, 2)
    mode_c = tuple(mode_c) if mode_c is not None else mode_d
    return mode_a == (0, 1) and mode_b == (1, 2) and mode_c == (0, 2) and mode_d == (0, 2)


def _supports_triton_ttgt(a, b, c, mode_a, mode_b, mode_c, mode_d):
    if not a.is_cuda or not b.is_cuda:
        return False
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return False
    if c is not None and (not c.is_cuda or c.dtype != a.dtype):
        return False
    return _is_default_2d_ttgt_case(a, b, c, mode_a, mode_b, mode_c, mode_d)


def _validate_triton_ttgt_inputs(a, b, c, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("2D Triton TTGT requires rank-2 inputs")
    if a.shape[0] != b.shape[1]:
        raise ValueError("inner dimensions must match for TTGT")
    expected_shape = (a.shape[1], b.shape[0])
    if c is not None and tuple(c.shape) != expected_shape:
        raise ValueError(f"addend tensor shape mismatch: expected {expected_shape}, got {tuple(c.shape)}")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("output tensor must be on CUDA")
        if out.dtype != a.dtype:
            raise TypeError("output tensor must have the same dtype as inputs")
        if tuple(out.shape) != expected_shape:
            raise ValueError(f"output tensor shape mismatch: expected {expected_shape}, got {tuple(out.shape)}")


def ttgt(a, b, *, c=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
    """Transposed-transposed general tensor contraction: ``alpha * A^T @ B^T + beta * C``."""
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("input tensors must be on CUDA")
    if a.dtype != b.dtype:
        raise TypeError("input tensors must have the same dtype")
    if c is not None and not c.is_cuda:
        raise ValueError("addend tensor must be on CUDA")
    if c is not None and c.dtype != a.dtype:
        raise TypeError("addend tensor must have the same dtype as inputs")
    if out is not None and not out.is_cuda:
        raise ValueError("output tensor must be on CUDA")

    # Fast path: default 2D case
    if _supports_triton_ttgt(a, b, c, mode_a, mode_b, mode_c, mode_d):
        _validate_triton_ttgt_inputs(a, b, c, out)
        if out is None:
            out = torch.empty((a.shape[1], b.shape[0]), device=a.device, dtype=a.dtype)
        # Physically transpose to coalesce memory access in the K-loop.
        # A is (K,M) → A^T=(M,K); B is (N,K) → B^T=(K,N).
        # Then use standard GETT kernel (no transpose flags) for full-speed access.
        a_t = a.transpose(-1, -2).contiguous()
        b_t = b.transpose(-1, -2).contiguous()
        return _launch_gett_like_kernel(a_t, b_t, c, out, alpha, beta, trans_a=False, trans_b=False)

    # General path: TTGT → GETT with last two modes swapped (implicit A^T and B^T)
    if mode_a is not None:
        mode_a = _normalize_modes(mode_a, a.ndim)
    else:
        mode_a = tuple(range(a.ndim))
    if mode_b is not None:
        mode_b = _normalize_modes(mode_b, b.ndim)
    else:
        mode_b = tuple(range(a.ndim, a.ndim + b.ndim))
    if len(mode_a) >= 2:
        mode_a = mode_a[:-2] + (mode_a[-1], mode_a[-2])
    if len(mode_b) >= 2:
        mode_b = mode_b[:-2] + (mode_b[-1], mode_b[-2])

    return _contract_via_triton_gett(a, b, c, mode_a, mode_b, mode_c, mode_d, alpha, beta, out)
