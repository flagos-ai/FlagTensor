import torch
import triton
import triton.language as tl

from flagtensor.cutensor import BlockSparseTensor
from flagtensor.cutensor import BlockSparseTensorContraction
from flagtensor.cutensor import BlockSparseTensorDescriptor
from flagtensor.cutensor import _get_block_sparse_contraction_executor
from flagtensor.cutensor import _infer_contraction_output_shape
from flagtensor.cutensor import _normalize_modes
from flagtensor.ops.CUTENSOR_OP_GETT import _launch_gett_kernel


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _block_sparse_gemm_kernel(
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
    HAS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Dense GEMM kernel for a single block pair in block-sparse contraction.
    C = alpha * A @ B + beta * C
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = alpha * acc

    if HAS_C:
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc = acc + beta * c.to(tl.float32)

    d_ptrs = d_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    tl.store(d_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _launch_block_sparse_gemm(a, b, c, out, alpha, beta):
    """Launch dense GEMM for a single block pair."""
    M, K = a.shape
    _, N = b.shape

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))

    HAS_C = c is not None

    _block_sparse_gemm_kernel[grid](
        a, b, c, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0) if c is not None else 0,
        c.stride(1) if c is not None else 0,
        out.stride(0), out.stride(1),
        alpha, beta,
        HAS_C=HAS_C,
    )
    return out


def _build_block_contraction_plan(a_desc, b_desc, mode_a, mode_b, mode_d):
    """
    Build contraction plan for block-sparse tensors.
    Returns: dict mapping (out_coord) -> list of (a_coord, b_coord) pairs
    """
    # For 2D case: mode_a=(0,1), mode_b=(1,2), mode_d=(0,2)
    # Shared mode is 1 (the K dimension)
    # A has blocks at (i, k), B has blocks at (k, j), output at (i, j)

    shared_mode_idx_a = 1 if mode_a == (0, 1) else None
    shared_mode_idx_b = 0 if mode_b == (1, 2) else None

    if shared_mode_idx_a is None or shared_mode_idx_b is None:
        return None  # Unsupported mode configuration

    plan = {}

    for a_coord in a_desc.canonical_nonzero_coordinates:
        for b_coord in b_desc.canonical_nonzero_coordinates:
            # Check if blocks can contract (share the same K index)
            if a_coord[shared_mode_idx_a] == b_coord[shared_mode_idx_b]:
                # Output coordinate
                if mode_a == (0, 1) and mode_b == (1, 2):
                    out_coord = (a_coord[0], b_coord[1])  # (i, j)
                else:
                    return None  # Unsupported mode

                if out_coord not in plan:
                    plan[out_coord] = []
                plan[out_coord].append((a_coord, b_coord))

    return plan


def _is_default_2d_block_sparse_case(a, b, c, mode_a, mode_b, mode_c, mode_d):
    """Check if this is a supported 2D block-sparse contraction case."""
    if a.ndim != 2 or b.ndim != 2:
        return False
    if c is not None and c.ndim != 2:
        return False

    mode_a = tuple(mode_a) if mode_a is not None else (0, 1)
    mode_b = tuple(mode_b) if mode_b is not None else (1, 2)
    mode_d = tuple(mode_d) if mode_d is not None else (0, 2)
    mode_c = tuple(mode_c) if mode_c is not None else mode_d

    return mode_a == (0, 1) and mode_b == (1, 2) and mode_c == (0, 2) and mode_d == (0, 2)


def _supports_triton_block_sparse(a, b, c, mode_a, mode_b, mode_c, mode_d):
    """Check if we can use Triton block-sparse path."""
    # Check device using BlockSparseTensor.device property
    if a.device is None or b.device is None:
        return False
    if a.device.type != 'cuda' or b.device.type != 'cuda':
        return False
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.float32):
        return False
    if c is not None:
        if c.device is None or c.device.type != 'cuda' or c.dtype != a.dtype:
            return False

    return _is_default_2d_block_sparse_case(a, b, c, mode_a, mode_b, mode_c, mode_d)


def _validate_triton_block_sparse_inputs(a, b, c, out):
    """Validate inputs for Triton block-sparse path."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("2D Triton block-sparse requires rank-2 inputs")


def _get_section_extents_for_coord(descriptor, coord):
    """Get the actual block shape for a given coordinate."""
    return tuple(
        descriptor.section_extents[mode][index]
        for mode, index in enumerate(coord)
    )


_BLOCK_SPARSE_OUTPUT_CACHE = {}


def _get_output_tensor(coord, shape, device, dtype):
    """Get or create output block tensor."""
    key = (coord, torch.device(device), tuple(shape), dtype)
    tensor = _BLOCK_SPARSE_OUTPUT_CACHE.get(key)
    if tensor is None or tuple(tensor.shape) != tuple(shape) or tensor.device != torch.device(device) or tensor.dtype != dtype:
        tensor = torch.empty(shape, device=device, dtype=dtype)
        _BLOCK_SPARSE_OUTPUT_CACHE[key] = tensor
    return tensor


def block_sparse_tensor_contraction(
    a: BlockSparseTensor,
    b: BlockSparseTensor,
    *,
    c=None,
    alpha=1.0,
    beta=0.0,
    mode_a=None,
    mode_b=None,
    mode_c=None,
    mode_d=None,
    out=None,
):
    """
    Block-sparse tensor contraction with Triton kernel support.
    Falls back to dense GETT for unsupported cases.
    """
    # Normalize modes - only for explicitly provided modes
    try:
        if mode_a is not None:
            mode_a = _normalize_modes(mode_a, a.ndim)
        if mode_b is not None:
            mode_b = _normalize_modes(mode_b, b.ndim)
        if mode_d is not None:
            mode_d = _normalize_modes(mode_d, a.ndim)
        if mode_c is not None:
            mode_c = _normalize_modes(mode_c, a.ndim)
    except ValueError:
        # Mode validation failed, fall back to dense path
        mode_a = mode_a
        mode_b = mode_b
        mode_c = mode_c
        mode_d = mode_d

    # Check if we can use Triton path
    if _supports_triton_block_sparse(a, b, c, mode_a, mode_b, mode_c, mode_d):
        _validate_triton_block_sparse_inputs(a, b, c, out)

        # Build contraction plan
        plan = _build_block_contraction_plan(a.descriptor, b.descriptor, mode_a, mode_b, mode_d)
        if plan is None:
            # Fall through to dense fallback
            pass
        else:
            # Get output descriptor shape
            output_shape = _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d)

            # Build output blocks
            out_blocks = {}
            for out_coord in plan.keys():
                # Output block shape: (A's row section extent, B's col section extent)
                row_extent = a.descriptor.section_extents[0][out_coord[0]]
                col_extent = b.descriptor.section_extents[1][out_coord[1]]
                out_block_shape = (row_extent, col_extent)
                out_blocks[out_coord] = torch.empty(out_block_shape, device=a.device, dtype=a.dtype)

            # Process each output block
            for out_coord, pairs in plan.items():
                out_block = out_blocks[out_coord]

                # Initialize with beta * C if C is provided and matches this output block
                addend_block = None
                if c is not None and out_coord in c.blocks:
                    addend_block = c.blocks[out_coord]
                    if beta != 0.0:
                        out_block.copy_(addend_block * beta)
                    else:
                        out_block.zero_()
                else:
                    out_block.zero_()

                # Accumulate contributions from all (A, B) pairs
                for a_coord, b_coord in pairs:
                    a_block = a.blocks[a_coord]
                    b_block = b.blocks[b_coord]

                    # Check if we need intermediate buffer
                    temp_out = torch.empty_like(out_block) if len(pairs) > 1 else out_block

                    _launch_block_sparse_gemm(
                        a_block, b_block,
                        None if len(pairs) > 1 else addend_block,
                        temp_out,
                        alpha if len(pairs) == 1 else alpha,  # Only apply alpha once if single pair
                        0.0 if len(pairs) > 1 else (beta if addend_block is not None else 0.0),
                    )

                    if len(pairs) > 1:
                        out_block.add_(temp_out)

            # Create output BlockSparseTensor
            # Derive output block_shape and section_extents from actual output blocks
            if out_blocks:
                first_coord = next(iter(out_blocks.keys()))
                first_block = out_blocks[first_coord]
                out_block_shape = first_block.shape

                # Build section_extents for output
                num_rows = max(coord[0] for coord in out_blocks.keys()) + 1
                num_cols = max(coord[1] for coord in out_blocks.keys()) + 1
                row_extents = [0] * num_rows
                col_extents = [0] * num_cols
                for (r, c), block in out_blocks.items():
                    row_extents[r] = block.shape[0]
                    col_extents[c] = block.shape[1]
                out_section_extents = (tuple(row_extents), tuple(col_extents))

                out_descriptor = BlockSparseTensorDescriptor(
                    shape=output_shape,
                    block_shape=out_block_shape,
                    num_sections_per_mode=(num_rows, num_cols),
                    section_extents=out_section_extents,
                    nonzero_coordinates=tuple(sorted(out_blocks.keys())),
                )
                return BlockSparseTensor(out_descriptor, out_blocks).to_dense()
            else:
                # Empty output
                return torch.zeros(output_shape, device=a.device, dtype=a.dtype)

    # Fallback to dense GETT via the existing executor
    executor = _get_block_sparse_contraction_executor(a.dtype)
    return executor(
        a, b, c=c, alpha=alpha, beta=beta,
        mode_a=mode_a, mode_b=mode_b, mode_c=mode_c, mode_d=mode_d,
        out=out,
    )


__all__ = ["BlockSparseTensor", "BlockSparseTensorContraction", "BlockSparseTensorDescriptor", "block_sparse_tensor_contraction"]
