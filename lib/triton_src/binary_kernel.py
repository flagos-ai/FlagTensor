# Standalone binary kernel for C++ wrapper (loaded via TritonJIT).
# op_mode selects the operation:
#   0=add, 1=mul, 2=max, 3=min

import triton
import triton.language as tl


@triton.jit
def binary_kernel(
    a_ptr, b_ptr, out_ptr, n_elements,
    op_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    if op_mode == 0:   # add
        y = a + b
    elif op_mode == 1:  # mul
        y = a * b
    elif op_mode == 2:  # max
        y = tl.where(a > b, a, b)
    elif op_mode == 3:  # min
        y = tl.where(a < b, a, b)
    else:
        y = a  # fallback

    tl.store(out_ptr + offsets, y, mask=mask)
