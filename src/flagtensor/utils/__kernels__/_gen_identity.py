import triton
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
def _identity_kernel(
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

result = _identity_kernel
result.__name__ = "_identity_kernel"
