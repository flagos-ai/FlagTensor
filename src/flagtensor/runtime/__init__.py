import triton

_TUNED_CONFIGS = {
    "CUTENSOR_OP_IDENTITY": [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    "CUTENSOR_OP_SQRT": [
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
}


def get_tuned_config(op_name: str):
    if op_name not in _TUNED_CONFIGS:
        raise KeyError(f"No tuned config registered for op: {op_name}")
    return _TUNED_CONFIGS[op_name]


__all__ = ["get_tuned_config"]
