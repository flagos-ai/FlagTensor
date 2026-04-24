import torch
from flagtensor.ops.CUTENSOR_OP_BINARY_GENERIC import binary_generic


def max(x: torch.Tensor, y: torch.Tensor, *, mode_x=None, mode_y=None, mode_out=None, out=None) -> torch.Tensor:
    return binary_generic(x, y, op="max", mode_x=mode_x, mode_y=mode_y, mode_out=mode_out, out=out)
