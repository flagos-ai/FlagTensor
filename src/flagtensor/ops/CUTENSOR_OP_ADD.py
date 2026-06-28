import torch
from flagtensor.ops.CUTENSOR_OP_BINARY_GENERIC import binary_generic


def add(x: torch.Tensor, y: torch.Tensor, *, alpha=1, mode_x=None, mode_y=None, mode_out=None, out=None) -> torch.Tensor:
    if alpha != 1:
        y = y * alpha  # aten::add.Tensor passes alpha; mul.Scalar path, does not recurse
    return binary_generic(x, y, op="add", mode_x=mode_x, mode_y=mode_y, mode_out=mode_out, out=out)
