"""Dtype utilities for testing."""

from typing import List

import torch


def correctness_dtypes(
    include_float64: bool = True, include_bfloat16: bool = False
) -> List[torch.dtype]:
    """Return default dtypes for correctness testing.

    Args:
        include_float64: Whether to include float64 dtype.
        include_bfloat16: Whether to include bfloat16 dtype.

    Returns:
        List of torch dtypes for testing.
    """
    dtypes = [torch.float16, torch.float32]
    if include_float64:
        dtypes.append(torch.float64)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes
