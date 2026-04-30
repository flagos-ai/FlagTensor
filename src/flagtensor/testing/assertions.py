"""Assertion utilities for correctness testing."""

from typing import Dict
from typing import Optional
from typing import Tuple

import torch

DEFAULT_CORRECTNESS_TOLERANCES: Dict[torch.dtype, Tuple[float, float]] = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float32: (1e-5, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex64: (1e-5, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}


def get_tolerance(
    dtype: torch.dtype, atol: Optional[float] = None, rtol: Optional[float] = None
) -> Tuple[float, float]:
    """Get tolerance values for a given dtype.

    Args:
        dtype: The torch dtype to get tolerances for.
        atol: Override for absolute tolerance. Uses default if None.
        rtol: Override for relative tolerance. Uses default if None.

    Returns:
        Tuple of (atol, rtol) for the given dtype.
    """
    default_atol, default_rtol = DEFAULT_CORRECTNESS_TOLERANCES.get(dtype, (1e-5, 1e-5))
    return (default_atol if atol is None else atol, default_rtol if rtol is None else rtol)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
):
    """Assert that two tensors are close within dtype-specific tolerances.

    Args:
        actual: The actual tensor from FlagTensor.
        expected: The expected tensor from reference implementation.
        dtype: The dtype to use for tolerance lookup. Uses actual.dtype if None.
        atol: Override for absolute tolerance.
        rtol: Override for relative tolerance.
    """
    resolved_dtype = dtype or actual.dtype
    resolved_atol, resolved_rtol = get_tolerance(resolved_dtype, atol=atol, rtol=rtol)
    assert torch.allclose(actual, expected, atol=resolved_atol, rtol=resolved_rtol)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor):
    """Assert that two tensors are exactly equal.

    Use for bit-exact operations or integer dtypes.

    Args:
        actual: The actual tensor.
        expected: The expected tensor.
    """
    assert torch.equal(actual, expected)
