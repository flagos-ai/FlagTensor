import pytest
import torch

from flagtensor import abs
from flagtensor.config import DEFAULT_ABS_TEST_SHAPES, DEFAULT_CORRECTNESS_DTYPES
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ABS_TEST_SHAPES)
def test_abs_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = abs(x)
    expected = torch.abs(x)
    assert_close(y, expected, dtype)
