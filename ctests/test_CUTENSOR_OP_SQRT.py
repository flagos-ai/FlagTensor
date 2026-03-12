import pytest
import torch

from flagtensor import sqrt
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_SQRT_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorSqrt
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SQRT_TEST_SHAPES)
def test_sqrt_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.rand(shape, device="cuda", dtype=dtype) + 1e-3
    y = sqrt(x)
    expected = torch.sqrt(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSqrt(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
