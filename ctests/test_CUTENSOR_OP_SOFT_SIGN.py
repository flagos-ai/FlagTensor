import pytest
import torch

from flagtensor import soft_sign
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_SOFT_SIGN_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorSoftSign
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SOFT_SIGN_TEST_SHAPES)
def test_soft_sign_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = soft_sign(x)
    expected = x / (torch.abs(x) + 1)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSoftSign(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
