import pytest
import torch

from flagtensor import exp
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_EXP_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorExp
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_EXP_TEST_SHAPES)
def test_exp_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = exp(x)
    expected = torch.exp(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorExp(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
