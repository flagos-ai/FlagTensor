import pytest
import torch

from flagtensor import neg
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_NEG_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorNeg
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_NEG_TEST_SHAPES)
def test_neg_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = neg(x)
    expected = torch.neg(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorNeg(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
