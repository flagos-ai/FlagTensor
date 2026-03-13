import pytest
import torch

from flagtensor import asin
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_ASIN_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorAsin
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ASIN_TEST_SHAPES)
def test_asin_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-1, 1)
    y = asin(x)
    expected = torch.asin(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAsin(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
