import pytest
import torch

from flagtensor import cosh
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_COSH_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorCosh
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_COSH_TEST_SHAPES)
def test_cosh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = cosh(x)
    expected = torch.cosh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorCosh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
