import pytest
import torch

from flagtensor import asinh
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_ASINH_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorAsinh
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ASINH_TEST_SHAPES)
def test_asinh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = asinh(x)
    expected = torch.asinh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAsinh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
