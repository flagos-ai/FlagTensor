import pytest
import torch

from flagtensor import atan
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_ATAN_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorAtan
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ATAN_TEST_SHAPES)
def test_atan_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = atan(x)
    expected = torch.atan(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAtan(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
