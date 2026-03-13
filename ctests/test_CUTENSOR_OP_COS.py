import pytest
import torch

from flagtensor import cos
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_COS_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorCos
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_COS_TEST_SHAPES)
def test_cos_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = cos(x)
    expected = torch.cos(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorCos(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
