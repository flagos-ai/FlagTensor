import pytest
import torch

from flagtensor import ceil
from flagtensor.config import DEFAULT_CEIL_TEST_SHAPES, DEFAULT_CORRECTNESS_DTYPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorCeil
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_CEIL_TEST_SHAPES)
def test_ceil_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = ceil(x)
    expected = torch.ceil(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorCeil(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
