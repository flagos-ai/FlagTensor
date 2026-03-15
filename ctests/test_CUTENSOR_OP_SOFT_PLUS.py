import pytest
import torch
import torch.nn.functional as F

from flagtensor import soft_plus
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_SOFT_PLUS_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorSoftPlus
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SOFT_PLUS_TEST_SHAPES)
def test_soft_plus_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = soft_plus(x)
    expected = F.softplus(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSoftPlus(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
