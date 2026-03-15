import pytest
import torch
import torch.nn.functional as F

from flagtensor import mish
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_MISH_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorMish
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_MISH_TEST_SHAPES)
def test_mish_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = mish(x)
    expected = F.mish(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorMish(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
