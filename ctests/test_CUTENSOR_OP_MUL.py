import pytest
import torch

from flagtensor import mul
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_MUL_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorMul
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_MUL_TEST_SHAPES)
def test_mul_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    z = mul(x, y)
    expected = x * y
    assert_close(z, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorMul(dtype=dtype)
        z_base = baseline(x, y)
        assert_close(z_base, expected, dtype)
        assert_close(z, z_base, dtype)
