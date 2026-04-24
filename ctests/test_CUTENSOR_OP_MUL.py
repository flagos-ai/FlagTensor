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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mul_broadcast_correctness(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty((2, 3, 4), device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    z = mul(x, y, mode_x=(0, 1, 2), mode_y=(2,), mode_out=(0, 1, 2))
    expected = x * y.view(1, 1, 4)
    assert_close(z, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mul_mode_permute_correctness(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty((3, 4), device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    z = mul(x, y, mode_x=(1, 0), mode_y=(0,), mode_out=(1, 0))
    expected = x * y.view(1, 4)
    assert_close(z, expected, dtype)
