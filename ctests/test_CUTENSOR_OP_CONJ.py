import pytest
import torch

from flagtensor import conj
from flagtensor.config import DEFAULT_CONJ_CORRECTNESS_DTYPES, DEFAULT_CONJ_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorConj
from flagtensor.testing import assert_close


def x_real_dtype(dtype):
    return torch.float32 if dtype == torch.complex64 else torch.float64


@pytest.mark.parametrize("dtype", DEFAULT_CONJ_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_CONJ_TEST_SHAPES)
def test_conj_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    real_dtype = x_real_dtype(dtype)
    x = torch.randn(shape, device="cuda", dtype=real_dtype) + 1j * torch.randn(
        shape, device="cuda", dtype=real_dtype
    )
    x = x.to(dtype)
    y = conj(x)
    expected = torch.conj(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorConj(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
