import pytest
import torch

from flagtensor import rcp
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_RCP_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorRcp
from flagtensor.testing import assert_close


def make_nonzero_tensor(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    eps = torch.tensor(1e-3, device="cuda", dtype=dtype)
    return torch.where(x >= 0, x + eps, x - eps)


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_RCP_TEST_SHAPES)
def test_rcp_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = make_nonzero_tensor(shape, dtype)
    y = rcp(x)
    expected = torch.reciprocal(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorRcp(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
