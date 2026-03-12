import pytest
import torch

from flagtensor import identity
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_IDENTITY_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorIdentity
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_IDENTITY_TEST_SHAPES)
def test_identity_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = identity(x)
    assert_close(y, x, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorIdentity(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, x, dtype)
        assert_close(y, y_base, dtype)
