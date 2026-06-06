import pytest
import torch

from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import sqrt


@pytest.mark.sqrt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sqrt_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.rand(shape, device="cuda", dtype=dtype) + 1e-3
    ref = to_reference(x, upcast=True)
    ref_out = torch.sqrt(ref)
    y = sqrt(x)
    gems_assert_close(y, ref_out, dtype)
