import pytest
import torch

from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import asinh


@pytest.mark.asinh
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = to_reference(x, upcast=True)
    ref_out = torch.asinh(x)
    y = asinh(x)
    gems_assert_close(y, ref_out, dtype)
