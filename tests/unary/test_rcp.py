import pytest
import torch

from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import rcp


@pytest.mark.rcp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rcp_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.where(torch.randn(shape, device="cuda", dtype=dtype) >= 0, torch.randn(shape, device="cuda", dtype=dtype) + 1e-3, torch.randn(shape, device="cuda", dtype=dtype) - 1e-3)
    ref = to_reference(x, upcast=True)
    ref_out = torch.reciprocal(x)
    y = rcp(x)
    gems_assert_close(y, ref_out, dtype)
