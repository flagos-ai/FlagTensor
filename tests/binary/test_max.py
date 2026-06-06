import pytest
import torch

from tests.accuracy_utils import gems_assert_close, gems_assert_equal, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import max


@pytest.mark.max
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = torch.randn(shape, device="cuda", dtype=dtype)
    ref_x = to_reference(x, upcast=True)
    ref_y = to_reference(y, upcast=(dtype in (torch.float16, torch.float32, torch.bfloat16)))
    ref_out = torch.maximum(x, y)
    z = max(x, y)
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        gems_assert_close(z, ref_out, dtype)
    else:
        gems_assert_equal(z, ref_out)
