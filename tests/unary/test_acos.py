import pytest
import torch

from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import acos


@pytest.mark.acos
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_acos_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-1, 1)
    ref = to_reference(x, upcast=True)
    ref_out = torch.acos(x)
    y = acos(x)
    gems_assert_close(y, ref_out, dtype)
