import pytest
import torch

from tests.accuracy_utils import gems_assert_equal, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import conj


@pytest.mark.conj
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_conj_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = to_reference(x, upcast=(dtype in (torch.float16, torch.float32, torch.bfloat16)))
    ref_out = torch.conj(x)
    y = conj(x)
    gems_assert_equal(y, ref_out)
