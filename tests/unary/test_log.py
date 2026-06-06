import pytest
import torch

from tests.accuracy_utils import gems_assert_close, to_reference
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import log


@pytest.mark.log
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_log_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(1e-3, 8.0)
    ref = to_reference(x, upcast=True)
    ref_out = torch.log(x)
    y = log(x)
    gems_assert_close(y, ref_out, dtype)
