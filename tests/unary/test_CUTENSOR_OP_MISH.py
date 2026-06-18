import pytest
import torch

from tests.accuracy_utils import gems_assert_close, to_reference, get_tolerance
from tests.accuracy_utils import POINTWISE_SHAPES, FLOAT_DTYPES
from flagtensor import mish


@pytest.mark.CUTENSOR_OP_MISH
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mish_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = to_reference(x, upcast=True)
    ref_out = ref * torch.tanh(torch.log(1 + torch.exp(ref)))
    y = mish(x)
    # mish involves chained ops (exp, log, tanh, mul); allow wider tolerances
    atol = 2e-3 if dtype == torch.float16 else None
    gems_assert_close(y, ref_out, dtype, atol=atol)
