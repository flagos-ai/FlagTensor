import pytest
import torch

from flagtensor import gett
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_GETT_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorContraction
from flagtensor.testing import assert_close


def _gett_case(shape_a, shape_b):
    if len(shape_a) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[0], shape_b[1]), lambda a, b: torch.matmul(a, b)
    return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[1], shape_b[1]), lambda a, b: torch.einsum("abc,cd->abd", a, b)


def _gett_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.parametrize("dtype", [dtype for dtype in DEFAULT_CORRECTNESS_DTYPES if dtype in (torch.float16, torch.float32, torch.float64)])
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_GETT_TEST_SHAPES)
def test_gett_correctness(dtype, shape_a, shape_b):
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.empty(shape_a, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_d, c_shape, reference = _gett_case(shape_a, shape_b)
    c = torch.empty(c_shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    out = gett(
        a,
        b,
        c=c,
        alpha=1.25,
        beta=0.5,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_d,
        mode_d=mode_d,
    )
    expected = _gett_reference(a, b, c, reference)
    assert_close(out, expected, dtype)

    baseline = CuTensorContraction(dtype=dtype)
    out_base = baseline(
        a,
        b,
        c=c,
        alpha=1.25,
        beta=0.5,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_d,
        mode_d=mode_d,
    )
    assert_close(out_base, expected, dtype)
    assert_close(out, out_base, dtype)
