import pytest
import torch

from flagtensor import tensor_contraction_trinary
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES, DEFAULT_TENSOR_CONTRACTION_TRINARY_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE, CuTensorContractionTrinary
from flagtensor.testing import assert_close


def _tensor_contraction_trinary_case(shape_a, shape_b, shape_c):
    return (
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
        (0, 3),
        (shape_a[0], shape_c[1]),
        lambda a, b, c: torch.matmul(torch.matmul(a, b), c),
    )


def _tensor_contraction_trinary_reference(a, b, c, d, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        temp = torch.matmul(a.float(), b.float()).to(a.dtype)
        return (1.25 * torch.matmul(temp.float(), c.float()) + 0.5 * d.float()).to(a.dtype)
    return 1.25 * reference(a, b, c) + 0.5 * d


@pytest.mark.parametrize("dtype", [dtype for dtype in DEFAULT_CORRECTNESS_DTYPES if dtype in (torch.float16, torch.float32, torch.bfloat16)])
@pytest.mark.parametrize("shape_a,shape_b,shape_c", DEFAULT_TENSOR_CONTRACTION_TRINARY_TEST_SHAPES)
def test_tensor_contraction_trinary_correctness(dtype, shape_a, shape_b, shape_c):
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.empty(shape_a, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    c = torch.empty(shape_c, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_c, mode_d, mode_e, d_shape, reference = _tensor_contraction_trinary_case(shape_a, shape_b, shape_c)
    d = torch.empty(d_shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)

    out = tensor_contraction_trinary(
        a,
        b,
        c,
        d=d,
        alpha=1.25,
        beta=0.5,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        mode_e=mode_e,
    )
    expected = _tensor_contraction_trinary_reference(a, b, c, d, reference)
    assert_close(out, expected, dtype)

    baseline = CuTensorContractionTrinary(dtype=dtype)
    out_base = baseline(
        a,
        b,
        c,
        d=d,
        alpha=1.25,
        beta=0.5,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        mode_e=mode_e,
    )
    assert_close(out_base, expected, dtype)
    assert_close(out, out_base, dtype)
