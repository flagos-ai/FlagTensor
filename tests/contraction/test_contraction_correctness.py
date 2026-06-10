import pytest
import torch

from flagtensor import gett
from flagtensor import tgett
from flagtensor import ttgt
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES
from flagtensor.config import DEFAULT_GETT_TEST_SHAPES
from flagtensor.config import DEFAULT_TGETT_TEST_SHAPES
from flagtensor.config import DEFAULT_TTGT_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.cutensor import CuTensorContraction
from flagtensor.testing import assert_close


CONTRACT_DTYPES = list(DEFAULT_CORRECTNESS_DTYPES)  # f16, f32, f64, bf16


def _gett_case(shape_a, shape_b):
    if len(shape_a) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[0], shape_b[1]), lambda a, b: torch.matmul(a, b)
    return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[1], shape_b[1]), lambda a, b: torch.einsum("abc,cd->abd", a, b)


def _gett_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.gett
@pytest.mark.parametrize("dtype", CONTRACT_DTYPES)
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


def _tgett_case(shape_a, shape_b):
    if len(shape_a) == 2 and len(shape_b) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[1], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b)
    if len(shape_a) == 3 and len(shape_b) == 2:
        return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[2], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b)
    raise ValueError("unsupported TGETT test shape combination")


def _tgett_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.tgett
@pytest.mark.parametrize("dtype", CONTRACT_DTYPES)
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_TGETT_TEST_SHAPES)
def test_tgett_correctness(dtype, shape_a, shape_b):
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.empty(shape_a, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_d, c_shape, reference = _tgett_case(shape_a, shape_b)
    c = torch.empty(c_shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    out = tgett(
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
    expected = _tgett_reference(a, b, c, reference)
    assert_close(out, expected, dtype)

    baseline = CuTensorContraction(dtype=dtype)
    a_t = a.transpose(-1, -2).contiguous()
    out_base = baseline(
        a_t,
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


def _ttgt_case(shape_a, shape_b):
    if len(shape_a) == 2 and len(shape_b) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[1], shape_b[0]), lambda a, b: torch.matmul(a.transpose(-1, -2), b.transpose(-1, -2))
    if len(shape_a) == 3 and len(shape_b) == 3:
        return (0, 1, 2), (0, 2, 3), (0, 1, 3), (shape_a[0], shape_a[2], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b.transpose(-1, -2))
    raise ValueError("unsupported TTGT test shape combination")


def _ttgt_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.ttgt
@pytest.mark.parametrize("dtype", CONTRACT_DTYPES)
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_TTGT_TEST_SHAPES)
def test_ttgt_correctness(dtype, shape_a, shape_b):
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.empty(shape_a, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_d, c_shape, reference = _ttgt_case(shape_a, shape_b)
    c = torch.empty(c_shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    out = ttgt(
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
    expected = _ttgt_reference(a, b, c, reference)
    assert_close(out, expected, dtype)

    baseline = CuTensorContraction(dtype=dtype)
    a_t = a.transpose(-1, -2).contiguous()
    b_t = b.transpose(-1, -2).contiguous()
    out_base = baseline(
        a_t,
        b_t,
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



