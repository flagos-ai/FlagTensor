# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flagtensor import contraction
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES
from flagtensor.config import DEFAULT_GETT_TEST_SHAPES
from flagtensor.config import DEFAULT_TGETT_TEST_SHAPES
from flagtensor.config import DEFAULT_TTGT_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)
from flagtensor.testing import assert_close

# On Ascend there is no cuTensor; use the torch_npu-aten baseline as the
# vendor reference instead.
if CUTENSOR_AVAILABLE:
    from flagtensor.cutensor import CuTensorContraction as _ContractionBaseline
else:
    try:
        from flagtensor.torch_npu_baseline import CuTensorContraction as _ContractionBaseline
    except ImportError:
        _ContractionBaseline = None

_BASELINE_AVAILABLE = CUTENSOR_AVAILABLE or _ContractionBaseline is not None


CONTRACT_DTYPES = list(DEFAULT_CORRECTNESS_DTYPES)  # f16, f32, f64, bf16


def _contraction_case(shape_a, shape_b):
    if len(shape_a) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[0], shape_b[1]), lambda a, b: torch.matmul(a, b)
    return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[1], shape_b[1]), lambda a, b: torch.einsum("abc,cd->abd", a, b)


def _contraction_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.Contraction
@pytest.mark.parametrize("dtype", CONTRACT_DTYPES)
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_GETT_TEST_SHAPES)
def test_contraction_correctness(dtype, shape_a, shape_b):
    if not _is_accelerator_available() or not _BASELINE_AVAILABLE:
        pytest.skip("Accelerator/baseline unavailable")

    a = torch.empty(shape_a, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_d, c_shape, reference = _contraction_case(shape_a, shape_b)
    c = torch.empty(c_shape, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    out = contraction(
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
    expected = _contraction_reference(a, b, c, reference)
    assert_close(out, expected, dtype)

    baseline = _ContractionBaseline(dtype=dtype)
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


def _contraction_trans_a_case(shape_a, shape_b):
    if len(shape_a) == 2 and len(shape_b) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[1], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b)
    if len(shape_a) == 3 and len(shape_b) == 2:
        return (0, 1, 2), (2, 3), (0, 1, 3), (shape_a[0], shape_a[2], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b)
    raise ValueError("unsupported trans-A test shape combination")


def _contraction_trans_a_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.Contraction
@pytest.mark.parametrize("dtype", CONTRACT_DTYPES)
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_TGETT_TEST_SHAPES)
def test_contraction_trans_a_correctness(dtype, shape_a, shape_b):
    """Contraction with first operand transposed (formerly tgett)."""
    if not _is_accelerator_available() or not _BASELINE_AVAILABLE:
        pytest.skip("Accelerator/baseline unavailable")

    a = torch.empty(shape_a, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_d, c_shape, reference = _contraction_trans_a_case(shape_a, shape_b)
    c = torch.empty(c_shape, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    a_t = a.transpose(-1, -2).contiguous()
    out = contraction(
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
    expected = _contraction_trans_a_reference(a, b, c, reference)
    assert_close(out, expected, dtype)

    baseline = _ContractionBaseline(dtype=dtype)
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


def _contraction_trans_ab_case(shape_a, shape_b):
    if len(shape_a) == 2 and len(shape_b) == 2:
        return (0, 1), (1, 2), (0, 2), (shape_a[1], shape_b[0]), lambda a, b: torch.matmul(a.transpose(-1, -2), b.transpose(-1, -2))
    if len(shape_a) == 3 and len(shape_b) == 3:
        return (0, 1, 2), (0, 2, 3), (0, 1, 3), (shape_a[0], shape_a[2], shape_b[1]), lambda a, b: torch.matmul(a.transpose(-1, -2), b.transpose(-1, -2))
    raise ValueError("unsupported trans-AB test shape combination")


def _contraction_trans_ab_reference(a, b, c, reference):
    if a.dtype in (torch.float16, torch.bfloat16):
        return (1.25 * reference(a.float(), b.float()) + 0.5 * c.float()).to(a.dtype)
    return 1.25 * reference(a, b) + 0.5 * c


@pytest.mark.Contraction
@pytest.mark.parametrize("dtype", CONTRACT_DTYPES)
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_TTGT_TEST_SHAPES)
def test_contraction_trans_ab_correctness(dtype, shape_a, shape_b):
    """Contraction with both operands transposed (formerly ttgt)."""
    if not _is_accelerator_available() or not _BASELINE_AVAILABLE:
        pytest.skip("Accelerator/baseline unavailable")

    a = torch.empty(shape_a, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    b = torch.empty(shape_b, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    mode_a, mode_b, mode_d, c_shape, reference = _contraction_trans_ab_case(shape_a, shape_b)
    c = torch.empty(c_shape, device=_device_str, dtype=dtype).uniform_(-2.0, 2.0)
    a_t = a.transpose(-1, -2).contiguous()
    b_t = b.transpose(-1, -2).contiguous()
    out = contraction(
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
    expected = _contraction_trans_ab_reference(a, b, c, reference)
    assert_close(out, expected, dtype)

    baseline = _ContractionBaseline(dtype=dtype)
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
