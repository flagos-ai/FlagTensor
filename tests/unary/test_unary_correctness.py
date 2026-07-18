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

from flagtensor import abs
from flagtensor import acos
from flagtensor import acosh
from flagtensor import asin
from flagtensor import asinh
from flagtensor import atan
from flagtensor import atanh
from flagtensor import ceil
from flagtensor import conj
from flagtensor import cos
from flagtensor import cosh
from flagtensor import floor
from flagtensor import identity
from flagtensor import log
from flagtensor import mish
from flagtensor import neg
from flagtensor import rcp
from flagtensor import relu
from flagtensor import sigmoid
from flagtensor import sin
from flagtensor import sinh
from flagtensor import soft_plus
from flagtensor import soft_sign
from flagtensor import sqrt
from flagtensor import swish
from flagtensor import tan
from flagtensor import tanh
from flagtensor.config import DEFAULT_ABS_TEST_SHAPES
from flagtensor.config import DEFAULT_ACOS_TEST_SHAPES
from flagtensor.config import DEFAULT_ACOSH_TEST_SHAPES
from flagtensor.config import DEFAULT_ASIN_TEST_SHAPES
from flagtensor.config import DEFAULT_ASINH_TEST_SHAPES
from flagtensor.config import DEFAULT_ATAN_TEST_SHAPES
from flagtensor.config import DEFAULT_ATANH_TEST_SHAPES
from flagtensor.config import DEFAULT_CEIL_TEST_SHAPES
from flagtensor.config import DEFAULT_CONJ_CORRECTNESS_DTYPES
from flagtensor.config import DEFAULT_CONJ_TEST_SHAPES
from flagtensor.config import DEFAULT_COS_TEST_SHAPES
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES
from flagtensor.config import DEFAULT_COSH_TEST_SHAPES
from flagtensor.config import DEFAULT_FLOOR_TEST_SHAPES
from flagtensor.config import DEFAULT_IDENTITY_TEST_SHAPES
from flagtensor.config import DEFAULT_LOG_TEST_SHAPES
from flagtensor.config import DEFAULT_MISH_TEST_SHAPES
from flagtensor.config import DEFAULT_NEG_TEST_SHAPES
from flagtensor.config import DEFAULT_RCP_TEST_SHAPES
from flagtensor.config import DEFAULT_RELU_TEST_SHAPES
from flagtensor.config import DEFAULT_SIGMOID_TEST_SHAPES
from flagtensor.config import DEFAULT_SIN_TEST_SHAPES
from flagtensor.config import DEFAULT_SINH_TEST_SHAPES
from flagtensor.config import DEFAULT_SOFT_PLUS_TEST_SHAPES
from flagtensor.config import DEFAULT_SOFT_SIGN_TEST_SHAPES
from flagtensor.config import DEFAULT_SQRT_TEST_SHAPES
from flagtensor.config import DEFAULT_SWISH_TEST_SHAPES
from flagtensor.config import DEFAULT_TAN_TEST_SHAPES
from flagtensor.config import DEFAULT_TANH_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.cutensor import CuTensorAcos
from flagtensor.cutensor import CuTensorAcosh
from flagtensor.cutensor import CuTensorAsin
from flagtensor.cutensor import CuTensorAsinh
from flagtensor.cutensor import CuTensorAtan
from flagtensor.cutensor import CuTensorAtanh
from flagtensor.cutensor import CuTensorCeil
from flagtensor.cutensor import CuTensorConj
from flagtensor.cutensor import CuTensorCos
from flagtensor.cutensor import CuTensorCosh
from flagtensor.cutensor import CuTensorFloor
from flagtensor.cutensor import CuTensorIdentity
from flagtensor.cutensor import CuTensorLog
from flagtensor.cutensor import CuTensorMish
from flagtensor.cutensor import CuTensorNeg
from flagtensor.cutensor import CuTensorRcp
from flagtensor.cutensor import CuTensorRelu
from flagtensor.cutensor import CuTensorSigmoid
from flagtensor.cutensor import CuTensorSin
from flagtensor.cutensor import CuTensorSinh
from flagtensor.cutensor import CuTensorSoftPlus
from flagtensor.cutensor import CuTensorSoftSign
from flagtensor.cutensor import CuTensorSqrt
from flagtensor.cutensor import CuTensorSwish
from flagtensor.cutensor import CuTensorTan
from flagtensor.cutensor import CuTensorTanh
from flagtensor.testing import assert_close


@pytest.mark.CUTENSOR_OP_ABS
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ABS_TEST_SHAPES)
def test_abs_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = abs(x)
    expected = torch.abs(x)
    assert_close(y, expected, dtype)


@pytest.mark.CUTENSOR_OP_ACOS
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ACOS_TEST_SHAPES)
def test_acos_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-1, 1)
    y = acos(x)
    expected = torch.acos(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAcos(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_IDENTITY
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_IDENTITY_TEST_SHAPES)
def test_identity_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = identity(x)
    assert_close(y, x, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorIdentity(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, x, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_NEG
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_NEG_TEST_SHAPES)
def test_neg_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = neg(x)
    expected = torch.neg(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorNeg(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_RELU
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_RELU_TEST_SHAPES)
def test_relu_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = relu(x)
    expected = torch.relu(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorRelu(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_CEIL
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_CEIL_TEST_SHAPES)
def test_ceil_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = ceil(x)
    expected = torch.ceil(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorCeil(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_FLOOR
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_FLOOR_TEST_SHAPES)
def test_floor_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = floor(x)
    expected = torch.floor(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorFloor(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_LOG
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_LOG_TEST_SHAPES)
def test_log_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(1e-3, 8.0)
    y = log(x)
    expected = torch.log(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorLog(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SQRT
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SQRT_TEST_SHAPES)
def test_sqrt_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.rand(shape, device="cuda", dtype=dtype) + 1e-3
    y = sqrt(x)
    expected = torch.sqrt(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSqrt(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SIN
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SIN_TEST_SHAPES)
def test_sin_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = sin(x)
    expected = torch.sin(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSin(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_COS
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_COS_TEST_SHAPES)
def test_cos_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = cos(x)
    expected = torch.cos(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorCos(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_TAN
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_TAN_TEST_SHAPES)
def test_tan_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = tan(x)
    expected = torch.tan(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorTan(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SINH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SINH_TEST_SHAPES)
def test_sinh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = sinh(x)
    expected = torch.sinh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSinh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_COSH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_COSH_TEST_SHAPES)
def test_cosh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = cosh(x)
    expected = torch.cosh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorCosh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_TANH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_TANH_TEST_SHAPES)
def test_tanh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = tanh(x)
    expected = torch.tanh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorTanh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_ASIN
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ASIN_TEST_SHAPES)
def test_asin_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-1, 1)
    y = asin(x)
    expected = torch.asin(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAsin(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_ATAN
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ATAN_TEST_SHAPES)
def test_atan_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = atan(x)
    expected = torch.atan(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAtan(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SIGMOID
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SIGMOID_TEST_SHAPES)
def test_sigmoid_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = sigmoid(x)
    expected = torch.sigmoid(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSigmoid(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_MISH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_MISH_TEST_SHAPES)
def test_mish_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import torch.nn.functional as F

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = mish(x)
    expected = F.mish(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorMish(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_ASINH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ASINH_TEST_SHAPES)
def test_asinh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = asinh(x)
    expected = torch.asinh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAsinh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_ACOSH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ACOSH_TEST_SHAPES)
def test_acosh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(1, 3)
    y = acosh(x)
    expected = torch.acosh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAcosh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_ATANH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ATANH_TEST_SHAPES)
def test_atanh_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-0.9, 0.9)
    y = atanh(x)
    expected = torch.atanh(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAtanh(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SOFT_PLUS
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SOFT_PLUS_TEST_SHAPES)
def test_soft_plus_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    import torch.nn.functional as F

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = soft_plus(x)
    expected = F.softplus(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSoftPlus(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SOFT_SIGN
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SOFT_SIGN_TEST_SHAPES)
def test_soft_sign_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = soft_sign(x)
    expected = x / (torch.abs(x) + 1)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSoftSign(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_SWISH
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_SWISH_TEST_SHAPES)
def test_swish_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-8.0, 8.0)
    y = swish(x)
    expected = x * torch.sigmoid(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorSwish(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_RCP
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_RCP_TEST_SHAPES)
def test_rcp_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def make_nonzero_tensor(shape, dtype):
        x = torch.randn(shape, device="cuda", dtype=dtype)
        eps = torch.tensor(1e-3, device="cuda", dtype=dtype)
        return torch.where(x >= 0, x + eps, x - eps)

    x = make_nonzero_tensor(shape, dtype)
    y = rcp(x)
    expected = torch.reciprocal(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorRcp(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)


@pytest.mark.CUTENSOR_OP_CONJ
@pytest.mark.parametrize("dtype", DEFAULT_CONJ_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_CONJ_TEST_SHAPES)
def test_conj_correctness(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def x_real_dtype(dtype):
        return torch.float32 if dtype == torch.complex64 else torch.float64

    real_dtype = x_real_dtype(dtype)
    x = torch.randn(shape, device="cuda", dtype=real_dtype) + 1j * torch.randn(
        shape, device="cuda", dtype=real_dtype
    )
    x = x.to(dtype)
    y = conj(x)
    expected = torch.conj(x)
    assert_close(y, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorConj(dtype=dtype)
        y_base = baseline(x)
        assert_close(y_base, expected, dtype)
        assert_close(y, y_base, dtype)
