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

from flagtensor.runtime import (
    device_str as _device_str,
    is_accelerator_available as _is_accelerator_available,
)

from flagtensor import add
from flagtensor import max
from flagtensor import min
from flagtensor import mul
from flagtensor.config import DEFAULT_ADD_TEST_SHAPES
from flagtensor.config import DEFAULT_CORRECTNESS_DTYPES
from flagtensor.config import DEFAULT_MAX_TEST_SHAPES
from flagtensor.config import DEFAULT_MIN_TEST_SHAPES
from flagtensor.config import DEFAULT_MUL_TEST_SHAPES
from flagtensor.cutensor import CUTENSOR_AVAILABLE
from flagtensor.cutensor import CuTensorAdd
from flagtensor.cutensor import CuTensorMax
from flagtensor.cutensor import CuTensorMin
from flagtensor.cutensor import CuTensorMul
from flagtensor.testing import assert_close


@pytest.mark.CUTENSOR_OP_ADD
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_ADD_TEST_SHAPES)
def test_add_correctness(dtype, shape):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = add(x, y)
    expected = x + y
    assert_close(z, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorAdd(dtype=dtype)
        z_base = baseline(x, y)
        assert_close(z_base, expected, dtype)
        assert_close(z, z_base, dtype)


@pytest.mark.CUTENSOR_OP_ADD
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_add_broadcast_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((3, 4, 5), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((5,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = add(x, y, mode_x=(0, 1, 2), mode_y=(2,), mode_out=(0, 1, 2))
    expected = x + y.view(1, 1, 5)
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_ADD
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_add_mode_permute_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = add(x, y, mode_x=(1, 0), mode_y=(0,), mode_out=(1, 0))
    expected = x + y.view(1, 4)
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_MUL
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_MUL_TEST_SHAPES)
def test_mul_correctness(dtype, shape):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = mul(x, y)
    expected = x * y
    assert_close(z, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorMul(dtype=dtype)
        z_base = baseline(x, y)
        assert_close(z_base, expected, dtype)
        assert_close(z, z_base, dtype)


@pytest.mark.CUTENSOR_OP_MUL
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mul_broadcast_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((2, 3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = mul(x, y, mode_x=(0, 1, 2), mode_y=(2,), mode_out=(0, 1, 2))
    expected = x * y.view(1, 1, 4)
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_MUL
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mul_mode_permute_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = mul(x, y, mode_x=(1, 0), mode_y=(0,), mode_out=(1, 0))
    expected = x * y.view(1, 4)
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_MAX
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_MAX_TEST_SHAPES)
def test_max_correctness(dtype, shape):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = max(x, y)
    expected = torch.maximum(x, y)
    assert_close(z, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorMax(dtype=dtype)
        z_base = baseline(x, y)
        assert_close(z_base, expected, dtype)
        assert_close(z, z_base, dtype)


@pytest.mark.CUTENSOR_OP_MAX
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_max_broadcast_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((2, 3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = max(x, y, mode_x=(0, 1, 2), mode_y=(2,), mode_out=(0, 1, 2))
    expected = torch.maximum(x, y.view(1, 1, 4))
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_MAX
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_max_mode_permute_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = max(x, y, mode_x=(1, 0), mode_y=(0,), mode_out=(1, 0))
    expected = torch.maximum(x, y.view(1, 4))
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_MIN
@pytest.mark.parametrize("dtype", DEFAULT_CORRECTNESS_DTYPES)
@pytest.mark.parametrize("shape", DEFAULT_MIN_TEST_SHAPES)
def test_min_correctness(dtype, shape):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty(shape, device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = min(x, y)
    expected = torch.minimum(x, y)
    assert_close(z, expected, dtype)

    if CUTENSOR_AVAILABLE:
        baseline = CuTensorMin(dtype=dtype)
        z_base = baseline(x, y)
        assert_close(z_base, expected, dtype)
        assert_close(z, z_base, dtype)


@pytest.mark.CUTENSOR_OP_MIN
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_min_broadcast_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((2, 3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = min(x, y, mode_x=(0, 1, 2), mode_y=(2,), mode_out=(0, 1, 2))
    expected = torch.minimum(x, y.view(1, 1, 4))
    assert_close(z, expected, dtype)


@pytest.mark.CUTENSOR_OP_MIN
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_min_mode_permute_correctness(dtype):
    if not _is_accelerator_available():
        pytest.skip("Accelerator unavailable")

    x = torch.empty((3, 4), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    y = torch.empty((4,), device=_device_str, dtype=dtype).uniform_(-8.0, 8.0)
    z = min(x, y, mode_x=(1, 0), mode_y=(0,), mode_out=(1, 0))
    expected = torch.minimum(x, y.view(1, 4))
    assert_close(z, expected, dtype)
