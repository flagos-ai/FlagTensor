import pytest
import torch

from flagtensor import add
from flagtensor.runtime import (
    device,
    device_str,
    is_accelerator_available,
)
from flagtensor.testing import assert_close
from flagtensor.torch_npu_baseline import CuTensorAdd
from flagtensor.torch_npu_baseline import torch_npu_available


pytestmark = pytest.mark.skipif(
    not is_accelerator_available() or device.vendor_name != "mthreads",
    reason="MThreads MUSA backend is not active",
)


def test_mthreads_torch_baseline_available():
    assert torch_npu_available()


@pytest.mark.CUTENSOR_OP_ADD
def test_mthreads_add_smoke():
    x = torch.empty((1024,), device=device_str, dtype=torch.float32).uniform_(-1, 1)
    y = torch.empty((1024,), device=device_str, dtype=torch.float32).uniform_(-1, 1)

    expected = torch.add(x, y)
    baseline = CuTensorAdd(dtype=x.dtype)

    assert_close(baseline(x, y), expected, x.dtype)
    assert_close(add(x, y), expected, x.dtype)
