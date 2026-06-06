import pytest
import torch
from flagtensor import gett

@pytest.mark.gett
def test_gett_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(gett), "gett should be callable"
