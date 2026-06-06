import pytest
import torch
from flagtensor import tgett

@pytest.mark.tgett
def test_tgett_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(tgett), "tgett should be callable"
