import pytest
import torch
from flagtensor import ttgt

@pytest.mark.ttgt
def test_ttgt_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(ttgt), "ttgt should be callable"
