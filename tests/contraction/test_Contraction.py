import pytest
import torch
from flagtensor import contraction

@pytest.mark.Contraction
def test_contraction_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(contraction), "contraction should be callable"
