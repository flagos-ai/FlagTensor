import pytest
import torch
from flagtensor import elementwise_trinary

@pytest.mark.ElementwiseTrinary
def test_elementwise_trinary_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(elementwise_trinary), "elementwise_trinary should be callable"
