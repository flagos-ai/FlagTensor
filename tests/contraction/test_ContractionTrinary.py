import pytest
import torch
from flagtensor import contraction_trinary

@pytest.mark.ContractionTrinary
def test_contraction_trinary_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(contraction_trinary), "contraction_trinary should be callable"
