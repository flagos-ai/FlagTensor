import pytest
import torch
from flagtensor import trinary

@pytest.mark.trinary_generic
def test_trinary_generic_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(trinary), "trinary_generic should be callable"
