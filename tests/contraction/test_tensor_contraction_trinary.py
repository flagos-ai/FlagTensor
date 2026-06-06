import pytest
import torch
from flagtensor import tensor_contraction_trinary

@pytest.mark.tensor_contraction_trinary
def test_tensor_contraction_trinary_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(tensor_contraction_trinary), "tensor_contraction_trinary should be callable"
