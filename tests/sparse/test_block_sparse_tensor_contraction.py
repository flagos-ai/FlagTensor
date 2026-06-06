import pytest
import torch
from flagtensor import block_sparse_tensor_contraction

@pytest.mark.block_sparse_tensor_contraction
def test_block_sparse_tensor_contraction_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(block_sparse_tensor_contraction), "block_sparse_tensor_contraction should be callable"
