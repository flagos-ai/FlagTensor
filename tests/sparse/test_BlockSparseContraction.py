import pytest
import torch
from flagtensor import block_sparse_contraction

@pytest.mark.BlockSparseContraction
def test_block_sparse_contraction_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert callable(block_sparse_contraction), "block_sparse_contraction should be callable"
