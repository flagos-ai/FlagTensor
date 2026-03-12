from typing import Iterable

import torch


def assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype):
    atol = 1e-5 if dtype != torch.float16 else 1e-3
    rtol = 1e-5 if dtype != torch.float16 else 1e-3
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol)


def default_identity_shapes() -> Iterable[tuple]:
    return [(1024,), (4096,), (128, 128), (32, 64, 16)]
