import pytest
import torch


def _mthreads_available():
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        pass
    musa = getattr(torch, "musa", None)
    return musa is not None and musa.is_available()


pytestmark = pytest.mark.skipif(
    not _mthreads_available(),
    reason="MThreads MUSA backend is not active",
)


def test_mthreads_torch_baseline_available():
    from flagtensor.torch_npu_baseline import torch_npu_available

    assert torch_npu_available()


@pytest.mark.CUTENSOR_OP_ADD
def test_mthreads_add_smoke():
    from flagtensor import add
    from flagtensor.runtime import device_str
    from flagtensor.testing import assert_close
    from flagtensor.torch_npu_baseline import CuTensorAdd

    x = torch.empty((1024,), device=device_str, dtype=torch.float32).uniform_(-1, 1)
    y = torch.empty((1024,), device=device_str, dtype=torch.float32).uniform_(-1, 1)

    expected = torch.add(x, y)
    baseline = CuTensorAdd(dtype=x.dtype)

    assert_close(baseline(x, y), expected, x.dtype)
    assert_close(add(x, y), expected, x.dtype)


@pytest.mark.parametrize(
    ("name", "lower", "upper"),
    [
        ("acos", -0.95, 0.95),
        ("acosh", 1.01, 4.0),
        ("asin", -0.95, 0.95),
        ("asinh", -4.0, 4.0),
        ("atan", -4.0, 4.0),
        ("atanh", -0.95, 0.95),
        ("tan", -1.0, 1.0),
    ],
)
def test_mthreads_unary_rewrites(name, lower, upper):
    import flagtensor
    from flagtensor.runtime import device_str
    from flagtensor.testing import assert_close

    x = torch.linspace(
        lower,
        upper,
        1024,
        device=device_str,
        dtype=torch.float32,
    )
    expected = getattr(torch, name)(x)
    actual = getattr(flagtensor, name)(x)
    assert_close(actual, expected, x.dtype)
