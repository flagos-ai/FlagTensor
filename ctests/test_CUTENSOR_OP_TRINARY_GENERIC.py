import pytest
import torch

from flagtensor import trinary
from flagtensor.cutensor import CUTENSOR_AVAILABLE, trinary as cutensor_trinary
from flagtensor.ops import CUTENSOR_OP_TRINARY_GENERIC as trinary_module
from flagtensor.testing import assert_close


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_trinary_generic_mul_add(dtype):
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.empty((128, 64), device="cuda", dtype=dtype).uniform_(-4.0, 4.0)
    b = torch.empty((128, 64), device="cuda", dtype=dtype).uniform_(-4.0, 4.0)
    c = torch.empty((128, 64), device="cuda", dtype=dtype).uniform_(-4.0, 4.0)
    out = trinary(a, b, c, op_ab="mul", op_abc="add")
    expected = (a.float() * b.float() + c.float()).to(dtype) if dtype == torch.float16 else a * b + c
    assert_close(out, expected, dtype)
    out_base = cutensor_trinary(a, b, c, op_ab="mul", op_abc="add")
    assert_close(out_base, expected, dtype)
    assert_close(out, out_base, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_trinary_generic_with_unary_and_scalars(dtype):
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.empty((256,), device="cuda", dtype=dtype).uniform_(0.5, 4.0)
    b = torch.empty((256,), device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    c = torch.empty((256,), device="cuda", dtype=dtype).uniform_(0.5, 4.0)
    out = trinary(
        a,
        b,
        c,
        op_a="log",
        op_b="neg",
        op_c="sqrt",
        op_ab="add",
        op_abc="max",
        alpha=1.5,
        beta=0.5,
        gamma=2.0,
    )
    ref_a = 1.5 * torch.log(a.float())
    ref_b = 0.5 * (-b.float())
    ref_c = 2.0 * torch.sqrt(c.float())
    expected = torch.maximum(ref_a + ref_b, ref_c).to(dtype) if dtype == torch.float16 else torch.maximum(ref_a.to(dtype) + ref_b.to(dtype), ref_c.to(dtype))
    assert_close(out, expected, dtype)
    out_base = cutensor_trinary(
        a,
        b,
        c,
        op_a="log",
        op_b="neg",
        op_c="sqrt",
        op_ab="add",
        op_abc="max",
        alpha=1.5,
        beta=0.5,
        gamma=2.0,
    )
    assert_close(out_base, expected, dtype)
    assert_close(out, out_base, dtype)


def test_trinary_generic_broadcast_and_modes():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    b = torch.arange(4, device="cuda", dtype=torch.float32).reshape(4)
    c = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    out = trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="add",
        mode_a=(1, 0),
        mode_b=(0,),
        mode_c=(1, 0),
        mode_d=(1, 0),
    )
    expected = a + b.view(1, 4) + c
    assert_close(out, expected, torch.float32)
    out_base = cutensor_trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="add",
        mode_a=(1, 0),
        mode_b=(0,),
        mode_c=(1, 0),
        mode_d=(1, 0),
    )
    assert_close(out_base, expected, torch.float32)
    assert_close(out, out_base, torch.float32)


def test_trinary_generic_composed_path_uses_custom_binary_impl(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    calls = []
    original_load_binary_impl = trinary_module._load_binary_impl

    def wrapped_load_binary_impl(name):
        impl = original_load_binary_impl(name)

        def wrapped(*args, **kwargs):
            calls.append(name)
            return impl(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(trinary_module, "_load_binary_impl", wrapped_load_binary_impl)

    a = torch.empty((32,), device="cuda", dtype=torch.float32).uniform_(0.2, 1.5)
    b = torch.empty((32,), device="cuda", dtype=torch.float32).uniform_(0.2, 1.5)
    c = torch.empty((32,), device="cuda", dtype=torch.float32).uniform_(0.2, 1.5)

    out = trinary(
        a,
        b,
        c,
        op_a="sin",
        op_b="cos",
        op_c="tan",
        op_ab="add",
        op_abc="max",
    )
    expected = torch.maximum(torch.sin(a) + torch.cos(b), torch.tan(c))
    assert_close(out, expected, torch.float32)
    assert calls == ["add", "max"]


def test_trinary_generic_mode_c_differs_from_mode_d():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    b = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    c = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 2, 3)
    out = trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="add",
        mode_a=(0, 1, 2),
        mode_b=(0, 1, 2),
        mode_c=(2, 0, 1),
        mode_d=(0, 1, 2),
    )
    expected = a + b + c.permute(1, 2, 0)
    assert_close(out, expected, torch.float32)


def test_trinary_generic_c_broadcast_to_output_shape():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    b = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    c = torch.arange(4, device="cuda", dtype=torch.float32).reshape(4)
    out = trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="max",
        mode_a=(0, 1, 2),
        mode_b=(0, 1, 2),
        mode_c=(2,),
        mode_d=(0, 1, 2),
    )
    expected = torch.maximum(a + b, c.view(1, 1, 4))
    assert_close(out, expected, torch.float32)


def test_trinary_generic_high_rank_indexed_fused_scope():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(2 * 3 * 4 * 5 * 2, device="cuda", dtype=torch.float32).reshape(2, 3, 4, 5, 2)
    b = torch.arange(3 * 4 * 2 * 2 * 5, device="cuda", dtype=torch.float32).reshape(3, 4, 2, 2, 5)
    c = torch.arange(4 * 5 * 2 * 2 * 3, device="cuda", dtype=torch.float32).reshape(4, 5, 2, 2, 3)
    out = trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="max",
        mode_a=(0, 1, 2, 3, 4),
        mode_b=(1, 2, 4, 0, 3),
        mode_c=(2, 3, 4, 0, 1),
        mode_d=(2, 3, 4, 0, 1),
    )
    a_ref = a.permute(2, 3, 4, 0, 1)
    b_ref = b.permute(1, 4, 2, 3, 0)
    c_ref = c
    expected = torch.maximum(a_ref + b_ref, c_ref)
    assert_close(out, expected, torch.float32)
    out_base = cutensor_trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="max",
        mode_a=(0, 1, 2, 3, 4),
        mode_b=(1, 2, 4, 0, 3),
        mode_c=(2, 3, 4, 0, 1),
        mode_d=(2, 3, 4, 0, 1),
    )
    assert_close(out_base, expected, torch.float32)
    assert_close(out, out_base, torch.float32)


def test_trinary_generic_complex_broadcast_indexed_fused_scope():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(2 * 1 * 4 * 1 * 3, device="cuda", dtype=torch.float32).reshape(2, 1, 4, 1, 3)
    b = torch.arange(1 * 4 * 1 * 2 * 3, device="cuda", dtype=torch.float32).reshape(1, 4, 1, 2, 3)
    c = torch.arange(4 * 2 * 3 * 2 * 1, device="cuda", dtype=torch.float32).reshape(4, 2, 3, 2, 1)
    out = trinary(
        a,
        b,
        c,
        op_a="relu",
        op_b="identity",
        op_c="identity",
        op_ab="add",
        op_abc="mul",
        mode_a=(0, 4, 1, 3, 2),
        mode_b=(4, 1, 3, 0, 2),
        mode_c=(1, 0, 2, 3, 4),
        mode_d=(1, 0, 2, 3, 4),
    )
    a_ref = torch.relu(a).permute(2, 0, 4, 3, 1)
    b_ref = b.permute(1, 3, 4, 2, 0)
    c_ref = c
    expected = (a_ref + b_ref) * c_ref
    assert_close(out, expected, torch.float32)
    out_base = cutensor_trinary(
        a,
        b,
        c,
        op_a="relu",
        op_b="identity",
        op_c="identity",
        op_ab="add",
        op_abc="mul",
        mode_a=(0, 4, 1, 3, 2),
        mode_b=(4, 1, 3, 0, 2),
        mode_c=(1, 0, 2, 3, 4),
        mode_d=(1, 0, 2, 3, 4),
    )
    assert_close(out_base, expected, torch.float32)
    assert_close(out, out_base, torch.float32)


def test_trinary_generic_permutation_consistency():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    b = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    c = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    out = trinary(
        a,
        b,
        c,
        op_ab="mul",
        op_abc="add",
        mode_a=(2, 0, 1),
        mode_b=(2, 0, 1),
        mode_c=(2, 0, 1),
        mode_d=(2, 0, 1),
    )
    expected = a * b + c
    assert_close(out, expected, torch.float32)
    out_base = cutensor_trinary(
        a,
        b,
        c,
        op_ab="mul",
        op_abc="add",
        mode_a=(2, 0, 1),
        mode_b=(2, 0, 1),
        mode_c=(2, 0, 1),
        mode_d=(2, 0, 1),
    )
    assert_close(out_base, expected, torch.float32)
    assert_close(out, out_base, torch.float32)


def test_trinary_generic_canonical_broadcast_fused_scope():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    b = torch.arange(4, device="cuda", dtype=torch.float32).reshape(1, 4)
    c = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    out = trinary(a, b, c, op_a="relu", op_b="identity", op_c="identity", op_ab="add", op_abc="max")
    expected = torch.maximum(torch.relu(a) + b, c)
    assert_close(out, expected, torch.float32)
    out_base = cutensor_trinary(a, b, c, op_a="relu", op_b="identity", op_c="identity", op_ab="add", op_abc="max")
    assert_close(out_base, expected, torch.float32)
    assert_close(out, out_base, torch.float32)


def test_trinary_generic_partial_permutation_fused_scope():
    if not torch.cuda.is_available() or not CUTENSOR_AVAILABLE:
        pytest.skip("CUDA/cuTensor unavailable")

    a = torch.arange(24, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    b = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 2, 3)
    c = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 4, 2)
    out = trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="mul",
        mode_a=(0, 1, 2),
        mode_b=(2, 0, 1),
        mode_c=(1, 2, 0),
        mode_d=(1, 2, 0),
    )
    a_ref = a.permute(1, 2, 0)
    b_ref = b.permute(2, 0, 1)
    c_ref = c
    expected = (a_ref + b_ref) * c_ref
    assert_close(out, expected, torch.float32)
    out_base = cutensor_trinary(
        a,
        b,
        c,
        op_ab="add",
        op_abc="mul",
        mode_a=(0, 1, 2),
        mode_b=(2, 0, 1),
        mode_c=(1, 2, 0),
        mode_d=(1, 2, 0),
    )
    assert_close(out_base, expected, torch.float32)
    assert_close(out, out_base, torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_trinary_composed_path_numerical(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    torch.manual_seed(0)
    a = torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(0.2, 1.5)
    b = torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(0.2, 1.5)
    c = torch.empty((64, 32), device="cuda", dtype=dtype).uniform_(0.2, 1.5)

    alpha, beta, gamma = 1.5, -0.75, 0.5
    out = trinary(
        a,
        b,
        c,
        op_a="sin",
        op_b="cos",
        op_c="tan",
        op_ab="add",
        op_abc="max",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    expected = torch.maximum(
        alpha * torch.sin(a.float()) + beta * torch.cos(b.float()),
        gamma * torch.tan(c.float()),
    ).to(dtype)
    assert_close(out, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_trinary_composed_path_broadcast(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    torch.manual_seed(1)
    a = torch.empty((4, 8, 16), device="cuda", dtype=dtype).uniform_(0.2, 1.5)
    b = torch.empty((8, 16), device="cuda", dtype=dtype).uniform_(0.2, 1.5)
    c = torch.empty((16,), device="cuda", dtype=dtype).uniform_(0.2, 1.5)

    alpha, beta, gamma = 2.0, 0.25, -1.25
    out = trinary(
        a,
        b,
        c,
        op_a="sin",
        op_b="cos",
        op_c="tan",
        op_ab="add",
        op_abc="min",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        mode_a=(0, 1, 2),
        mode_b=(1, 2),
        mode_c=(2,),
        mode_d=(0, 1, 2),
    )
    expected = torch.minimum(
        alpha * torch.sin(a.float()) + beta * torch.cos(b.float()).view(1, 8, 16),
        gamma * torch.tan(c.float()).view(1, 1, 16),
    ).to(dtype)
    assert_close(out, expected, dtype)


def test_trinary_fused_support_set_locked():
    from flagtensor.cutensor import (
        CUTENSOR_OP_ABS,
        CUTENSOR_OP_ADD,
        CUTENSOR_OP_EXP,
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_LOG,
        CUTENSOR_OP_MAX,
        CUTENSOR_OP_MIN,
        CUTENSOR_OP_MUL,
        CUTENSOR_OP_NEG,
        CUTENSOR_OP_RELU,
        CUTENSOR_OP_SIGMOID,
        CUTENSOR_OP_SQRT,
        CUTENSOR_OP_TANH,
    )

    expected_unary = frozenset({
        CUTENSOR_OP_IDENTITY,
        CUTENSOR_OP_NEG,
        CUTENSOR_OP_RELU,
        CUTENSOR_OP_SIGMOID,
        CUTENSOR_OP_TANH,
        CUTENSOR_OP_ABS,
        CUTENSOR_OP_EXP,
        CUTENSOR_OP_LOG,
        CUTENSOR_OP_SQRT,
    })
    expected_binary = frozenset({
        CUTENSOR_OP_ADD,
        CUTENSOR_OP_MUL,
        CUTENSOR_OP_MAX,
        CUTENSOR_OP_MIN,
    })

    assert trinary_module._FUSED_KERNEL_UNARY_OPS == expected_unary, (
        "fused trinary unary support set drifted; update the kernel branches "
        "and this lock test together"
    )
    assert trinary_module._FUSED_KERNEL_BINARY_OPS == expected_binary, (
        "fused trinary binary support set drifted; update the kernel branches "
        "and this lock test together"
    )


def test_trinary_validate_fused_codes_rejects_unknown():
    from flagtensor.cutensor import CUTENSOR_OP_ADD, CUTENSOR_OP_IDENTITY

    with pytest.raises(AssertionError, match="unary op code"):
        trinary_module._validate_fused_codes(
            CUTENSOR_OP_IDENTITY,
            CUTENSOR_OP_IDENTITY,
            9999,
            CUTENSOR_OP_ADD,
            CUTENSOR_OP_ADD,
        )
    with pytest.raises(AssertionError, match="binary op code"):
        trinary_module._validate_fused_codes(
            CUTENSOR_OP_IDENTITY,
            CUTENSOR_OP_IDENTITY,
            CUTENSOR_OP_IDENTITY,
            9999,
            CUTENSOR_OP_ADD,
        )


def test_trinary_rank_limit_raises():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    # 17 distinct modes -> output rank 17 > _MAX_SUPPORTED_OUTPUT_RANK (16).
    modes_a = tuple(range(17))
    a = torch.zeros((1,) * 17, device="cuda", dtype=torch.float32)
    b = torch.zeros((1,) * 17, device="cuda", dtype=torch.float32)
    c = torch.zeros((1,) * 17, device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="exceeds the supported limit"):
        trinary(
            a,
            b,
            c,
            mode_a=modes_a,
            mode_b=modes_a,
            mode_c=modes_a,
            mode_d=modes_a,
        )


def test_trinary_composed_path_does_not_mutate_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    torch.manual_seed(2)
    a = torch.empty((128,), device="cuda", dtype=torch.float32).uniform_(0.2, 1.5)
    b = torch.empty((128,), device="cuda", dtype=torch.float32).uniform_(0.2, 1.5)
    c = torch.empty((128,), device="cuda", dtype=torch.float32).uniform_(0.2, 1.5)
    a_ref = a.clone()
    b_ref = b.clone()
    c_ref = c.clone()

    # identity + non-1.0 scale forces the "x_is_owned=False" branch of
    # _apply_unary_with_scale in fp32 (no upcast). We must NOT mutate inputs.
    out = trinary(
        a,
        b,
        c,
        op_a="identity",
        op_b="sin",
        op_c="cos",
        op_ab="add",
        op_abc="max",
        alpha=2.0,
        beta=0.5,
        gamma=-0.25,
    )
    assert torch.equal(a, a_ref), "composed path mutated input a"
    assert torch.equal(b, b_ref), "composed path mutated input b"
    assert torch.equal(c, c_ref), "composed path mutated input c"

    expected = torch.maximum(2.0 * a_ref + 0.5 * torch.sin(b_ref), -0.25 * torch.cos(c_ref))
    assert_close(out, expected, torch.float32)
