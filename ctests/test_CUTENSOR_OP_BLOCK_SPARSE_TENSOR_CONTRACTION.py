import pytest
import torch

from flagtensor import BlockSparseTensor
from flagtensor import BlockSparseTensorContraction
from flagtensor import BlockSparseTensorDescriptor
from flagtensor import block_sparse_tensor_contraction
from flagtensor.config import DEFAULT_BLOCK_SPARSE_TENSOR_CONTRACTION_TEST_SHAPES
from flagtensor.testing import assert_close


def _make_block_sparse(shape, block_shape, dtype, device):
    block_h, block_w = block_shape
    blocks = {}
    num_block_rows = shape[0] // block_h
    num_block_cols = shape[1] // block_w
    for i in range(num_block_rows):
        for j in range(num_block_cols):
            if (i + j) % 2 == 0:
                blocks[(i, j)] = torch.empty(block_shape, device=device, dtype=dtype).uniform_(-2.0, 2.0)
    desc = BlockSparseTensorDescriptor(
        shape=shape,
        block_shape=block_shape,
        nonzero_coordinates=tuple(sorted(blocks.keys())),
    )
    return BlockSparseTensor(desc, blocks)


def _make_block_sparse_from_sections(shape, section_extents, coords, dtype, device):
    desc = BlockSparseTensorDescriptor(
        shape=shape,
        num_sections_per_mode=tuple(len(mode_extents) for mode_extents in section_extents),
        section_extents=section_extents,
        nonzero_coordinates=coords,
    )
    blocks = {}
    for coord in coords:
        block_shape = tuple(section_extents[mode][index] for mode, index in enumerate(coord))
        real = torch.empty(block_shape, device=device, dtype=torch.float32).uniform_(-2.0, 2.0)
        if dtype in (torch.complex64, torch.complex128):
            imag = torch.empty(block_shape, device=device, dtype=torch.float32).uniform_(-2.0, 2.0)
            block = torch.complex(real, imag).to(dtype)
        else:
            block = real.to(dtype)
        blocks[coord] = block
    return BlockSparseTensor(desc, blocks)


def _output_mask(tensor: BlockSparseTensor):
    mask = torch.zeros(tensor.shape, device=tensor.device, dtype=torch.float32)
    offsets_per_mode = []
    for mode_extents in tensor.descriptor.section_extents:
        offsets = [0]
        for extent in mode_extents:
            offsets.append(offsets[-1] + extent)
        offsets_per_mode.append(offsets)
    for coord in tensor.descriptor.canonical_nonzero_coordinates:
        slices = tuple(
            slice(offsets_per_mode[mode][index], offsets_per_mode[mode][index + 1])
            for mode, index in enumerate(coord)
        )
        mask[slices] = 1.0
    return mask


def _block_sparse_reference(a, b, c, alpha=1.25, beta=0.5):
    out = alpha * torch.matmul(a.to_dense(), b.to_dense()) + beta * c.to_dense()
    mask = _output_mask(c)
    if out.dtype.is_complex:
        mask = mask.to(out.real.dtype)
    out = out * mask.to(out.dtype)
    return out


def _block_sparse_reference_modes(a, b, c, mode_a, mode_b, mode_d, alpha=1.25, beta=0.5):
    labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_map = {}

    def _encode(modes):
        encoded = []
        for mode in modes:
            if mode not in label_map:
                label_map[mode] = labels[len(label_map)]
            encoded.append(label_map[mode])
        return "".join(encoded)

    expr = f"{_encode(mode_a)},{_encode(mode_b)}->{_encode(mode_d)}"
    out = alpha * torch.einsum(expr, a.to_dense(), b.to_dense()) + beta * c.to_dense()
    mask = _output_mask(c)
    if out.dtype.is_complex:
        mask = mask.to(out.real.dtype)
    return out * mask.to(out.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape_a,shape_b", DEFAULT_BLOCK_SPARSE_TENSOR_CONTRACTION_TEST_SHAPES)
def test_block_sparse_tensor_contraction_correctness(dtype, shape_a, shape_b):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    block_k = 4 if shape_a[1] % 4 == 0 and shape_b[0] % 4 == 0 else shape_a[1]
    block_shape_a = (shape_a[0] // 2, block_k)
    block_shape_b = (block_k, shape_b[1] // 2)
    a = _make_block_sparse(shape_a, block_shape_a, dtype, "cuda")
    b = _make_block_sparse(shape_b, block_shape_b, dtype, "cuda")
    c_coords = ((0, 0), (1, 1))
    c_desc = BlockSparseTensorDescriptor(
        shape=(shape_a[0], shape_b[1]),
        block_shape=(block_shape_a[0], block_shape_b[1]),
        nonzero_coordinates=c_coords,
    )
    c = BlockSparseTensor(
        c_desc,
        {
            (0, 0): torch.empty(c_desc.block_shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
            (1, 1): torch.empty(c_desc.block_shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0),
        },
    )

    out = block_sparse_tensor_contraction(a, b, c=c, alpha=1.25, beta=0.5)
    expected = _block_sparse_reference(a, b, c, alpha=1.25, beta=0.5).to(dtype)
    assert_close(out, expected, dtype)

    baseline = BlockSparseTensorContraction()
    out_base = baseline(a, b, c=c, alpha=1.25, beta=0.5)
    assert_close(out_base, expected, dtype)
    assert_close(out, out_base, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_block_sparse_tensor_contraction_irregular_sections(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    a = _make_block_sparse_from_sections(
        (8, 9),
        ((3, 5), (2, 4, 3)),
        ((0, 0), (0, 2), (1, 1)),
        dtype,
        "cuda",
    )
    b = _make_block_sparse_from_sections(
        (9, 7),
        ((2, 4, 3), (3, 4)),
        ((0, 1), (1, 0), (2, 1)),
        dtype,
        "cuda",
    )
    c = _make_block_sparse_from_sections(
        (8, 7),
        ((3, 5), (3, 4)),
        ((0, 1), (1, 0)),
        dtype,
        "cuda",
    )

    out = block_sparse_tensor_contraction(a, b, c=c, alpha=1.25, beta=0.5)
    expected = _block_sparse_reference(a, b, c, alpha=1.25, beta=0.5)
    assert_close(out, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_block_sparse_tensor_contraction_complex_pattern(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    a = _make_block_sparse_from_sections(
        (10, 12),
        ((2, 3, 5), (4, 3, 5)),
        ((0, 1), (1, 0), (1, 2), (2, 1)),
        dtype,
        "cuda",
    )
    b = _make_block_sparse_from_sections(
        (12, 9),
        ((4, 3, 5), (2, 4, 3)),
        ((0, 0), (1, 2), (2, 1)),
        dtype,
        "cuda",
    )
    c = _make_block_sparse_from_sections(
        (10, 9),
        ((2, 3, 5), (2, 4, 3)),
        ((0, 0), (1, 2), (2, 1)),
        dtype,
        "cuda",
    )

    out = block_sparse_tensor_contraction(a, b, c=c, alpha=0.75, beta=-0.25)
    expected = _block_sparse_reference(a, b, c, alpha=0.75, beta=-0.25)
    assert_close(out, expected, dtype)


def test_block_sparse_tensor_contraction_only_materializes_output_pattern():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    a = _make_block_sparse_from_sections(
        (8, 8),
        ((4, 4), (4, 4)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        torch.float32,
        "cuda",
    )
    b = _make_block_sparse_from_sections(
        (8, 8),
        ((4, 4), (4, 4)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        torch.float32,
        "cuda",
    )
    c = _make_block_sparse_from_sections(
        (8, 8),
        ((4, 4), (4, 4)),
        ((0, 0),),
        torch.float32,
        "cuda",
    )

    out = block_sparse_tensor_contraction(a, b, c=c, alpha=1.0, beta=0.0)
    dense_expected = torch.matmul(a.to_dense(), b.to_dense())
    masked_expected = _block_sparse_reference(a, b, c, alpha=1.0, beta=0.0)

    assert torch.count_nonzero(dense_expected[4:, 4:]).item() > 0
    assert_close(out, masked_expected, torch.float32)
    assert torch.count_nonzero(out[4:, :]).item() == 0
    assert torch.count_nonzero(out[:, 4:]).item() == 0


def test_block_sparse_descriptor_section_schema():
    desc = BlockSparseTensorDescriptor(
        shape=(8, 10),
        num_sections_per_mode=(2, 2),
        section_extents=((3, 5), (4, 6)),
        nonzero_coordinates=((0, 0), (1, 1)),
    )
    assert desc.canonical_section_extents == ((3, 5), (4, 6))
    assert desc.canonical_nonzero_coordinates == ((0, 0), (1, 1))


def test_block_sparse_descriptor_nd_schema_and_to_dense():
    desc = BlockSparseTensorDescriptor(
        shape=(5, 6, 7),
        num_sections_per_mode=(2, 2, 2),
        section_extents=((2, 3), (1, 5), (4, 3)),
        nonzero_coordinates=((0, 0, 0), (1, 1, 1)),
    )
    assert desc.canonical_section_extents == ((2, 3), (1, 5), (4, 3))
    assert desc.canonical_nonzero_coordinates == ((0, 0, 0), (1, 1, 1))

    tensor = BlockSparseTensor(
        desc,
        {
            (0, 0, 0): torch.ones((2, 1, 4), device="cuda", dtype=torch.float32),
            (1, 1, 1): 2 * torch.ones((3, 5, 3), device="cuda", dtype=torch.float32),
        },
    )
    dense = tensor.to_dense()
    assert dense.shape == (5, 6, 7)
    assert torch.all(dense[:2, :1, :4] == 1)
    assert torch.all(dense[2:, 1:, 4:] == 2)
    assert torch.count_nonzero(dense[:2, 1:, :]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_block_sparse_tensor_contraction_nd_modes(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    a = _make_block_sparse_from_sections(
        (5, 6, 7),
        ((2, 3), (2, 4), (3, 4)),
        ((0, 0, 0), (0, 1, 1), (1, 1, 0)),
        dtype,
        "cuda",
    )
    b = _make_block_sparse_from_sections(
        (7, 6, 4),
        ((3, 4), (2, 4), (1, 3)),
        ((0, 0, 0), (1, 1, 1), (1, 0, 1)),
        dtype,
        "cuda",
    )
    c = _make_block_sparse_from_sections(
        (5, 4),
        ((2, 3), (1, 3)),
        ((0, 0), (1, 1)),
        dtype,
        "cuda",
    )

    mode_a = (0, 1, 2)
    mode_b = (2, 1, 3)
    mode_d = (0, 3)

    out = block_sparse_tensor_contraction(
        a,
        b,
        c=c,
        alpha=1.25,
        beta=0.5,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_d,
        mode_d=mode_d,
    )
    expected = _block_sparse_reference_modes(a, b, c, mode_a, mode_b, mode_d, alpha=1.25, beta=0.5)
    assert_close(out, expected, dtype)

    baseline = BlockSparseTensorContraction()
    out_base = baseline(
        a,
        b,
        c=c,
        alpha=1.25,
        beta=0.5,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_d,
        mode_d=mode_d,
    )
    assert_close(out_base, expected, dtype)


def test_block_sparse_tensor_contraction_nd_requires_explicit_modes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    a = _make_block_sparse_from_sections(
        (4, 4, 4),
        ((2, 2), (2, 2), (2, 2)),
        ((0, 0, 0),),
        torch.float32,
        "cuda",
    )
    b = _make_block_sparse_from_sections(
        (4, 4, 4),
        ((2, 2), (2, 2), (2, 2)),
        ((0, 0, 0),),
        torch.float32,
        "cuda",
    )
    c = _make_block_sparse_from_sections(
        (4, 4),
        ((2, 2), (2, 2)),
        ((0, 0),),
        torch.float32,
        "cuda",
    )

    with pytest.raises(ValueError):
        block_sparse_tensor_contraction(a, b, c=c)


def test_block_sparse_descriptor_rejects_mismatched_coordinates():
    desc = BlockSparseTensorDescriptor(
        shape=(8, 8),
        block_shape=(4, 4),
        nonzero_coordinates=((0, 0),),
    )
    with pytest.raises(ValueError):
        BlockSparseTensor(
            desc,
            {
                (0, 0): torch.ones((4, 4), device="cuda", dtype=torch.float32),
                (1, 1): torch.ones((4, 4), device="cuda", dtype=torch.float32),
            },
        )
