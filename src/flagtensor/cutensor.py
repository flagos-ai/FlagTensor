# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
from dataclasses import dataclass
from ctypes import POINTER, byref, c_double, c_float, c_int, c_int32, c_int64, c_uint32, c_uint64, c_void_p

import torch

CUDA_R_16F = 2
CUDA_R_32F = 0
CUDA_R_64F = 1
CUDA_R_16BF = 14
CUDA_C_32F = 4
CUDA_C_64F = 5

CUTENSOR_OP_IDENTITY = 1
CUTENSOR_OP_SQRT = 2
CUTENSOR_OP_RELU = 8
CUTENSOR_OP_CONJ = 9
CUTENSOR_OP_RCP = 10
CUTENSOR_OP_SIGMOID = 11
CUTENSOR_OP_TANH = 12
CUTENSOR_OP_ABS = 24
CUTENSOR_OP_EXP = 22
CUTENSOR_OP_LOG = 23
CUTENSOR_OP_NEG = 25
CUTENSOR_OP_SIN = 26
CUTENSOR_OP_COS = 27
CUTENSOR_OP_TAN = 28
CUTENSOR_OP_SINH = 29
CUTENSOR_OP_COSH = 30
CUTENSOR_OP_ASIN = 31
CUTENSOR_OP_ACOS = 32
CUTENSOR_OP_ATAN = 33
CUTENSOR_OP_ASINH = 34
CUTENSOR_OP_ACOSH = 35
CUTENSOR_OP_ATANH = 36
CUTENSOR_OP_CEIL = 37
CUTENSOR_OP_FLOOR = 38
CUTENSOR_OP_MISH = 39
CUTENSOR_OP_SWISH = 40
CUTENSOR_OP_SOFT_PLUS = 41
CUTENSOR_OP_SOFT_SIGN = 42
CUTENSOR_OP_ADD = 3
CUTENSOR_OP_MUL = 5
CUTENSOR_OP_MAX = 6
CUTENSOR_OP_MIN = 7
CUTENSOR_ALGO_DEFAULT = -1
CUTENSOR_JIT_MODE_NONE = 0
CUTENSOR_WORKSPACE_DEFAULT = 2

UNARY_OPERATOR_MAP = {
    "identity": CUTENSOR_OP_IDENTITY,
    "sqrt": CUTENSOR_OP_SQRT,
    "relu": CUTENSOR_OP_RELU,
    "conj": CUTENSOR_OP_CONJ,
    "rcp": CUTENSOR_OP_RCP,
    "sigmoid": CUTENSOR_OP_SIGMOID,
    "tanh": CUTENSOR_OP_TANH,
    "abs": CUTENSOR_OP_ABS,
    "exp": CUTENSOR_OP_EXP,
    "log": CUTENSOR_OP_LOG,
    "neg": CUTENSOR_OP_NEG,
    "sin": CUTENSOR_OP_SIN,
    "cos": CUTENSOR_OP_COS,
    "tan": CUTENSOR_OP_TAN,
    "sinh": CUTENSOR_OP_SINH,
    "cosh": CUTENSOR_OP_COSH,
    "asin": CUTENSOR_OP_ASIN,
    "acos": CUTENSOR_OP_ACOS,
    "atan": CUTENSOR_OP_ATAN,
    "asinh": CUTENSOR_OP_ASINH,
    "acosh": CUTENSOR_OP_ACOSH,
    "atanh": CUTENSOR_OP_ATANH,
    "ceil": CUTENSOR_OP_CEIL,
    "floor": CUTENSOR_OP_FLOOR,
    "mish": CUTENSOR_OP_MISH,
    "swish": CUTENSOR_OP_SWISH,
    "soft_plus": CUTENSOR_OP_SOFT_PLUS,
    "soft_sign": CUTENSOR_OP_SOFT_SIGN,
}

BINARY_OPERATOR_MAP = {
    "add": CUTENSOR_OP_ADD,
    "mul": CUTENSOR_OP_MUL,
    "max": CUTENSOR_OP_MAX,
    "min": CUTENSOR_OP_MIN,
}


def _normalize_modes(modes, ndim):
    if modes is None:
        return tuple(range(ndim))
    if len(modes) != ndim:
        raise ValueError("mode length must match tensor ndim")
    normalized = tuple(modes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("each mode may appear at most once in a tensor")
    return normalized


def _infer_output_modes(mode_a, mode_b, mode_c, mode_d=None):
    if mode_d is not None:
        normalized = tuple(mode_d)
        if len(set(normalized)) != len(normalized):
            raise ValueError("each output mode may appear at most once")
        return normalized
    ordered = []
    for mode_group in (mode_a, mode_b, mode_c):
        for mode in mode_group:
            if mode not in ordered:
                ordered.append(mode)
    return tuple(ordered)


def _infer_output_shape(a, mode_a, b, mode_b, c, mode_c, mode_d):
    extents = {}
    for tensor, modes in ((a, mode_a), (b, mode_b), (c, mode_c)):
        for extent, mode in zip(tensor.shape, modes):
            previous = extents.get(mode, 1)
            if previous != 1 and extent != 1 and previous != extent:
                raise ValueError(f"incompatible broadcast extent for mode {mode}: {previous} vs {extent}")
            extents[mode] = max(previous, extent)
    return tuple(extents.get(mode, 1) for mode in mode_d)


def _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d):
    extents = {}
    for tensor, modes in ((a, mode_a), (b, mode_b)):
        for extent, mode in zip(tensor.shape, modes):
            previous = extents.get(mode)
            if previous is not None and previous != extent:
                raise ValueError(f"incompatible extent for mode {mode}: {previous} vs {extent}")
            extents[mode] = extent
    missing_modes = [mode for mode in mode_d if mode not in extents]
    if missing_modes:
        raise ValueError(f"output modes not present in inputs: {missing_modes}")
    return tuple(extents[mode] for mode in mode_d)


def _validate_contraction_addend(c, mode_c, mode_d, output_shape):
    mode_c = _normalize_modes(mode_c, c.ndim)
    if tuple(mode_c) != tuple(mode_d):
        raise ValueError("cuTensor contraction currently requires mode_c to be identical to mode_d")
    if tuple(c.shape) != tuple(output_shape):
        raise ValueError(f"addend tensor shape mismatch: expected {output_shape}, got {tuple(c.shape)}")
    return mode_c


def _infer_trinary_contraction_output_shape(a, mode_a, b, mode_b, c, mode_c, mode_e):
    extents = {}
    for tensor, modes in ((a, mode_a), (b, mode_b), (c, mode_c)):
        for extent, mode in zip(tensor.shape, modes):
            previous = extents.get(mode)
            if previous is not None and previous != extent:
                raise ValueError(f"incompatible extent for mode {mode}: {previous} vs {extent}")
            extents[mode] = extent
    missing_modes = [mode for mode in mode_e if mode not in extents]
    if missing_modes:
        raise ValueError(f"output modes not present in inputs: {missing_modes}")
    return tuple(extents[mode] for mode in mode_e)


def _validate_trinary_contraction_addend(d, mode_d, mode_e, output_shape):
    mode_d = _normalize_modes(mode_d, d.ndim)
    if tuple(mode_d) != tuple(mode_e):
        raise ValueError("cuTensor contraction trinary currently requires mode_d to be identical to mode_e")
    if tuple(d.shape) != tuple(output_shape):
        raise ValueError(f"addend tensor shape mismatch: expected {output_shape}, got {tuple(d.shape)}")
    return mode_d


@dataclass(frozen=True)
class BlockSparseTensorDescriptor:
    shape: tuple[int, ...]
    block_shape: tuple[int, ...] | None = None
    num_sections_per_mode: tuple[int, ...] | None = None
    section_extents: tuple[tuple[int, ...], ...] | None = None
    nonzero_coordinates: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if len(self.shape) == 0:
            raise ValueError("BlockSparseTensorDescriptor requires at least one mode")
        block_shape = self.block_shape
        section_extents = self.section_extents
        num_sections_per_mode = self.num_sections_per_mode
        ndim = len(self.shape)

        if block_shape is not None:
            if len(block_shape) != ndim:
                raise ValueError("block_shape rank must match shape rank")
            if any(extent <= 0 for extent in block_shape):
                raise ValueError("block_shape values must be positive")
            if any(shape_extent % block_extent != 0 for shape_extent, block_extent in zip(self.shape, block_shape)):
                raise ValueError("shape must be divisible by block_shape")
            inferred_num_sections = tuple(shape_extent // block_extent for shape_extent, block_extent in zip(self.shape, block_shape))
            inferred_section_extents = tuple(
                tuple([block_extent] * num_sections)
                for block_extent, num_sections in zip(block_shape, inferred_num_sections)
            )
            if num_sections_per_mode is None:
                num_sections_per_mode = inferred_num_sections
                object.__setattr__(self, "num_sections_per_mode", inferred_num_sections)
            if section_extents is None:
                section_extents = inferred_section_extents
                object.__setattr__(self, "section_extents", inferred_section_extents)

        if num_sections_per_mode is None or section_extents is None:
            raise ValueError("either block_shape or both num_sections_per_mode and section_extents must be provided")
        if len(num_sections_per_mode) != ndim or len(section_extents) != ndim:
            raise ValueError("num_sections_per_mode and section_extents rank must match shape rank")
        if any(len(mode_extents) != num_sections for mode_extents, num_sections in zip(section_extents, num_sections_per_mode)):
            raise ValueError("section_extents lengths must match num_sections_per_mode")
        if any(sum(mode_extents) != shape_extent for mode_extents, shape_extent in zip(section_extents, self.shape)):
            raise ValueError("section_extents must sum to the dense shape")
        for coord in self.nonzero_coordinates:
            if len(coord) != ndim:
                raise ValueError("nonzero_coordinates rank must match shape rank")
            if any(not (0 <= index < num_sections) for index, num_sections in zip(coord, num_sections_per_mode)):
                raise ValueError(f"nonzero coordinate {coord} is out of range")

    @property
    def canonical_nonzero_coordinates(self):
        return tuple(sorted(self.nonzero_coordinates))

    @property
    def canonical_section_extents(self):
        return self.section_extents


class BlockSparseTensor:
    def __init__(self, descriptor: BlockSparseTensorDescriptor, blocks: dict[tuple[int, ...], torch.Tensor]):
        self.descriptor = descriptor
        self.blocks = dict(blocks)
        if descriptor.nonzero_coordinates and set(descriptor.nonzero_coordinates) != set(self.blocks.keys()):
            raise ValueError("descriptor nonzero_coordinates must match provided block keys")
        for coord, block in self.blocks.items():
            if len(coord) != len(descriptor.shape):
                raise ValueError("block coordinates rank must match descriptor rank")
            expected_shape = tuple(
                descriptor.section_extents[mode][index]
                for mode, index in enumerate(coord)
            )
            if block.shape != expected_shape:
                raise ValueError(
                    f"block at {coord} has shape {tuple(block.shape)}, expected {expected_shape}"
                )

    @property
    def shape(self):
        return self.descriptor.shape

    @property
    def ndim(self):
        return len(self.descriptor.shape)

    @property
    def device(self):
        devices = {block.device for block in self.blocks.values()}
        if not devices:
            return None
        if len(devices) != 1:
            raise ValueError("all blocks must be on the same device")
        return next(iter(devices))

    @property
    def dtype(self):
        dtypes = {block.dtype for block in self.blocks.values()}
        if not dtypes:
            return None
        if len(dtypes) != 1:
            raise ValueError("all blocks must have the same dtype")
        return next(iter(dtypes))

    def to_dense(self):
        device = self.device
        dtype = self.dtype
        if device is None or dtype is None:
            raise ValueError("block-sparse tensor must contain at least one block")
        dense = torch.zeros(self.descriptor.shape, device=device, dtype=dtype)
        mode_offsets = []
        for mode_extents in self.descriptor.section_extents:
            offsets = [0]
            for extent in mode_extents:
                offsets.append(offsets[-1] + extent)
            mode_offsets.append(offsets)
        for coord, block in self.blocks.items():
            slices = tuple(
                slice(mode_offsets[mode][index], mode_offsets[mode][index + 1])
                for mode, index in enumerate(coord)
            )
            dense[slices] = block
        return dense

    def block_ptrs(self):
        ordered_coords = self.descriptor.canonical_nonzero_coordinates
        if not ordered_coords:
            ordered_coords = tuple(sorted(self.blocks.keys()))
        return [self.blocks[coord] for coord in ordered_coords]

    def block_strides(self):
        ordered_blocks = self.block_ptrs()
        strides = []
        for block in ordered_blocks:
            strides.extend(int(s) for s in block.stride())
        return tuple(strides)

    def block_ptr_array(self):
        ordered_blocks = self.block_ptrs()
        return (c_void_p * len(ordered_blocks))(*(c_void_p(block.data_ptr()) for block in ordered_blocks))


class CuTensorBlockSparseContraction:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.handle = c_void_p()
        self.desc_a = c_void_p()
        self.desc_b = c_void_p()
        self.desc_c = c_void_p()
        self.op_desc = c_void_p()
        self.plan_pref = c_void_p()
        self.plan = c_void_p()
        self.signature = None
        self.initialized = False

        if not CUTENSOR_AVAILABLE:
            return

        status = libcutensor.cutensorCreate(byref(self.handle))
        if status != 0:
            raise RuntimeError(f"cutensorCreate failed: {status}")
        self.initialized = True

    def _cuda_type(self, dtype):
        if dtype == torch.float32:
            return CUDA_R_32F
        if dtype == torch.float64:
            return CUDA_R_64F
        if dtype == torch.complex64:
            return CUDA_C_32F
        if dtype == torch.complex128:
            return CUDA_C_64F
        raise TypeError(f"unsupported block-sparse dtype: {dtype}")

    def _compute_desc(self, dtype):
        if dtype == torch.float32 or dtype == torch.complex64:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float64 or dtype == torch.complex128:
            return CUTENSOR_COMPUTE_DESC_64F
        raise TypeError(f"unsupported block-sparse dtype: {dtype}")

    def _scalar_value(self, value, dtype):
        if dtype == torch.complex64:
            value = complex(value)
            return (c_float * 2)(value.real, value.imag)
        if dtype == torch.complex128:
            value = complex(value)
            return (c_double * 2)(value.real, value.imag)
        if dtype == torch.float64:
            return c_double(value)
        return c_float(value)

    def _mode_array(self, modes):
        return (c_int32 * len(modes))(*modes)

    def _flatten_section_extents(self, descriptor):
        values = []
        for mode_extents in descriptor.section_extents:
            values.extend(int(v) for v in mode_extents)
        return (c_int64 * len(values))(*values)

    def _num_sections_array(self, descriptor):
        return (c_uint32 * len(descriptor.num_sections_per_mode))(*descriptor.num_sections_per_mode)

    def _flatten_nonzero_coordinates(self, descriptor):
        coords = []
        for coord in descriptor.canonical_nonzero_coordinates:
            coords.extend(int(v) for v in coord)
        return (c_int32 * len(coords))(*coords)

    def _flatten_strides(self, tensor):
        strides = tensor.block_strides()
        return (c_int64 * len(strides))(*strides)

    def _create_block_sparse_tensor_descriptor(self, tensor):
        desc = c_void_p()
        num_sections = self._num_sections_array(tensor.descriptor)
        extents = self._flatten_section_extents(tensor.descriptor)
        nonzero_coordinates = self._flatten_nonzero_coordinates(tensor.descriptor)
        strides = self._flatten_strides(tensor)
        status = libcutensor.cutensorCreateBlockSparseTensorDescriptor(
            self.handle,
            byref(desc),
            c_uint32(tensor.ndim),
            c_uint64(len(tensor.blocks)),
            num_sections,
            extents,
            nonzero_coordinates,
            strides,
            c_int(self._cuda_type(tensor.dtype)),
        )
        if status != 0:
            raise RuntimeError(f"create block-sparse tensor descriptor failed: {status}")
        return desc

    def _signature(self, a, mode_a, b, mode_b, c, mode_c, mode_d):
        return (
            a.dtype,
            a.descriptor.shape,
            a.descriptor.canonical_section_extents,
            a.descriptor.canonical_nonzero_coordinates,
            tuple(mode_a),
            b.descriptor.shape,
            b.descriptor.canonical_section_extents,
            b.descriptor.canonical_nonzero_coordinates,
            tuple(mode_b),
            c.descriptor.shape,
            c.descriptor.canonical_section_extents,
            c.descriptor.canonical_nonzero_coordinates,
            tuple(mode_c),
            tuple(mode_d),
        )

    def _validate_mode_sections(self, a, mode_a, b, mode_b, c, mode_c):
        mode_sections = {}
        for descriptor, modes in ((a.descriptor, mode_a), (b.descriptor, mode_b), (c.descriptor, mode_c)):
            for axis, mode in enumerate(modes):
                section_extents = descriptor.section_extents[axis]
                previous = mode_sections.get(mode)
                if previous is None:
                    mode_sections[mode] = section_extents
                elif previous != section_extents:
                    raise ValueError(f"section extents must match for shared mode {mode}")

    def _destroy_cache(self):
        if self.plan:
            libcutensor.cutensorDestroyPlan(self.plan)
            self.plan = c_void_p()
        if self.plan_pref:
            libcutensor.cutensorDestroyPlanPreference(self.plan_pref)
            self.plan_pref = c_void_p()
        if self.op_desc:
            libcutensor.cutensorDestroyOperationDescriptor(self.op_desc)
            self.op_desc = c_void_p()
        if self.desc_c:
            libcutensor.cutensorDestroyBlockSparseTensorDescriptor(self.desc_c)
            self.desc_c = c_void_p()
        if self.desc_b:
            libcutensor.cutensorDestroyBlockSparseTensorDescriptor(self.desc_b)
            self.desc_b = c_void_p()
        if self.desc_a:
            libcutensor.cutensorDestroyBlockSparseTensorDescriptor(self.desc_a)
            self.desc_a = c_void_p()
        self.signature = None

    def prepare(self, a, mode_a, b, mode_b, c, mode_c, mode_d):
        if not self.initialized:
            raise RuntimeError("cuTensor not initialized")
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise TypeError("block-sparse tensors must share the same dtype")
        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        mode_c = _normalize_modes(mode_c, c.ndim)
        if tuple(mode_c) != tuple(mode_d):
            raise ValueError("block-sparse contraction currently requires mode_c to be identical to mode_d")

        output_shape = _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d)
        if tuple(c.shape) != tuple(output_shape):
            raise ValueError(f"addend tensor shape mismatch: expected {output_shape}, got {tuple(c.shape)}")
        self._validate_mode_sections(a, mode_a, b, mode_b, c, mode_c)

        signature = self._signature(a, mode_a, b, mode_b, c, mode_c, mode_d)
        if self.signature == signature and self.plan:
            return

        self._destroy_cache()
        self.desc_a = self._create_block_sparse_tensor_descriptor(a)
        self.desc_b = self._create_block_sparse_tensor_descriptor(b)
        self.desc_c = self._create_block_sparse_tensor_descriptor(c)
        mode_a_arr = self._mode_array(mode_a)
        mode_b_arr = self._mode_array(mode_b)
        mode_c_arr = self._mode_array(mode_c)
        compute_desc = self._compute_desc(a.dtype)

        status = libcutensor.cutensorCreateBlockSparseContraction(
            self.handle,
            byref(self.op_desc),
            self.desc_a,
            mode_a_arr,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_b,
            mode_b_arr,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_c,
            mode_c_arr,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_c,
            mode_c_arr,
            compute_desc,
        )
        if status != 0:
            raise RuntimeError(f"create block-sparse contraction descriptor failed: {status}")

        status = libcutensor.cutensorCreatePlanPreference(
            self.handle,
            byref(self.plan_pref),
            c_int(CUTENSOR_ALGO_DEFAULT),
            c_int(CUTENSOR_JIT_MODE_NONE),
        )
        if status != 0:
            raise RuntimeError(f"create plan preference failed: {status}")

        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")

        status = libcutensor.cutensorCreatePlan(
            self.handle,
            byref(self.plan),
            self.op_desc,
            self.plan_pref,
            workspace_size,
        )
        if status != 0:
            raise RuntimeError(f"create plan failed: {status}")

        self.signature = signature

    def __call__(self, a, mode_a, b, mode_b, c, mode_c, mode_d, alpha=1.0, beta=0.0):
        self.prepare(a, mode_a, b, mode_b, c, mode_c, mode_d)
        alpha_val = self._scalar_value(alpha, a.dtype)
        beta_val = self._scalar_value(beta, a.dtype)
        a_ptrs = a.block_ptr_array()
        b_ptrs = b.block_ptr_array()
        c_ptrs = c.block_ptr_array()
        out_blocks = {
            coord: torch.empty_like(c.blocks[coord])
            for coord in c.descriptor.canonical_nonzero_coordinates
        }
        out_tensor = BlockSparseTensor(c.descriptor, out_blocks)
        d_ptrs = out_tensor.block_ptr_array()

        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")
        workspace = c_void_p(0)
        workspace_tensor = None
        if workspace_size.value > 0:
            workspace_tensor = torch.empty(int(workspace_size.value), device=next(iter(out_blocks.values())).device, dtype=torch.uint8)
            workspace = c_void_p(workspace_tensor.data_ptr())

        status = libcutensor.cutensorBlockSparseContract(
            self.handle,
            self.plan,
            byref(alpha_val),
            a_ptrs,
            b_ptrs,
            byref(beta_val),
            c_ptrs,
            d_ptrs,
            workspace,
            workspace_size,
            c_void_p(0),
        )
        if status != 0:
            raise RuntimeError(f"cutensorBlockSparseContract failed: {status}")
        return out_tensor

    def __del__(self):
        if CUTENSOR_AVAILABLE:
            self._destroy_cache()
            if self.initialized and self.handle:
                try:
                    libcutensor.cutensorDestroy(self.handle)
                except Exception:
                    pass


class BlockSparseTensorContraction:
    def __init__(self):
        self.cutensor_executor = None

    def _supports_cutensor(self, a, b, c):
        if not CUTENSOR_AVAILABLE or c is None:
            return False
        return a.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)

    def _default_modes(self, a, b, c):
        if a.ndim != 2 or b.ndim != 2 or (c is not None and c.ndim != 2):
            raise ValueError("mode_a/mode_b/mode_d must be provided for ND block-sparse contraction")
        return (0, 1), (1, 2), (0, 2), (0, 2)

    def __call__(
        self,
        a: BlockSparseTensor,
        b: BlockSparseTensor,
        *,
        c=None,
        alpha=1.0,
        beta=0.0,
        mode_a=None,
        mode_b=None,
        mode_c=None,
        mode_d=None,
        out=None,
    ):
        if not isinstance(a, BlockSparseTensor) or not isinstance(b, BlockSparseTensor):
            raise TypeError("a and b must be BlockSparseTensor instances")
        if a.dtype != b.dtype:
            raise TypeError("block-sparse inputs must have the same dtype")

        if mode_a is None or mode_b is None or mode_d is None:
            mode_a, mode_b, inferred_mode_c, mode_d = self._default_modes(a, b, c)
            if mode_c is None:
                mode_c = inferred_mode_c
        else:
            mode_a = _normalize_modes(mode_a, a.ndim)
            mode_b = _normalize_modes(mode_b, b.ndim)
            mode_d = tuple(mode_d)
            if mode_c is None and c is not None:
                mode_c = tuple(mode_d)

        output_shape = _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d)

        dense_a = a.to_dense()
        dense_b = b.to_dense()

        if c is not None:
            if not isinstance(c, BlockSparseTensor):
                raise TypeError("c must be a BlockSparseTensor when provided")
            mode_c = _validate_contraction_addend(c, mode_c, mode_d, output_shape)
            dense_c = c.to_dense()
        else:
            mode_c = None
            dense_c = None

        if self._supports_cutensor(a, b, c):
            if self.cutensor_executor is None or self.cutensor_executor.dtype != a.dtype:
                self.cutensor_executor = CuTensorBlockSparseContraction(dtype=a.dtype)
            out_sparse = self.cutensor_executor(a, mode_a, b, mode_b, c, mode_c, mode_d, alpha=alpha, beta=beta)
            return out_sparse.to_dense()

        # No cuTensor available — fall through to the vendor-agnostic dense
        # contraction path below. On PPU this is the native baseline (acblas-
        # backed torch.matmul); on NVIDIA without cuTensor it is also the
        # correct dense fallback. We use torch.matmul directly (rather than
        # the cutensor.contraction() function) because the latter would route
        # through CuTensorContraction which requires cuTensor to be
        # initialised. The BlockSparseTensor benchmark loads the
        # vendor-specific kernel-mode baseline separately via
        # benchmark_core.get_baseline_class("BlockSparseContraction").
        if c is not None:
            mode_c = _validate_contraction_addend(c, mode_c, mode_d, output_shape)
        if len(mode_a) == 2 and len(mode_b) == 2 and mode_a == (0, 1) and mode_b == (1, 2) and mode_d == (0, 2):
            # Fast path: standard 2D matmul C = alpha * A @ B + beta * C
            result = alpha * torch.matmul(dense_a, dense_b)
            if dense_c is not None and beta != 0.0:
                result = result + beta * dense_c
        else:
            # General ND contraction via einsum
            _cls = _get_torch_contraction_baseline_cls()
            eq = _cls._einsum_equation(mode_a, mode_b, mode_d)
            result = alpha * torch.einsum(eq, dense_a, dense_b)
            if dense_c is not None and beta != 0.0:
                result = result + beta * dense_c
        # Apply output block-sparsity mask: the cuTensor path produces a
        # BlockSparseTensor whose non-zero blocks match c's descriptor, so
        # the dense fallback must zero out everything outside those blocks
        # to keep semantics consistent.
        if c is not None:
            _bs_cls = _get_torch_block_sparse_baseline_cls()
            mask = _bs_cls._sparsity_mask(c)
            result = result * mask.to(result.dtype)
        return result


_GETT_EXECUTORS = {}
_TRINARY_CONTRACTION_EXECUTORS = {}
_BLOCK_SPARSE_CONTRACTION_EXECUTORS = {}


def _get_gett_executor(dtype):
    executor = _GETT_EXECUTORS.get(dtype)
    if executor is None:
        executor = CuTensorContraction(dtype=dtype)
        _GETT_EXECUTORS[dtype] = executor
    return executor


def _get_trinary_contraction_executor(dtype):
    executor = _TRINARY_CONTRACTION_EXECUTORS.get(dtype)
    if executor is None:
        executor = CuTensorContractionTrinary(dtype=dtype)
        _TRINARY_CONTRACTION_EXECUTORS[dtype] = executor
    return executor


def _get_block_sparse_contraction_executor(dtype):
    executor = _BLOCK_SPARSE_CONTRACTION_EXECUTORS.get(dtype)
    if executor is None:
        executor = BlockSparseTensorContraction()
        _BLOCK_SPARSE_CONTRACTION_EXECUTORS[dtype] = executor
    return executor


def _resolve_operator(op, mapping, kind):
    if isinstance(op, int):
        return op
    if not isinstance(op, str):
        raise TypeError(f"unsupported {kind} operator type: {type(op)!r}")
    key = op.strip().lower()
    if key not in mapping:
        raise ValueError(f"unsupported {kind} operator: {op}")
    return mapping[key]

try:
    libcutensor = ctypes.CDLL("libcutensor.so")
    CUTENSOR_AVAILABLE = True
except OSError:
    libcutensor = None
    CUTENSOR_AVAILABLE = False

# Lazy import of the PyTorch-native baseline used by BlockSparseTensorContraction
# when cuTensor is unavailable. Imported here (not at top of file) to avoid a
# circular dependency: torch_baseline.py imports from cutensor.py.
_TorchContractionBaseline = None
def _get_torch_contraction_baseline_cls():
    global _TorchContractionBaseline
    if _TorchContractionBaseline is None:
        from flagtensor.torch_baseline import TorchContractionBaseline as _T
        _TorchContractionBaseline = _T
    return _TorchContractionBaseline


_TorchBlockSparseContractionBaseline = None
def _get_torch_block_sparse_baseline_cls():
    global _TorchBlockSparseContractionBaseline
    if _TorchBlockSparseContractionBaseline is None:
        from flagtensor.torch_baseline import TorchBlockSparseContractionBaseline as _T
        _TorchBlockSparseContractionBaseline = _T
    return _TorchBlockSparseContractionBaseline

if CUTENSOR_AVAILABLE:
    CUTENSOR_COMPUTE_DESC_16F = c_void_p.in_dll(libcutensor, "CUTENSOR_COMPUTE_DESC_16F")
    CUTENSOR_COMPUTE_DESC_16BF = c_void_p.in_dll(libcutensor, "CUTENSOR_COMPUTE_DESC_16BF")
    CUTENSOR_COMPUTE_DESC_32F = c_void_p.in_dll(libcutensor, "CUTENSOR_COMPUTE_DESC_32F")
    CUTENSOR_COMPUTE_DESC_64F = c_void_p.in_dll(libcutensor, "CUTENSOR_COMPUTE_DESC_64F")

    libcutensor.cutensorCreate.restype = c_int
    libcutensor.cutensorCreate.argtypes = [POINTER(c_void_p)]

    libcutensor.cutensorDestroy.restype = c_int
    libcutensor.cutensorDestroy.argtypes = [c_void_p]

    libcutensor.cutensorCreateTensorDescriptor.restype = c_int
    libcutensor.cutensorCreateTensorDescriptor.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_uint32,
        POINTER(c_int64),
        POINTER(c_int64),
        c_int,
        c_uint32,
    ]

    libcutensor.cutensorDestroyTensorDescriptor.restype = c_int
    libcutensor.cutensorDestroyTensorDescriptor.argtypes = [c_void_p]

    libcutensor.cutensorCreatePermutation.restype = c_int
    libcutensor.cutensorCreatePermutation.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_void_p,
    ]

    libcutensor.cutensorCreateElementwiseBinary.restype = c_int
    libcutensor.cutensorCreateElementwiseBinary.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
    ]

    libcutensor.cutensorCreateElementwiseTrinary.restype = c_int
    libcutensor.cutensorCreateElementwiseTrinary.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_int,
        c_void_p,
    ]

    libcutensor.cutensorDestroyOperationDescriptor.restype = c_int
    libcutensor.cutensorDestroyOperationDescriptor.argtypes = [c_void_p]

    libcutensor.cutensorCreatePlanPreference.restype = c_int
    libcutensor.cutensorCreatePlanPreference.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_int,
        c_int,
    ]

    libcutensor.cutensorDestroyPlanPreference.restype = c_int
    libcutensor.cutensorDestroyPlanPreference.argtypes = [c_void_p]

    libcutensor.cutensorEstimateWorkspaceSize.restype = c_int
    libcutensor.cutensorEstimateWorkspaceSize.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_int,
        POINTER(c_uint64),
    ]

    libcutensor.cutensorCreatePlan.restype = c_int
    libcutensor.cutensorCreatePlan.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        c_void_p,
        c_uint64,
    ]

    libcutensor.cutensorDestroyPlan.restype = c_int
    libcutensor.cutensorDestroyPlan.argtypes = [c_void_p]

    libcutensor.cutensorPermute.restype = c_int
    libcutensor.cutensorPermute.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    libcutensor.cutensorCreateContraction.restype = c_int
    libcutensor.cutensorCreateContraction.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_void_p,
    ]

    libcutensor.cutensorContract.restype = c_int
    libcutensor.cutensorContract.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
        c_void_p,
    ]

    libcutensor.cutensorCreateBlockSparseTensorDescriptor.restype = c_int
    libcutensor.cutensorCreateBlockSparseTensorDescriptor.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_uint32,
        c_uint64,
        POINTER(c_uint32),
        POINTER(c_int64),
        POINTER(c_int32),
        POINTER(c_int64),
        c_int,
    ]

    libcutensor.cutensorDestroyBlockSparseTensorDescriptor.restype = c_int
    libcutensor.cutensorDestroyBlockSparseTensorDescriptor.argtypes = [c_void_p]

    libcutensor.cutensorCreateBlockSparseContraction.restype = c_int
    libcutensor.cutensorCreateBlockSparseContraction.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_void_p,
    ]

    libcutensor.cutensorBlockSparseContract.restype = c_int
    libcutensor.cutensorBlockSparseContract.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        POINTER(c_void_p),
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_void_p),
        POINTER(c_void_p),
        c_void_p,
        c_uint64,
        c_void_p,
    ]

    libcutensor.cutensorCreateContractionTrinary.restype = c_int
    libcutensor.cutensorCreateContractionTrinary.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_int,
        c_void_p,
        POINTER(c_int32),
        c_void_p,
    ]

    libcutensor.cutensorContractTrinary.restype = c_int
    libcutensor.cutensorContractTrinary.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
        c_void_p,
    ]

    libcutensor.cutensorElementwiseBinaryExecute.restype = c_int
    libcutensor.cutensorElementwiseBinaryExecute.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    libcutensor.cutensorElementwiseTrinaryExecute.restype = c_int
    libcutensor.cutensorElementwiseTrinaryExecute.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]


class CuTensorUnary:
    def __init__(self, op, dtype=torch.float32):
        self.op = op
        self.dtype = dtype
        self.handle = c_void_p()
        self.desc_a = c_void_p()
        self.desc_b = c_void_p()
        self.op_desc = c_void_p()
        self.plan_pref = c_void_p()
        self.plan = c_void_p()
        self.signature = None
        self.initialized = False

        if not CUTENSOR_AVAILABLE:
            return

        status = libcutensor.cutensorCreate(byref(self.handle))
        if status != 0:
            raise RuntimeError(f"cutensorCreate failed: {status}")
        self.initialized = True

    def _create_tensor_descriptor(self, tensor):
        ndim = tensor.ndim
        extents = (c_int64 * ndim)(*tensor.shape)
        strides = (c_int64 * ndim)(*tensor.stride())
        cuda_type = self._cuda_type(tensor.dtype)
        alignment = c_uint32(max(1, tensor.element_size()))
        desc = c_void_p()
        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(desc),
            c_uint32(ndim),
            extents,
            strides,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create tensor descriptor failed: {status}")
        return desc

    def _mode_array(self, modes):
        return (c_int32 * len(modes))(*modes)

    def _cuda_type(self, dtype):
        if dtype == torch.float16:
            return CUDA_R_16F
        if dtype == torch.float32:
            return CUDA_R_32F
        if dtype == torch.float64:
            return CUDA_R_64F
        if dtype == torch.bfloat16:
            return CUDA_R_16BF
        if dtype == torch.complex64:
            return CUDA_C_32F
        if dtype == torch.complex128:
            return CUDA_C_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _compute_desc(self, dtype):
        if dtype == torch.float16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float32:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float64:
            return CUTENSOR_COMPUTE_DESC_64F
        if dtype == torch.bfloat16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex64:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex128:
            return CUTENSOR_COMPUTE_DESC_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _scalar_value(self, value, dtype):
        if dtype == torch.complex64:
            value = complex(value)
            return (c_float * 2)(value.real, value.imag)
        if dtype == torch.complex128:
            value = complex(value)
            return (c_double * 2)(value.real, value.imag)
        if dtype == torch.float64:
            return c_double(value)
        return c_float(value)

    def _destroy_cache(self):
        if self.plan:
            libcutensor.cutensorDestroyPlan(self.plan)
            self.plan = c_void_p()
        if self.plan_pref:
            libcutensor.cutensorDestroyPlanPreference(self.plan_pref)
            self.plan_pref = c_void_p()
        if self.op_desc:
            libcutensor.cutensorDestroyOperationDescriptor(self.op_desc)
            self.op_desc = c_void_p()
        if self.desc_b:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_b)
            self.desc_b = c_void_p()
        if self.desc_a:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_a)
            self.desc_a = c_void_p()
        self.signature = None

    def _signature(self, x):
        return (x.dtype, tuple(x.shape), tuple(x.stride()))

    def prepare(self, x):
        if not self.initialized:
            raise RuntimeError("cuTensor not initialized")
        if not x.is_cuda:
            raise ValueError("input tensor must be on CUDA")

        # Check instance-level cache first (exact match)
        signature = self._signature(x)
        if self.signature == signature and self.plan:
            return

        self._destroy_cache()

        ndim = x.ndim
        mode = (c_int32 * ndim)(*range(ndim))
        extents = (c_int64 * ndim)(*x.shape)
        strides = (c_int64 * ndim)(*x.stride())
        cuda_type = self._cuda_type(x.dtype)
        compute_desc = self._compute_desc(x.dtype)
        alignment = c_uint32(max(1, x.element_size()))

        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(self.desc_a),
            c_uint32(ndim),
            extents,
            strides,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create input descriptor failed: {status}")

        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(self.desc_b),
            c_uint32(ndim),
            extents,
            strides,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create output descriptor failed: {status}")

        status = libcutensor.cutensorCreatePermutation(
            self.handle,
            byref(self.op_desc),
            self.desc_a,
            mode,
            c_int(self.op),
            self.desc_b,
            mode,
            compute_desc,
        )
        if status != 0:
            raise RuntimeError(f"create permutation descriptor failed: {status}")

        status = libcutensor.cutensorCreatePlanPreference(
            self.handle,
            byref(self.plan_pref),
            c_int(CUTENSOR_ALGO_DEFAULT),
            c_int(CUTENSOR_JIT_MODE_NONE),
        )
        if status != 0:
            raise RuntimeError(f"create plan preference failed: {status}")

        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")

        status = libcutensor.cutensorCreatePlan(
            self.handle,
            byref(self.plan),
            self.op_desc,
            self.plan_pref,
            workspace_size,
        )
        if status != 0:
            raise RuntimeError(f"create plan failed: {status}")

        self.signature = signature

    def build_kernel_callable(self, x, alpha=1.0):
        self.prepare(x)
        y = torch.empty_like(x)
        alpha_val = self._scalar_value(alpha, x.dtype)

        def run_kernel():
            status = libcutensor.cutensorPermute(
                self.handle,
                self.plan,
                byref(alpha_val),
                c_void_p(x.data_ptr()),
                c_void_p(y.data_ptr()),
                c_void_p(0),
            )
            if status != 0:
                raise RuntimeError(f"cutensorPermute failed: {status}")
            return y

        return run_kernel

    def __call__(self, x, alpha=1.0):
        self.prepare(x)
        y = torch.empty_like(x)
        alpha_val = self._scalar_value(alpha, x.dtype)
        status = libcutensor.cutensorPermute(
            self.handle,
            self.plan,
            byref(alpha_val),
            c_void_p(x.data_ptr()),
            c_void_p(y.data_ptr()),
            c_void_p(0),
        )
        if status != 0:
            raise RuntimeError(f"cutensorPermute failed: {status}")
        return y

    def __del__(self):
        if CUTENSOR_AVAILABLE:
            self._destroy_cache()
            if self.initialized and self.handle:
                try:
                    libcutensor.cutensorDestroy(self.handle)
                except Exception:
                    pass


class CuTensorBinary:
    def __init__(self, op, dtype=torch.float32):
        self.op = op
        self.dtype = dtype
        self.handle = c_void_p()
        self.desc_a = c_void_p()
        self.desc_c = c_void_p()
        self.desc_d = c_void_p()
        self.op_desc = c_void_p()
        self.plan_pref = c_void_p()
        self.plan = c_void_p()
        self.signature = None
        self.initialized = False

        if not CUTENSOR_AVAILABLE:
            return

        status = libcutensor.cutensorCreate(byref(self.handle))
        if status != 0:
            raise RuntimeError(f"cutensorCreate failed: {status}")
        self.initialized = True

    def _create_tensor_descriptor(self, tensor):
        ndim = tensor.ndim
        extents = (c_int64 * ndim)(*tensor.shape)
        strides = (c_int64 * ndim)(*tensor.stride())
        cuda_type = self._cuda_type(tensor.dtype)
        alignment = c_uint32(max(1, tensor.element_size()))
        desc = c_void_p()
        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(desc),
            c_uint32(ndim),
            extents,
            strides,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create tensor descriptor failed: {status}")
        return desc

    def _mode_array(self, modes):
        return (c_int32 * len(modes))(*modes)

    def _cuda_type(self, dtype):
        if dtype == torch.float16:
            return CUDA_R_16F
        if dtype == torch.float32:
            return CUDA_R_32F
        if dtype == torch.float64:
            return CUDA_R_64F
        if dtype == torch.bfloat16:
            return CUDA_R_16BF
        if dtype == torch.complex64:
            return CUDA_C_32F
        if dtype == torch.complex128:
            return CUDA_C_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _compute_desc(self, dtype):
        if dtype == torch.float16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float32:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float64:
            return CUTENSOR_COMPUTE_DESC_64F
        if dtype == torch.bfloat16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex64:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex128:
            return CUTENSOR_COMPUTE_DESC_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _scalar_value(self, value, dtype):
        if dtype == torch.complex64:
            value = complex(value)
            return (c_float * 2)(value.real, value.imag)
        if dtype == torch.complex128:
            value = complex(value)
            return (c_double * 2)(value.real, value.imag)
        if dtype == torch.float64:
            return c_double(value)
        return c_float(value)

    def _destroy_cache(self):
        if self.plan:
            libcutensor.cutensorDestroyPlan(self.plan)
            self.plan = c_void_p()
        if self.plan_pref:
            libcutensor.cutensorDestroyPlanPreference(self.plan_pref)
            self.plan_pref = c_void_p()
        if self.op_desc:
            libcutensor.cutensorDestroyOperationDescriptor(self.op_desc)
            self.op_desc = c_void_p()
        if self.desc_d and self.desc_d != self.desc_c:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_d)
            self.desc_d = c_void_p()
        if self.desc_c:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_c)
            self.desc_c = c_void_p()
        self.desc_d = c_void_p()
        if self.desc_a:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_a)
            self.desc_a = c_void_p()
        self.signature = None

    def _signature(self, x, y):
        return (
            x.dtype,
            tuple(x.shape),
            tuple(x.stride()),
            y.dtype,
            tuple(y.shape),
            tuple(y.stride()),
        )

    def prepare(self, x, y):
        if not self.initialized:
            raise RuntimeError("cuTensor not initialized")
        if not x.is_cuda or not y.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if x.dtype != y.dtype:
            raise TypeError("input tensors must have the same dtype")
        if x.shape != y.shape:
            raise ValueError("input tensors must have the same shape")
        if x.stride() != y.stride():
            raise ValueError("input tensors must have the same stride")

        # Check instance-level cache first (exact match)
        signature = self._signature(x, y)
        if self.signature == signature and self.plan:
            return

        self._destroy_cache()

        ndim = x.ndim
        mode = (c_int32 * ndim)(*range(ndim))
        extents = (c_int64 * ndim)(*x.shape)
        strides_x = (c_int64 * ndim)(*x.stride())
        strides_y = (c_int64 * ndim)(*y.stride())
        cuda_type = self._cuda_type(x.dtype)
        compute_desc = self._compute_desc(x.dtype)
        alignment = c_uint32(max(1, x.element_size()))

        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(self.desc_a),
            c_uint32(ndim),
            extents,
            strides_x,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create input A descriptor failed: {status}")

        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(self.desc_c),
            c_uint32(ndim),
            extents,
            strides_y,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create input C descriptor failed: {status}")

        self.desc_d = self.desc_c

        status = libcutensor.cutensorCreateElementwiseBinary(
            self.handle,
            byref(self.op_desc),
            self.desc_a,
            mode,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_c,
            mode,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_d,
            mode,
            c_int(self.op),
            compute_desc,
        )
        if status != 0:
            raise RuntimeError(f"create elementwise binary descriptor failed: {status}")

        status = libcutensor.cutensorCreatePlanPreference(
            self.handle,
            byref(self.plan_pref),
            c_int(CUTENSOR_ALGO_DEFAULT),
            c_int(CUTENSOR_JIT_MODE_NONE),
        )
        if status != 0:
            raise RuntimeError(f"create plan preference failed: {status}")

        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")

        status = libcutensor.cutensorCreatePlan(
            self.handle,
            byref(self.plan),
            self.op_desc,
            self.plan_pref,
            workspace_size,
        )
        if status != 0:
            raise RuntimeError(f"create plan failed: {status}")

        self.signature = signature

    def build_kernel_callable(self, x, y, alpha=1.0, gamma=1.0):
        self.prepare(x, y)
        out = torch.empty_like(x)
        alpha_val = self._scalar_value(alpha, x.dtype)
        gamma_val = self._scalar_value(gamma, x.dtype)

        def run_kernel():
            status = libcutensor.cutensorElementwiseBinaryExecute(
                self.handle,
                self.plan,
                byref(alpha_val),
                c_void_p(x.data_ptr()),
                byref(gamma_val),
                c_void_p(y.data_ptr()),
                c_void_p(out.data_ptr()),
                c_void_p(0),
            )
            if status != 0:
                raise RuntimeError(f"cutensorElementwiseBinaryExecute failed: {status}")
            return out

        return run_kernel

    def __call__(self, x, y, alpha=1.0, gamma=1.0):
        self.prepare(x, y)
        out = torch.empty_like(x)
        alpha_val = self._scalar_value(alpha, x.dtype)
        gamma_val = self._scalar_value(gamma, x.dtype)
        status = libcutensor.cutensorElementwiseBinaryExecute(
            self.handle,
            self.plan,
            byref(alpha_val),
            c_void_p(x.data_ptr()),
            byref(gamma_val),
            c_void_p(y.data_ptr()),
            c_void_p(out.data_ptr()),
            c_void_p(0),
        )
        if status != 0:
            raise RuntimeError(f"cutensorElementwiseBinaryExecute failed: {status}")
        return out

    def __del__(self):
        if CUTENSOR_AVAILABLE:
            self._destroy_cache()
            if self.initialized and self.handle:
                try:
                    libcutensor.cutensorDestroy(self.handle)
                except Exception:
                    pass


class CuTensorTrinary:
    def __init__(self, op_ab, op_abc, op_a=CUTENSOR_OP_IDENTITY, op_b=CUTENSOR_OP_IDENTITY, op_c=CUTENSOR_OP_IDENTITY, dtype=torch.float32):
        self.op_ab = _resolve_operator(op_ab, BINARY_OPERATOR_MAP, "binary")
        self.op_abc = _resolve_operator(op_abc, BINARY_OPERATOR_MAP, "binary")
        self.op_a = _resolve_operator(op_a, UNARY_OPERATOR_MAP, "unary")
        self.op_b = _resolve_operator(op_b, UNARY_OPERATOR_MAP, "unary")
        self.op_c = _resolve_operator(op_c, UNARY_OPERATOR_MAP, "unary")
        self.dtype = dtype
        self.handle = c_void_p()
        self.desc_a = c_void_p()
        self.desc_b = c_void_p()
        self.desc_c = c_void_p()
        self.desc_d = c_void_p()
        self.op_desc = c_void_p()
        self.plan_pref = c_void_p()
        self.plan = c_void_p()
        self.signature = None
        self.initialized = False

        if not CUTENSOR_AVAILABLE:
            return

        status = libcutensor.cutensorCreate(byref(self.handle))
        if status != 0:
            raise RuntimeError(f"cutensorCreate failed: {status}")
        self.initialized = True

    def _create_tensor_descriptor(self, tensor):
        ndim = tensor.ndim
        extents = (c_int64 * ndim)(*tensor.shape)
        strides = (c_int64 * ndim)(*tensor.stride())
        cuda_type = self._cuda_type(tensor.dtype)
        alignment = c_uint32(max(1, tensor.element_size()))
        desc = c_void_p()
        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(desc),
            c_uint32(ndim),
            extents,
            strides,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create tensor descriptor failed: {status}")
        return desc

    def _mode_array(self, modes):
        return (c_int32 * len(modes))(*modes)

    def _cuda_type(self, dtype):
        if dtype == torch.float16:
            return CUDA_R_16F
        if dtype == torch.float32:
            return CUDA_R_32F
        if dtype == torch.float64:
            return CUDA_R_64F
        if dtype == torch.bfloat16:
            return CUDA_R_16BF
        if dtype == torch.complex64:
            return CUDA_C_32F
        if dtype == torch.complex128:
            return CUDA_C_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _compute_desc(self, dtype):
        if dtype == torch.float16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float32:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float64:
            return CUTENSOR_COMPUTE_DESC_64F
        if dtype == torch.bfloat16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex64:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex128:
            return CUTENSOR_COMPUTE_DESC_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _scalar_value(self, value, dtype):
        if dtype == torch.complex64:
            value = complex(value)
            return (c_float * 2)(value.real, value.imag)
        if dtype == torch.complex128:
            value = complex(value)
            return (c_double * 2)(value.real, value.imag)
        if dtype == torch.float64:
            return c_double(value)
        return c_float(value)

    def _destroy_cache(self):
        if self.plan:
            libcutensor.cutensorDestroyPlan(self.plan)
            self.plan = c_void_p()
        if self.plan_pref:
            libcutensor.cutensorDestroyPlanPreference(self.plan_pref)
            self.plan_pref = c_void_p()
        if self.op_desc:
            libcutensor.cutensorDestroyOperationDescriptor(self.op_desc)
            self.op_desc = c_void_p()
        if self.desc_d and self.desc_d != self.desc_c:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_d)
            self.desc_d = c_void_p()
        if self.desc_c:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_c)
            self.desc_c = c_void_p()
        if self.desc_b:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_b)
            self.desc_b = c_void_p()
        if self.desc_a:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_a)
            self.desc_a = c_void_p()
        self.desc_d = c_void_p()
        self.signature = None

    def _signature(self, x, y, z, mode_a, mode_b, mode_c, mode_d):
        return (
            x.dtype,
            tuple(x.shape),
            tuple(x.stride()),
            tuple(mode_a),
            y.dtype,
            tuple(y.shape),
            tuple(y.stride()),
            tuple(mode_b),
            z.dtype,
            tuple(z.shape),
            tuple(z.stride()),
            tuple(mode_c),
            tuple(mode_d),
        )

    def prepare(self, x, y, z, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        if not self.initialized:
            raise RuntimeError("cuTensor not initialized")
        if not x.is_cuda or not y.is_cuda or not z.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if x.dtype != y.dtype or x.dtype != z.dtype:
            raise TypeError("input tensors must have the same dtype")

        mode_a = _normalize_modes(mode_a, x.ndim)
        mode_b = _normalize_modes(mode_b, y.ndim)
        mode_c = _normalize_modes(mode_c, z.ndim)
        inferred_mode_d = _infer_output_modes(mode_a, mode_b, mode_c, mode_d)
        if tuple(inferred_mode_d) != tuple(mode_c):
            raise ValueError("cuTensor elementwise trinary currently requires mode_d to be identical to mode_c")
        mode_d = tuple(mode_c)

        expected_shape = _infer_output_shape(x, mode_a, y, mode_b, z, mode_c, mode_d)
        if tuple(expected_shape) != tuple(z.shape):
            raise ValueError("cuTensor elementwise trinary currently requires the output descriptor to match tensor C")

        if out is not None:
            if not out.is_cuda:
                raise ValueError("output tensor must be on CUDA")
            if out.dtype != x.dtype:
                raise TypeError("output tensor must have the same dtype as inputs")
            if tuple(out.shape) != tuple(expected_shape):
                raise ValueError(f"output tensor shape mismatch: expected {expected_shape}, got {tuple(out.shape)}")

        # Check instance-level cache first (exact match)
        signature = self._signature(x, y, z, mode_a, mode_b, mode_c, mode_d)
        if self.signature == signature and self.plan:
            return mode_a, mode_b, mode_c, mode_d

        self._destroy_cache()

        cuda_type = self._cuda_type(x.dtype)
        compute_desc = self._compute_desc(x.dtype)
        if out is None:
            out = torch.empty_like(z)

        self.desc_a = self._create_tensor_descriptor(x)
        self.desc_b = self._create_tensor_descriptor(y)
        self.desc_c = self._create_tensor_descriptor(z)
        self.desc_d = self._create_tensor_descriptor(out)

        mode_a_arr = self._mode_array(mode_a)
        mode_b_arr = self._mode_array(mode_b)
        mode_c_arr = self._mode_array(mode_c)
        mode_d_arr = self._mode_array(mode_d)

        status = libcutensor.cutensorCreateElementwiseTrinary(
            self.handle,
            byref(self.op_desc),
            self.desc_a,
            mode_a_arr,
            c_int(self.op_a),
            self.desc_b,
            mode_b_arr,
            c_int(self.op_b),
            self.desc_c,
            mode_c_arr,
            c_int(self.op_c),
            self.desc_d,
            mode_d_arr,
            c_int(self.op_ab),
            c_int(self.op_abc),
            compute_desc,
        )
        if status != 0:
            raise RuntimeError(f"create elementwise trinary descriptor failed: {status}")

        status = libcutensor.cutensorCreatePlanPreference(
            self.handle,
            byref(self.plan_pref),
            c_int(CUTENSOR_ALGO_DEFAULT),
            c_int(CUTENSOR_JIT_MODE_NONE),
        )
        if status != 0:
            raise RuntimeError(f"create plan preference failed: {status}")

        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")

        status = libcutensor.cutensorCreatePlan(
            self.handle,
            byref(self.plan),
            self.op_desc,
            self.plan_pref,
            workspace_size,
        )
        if status != 0:
            raise RuntimeError(f"create plan failed: {status}")

        self.signature = signature

        return mode_a, mode_b, mode_c, mode_d

    def build_kernel_callable(self, x, y, z, alpha=1.0, beta=1.0, gamma=1.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        mode_a, mode_b, mode_c, mode_d = self.prepare(x, y, z, mode_a=mode_a, mode_b=mode_b, mode_c=mode_c, mode_d=mode_d, out=out)
        output = out
        if output is None:
            output_shape = _infer_output_shape(x, mode_a, y, mode_b, z, mode_c, mode_d)
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        alpha_val = self._scalar_value(alpha, x.dtype)
        beta_val = self._scalar_value(beta, x.dtype)
        gamma_val = self._scalar_value(gamma, x.dtype)

        def run_kernel():
            status = libcutensor.cutensorElementwiseTrinaryExecute(
                self.handle,
                self.plan,
                byref(alpha_val),
                c_void_p(x.data_ptr()),
                byref(beta_val),
                c_void_p(y.data_ptr()),
                byref(gamma_val),
                c_void_p(z.data_ptr()),
                c_void_p(output.data_ptr()),
                c_void_p(0),
            )
            if status != 0:
                raise RuntimeError(f"cutensorElementwiseTrinaryExecute failed: {status}")
            return output

        return run_kernel

    def __call__(self, x, y, z, alpha=1.0, beta=1.0, gamma=1.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        mode_a, mode_b, mode_c, mode_d = self.prepare(x, y, z, mode_a=mode_a, mode_b=mode_b, mode_c=mode_c, mode_d=mode_d, out=out)
        output = out
        if output is None:
            output_shape = _infer_output_shape(x, mode_a, y, mode_b, z, mode_c, mode_d)
            output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
        alpha_val = self._scalar_value(alpha, x.dtype)
        beta_val = self._scalar_value(beta, x.dtype)
        gamma_val = self._scalar_value(gamma, x.dtype)
        status = libcutensor.cutensorElementwiseTrinaryExecute(
            self.handle,
            self.plan,
            byref(alpha_val),
            c_void_p(x.data_ptr()),
            byref(beta_val),
            c_void_p(y.data_ptr()),
            byref(gamma_val),
            c_void_p(z.data_ptr()),
            c_void_p(output.data_ptr()),
            c_void_p(0),
        )
        if status != 0:
            raise RuntimeError(f"cutensorElementwiseTrinaryExecute failed: {status}")
        return output

    def __del__(self):
        if CUTENSOR_AVAILABLE:
            self._destroy_cache()
            if self.initialized and self.handle:
                try:
                    libcutensor.cutensorDestroy(self.handle)
                except Exception:
                    pass


class CuTensorIdentity(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_IDENTITY, dtype=dtype)


class CuTensorSqrt(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SQRT, dtype=dtype)


class CuTensorRelu(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_RELU, dtype=dtype)


class CuTensorConj(CuTensorUnary):
    def __init__(self, dtype=torch.complex64):
        super().__init__(CUTENSOR_OP_CONJ, dtype=dtype)


class CuTensorRcp(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_RCP, dtype=dtype)


class CuTensorSigmoid(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SIGMOID, dtype=dtype)


class CuTensorTanh(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_TANH, dtype=dtype)


class CuTensorAbs(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ABS, dtype=dtype)


class CuTensorExp(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_EXP, dtype=dtype)


class CuTensorLog(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_LOG, dtype=dtype)


class CuTensorNeg(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_NEG, dtype=dtype)


class CuTensorSin(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SIN, dtype=dtype)


class CuTensorCos(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_COS, dtype=dtype)


class CuTensorTan(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_TAN, dtype=dtype)


class CuTensorSinh(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SINH, dtype=dtype)


class CuTensorCosh(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_COSH, dtype=dtype)


class CuTensorAsin(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ASIN, dtype=dtype)


class CuTensorAcos(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ACOS, dtype=dtype)


class CuTensorAtan(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ATAN, dtype=dtype)


class CuTensorAsinh(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ASINH, dtype=dtype)


class CuTensorAcosh(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ACOSH, dtype=dtype)


class CuTensorAtanh(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ATANH, dtype=dtype)


class CuTensorCeil(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_CEIL, dtype=dtype)


class CuTensorFloor(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_FLOOR, dtype=dtype)


class CuTensorMish(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_MISH, dtype=dtype)


class CuTensorSwish(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SWISH, dtype=dtype)


class CuTensorSoftPlus(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SOFT_PLUS, dtype=dtype)


class CuTensorSoftSign(CuTensorUnary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_SOFT_SIGN, dtype=dtype)


class CuTensorAdd(CuTensorBinary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_ADD, dtype=dtype)


class CuTensorMul(CuTensorBinary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_MUL, dtype=dtype)


class CuTensorMax(CuTensorBinary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_MAX, dtype=dtype)


class CuTensorMin(CuTensorBinary):
    def __init__(self, dtype=torch.float32):
        super().__init__(CUTENSOR_OP_MIN, dtype=dtype)


class CuTensorContraction:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.handle = c_void_p()
        self.desc_a = c_void_p()
        self.desc_b = c_void_p()
        self.desc_c = c_void_p()
        self.desc_d = c_void_p()
        self.op_desc = c_void_p()
        self.plan_pref = c_void_p()
        self.plan = c_void_p()
        self.signature = None
        self.initialized = False

        if not CUTENSOR_AVAILABLE:
            return

        status = libcutensor.cutensorCreate(byref(self.handle))
        if status != 0:
            raise RuntimeError(f"cutensorCreate failed: {status}")
        self.initialized = True

    def _create_tensor_descriptor(self, tensor):
        ndim = tensor.ndim
        extents = (c_int64 * ndim)(*tensor.shape)
        strides = (c_int64 * ndim)(*tensor.stride())
        cuda_type = self._cuda_type(tensor.dtype)
        alignment = c_uint32(max(1, tensor.element_size()))
        desc = c_void_p()
        status = libcutensor.cutensorCreateTensorDescriptor(
            self.handle,
            byref(desc),
            c_uint32(ndim),
            extents,
            strides,
            c_int(cuda_type),
            alignment,
        )
        if status != 0:
            raise RuntimeError(f"create tensor descriptor failed: {status}")
        return desc

    def _mode_array(self, modes):
        return (c_int32 * len(modes))(*modes)

    def _cuda_type(self, dtype):
        if dtype == torch.float16:
            return CUDA_R_16F
        if dtype == torch.float32:
            return CUDA_R_32F
        if dtype == torch.float64:
            return CUDA_R_64F
        if dtype == torch.bfloat16:
            return CUDA_R_16BF
        if dtype == torch.complex64:
            return CUDA_C_32F
        if dtype == torch.complex128:
            return CUDA_C_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _compute_desc(self, dtype):
        if dtype == torch.float16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float32:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.float64:
            return CUTENSOR_COMPUTE_DESC_64F
        if dtype == torch.bfloat16:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex64:
            return CUTENSOR_COMPUTE_DESC_32F
        if dtype == torch.complex128:
            return CUTENSOR_COMPUTE_DESC_64F
        raise TypeError(f"unsupported dtype: {dtype}")

    def _scalar_value(self, value, dtype):
        if dtype == torch.complex64:
            value = complex(value)
            return (c_float * 2)(value.real, value.imag)
        if dtype == torch.complex128:
            value = complex(value)
            return (c_double * 2)(value.real, value.imag)
        if dtype == torch.float64:
            return c_double(value)
        return c_float(value)

    def _destroy_cache(self):
        if self.plan:
            libcutensor.cutensorDestroyPlan(self.plan)
            self.plan = c_void_p()
        if self.plan_pref:
            libcutensor.cutensorDestroyPlanPreference(self.plan_pref)
            self.plan_pref = c_void_p()
        if self.op_desc:
            libcutensor.cutensorDestroyOperationDescriptor(self.op_desc)
            self.op_desc = c_void_p()
        if self.desc_d:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_d)
            self.desc_d = c_void_p()
        if self.desc_c:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_c)
            self.desc_c = c_void_p()
        if self.desc_b:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_b)
            self.desc_b = c_void_p()
        if self.desc_a:
            libcutensor.cutensorDestroyTensorDescriptor(self.desc_a)
            self.desc_a = c_void_p()
        self.signature = None

    def _signature(self, a, b, c, mode_a, mode_b, mode_c, mode_d):
        return (
            a.dtype,
            tuple(a.shape),
            tuple(a.stride()),
            tuple(mode_a),
            tuple(b.shape),
            tuple(b.stride()),
            tuple(mode_b),
            tuple(c.shape),
            tuple(c.stride()),
            tuple(mode_c),
            tuple(mode_d),
        )

    def prepare(self, a, b, c=None, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        if not self.initialized:
            raise RuntimeError("cuTensor not initialized")
        if not a.is_cuda or not b.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if a.dtype != b.dtype:
            raise TypeError("input tensors must have the same dtype")

        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        mode_d = tuple(mode_d) if mode_d is not None else tuple(mode for mode in mode_a + mode_b if mode not in set(mode_a).intersection(mode_b))
        if len(set(mode_d)) != len(mode_d):
            raise ValueError("each output mode may appear at most once")
        output_shape = _infer_contraction_output_shape(a, mode_a, b, mode_b, mode_d)

        if c is None:
            c = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
        else:
            if not c.is_cuda:
                raise ValueError("addend tensor must be on CUDA")
            if c.dtype != a.dtype:
                raise TypeError("addend tensor must have the same dtype as inputs")
        mode_c = _validate_contraction_addend(c, mode_c if mode_c is not None else mode_d, mode_d, output_shape)

        if out is not None:
            if not out.is_cuda:
                raise ValueError("output tensor must be on CUDA")
            if out.dtype != a.dtype:
                raise TypeError("output tensor must have the same dtype as inputs")
            if tuple(out.shape) != tuple(output_shape):
                raise ValueError(f"output tensor shape mismatch: expected {output_shape}, got {tuple(out.shape)}")
        else:
            out = torch.empty(output_shape, device=a.device, dtype=a.dtype)

        signature = self._signature(a, b, c, mode_a, mode_b, mode_c, mode_d)
        if self.signature == signature and self.plan:
            return a, b, c, out, mode_a, mode_b, mode_c, mode_d

        self._destroy_cache()

        compute_desc = self._compute_desc(a.dtype)
        self.desc_a = self._create_tensor_descriptor(a)
        self.desc_b = self._create_tensor_descriptor(b)
        self.desc_c = self._create_tensor_descriptor(c)
        self.desc_d = self._create_tensor_descriptor(out)

        mode_a_arr = self._mode_array(mode_a)
        mode_b_arr = self._mode_array(mode_b)
        mode_c_arr = self._mode_array(mode_c)
        mode_d_arr = self._mode_array(mode_d)

        status = libcutensor.cutensorCreateContraction(
            self.handle,
            byref(self.op_desc),
            self.desc_a,
            mode_a_arr,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_b,
            mode_b_arr,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_c,
            mode_c_arr,
            c_int(CUTENSOR_OP_IDENTITY),
            self.desc_d,
            mode_d_arr,
            compute_desc,
        )
        if status != 0:
            raise RuntimeError(f"create contraction descriptor failed: {status}")

        status = libcutensor.cutensorCreatePlanPreference(
            self.handle,
            byref(self.plan_pref),
            c_int(CUTENSOR_ALGO_DEFAULT),
            c_int(CUTENSOR_JIT_MODE_NONE),
        )
        if status != 0:
            raise RuntimeError(f"create plan preference failed: {status}")

        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")

        status = libcutensor.cutensorCreatePlan(
            self.handle,
            byref(self.plan),
            self.op_desc,
            self.plan_pref,
            workspace_size,
        )
        if status != 0:
            raise RuntimeError(f"create plan failed: {status}")

        self.signature = signature
        return a, b, c, out, mode_a, mode_b, mode_c, mode_d

    def __call__(self, a, b, c=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
        a, b, c, output, mode_a, mode_b, mode_c, mode_d = self.prepare(
            a,
            b,
            c=c,
            mode_a=mode_a,
            mode_b=mode_b,
            mode_c=mode_c,
            mode_d=mode_d,
            out=out,
        )
        alpha_val = self._scalar_value(alpha, a.dtype)
        beta_val = self._scalar_value(beta, a.dtype)
        workspace_size = c_uint64(0)
        status = libcutensor.cutensorEstimateWorkspaceSize(
            self.handle,
            self.op_desc,
            self.plan_pref,
            c_int(CUTENSOR_WORKSPACE_DEFAULT),
            byref(workspace_size),
        )
        if status != 0:
            raise RuntimeError(f"estimate workspace failed: {status}")
        workspace = c_void_p(0)
        if workspace_size.value > 0:
            workspace_tensor = torch.empty(int(workspace_size.value), device=a.device, dtype=torch.uint8)
            workspace = c_void_p(workspace_tensor.data_ptr())
        status = libcutensor.cutensorContract(
            self.handle,
            self.plan,
            byref(alpha_val),
            c_void_p(a.data_ptr()),
            c_void_p(b.data_ptr()),
            byref(beta_val),
            c_void_p(c.data_ptr()),
            c_void_p(output.data_ptr()),
            workspace,
            workspace_size,
            c_void_p(0),
        )
        if status != 0:
            raise RuntimeError(f"cutensorContract failed: {status}")
        return output

    def __del__(self):
        if CUTENSOR_AVAILABLE:
            self._destroy_cache()
            if self.initialized and self.handle:
                try:
                    libcutensor.cutensorDestroy(self.handle)
                except Exception:
                    pass


class CuTensorContractionTrinary:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
        self.first = CuTensorContraction(dtype=dtype)
        self.second = CuTensorContraction(dtype=dtype)

    def __call__(self, a, b, c, d=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, mode_e=None, out=None):
        if not a.is_cuda or not b.is_cuda or not c.is_cuda:
            raise ValueError("input tensors must be on CUDA")
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise TypeError("input tensors must have the same dtype")

        mode_a = _normalize_modes(mode_a, a.ndim)
        mode_b = _normalize_modes(mode_b, b.ndim)
        mode_c = _normalize_modes(mode_c, c.ndim)
        contracted_modes = (set(mode_a) & set(mode_b)) | (set(mode_a) & set(mode_c)) | (set(mode_b) & set(mode_c))
        mode_e = tuple(mode_e) if mode_e is not None else tuple(mode for mode in mode_a + mode_b + mode_c if mode not in contracted_modes)
        if len(set(mode_e)) != len(mode_e):
            raise ValueError("each output mode may appear at most once")
        output_shape = _infer_trinary_contraction_output_shape(a, mode_a, b, mode_b, c, mode_c, mode_e)

        if d is None:
            d = torch.zeros(output_shape, device=a.device, dtype=a.dtype)
        else:
            if not d.is_cuda:
                raise ValueError("addend tensor must be on CUDA")
            if d.dtype != a.dtype:
                raise TypeError("addend tensor must have the same dtype as inputs")
        mode_d = _validate_trinary_contraction_addend(d, mode_d if mode_d is not None else mode_e, mode_e, output_shape)

        shared_modes = tuple(mode for mode in mode_a if mode in set(mode_b) and mode not in mode_e)
        intermediate_modes = tuple(mode for mode in mode_a + mode_b if mode not in shared_modes)
        temp = self.first(
            a,
            b,
            c=None,
            alpha=1.0,
            beta=0.0,
            mode_a=mode_a,
            mode_b=mode_b,
            mode_c=intermediate_modes,
            mode_d=intermediate_modes,
        )
        return self.second(
            temp,
            c,
            c=d,
            alpha=alpha,
            beta=beta,
            mode_a=intermediate_modes,
            mode_b=mode_c,
            mode_c=mode_d,
            mode_d=mode_e,
            out=out,
        )

    def __del__(self):
        pass


# Module-level executor cache for the `elementwise_trinary(...)` wrapper. Building a
# CuTensorTrinary (handle + tensor/operation descriptors + plan) is expensive
# (millisecond range); re-using one across calls with the same operator
# signature keeps the Python wrapper cost down to the actual GPU work.
_TRINARY_EXECUTOR_CACHE = {}


def _get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, dtype):
    key = (
        _resolve_operator(op_ab, BINARY_OPERATOR_MAP, "binary"),
        _resolve_operator(op_abc, BINARY_OPERATOR_MAP, "binary"),
        _resolve_operator(op_a, UNARY_OPERATOR_MAP, "unary"),
        _resolve_operator(op_b, UNARY_OPERATOR_MAP, "unary"),
        _resolve_operator(op_c, UNARY_OPERATOR_MAP, "unary"),
        dtype,
    )
    executor = _TRINARY_EXECUTOR_CACHE.get(key)
    if executor is None:
        executor = CuTensorTrinary(
            op_ab=op_ab,
            op_abc=op_abc,
            op_a=op_a,
            op_b=op_b,
            op_c=op_c,
            dtype=dtype,
        )
        _TRINARY_EXECUTOR_CACHE[key] = executor
    return executor


def elementwise_trinary(
    a,
    b,
    c,
    *,
    op_a="identity",
    op_b="identity",
    op_c="identity",
    op_ab="add",
    op_abc="add",
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    mode_a=None,
    mode_b=None,
    mode_c=None,
    mode_d=None,
    out=None,
):
    executor = _get_trinary_executor(op_ab, op_abc, op_a, op_b, op_c, a.dtype)
    return executor(
        a,
        b,
        c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        out=out,
    )


def contraction_trinary(a, b, c, *, d=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, mode_e=None, out=None):
    executor = _get_trinary_contraction_executor(a.dtype)
    return executor(
        a,
        b,
        c,
        d=d,
        alpha=alpha,
        beta=beta,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        mode_e=mode_e,
        out=out,
    )


def contraction(a, b, *, c=None, alpha=1.0, beta=0.0, mode_a=None, mode_b=None, mode_c=None, mode_d=None, out=None):
    executor = _get_gett_executor(a.dtype)
    return executor(
        a,
        b,
        c=c,
        alpha=alpha,
        beta=beta,
        mode_a=mode_a,
        mode_b=mode_b,
        mode_c=mode_c,
        mode_d=mode_d,
        out=out,
    )
