import ctypes
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

try:
    libcutensor = ctypes.CDLL("libcutensor.so")
    CUTENSOR_AVAILABLE = True
except OSError:
    libcutensor = None
    CUTENSOR_AVAILABLE = False

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
