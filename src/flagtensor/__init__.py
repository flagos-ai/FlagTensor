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

import sys

from flagtensor import runtime
from flagtensor.cutensor import BlockSparseTensor
from flagtensor.cutensor import BlockSparseTensorContraction
from flagtensor.cutensor import BlockSparseTensorDescriptor
from flagtensor.ops.CUTENSOR_OP_ADD import add
from flagtensor.ops.CUTENSOR_OP_ABS import abs
from flagtensor.ops.CUTENSOR_OP_ACOSH import acosh
from flagtensor.ops.CUTENSOR_OP_ACOS import acos
from flagtensor.ops.CUTENSOR_OP_ASIN import asin
from flagtensor.ops.CUTENSOR_OP_ASINH import asinh
from flagtensor.ops.CUTENSOR_OP_ATAN import atan
from flagtensor.ops.CUTENSOR_OP_ATANH import atanh
from flagtensor.ops.CUTENSOR_OP_BLOCK_SPARSE_TENSOR_CONTRACTION import block_sparse_contraction
from flagtensor.ops.CUTENSOR_OP_CEIL import ceil
from flagtensor.ops.CUTENSOR_OP_CONJ import conj
from flagtensor.ops.CUTENSOR_OP_COS import cos
from flagtensor.ops.CUTENSOR_OP_COSH import cosh
from flagtensor.ops.CUTENSOR_OP_EXP import exp
from flagtensor.ops.CUTENSOR_OP_FLOOR import floor
from flagtensor.ops.CUTENSOR_OP_GETT import contraction
from flagtensor.ops.CUTENSOR_OP_IDENTITY import identity
from flagtensor.ops.CUTENSOR_OP_LOG import log
from flagtensor.ops.CUTENSOR_OP_MISH import mish
from flagtensor.ops.CUTENSOR_OP_MIN import min
from flagtensor.ops.CUTENSOR_OP_MAX import max
from flagtensor.ops.CUTENSOR_OP_MUL import mul
from flagtensor.ops.CUTENSOR_OP_NEG import neg
from flagtensor.ops.CUTENSOR_OP_RCP import rcp
from flagtensor.ops.CUTENSOR_OP_RELU import relu
from flagtensor.ops.CUTENSOR_OP_SIGMOID import sigmoid
from flagtensor.ops.CUTENSOR_OP_SIN import sin
from flagtensor.ops.CUTENSOR_OP_SINH import sinh
from flagtensor.ops.CUTENSOR_OP_SOFT_PLUS import soft_plus
from flagtensor.ops.CUTENSOR_OP_SOFT_SIGN import soft_sign
from flagtensor.ops.CUTENSOR_OP_SQRT import sqrt
from flagtensor.ops.CUTENSOR_OP_SWISH import swish
from flagtensor.ops.CUTENSOR_OP_TAN import tan
from flagtensor.ops.CUTENSOR_OP_TANH import tanh
from flagtensor.ops.CUTENSOR_OP_TENSOR_CONTRACTION_TRINARY import contraction_trinary
try:
    from flagtensor.ops.CUTENSOR_OP_TRINARY_GENERIC import elementwise_trinary
except Exception:
    elementwise_trinary = None

__all__ = ["BlockSparseTensor", "BlockSparseTensorContraction", "BlockSparseTensorDescriptor", "add", "abs", "acosh", "acos", "asin", "asinh", "atan", "atanh", "block_sparse_contraction", "ceil", "exp", "floor", "contraction", "identity", "log", "mish", "min", "max", "mul", "soft_plus", "soft_sign", "sqrt", "relu", "conj", "cos", "cosh", "neg", "rcp", "sigmoid", "sin", "sinh", "swish", "tan", "tanh", "contraction_trinary", "elementwise_trinary"]

# Attempt to load C++ accelerated operators (requires building C extensions).
# The C++ ops use TritonJIT to launch kernels from C++, offering lower dispatch
# overhead. When available, they are exposed as flagtensor.c_ops.<name>.
HAS_CPP_OPS = False
try:
    from flagtensor import c_operators
    HAS_CPP_OPS = True

    # Build a mapping from C++ op names to Python convenience objects.
    c_ops = type(sys.modules[__name__].__name__ + ".c_ops", (), {})()
    _unary_names = [
        "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh",
        "ceil", "conj", "cos", "cosh", "exp", "floor", "identity",
        "log", "mish", "neg", "rcp", "relu", "sigmoid", "sin", "sinh",
        "soft_plus", "soft_sign", "sqrt", "swish", "tan", "tanh",
    ]
    for _name in _unary_names:
        _cpp_fn = getattr(c_operators, _name)
        setattr(c_ops, _name, staticmethod(lambda x, _fn=_cpp_fn: _fn(x)))
    _binary_names = ["add", "mul", "max", "min"]
    for _name in _binary_names:
        _cpp_fn = getattr(c_operators, _name)
        setattr(c_ops, _name, staticmethod(lambda a, b, _fn=_cpp_fn: _fn(a, b)))
    c_ops.contraction = c_operators.contraction
    c_ops.contraction_trinary = c_operators.contraction_trinary
    c_ops.elementwise_trinary = c_operators.elementwise_trinary
    c_ops.block_sparse_contraction = c_operators.block_sparse_contraction
except ImportError:
    c_ops = None

runtime.replace_customized_ops(globals())
