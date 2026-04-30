"""FlagTensor testing utilities package.

This package provides centralized testing helpers for correctness validation,
including assertion utilities, shape generators, and dtype helpers.
"""

from .assertions import assert_close
from .assertions import assert_equal
from .assertions import DEFAULT_CORRECTNESS_TOLERANCES
from .assertions import get_tolerance
from .dtypes import correctness_dtypes
from .shapes import DEFAULT_CONTRACTION_TEST_SHAPES
from .shapes import DEFAULT_POINTWISE_TEST_SHAPES
from .shapes import default_binary_shapes
from .shapes import default_contraction_shapes
from .shapes import default_identity_shapes
from .shapes import default_pointwise_shapes

__all__ = [
    "assert_close",
    "assert_equal",
    "get_tolerance",
    "correctness_dtypes",
    "DEFAULT_CONTRACTION_TEST_SHAPES",
    "DEFAULT_CORRECTNESS_TOLERANCES",
    "DEFAULT_POINTWISE_TEST_SHAPES",
    "default_binary_shapes",
    "default_contraction_shapes",
    "default_identity_shapes",
    "default_pointwise_shapes",
]
