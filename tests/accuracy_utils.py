import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flagtensor.testing import assert_close
from flagtensor.testing import assert_equal
from flagtensor.testing import correctness_dtypes
from flagtensor.testing import DEFAULT_CONTRACTION_TEST_SHAPES
from flagtensor.testing import DEFAULT_CORRECTNESS_TOLERANCES
from flagtensor.testing import DEFAULT_POINTWISE_TEST_SHAPES
from flagtensor.testing import default_binary_shapes
from flagtensor.testing import default_contraction_shapes
from flagtensor.testing import default_pointwise_shapes
from flagtensor.testing import get_tolerance

__all__ = [
    "assert_close",
    "assert_equal",
    "correctness_dtypes",
    "DEFAULT_CONTRACTION_TEST_SHAPES",
    "DEFAULT_CORRECTNESS_TOLERANCES",
    "DEFAULT_POINTWISE_TEST_SHAPES",
    "default_binary_shapes",
    "default_contraction_shapes",
    "default_pointwise_shapes",
    "get_tolerance",
]
