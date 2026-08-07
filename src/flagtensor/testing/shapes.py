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

"""Shape utilities for testing."""

from typing import Iterable
from typing import Sequence

# Default shapes for pointwise operators (unary, binary)
DEFAULT_POINTWISE_TEST_SHAPES = (
    (1024,),  # 1D small
    (4096,),  # 1D medium
    (128, 128),  # 2D square
    (32, 64, 16),  # 3D multi-dimensional
)

# Default shapes for contraction operators
DEFAULT_CONTRACTION_TEST_SHAPES = (
    ((64, 32), (32, 48)),  # Standard matmul-like
    ((16, 8), (8, 5)),  # Small contraction
    ((4, 8, 16), (16, 10)),  # Tensor contraction
)


def default_identity_shapes() -> Iterable[tuple]:
    """Return default shapes for identity-like operators.

    Returns:
        Iterable of shape tuples.
    """
    return DEFAULT_POINTWISE_TEST_SHAPES


def default_pointwise_shapes() -> Sequence[tuple]:
    """Return default shapes for pointwise operators.

    Returns:
        Sequence of shape tuples.
    """
    return DEFAULT_POINTWISE_TEST_SHAPES


def default_binary_shapes() -> Sequence[tuple]:
    """Return default shapes for binary operators.

    Returns:
        Sequence of shape tuples.
    """
    return DEFAULT_POINTWISE_TEST_SHAPES


def default_contraction_shapes() -> Sequence[tuple]:
    """Return default shapes for contraction operators.

    Returns:
        Sequence of shape tuple pairs (input1_shape, input2_shape).
    """
    return DEFAULT_CONTRACTION_TEST_SHAPES
