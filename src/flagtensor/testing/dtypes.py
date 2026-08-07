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

"""Dtype utilities for testing."""

from typing import List

import torch

from flagtensor.runtime.dtype_capability import dtype_capability


def correctness_dtypes(
    include_float64: bool = False,
    include_bfloat16: bool = False,
    include_fp8: bool = False,
    include_int: bool = False,
) -> List[torch.dtype]:
    """Return default dtypes for correctness testing.

    Args:
        include_float64: Whether to include float64 dtype.
        include_bfloat16: Whether to include bfloat16 dtype.
        include_fp8: Whether to include FP8 dtypes supported by the device.
        include_int: Whether to include integer dtypes supported by the device.

    Returns:
        List of torch dtypes for testing.
    """
    dtypes = [torch.float16, torch.float32]
    if include_bfloat16 and torch.bfloat16 in dtype_capability.supported_dtypes:
        dtypes.append(torch.bfloat16)
    if include_float64:
        dtypes.append(torch.float64)
    if include_fp8:
        dtypes.extend(sorted(dtype_capability.supported_fp8, key=str))
    if include_int:
        dtypes.extend(sorted(dtype_capability.supported_int, key=str))
    return dtypes
