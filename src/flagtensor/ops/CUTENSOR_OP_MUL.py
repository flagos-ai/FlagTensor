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

import torch
from flagtensor.ops.CUTENSOR_OP_BINARY_GENERIC import binary_generic


def mul(x: torch.Tensor, y: torch.Tensor, *, mode_x=None, mode_y=None, mode_out=None, out=None) -> torch.Tensor:
    return binary_generic(x, y, op="mul", mode_x=mode_x, mode_y=mode_y, mode_out=mode_out, out=out)
