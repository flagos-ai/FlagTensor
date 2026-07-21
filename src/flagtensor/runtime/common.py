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

"""Vendor enumeration and hardware capability constants.

Following FlagGems runtime/common.py pattern.
"""
from enum import Enum


class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    METAX = 2
    ILUVATAR = 3
    MTHREADS = 4
    KUNLUNXIN = 5
    HYGON = 6
    AMD = 7
    AIPU = 8
    ASCEND = 9
    TSINGMICRO = 10
    SUNRISE = 11
    ENFLAME = 12
    SPACEMIT = 13
    THEAD = 14
    PPU = 15

    @classmethod
    def get_all_vendors(cls) -> dict:
        """Return {lowercase_name: enum_value} for all registered vendors."""
        return {member.name.lower(): member for member in cls}


# ---------------------------------------------------------------------------
# Quick detection: vendor_name → torch attribute
# ---------------------------------------------------------------------------
_VENDOR_TORCH_ATTR = {
    "ascend": "npu",
    "cambricon": "mlu",
    "enflame": "gcu",
    "hygon": "__hcu_version__",
    "iluvatar": "corex",
    "mthreads": "musa",
    "sunrise": "ptpu",
}

# ---------------------------------------------------------------------------
# Vendors that do NOT support certain dtypes
# ---------------------------------------------------------------------------
UNSUPPORT_FP64 = {
    vendors.AIPU,
    vendors.ASCEND,
    vendors.CAMBRICON,
    vendors.ENFLAME,
    vendors.ILUVATAR,
    vendors.KUNLUNXIN,
    vendors.MTHREADS,
    vendors.SUNRISE,
    vendors.SPACEMIT,
    vendors.TSINGMICRO,
}

UNSUPPORT_BF16 = {
    vendors.AIPU,
    vendors.SUNRISE,
    vendors.SPACEMIT,
}

UNSUPPORT_INT64 = {
    vendors.AIPU,
    vendors.ENFLAME,
    vendors.SPACEMIT,
    vendors.SUNRISE,
    vendors.TSINGMICRO,
}
