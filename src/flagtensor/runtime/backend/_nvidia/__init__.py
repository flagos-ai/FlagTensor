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

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='nvidia', device_name='cuda', device_query_cmd='nvidia-smi'
)
ARCH_MAP = {'9': 'hopper', '8': 'ampere'}
CUSTOMIZED_UNUSED_OPS = ()

# ---------------------------------------------------------------------------
# Baseline availability + factory
# ---------------------------------------------------------------------------
# NVIDIA's native baseline is cuTensor (libcutensor.so). It is available iff
# CUTENSOR_AVAILABLE is True (i.e. cuTensor is installed on the host). When
# cuTensor is missing, BASELINE_AVAILABLE is False and the benchmark tests
# are skipped with "baseline unavailable" — preserving the historical
# NVIDIA behaviour exactly.
from flagtensor.cutensor import CUTENSOR_AVAILABLE as _CUTENSOR_AVAILABLE
BASELINE_AVAILABLE = _CUTENSOR_AVAILABLE


def get_baseline_class(op_slug: str):
    """Return the NVIDIA-native baseline class for an operator slug.

    ``op_slug`` is the lowercased operator name with the ``CUTENSOR_OP_``
    prefix stripped, e.g. ``'abs'``, ``'add'``, ``'contraction'``,
    ``'elementwise_trinary'``, ``'block_sparse_contraction'``.

    Returns ``None`` if cuTensor is not installed or no baseline is
    registered for the slug.
    """
    from . import baseline as _baseline
    return _baseline.BASELINE_CLASSES.get(op_slug)


__all__ = ['*']
