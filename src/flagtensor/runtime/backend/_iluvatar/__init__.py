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

"""Iluvatar CoreX (BI-V150) backend module.

Iluvatar GPUs are driven through a CUDA-compatible stack: the CoreX SDK
(``/usr/local/corex``) provides CUDA shims (``libcudart``, ``libcublas``,
``libcudnn``, ...) and PyTorch exposes the device through ``torch.cuda``
(plus the ``torch.corex`` marker attribute). Devices report names like
``Iluvatar BI-V150`` and the SDK ships the ``ixsmi`` query tool.

The CoreX SDK also ships a placeholder ``libcutensor.so`` that dlopens but
lacks the data symbols (``CUTENSOR_COMPUTE_DESC_*``) required by
``flagtensor.cutensor`` — there is no usable cuTensor on Iluvatar, so the
Iluvatar baseline for FlagTensor benchmarks is the PyTorch-native op path
(which dispatches to Iluvatar vendor libraries), exactly analogous to the
PPU backend. See ``_iluvatar/baseline.py``.

This module exposes:
    * ``vendor_info``        — VendorInfoBase used by the runtime detector
    * ``ARCH_MAP``           — no arch specialisation (configs fall back to
                               the shared NVIDIA defaults)
    * ``BASELINE_AVAILABLE`` — whether this vendor's baseline can run
    * ``get_baseline_class`` — factory returning the per-op baseline class
"""

import os

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='iluvatar',
    device_name='cuda',
    device_query_cmd='ixsmi',
)

# No Iluvatar-specific arch specialisation: the vendor-level tune configs
# fall back to the shared NVIDIA defaults (backend_utils.get_tune_config)
# and the elementwise heuristics ship in heuristics_config_utils.py.
ARCH_MAP = {}
CUSTOMIZED_UNUSED_OPS = ()

# ---------------------------------------------------------------------------
# Baseline availability + factory
# ---------------------------------------------------------------------------
# Iluvatar's native baseline is PyTorch-native ops (which dispatch to the
# CoreX vendor libraries via the standard aten dispatcher). It is always
# available on a real Iluvatar device.
BASELINE_AVAILABLE = True


def get_baseline_class(op_slug: str):
    """Return the Iluvatar-native baseline class for an operator slug.

    ``op_slug`` is the lowercased operator name with the ``CUTENSOR_OP_``
    prefix stripped, e.g. ``'abs'``, ``'add'``, ``'contraction'``,
    ``'elementwise_trinary'``, ``'block_sparse_contraction'``.

    Returns ``None`` if no baseline is registered for the slug.
    """
    from . import baseline as _baseline
    return _baseline.BASELINE_CLASSES.get(op_slug)


__all__ = ['*']
