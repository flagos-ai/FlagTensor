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

"""Alibaba PPU (PPU-ZW810E) backend module.

PPU is a CUDA-compatible accelerator (sm80, CUDA 12.9 SDK at
``/usr/local/PPU_SDK``). It speaks through ``torch.cuda`` and reuses
NVIDIA's Ampere architecture specialisation (autotune configs, kernel
tuning) because its compute capability is 8.x.

PPU ships its own vendor math libraries (``libacblas``, ``libacdnn``,
``libacsparse``, ``libacfft``, ``libacsolver``, ``libacrand``) which
PyTorch dispatches to natively via ``torch.matmul`` / elementwise ops.
There is no PPU-native tensor library equivalent to cuTensor, so the
PPU baseline for FlagTensor benchmarks is the PyTorch-native op path
(see ``_ppu/baseline.py`` for the full rationale and implementation).

This module exposes:
    * ``vendor_info``       — VendorInfoBase used by the runtime detector
    * ``ARCH_MAP``          — maps compute-capability major → arch name
    * ``BASELINE_AVAILABLE`` — whether this vendor's baseline can run
    * ``get_baseline_class``— factory returning the per-op baseline class
"""

import os

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='ppu',
    device_name='cuda',
    device_query_cmd='ppu-smi',
)

# PPU is sm80 (Ampere-class). It reuses the NVIDIA Ampere architecture
# specialisation — the autotune configs and kernel tunings are identical
# because both target the same compute capability. The runtime backend
# loader resolves arch paths by scanning sibling directories of the
# vendor module, so we expose 'ampere' here and let the BackendArchEvent
# resolve it against the _nvidia/ampere/ path via the shared ARCH_MAP
# convention.
ARCH_MAP = {'8': 'ampere'}
CUSTOMIZED_UNUSED_OPS = ()

# ---------------------------------------------------------------------------
# Baseline availability + factory
# ---------------------------------------------------------------------------
# PPU's native baseline is PyTorch-native ops (which dispatch to acblas /
# acdnn vendor kernels). It is always available on a real PPU device.
BASELINE_AVAILABLE = True


def get_baseline_class(op_slug: str):
    """Return the PPU-native baseline class for an operator slug.

    ``op_slug`` is the lowercased operator name with the ``CUTENSOR_OP_``
    prefix stripped, e.g. ``'abs'``, ``'add'``, ``'contraction'``,
    ``'elementwise_trinary'``, ``'block_sparse_contraction'``.

    Returns ``None`` if no baseline is registered for the slug.
    """
    from . import baseline as _baseline
    return _baseline.BASELINE_CLASSES.get(op_slug)


__all__ = ['*']
