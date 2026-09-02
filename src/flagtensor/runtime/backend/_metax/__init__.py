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

"""MetaX (C500/C550) backend module.

MetaX accelerators are driven through the MACA SDK (``/opt/maca-3.7.1``):
MACA is MetaX's CUDA-compatible runtime, and the ``torch-maca`` PyTorch
plugin exposes the device through ``torch.cuda`` (``get_device_properties``
reports a board name like ``MetaX C550``). PyTorch dispatches elementwise
and GEMM ops to MetaX's vendor libraries (``libmcblas`` / ``libmcdnn`` /
``libmcfft`` ...) via the standard ``aten`` dispatcher.

There is no MetaX-native tensor library equivalent to cuTensor (no
``libcutensor`` ships with the MACA SDK), so the MetaX baseline for
FlagTensor benchmarks is the PyTorch-native op path (which dispatches to
MetaX vendor kernels), exactly analogous to the PPU / Iluvatar backends.
See ``_metax/baseline.py`` for the implementation.

Unlike PPU / Iluvatar (whose baseline module is prepared but not yet
wired into the benchmark harness), the MetaX baseline is actively loaded
by ``benchmark_core.Benchmark._baseline_module`` via the
``BASELINE_MODULE_NAME`` sentinel defined below. The sentinel is opt-in:
no other vendor module sets it, so NVIDIA (cuTensor), Ascend
(torch_npu) and PPU/Iluvatar (triton-only) keep their existing baseline
behaviour byte-for-byte.

Detection (see ``runtime/backend/device.py``):
    1. Env vars: ``GEMS_VENDOR=metax`` / ``FLAGGEMS_VENDOR=metax`` /
       ``GEMS_BACKEND=metax`` / ``FLAGGEMS_BACKEND=metax`` (most reliable
       when running under a stock PyTorch that does not see MetaX via
       ``torch.cuda``).
    2. Quick torch probe: ``torch.cuda.get_device_properties(0).name``
       starting with ``METAX`` (active under the torch-maca plugin).
    3. System query: ``mx-smi`` (MX-SMI, shipped with the MACA SDK).

This module exposes:
    * ``vendor_info``           — VendorInfoBase used by the runtime detector
    * ``ARCH_MAP``              — no arch specialisation (configs fall back
                                  to the shared NVIDIA defaults)
    * ``BASELINE_AVAILABLE``    — whether this vendor's baseline can run
    * ``BASELINE_MODULE_NAME``  — sentinel opting into benchmark_core wiring
    * ``get_baseline_class``    — factory returning the per-op baseline class
"""

import os

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='metax',
    device_name='cuda',
    device_query_cmd='mx-smi',
)

# No MetaX-specific arch specialisation: the vendor-level tune configs
# (tune_configs.yaml) ship the generic elementwise configs, and other ops
# fall back to the shared NVIDIA defaults (backend_utils.get_tune_config).
ARCH_MAP = {}
CUSTOMIZED_UNUSED_OPS = ()

# ---------------------------------------------------------------------------
# Baseline availability + factory
# ---------------------------------------------------------------------------
# MetaX's native baseline is PyTorch-native ops (which dispatch to the
# MetaX vendor libraries via the standard aten dispatcher). It is always
# available on a real MetaX device (torch-maca makes torch.cuda work).
BASELINE_AVAILABLE = True

# Opt-in sentinel consumed by ``benchmark_core.Benchmark._baseline_module``.
# When set, the benchmark harness loads ``_{vendor_name}.{BASELINE_MODULE_NAME}``
# and resolves ``CuTensor{Op}`` classes from it. Only this module sets the
# sentinel, so other vendors' baseline resolution is unchanged.
BASELINE_MODULE_NAME = 'baseline'


def get_baseline_class(op_slug: str):
    """Return the MetaX-native baseline class for an operator slug.

    ``op_slug`` is the lowercased operator name with the ``CUTENSOR_OP_``
    prefix stripped, e.g. ``'abs'``, ``'add'``, ``'contraction'``,
    ``'elementwise_trinary'``, ``'block_sparse_contraction'``.

    Returns ``None`` if no baseline is registered for the slug.
    """
    from . import baseline as _baseline
    return _baseline.BASELINE_CLASSES.get(op_slug)


__all__ = ['*']
