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

"""Kunlunxin XPU (P800 / R200) backend module.

Kunlunxin XPU devices are driven through a CUDA-compatible software stack:
the ``torch_xmlir`` plugin (loaded via ``xpytorch_import_hook`` at Python
startup) hooks into ``torch.cuda`` and redirects CUDA runtime / driver
calls to the Kunlunxin XPU runtime (``libxpurt.so``).  From the
application's perspective the device is accessed through
``torch.cuda`` exactly like an NVIDIA GPU, but the underlying kernels are
XPU-native (``libxpu_blas.so``, ``libxpu_dnn.so``, ...).

The SDK ships the ``xpu-smi`` query tool and installs the runtime under
``/usr/local/xpu`` (or ``/opt/xre`` inside the official Docker image).
No ``libcutensor.so`` is provided, so cuTensor is unavailable and the
vendor-native baseline for FlagTensor benchmarks is the PyTorch-native op
path (which dispatches to the Kunlunxin vendor libraries via the
``torch_xmlir`` symbol-rewrite hook), exactly analogous to the PPU and
Iluvatar backends.

This module exposes:
    * ``vendor_info``        — VendorInfoBase used by the runtime detector
    * ``ARCH_MAP``           — maps compute-capability major → arch name
    * ``BASELINE_AVAILABLE`` — whether this vendor's baseline can run
    * ``get_baseline_class`` — factory returning the per-op baseline class
"""

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='kunlunxin',
    device_name='cuda',
    device_query_cmd='xpu-smi',
)

# Kunlunxin P800 reports compute capability 8.6 (Ampere-class).  It reuses
# the NVIDIA Ampere architecture specialisation — the autotune configs and
# kernel tunings are identical because both target the same compute
# capability.  The runtime backend loader resolves arch paths by scanning
# sibling directories of the vendor module, so we expose 'ampere' here and
# let the BackendArchEvent resolve it against the _nvidia/ampere/ path via
# the shared ARCH_MAP convention.
ARCH_MAP = {'8': 'ampere'}
CUSTOMIZED_UNUSED_OPS = ()

# ---------------------------------------------------------------------------
# Baseline availability + factory
# ---------------------------------------------------------------------------
# Kunlunxin's native baseline is PyTorch-native ops (which dispatch to the
# XPU vendor libraries via the torch_xmlir hook).  It is always available
# on a real Kunlunxin device.
BASELINE_AVAILABLE = True


def get_baseline_class(op_slug: str):
    """Return the Kunlunxin-native baseline class for an operator slug.

    ``op_slug`` is the lowercased operator name with the ``CUTENSOR_OP_``
    prefix stripped, e.g. ``'abs'``, ``'add'``, ``'contraction'``,
    ``'elementwise_trinary'``, ``'block_sparse_contraction'``.

    Returns ``None`` if no baseline is registered for the slug.
    """
    from . import baseline as _baseline
    return _baseline.BASELINE_CLASSES.get(op_slug)


__all__ = ['*']
