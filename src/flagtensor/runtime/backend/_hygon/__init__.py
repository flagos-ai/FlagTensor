# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS BASIS",
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hygon DCU backend module.

Hygon DCU (Deep Computing Unit) is driven through a CUDA-compatible
software stack: the ``torch_hcu`` plugin (loaded at Python startup) hooks
into ``torch.cuda`` and redirects CUDA runtime / driver calls to the
Hygon DCU runtime. From the application's perspective the device is
accessed through ``torch.cuda`` exactly like an NVIDIA GPU, but the
underlying kernels are DCU-native (``libdtkblas``, ``libdtkdnn``, ...).

The SDK ships the ``hygon-smi`` query tool and installs the runtime
under ``/opt/hYGON/dtk`` (or equivalent path).  The ``torch.__hcu_version__``
marker attribute distinguishes Hygon DCU hosts from NVIDIA CUDA hosts.

No ``libcutensor.so`` is provided, so cuTensor is unavailable and the
vendor-native baseline for FlagTensor benchmarks is the PyTorch-native op
path (which dispatches to the Hygon DCU vendor libraries via the
``torch_hcu`` symbol-rewrite hook), exactly analogous to the PPU,
Iluvatar and Kunlunxin backends.

This module exposes:
    * ``vendor_info``        — VendorInfoBase used by the runtime detector
    * ``ARCH_MAP``           — maps compute-capability major → arch name
    * ``BASELINE_AVAILABLE`` — whether this vendor's baseline can run
    * ``get_baseline_class`` — factory returning the per-op baseline class
"""

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='hygon',
    device_name='cuda',
    device_query_cmd='hygon-smi',
)

# Hygon DCU reports compute capability through the CUDA-compat layer.
# The DCU K100 / K100 AI variants report compute capability 9.0 (gfx9-class)
# while older DCU variants report 8.0. We reuse the NVIDIA Ampere / Hopper
# architecture specialisations — the autotune configs and kernel tunings
# are identical because both target the same compute capability family.
# The runtime backend loader resolves arch paths by scanning sibling
# directories of the vendor module, so we expose 'ampere' / 'hopper' here
# and let the BackendArchEvent resolve it against the _nvidia/ampere/
# / _nvidia/hopper/ path via the shared ARCH_MAP convention.
ARCH_MAP = {'8': 'ampere', '9': 'hopper'}
CUSTOMIZED_UNUSED_OPS = ()

# ---------------------------------------------------------------------------
# Baseline availability + factory
# ---------------------------------------------------------------------------
# Hygon's native baseline is PyTorch-native ops (which dispatch to the
# DCU vendor libraries via the torch_hcu hook).  It is always available
# on a real Hygon DCU device.
BASELINE_AVAILABLE = True


def get_baseline_class(op_slug: str):
    """Return the Hygon-native baseline class for an operator slug.

    ``op_slug`` is the lowercased operator name with the ``CUTENSOR_OP_``
    prefix stripped, e.g. ``'abs'``, ``'add'``, ``'contraction'``,
    ``'elementwise_trinary'``, ``'block_sparse_contraction'``.

    Returns ``None`` if no baseline is registered for the slug.
    """
    from . import baseline as _baseline
    return _baseline.BASELINE_CLASSES.get(op_slug)


__all__ = ['*']
