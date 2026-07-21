"""Operator registrar — registers FlagTensor ops into PyTorch's aten dispatch.

Ported from FlagGems runtime/op_registrar.py, simplified for FlagTensor.
The dispatch key is dynamically resolved from DeviceDetector — never hardcoded.
Adding a new vendor requires zero changes to this file.
"""

import warnings

from .backend.device import DeviceDetector


class GeneralOpRegistrar:
    """Register FlagTensor operator implementations into torch.library.

    On construction, filters the config list against include/exclude lists,
    then registers each surviving (aten_key, flagtensor_fn) pair into
    ``torch.library.Library("aten", "IMPL").impl()`` under the dispatch key
    returned by DeviceDetector (e.g. "CUDA" for NVIDIA, "PrivateUse1" for AMD).

    Parameters
    ----------
    config : iterable of (str, callable)
        Pairs of ``(aten_operator_name, flagtensor_implementation)``.
    lib : torch.library.Library
        A library instance created with ``torch.library.Library("aten", "IMPL")``.
    include_ops : list of str, optional
        Whitelist of function names. When set, *only* these ops are registered
        (the exclude list is ignored).
    exclude_ops : list of str, optional
        Blacklist of function names. Ignored when *include_ops* is set.
    """

    def __init__(self, config, lib, include_ops=None, exclude_ops=None):
        self._device = DeviceDetector()
        self._reg_key = self._device.dispatch_key   # e.g. "CUDA"
        self._lib = lib

        self._all_ops = []
        self._all_keys = []

        if include_ops:
            self._include_ops = list(include_ops)
            self._exclude_ops = []
            self._config = self._filter_by_include(config)
        else:
            self._include_ops = []
            self._exclude_ops = list(exclude_ops or [])
            self._config = self._filter_by_exclude(config)

        self._for_each()

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _filter_by_include(self, config):
        """Keep only items whose op name or function name is in include_ops."""
        included = []
        for item in config:
            op_name, fn = item[0], item[1]
            fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
            if fn_name in self._include_ops or op_name in self._include_ops:
                included.append(item)
        if not included:
            warnings.warn(
                "GeneralOpRegistrar: include list matched zero ops. "
                "No operators will be registered."
            )
        return included

    def _filter_by_exclude(self, config):
        """Remove items whose function name is in exclude_ops."""
        return [
            item for item in config
            if item[1].__name__ not in self._exclude_ops
        ]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_impl(self, key, fn):
        """Register a single implementation under the current dispatch key."""
        self._lib.impl(key, fn, self._reg_key)

    def _for_each(self):
        for key, func in self._config:
            try:
                self.register_impl(key, func)
                self._all_ops.append(func.__name__)
                self._all_keys.append(key)
            except RuntimeError as e:
                msg = str(e)
                if "already a kernel registered" in msg:
                    # Already registered (e.g. prior enable() call) — skip silently.
                    continue
                warnings.warn(
                    f"Failed to register '{key}' → '{func.__name__}': {e}"
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to register '{key}' → '{func.__name__}': {e}"
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_all_ops(self):
        """Return list of registered function names."""
        return self._all_ops

    def get_all_keys(self):
        """Return list of registered aten operator keys."""
        return self._all_keys

    @property
    def reg_key(self):
        """The PyTorch dispatch key this registrar targets (e.g. 'CUDA')."""
        return self._reg_key

    @property
    def vendor_name(self):
        """The detected vendor name (e.g. 'nvidia')."""
        return self._device.vendor_name

    @property
    def device_name(self):
        """The detected device name (e.g. 'cuda')."""
        return self._device.name
