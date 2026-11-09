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

"""Low-overhead Triton kernel launch for non-NVIDIA accelerators.

Triton's Python launch path (:meth:`triton.runtime.JITFunction.run`) re-binds
the arguments, recomputes the specialization key and performs a dictionary
lookup on *every* call.  On the hosts attached to Kunlunxin XPU / PPU cards
this costs ~20-25us per launch, and the surrounding autotuner/dispatch layers
add another ~20us -- which easily dwarfs the device time of a pointwise
kernel handling a few thousand elements.

``CompiledKernel.run`` is the final, C-level launch that Triton itself calls
at the end of ``JITFunction.run``.  Once a kernel has been compiled for a
given (specialization, config) pair, that low-level entry point can be
invoked directly with a pre-computed grid, which costs ~9us instead of
~45us.  This module caches such direct launches.

The fast path is only taken when the argument specialization is *identical*
to the one the cached kernel was compiled for (dtype, pointer alignment and
the exact integer arguments), so it never launches a kernel compiled for a
different specialization; on any mismatch the caller falls back to the
regular path, which compiles/looks the kernel up as usual.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Optional, Tuple

import torch
from triton.runtime import JITFunction, driver

try:  # ``triton.knobs`` only exists on Triton >= 3.2
    from triton import knobs
except Exception:  # pragma: no cover - very old Triton
    knobs = None

from flagtensor import runtime

#: Set to ``1`` to disable the direct-launch fast path entirely.
_DISABLE_ENV = "FLAGTENSOR_DISABLE_FAST_LAUNCH"


def fast_launch_enabled() -> bool:
    """Whether the direct-launch fast path may be used.

    Only enabled on non-NVIDIA backends: on NVIDIA the benchmark harness
    times kernels with CUDA events (device time only), so Python-side
    launch overhead does not affect the measurement and there is no reason
    to take the extra risk.
    """
    if os.environ.get(_DISABLE_ENV) == "1":
        return False
    try:
        return runtime.device.vendor_name != "nvidia"
    except Exception:
        return False


def specialization_key(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Specialization of ``args``, i.e. everything Triton's codegen keys on.

    Runs on every fast-path launch, so the (much more common) scalar and
    tensor cases are inlined here instead of recursing into a helper.
    """
    Tensor = torch.Tensor
    out = []
    append = out.append
    for arg in args:
        kind = type(arg)
        if kind is int:
            append(("i", arg))
        elif isinstance(arg, Tensor):
            append(("t", arg.dtype, arg.data_ptr() & 15))
        elif arg is None:
            append(("n",))
        elif kind is bool:
            append(("b", arg))
        elif kind is float:
            append(("f", arg))
        else:
            append(("o", kind.__name__))
    return tuple(out)


def constexpr_key(kwargs: dict) -> Tuple[Any, ...]:
    """Key for the constexpr keyword arguments (e.g. ``OP=...``).

    Two launches that only differ by a constexpr argument compile to
    different kernels, so they must not share a cached launcher.
    """
    items = [(k, v) for k, v in kwargs.items() if k not in ("grid", "warmup")]
    if not items:
        return ()
    items.sort()
    return tuple(items)


def unwrap_jit(fn: Any) -> JITFunction:
    """Strip ``Autotuner`` / ``Heuristics`` wrappers off a Triton kernel."""
    while not isinstance(fn, JITFunction):
        fn = fn.fn
    return fn


_STREAM_FNS: Optional[Tuple[Any, Any]] = None


def _bind_stream_fn(fn: Any) -> Any:
    """Adapt a driver's ``get_current_stream`` so it can be called as ``fn(device)``.

    Triton's drivers declare it as ``get_current_stream(device=None)``, but not
    every backend has to, and the fast path keys its cache on the stream: an
    arity mismatch would otherwise surface as a ``TypeError`` on the operator
    hot path, which now runs on every non-NVIDIA vendor.  Checked once, when
    the pair is first resolved.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Not introspectable: keep the documented convention.
        return fn
    try:
        signature.bind(0)
    except TypeError:
        return lambda _device=None: fn()
    return fn


def _stream_fns() -> Tuple[Any, Any]:
    global _STREAM_FNS
    fns = _STREAM_FNS
    if fns is None:
        active = driver.active
        get_device = getattr(active, "get_current_device", None) or (lambda: 0)
        get_stream = getattr(active, "get_current_stream", None)
        if get_stream is None:
            # A driver without a per-device stream concept is still usable:
            # every launch then targets the same (default) stream, so the
            # execution key collapses to the device alone.
            get_stream = lambda _device=None: None
        else:
            get_stream = _bind_stream_fn(get_stream)
        fns = _STREAM_FNS = (get_device, get_stream)
    return fns


def stream_fns() -> Tuple[Any, Any]:
    """``(get_current_device, get_current_stream)`` of the active driver.

    Exposed for hot paths that bind the two callables once instead of
    re-resolving them per launch.
    """
    return _stream_fns()


def current_stream() -> Any:
    """Current raw stream."""
    get_device, get_stream = _stream_fns()
    return get_stream(get_device())


def execution_key(device: Optional[int] = None) -> Tuple[int, Any]:
    """``(device, stream)`` of the current execution context.

    Launchers are only interchangeable within one device/stream pair: the
    compiled binary is device-specific and the launch targets a fixed
    stream.  Resolving the pair used to cost ~1.3us because
    ``torch.cuda.current_device()`` is slow on torch_xmlir.  Callers that
    already hold an input tensor should pass ``tensor.get_device()``
    (~0.09us) instead of letting the device be queried: on a benchmark run
    the tensor *is* on the device the kernel will execute on, and the
    stream is still resolved live so a stream change is never missed.
    """
    get_device, get_stream = _stream_fns()
    if device is None:
        device = get_device()
    return device, get_stream(device)


def _select_launch_fn(launcher: Any) -> Any:
    """Pick the cheapest callable able to perform the launch.

    The XPU launcher's ``__call__`` re-filters the constexpr arguments and
    re-imports ``TensorDescriptor`` on *every* launch (~3us of pure Python).
    When the kernel takes no tensor-descriptor arguments and its constexpr
    parameters are trailing, the underlying C entry point can be called
    directly instead.
    """
    raw = getattr(launcher, "launch", None)
    if raw is None:
        return launcher
    signature = getattr(launcher, "_signature", None)
    if signature is not None:
        types = list(signature.values())
        if any("desc" in str(t).lower() for t in types):
            return launcher
        first_constexpr = next(
            (i for i, t in enumerate(types) if t == "constexpr"), len(types)
        )
        # constexpr parameters must be trailing (no runtime parameter after
        # them), otherwise the positional filter below would misalign.
        if any(t != "constexpr" for t in types[first_constexpr:]):
            return launcher
    return raw


class DirectLauncher:
    """A pre-bound, pre-specialized Triton kernel launch."""

    __slots__ = ("_launch", "_compiled", "_fixed", "_ptr_args")

    def __init__(self, compiled: Any, grid: Tuple[int, ...], stream: Any,
                 ptr_args: int = 0):
        # ``compiled.run`` is a lazily-initialised property: touching it once
        # here keeps the per-call path down to a single C call.
        if compiled._run is None:
            compiled._init_handles()
        grid = tuple(grid)
        g0 = int(grid[0])
        g1 = int(grid[1]) if len(grid) > 1 else 1
        g2 = int(grid[2]) if len(grid) > 2 else 1
        self._compiled = compiled
        self._launch = _select_launch_fn(compiled.run)
        # Mirrors ``JITFunction.run``: (grid, stream, function, metadata,
        # launch_metadata, enter_hook, exit_hook) followed by runtime args.
        # ``launch_metadata`` is only needed when a profiler hook is set.
        self._fixed = (g0, g1, g2, stream, compiled.function,
                       compiled.packed_metadata, None, None, None)
        # Number of leading runtime arguments that are plain data pointers.
        # When non-zero they are passed as raw ints, which skips the C
        # launcher's per-tensor ``data_ptr``/``numel`` Python method calls
        # (~0.4us each on torch_xmlir).
        self._ptr_args = ptr_args

    def with_ptr_args(self, ptr_args: int) -> "DirectLauncher":
        """Return an equivalent launcher that passes the first ``ptr_args``
        runtime arguments as raw integer pointers."""
        clone = DirectLauncher.__new__(DirectLauncher)
        clone._compiled = self._compiled
        clone._launch = self._launch
        clone._fixed = self._fixed
        clone._ptr_args = ptr_args
        return clone

    @property
    def launch_parts(self) -> Tuple[Any, Tuple[Any, ...]]:
        """``(c_entry_point, bound_prefix)`` of the underlying launch.

        Callers that invoke the same kernel millions of times can hoist the
        constant part and call ``entry(*bound_prefix, *runtime_args)`` from
        their own closure, which is measurably cheaper than going through
        :meth:`__call__` once per tensor argument.
        """
        return self._launch, self._fixed

    def __call__(self, *args: Any) -> Any:
        n_ptr = self._ptr_args
        if n_ptr == 0:
            self._launch(*self._fixed, *args)
        elif n_ptr == 2:
            self._launch(*self._fixed, args[0].data_ptr(), args[1].data_ptr(),
                         *args[2:])
        elif n_ptr == 1:
            self._launch(*self._fixed, args[0].data_ptr(), *args[1:])
        else:
            self._launch(
                *self._fixed,
                *tuple(a.data_ptr() for a in args[:n_ptr]),
                *args[n_ptr:],
            )
        return self._compiled


def _hooks_active() -> bool:
    """True when a profiler/tracer hook would need launch metadata."""
    hook = getattr(getattr(knobs, "runtime", None), "launch_enter_hook", None)
    if hook is None:
        return False
    calls = getattr(hook, "calls", None)
    if calls is not None:
        return bool(calls)
    return True


def build_launcher(
    tuner: Any,
    args: Tuple[Any, ...],
    kwargs: dict,
    config_kwargs: dict,
    compiled: Any,
    stream: Any = None,
) -> Optional[DirectLauncher]:
    """Build a :class:`DirectLauncher` for ``compiled``, or ``None``.

    Returns ``None`` whenever the call cannot be reproduced exactly -- e.g.
    because a profiler hook is active, the kernel takes constexpr arguments
    that are not trailing, or the caller passes keyword arguments.
    """
    if compiled is None or _hooks_active():
        return None
    if kwargs.get("warmup"):
        return None

    jit = unwrap_jit(tuner.fn)
    n_runtime = 0
    for param in jit.params:
        if not param.is_constexpr:
            n_runtime += 1
    # Constexpr parameters must be trailing so that ``args`` maps one-to-one
    # onto the runtime parameters, in signature order.
    if len(args) != n_runtime:
        return None
    if any(p.is_constexpr for p in jit.params[:n_runtime]):
        return None

    grid = kwargs.get("grid")
    if grid is None:
        return None
    if callable(grid):
        # Reproduce the argument dictionary Triton builds before it resolves
        # the launch grid: positional args + keyword args + the config, then
        # any heuristic-computed meta parameters (BLOCK_SIZE, ...).
        arg_dict = {**dict(zip(tuner.arg_names, args)), **kwargs, **config_kwargs}
        heuristics = getattr(tuner.fn, "values", None) or {}
        for name, heuristic in heuristics.items():
            try:
                arg_dict[name] = heuristic(arg_dict)
            except Exception:
                return None
        try:
            grid = grid(arg_dict)
        except Exception:
            return None

    try:
        if stream is None:
            stream = execution_key()[1]
        # When the kernel has no tensor-descriptor arguments, the launcher's
        # per-pointer ``numel`` is never consumed, so data pointers may be
        # passed as raw ints.  That skips two Python method calls per tensor
        # argument inside the C launcher (~0.25us each on torch_xmlir).
        ptr_args = 0
        if not getattr(compiled.metadata, "tensor_args", None):
            for arg in args:
                if not isinstance(arg, torch.Tensor):
                    break
                ptr_args += 1
        return DirectLauncher(compiled, grid, stream, ptr_args=ptr_args)
    except Exception:
        return None
