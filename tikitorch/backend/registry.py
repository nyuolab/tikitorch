"""tikitorch.backend.registry — backend + op registries and the dispatch rule.

This is the internal machinery behind the :mod:`tikitorch.backend` facade. It
holds three module-global tables and the single fallback rule that makes the seam
"never silently break":

* ``_BACKENDS``  : ``name -> Backend``           (which accelerators are registered)
* ``_OPS``       : ``(op_name, device) -> impl`` (per-device op implementations)
* ``_WARNED``    : a dedup set for one-time fallback warnings

Design invariants:

* **Additive / same callable.** The CPU implementations are the *same callable
  objects* exported by :mod:`tikitorch.functional` (registered via
  :func:`_autoregister_cpu_ops`). ``dispatch('tropical_linear', 'cpu')`` returns
  ``functional.tropical_linear`` itself. Registry lookup still has normal Python
  overhead and does not make a claim about other implementations.
  Nothing here edits or wraps ``functional``.
* **Transparent fallback.** Requesting an op for a device that has no
  implementation (or whose backend is unavailable) returns the CPU implementation
  and emits a one-time :class:`BackendFallbackWarning`. The CPU reference semantics
  are then used; backend equivalence must be established separately.
"""
from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

from .device import Device, device

__all__ = [
    "Backend",
    "BackendCapabilities",
    "CpuBackend",
    "BackendFallbackWarning",
    "register_backend",
    "get_backend",
    "available_backends",
    "list_backends",
    "register_op",
    "dispatch",
    "is_op_supported",
    "list_ops",
]


class BackendFallbackWarning(UserWarning):
    """Emitted once per ``(subject, device)`` when execution silently falls back to CPU."""


# ---------------------------------------------------------------------------
# Capability declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do — branch on capability, not ``isinstance``.

    ``op_dispatch``       : routes individual tropical ops to device kernels.
    ``model_binding``     : can wrap a whole model (the Tier-1 fused-step path).
    ``on_device_training``: runs forward+backward+update on device.
    ``integer_only``      : operates in the integer-exact / multiplier-free domain.

    The bundled FPGA backend conservatively reports only model wrapping
    (``model_binding=True, op_dispatch=False``). It reports neither on-device
    training nor integer-only execution because no safe native runner ships.
    An external backend may declare broader capabilities only for the concrete
    implementation it supplies and validates.
    """

    op_dispatch: bool = False
    model_binding: bool = False
    on_device_training: bool = False
    integer_only: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# The single extension point
# ---------------------------------------------------------------------------


class Backend(abc.ABC):
    """Base class for a TikiTorch execution backend.

    Subclass this, set :attr:`name`, implement :meth:`is_available`, and call
    :meth:`register`. Override :meth:`ops` to supply per-op device kernels and
    :meth:`accelerate` to supply a model-level wrapper.
    """

    name: ClassVar[str] = "?"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Cheap, never-raising probe for whether this backend can run *now*."""

    def capabilities(self) -> BackendCapabilities:
        """Declare what this backend can do (default: nothing)."""
        return BackendCapabilities()

    def ops(self) -> Dict[str, Callable]:
        """Map ``op_name -> implementation`` for this backend (default: none)."""
        return {}

    def accelerate(self, model, **opts):
        """Return a (possibly wrapped) model bound to this backend (default: identity)."""
        return model

    def register(self, *, default: bool = False) -> "Backend":
        """Register this backend and all of its :meth:`ops` into the global tables."""
        register_backend(self, default=default)
        for op_name, impl in self.ops().items():
            register_op(op_name, self.name, impl)
        return self


# ---------------------------------------------------------------------------
# Module-global registries
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, Backend] = {}
_DEFAULT_BACKEND: Optional[str] = None
_OPS: Dict[Tuple[str, str], Callable] = {}
_WARNED: set = set()


def _coerce_device_type(name: Union[str, Device]) -> str:
    return name.type if isinstance(name, Device) else device(name).type


def _backend_up(backend: Optional[Backend]) -> bool:
    """Whether ``backend`` is registered and available, treating a raising
    ``is_available()`` as unavailable (the probe must never break dispatch)."""
    if backend is None:
        return False
    try:
        return backend.is_available()
    except Exception:  # noqa: BLE001 — a misbehaving probe degrades to "unavailable"
        return False


def register_backend(backend: Backend, *, default: bool = False) -> Backend:
    """Insert ``backend`` keyed by ``backend.name`` (idempotent). Returns it."""
    global _DEFAULT_BACKEND
    if not isinstance(backend, Backend):
        raise TypeError("backend must be an instance of Backend")
    name = _coerce_device_type(backend.name)
    _BACKENDS[name] = backend
    if default or _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = name
    return backend


def get_backend(name: Union[str, Device]) -> Optional[Backend]:
    """Look up a backend by name or :class:`Device`; ``None`` if well-formed but absent."""
    return _BACKENDS.get(_coerce_device_type(name))


def available_backends() -> List[str]:
    """Names of registered backends whose :meth:`is_available` is ``True``."""
    return [name for name, backend in _BACKENDS.items() if _backend_up(backend)]


def list_backends() -> List[str]:
    """Names of *all* registered backends (available or not)."""
    return list(_BACKENDS.keys())


def register_op(op_name: str, device_type: Union[str, Device], impl: Callable) -> None:
    """Register ``impl`` as the implementation of ``op_name`` on ``device_type``."""
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")
    if not callable(impl):
        raise TypeError("impl must be callable")
    _OPS[(op_name, _coerce_device_type(device_type))] = impl


def _warn_fallback(key: Tuple, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, BackendFallbackWarning, stacklevel=3)


def dispatch(op_name: str, device_type: Union[str, Device] = "cpu") -> Callable:
    """Return the implementation of ``op_name`` for ``device_type``.

    If the device has a registered, *available* implementation, it is returned.
    Otherwise this emits a one-time :class:`BackendFallbackWarning` and returns
    the CPU implementation. Raises :class:`KeyError` only if no CPU implementation
    exists (a genuine bug — the op name is unknown).
    """
    dev = _coerce_device_type(device_type)
    impl = _OPS.get((op_name, dev))

    if dev != "cpu":
        if impl is not None and _backend_up(_BACKENDS.get(dev)):
            return impl
        cpu_impl = _OPS.get((op_name, "cpu"))
        if cpu_impl is None:
            raise KeyError(f"no implementation registered for op {op_name!r} (cpu or {dev!r})")
        _warn_fallback(
            ("dispatch", op_name, dev),
            f"tikitorch op {op_name!r} has no available {dev!r} implementation; "
            f"falling back to the registered CPU reference. "
            f"This warning is shown once per (op, device).",
        )
        return cpu_impl

    if impl is None:
        raise KeyError(f"no CPU implementation registered for op {op_name!r}")
    return impl


def is_op_supported(op_name: str, device_type: Union[str, Device]) -> bool:
    """Pure query: is ``op_name`` natively supported on ``device_type``? (No warning.)"""
    dev = _coerce_device_type(device_type)
    if (op_name, dev) not in _OPS:
        return False
    if dev == "cpu":
        return True
    return _backend_up(_BACKENDS.get(dev))


def list_ops(device_type: Union[str, Device] = "cpu") -> List[str]:
    """All op names registered for ``device_type``."""
    dev = _coerce_device_type(device_type)
    return [op for (op, d) in _OPS if d == dev]


def _reset_fallback_warnings() -> None:
    """Test hook: clear the one-time fallback-warning dedup set."""
    _WARNED.clear()


# ---------------------------------------------------------------------------
# The always-present CPU backend
# ---------------------------------------------------------------------------


class CpuBackend(Backend):
    """The default backend: ordinary PyTorch/CPU execution of the tropical ops."""

    name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            op_dispatch=True,
            model_binding=False,
            on_device_training=True,
            integer_only=False,
            notes="Reference PyTorch execution; the always-available default.",
        )

    def accelerate(self, model, **opts):
        return model  # CPU is the native path; nothing to wrap


def _autoregister_cpu_ops() -> int:
    """Register the dispatchable callables in ``tikitorch.functional`` as CPU ops.

    Reads :mod:`tikitorch.functional` without modifying it and registers the *same*
    callable objects, so ``dispatch(name, 'cpu')`` is byte-identical to calling
    ``functional.<name>`` directly. The ``*Fn`` :class:`torch.autograd.Function`
    subclasses are skipped: they are callable but must be invoked via their
    ``.apply`` alias (which *is* registered, e.g. ``tropical_linear``) — registering
    the class itself would make ``dispatch(name)(...)`` *instantiate* it (deprecated,
    wrong result). Returns the number of ops registered.
    """
    import torch
    import tikitorch.functional as F  # local import: functional is loaded first

    count = 0
    for name in getattr(F, "__all__", []):
        obj = getattr(F, name, None)
        if not callable(obj):
            continue
        if isinstance(obj, type) and issubclass(obj, torch.autograd.Function):
            continue  # the .apply alias is the dispatchable op, not the class
        register_op(name, "cpu", obj)
        count += 1
    return count
