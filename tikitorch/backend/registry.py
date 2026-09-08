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
* **Transparent fallback.** Requesting an op for a logical device that has no
  implementation (or whose backend is unavailable) returns the registered PyTorch
  reference and emits a one-time :class:`BackendFallbackWarning`. This does not move
  tensor storage: CUDA inputs remain CUDA inputs. Backend equivalence must be
  established separately.
"""
from __future__ import annotations

import abc
from contextlib import contextmanager
from threading import RLock
import warnings
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Iterator, List, Mapping, Optional, Tuple, Union

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
    "is_reference_dispatch",
    "is_op_supported",
    "list_ops",
    "set_default_device",
    "get_default_device",
    "default_device",
]


class BackendFallbackWarning(UserWarning):
    """Emitted when a logical-device operation uses the PyTorch reference."""


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
        """Atomically register this backend and the exact result of :meth:`ops`."""
        _install_backend(self, self.ops(), default=default, replace_ops=True)
        return self


# ---------------------------------------------------------------------------
# Module-global registries
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, Backend] = {}
_DEFAULT_BACKEND: Optional[str] = None
_OPS: Dict[Tuple[str, str], Callable] = {}
_CANONICAL_REFERENCE_IMPLS: Dict[str, Callable] = {}
_REFERENCE_OPS = frozenset()
_REGISTRY_SNAPSHOT = (_BACKENDS, _OPS, _REFERENCE_OPS)
_WARNED: set = set()
_DEFAULT_DEVICE: Device = Device("cpu")
_REGISTRY_LOCK = RLock()
_REGISTRY_GENERATION = 0


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


def _backend_op_dispatch_up(backend: Optional[Backend]) -> bool:
    if not _backend_up(backend):
        return False
    try:
        return bool(backend.capabilities().op_dispatch)
    except Exception:  # noqa: BLE001 — invalid capability probes are unavailable
        return False


def _validated_ops(implementations: Mapping[str, Callable]) -> Dict[str, Callable]:
    if not isinstance(implementations, Mapping):
        raise TypeError("Backend.ops() must return a mapping of op names to callables")
    validated: Dict[str, Callable] = {}
    for op_name, impl in implementations.items():
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValueError("backend op names must be non-empty strings")
        if not callable(impl):
            raise TypeError(f"implementation for op {op_name!r} must be callable")
        validated[op_name] = impl
    return validated


def _install_backend(
    backend: Backend,
    implementations: Mapping[str, Callable],
    *,
    default: bool,
    replace_ops: bool,
) -> Backend:
    """Install validated backend state without exposing a partial replacement."""

    global _BACKENDS, _DEFAULT_BACKEND, _DEFAULT_DEVICE, _OPS, _REFERENCE_OPS
    global _REGISTRY_GENERATION, _REGISTRY_SNAPSHOT
    if not isinstance(backend, Backend):
        raise TypeError("backend must be an instance of Backend")
    name = _coerce_device_type(backend.name)
    validated = _validated_ops(implementations)
    if validated:
        try:
            op_dispatch = bool(backend.capabilities().op_dispatch)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("backend capabilities() failed during registration") from exc
        if not op_dispatch:
            raise ValueError(
                f"backend {name!r} supplies ops but capabilities().op_dispatch is False"
            )
    with _REGISTRY_LOCK:
        current_backends, current_ops, current_references = _REGISTRY_SNAPSHOT
        backends = dict(current_backends)
        ops = dict(current_ops)
        references = set(current_references)
        previous = backends.get(name)
        if replace_ops or (previous is not None and previous is not backend):
            stale = [key for key in ops if key[1] == name]
            for key in stale:
                del ops[key]
                references.discard(key)
        backends[name] = backend
        ops.update({(op_name, name): impl for op_name, impl in validated.items()})
        # Publish immutable-by-convention snapshots in one assignment each. Public
        # execution reads these without acquiring a lock, which keeps torch.compile
        # fullgraph-compatible; Dynamo guards the global snapshot identities.
        reference_snapshot = frozenset(references)
        _REGISTRY_SNAPSHOT = (backends, ops, reference_snapshot)
        _BACKENDS, _OPS = backends, ops
        _REFERENCE_OPS = reference_snapshot
        _REGISTRY_GENERATION += 1
        if default or _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = name
        if default:
            _DEFAULT_DEVICE = Device(name)
    return backend


def register_backend(backend: Backend, *, default: bool = False) -> Backend:
    """Register a backend object, clearing stale ops when replacing another object.

    Prefer :meth:`Backend.register` when the backend supplies operations; it
    validates and installs the backend and its exact operation set together.
    Re-registering the same object through this low-level function is idempotent.
    """

    return _install_backend(backend, {}, default=default, replace_ops=False)


def get_backend(name: Union[str, Device]) -> Optional[Backend]:
    """Look up a backend by name or :class:`Device`; ``None`` if well-formed but absent."""
    backends, _, _ = _REGISTRY_SNAPSHOT
    return backends.get(_coerce_device_type(name))


def available_backends() -> List[str]:
    """Names of registered backends whose :meth:`is_available` is ``True``."""
    backends, _, _ = _REGISTRY_SNAPSHOT
    registered = list(backends.items())
    return [name for name, backend in registered if _backend_up(backend)]


def list_backends() -> List[str]:
    """Names of *all* registered backends (available or not)."""
    backends, _, _ = _REGISTRY_SNAPSHOT
    return list(backends.keys())


def register_op(
    op_name: str,
    device_type: Union[str, Device],
    impl: Callable,
    *,
    _reference: bool = False,
) -> None:
    """Register ``impl`` as the implementation of ``op_name`` on ``device_type``."""
    if not isinstance(op_name, str) or not op_name.strip():
        raise ValueError("op_name must be a non-empty string")
    if not callable(impl):
        raise TypeError("impl must be callable")
    global _OPS, _REFERENCE_OPS, _REGISTRY_GENERATION, _REGISTRY_SNAPSHOT
    key = (op_name, _coerce_device_type(device_type))
    with _REGISTRY_LOCK:
        backends, current_ops, current_references = _REGISTRY_SNAPSHOT
        backend = backends.get(key[1])
        if backend is not None and not backend.capabilities().op_dispatch:
            raise ValueError(
                f"backend {key[1]!r} does not declare op_dispatch capability"
            )
        ops = dict(current_ops)
        references = set(current_references)
        ops[key] = impl
        canonical_reference = _CANONICAL_REFERENCE_IMPLS.get(op_name)
        if _reference or impl is canonical_reference:
            references.add(key)
        else:
            references.discard(key)
        reference_snapshot = frozenset(references)
        _REGISTRY_SNAPSHOT = (backends, ops, reference_snapshot)
        _OPS = ops
        _REFERENCE_OPS = reference_snapshot
        _REGISTRY_GENERATION += 1


def _warn_fallback(key: Tuple, message: str) -> None:
    with _REGISTRY_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    warnings.warn(message, BackendFallbackWarning, stacklevel=3)


def dispatch(op_name: str, device_type: Union[str, Device, None] = None) -> Callable:
    """Return the implementation of ``op_name`` for ``device_type``.

    Omitting ``device_type`` consults :func:`get_default_device`.
    If the device has a registered, *available* implementation, it is returned.
    Otherwise this emits a one-time :class:`BackendFallbackWarning` and returns
    the registered PyTorch reference. The fallback executes wherever its tensor
    inputs already live; dispatch never copies a CUDA tensor to CPU. Raises
    :class:`KeyError` only if no reference implementation exists.
    """
    dev = get_default_device().type if device_type is None else _coerce_device_type(device_type)
    # Deliberately read the published snapshots without a lock. Apart from making
    # dispatch cheap, this lets torch.compile guard the registry/default globals
    # instead of graph-breaking on a lock context manager.
    backends, ops, _ = _REGISTRY_SNAPSHOT
    impl = ops.get((op_name, dev))
    backend = backends.get(dev)
    reference_impl = ops.get((op_name, "cpu"))

    if dev != "cpu":
        if impl is not None and _backend_op_dispatch_up(backend):
            return impl
        if reference_impl is None:
            raise KeyError(f"no implementation registered for op {op_name!r} (cpu or {dev!r})")
        import torch
        if torch._dynamo.is_compiling():
            raise RuntimeError(
                f"cannot compile an implicit fallback for explicitly selected logical "
                f"device {dev!r}; make the backend available or select the PyTorch "
                "reference device before calling torch.compile"
            )
        _warn_fallback(
            ("dispatch", op_name, dev),
            f"tikitorch op {op_name!r} has no available {dev!r} implementation; "
            f"using the registered PyTorch reference on the inputs' existing device. "
            f"This warning is shown once per (op, device).",
        )
        return reference_impl

    if impl is None:
        raise KeyError(f"no PyTorch reference implementation registered for op {op_name!r}")
    return impl


def is_reference_dispatch(
    op_name: str,
    device_type: Union[str, Device, None] = None,
) -> bool:
    """Whether dispatch resolves to the canonical PyTorch reference callable.

    This metadata query avoids comparing callable identities inside compiled
    module code. Registry snapshot replacement gives Dynamo a normal global
    guard, so changing a backend invalidates the compiled specialization.
    """

    dev = get_default_device().type if device_type is None else _coerce_device_type(device_type)
    backends, ops, references = _REGISTRY_SNAPSHOT
    key = (op_name, dev)
    if dev == "cpu":
        return key in references
    if key in ops and _backend_op_dispatch_up(backends.get(dev)):
        return key in references
    return (op_name, "cpu") in references


def is_op_supported(op_name: str, device_type: Union[str, Device]) -> bool:
    """Pure query: is ``op_name`` natively supported on ``device_type``? (No warning.)"""
    dev = _coerce_device_type(device_type)
    backends, ops, _ = _REGISTRY_SNAPSHOT
    registered = (op_name, dev) in ops
    backend = backends.get(dev)
    if not registered:
        return False
    if dev == "cpu":
        return True
    return _backend_op_dispatch_up(backend)


def list_ops(device_type: Union[str, Device] = "cpu") -> List[str]:
    """All op names registered for ``device_type``."""
    dev = _coerce_device_type(device_type)
    _, ops, _ = _REGISTRY_SNAPSHOT
    return [op for (op, d) in ops if d == dev]


def _reset_fallback_warnings() -> None:
    """Test hook: clear the one-time fallback-warning dedup set."""
    with _REGISTRY_LOCK:
        _WARNED.clear()


def get_default_device() -> Device:
    """Return the logical device used when :func:`dispatch` omits one."""

    return _DEFAULT_DEVICE


def set_default_device(dev: Union[str, Device]) -> None:
    """Set the process-wide default logical dispatch device."""

    global _DEFAULT_BACKEND, _DEFAULT_DEVICE, _REGISTRY_GENERATION
    selected = device(dev)
    with _REGISTRY_LOCK:
        _DEFAULT_DEVICE = selected
        _DEFAULT_BACKEND = selected.type
        _REGISTRY_GENERATION += 1


@contextmanager
def default_device(dev: Union[str, Device]) -> Iterator[Device]:
    """Temporarily select the logical device used by implicit dispatch."""

    previous = get_default_device()
    set_default_device(dev)
    try:
        yield get_default_device()
    finally:
        set_default_device(previous)


# ---------------------------------------------------------------------------
# The always-present CPU backend
# ---------------------------------------------------------------------------


class CpuBackend(Backend):
    """The default backend: ordinary PyTorch execution of the tropical ops."""

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
        return model  # The PyTorch reference needs no logical-device wrapper.


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
        _CANONICAL_REFERENCE_IMPLS[name] = obj
        register_op(name, "cpu", obj, _reference=True)
        count += 1
    return count
