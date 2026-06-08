"""tikitorch.backend — the pluggable, FPGA-first execution-backend seam.

TikiTorch's whole workload collapses to one tropical primitive, which makes it an
unusually clean target for non-GPU accelerators. This package is the *seam* those
accelerators plug into — a small, PyTorch-flavored dispatch layer:

* :class:`Device` / :func:`device` — a logical device handle (``tt.device('fpga')``),
  spelled like ``torch.device`` but living one level up (tropical tensors stay
  ordinary CPU ``torch.Tensor`` s; see :mod:`tikitorch.backend.device`).
* :class:`Backend` / :class:`BackendCapabilities` — the single extension point. A
  new backend subclasses :class:`Backend`, declares its capabilities, and calls
  ``.register()``.
* :func:`register_op` / :func:`dispatch` — a per-device op registry with
  **transparent CPU fallback**: an op with no available device implementation
  silently runs the CPU reference (one-time :class:`BackendFallbackWarning`).
  Correctness never breaks; performance degrades gracefully.
* :func:`to` — a logical mover, the ``.to(device)`` analogue.

The CPU backend is registered at import and **auto-binds every op in
:mod:`tikitorch.functional` as the default implementation** — the *same* callable
objects, so the existing CPU path is byte-identical and carries zero overhead.
The seam is consulted only when a non-CPU device is explicitly requested.

Idiomatic usage::

    import tikitorch as tt
    print(tt.backend.available_backends())        # ['cpu']  (+ 'fpga' with a board)
    dev = tt.device('fpga')                        # logical device
    model = tt.to(model, dev)                      # -> CPU fallback wrapper, or on-board

Other backends (a torch-free C/Cython CPU runtime, WASM, integer NPUs) register
the same way — FPGA is simply the first consumer.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union

from .device import Device, FpgaAccelerable, device
from .registry import (
    Backend,
    BackendCapabilities,
    BackendFallbackWarning,
    CpuBackend,
    _autoregister_cpu_ops,
    _reset_fallback_warnings as _reset_fallback_warnings,  # re-exported for tests
    _warn_fallback,
    available_backends,
    dispatch,
    get_backend,
    is_op_supported,
    list_backends,
    list_ops,
    register_backend,
    register_op,
)

__all__ = [
    "Device",
    "device",
    "FpgaAccelerable",
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
    "to",
    "set_default_device",
    "default_device",
    "get_default_device",
]

# A private handle to the device() factory so the `to(obj, device)` parameter name
# can shadow it without losing the constructor.
_as_device = device

# Register the always-present CPU backend and auto-bind the functional ops to it.
CpuBackend().register(default=True)
_autoregister_cpu_ops()

# Process-wide default logical device (consulted when a device is omitted).
_DEFAULT_DEVICE: Device = Device("cpu")


def get_default_device() -> Device:
    """Return the current process-wide default logical device."""
    return _DEFAULT_DEVICE


def set_default_device(dev: Union[str, Device]) -> None:
    """Set the process-wide default logical device (mirrors ``torch.set_default_device``)."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = _as_device(dev)


@contextmanager
def default_device(dev: Union[str, Device]) -> Iterator[Device]:
    """Scope the default logical device for the duration of a ``with`` block."""
    global _DEFAULT_DEVICE
    previous = _DEFAULT_DEVICE
    _DEFAULT_DEVICE = _as_device(dev)
    try:
        yield _DEFAULT_DEVICE
    finally:
        _DEFAULT_DEVICE = previous


def to(obj, device: Union[str, Device, None] = None):
    """Move a model (logically) to a TikiTorch device — the ``.to(device)`` analogue.

    * ``nn.Module`` + CPU device  -> the module unchanged (zero overhead).
    * ``nn.Module`` + a backend that *can* bind models (``capabilities().model_binding``,
      e.g. ``'fpga'``) -> ``backend.accelerate(obj)``. This holds regardless of board
      availability: the returned wrapper itself runs on device when bound and does the
      transparent CPU fallback (with a one-time warning at ``train_step`` time) when not.
    * ``nn.Module`` + a non-binding or unknown device -> the module unchanged, plus a
      one-time :class:`BackendFallbackWarning`.
    * ``torch.Tensor`` + a non-CPU device -> the tensor unchanged (tropical tensors
      stay on torch CPU; the logical device only selects a backend), plus a
      one-time warning.

    Never raises for an unsupported device — the whole point of the seam is that an
    unsupported target degrades to CPU rather than erroring.
    """
    dev = _as_device(device) if device is not None else _DEFAULT_DEVICE

    if dev.is_cpu:
        return obj

    backend = get_backend(dev)

    # torch.Tensor: logical devices never move storage.
    import torch

    if isinstance(obj, torch.Tensor):
        _warn_fallback(
            ("to-tensor", dev.type),
            f"tikitorch.to(): tensors are not moved to logical device {dev!s} "
            f"(tropical tensors stay on torch CPU; the device selects a backend). "
            f"Returning the tensor unchanged.",
        )
        return obj

    # nn.Module: defer to the backend's model-binding hook when available.
    if backend is not None and backend.capabilities().model_binding:
        return backend.accelerate(obj, device=dev)

    _warn_fallback(
        ("to-module", dev.type),
        f"tikitorch.to(): no model-binding backend available for device {dev!s}; "
        f"returning the model unchanged (it will run on CPU).",
    )
    if hasattr(obj, "__dict__"):
        try:
            obj._tt_device = dev  # advisory tag; harmless if unused
        except Exception:  # noqa: BLE001
            pass
    return obj
