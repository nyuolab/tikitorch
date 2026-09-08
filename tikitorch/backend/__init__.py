"""tikitorch.backend — the pluggable execution-backend seam.

Several TikiTorch operators have compact compare/add-oriented implementations,
which makes them useful targets for non-GPU accelerator experiments. This package
is the seam those accelerators plug into: a small, PyTorch-flavored dispatch layer.

* :class:`Device` / :func:`device` — a logical device handle (``tt.device('fpga')``),
  spelled like ``torch.device`` but living one level up (tensor storage remains
  on its existing PyTorch device; see :mod:`tikitorch.backend.device`).
* :class:`Backend` / :class:`BackendCapabilities` — the single extension point. A
  new backend subclasses :class:`Backend`, declares its capabilities, and calls
  ``.register()``.
* :func:`register_op` / :func:`dispatch` — a per-device op registry with
  **transparent PyTorch fallback**: an op with no available logical-device
  implementation runs the reference on its input tensors' existing device
  (one-time :class:`BackendFallbackWarning`). Equivalence remains the responsibility
  of the registered implementation and its tests.
* :func:`to` — a logical mover, the ``.to(device)`` analogue.

The CPU backend is registered at import and **auto-binds every op in
:mod:`tikitorch.functional` as the default implementation** — the *same* callable
objects. Dispatch lookup itself still has ordinary Python call overhead.
The public tropical linear modules consult this seam. Omitting their logical
device uses :func:`get_default_device`; :func:`to` can tag a module tree for an
op-dispatch backend without moving its tensor storage.

Idiomatic usage::

    import tikitorch as tt
    print(tt.backend.available_backends())        # ['cpu']  (+ 'fpga' with a board)
    dev = tt.device('fpga')                        # logical device
    model = tt.to(model, dev)                      # -> CPU fallback wrapper, or on-board

Other backends (a torch-free C/Cython CPU runtime, WASM, integer NPUs) register
the same way — FPGA is simply the first consumer.
"""
from __future__ import annotations

from typing import Union

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
    default_device,
    dispatch,
    get_default_device,
    get_backend,
    is_reference_dispatch,
    is_op_supported,
    list_backends,
    list_ops,
    register_backend,
    register_op,
    set_default_device,
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
    "is_reference_dispatch",
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

def to(obj, device: Union[str, Device, None] = None):
    """Move a model (logically) to a TikiTorch device — the ``.to(device)`` analogue.

    * ``nn.Module`` + CPU device -> the same module, tagged for reference dispatch.
    * ``nn.Module`` + a backend that *can* bind models (``capabilities().model_binding``,
      e.g. ``'fpga'``) -> ``backend.accelerate(obj)``. This holds regardless of board
      availability: the returned wrapper itself runs on device when bound and does the
      transparent CPU fallback (with a one-time warning at ``train_step`` time) when not.
    * ``nn.Module`` + an op-dispatch backend -> the same module tree, tagged so its
      public tropical layers resolve operations from that backend.
    * ``nn.Module`` + an unknown device -> the tagged module unchanged, plus a
      one-time :class:`BackendFallbackWarning`; operations use their PyTorch reference.
    * ``torch.Tensor`` + a non-CPU logical device -> the tensor unchanged, including
      its physical PyTorch device, plus a one-time warning.

    Unsupported logical devices use the PyTorch reference on the tensor's existing
    physical device, with a warning. Errors raised by the selected implementation
    itself are not swallowed.
    """
    dev = _as_device(device) if device is not None else get_default_device()

    backend = get_backend(dev)

    import torch

    def tag_module_tree(module) -> None:
        for child in module.modules():
            child._tt_device = dev

    if dev.is_cpu:
        if isinstance(obj, torch.nn.Module):
            tag_module_tree(obj)
        return obj

    # torch.Tensor: logical devices never move storage.
    if isinstance(obj, torch.Tensor):
        _warn_fallback(
            ("to-tensor", dev.type),
            f"tikitorch.to(): tensors are not moved to logical device {dev!s} "
            f"(storage remains on {obj.device}; the logical device selects a backend). "
            f"Returning the tensor unchanged.",
        )
        return obj

    # nn.Module: defer to the backend's model-binding hook when available.
    if backend is not None and backend.capabilities().model_binding:
        return backend.accelerate(obj, device=dev)

    if isinstance(obj, torch.nn.Module) and backend is not None \
            and backend.capabilities().op_dispatch:
        tag_module_tree(obj)
        return obj

    _warn_fallback(
        ("to-module", dev.type),
        f"tikitorch.to(): no model-binding backend available for device {dev!s}; "
        f"returning the model unchanged with PyTorch reference fallback.",
    )
    if isinstance(obj, torch.nn.Module):
        tag_module_tree(obj)
    elif hasattr(obj, "__dict__"):
        try:
            obj._tt_device = dev  # advisory tag; harmless if unused
        except Exception:  # noqa: BLE001
            pass
    return obj
