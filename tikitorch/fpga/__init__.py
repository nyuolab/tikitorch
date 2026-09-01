"""tikitorch.fpga — an experimental FPGA/native integration for TikiTorch.

Importing this module loads an optional native library and registers an
``'fpga'`` backend into :mod:`tikitorch.backend`. A loaded library may expose a
LOCAL host-reference endpoint or a compatible physical-FPGA endpoint; loading
the library alone does not open or probe either one.

The bundled :class:`KernelCache` serves two distinct purposes. ``lookup()``
inspects a catalog record by its compact key and is not a compatibility check.
``match(model)`` additionally validates the full model-observable structural
recipe represented by that record. Stock model constructors currently fail this
conservative match, and no safe bundled runner is registered automatically.

``accelerate(model)`` therefore returns a PyTorch-fallback wrapper unless all of
the following are present: a usable native endpoint, a structural match, and an
explicitly registered kernel-specific runner. A custom runner must still check
runtime tensor shapes, dtypes and ranges; named buffer marshalling; quantization;
native symbol/ABI identity; and a configuration-matched build manifest. The
``bound`` flag means that the wrapper is eligible to invoke that runner. It is
not proof of those runner-owned checks or of physical FPGA execution; LOCAL is a
host endpoint.

The bundled kernels are fused training-step monoliths, not per-op
``.to('fpga')`` implementations. See ``examples/fpga_quickstart.py`` for safe
catalog and fallback inspection, and ``BUILD_NATIVE.md`` before registering a
custom runner.

This package never ``import forge`` — the HLS sources it references are the copies
bundled under :mod:`tikitorch.hardware`.
"""
from __future__ import annotations

from typing import Optional, Union

from ..backend.device import Device
from . import _native
from ._native import NativeHandle, NativeUnavailable
from ._session import register_board_runner
from .accelerate import AcceleratedModel
from .backend import FpgaBackend
from .runners import register_reference_runners
from .kernels import (
    KernelABI,
    KernelCache,
    KernelSpec,
    LMKernelConfig,
    MorphKernelConfig,
    MuZeroKernelConfig,
)
from .dyadic_export import export_net, verify_schedule, export_layer, reconstruct_weight

__all__ = [
    "FpgaBackend",
    "AcceleratedModel",
    "accelerate",
    "is_available",
    "device_count",
    "get_backend",
    "LMKernelConfig",
    "MorphKernelConfig",
    "MuZeroKernelConfig",
    "KernelSpec",
    "KernelABI",
    "KernelCache",
    "NativeHandle",
    "NativeUnavailable",
    "register_board_runner",
    "register_reference_runners",
    # dyadic weight -> software shift-schedule export
    "export_net",
    "verify_schedule",
    "export_layer",
    "reconstruct_weight",
]

# Load the optional native driver (``None`` in a normal wheel/CI), build the kernel
# allow-list, and register the singleton backend. ``_native.load`` owns the narrow
# missing-library/ABI fallback; programming errors in registration fail loudly.
_native_handle: Optional[NativeHandle] = _native.load()
_backend: Optional[FpgaBackend] = FpgaBackend(
    native=_native_handle, cache=KernelCache.default())
_backend.register()
# No runner is registered here. The historical one-layer flat-parameter runner
# was unsafe; ``register_reference_runners()`` remains only as a fail-loud
# compatibility hook. Applications must explicitly register a validated custom
# runner for a specific kernel/build contract.


def get_backend() -> Optional[FpgaBackend]:
    """Return the registered :class:`FpgaBackend` singleton."""
    return _backend


def is_available() -> bool:
    """Whether a native endpoint is usable (a LOCAL host endpoint may be usable)."""
    return _backend is not None and _backend.is_available()


def device_count() -> int:
    """Number of usable native endpoints; LOCAL may report one host endpoint."""
    return _backend.device_count() if _backend is not None else 0


def accelerate(model, *, device: Union[str, Device] = "fpga", optimizer=None,
               strict: bool = False) -> AcceleratedModel:
    """Wrap ``model`` for native-kernel execution (PyTorch fallback otherwise).

    See :class:`~tikitorch.fpga.accelerate.AcceleratedModel`. With ``strict=True``,
    raises :class:`NativeUnavailable` unless an endpoint, full structural match,
    and explicitly registered runner are all present. Strict mode disables the
    fallback; it cannot validate a custom runner's runtime inputs or build
    manifest. A bound LOCAL bridge executes on the host.
    """
    if isinstance(model, AcceleratedModel):
        model = model.module  # idempotent: never nest wrappers
    if _backend is None:  # extremely defensive: seam failed to register
        return AcceleratedModel._build(
            model, spec=None, native=None, device=Device("fpga"),
            optimizer=optimizer, strict=strict,
        )
    return _backend.accelerate(model, device=device, optimizer=optimizer, strict=strict)
