"""tikitorch.fpga — the FPGA backend for TikiTorch.

Importing this module detects a board (via a native driver) and registers an
``'fpga'`` backend into :mod:`tikitorch.backend`. The headline entry point is the
DDP-style wrapper::

    import tikitorch as tt
    import tikitorch.fpga as fpga

    model = tt.models.TropicalTransformer(d_model=128, n_layers=2,
                                          task_type="lm", vocab_size=128)
    model = fpga.accelerate(model)        # bound on a board (see below); else CPU fallback
    print(fpga.is_available(), model.bound)

    # keep your ordinary tropical training loop unchanged:
    loss = model.compute_loss(x, y); loss.backward(); opt.step()

**Honest scope (Tier 1).** The bundled silicon kernels are *fused whole-training-
step monoliths* (no multi-kernel mux), so ``accelerate`` binds a whole *model* to
a compiled kernel — it is not per-op ``.to('fpga')`` dispatch (that is Tier 2,
F-gated). A model is *bound*
(runs on silicon) only when **all three** hold: a board is present, its shape
matches a compiled kernel, *and* an on-board runner is registered for that kernel
(:func:`register_board_runner`) — the hardware-validated last mile, of which
**none ship by default** (see ``BUILD_NATIVE.md``). On a machine with no board, no
matching kernel, or no runner (this repo, CI) everything transparently falls back
to CPU and stays correct.

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
)

__all__ = [
    "FpgaBackend",
    "AcceleratedModel",
    "accelerate",
    "is_available",
    "device_count",
    "get_backend",
    "LMKernelConfig",
    "MorphKernelConfig",
    "KernelSpec",
    "KernelABI",
    "KernelCache",
    "NativeHandle",
    "NativeUnavailable",
    "register_board_runner",
    "register_reference_runners",
]

# Load the native driver (None in this repo / CI), build the kernel allow-list,
# and register the singleton backend into the seam. Guarded so a half-built native
# layer can never make `import tikitorch.fpga` (or `import tikitorch`) fail. The
# singleton is module-private (``_backend``) so it does not shadow the
# ``tikitorch.fpga.backend`` submodule attribute; reach it via :func:`get_backend`.
_native_handle: Optional[NativeHandle] = None
_backend: Optional[FpgaBackend] = None
try:
    _native_handle = _native.load()
    _backend = FpgaBackend(native=_native_handle, cache=KernelCache.default())
    _backend.register()
    # Install the bundled reference runner(s). Harmless without a board: a model
    # only ever *binds* when a native handle is also present (none here / in CI),
    # so auto-registering just makes `.fpga()` work out-of-the-box once
    # `_libtikifpga` is built (LOCAL or XRT) — no extra call needed.
    register_reference_runners()
except Exception:  # noqa: BLE001 — never break import on backend bring-up
    _backend = None


def get_backend() -> Optional[FpgaBackend]:
    """The registered :class:`FpgaBackend` singleton (``None`` if registration failed)."""
    return _backend


def is_available() -> bool:
    """``True`` iff a native driver loaded and a board is present (``False`` here)."""
    return _backend is not None and _backend.is_available()


def device_count() -> int:
    """Number of usable FPGA boards (0 when unavailable)."""
    return _backend.device_count() if _backend is not None else 0


def accelerate(model, *, device: Union[str, Device] = "fpga", optimizer=None,
               strict: bool = False) -> AcceleratedModel:
    """Wrap ``model`` for FPGA execution (bound on a board, CPU fallback otherwise).

    See :class:`~tikitorch.fpga.accelerate.AcceleratedModel`. With ``strict=True``,
    raises :class:`NativeUnavailable` instead of falling back when the model cannot
    run on a board.
    """
    if isinstance(model, AcceleratedModel):
        model = model.module  # idempotent: never nest wrappers
    if _backend is None:  # extremely defensive: seam failed to register
        return AcceleratedModel._build(
            model, spec=None, native=None, device=Device("fpga"),
            optimizer=optimizer, strict=strict,
        )
    return _backend.accelerate(model, device=device, optimizer=optimizer, strict=strict)
