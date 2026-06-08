"""tikitorch.fpga.backend — the FpgaBackend (Tier-1 model binding).

Registers ``'fpga'`` into the :mod:`tikitorch.backend` seam. Its capability set is
the honest Tier-1 shape: ``model_binding=True`` (whole fused training-step), and
``op_dispatch=False`` (per-op kernels are Tier-2, F-gated — the :meth:`ops` table
is intentionally empty so every tropical op resolves to the CPU fallback today).

``is_available()`` is ``True`` only when a native driver loaded *and* a board is
present — ``False`` in this repo/CI. The whole-model offload contract is honored
through :meth:`accelerate`, which returns a DDP-style wrapper (bound on board,
transparent CPU fallback otherwise).
"""
from __future__ import annotations

from typing import Optional, Union

from ..backend.device import Device, device as make_device
from ..backend.registry import Backend, BackendCapabilities
from .kernels import KernelCache


class FpgaBackend(Backend):
    """TikiTorch FPGA backend: Tier-1 whole-model fused-step binding over the seam."""

    name = "fpga"

    def __init__(self, native=None, cache: Optional[KernelCache] = None) -> None:
        self._native = native
        self._cache = cache if cache is not None else KernelCache.default()

    # -- Backend interface --------------------------------------------------

    def is_available(self) -> bool:
        if self._native is None:
            return False
        try:
            return self._native.board_present()
        except Exception:  # noqa: BLE001 — availability probes never raise
            return False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            op_dispatch=False,        # Tier-2 (custom codegen) flips this on
            model_binding=True,       # Tier-1: bind a whole model to a fused kernel
            on_device_training=True,  # fwd + bwd + sign-SGD all on board
            integer_only=True,        # INT16 weights / INT32 grads, DSP=0
            notes="Tier-1 fused whole-training-step binding; per-op dispatch is Tier-2.",
        )

    def ops(self) -> dict:
        return {}  # Tier-2 op kernels slot in here without an API change

    def accelerate(self, model, *, device: Union[str, Device, None] = None,
                   optimizer=None, strict: bool = False):
        """Wrap ``model`` in an :class:`~tikitorch.fpga.accelerate.AcceleratedModel`.

        Binds to a compiled kernel iff a board is available, the model's shape
        matches an entry in the :class:`KernelCache`, *and* an on-board runner is
        registered for that kernel (:func:`~tikitorch.fpga.register_board_runner`).
        Otherwise returns a transparent CPU-fallback wrapper (or raises
        :class:`NativeUnavailable` when ``strict=True``). Idempotent: re-wrapping an
        ``AcceleratedModel`` rebinds its inner module rather than nesting wrappers.
        """
        from .accelerate import AcceleratedModel  # local import: avoid cycle

        if isinstance(model, AcceleratedModel):
            model = model.module  # idempotent: never nest wrappers

        dev = device if isinstance(device, Device) else make_device(device or "fpga")
        spec = self._cache.match(model)
        on_board = self.is_available()
        native = self._native if on_board else None
        return AcceleratedModel._build(
            model, spec=spec, native=native, device=dev, optimizer=optimizer, strict=strict,
        )

    # -- Conveniences -------------------------------------------------------

    @property
    def cache(self) -> KernelCache:
        return self._cache

    @property
    def native(self):
        return self._native

    def device_count(self) -> int:
        """Number of usable boards (0 when unavailable)."""
        return 1 if self.is_available() else 0
