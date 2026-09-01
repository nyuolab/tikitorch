"""tikitorch.fpga.backend — fail-closed whole-model native binding.

The backend registers the ``'fpga'`` wrapper seam but no per-op implementation.
It can wrap a model and fall back to the model's PyTorch path; native execution
additionally requires a usable endpoint, a full structural kernel match, and an
application-supplied runner. No safe runner ships by default.

``is_available()`` reports native endpoint usability, not physical-board
presence. In particular, a LOCAL host-reference endpoint may return ``True``.
The conservative capability declaration describes built-in behavior: it does
not claim on-device training or integer-only execution merely because catalog
artifacts or externally registered runners exist.
"""
from __future__ import annotations

from typing import Optional, Union

from ..backend.device import Device, device as make_device
from ..backend.registry import Backend, BackendCapabilities
from .kernels import KernelCache


class FpgaBackend(Backend):
    """Whole-model wrapper with fail-closed, explicitly supplied native runners."""

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
            op_dispatch=False,
            model_binding=True,
            on_device_training=False,
            integer_only=False,
            notes=(
                "Fail-closed model wrapper and PyTorch fallback; native fused-step "
                "execution requires an explicitly registered, build-specific runner."
            ),
        )

    def ops(self) -> dict:
        return {}  # Tier-2 op kernels slot in here without an API change

    def accelerate(self, model, *, device: Union[str, Device, None] = None,
                   optimizer=None, strict: bool = False):
        """Wrap ``model`` in an :class:`~tikitorch.fpga.accelerate.AcceleratedModel`.

        Native binding requires a usable endpoint, a full model-observable
        structural match from :class:`KernelCache`, and an explicitly registered
        kernel-specific runner. Otherwise the wrapper uses its PyTorch fallback,
        or raises :class:`NativeUnavailable` when ``strict=True``. Strict mode does
        not replace the runtime and build-manifest validation owned by a custom
        runner. Re-wrapping an ``AcceleratedModel`` never nests wrappers.
        """
        from .accelerate import AcceleratedModel  # local import: avoid cycle

        if isinstance(model, AcceleratedModel):
            model = model.module  # idempotent: never nest wrappers

        dev = device if isinstance(device, Device) else make_device(device or "fpga")
        if dev.type != self.name:
            raise ValueError(
                f"FPGA acceleration requires logical device type 'fpga', got {dev!s}"
            )
        if dev.index not in (None, 0):
            raise ValueError(
                f"unsupported FPGA endpoint {dev!s}; only 'fpga' and 'fpga:0' are available"
            )
        spec = self._cache.match(model)
        endpoint_usable = self.is_available()
        native = self._native if endpoint_usable else None
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
        """Number of usable native endpoints; LOCAL may contribute one."""
        return 1 if self.is_available() else 0
