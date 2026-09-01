"""tikitorch.backend.device — the logical TikiTorch device handle.

:class:`Device` mirrors ``torch.device`` in *spelling* (``device('fpga:0')``) but
is a **logical TikiTorch concept, not a ``torch.Tensor`` device**. PyTorch only
accepts a fixed set of tensor device types (cpu, cuda, mps, ...) and rejects
custom ones, and tropical tensors stay ordinary CPU ``torch.Tensor`` s. A
:class:`Device` therefore only selects which TikiTorch *backend* an op or a model
dispatches to — carrying one never moves a tensor's storage. This is the honest
resolution of "torch cannot represent an ``fpga`` tensor": the device lives one
level up, at the tikitorch dispatch layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Device:
    """A logical TikiTorch device: a ``(type, index)`` pair.

    Frozen and hashable so it can key the backend / op registries. Value-based
    equality means ``Device('fpga', 0) == Device('fpga', 0)``. Use the
    :func:`device` factory for string parsing (``device('fpga:0')``).
    """

    type: str = "cpu"
    index: Optional[int] = None

    @property
    def is_cpu(self) -> bool:
        """``True`` for the canonical CPU device (the always-present default)."""
        return self.type == "cpu"

    def with_index(self, index: Optional[int]) -> "Device":
        """Return a copy of this device with a different ordinal."""
        return Device(self.type, index)

    def __repr__(self) -> str:  # echoes torch.device's repr
        if self.index is None:
            return f"device(type={self.type!r})"
        return f"device(type={self.type!r}, index={self.index})"

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


class FpgaAccelerable:
    """Mixin giving a tropical model a torch-idiomatic ``.fpga()`` mover.

    ``model = model.fpga()`` is the one-liner that mirrors ``model.cuda()``: it
    returns a DDP-style :class:`~tikitorch.fpga.accelerate.AcceleratedModel` that
    invokes a native runner only when one is *bound* (usable endpoint + full
    structural kernel match + explicitly registered runner) and uses the
    ordinary PyTorch path otherwise. A bound LOCAL endpoint still executes on
    the host, and the custom runner owns runtime and build-identity checks. The
    inverse (unwrap to the plain model) is ``accelerated.cpu()`` /
    ``accelerated.to(tt.device('cpu'))`` on the returned wrapper.

    The mixin is intentionally dependency-free and imports :mod:`tikitorch.fpga`
    *lazily* inside the method, so mixing it into a model never pulls the FPGA
    backend (or a native driver) in at class-definition time. It does **not**
    override ``nn.Module.cpu()`` (which moves parameters to the CPU torch device).
    """

    def fpga(self, *, optimizer=None, strict: bool = False):
        """Wrap ``self`` for FPGA execution (``tikitorch.fpga.accelerate``).

        With ``strict=True`` raises :class:`~tikitorch.fpga.NativeUnavailable`
        instead of falling back when the complete native binding is unavailable.
        """
        import tikitorch.fpga as _fpga

        return _fpga.accelerate(self, optimizer=optimizer, strict=strict)


def device(spec: Union[str, "Device", None] = "cpu") -> Device:
    """Construct a :class:`Device`, mirroring ``torch.device``.

    Accepts ``'cpu'``, ``'fpga'``, ``'fpga:0'``, an existing :class:`Device`
    (idempotent pass-through), or ``None`` (canonical CPU). Never raises for a
    well-formed string; raises :class:`ValueError` only on a malformed spec.

        >>> device('fpga:0')
        device(type='fpga', index=0)
        >>> device('cpu').is_cpu
        True
    """
    if spec is None:
        return Device("cpu")
    if isinstance(spec, Device):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"device spec must be a str, Device, or None, got {type(spec).__name__}")

    text = spec.strip()
    if not text:
        raise ValueError("device spec must be a non-empty string")

    if ":" in text:
        type_part, _, index_part = text.partition(":")
        type_part = type_part.strip()
        index_part = index_part.strip()
        if not type_part or not index_part:
            raise ValueError(f"malformed device spec {spec!r} (expected 'type' or 'type:index')")
        try:
            index = int(index_part)
        except ValueError:
            raise ValueError(f"device index in {spec!r} must be an integer") from None
        if index < 0:
            raise ValueError(f"device index in {spec!r} must be non-negative")
        return Device(type_part, index)

    return Device(text)
