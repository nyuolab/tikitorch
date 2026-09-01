"""tikitorch.fpga.accelerate — the DDP-style AcceleratedModel wrapper.

``model = tt.fpga.accelerate(model)`` wraps a model while preserving normal
``nn.Module`` registration and delegation. The wrapper:

* registers the inner model as a submodule, so ``.parameters()`` / ``.state_dict()``
  / optimizers work unchanged;
* delegates ``forward`` / ``compute_loss`` / ``param_count`` and proxies other
  attributes (``task_type``, ``set_tau``, ...) to the inner model;
* is **bound** only with a usable native endpoint, a full model-observable
  structural match, and an explicitly registered runner;
* when unbound, calls the inner model's ordinary PyTorch loss/backward/optimizer
  path and emits one fallback warning from :meth:`train_step`.

No safe runner ships by default. ``bound`` means that the wrapper will delegate
to an application-supplied runner; it does not establish runtime tensor validity,
build-manifest identity, or physical FPGA execution. LOCAL is a host endpoint.
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch.nn as nn

from ..backend.device import Device
from ..backend.registry import BackendFallbackWarning
from ._native import NativeUnavailable
from ._session import has_board_runner
from .kernels import KernelSpec


class AcceleratedModel(nn.Module):
    """A model wrapper for explicit native runners, with a PyTorch fallback."""

    def __init__(self, model: nn.Module, *, spec: Optional[KernelSpec], native,
                 device: Device, optimizer=None) -> None:
        super().__init__()
        self.module = model            # registered as a submodule
        self.kernel_spec = spec
        self._native = native
        self.device = device           # a logical tikitorch Device (not a torch device)
        self._optimizer = optimizer
        self._fallback_warned = False

    # -- construction (single source of truth, used by FpgaBackend + accelerate) --

    @classmethod
    def _build(cls, model, *, spec, native, device, optimizer=None, strict=False) -> "AcceleratedModel":
        am = cls(model, spec=spec, native=native, device=device, optimizer=optimizer)
        if strict and not am.bound:
            raise NativeUnavailable(am._unbound_reason())
        return am

    # -- status -------------------------------------------------------------

    @property
    def bound(self) -> bool:
        """Whether all dispatch prerequisites for a native runner are present.

        This checks a usable handle supplied by the backend, a structural match,
        and explicit runner registration. The runner remains responsible for
        runtime inputs, named buffer layout, ABI/build identity, and transport
        mode; a bound LOCAL endpoint still executes on the host.
        """
        return (
            self._native is not None
            and self.kernel_spec is not None
            and has_board_runner(self.kernel_spec)
        )

    def _unbound_reason(self) -> str:
        reasons = []
        if self._native is None:
            reasons.append("no usable native endpoint")
        if self.kernel_spec is None:
            reasons.append(
                f"no full structural kernel match for {type(self.module).__name__} "
                "(catalog lookup alone is insufficient)"
            )
        elif not has_board_runner(self.kernel_spec):
            reasons.append(
                f"no explicitly registered runner for {self.kernel_spec.entry!r}"
            )
        detail = "; ".join(reasons) if reasons else "unknown binding failure"
        return (
            f"native FPGA binding unavailable (strict=True): {detail}. "
            "See tikitorch/fpga/BUILD_NATIVE.md; strict mode does not validate a "
            "custom runner's runtime inputs or build manifest."
        )

    # -- delegation ---------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def compute_loss(self, x, y):
        """Compute the loss through the inner model's ordinary PyTorch path."""
        return self.module.compute_loss(x, y)

    @property
    def param_count(self) -> int:
        pc = getattr(self.module, "param_count", None)
        if isinstance(pc, int):
            return pc
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def __getattr__(self, name):
        # nn.Module resolves params/buffers/submodules here first.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        module = self.__dict__.get("_modules", {}).get("module")
        if module is not None and name != "module":
            return getattr(module, name)
        raise AttributeError(name)

    # -- training -----------------------------------------------------------

    def train_step(self, x, targets=None, *, lr=None, threshold=None, optimizer=None) -> float:
        """Run one native fused step when bound, otherwise use PyTorch.

        The bound path delegates to the explicitly registered runner, which may
        target LOCAL host-reference execution or a physical FPGA transport. The
        fallback path runs the inner model's own
        ``compute_loss(x, targets).backward()`` + ``optimizer.step()/zero_grad()``
        and warns once.
        """
        if self.bound:
            from ._session import run_board_step
            return run_board_step(
                self._native, self.kernel_spec, self.module, x,
                targets=targets, lr=lr if lr is not None else 1,
                threshold=threshold if threshold is not None else 0,
            )

        if targets is None:
            raise ValueError(
                "AcceleratedModel.train_step(x, targets): targets are required on the "
                "PyTorch-fallback path (or run the ordinary loop: "
                "loss = model.compute_loss(x, y); loss.backward(); opt.step())."
            )
        self._warn_fallback_once()
        opt = optimizer if optimizer is not None else self._optimizer
        loss = self.module.compute_loss(x, targets)
        loss.backward()
        if opt is not None:
            opt.step()
            opt.zero_grad()
        return float(loss.item())

    def fpga(self, *, optimizer=None, strict: bool = False):  # type: ignore[override]
        """Idempotent: re-accelerate the inner model (never nests wrappers)."""
        import tikitorch.fpga as _fpga

        return _fpga.accelerate(self.module, optimizer=optimizer, strict=strict)

    def cpu(self):  # type: ignore[override]
        """Unwrap to the plain inner model (the inverse of ``.fpga()``)."""
        return self.module

    def to(self, device, *args, **kwargs):  # type: ignore[override]
        """Re-target the logical device, or defer to ``nn.Module.to`` for torch moves.

        Only a tikitorch :class:`~tikitorch.backend.Device` unwraps/re-targets:
        ``to(tt.device('cpu'))`` returns the plain inner model and
        ``to(tt.device('fpga'))`` re-binds the wrapper. Everything else — including
        the *string* ``to('cpu')`` and any torch device/dtype — forwards to
        ``nn.Module.to`` (which moves the inner parameters and keeps the wrapper), so
        ordinary PyTorch ``.to(...)`` semantics are preserved.
        """
        if isinstance(device, Device):
            if device.is_cpu:
                return self.module
            if device.type != "fpga":
                raise ValueError(
                    f"AcceleratedModel accepts logical 'cpu' or 'fpga' devices, got {device!s}"
                )
            if device.index not in (None, 0):
                raise ValueError(
                    f"unsupported FPGA endpoint {device!s}; only 'fpga' and 'fpga:0' are available"
                )
            self.device = device
            return self
        return super().to(device, *args, **kwargs)

    def _warn_fallback_once(self) -> None:
        if not self._fallback_warned:
            self._fallback_warned = True
            warnings.warn(
                f"tikitorch.fpga: model {type(self.module).__name__} is running on the "
                f"PyTorch fallback ({self._unbound_reason_short()}). No native runner "
                f"was invoked; see tikitorch/fpga/BUILD_NATIVE.md for the explicit "
                f"binding contract.",
                BackendFallbackWarning,
                stacklevel=2,
            )

    def _unbound_reason_short(self) -> str:
        reasons = []
        if self._native is None:
            reasons.append("no usable native endpoint")
        if self.kernel_spec is None:
            reasons.append("no full structural kernel match")
        elif not has_board_runner(self.kernel_spec):
            reasons.append("no explicitly registered runner")
        return "; ".join(reasons) if reasons else "binding unavailable"

    def __repr__(self) -> str:
        spec = self.kernel_spec.entry if self.kernel_spec is not None else None
        return (f"AcceleratedModel(module={type(self.module).__name__}, device={self.device!s}, "
                f"bound={self.bound}, kernel={spec!r})")
