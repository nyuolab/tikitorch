"""tikitorch.fpga.accelerate — the DDP-style AcceleratedModel wrapper.

``model = tt.fpga.accelerate(model)`` is the headline entry point, deliberately
analogous to ``model = DistributedDataParallel(model)``: it wraps a model and you
keep your ordinary training loop. The wrapper:

* registers the inner model as a submodule, so ``.parameters()`` / ``.state_dict()``
  / optimizers work unchanged;
* delegates ``forward`` / ``compute_loss`` / ``param_count`` and proxies other
  attributes (``task_type``, ``set_tau``, ...) to the inner model;
* when **bound** (board present + shape matched + on-board runner registered), runs
  the fused on-board training step via :meth:`AcceleratedModel.train_step`;
* when **unbound** (the case in this repo/CI), is a *transparent CPU fallback*: it
  calls the inner model's own ``compute_loss``/optimizer verbatim, so results are
  byte-identical to the unwrapped model — never an error.
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
    """A model wrapped for FPGA execution (bound on board, else CPU fallback)."""

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
        """``True`` iff this model actually executes on the board.

        Requires a native handle, a matched kernel spec, *and* a registered on-board
        runner for that kernel (the last hardware-validated mile). When any is
        missing the wrapper runs on CPU.
        """
        return (
            self._native is not None
            and self.kernel_spec is not None
            and has_board_runner(self.kernel_spec)
        )

    def _unbound_reason(self) -> str:
        if self._native is None:
            return ("no FPGA board / native driver available (strict=True). "
                    "Build the driver per tikitorch/fpga/BUILD_NATIVE.md or omit strict.")
        if self.kernel_spec is None:
            return (f"no compiled kernel matches model {type(self.module).__name__} "
                    f"of this shape (strict=True). See KernelCache.default() for the allow-list.")
        return (f"no on-board runner registered for kernel {self.kernel_spec.entry!r} "
                f"(strict=True). See tikitorch/fpga/BUILD_NATIVE.md.")

    # -- delegation ---------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def compute_loss(self, x, y):
        """Compute the model's loss (always the inner model's own CPU computation)."""
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
        """One training step. On board when :attr:`bound`; else a CPU fallback.

        The bound path drives the fused ``FWD_LOSS -> BWD -> SIGNSGD`` loop on the
        FPGA. The fallback path runs the inner model's own
        ``compute_loss(x, targets).backward()`` + ``optimizer.step()/zero_grad()``
        (byte-identical to training the unwrapped model) and warns once.
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
                "CPU-fallback path (or run the ordinary loop: "
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
            self.device = device
            return self
        return super().to(device, *args, **kwargs)

    def _warn_fallback_once(self) -> None:
        if not self._fallback_warned:
            self._fallback_warned = True
            warnings.warn(
                f"tikitorch.fpga: model {type(self.module).__name__} is running on the "
                f"CPU fallback ({self._unbound_reason_short()}). Results are correct; "
                f"see tikitorch/fpga/BUILD_NATIVE.md to run on a board.",
                BackendFallbackWarning,
                stacklevel=2,
            )

    def _unbound_reason_short(self) -> str:
        if self._native is None:
            return "no board available"
        if self.kernel_spec is None:
            return "no matching compiled kernel"
        return "no on-board runner for the matched kernel"

    def __repr__(self) -> str:
        spec = self.kernel_spec.entry if self.kernel_spec is not None else None
        return (f"AcceleratedModel(module={type(self.module).__name__}, device={self.device!s}, "
                f"bound={self.bound}, kernel={spec!r})")
