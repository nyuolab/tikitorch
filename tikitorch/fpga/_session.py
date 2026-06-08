"""tikitorch.fpga._session — the on-board training-step boundary.

When an :class:`~tikitorch.fpga.accelerate.AcceleratedModel` is *bound*, its
``train_step`` delegates to a registered **on-board runner** for the matched
kernel. The runner is the hardware-validated last mile — it owns the kernel's flat
INT32 buffer layout, marshals the model's weights to/from INT16, and drives the
*exact* fused sequence the bundled testbench (``forge_train_step_*_tb.cpp``)
prescribes: cache handling, per-layer phasing, position ordering, and the final
``SIGNSGD`` kick.

That sequence is **buffer-layout-coupled and kernel-specific** (e.g. the K/V cache
regions are zeroed directly in the buffer between the forward and backward phases,
and the multi-layer LM kernels phase forward/backward by ``layer_id`` while the
morph kernels run a single whole-sequence pass) — so it is *not* a generic Python
loop. It is authored and validated against a real board, mirroring the testbench.

**No runner ships by default.** A model is therefore *bound* only when a runner is
registered for its kernel (see :func:`has_board_runner`), so the absence of the
last mile is a clean CPU fallback — never a silently-wrong on-board run. See
``BUILD_NATIVE.md`` for how to author one against the bundled testbench.
"""
from __future__ import annotations

from typing import Callable, Dict

from ._native import NativeHandle, NativeUnavailable
from .kernels import KernelSpec

# entry-symbol -> on-board runner. Signature:
#   runner(handle: NativeHandle, spec: KernelSpec, model, tokens, *,
#          targets=None, lr: int, threshold: int) -> float   # scalar loss proxy
# The runner does the whole on-board step: marshal weights into the kernel buffer,
# drive the validated fused FWD_LOSS/BWD/SIGNSGD sequence for `spec.entry`, sync the
# trained weights back into `model`, and return a loss proxy. None ship by default.
BOARD_RUNNERS: Dict[str, Callable] = {}


def register_board_runner(entry: str, fn: Callable) -> None:
    """Register an on-board step runner for the kernel whose entry symbol is ``entry``."""
    BOARD_RUNNERS[entry] = fn


def has_board_runner(spec: KernelSpec) -> bool:
    """Whether an on-board runner is registered for ``spec``'s kernel."""
    return spec.entry in BOARD_RUNNERS


def run_board_step(handle: NativeHandle, spec: KernelSpec, model, tokens, *,
                   targets=None, lr: int = 1, threshold: int = 0) -> float:
    """Run one fused on-board training step via the kernel's registered runner.

    Requires both a live ``handle`` and a registered runner for ``spec``; otherwise
    raises :class:`NativeUnavailable` — the honest signal that the hardware-validated
    last mile is not in place for this kernel. (With no runner registered a model is
    never *bound*, so this is unreachable on the CPU-fallback path.)
    """
    runner = BOARD_RUNNERS.get(spec.entry)
    if handle is None or runner is None:
        raise NativeUnavailable(
            f"no on-board runner for kernel {spec.entry!r}: "
            f"{'no native handle' if handle is None else 'none registered'}. "
            f"Author one against forge_kernels/{spec.entry}_tb.cpp and register it "
            f"with tikitorch.fpga.register_board_runner(); see "
            f"tikitorch/fpga/BUILD_NATIVE.md."
        )
    return runner(handle, spec, model, tokens, targets=targets, lr=lr, threshold=threshold)
