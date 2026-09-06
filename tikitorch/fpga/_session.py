"""tikitorch.fpga._session — the native fused-step boundary.

When an :class:`~tikitorch.fpga.accelerate.AcceleratedModel` is *bound*, its
``train_step`` delegates to a registered kernel-specific runner for the matched
kernel. The runner is the transport-facing last mile—it owns the kernel's flat
INT32 buffer layout, marshals the model's weights to/from INT16, and drives the
*exact* fused sequence the bundled testbench (``forge_train_step_*_tb.cpp``)
prescribes: cache handling, per-layer phasing, position ordering, and the final
``SIGNSGD`` kick.

That sequence is **buffer-layout-coupled and kernel-specific** (e.g. the K/V cache
regions are zeroed directly in the buffer between the forward and backward phases,
and the multi-layer LM kernels phase forward/backward by ``layer_id`` while the
morph kernels run a single whole-sequence pass) — so it is *not* a generic Python
loop. A runner must validate named model-to-buffer regions, runtime inputs, and
the native library/build manifest against the functional testbench, then be
re-verified on each target synthesized image.

**No safe runner ships or registers by default.** The historical one-layer
flat-parameter runner misordered named regions and omitted ``B1``/``B2``
synchronization, so its compatibility registration hook fails loudly. A model
is *bound* only when an application explicitly registers a runner for its
matched kernel (see :func:`has_board_runner`) and a usable native endpoint
exists. Registration asserts responsibility for the runner-owned runtime and
build checks; ``bound`` does not prove them. LOCAL is a host endpoint, not a
board. See ``BUILD_NATIVE.md`` for the custom-runner contract.
"""
from __future__ import annotations

from typing import Callable, Dict

from ._native import NativeHandle, NativeUnavailable
from .kernels import KernelSpec

# entry-symbol -> kernel-specific native runner. Signature:
#   runner(handle: NativeHandle, spec: KernelSpec, model, tokens, *,
#          targets=None, lr: int, threshold: int) -> float   # scalar loss proxy
# The runner does the whole fused step: validate runtime/build identity, marshal
# named weights into the kernel buffer, drive the kernel-specific sequence, sync
# trained weights back into `model`, and return a loss proxy. None ship by default.
BOARD_RUNNERS: Dict[str, Callable] = {}


def register_board_runner(entry: str, fn: Callable, *, replace: bool = False) -> None:
    """Register an application-supplied native runner for ``entry``.

    Registration is an explicit trust boundary: the caller is responsible for
    the runtime, named-layout, ABI, and build-manifest checks in ``BUILD_NATIVE.md``.
    The legacy function name is retained for compatibility; LOCAL host runners
    use the same registry. Replacing a different runner requires
    ``replace=True`` so an import cannot silently change the trusted boundary.
    """
    if not isinstance(entry, str) or not entry.strip() or "\x00" in entry:
        raise ValueError("entry must be a non-empty kernel symbol without NUL bytes")
    if entry != entry.strip():
        raise ValueError("entry must not contain leading or trailing whitespace")
    if not callable(fn):
        raise TypeError("fn must be callable")
    if entry in BOARD_RUNNERS and BOARD_RUNNERS[entry] is not fn and not replace:
        raise ValueError(
            f"a board runner is already registered for {entry!r}; "
            "pass replace=True to replace it explicitly"
        )
    BOARD_RUNNERS[entry] = fn


def has_board_runner(spec: KernelSpec) -> bool:
    """Whether an application supplied a native runner for ``spec``'s kernel."""
    return spec.entry in BOARD_RUNNERS


def run_board_step(handle: NativeHandle, spec: KernelSpec, model, tokens, *,
                   targets=None, lr: int = 1, threshold: int = 0) -> float:
    """Run one fused native step via the kernel's registered runner.

    Requires both a live ``handle`` and an explicitly registered runner for
    ``spec``; otherwise raises :class:`NativeUnavailable`. The runner owns all
    runtime, marshalling, and build-identity validation. With no runner registered
    a model is never *bound*, so this is unreachable on the fallback path.
    """
    runner = BOARD_RUNNERS.get(spec.entry)
    if handle is None or runner is None:
        raise NativeUnavailable(
            f"no explicitly registered native runner for {spec.entry!r}: "
            f"{'no native handle' if handle is None else 'none registered'}. "
            f"Author one against forge_kernels/{spec.entry}_tb.cpp and register it "
            f"with tikitorch.fpga.register_board_runner(); see "
            f"tikitorch/fpga/BUILD_NATIVE.md."
        )
    return runner(handle, spec, model, tokens, targets=targets, lr=lr, threshold=threshold)
