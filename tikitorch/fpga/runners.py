"""Compatibility surface for the retired bundled FPGA runner.

TikiTorch previously exposed a one-layer ``forge_train_step`` runner that
flattened ``model.parameters()`` into the kernel buffer. That mapping is unsafe:
PyTorch registration order does not match the HLS named-region order, and the
runner did not seed or synchronize the ``B1``/``B2`` regions. It is therefore
not registered automatically and cannot be re-enabled through the legacy
:func:`register_reference_runners` hook.

There is currently no safe bundled native runner. Kernel catalog inspection and
structural matching remain available, while execution falls back to PyTorch
unless an application explicitly registers a kernel- and build-specific runner
with :func:`tikitorch.fpga.register_board_runner`. See ``BUILD_NATIVE.md`` for
the remaining runtime and build-manifest checks that such a runner must own.
"""
from __future__ import annotations

from ._native import NativeUnavailable


def register_reference_runners() -> None:
    """Fail closed: the historical reference runner is not safe to register.

    This compatibility name is retained so older callers receive an actionable
    error instead of silently activating the legacy flat-parameter marshaller.
    """
    raise NativeUnavailable(
        "the bundled forge_train_step reference runner is disabled: its flat-parameter "
        "mapping does not match the kernel's named buffer regions and it omits B1/B2 "
        "synchronization. Implement a named, runtime-validated, build-specific runner "
        "and register it explicitly with tikitorch.fpga.register_board_runner(); see "
        "tikitorch/fpga/BUILD_NATIVE.md."
    )
