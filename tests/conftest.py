"""Shared pytest fixtures / skip helpers for the TikiTorch suite.

Some modules (``functional`` + the core ``nn`` layers) are always present, so
those tests run unconditionally. Others (``optim``, ``models``, the higher-level
``nn`` blocks) may not be available in a given working tree. Tests that depend on
those optional symbols are guarded with the ``requires_*`` skip markers below:
they turn green the moment
the corresponding module is populated, and skip cleanly until then instead of
hard-failing the whole run.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture(autouse=True)
def _isolate_backend_globals():
    """Snapshot/restore the backend seam's process-global state around each test.

    The backend registry keeps two process-globals (the default logical device and
    the one-time fallback-warning dedup set). Reset them around every test so a
    stray ``set_default_device`` or an emitted warning can't make tests
    order-dependent. No-op if the backend isn't importable.
    """
    try:
        import tikitorch.backend as B
    except Exception:
        yield
        return
    before = B.get_default_device()
    yield
    B.set_default_device(before)
    B._reset_fallback_warnings()


def _attr_present(module_name: str, attr: str) -> bool:
    try:
        mod = __import__(module_name, fromlist=[attr])
    except Exception:
        return False
    return hasattr(mod, attr)


def requires_optim(name: str):
    """Skip unless ``tikitorch.optim.<name>`` exists."""
    return pytest.mark.skipif(
        not _attr_present("tikitorch.optim", name),
        reason=f"tikitorch.optim.{name} not implemented yet",
    )


def requires_model(name: str):
    """Skip unless ``tikitorch.models.<name>`` exists."""
    return pytest.mark.skipif(
        not _attr_present("tikitorch.models", name),
        reason=f"tikitorch.models.{name} not implemented yet",
    )


def requires_nn(name: str):
    """Skip unless ``tikitorch.nn.<name>`` exists."""
    return pytest.mark.skipif(
        not _attr_present("tikitorch.nn", name),
        reason=f"tikitorch.nn.{name} not implemented yet",
    )


def _fpga_native_available() -> bool:
    try:
        import tikitorch.fpga as fpga
    except Exception:
        return False
    return bool(fpga.is_available())


def requires_fpga_native():
    """Skip unless a native FPGA driver + board is present (board-only assertions)."""
    return pytest.mark.skipif(
        not _fpga_native_available(),
        reason="native FPGA driver/board unavailable (CPU-fallback path runs instead)",
    )
