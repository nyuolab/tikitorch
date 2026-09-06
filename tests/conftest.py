"""Shared pytest fixtures / skip helpers for the TikiTorch suite.

The public modules are present in a release checkout. The ``requires_*`` helpers
make an individual test's symbol or native-board dependency explicit; CI's import
and export-contract tests separately require the complete public package surface.
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
    """Mark a test that requires ``tikitorch.optim.<name>``."""
    return pytest.mark.skipif(
        not _attr_present("tikitorch.optim", name),
        reason=f"required tikitorch.optim.{name} symbol is unavailable",
    )


def requires_model(name: str):
    """Mark a test that requires ``tikitorch.models.<name>``."""
    return pytest.mark.skipif(
        not _attr_present("tikitorch.models", name),
        reason=f"required tikitorch.models.{name} symbol is unavailable",
    )


def requires_nn(name: str):
    """Mark a test that requires ``tikitorch.nn.<name>``."""
    return pytest.mark.skipif(
        not _attr_present("tikitorch.nn", name),
        reason=f"required tikitorch.nn.{name} symbol is unavailable",
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
