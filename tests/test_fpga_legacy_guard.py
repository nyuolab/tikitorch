"""Obsolete unprobed MorphLM drivers must fail before a native library is loaded."""
import ctypes
import importlib.util
from pathlib import Path

import pytest


def test_legacy_morph_lm_never_loads_native_library(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/_validate_kernel_morph_lm.py"
    spec = importlib.util.spec_from_file_location("legacy_morph_lm_guard", path)
    module = importlib.util.module_from_spec(spec)

    def forbidden(*args, **kwargs):
        pytest.fail("obsolete driver attempted a native load")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    spec.loader.exec_module(module)
    with pytest.raises(RuntimeError, match="CE_LUT.*fpga_lm_pretrain"):
        module.Kernel()
