"""tikitorch.fpga._native — the ctypes bridge to the compiled native driver.

This is the **only** place TikiTorch touches native code. It loads a shared
library (``_libtikifpga``) built out-of-band from the bundled HLS sources +
a thin C bridge (see :doc:`BUILD_NATIVE.md`) and binds the two verified extern-C
ABIs plus the buffer/board helpers.

**It degrades gracefully.** :func:`load` searches for the library and returns a
:class:`NativeHandle` on success, or ``None`` on *any* failure — missing library,
missing XRT runtime, or a board-probe miss. It NEVER raises and it NEVER calls a
board. In this repo/sandbox and in CI there is no library, so :func:`load`
returns ``None`` and the whole FPGA path transparently falls back to CPU.

The library is intentionally *not* shipped in the wheel (platform/ABI mismatch);
point the loader at a local build via ``$TIKITORCH_FPGA_LIB`` or drop the built
``_libtikifpga.{so,dylib}`` next to this file. This module must never ``import
forge`` (enforced by a package-wide test).
"""
from __future__ import annotations

import os
from typing import Optional

# Canonical fused-step mode codes (mirrors kernels.MODE_*).
MODE_FWD_LOSS = 0
MODE_BWD = 1
MODE_SIGNSGD = 2

# extern-C symbols the bridge is expected to export.
_BRIDGE_SYMBOLS = (
    "tt_fpga_open",        # (const char* xclbin_or_null) -> void* handle
    "tt_fpga_close",       # (void* handle)
    "tt_fpga_board_present",  # (void* handle) -> int (0/1)
    "tt_fpga_alloc",       # (void* handle, long words) -> void* buffer
    "tt_fpga_free",        # (void* handle, void* buffer)
    "tt_fpga_load_weights",   # (void* handle, void* buffer, const int16_t* w, long n)
    "tt_fpga_save_weights",   # (void* handle, void* buffer, int16_t* w, long n)
    "tt_fpga_run",         # (void* handle, const char* entry, void* buffer, int mode,
                           #  int a1, int a2, int a3, int layer_id) -> int rc
    "tt_fpga_write_region",   # (void* h, void* buf, long off_words, const int32_t* src, long n)
    "tt_fpga_read_region",    # (void* h, void* buf, long off_words, int32_t* dst, long n)
)


class NativeUnavailable(RuntimeError):
    """Raised on the ``strict=True`` opt-in path when no native handle is present.

    :func:`load` itself returns ``None`` rather than raising; this is the explicit
    error a strict caller asked for.
    """


class NativeHandle:
    """A thin wrapper over a loaded ``_libtikifpga`` and its argtype bindings.

    Constructed only by :func:`load`. Holds the ``ctypes.CDLL`` and the bound
    functions. :meth:`board_present` is the on-board probe used by
    ``FpgaBackend.is_available``.
    """

    def __init__(self, lib, libpath: str) -> None:
        self._lib = lib
        self.libpath = libpath

    def board_present(self) -> bool:
        """Probe whether a usable FPGA board is attached (never raises)."""
        try:
            handle = self._lib.tt_fpga_open(None)
            if not handle:
                return False
            present = bool(self._lib.tt_fpga_board_present(handle))
            self._lib.tt_fpga_close(handle)
            return present
        except Exception:  # noqa: BLE001 — a probe must never crash availability checks
            return False


def _candidate_paths() -> list:
    paths = []
    env = os.environ.get("TIKITORCH_FPGA_LIB")
    if env:
        paths.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("_libtikifpga.so", "_libtikifpga.dylib", "_libtikifpga.dll"):
        paths.append(os.path.join(here, name))
    return paths


def load(libpath: Optional[str] = None) -> Optional[NativeHandle]:
    """Load the native driver and bind its ABIs, or return ``None`` (never raises).

    Search order: an explicit ``libpath``, then ``$TIKITORCH_FPGA_LIB``, then a
    bundled ``_libtikifpga.{so,dylib,dll}`` next to this module. Returns ``None``
    if nothing loads or the symbols/argtypes don't bind.
    """
    import ctypes

    candidates = ([libpath] if libpath else []) + _candidate_paths()
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        try:
            _bind_signatures(ctypes, lib)
        except Exception:  # noqa: BLE001 — missing symbol / binding error -> skip this lib
            continue
        return NativeHandle(lib, path)
    return None


def _bind_signatures(ctypes, lib) -> None:
    """Bind argtypes/restypes for the bridge symbols (both kernel ABIs covered).

    ``tt_fpga_run`` carries a ``layer_id`` slot for every call; 5-int (FUSED_5)
    kernels ignore it (the bridge passes -1), 6-int (FUSED_6) kernels consume it.
    Routing arity through one bridge entry keeps the ctypes surface small while the
    C side dispatches by ``entry`` name.
    """
    c_void_p, c_char_p, c_int, c_long = (
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_long,
    )
    i16p = ctypes.POINTER(ctypes.c_int16)

    lib.tt_fpga_open.argtypes = [c_char_p]
    lib.tt_fpga_open.restype = c_void_p
    lib.tt_fpga_close.argtypes = [c_void_p]
    lib.tt_fpga_close.restype = None
    lib.tt_fpga_board_present.argtypes = [c_void_p]
    lib.tt_fpga_board_present.restype = c_int
    lib.tt_fpga_alloc.argtypes = [c_void_p, c_long]
    lib.tt_fpga_alloc.restype = c_void_p
    lib.tt_fpga_free.argtypes = [c_void_p, c_void_p]
    lib.tt_fpga_free.restype = None
    lib.tt_fpga_load_weights.argtypes = [c_void_p, c_void_p, i16p, c_long]
    lib.tt_fpga_load_weights.restype = None
    lib.tt_fpga_save_weights.argtypes = [c_void_p, c_void_p, i16p, c_long]
    lib.tt_fpga_save_weights.restype = None
    lib.tt_fpga_run.argtypes = [c_void_p, c_char_p, c_void_p, c_int, c_int, c_int, c_int, c_int]
    lib.tt_fpga_run.restype = c_int
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.tt_fpga_write_region.argtypes = [c_void_p, c_void_p, c_long, i32p, c_long]
    lib.tt_fpga_write_region.restype = None
    lib.tt_fpga_read_region.argtypes = [c_void_p, c_void_p, c_long, i32p, c_long]
    lib.tt_fpga_read_region.restype = None
