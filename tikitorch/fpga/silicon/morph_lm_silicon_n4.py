#!/usr/bin/env python3
"""morph_lm_silicon_n4.py — N-WAY REPLICATED FPGA silicon driver for the
config-B MorphSSM LM training kernel on the v3 multi-kernel CL.

The v3 CL instantiates NK independent kernels, each with its own 2 MB on-CL
staging URAM (DMA-PCIS window i at 0x0100_0000 + i*0x20_0000) and its own
resident weights. OCL control is BROADCAST, so a single AP_START kick starts
all NK at once; reads are muxed per instance by the control-address window.

This driver stages the SAME ground-truth buffer into all NK windows, kicks all
NK with ONE non-blocking mode-3 call (whole 30-step run on fabric), polls each
instance's ap_done, and reports the AGGREGATE throughput (NK*30*128 tokens over
the wall of the slowest instance). It then reads back each instance's five
weight regions and verifies every instance is BIT-EXACT vs the host golden
(wcrc 14650122959732499763) — proving NK independent correct trainings ran in
parallel.

  NK              via env TT_NK (default 4) — MUST match the CL's localparam NK
  TT_FUSED_STEPS  steps per kick (default 30)
Window stride = 2 MB = 0x80000 int32 words (matches the CL 2 MB URAM window).
"""
import ctypes
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_buffer_init as L   # noqa: E402  (layout constants; ML_BUFFER_WORDS, OFF_*)

NK       = int(os.environ.get("TT_NK", "4"))
STEPS    = int(os.environ.get("TT_FUSED_STEPS", "30"))
SGD_LR   = 1
SGD_THR  = int(os.environ.get("TT_SGD_THR", "2048"))
GATE_CRC = 14650122959732499763
WIN_WORDS = 0x80000            # 2 MB window in int32 words (must match CL stride)

ENTRY    = b"forge_train_step_morph_lm"
LIB_PATH = os.environ.get("TIKITORCH_FPGA_LIB", "/tmp/_libtikifpga_pci.so")
i32p = ctypes.POINTER(ctypes.c_int32)


def load_lib(path):
    lib = ctypes.CDLL(path)
    c_void_p, c_char_p, c_int, c_long = (ctypes.c_void_p, ctypes.c_char_p,
                                         ctypes.c_int, ctypes.c_long)
    lib.tt_fpga_open.argtypes = [c_char_p]; lib.tt_fpga_open.restype = c_void_p
    lib.tt_fpga_alloc.argtypes = [c_void_p, c_long]; lib.tt_fpga_alloc.restype = c_void_p
    lib.tt_fpga_write_region.argtypes = [c_void_p, c_void_p, c_long, i32p, c_long]
    lib.tt_fpga_read_region.argtypes = [c_void_p, c_void_p, c_long, i32p, c_long]
    lib.tt_fpga_kick.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
    lib.tt_fpga_kick.restype = None
    lib.tt_fpga_done.argtypes = [c_void_p, c_int]; lib.tt_fpga_done.restype = c_int
    return lib


def fnv1a_words(words, h):
    MASK = 0xFFFFFFFFFFFFFFFF; PRIME = 1099511628211
    for w in words:
        u = w & 0xFFFFFFFF
        for b in range(4):
            h ^= (u >> (b * 8)) & 0xFF
            h = (h * PRIME) & MASK
    return h


def weight_crc(buf):
    h = 1469598103934665603
    for off, n in ((L.OFF_EMBED, L.WORDS_EMBED), (L.OFF_MH, L.WORDS_MH),
                   (L.OFF_MP, L.WORDS_MP), (L.OFF_MB, L.WORDS_MB),
                   (L.OFF_W_HEAD, L.WORDS_W_HEAD)):
        h = fnv1a_words(buf[off:off + n], h)
    return h


def write_region(lib, h, dbuf, off, ints):
    n = len(ints); arr = (ctypes.c_int32 * n)(*ints)
    lib.tt_fpga_write_region(h, dbuf, off, arr, n)


def read_region(lib, h, dbuf, off, n):
    arr = (ctypes.c_int32 * n)(); lib.tt_fpga_read_region(h, dbuf, off, arr, n)
    return list(arr)


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: morph_lm_silicon_n4.py <buffer_init.bin>\n"); return 2
    with open(argv[1], "rb") as f:
        raw = f.read()
    nwords = len(raw) // 4
    if nwords != L.ML_BUFFER_WORDS:
        sys.stderr.write("bad init size: %d words, want %d\n" % (nwords, L.ML_BUFFER_WORDS)); return 3
    init = list(struct.unpack("<%di" % nwords, raw))
    tgts = [init[L.OFF_TARGETS + m] for m in range(L.ML_L_SEQ)]

    lib = load_lib(LIB_PATH)
    h = lib.tt_fpga_open(None)
    if not h:
        sys.stderr.write("tt_fpga_open returned NULL\n"); return 4
    dbuf = ctypes.c_void_p(lib.tt_fpga_alloc(h, NK * WIN_WORDS))

    # ---- stage the SAME init buffer into all NK windows ----
    for i in range(NK):
        write_region(lib, h, dbuf, i * WIN_WORDS, init)
    sys.stderr.write("staged %d instances (%d words each)\n" % (NK, nwords))

    # ---- ONE broadcast kick (mode 3, whole run) -> all NK start ----
    t0 = time.time()
    lib.tt_fpga_kick(h, 3, STEPS, SGD_LR, SGD_THR)
    done_t = [None] * NK
    deadline = t0 + 60.0
    while any(d is None for d in done_t):
        for i in range(NK):
            if done_t[i] is None and lib.tt_fpga_done(h, i):
                done_t[i] = time.time()
        if time.time() > deadline:
            sys.stderr.write("TIMEOUT: done=%s\n" % [d is not None for d in done_t]); return 5
    wall = max(done_t) - t0

    # ---- per-instance verification ----
    L_SEQ = L.ML_L_SEQ
    all_ok = True; crcs = []; accs = []
    for i in range(NK):
        base = i * WIN_WORDS
        shadow = [0] * L.ML_BUFFER_WORDS
        for off, nw in ((L.OFF_EMBED, L.WORDS_EMBED), (L.OFF_MH, L.WORDS_MH),
                        (L.OFF_MP, L.WORDS_MP), (L.OFF_MB, L.WORDS_MB),
                        (L.OFF_W_HEAD, L.WORDS_W_HEAD)):
            shadow[off:off + nw] = read_region(lib, h, dbuf, base + off, nw)
        crc = weight_crc(shadow); crcs.append(crc)
        preds = read_region(lib, h, dbuf, base + L.OFF_PRED, L_SEQ)
        accs.append(sum(1 for m in range(L_SEQ) if preds[m] == tgts[m]) / float(L_SEQ))
        if crc != GATE_CRC:
            all_ok = False

    toks = NK * STEPS * L_SEQ
    print('{"nk":%d,"fused_steps":%d,"wall_s":%.4f,"agg_tok_s":%.0f,'
          '"per_kernel_tok_s":%.0f,"all_bit_exact":%s,"acc":%.6f}'
          % (NK, STEPS, wall, toks / wall, (STEPS * L_SEQ) / wall,
             "true" if all_ok else "false", accs[0]))
    sys.stderr.write("done_times(rel ms): %s\n" % [round((d - t0) * 1e3, 1) for d in done_t])
    sys.stderr.write("per-instance wcrc: %s\n" % crcs)
    sys.stderr.write("GATE %s (want %d)\n" % ("PASS" if all_ok else "FAIL", GATE_CRC))
    return 0 if all_ok else 6


if __name__ == "__main__":
    sys.exit(main(sys.argv))
