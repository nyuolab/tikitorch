#!/usr/bin/env python3
"""morph_lm_silicon.py — FPGA SILICON driver for the config-B MorphSSM LM
training kernel (forge_train_step_morph_lm) on the AWS F2 box.

ctypes-loads the PCI bridge (_libtikifpga_pci.so), opens the board, allocates
the ML_BUFFER_WORDS int32 buffer, writes the SHARED ground-truth buffer_init.bin
(the SAME bytes the host CSIM golden consumes), and runs the IDENTICAL optimizer
schedule on hardware via tt_fpga_run. After each step it reads the buffer back,
computes the SAME {step,acc,wcrc,preds16} JSONL, and times the fwd+bwd inner
loop to report tokens/sec and total wall.

If the hardware is correct, this emits JSONL BYTE-IDENTICAL to the golden:
    python3 gen_buffer_init.py /tmp/buffer_init.bin
    clang++ ... golden_morph_lm.cpp ... -o /tmp/golden
    /tmp/golden /tmp/buffer_init.bin > /tmp/golden_traj.jsonl
    python3 morph_lm_silicon.py /tmp/buffer_init.bin > /tmp/silicon_traj.jsonl
    diff /tmp/golden_traj.jsonl /tmp/silicon_traj.jsonl   # must be empty

Library path defaults to /tmp/_libtikifpga_pci.so, overridable via the
TIKITORCH_FPGA_LIB env var. Built on the F2 box from bridge.cpp with:
    g++ -O2 -std=c++17 -fPIC -shared -DTT_FPGA_PCI \
        -I"$SDK_DIR/userspace/include" tikitorch/fpga/bridge.cpp \
        -lfpga_mgmt -o _libtikifpga_pci.so
"""
import ctypes
import os
import struct
import sys
import time

# Layout + run params come from the SHARED generator so the driver and golden
# can never drift. (Pure-constants import; no RNG side effects.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_buffer_init as L   # noqa: E402

# ---- run parameters — MUST match golden_morph_lm.cpp ----
STEPS   = 30
SGD_LR  = 1
SGD_THR = 2048

ENTRY = b"forge_train_step_morph_lm"
LIB_PATH = os.environ.get("TIKITORCH_FPGA_LIB", "/tmp/_libtikifpga_pci.so")

i32p = ctypes.POINTER(ctypes.c_int32)


def load_lib(path):
    lib = ctypes.CDLL(path)
    c_void_p, c_char_p = ctypes.c_void_p, ctypes.c_char_p
    c_int, c_long = ctypes.c_int, ctypes.c_long
    lib.tt_fpga_open.argtypes = [c_char_p]
    lib.tt_fpga_open.restype = c_void_p
    lib.tt_fpga_alloc.argtypes = [c_void_p, c_long]
    lib.tt_fpga_alloc.restype = c_void_p
    lib.tt_fpga_write_region.argtypes = [c_void_p, c_void_p, c_long, i32p, c_long]
    lib.tt_fpga_write_region.restype = None
    lib.tt_fpga_read_region.argtypes = [c_void_p, c_void_p, c_long, i32p, c_long]
    lib.tt_fpga_read_region.restype = None
    # 8 args: (handle, entry, buf, mode, a1, a2, a3, layer_id) -> rc
    lib.tt_fpga_run.argtypes = [c_void_p, c_char_p, c_void_p, c_int,
                                c_int, c_int, c_int, c_int]
    lib.tt_fpga_run.restype = c_int
    return lib


def fnv1a_words(words, h):
    """64-bit FNV-1a over int32 words, little-endian bytes. Byte-identical to
    the C++ fnv1a_words in golden_morph_lm.cpp."""
    MASK = 0xFFFFFFFFFFFFFFFF
    PRIME = 1099511628211
    for w in words:
        u = w & 0xFFFFFFFF
        for b in range(4):
            byte = (u >> (b * 8)) & 0xFF
            h ^= byte
            h = (h * PRIME) & MASK
    return h


def weight_crc(buf):
    """FNV-1a over the five weight regions in order. `buf` is a list of signed
    int32 (the host shadow). Matches golden_morph_lm.cpp::weight_crc exactly."""
    h = 1469598103934665603  # FNV offset basis
    for off, n in ((L.OFF_EMBED, L.WORDS_EMBED),
                   (L.OFF_MH, L.WORDS_MH),
                   (L.OFF_MP, L.WORDS_MP),
                   (L.OFF_MB, L.WORDS_MB),
                   (L.OFF_W_HEAD, L.WORDS_W_HEAD)):
        h = fnv1a_words(buf[off:off + n], h)
    return h


# A small zero block reused for region zeroing on-device.
_ZERO_CACHE = {}


def zero_region(lib, h, dbuf, off, n):
    z = _ZERO_CACHE.get(n)
    if z is None:
        z = (ctypes.c_int32 * n)()   # zero-initialized
        _ZERO_CACHE[n] = z
    lib.tt_fpga_write_region(h, dbuf, off, z, n)


def write_region(lib, h, dbuf, off, ints):
    n = len(ints)
    arr = (ctypes.c_int32 * n)(*ints)
    lib.tt_fpga_write_region(h, dbuf, off, arr, n)


def read_region(lib, h, dbuf, off, n):
    arr = (ctypes.c_int32 * n)()
    lib.tt_fpga_read_region(h, dbuf, off, arr, n)
    return list(arr)


def run(lib, h, dbuf, mode, a1, a2, a3):
    rc = lib.tt_fpga_run(h, ENTRY, dbuf, mode, a1, a2, a3, -1)  # layer_id ignored
    if rc < -90:
        raise RuntimeError(
            "tt_fpga_run mode=%d args=(%d,%d,%d) rc=%d (hang/transport error)"
            % (mode, a1, a2, a3, rc))
    if rc != 0:
        raise RuntimeError(
            "tt_fpga_run mode=%d args=(%d,%d,%d) rc=%d (kernel error)"
            % (mode, a1, a2, a3, rc))
    return rc


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: morph_lm_silicon.py <buffer_init.bin>\n")
        return 2
    path = argv[1]

    # ---- load the SHARED init buffer (raw signed int32, little-endian) ----
    with open(path, "rb") as f:
        raw = f.read()
    nwords = len(raw) // 4
    if nwords != L.ML_BUFFER_WORDS:
        sys.stderr.write("bad init size: %d words, want %d\n"
                         % (nwords, L.ML_BUFFER_WORDS))
        return 3
    init = list(struct.unpack("<%di" % nwords, raw))

    toks = [init[L.OFF_TOKENS + m] for m in range(L.ML_L_SEQ)]
    tgts = [init[L.OFF_TARGETS + m] for m in range(L.ML_L_SEQ)]

    # ---- open board, alloc, write the whole init buffer ----
    lib = load_lib(LIB_PATH)
    h = lib.tt_fpga_open(None)            # slot 0; pokes buffer base at open
    if not h:
        sys.stderr.write("tt_fpga_open returned NULL\n")
        return 4
    dbuf = ctypes.c_void_p(lib.tt_fpga_alloc(h, L.ML_BUFFER_WORDS))
    write_region(lib, h, dbuf, 0, init)  # full buffer (weights, CFG, tok/tgt)

    L_SEQ = L.ML_L_SEQ
    inner_tokens = 0       # fwd+bwd token-positions processed (for tok/s)
    t_inner = 0.0
    t0_wall = time.time()

    for step in range(STEPS):
        # (1) zero BS stream before forward sweep.
        zero_region(lib, h, dbuf, L.OFF_BS_STREAM, L.WORDS_BS_STREAM)

        # ---- timed fwd+bwd inner loop ----
        t_a = time.time()

        # (2) FORWARD m = 0..L-1.
        for m in range(L_SEQ):
            run(lib, h, dbuf, 0, m, toks[m], tgts[m])

        # (3) zero 5 grad regions + re-zero BS stream (now grad_BS accum).
        zero_region(lib, h, dbuf, L.OFF_G_EMBED,  L.WORDS_G_EMBED)
        zero_region(lib, h, dbuf, L.OFF_G_MH,     L.WORDS_G_MH)
        zero_region(lib, h, dbuf, L.OFF_G_MP,     L.WORDS_G_MP)
        zero_region(lib, h, dbuf, L.OFF_G_MB,     L.WORDS_G_MB)
        zero_region(lib, h, dbuf, L.OFF_G_W_HEAD, L.WORDS_G_W_HEAD)
        zero_region(lib, h, dbuf, L.OFF_BS_STREAM, L.WORDS_BS_STREAM)

        # (4) BACKWARD m = L-1..0 (REVERSE).
        for m in range(L_SEQ - 1, -1, -1):
            run(lib, h, dbuf, 1, m, toks[m], 0)

        t_inner += time.time() - t_a
        inner_tokens += 2 * L_SEQ   # fwd L + bwd L positions

        # accuracy + preds16 from THIS step's forward (PRED written in mode 0).
        preds = read_region(lib, h, dbuf, L.OFF_PRED, L_SEQ)
        correct = sum(1 for m in range(L_SEQ) if preds[m] == tgts[m])
        acc = correct / float(L_SEQ)

        # (5) SignSGD: update weights, zero grads in-kernel.
        run(lib, h, dbuf, 2, SGD_LR, SGD_THR, 0)

        # read back the five weight regions for the checksum (post-update).
        shadow = {}
        for off, n in ((L.OFF_EMBED, L.WORDS_EMBED),
                       (L.OFF_MH, L.WORDS_MH),
                       (L.OFF_MP, L.WORDS_MP),
                       (L.OFF_MB, L.WORDS_MB),
                       (L.OFF_W_HEAD, L.WORDS_W_HEAD)):
            shadow[off] = read_region(lib, h, dbuf, off, n)
        # assemble a sparse view sufficient for weight_crc (only weight regions)
        wbuf = [0] * L.ML_BUFFER_WORDS
        for off, vals in shadow.items():
            wbuf[off:off + len(vals)] = vals
        wcrc = weight_crc(wbuf)

        preds16 = ",".join(str(int(preds[m])) for m in range(16))
        # %.6f / %llu to match the C printf formatting exactly.
        sys.stdout.write(
            '{"step":%d,"acc":%.6f,"wcrc":%d,"preds16":[%s]}\n'
            % (step, acc, wcrc, preds16))
        sys.stdout.flush()

    wall = time.time() - t0_wall
    tok_per_s = inner_tokens / t_inner if t_inner > 0 else 0.0
    sys.stderr.write(
        "silicon: %d steps, fwd+bwd inner %.3fs, total wall %.3fs, "
        "%.0f tok/s (%d token-positions)\n"
        % (STEPS, t_inner, wall, tok_per_s, inner_tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
