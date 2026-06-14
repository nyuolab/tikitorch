#!/usr/bin/env python3
"""gen_buffer_init.py — SHARED ground-truth buffer initializer for the
config-B MorphSSM LM training kernel (forge_train_step_morph_lm).

This module is the single source of truth that BOTH the host CSIM golden
(golden_morph_lm.cpp) and the FPGA silicon driver (morph_lm_silicon.py)
consume. It builds the flat int32 buffer of length ML_BUFFER_WORDS exactly
matching forge_buffer_morph_lm.hpp, deterministically (fixed-seed LCG — NO
unseeded RNG), and writes the raw little-endian int32 bytes to a file. Both
consumers load THE SAME bytes, so any divergence in their per-step JSONL is a
hardware/schedule bug, not an init mismatch.

Layout (config B, all offsets/counts in int32 WORDS; verified against
forge_buffer_morph_lm.hpp lines 83-178):

    ML_D = ML_V = ML_L_SEQ = 128, ML_N_LAYERS = 4, ML_K = 7,
    ML_LOG2_D = 7, ML_DIL_CAP = 64.

Weight regions get small int16-range values (-4..4, matching the reference
validator's rng.integers(-4, 5)); the kernel reads each weight word as
(int16_t)(w & 0xFFFF), so the value lives in the low 16 bits. Grad regions are
zero. TOKENS/TARGETS hold a LEARNABLE next-char stream (ASCII of a fixed
repeated text, clipped <128), with targets[m] = tokens[m+1]. CFG selects the
balanced LUT cross-entropy path (loss_kind=1, zshift=4).

Usage:
    python3 gen_buffer_init.py /tmp/buffer_init.bin
"""
import os
import struct
import sys

# ---- Shape constants (config B) — mirror forge_buffer_morph_lm.hpp exactly ----
ML_D        = 128
ML_V        = 128
ML_L_SEQ    = 128
ML_N_LAYERS = 4
ML_K        = 7
ML_LOG2_D   = 7
ML_DIL_CAP  = 64

# ---- BATCH parallelism (mirror header TT_BATCH / ML_BATCH_N) ----
# ML_BATCH = number of INDEPENDENT sequences a fused step processes. MUST match
# the kernel build (-DTT_BATCH -DML_BATCH=N). Default 1 => the shipped (non-batch)
# layout, byte-identical to the AFI buffer. Override via env ML_BATCH (so the
# golden harness and this generator stay in lockstep) or the 3rd CLI arg.
ML_BATCH = int(os.environ.get("ML_BATCH", "1"))
assert ML_BATCH >= 1, "ML_BATCH must be >= 1"

# ---- Per-sequence region strides (one sequence's worth of words) ----
STRIDE_TAP_IDX    = ML_N_LAYERS * ML_L_SEQ * ML_D   # 65536
STRIDE_PROJ_IDX   = ML_N_LAYERS * ML_L_SEQ * ML_D   # 65536
STRIDE_AD_HEAD    = ML_L_SEQ * ML_V                 # 16384
STRIDE_BS_STREAM  = ML_N_LAYERS * ML_L_SEQ * ML_D   # 65536
STRIDE_TOKENS     = ML_L_SEQ                        # 128
STRIDE_TARGETS    = ML_L_SEQ                        # 128
STRIDE_GRAD_LOGIT = ML_L_SEQ * ML_V                 # 16384
STRIDE_PRED       = ML_L_SEQ                        # 128

# ---- Per-tensor word counts (per-seq regions grown by ML_BATCH) ----
WORDS_EMBED    = ML_V * ML_D                 # 16384
WORDS_MH       = ML_N_LAYERS * ML_D * ML_K   # 3584
WORDS_MP       = ML_N_LAYERS * ML_D * ML_D   # 65536
WORDS_MB       = ML_N_LAYERS * ML_D          # 512
WORDS_W_HEAD   = ML_V * ML_D                 # 16384
WORDS_G_EMBED  = WORDS_EMBED
WORDS_G_MH     = WORDS_MH
WORDS_G_MP     = WORDS_MP
WORDS_G_MB     = WORDS_MB
WORDS_G_W_HEAD = WORDS_W_HEAD
WORDS_TAP_IDX  = ML_BATCH * STRIDE_TAP_IDX
WORDS_PROJ_IDX = ML_BATCH * STRIDE_PROJ_IDX
WORDS_AD_HEAD  = ML_BATCH * STRIDE_AD_HEAD
WORDS_BS_STREAM = ML_BATCH * STRIDE_BS_STREAM
WORDS_TOKENS     = ML_BATCH * STRIDE_TOKENS
WORDS_TARGETS    = ML_BATCH * STRIDE_TARGETS
WORDS_GRAD_LOGIT = ML_BATCH * STRIDE_GRAD_LOGIT
WORDS_PRED       = ML_BATCH * STRIDE_PRED
WORDS_CFG        = 2
WORDS_CE_LUT     = 256   # host-staged exp2 table (FT_CE_EXP2_Q16)

# ---- Cumulative offsets (word indices) ----
OFF_EMBED      = 0
OFF_MH         = OFF_EMBED      + WORDS_EMBED
OFF_MP         = OFF_MH         + WORDS_MH
OFF_MB         = OFF_MP         + WORDS_MP
OFF_W_HEAD     = OFF_MB         + WORDS_MB
OFF_G_EMBED    = OFF_W_HEAD     + WORDS_W_HEAD
OFF_G_MH       = OFF_G_EMBED    + WORDS_G_EMBED
OFF_G_MP       = OFF_G_MH       + WORDS_G_MH
OFF_G_MB       = OFF_G_MP       + WORDS_G_MP
OFF_G_W_HEAD   = OFF_G_MB       + WORDS_G_MB
OFF_TAP_IDX    = OFF_G_W_HEAD   + WORDS_G_W_HEAD
OFF_PROJ_IDX   = OFF_TAP_IDX    + WORDS_TAP_IDX
OFF_AD_HEAD    = OFF_PROJ_IDX   + WORDS_PROJ_IDX
OFF_BS_STREAM  = OFF_AD_HEAD    + WORDS_AD_HEAD
OFF_TOKENS     = OFF_BS_STREAM  + WORDS_BS_STREAM
OFF_TARGETS    = OFF_TOKENS     + WORDS_TOKENS
OFF_GRAD_LOGIT = OFF_TARGETS    + WORDS_TARGETS
OFF_PRED       = OFF_GRAD_LOGIT + WORDS_GRAD_LOGIT
OFF_CFG        = OFF_PRED       + WORDS_PRED
# 16-word aligned (the kernel's wide burst_load truncates the offset to /16; a
# non-aligned base reads the LUT shifted on silicon — see forge_buffer_morph_lm.hpp)
OFF_CE_LUT     = ((OFF_CFG + WORDS_CFG + 15) // 16) * 16

ML_BUFFER_WORDS = OFF_CE_LUT + WORDS_CE_LUT
ML_BUFFER_BYTES = ML_BUFFER_WORDS * 4

# Per-sequence offset helpers (mirror off_tokens_b / off_targets_b in the header).
def off_tokens_b(b):  return OFF_TOKENS  + b * STRIDE_TOKENS
def off_targets_b(b): return OFF_TARGETS + b * STRIDE_TARGETS

# ---- CFG: balanced LUT cross-entropy path (matches the reference validator) ----
CFG_LOSS_KIND = 1   # 1 = balanced LUT-CE; 0 = 2-hot perceptron
CFG_CE_ZSHIFT = 4   # LUT index = (zmax - logit) >> zshift  (only read when CE)

# ---- Learnable next-char training stream ----
# A fixed repeated text so the same character context recurs and next-char
# prediction is genuinely learnable (low-entropy, strongly auto-correlated).
TRAIN_TEXT = "to be or not to be that is the question "


class _LCG:
    """Deterministic 32-bit linear congruential generator (Numerical Recipes
    constants). Fixed seed => byte-identical init every run, on every machine,
    in any language that reproduces these constants. NO unseeded RNG."""
    __slots__ = ("s",)

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def next_u32(self):
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        """Uniform int in [lo, hi] inclusive, low-bias-free enough for tiny
        ranges; deterministic."""
        span = hi - lo + 1
        return lo + (self.next_u32() % span)


def build_init(seed=0):
    """Return a flat list of ML_BUFFER_WORDS signed int32 values, fully
    deterministic. Weight regions: small int16-range (-4..4). Grads: 0. Saved
    indices / streams / I-O scratch: 0. TOKENS/TARGETS: the learnable next-char
    stream. CFG: LUT-CE path."""
    buf = [0] * ML_BUFFER_WORDS
    rng = _LCG(seed)

    # Small symmetric weights in [-4, 4] for every weight region. These land in
    # the low 16 bits; the kernel reads (int16_t)(w & 0xFFFF), and small +ve
    # values are unchanged. Negative values are stored as the natural Python
    # int (e.g. -3); both golden and silicon write the SAME int32 word, and the
    # kernel's sign-extend of the low 16 bits recovers -3 identically.
    def fill_weight(off, n):
        for i in range(n):
            buf[off + i] = rng.randint(-4, 4)

    fill_weight(OFF_EMBED,  WORDS_EMBED)
    fill_weight(OFF_MH,     WORDS_MH)
    fill_weight(OFF_MP,     WORDS_MP)
    fill_weight(OFF_MB,     WORDS_MB)
    fill_weight(OFF_W_HEAD, WORDS_W_HEAD)

    # Grad regions, saved-index regions, BS stream, GRAD_LOGIT, PRED, AD_HEAD
    # all stay zero (already 0). The host schedule zeroes BS stream + grads at
    # the proper points anyway; zero-init here keeps the very first step clean.

    # Learnable next-char stream(s). For each of the ML_BATCH sequences b, take a
    # DISTINCT shifted slice of the repeated training text so the sequences differ
    # (each b sees a different phase of the same low-entropy auto-correlated text,
    # so each is genuinely learnable AND distinct => the per-sequence grads differ,
    # which is what makes the summed-grad bit-exactness check meaningful). Within a
    # sequence: tokens[m] = char (b_shift+m), targets[m] = tokens[m+1] (next-char).
    # Sequence b lives at OFF_TOKENS + b*STRIDE_TOKENS / OFF_TARGETS + b*STRIDE.
    T = len(TRAIN_TEXT)
    for b in range(ML_BATCH):
        shift = (b * 7) % T   # coprime-ish phase per sequence (T=40); distinct b
        ids = [min(ord(TRAIN_TEXT[(shift + i) % T]), ML_V - 1)
               for i in range(ML_L_SEQ + 1)]
        ot = off_tokens_b(b)
        og = off_targets_b(b)
        for m in range(ML_L_SEQ):
            buf[ot + m] = ids[m]
            buf[og + m] = ids[m + 1]   # next-char target

    # CFG (read by do_fwd_loss): balanced LUT cross-entropy.
    buf[OFF_CFG + 0] = CFG_LOSS_KIND
    buf[OFF_CFG + 1] = CFG_CE_ZSHIFT

    # exp2 table for the balanced CE gradient, host-staged at OFF_CE_LUT so the
    # kernel does not depend on a synthesized-ROM init (which read all-zero on
    # the F2 AFI). FT_CE_EXP2_Q16[i] = round(65536 * 2^(-i/16)) -- verified
    # bit-identical to forge_ce_lut.hpp for all 256 entries.
    for i in range(WORDS_CE_LUT):
        buf[OFF_CE_LUT + i] = round(65536 * 2.0 ** (-i / 16.0))

    return buf


def write_bin(path, buf):
    """Write the buffer as raw little-endian signed int32 to `path`."""
    # struct pack as '<i' (signed) so negative weights round-trip exactly.
    packed = struct.pack("<%di" % len(buf), *buf)
    with open(path, "wb") as f:
        f.write(packed)
    return len(packed)


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: gen_buffer_init.py <out.bin> [seed]\n")
        return 2
    out = argv[1]
    seed = int(argv[2]) if len(argv) > 2 else 0
    buf = build_init(seed)
    assert len(buf) == ML_BUFFER_WORDS, (len(buf), ML_BUFFER_WORDS)
    nbytes = write_bin(out, buf)
    assert nbytes == ML_BUFFER_BYTES, (nbytes, ML_BUFFER_BYTES)
    sys.stderr.write(
        "wrote %s: %d words, %d bytes (ML_BUFFER_WORDS=%d) ML_BATCH=%d seed=%d\n"
        % (out, len(buf), nbytes, ML_BUFFER_WORDS, ML_BATCH, seed))
    sys.stderr.write(
        "  OFF_TOKENS=%d OFF_TARGETS=%d OFF_CFG=%d OFF_CE_LUT=%d (loss_kind=%d zshift=%d)\n"
        % (OFF_TOKENS, OFF_TARGETS, OFF_CFG, OFF_CE_LUT, CFG_LOSS_KIND, CFG_CE_ZSHIFT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
