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

# ---- Per-tensor word counts ----
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
WORDS_TAP_IDX  = ML_N_LAYERS * ML_L_SEQ * ML_D   # 65536
WORDS_PROJ_IDX = ML_N_LAYERS * ML_L_SEQ * ML_D   # 65536
WORDS_AD_HEAD  = ML_L_SEQ * ML_V                 # 16384
WORDS_BS_STREAM = ML_N_LAYERS * ML_L_SEQ * ML_D  # 65536
WORDS_TOKENS     = ML_L_SEQ                   # 128
WORDS_TARGETS    = ML_L_SEQ                   # 128
WORDS_GRAD_LOGIT = ML_L_SEQ * ML_V            # 16384
WORDS_PRED       = ML_L_SEQ                   # 128
WORDS_CFG        = 2
WORDS_CE_LUT     = 256   # host-staged exp2 table (FT_CE_EXP2_Q16)

# ---- Cumulative offsets (word indices) ----
OFF_EMBED      = 0
OFF_MH         = OFF_EMBED      + WORDS_EMBED       # 16384
OFF_MP         = OFF_MH         + WORDS_MH          # 19968
OFF_MB         = OFF_MP         + WORDS_MP          # 85504
OFF_W_HEAD     = OFF_MB         + WORDS_MB          # 86016
OFF_G_EMBED    = OFF_W_HEAD     + WORDS_W_HEAD      # 102400
OFF_G_MH       = OFF_G_EMBED    + WORDS_G_EMBED     # 118784
OFF_G_MP       = OFF_G_MH       + WORDS_G_MH        # 122368
OFF_G_MB       = OFF_G_MP       + WORDS_G_MP        # 187904
OFF_G_W_HEAD   = OFF_G_MB       + WORDS_G_MB        # 188416
OFF_TAP_IDX    = OFF_G_W_HEAD   + WORDS_G_W_HEAD    # 204800
OFF_PROJ_IDX   = OFF_TAP_IDX    + WORDS_TAP_IDX     # 270336
OFF_AD_HEAD    = OFF_PROJ_IDX   + WORDS_PROJ_IDX    # 335872
OFF_BS_STREAM  = OFF_AD_HEAD    + WORDS_AD_HEAD     # 352256
OFF_TOKENS     = OFF_BS_STREAM  + WORDS_BS_STREAM   # 417792
OFF_TARGETS    = OFF_TOKENS     + WORDS_TOKENS      # 417920
OFF_GRAD_LOGIT = OFF_TARGETS    + WORDS_TARGETS     # 418048
OFF_PRED       = OFF_GRAD_LOGIT + WORDS_GRAD_LOGIT  # 434432
OFF_CFG        = OFF_PRED       + WORDS_PRED        # 434560
OFF_CE_LUT     = OFF_CFG        + WORDS_CFG         # 434562

ML_BUFFER_WORDS = OFF_CE_LUT + WORDS_CE_LUT        # 434818
ML_BUFFER_BYTES = ML_BUFFER_WORDS * 4             # 1739272

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

    # Learnable next-char stream. tokens[m] = char m of the repeated text
    # (clipped < ML_V); targets[m] = tokens[m+1] (next-char), with the last
    # target set to tokens[0] (wrap) so it is in-range and deterministic.
    ids = [min(ord(TRAIN_TEXT[i % len(TRAIN_TEXT)]), ML_V - 1)
           for i in range(ML_L_SEQ + 1)]
    for m in range(ML_L_SEQ):
        buf[OFF_TOKENS  + m] = ids[m]
        buf[OFF_TARGETS + m] = ids[m + 1]   # next-char target

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
        "wrote %s: %d words, %d bytes (ML_BUFFER_WORDS=%d) seed=%d\n"
        % (out, len(buf), nbytes, ML_BUFFER_WORDS, seed))
    sys.stderr.write(
        "  OFF_TOKENS=%d OFF_TARGETS=%d OFF_CFG=%d (loss_kind=%d zshift=%d)\n"
        % (OFF_TOKENS, OFF_TARGETS, OFF_CFG, CFG_LOSS_KIND, CFG_CE_ZSHIFT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
