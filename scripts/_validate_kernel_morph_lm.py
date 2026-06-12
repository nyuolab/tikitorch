"""Validation for the config-B MorphSSM LM kernel (forge_train_step_morph_lm).

1. Controlled-logits bit-exact CE-gradient test: zero ALL weights except
   W_HEAD with constant rows (W_head[v,:] = c_v). With zero embed the residual
   stream is zero end to end (batch_shift(0)=0; conv max/min over h=0 taps = 0;
   proj max(0+0)+0 = 0; residual 0), so logits[v] = c_v exactly. The kernel's
   LUT-CE gradient row must match the integer reference (same spec as
   scripts/_validate_kernel_lutce.py, post-truncation balance) bit for bit,
   sum to exactly zero, and the CFG=0 perceptron path must be 2-hot.

2. End-to-end training smoke: 20 windows of tiny-shakespeare chars (V=128
   clip), per-window fwd 0..L-1 / bwd L-1..0 (REVERSE: the grad_BS
   complete-by-read invariant) / signsgd(lr=1, threshold=2048), BS stream
   zeroed between fwd and bwd loops. Tail accuracy must exceed 3x chance.

Usage:  PYTHONPATH=. python scripts/_validate_kernel_morph_lm.py
"""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "_libtikifpga_morphlm.dylib"

V, D, L_SEQ, N_LAYERS, K, ZSHIFT = 128, 128, 128, 4, 7, 4
LUT = np.array([round(65536 * 2 ** (-i / 16)) for i in range(256)], dtype=np.int64)


def layout():
    """Python mirror of forge_buffer_morph_lm.hpp (order + arithmetic exact)."""
    w = {
        "EMBED": V * D, "MH": N_LAYERS * D * K, "MP": N_LAYERS * D * D,
        "MB": N_LAYERS * D, "W_HEAD": V * D,
    }
    order = ["EMBED", "MH", "MP", "MB", "W_HEAD"]
    off, cur = {}, 0
    for k in order:
        off["OFF_" + k] = cur
        cur += w[k]
    for k in order:
        off["OFF_G_" + k] = cur
        cur += w[k]
    for name, n in [("TAP_IDX", N_LAYERS * L_SEQ * D),
                    ("PROJ_IDX", N_LAYERS * L_SEQ * D),
                    ("AD_HEAD", L_SEQ * V),
                    ("BS_STREAM", N_LAYERS * L_SEQ * D),
                    ("TOKENS", L_SEQ), ("TARGETS", L_SEQ),
                    ("GRAD_LOGIT", L_SEQ * V), ("PRED", L_SEQ), ("CFG", 2)]:
        off["OFF_" + name] = cur
        off["W_" + name] = n
        cur += n
    off["WORDS"] = cur
    off["_w"] = w
    return off


def ref_grad(c, target, zshift=ZSHIFT):
    zneg = c.max() - c
    idx = np.minimum(zneg >> zshift, 255)
    e = LUT[idx]
    S = e.sum()
    sh = 0
    while (S >> sh) > 65536:
        sh += 1
    g = (e >> sh).astype(np.int64)
    g[target] -= g.sum()        # post-truncation sum: exactly balanced
    return g


class Kernel:
    def __init__(self):
        self.off = layout()
        self.lib = ctypes.CDLL(str(LIB))
        self.fn = self.lib.forge_train_step_morph_lm
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4
        self.buf = np.zeros(self.off["WORDS"], dtype=np.int32)
        self._p = self.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def kick(self, mode, a1, a2, a3):
        rc = self.fn(self._p, mode, a1, a2, a3)
        if rc != 0:
            raise RuntimeError(f"kick mode={mode} args=({a1},{a2},{a3}) rc={rc}")

    def zero_bs(self):
        o = self.off
        self.buf[o["OFF_BS_STREAM"]:o["OFF_BS_STREAM"] + o["W_BS_STREAM"]] = 0


def test_controlled_logits(n_cases=50):
    rng = np.random.default_rng(0)
    for case in range(n_cases):
        k = Kernel()
        o = k.off
        c = rng.integers(-200, 200, size=V).astype(np.int32)
        w_head = np.repeat(c[:, None], D, axis=1)
        k.buf[o["OFF_W_HEAD"]:o["OFF_W_HEAD"] + V * D] = w_head.reshape(-1)
        target = int(rng.integers(0, V))
        token = int(rng.integers(0, V))
        m = int(rng.integers(0, L_SEQ))
        k.buf[o["OFF_CFG"] + 0] = 1
        k.buf[o["OFF_CFG"] + 1] = ZSHIFT

        k.kick(0, m, token, target)
        got = k.buf[o["OFF_GRAD_LOGIT"] + m * V:o["OFF_GRAD_LOGIT"] + (m + 1) * V].astype(np.int64)
        want = ref_grad(c, target)
        assert got.sum() == 0, f"case {case}: row sum {got.sum()}"
        assert np.array_equal(got, want), f"case {case}: grad mismatch"
        assert int(k.buf[o["OFF_PRED"] + m]) == int(np.argmax(c))

        # perceptron path (CFG=0) must be 2-hot
        k.buf[o["OFF_CFG"] + 0] = 0
        k.kick(0, m, token, target)
        g2 = k.buf[o["OFF_GRAD_LOGIT"] + m * V:o["OFF_GRAD_LOGIT"] + (m + 1) * V]
        pred = int(np.argmax(c))
        if pred != target:
            assert g2[pred] == 1 and g2[target] == -1 and np.count_nonzero(g2) == 2
        else:
            assert np.count_nonzero(g2) == 0
    print(f"controlled-logits: {n_cases} cases bit-exact, rows sum to zero, "
          f"perceptron 2-hot OK")


def test_training_smoke(windows=20, lr=1, threshold=2048):
    text = (REPO / "data" / "tinyshakespeare.txt").read_text()
    ids = np.array([min(ord(ch), V - 1) for ch in text], dtype=np.int32)
    rng = np.random.default_rng(0)

    k = Kernel()
    o = k.off
    for name, n in o["_w"].items():
        k.buf[o["OFF_" + name]:o["OFF_" + name] + n] = \
            rng.integers(-4, 5, size=n).astype(np.int32)
    k.buf[o["OFF_CFG"] + 0] = 1
    k.buf[o["OFF_CFG"] + 1] = ZSHIFT

    accs = []
    t0 = time.time()
    for w in range(windows):
        s = int(rng.integers(0, len(ids) - L_SEQ - 1))
        toks, tgts = ids[s:s + L_SEQ], ids[s + 1:s + L_SEQ + 1]
        k.buf[o["OFF_TOKENS"]:o["OFF_TOKENS"] + L_SEQ] = toks
        k.buf[o["OFF_TARGETS"]:o["OFF_TARGETS"] + L_SEQ] = tgts
        k.zero_bs()
        for m in range(L_SEQ):
            k.kick(0, m, int(toks[m]), int(tgts[m]))
        pred = k.buf[o["OFF_PRED"]:o["OFF_PRED"] + L_SEQ]
        accs.append(float((pred == tgts).mean()))
        k.zero_bs()                              # stream -> grad_BS accumulator
        for m in range(L_SEQ - 1, -1, -1):       # REVERSE: complete-by-read
            k.kick(1, m, int(toks[m]), 0)
        k.kick(2, lr, threshold, 0)
    tail = float(np.mean(accs[-5:]))
    rate = windows * L_SEQ / (time.time() - t0)
    print(f"training smoke: acc {accs[0]:.4f} -> tail {tail:.4f} "
          f"({rate:.0f} tok/s)  curve={['%.3f' % a for a in accs]}")
    chance = 1.0 / V
    assert tail > 3 * chance, f"tail acc {tail:.4f} <= 3x chance {3*chance:.4f}"


if __name__ == "__main__":
    test_controlled_logits()
    test_training_smoke()
    print("OK: forge_train_step_morph_lm validated (CSIM).")
