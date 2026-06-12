"""CSIM end-to-end training of the 1-layer LM kernel: LUT-CE vs perceptron.

Drives the REAL kernel (CSIM build, _libtikifpga.dylib) through the same
mode sequence the silicon driver uses — per window: stage tokens/targets,
zero K/V caches, FWD_LOSS per position, zero caches (-> grad accumulators),
BWD in reverse, one SIGNSGD kick — training from random small-int weights on
tiny-shakespeare chars (clipped to V=128). The only variable is the on-chip
loss: CFG_LOSS_KIND 0 (2-hot perceptron, silicon baseline) vs 1 (balanced
LUT cross-entropy, this round's kernel feature).

Accuracy is read from OFF_PRED (the runner's grad-row==0 proxy is
perceptron-only). LUT-CE gradients are Q16, so the SignSGD threshold is
swept first.

Usage:  PYTHONPATH=. python scripts/_csim_train_lutce.py [--windows 300]
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tikitorch.fpga.runners import _L1  # noqa: E402

LIB = REPO / "_libtikifpga.dylib"
V, D, L_MAX, ZSHIFT = 128, 128, 128, 4


class Kernel:
    def __init__(self):
        self.off = _L1.offsets()
        self.lib = ctypes.CDLL(str(LIB))
        self.fn = self.lib.forge_train_step
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_int32)] + [ctypes.c_int] * 4
        self.buf = np.zeros(self.off["FT_BUFFER_WORDS"], dtype=np.int32)
        self._p = self.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def kick(self, mode, a1, a2, a3):
        rc = self.fn(self._p, mode, a1, a2, a3)
        if rc < -90:
            raise RuntimeError(f"kick mode={mode} rc={rc}")
        return rc

    def seed_weights(self, rng, scale=4):
        off = self.off
        for name, n in off["_weight_regions"]:
            self.buf[off["OFF_" + name]:off["OFF_" + name] + n] = \
                rng.integers(-scale, scale + 1, size=n).astype(np.int32)

    def zero_caches(self):
        off = self.off
        for r in ("K_CACHE", "V_CACHE"):
            self.buf[off["OFF_" + r]:off["OFF_" + r] + off["W_" + r]] = 0


def train(loss_kind, threshold, lr, windows, seed=0, log_every=20):
    rng = np.random.default_rng(seed)
    text = (REPO / "data" / "tinyshakespeare.txt").read_text()
    ids = np.array([min(ord(c), V - 1) for c in text], dtype=np.int32)

    k = Kernel()
    off = k.off
    k.seed_weights(rng)
    k.buf[off["OFF_CFG"] + 0] = loss_kind
    k.buf[off["OFF_CFG"] + 1] = ZSHIFT

    hist = []
    t0 = time.time()
    for w in range(windows):
        s = int(rng.integers(0, len(ids) - L_MAX - 1))
        toks = ids[s:s + L_MAX]
        tgts = ids[s + 1:s + L_MAX + 1]
        k.buf[off["OFF_TOKENS"]:off["OFF_TOKENS"] + L_MAX] = toks
        k.buf[off["OFF_TARGETS"]:off["OFF_TARGETS"] + L_MAX] = tgts
        k.zero_caches()
        for m in range(L_MAX):
            k.kick(0, m, int(toks[m]), int(tgts[m]))
        pred = k.buf[off["OFF_PRED"]:off["OFF_PRED"] + L_MAX]
        acc = float((pred == tgts).mean())
        k.zero_caches()                      # caches double as grad accumulators
        for m in range(L_MAX - 1, -1, -1):
            k.kick(1, m, int(toks[m]), 0)
        k.kick(2, lr, threshold, 0)
        hist.append(acc)
        if (w + 1) % log_every == 0:
            recent = float(np.mean(hist[-log_every:]))
            print(f"  [{ 'lut' if loss_kind else 'perc' } thr={threshold}] "
                  f"window {w+1:4d}  acc(last {log_every}) {recent:.4f}  "
                  f"({(w+1)*L_MAX/(time.time()-t0):.0f} tok/s)", flush=True)
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=300)
    p.add_argument("--sweep-windows", type=int, default=60)
    p.add_argument("--lr", type=int, default=1)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "lm_shakespeare" / "csim_lutce.json")
    args = p.parse_args()

    results = {}
    print("== threshold sweep (LUT-CE, Q16 grads) ==")
    best_thr, best_acc = 0, -1.0
    for thr in (0, 256, 2048, 8192, 32768):
        hist = train(1, thr, args.lr, args.sweep_windows)
        tail = float(np.mean(hist[-20:]))
        results[f"sweep_lut_thr{thr}"] = tail
        print(f"  threshold {thr:6d}: tail acc {tail:.4f}")
        if tail > best_acc:
            best_acc, best_thr = tail, thr

    print(f"\n== full runs ({args.windows} windows, lut thr={best_thr}) ==")
    hist_lut = train(1, best_thr, args.lr, args.windows)
    hist_perc = train(0, 0, args.lr, args.windows)
    results["lut_thr"] = best_thr
    results["lut_tail_acc"] = float(np.mean(hist_lut[-50:]))
    results["perc_tail_acc"] = float(np.mean(hist_perc[-50:]))
    results["lut_hist"] = hist_lut
    results["perc_hist"] = hist_perc
    args.out.write_text(json.dumps(results))
    print(f"\nLUT-CE  tail acc: {results['lut_tail_acc']:.4f}")
    print(f"percep  tail acc: {results['perc_tail_acc']:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
