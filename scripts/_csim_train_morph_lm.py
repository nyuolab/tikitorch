"""Long CSIM training of the config-B morph LM kernel (quality trajectory).

Reuses the validated driver from _validate_kernel_morph_lm.py; logs windowed
train accuracy toward the Python strict-hard morph reference (~29.7% val acc
at 3.46M tokens, results/lm_shakespeare/strictG_hard_*.jsonl).

Usage:  PYTHONPATH=. python scripts/_csim_train_morph_lm.py --windows 3000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _validate_kernel_morph_lm import Kernel, L_SEQ, V, ZSHIFT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=3000)
    p.add_argument("--lr", type=int, default=1)
    p.add_argument("--threshold", type=int, default=2048)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "lm_shakespeare" / "csim_morph_lm.json")
    args = p.parse_args()

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
    for w in range(args.windows):
        s = int(rng.integers(0, len(ids) - L_SEQ - 1))
        toks, tgts = ids[s:s + L_SEQ], ids[s + 1:s + L_SEQ + 1]
        k.buf[o["OFF_TOKENS"]:o["OFF_TOKENS"] + L_SEQ] = toks
        k.buf[o["OFF_TARGETS"]:o["OFF_TARGETS"] + L_SEQ] = tgts
        k.zero_bs()
        for m in range(L_SEQ):
            k.kick(0, m, int(toks[m]), int(tgts[m]))
        pred = k.buf[o["OFF_PRED"]:o["OFF_PRED"] + L_SEQ]
        accs.append(float((pred == tgts).mean()))
        k.zero_bs()
        for m in range(L_SEQ - 1, -1, -1):
            k.kick(1, m, int(toks[m]), 0)
        k.kick(2, args.lr, args.threshold, 0)
        if (w + 1) % args.log_every == 0:
            print(f"window {w+1:5d}  acc(last {args.log_every}) "
                  f"{np.mean(accs[-args.log_every:]):.4f}  "
                  f"({(w+1)*L_SEQ/(time.time()-t0):.0f} tok/s)", flush=True)

    out = {"windows": args.windows, "lr": args.lr, "threshold": args.threshold,
           "tail_acc": float(np.mean(accs[-200:])), "acc_hist": accs}
    args.out.write_text(json.dumps(out))
    print(f"tail acc (last 200 windows): {out['tail_acc']:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
