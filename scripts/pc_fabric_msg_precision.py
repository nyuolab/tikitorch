#!/usr/bin/env python3
"""pc_fabric_msg_precision.py — does the inter-chip MESSAGE precision cause the silicon plateau?

The silicon distributed run plateaus at bpb ~4.0 while the single-process CPU golden (same
config) reaches ~3.65 and keeps falling. The leading hypothesis: the distributed sharding
rescales the inter-chip neighbour messages (activation down, error up) to a lossy format
(int16 / bf16) — an approximation the single-process golden does not have.

This isolates it: train the SAME multi-block transformer char-LM on the fabric pipeline, varying
ONLY the inter-chip message encoding (exact float vs int16-per-tensor vs bf16). If a quantized
message plateaus the loss, the message precision is the culprit (fix = wider / better-scaled
encoding). If all three match, the plateau is elsewhere (depth, the dyadic forward, the optimizer).

Run:  PYTHONPATH=. python3 scripts/pc_fabric_msg_precision.py
"""
import os

import numpy as np

from tikitorch.fabric import PipelineRunner, make_thread_links
from tikitorch.fabric.transformer_stage import make_charlm_pipeline

SEQ, D, NBLOCKS, LR, EPOCHS = 24, 32, 3, 0.03, 120


def quant_int16(x):
    """Per-tensor max-scaled int16 (the silicon's scale-invariant error rescale; 15-bit)."""
    m = float(np.abs(x).max())
    if m == 0.0:
        return x.copy()
    s = m / 32767.0
    return np.round(x / s) * s


def quant_bf16(x):
    """Round to bfloat16 (7-bit mantissa) — the MP_GRIN/GROUT message format."""
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x8000) & np.uint32(0xFFFF0000)         # round-to-nearest to bf16
    return u.view(np.float32).astype(np.float64)


def load_text():
    p = os.path.expanduser("~/.cache/tiur/tinyshakespeare.txt")
    if os.path.exists(p):
        return open(p).read()[:3000]                  # a real slice
    return ("to be or not to be that is the question whether tis nobler in the mind " * 30)


def data():
    text = load_text()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
    toks = np.array([stoi[c] for c in text])
    M = (len(toks) - 1) // SEQ
    inp = [toks[k * SEQ:(k + 1) * SEQ] for k in range(M)]
    tgt = [toks[k * SEQ + 1:(k + 1) * SEQ + 1] for k in range(M)]
    return inp, tgt, V, M


def train(quantize, inp, tgt, V, M, S):
    stages = make_charlm_pipeline(V, D, SEQ, NBLOCKS, seed=0)   # identical init each run
    last = None
    for _ in range(EPOCHS):
        last = PipelineRunner(stages, make_thread_links(S, quantize=quantize),
                              lr=LR, window=M, schedule="1f1b").run(inp, tgt).mean_loss
    return last


def main():
    inp, tgt, V, M = data()
    S = NBLOCKS + 2
    print("tikitorch.fabric — inter-chip MESSAGE precision vs training quality (char-LM, NBLK=3)")
    print(f"  TinyShakespeare slice: vocab={V} seq={SEQ} microbatches={M}, depth S={S}, "
          f"epochs={EPOCHS}, lr={LR}\n")
    modes = [("exact float (single-process golden)", None),
             ("int16 per-tensor (15-bit rescale)", quant_int16),
             ("bf16 (7-bit, MP_GRIN/GROUT format)", quant_bf16)]
    base = None
    print(f"  {'inter-chip message encoding':<40} {'final bpb':>10} {'vs float':>9}")
    print("  " + "-" * 62)
    for name, q in modes:
        bpb = train(q, inp, tgt, V, M, S)
        if base is None:
            base = bpb
        print(f"  {name:<40} {bpb:>10.4f} {('--' if q is None else f'{bpb-base:+.4f}'):>9}")
    print("\n  read: a quantized message that ADDS bpb vs exact float = the message precision is")
    print("  costing quality (the silicon-vs-golden gap). If all ~equal, the plateau is elsewhere")
    print("  (NBLK depth, the dyadic forward, or the on-chip optimizer) -> next isolation target.\n")


if __name__ == "__main__":
    main()
