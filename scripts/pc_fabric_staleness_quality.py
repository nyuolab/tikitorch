#!/usr/bin/env python3
"""pc_fabric_staleness_quality.py — does the 1F1B pipeline's depth-proportional staleness
actually HURT training, and how much does the accumulation window W buy back?

This is the de-risk the audit asked for: a throughput model that assumes "PC bubble = 0"
must also pay the grad-accumulation cost the staleness needs. Here we MEASURE it.

Method (isolates staleness from the SGD-vs-batch confound):
  for each window W, train the SAME deep pipeline two ways for the same epochs/lr/init:
    CONTROL : synchronous windowed SGD (run_serial, window=W)  -> applies every W, ZERO staleness
    1F1B    : the pipeline at window=W                          -> same update freq, WITH staleness
  the final-loss GAP is the pure staleness penalty. W=M -> both are full-batch -> identical.

Run:  PYTHONPATH=. python3 scripts/pc_fabric_staleness_quality.py
"""
import numpy as np

from tikitorch.fabric import PipelineRunner, make_thread_links, run_serial
from tikitorch.fabric.stages import make_linear_pipeline

S, M, EPOCHS, LR, SEEDS = 8, 24, 60, 0.02, range(6)
DIMS = [6] + [16] * (S - 1) + [4]


def task(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((M, DIMS[0]))
    W1 = rng.standard_normal((DIMS[0], 12)); W2 = rng.standard_normal((12, DIMS[-1]))
    Y = np.tanh(np.tanh(X @ W1) @ W2)        # a 2-layer nonlinear target -> needs the depth
    return [X[k] for k in range(M)], [Y[k] for k in range(M)]


def train(stages, inputs, targets, W, kind):
    last = None
    for _ in range(EPOCHS):
        if kind == "control":
            last, _ = run_serial(stages, inputs, targets, lr=LR, window=W)
            last = float(np.mean(last))
        else:
            last = PipelineRunner(stages, make_thread_links(S), lr=LR, window=W,
                                  schedule="1f1b").run(inputs, targets).mean_loss
    return last


def main():
    print("tikitorch.fabric — staleness -> training-quality (1F1B vs same-W synchronous control)")
    print(f"  depth S={S}, microbatches M={M}, epochs={EPOCHS}, lr={LR}, seeds={len(list(SEEDS))} "
          f"(penalty = mean over seeds of 1F1B_loss / control_loss)\n")
    print(f"  {'W':>4} | {'max stale':>9} | {'ctrl loss':>10} | {'1F1B loss':>10} | {'penalty':>9}")
    print("  " + "-" * 56)
    for W in [1, 2, 4, S - 1, M]:
        cs, ps, stale = [], [], 0
        for seed in SEEDS:
            inputs, targets = task(seed)
            stale = PipelineRunner(make_linear_pipeline(DIMS, seed=seed), make_thread_links(S),
                                   lr=LR, window=W, schedule="1f1b").run(inputs, targets).max_staleness
            cs.append(train(make_linear_pipeline(DIMS, seed=seed), inputs, targets, W, "control"))
            ps.append(train(make_linear_pipeline(DIMS, seed=seed), inputs, targets, W, "1f1b"))
        c, p = float(np.mean(cs)), float(np.mean(ps))
        pen = float(np.mean([pp / cc for pp, cc in zip(ps, cs) if cc > 0]))
        print(f"  {W:>4} | {stale:>9} | {c:>10.5f} | {p:>10.5f} | {pen:>8.2f}x")
    print("\n  read: penalty > 1 = staleness HURTS final loss at that W; -> ~1.0x as W grows to")
    print("  cover the in-flight depth (W >= ~S-1). This is the measured cost a 'bubble-free'")
    print("  throughput model must pay: the overlap is real, but it needs a W-deep accumulation —")
    print("  both now measurable on CPU, before the silicon driver, with one knob (W).\n")


if __name__ == "__main__":
    main()
