#!/usr/bin/env python3
"""pc_fabric_comms_sweep.py — what does fixing chip-to-chip comms buy the PC fabric?

Runs the SAME bubble-free pipeline (tikitorch.fabric) under different neighbour-message
transports and measures the effect on throughput. The point: the architecture is fixed;
only the transport changes — so the speedup is exactly the value of building a better
chip-to-chip link.

Grounded: block compute = 10.25 ms and the host-relay = 61% of wall-clock are MEASURED
(results/predictive_coding/multi_fpga/). The WORD profile is calibrated to reproduce that
61%; BURST/P2P/FABRIC are realistic transport params (PROJECTED). The pipeline overlap is
MEASURED in-sim (real threads). So this UPGRADES a throughput model that just asserts
"comms = 0" — here comms is explicit, calibrated, and you can see every number.

Run:  PYTHONPATH=. python3 scripts/pc_fabric_comms_sweep.py
"""
import numpy as np

from tikitorch.fabric import PipelineRunner, comms_fraction, make_thread_links
from tikitorch.fabric.cost import CAL_COMPUTE_S, CAL_MSG_BYTES, PROFILES
from tikitorch.fabric.stages import make_linear_pipeline

SEQ = 64                       # tokens per microbatch kick (current kernel)
COMPUTE_MS = CAL_COMPUTE_S * 1e3   # 10.25 ms, MEASURED per-block fwd+bwd
MSG = CAL_MSG_BYTES            # 8192 B neighbour message (D=64 seq=64 int16)


def pipelined_toks(profile, S=4, M=20, reps=3, double_buffer=False):
    """Measured-in-sim throughput of the bubble-free pipeline under `profile` (median of
    `reps` runs — thread scheduling adds run-to-run jitter, largest when compute-bound).
    double_buffer overlaps a message's transit with the NEXT compute."""
    dims = [8] + [16] * (S - 1) + [4]
    inputs, targets = _task(M, dims[0], dims[-1])
    walls = []
    for _ in range(reps):
        stages = make_linear_pipeline(dims, seed=1, compute_ms=COMPUTE_MS / 2.0)
        links = make_thread_links(S, transport=profile, msg_bytes=MSG,
                                  double_buffer=double_buffer)
        walls.append(PipelineRunner(stages, links, lr=0.05, window=M).run(inputs, targets).wall_s)
    wall = float(np.median(walls))
    return M * SEQ / wall, wall


def _task(n, d_in, d_out):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, d_in))
    Y = np.tanh(X @ rng.standard_normal((d_in, d_out)))
    return [X[k] for k in range(n)], [Y[k] for k in range(n)]


def main():
    print("tikitorch.fabric — chip-to-chip comms sweep (same pipeline, different transport)")
    print(f"  neighbour message: {MSG} B (D=64 seq=64 int16) | block compute: "
          f"{COMPUTE_MS:.2f} ms (MEASURED) | tok/microbatch: {SEQ}\n")
    word = PROFILES[0]
    word_tok, _ = pipelined_toks(word)
    hdr = (f"  {'transport':<38} {'comms/msg':>10} {'comms-frac':>10} "
           f"{'serial tok/s':>12} {'+dbl-buf':>9} {'vs word':>8}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for p in PROFILES:
        cms = p.time_for(MSG) * 1e3
        frac = comms_fraction(p, MSG, CAL_COMPUTE_S)
        tok, _ = pipelined_toks(p)
        tok_db, _ = pipelined_toks(p, double_buffer=True)
        print(f"  {p.name:<38} {cms:>8.2f}ms {frac*100:>9.1f}% {tok:>12.0f} {tok_db:>9.0f} {tok_db/word_tok:>7.2f}x")
    print("  serial tok/s = comms serialized w/ compute (today); +dbl-buf = transit overlaps the NEXT")
    print("  compute (rate -> 1/max(compute,comms)). Double-buffer helps most when comms ~ compute;")
    print("  once comms << compute (burst/P2P) the rate is compute-bound and double-buffer is moot.")
    # the transport fix is ONE lever; it multiplies with two other INDEPENDENTLY-measured ones.
    burst_tok, _ = pipelined_toks(PROFILES[1])
    print("\n  the levers (each measured on its own; the PRODUCT is a projection, not one run):")
    print(f"    [MEASURED-in-sim] transport fix (word->burst, same pipeline):  {burst_tok/word_tok:>4.1f}x  (removes the 61% relay)")
    print(f"    [MEASURED-in-sim] bubble-free pipeline overlap (8-stage):       5.6x  (pc_fabric_sim_demo.py)")
    print(f"    [MEASURED-silicon] linear data-parallel replication N=4:        4.0x  (morph-LM, MULTI_FPGA_SCALING)")
    print("    -> a chip-to-chip link is the load-bearing first lever: it is what turns the")
    print("       measured 25.8->4.0 steps/s relay-bound REGRESSION into compute-bound scaling.")
    print("\n  NOTE: this runner uses an all-forward-then-all-backward (GPipe-like) schedule, so a")
    print("  fwd->bwd drain remains; a true 1F1B schedule (next) interleaves them for the full")
    print("  bubble-free rate. The transport conclusion above is schedule-independent.\n")


if __name__ == "__main__":
    main()
