#!/usr/bin/env python3
"""pc_fabric_sim_demo.py — demonstrate the tikitorch.fabric runtime on the CPU backend.

Two measurements, both on real threads against the NeighborLink interface (the same
logic that will drive silicon):

  1. OVERLAP  — a deep pipeline keeps every stage busy (the bubble PC removes). We give
     each block a simulated per-block wall-time and measure serial vs pipelined wall:
     the pipeline approaches  M*S / (M+S-1)  speedup.

  2. STALENESS vs W — the grad-accumulation window bounds the version gap between a
     microbatch's forward and its backward. W = M (one window over the sweep) => 0
     staleness and bit-exact to the synchronous baseline; smaller W trades a fuller
     pipeline for more staleness. The runtime MEASURES it, so W is tuned against a
     number, not a hope (silicon-measured: weight-staleness tolerates only ~1 step).

Run:  PYTHONPATH=. python3 scripts/pc_fabric_sim_demo.py
"""
import numpy as np

from tikitorch.fabric import PipelineRunner, make_thread_links, run_serial
from tikitorch.fabric.stages import make_linear_pipeline


def task(n, d_in, d_out, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))
    Y = np.tanh(X @ rng.standard_normal((d_in, d_out)))
    return [X[k] for k in range(n)], [Y[k] for k in range(n)]


def part1_overlap():
    print("\n=== 1. OVERLAP (deep pipeline, simulated 3ms/block) ===")
    S, M, ms = 8, 16, 3.0
    dims = [6] + [16] * (S - 1) + [4]
    inputs, targets = task(M, dims[0], dims[-1])

    serial_losses, serial_wall = run_serial(
        make_linear_pipeline(dims, seed=1, compute_ms=ms), inputs, targets, lr=0.05)
    res = PipelineRunner(make_linear_pipeline(dims, seed=1, compute_ms=ms),
                         make_thread_links(S), lr=0.05, window=M).run(inputs, targets)

    ideal = M * S / (M + S - 1)
    print(f"  stages={S} microbatches={M}")
    print(f"  serial wall   : {serial_wall*1e3:7.1f} ms")
    print(f"  pipelined wall: {res.wall_s*1e3:7.1f} ms   (steps/s={res.steps_per_s:.1f})")
    print(f"  overlap speedup: {serial_wall/res.wall_s:5.2f}x   (ideal {ideal:.2f}x for this depth)")
    by = res.link_stats[0]["bytes_down"]
    print(f"  neighbour traffic: {by} B activations down @ stage0, errors up @ last "
          f"(point-to-point, no all-reduce)")
    assert np.allclose(res.losses, serial_losses, atol=1e-9)  # same math, just overlapped


def part2_staleness():
    print("\n=== 2. STALENESS vs accumulation window W ===")
    S, M = 6, 24
    dims = [5] + [10] * (S - 1) + [3]
    inputs, targets = task(M, dims[0], dims[-1], seed=9)

    ref_losses, _ = run_serial(make_linear_pipeline(dims, seed=2), inputs, targets, lr=0.05)
    print(f"  pipeline depth S={S}, microbatches M={M}")
    print(f"  {'W':>4} | {'max stale':>9} | {'mean stale':>10} | {'weights == serial?':>18}")
    print("  " + "-" * 50)
    for W in [1, 2, 3, 6, 12, 24]:
        stages = make_linear_pipeline(dims, seed=2)
        ref_stages = make_linear_pipeline(dims, seed=2)
        run_serial(ref_stages, inputs, targets, lr=0.05)        # the synchronous weight update
        res = PipelineRunner(stages, make_thread_links(S), lr=0.05, window=W).run(inputs, targets)
        same = all(np.allclose(a.W, b.W, atol=1e-9) for a, b in zip(stages, ref_stages))
        print(f"  {W:>4} | {res.max_staleness:>9} | {res.mean_staleness:>10.2f} | "
              f"{('yes' if same else 'no'):>18}")
    print("  -> W=M closes the window over the whole sweep: zero staleness, and the UPDATED")
    print("     weights are bit-exact to synchronous backprop. Smaller W = fuller pipeline but")
    print("     a real version gap (this GPipe-like schedule's gap ~ M/W; true 1F1B is ~depth/W).")
    print("     Quality impact of that gap is the next experiment (1F1B + a real PC block);")
    print("     silicon-measured today: weight-staleness tolerates ~1 step (stale-8 = +2.0 bpb).")


if __name__ == "__main__":
    print("tikitorch.fabric — distributed predictive-coding runtime (CPU ThreadLink backend)")
    part1_overlap()
    part2_staleness()
    print("\nSame PipelineRunner logic runs on F2 by swapping ThreadLink -> HostRelayLink "
          "(existing AGFI) / PcieP2PLink, turning these into silicon-measured numbers.\n")
