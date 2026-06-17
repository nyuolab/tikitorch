"""muzero_scale_cost — cost the H2H-vs-scale learning-curve experiment from MEASURED throughput.

Self-play is embarrassingly parallel (one game per core), so total $ ~= core-seconds x $/core-hr, roughly
independent of instance size. Numbers below are from scripts/muzero_scale_bench.py (Connect4, AZ-tree,
100 sims, measured 2026-06-17).
"""
# --- measured per-CORE rates (8 workers on a 16-core box) ---
MOVES_PER_CORE_S = {"tropical": 79.1 / 8, "dense": 280.5 / 8}      # 9.89 / 35.06
TRAIN_STEPS_PER_S = {"tropical": 27.0, "dense": 138.6}             # single-stream
MOVES_PER_GAME = 18                                                # Connect4 avg (16-20)
SIMS = 100

# --- experiment design ---
GAMES_PER_ITER = 16
TRAIN_STEPS_PER_ITER = 40
SEEDS = 3
SUBSTRATES = ["tropical", "dense"]
H2H_GAMES = 60                                                     # trained-vs-untrained eval per run, x2 agents
SCALE = {"1x": 224, "3x": 672, "10x": 2240, "30x": 6720, "100x": 22400}   # total self-play games

# --- cloud pricing (AWS c7i CPU-optimized, us-east-1, approx) ---
USD_PER_VCPU_HR_OD = 0.045
USD_PER_VCPU_HR_SPOT = 0.015


def run_core_seconds(net, games):
    iters = max(1, games // GAMES_PER_ITER)
    selfplay = games * MOVES_PER_GAME / MOVES_PER_CORE_S[net]                 # core-s (parallel)
    train = iters * TRAIN_STEPS_PER_ITER / TRAIN_STEPS_PER_S[net]            # core-s (single-stream)
    h2h = H2H_GAMES * MOVES_PER_GAME * 2 / MOVES_PER_CORE_S[net]             # core-s (trained vs untrained)
    return selfplay + train + h2h


def main():
    print(f"H2H-vs-scale learning curve: {len(SCALE)} points x {len(SUBSTRATES)} substrates x {SEEDS} seeds "
          f"= {len(SCALE)*len(SUBSTRATES)*SEEDS} runs; metric = head-to-head (trained vs untrained self).\n")
    grand = 0.0
    for tag, games in SCALE.items():
        pt = 0.0
        for net in SUBSTRATES:
            pt += run_core_seconds(net, games) * SEEDS
        grand += pt
        ch = pt / 3600
        print(f"  {tag:>5s} ({games:6d} games): {ch:7.2f} core-hr  "
              f"-> ${ch*USD_PER_VCPU_HR_OD:7.2f} on-demand / ${ch*USD_PER_VCPU_HR_SPOT:6.2f} spot")
    gch = grand / 3600
    print(f"\n  TOTAL 100x curve: {gch:.1f} core-hr  -> ${gch*USD_PER_VCPU_HR_OD:.0f} on-demand / "
          f"${gch*USD_PER_VCPU_HR_SPOT:.0f} spot")
    print(f"  wall-clock on a 96-vCPU c7i.24xlarge (~$4.3/hr OD): ~{gch/96:.1f} h compute "
          f"(self-play parallelizes; the 100x point dominates)\n")

    # projections toward MuZero scale (single biggest point, tropical+dense, 3 seeds)
    print("  Projection — cost of the LARGEST single scale point alone (x2 substrates x3 seeds):")
    for tag, mult in [("1,000x", 224000), ("10,000x", 2240000)]:
        pt = sum(run_core_seconds(net, mult) for net in SUBSTRATES) * SEEDS / 3600
        print(f"    {tag:>8s} ({mult:8d} games): {pt:8.0f} core-hr -> ${pt*USD_PER_VCPU_HR_OD:8.0f} OD / "
              f"${pt*USD_PER_VCPU_HR_SPOT:7.0f} spot  (~{pt/96:.0f} h on 96 vCPU)")
    print("\n  NOTE: 'games' here are Connect4. Chess self-play is ~10-50x slower per game (python-chess "
          "rules + 769-d obs + 4096 actions) -> multiply chess costs accordingly.")


if __name__ == "__main__":
    main()
