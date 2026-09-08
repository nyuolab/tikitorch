# Connect4 — AlphaZero-style explicit-game tree + dyadic net

_Score = win + 0.5·draw vs depth-limited negamax, random openings (30 games). The explicit Connect4
state transition, terminal, and legality logic runs at every MCTS node; the dyadic net supplies the
prior and leaf value. This avoids learned-transition errors, but it is not a multiplier-free
whole-system result: PUCT and min-max normalization retain multiply/square-root/divide boundaries.
At the tested budgets this configuration passes the d2/d3 wall seen with the learned-model tree._

## Depth is the lever (untrained dyadic net — the search does the work)

| sims | vs negamax-d1 | vs negamax-d2 | vs negamax-d3 |
|---|---|---|---|
| 50  | 0.875 | 0.06 | 0.06 |
| 200 | 0.917 | 0.48 | 0.44 |
| 800 | 1.00  | **0.88** | **0.75** |
| *pure-MuZero (learned model) @200* | *0.58* | *0.00* | *0.00* |

Scores rise monotonically across this tested simulation grid with an *untrained* net, supporting the
interpretation that explicit terminals and legality drive the gain here. This finite sweep is not a
general guarantee that deeper search always improves play.

## What training buys (trained vs untrained, AlphaZero tree)

| sims | trained vs d2 | untrained vs d2 | trained vs d3 | untrained vs d3 |
|---|---|---|---|---|
| 50  | 0.40 | 0.37 | 0.20 | 0.53 |
| 100 | 0.73 | 0.65 | 0.70 | 0.73 |
| 200 | 0.82 | 0.77 | 0.87 | 0.70 |

**Honest read:** depth dominates; training's sim-efficiency benefit here is **small and within n=30
noise** (trained edges untrained vs d2 by ~0.05–0.08, but vs d3 it is mixed and *worse* at 50 sims,
0.20 vs 0.53 — consistent with the light 720-game training slightly *narrowing* the policy). So the
dyadic net + AlphaZero-style search is already **competitive via depth** in this benchmark
(≈0.82/0.87 vs d2/d3 at 200 sims),
but the net's prior is **not yet good enough to clearly reduce the sims needed**. Making the prior
genuinely valuable (a real sim-efficiency win → cheaper/faster FPGA actors) is what the **M1** levers
are for — more self-play, anti-collapse (entropy/KL), n-step value bootstrap — NOT crossing the wall
(depth already does that).

_Reproduce: `scripts/run_connect4_alphazero.py`. Data: `connect4_alphazero.json`. Depth-vs-untrained
table from the AlphaZeroBoardModel scaling probe (commit 821c7f9)._
