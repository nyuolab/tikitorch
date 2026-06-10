# Connect4 — AlphaZero-style (true model in tree) + zero-DSP dyadic net

_Score = win + 0.5·draw vs depth-limited negamax, random openings (30 games). The TRUE game model in
the MCTS tree (exact transitions / terminals / legality at every node) + the dyadic net as prior +
leaf value. Still **multiplier-free**: the net is the shift+add kernel, the tree is integer game-rule
logic. The point: this is the **competitive path** for board games (where we have the simulator), and
it sails past the negamax-d2/d3 wall that pure-MuZero (learned model in the tree) could not._

## Depth is the lever (untrained dyadic net — the search does the work)

| sims | vs negamax-d1 | vs negamax-d2 | vs negamax-d3 |
|---|---|---|---|
| 50  | 0.875 | 0.06 | 0.06 |
| 200 | 0.917 | 0.48 | 0.44 |
| 800 | 1.00  | **0.88** | **0.75** |
| *pure-MuZero (learned model) @200* | *0.58* | *0.00* | *0.00* |

Depth **monotonically demolishes the depth-2/3 wall** — with an *untrained* net, so it is the search
(true terminals + legality), not the net, doing it. The pure-MuZero learned-model tree is stuck at 0.

## What training buys (trained vs untrained, AlphaZero tree)

| sims | trained vs d2 | untrained vs d2 | trained vs d3 | untrained vs d3 |
|---|---|---|---|---|
| 50  | 0.40 | 0.37 | 0.20 | 0.53 |
| 100 | 0.73 | 0.65 | 0.70 | 0.73 |
| 200 | 0.82 | 0.77 | 0.87 | 0.70 |

**Honest read:** depth dominates; training's sim-efficiency benefit here is **small and within n=30
noise** (trained edges untrained vs d2 by ~0.05–0.08, but vs d3 it is mixed and *worse* at 50 sims,
0.20 vs 0.53 — consistent with the light 720-game training slightly *narrowing* the policy). So the
zero-DSP net + AlphaZero search is already **competitive via depth** (≈0.82/0.87 vs d2/d3 at 200 sims),
but the net's prior is **not yet good enough to clearly reduce the sims needed**. Making the prior
genuinely valuable (a real sim-efficiency win → cheaper/faster FPGA actors) is what the **M1** levers
are for — more self-play, anti-collapse (entropy/KL), n-step value bootstrap — NOT crossing the wall
(depth already does that).

_Reproduce: `scripts/run_connect4_alphazero.py`. Data: `connect4_alphazero.json`. Depth-vs-untrained
table from the AlphaZeroBoardModel scaling probe (commit 821c7f9)._
