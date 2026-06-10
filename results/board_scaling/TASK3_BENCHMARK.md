# Scaling MuZero from GridWorld → Connect4 → Chess (multiplier-free, zero-DSP substrate)

**What this is.** A benchmark of TikiTorch's multiplier-free (max-plus / dyadic, zero-DSP) MuZero
as it scales up the AlphaZero/MuZero ladder: GridWorld (prior) → Connect4 → chess. **What this is
NOT:** a strength match to reported MuZero. Reported MuZero (Schrittwieser et al., *Nature* 2020)
reaches **superhuman** chess (matching AlphaZero, beating Stockfish under match conditions) using
**TPU-pod compute and millions of self-play games**. These runs are workstation-CPU, **tens–hundreds**
of self-play games, tens of MCTS sims. The contribution is: (1) the **same** multiplier-free
self-play→replay→K-step-training loop runs **structurally end-to-end** on Connect4 and chess; and
(2) **honest, adversarially-checked** measurements of what that tractable scale buys.

> This write-up was reviewed by a 3-lens adversarial honesty pass (statistics / claim-vs-data /
> reproducibility); the corrections it raised are folded in below. Score = **win + 0.5·draw**.

---

> **UPDATE (eval bug found + fixed, follow-up to the M0 gate).** The gauntlet numbers in this section
> were measured with a **deterministic eval** (MCTS root-noise off + deterministic negamax), which
> replays the *same game per color* — so the "20–30 game" gauntlet is effectively **n=2**, score
> quantized to {0, 0.5, 1.0} and blind to gradual skill. This was confirmed directly (agent-as-X vs
> negamax-d2 is byte-identical every game) and explains why the score was an invariant 0.5/0.0/0.0
> across 48→200 sims and 320→2000 games and both optimizers. Fixed in `games.random_opening` +
> `BoardMuZero.evaluate` (random openings → distinct games). **Re-measured under the working eval**
> (random openings, 60 distinct games, dyadic+TropicalSGD, 1600 self-play games — `connect4_eval_diagnostic.json`):
>
> | opponent | trained | untrained control |
> |---|---|---|
> | random | 0.77 | 0.83 |
> | negamax-d1 | **0.72** | 0.57 |
> | negamax-d2 | 0.00 | 0.02 |
> | negamax-d3 | 0.00 | 0.02 |
>
> Corrected conclusions: (1) training gives a **real** gain vs the weak depth-1 opponent (0.57→0.72,
> ~2–3 SE) — it *is* learning, which the broken eval hid; (2) it is flat-to-slightly-down vs random
> (within ~1 SE — not a strong collapse, but no gain either); (3) **the depth-2 barrier is REAL, not an
> eval artifact** — trained and untrained both ~0 vs d2. So the plateau survives the eval fix: it is a
> genuine ceiling that is **not the optimizer** (M0: TropicalSGD≡AdamW, 8/8 runs) and **not curable by
> the search/volume tried** (the scaled 200-sim/2000-game run did not move it). The remaining blocker is
> training dynamics (the policy learns weak-opponent play but not depth-2 tactics) — the M1 levers in
> `tikitorch/fpga/COMPETITIVE_TRAINING_ROADMAP.md` (anti-collapse entropy/KL, n-step value bootstrap,
> more search) — not more raw games. The original deterministic-eval table is kept below for the record.

## Connect4 — dyadic vs max-plus vs dense (original deterministic-eval gauntlet — see UPDATE above)

Two-player negamax-MCTS self-play (latent 64, **48 sims, 20 iters × 16 games**, closed TropicalSGD —
multiplier-free), then a gauntlet vs depth-limited negamax of **increasing** strength. The n=4
aggregates below were produced by the `connect4-robust-sweep` workflow driving
`scripts/_one_board_run.py` (4 seeds/arm, one process per core); the committed `scaling_connect4_*.jsonl`
files hold smaller 1–2-seed runs at slightly different settings, so individual seeds there will differ.

| arm (n=4 seeds, mean±SD) | vs negamax-d1 | vs negamax-d2 | vs negamax-d3 |
|---|---|---|---|
| tropical-maxplus | 0.38 ± 0.22 | 0.00 | 0.00 |
| tropical-dyadic  | 0.50 ± 0.35 | 0.00 | 0.00 |
| dense (control)  | 0.38 ± 0.41 | 0.00 | 0.00 |
| *random floor (separately measured, 40 games)* | *0.175* | *0.025* | *0.025* |

(The random floor is a uniform-random legal mover vs the same negamax depths — `connect4_random_baseline.json`;
it is **not** part of the trained-net gauntlet, hence listed separately.)

**Honest reading:**
- **The dominant fact:** in this n=4 sweep no arm reliably scores above zero vs **depth-2 or depth-3**
  (all 0.00). (Caveat: an earlier 2-seed run had *one lucky dense seed* reach 0.5 at d2/d3 —
  `scaling_connect4_full_tropicalsgd.jsonl` — averaged away at n=4. So the honest statement is "no arm
  *reliably* beats depth-2+", not "never a single draw".) None **masters** Connect4 at this compute.
- At depth-1 all three arms' **point estimates** sit above the random floor (0.175), but with n=4 the
  spread is large (SD 0.22–0.41) and **none clears the floor significantly** (dense ≈0.9 SE above it).
  Dyadic is highest (0.50) but its 0.12 edge over max-plus is only **~0.5 SE of the difference of
  means** — the three arms are **statistically indistinguishable here**.
- So the oblique-core advantage that is clean in *supervised value-fitting* (the §2a probes: dyadic
  matches dense, max-plus underfits 1.5–1.7×) **does not separate in the compute-starved Connect4
  self-play loop** — all three cores are bottlenecked alike by search/data, not representation. An
  honest null result for the thesis *at this scale*.

## Chess — pipeline scaffold (dyadic, multiplier-free)

Chess is wired via `tikitorch.planning.muzero.chess_game.Chess` (python-chess; 4096-action
`from*64+to` encoding, queen-promotion collapse; 769-d piece-plane obs; 80-ply cap; optional material
adjudication of capped games). It plugs into the **identical** `BoardMuZero` loop. Two dyadic runs
(closed TropicalSGD), each evaluated over **14 games (n=1 run, percentages below)** vs a random mover,
a greedy 1-ply material grabber, and an **untrained control** (a fresh net's MCTS agent) to isolate
whether training helped.

| run | training loss | vs random | vs material(1-ply) | vs random (untrained control) |
|---|---|---|---|---|
| pure MuZero (5 iters, 10 sims) | 25.9 → 16.9 | 0.14 (0%W/29%D/71%L) | 0.50 (100%D) | **0.46** (0%W/93%D/7%L) |
| + material adjudication (8 iters, 12 sims) | 27.5 → 12.0 | 0.07 (0%W/14%D/86%L) | 0.50 (100%D) | **0.39** (0%W/79%D/21%L) |

(W/D/L are **percentages of 14 games**; data: `chess_scaffold_pure.json`, `chess_scaffold_adjudicated.json`.)

**Honest reading:**
- **The pipeline scales:** the multiplier-free net runs the full chess self-play→training loop
  end-to-end and the **training loss falls monotonically** (it fits the MCTS targets) — the substrate
  is not the blocker.
- **But the learned agent is *not* competitive at this compute** — it **loses to a random mover**
  (0.07–0.14) and is **worse than the untrained control** (0.39–0.46). Diagnosis: at the 80-ply cap
  with ~10-sim search over a 4096-way action space, MCTS targets are too weak/noisy and the policy head
  overfits, **collapsing onto a narrow losing move set**; chess rarely *resolves* at this scale, unlike
  Connect4/TicTacToe.
- **Material adjudication** of capped games gives the value head a non-zero signal. It did **not** fix
  strength (still below random/untrained). On the loss curve: both runs fall at essentially the **same
  per-iter rate (~2.2/iter)**; the adjudicated run's lower endpoint (12.0) is just **3 more iterations**,
  and the two losses are **not directly comparable** (different value targets, and different
  iters/sims/games) — so adjudication should *not* be read as "faster learning". The takeaway is that
  the binding constraint is **search/self-play starvation, not the value target**.
- 100%-draws vs the material agent is two weak players shuffling to the 50-move rule.

---

## Benchmark vs reported MuZero — the honest gap

| domain | reported result | this work (tractable, multiplier-free, zero-DSP) |
|---|---|---|
| GridWorld | n/a | **95–97% optimal-action agreement, 4/5 solved** (prior; dyadic + closed optimizer) |
| Connect4 | **game-theoretically solved in 1988** (Allis — *classical* search, cheap; AlphaZero also masters it) | beats the random floor at d1 only in point estimate; **0.00 vs d2+** — does not learn the game at this compute |
| Chess | **superhuman**, beats Stockfish (MuZero/AlphaZero; TPU-pod, millions of games) | pipeline runs end-to-end, loss falls; agent **below random** — search/data-starved |

**Bottom line.** The multiplier-free, zero-DSP substrate **scales structurally** to Connect4 and chess
on the *same* code path that reaches 95–97% on GridWorld — and it is *not* the bottleneck (chess loss
falls monotonically). What it does **not** do at workstation compute is reach competitive play on the
harder games: Connect4 stalls at depth-1, chess collapses below random. Note the gap is **compute**, of
two different kinds: for **chess**, reaching reported MuZero strength genuinely needs the large-scale
search + self-play (sims, games, steps) that TPU pods provide — the FPGA actor/learner scale-up this
substrate is designed for. For **Connect4**, the *game* is cheaply solvable by classical search (1988),
so our depth-2 stall reflects *our* tractable-MuZero training budget, not the game's difficulty. Either
way the lever is compute/search, the known and expected one — not an algorithmic or substrate change.
The dyadic oblique-core advantage that is clear in supervised value-fitting is **masked** by this
bottleneck in the self-play loop; separating the cores would itself require the scaled-up regime.

_Reproduce: `scripts/run_board_scaling.py` (Connect4 sweep) and `scripts/_one_board_run.py` (single
seed; driven in parallel by the `connect4-robust-sweep` workflow for the n=4 table);
`scripts/run_chess_scaffold.py` (chess, `--iters/--sims/--games`). Data: `scaling_connect4_*.jsonl`,
`connect4_random_baseline.json`, `chess_scaffold_pure.json`, `chess_scaffold_adjudicated.json`._
