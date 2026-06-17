# Scaled-run design + cost, and FPGA-native feasibility

Answers two questions: (1) design + cost an experiment to find whether MuZero can be made to **play well**
with more self-play compute; (2) can it run **natively on FPGA**. Built on measured throughput
([`muzero_scale_bench.py`](../../../scripts/muzero_scale_bench.py)) + cost model
([`muzero_scale_cost.py`](../../../scripts/muzero_scale_cost.py)); design adversarially audited.

## Honest framing first

The decisive n=5 head-to-head already showed training is **net-neutral at fair search and
substrate-independent** (dense H2H 0.505 ± 0.115 ≈ tropical 0.500 ± 0.091 — [ROOT_CAUSE](ROOT_CAUSE_TRAINING.md)).
So this experiment does **not** ask "can it play well" in the absolute — it asks the narrower, testable
question: **is that net-neutral wall a *budget* (more self-play would help) or a *ceiling* (it won't), at
workstation-reachable scale?** A flat curve is the most likely outcome and is itself a clean, decisive negative.

## The experiment — H2H vs self-play volume

| knob | value |
|---|---|
| primary axis | total self-play **games** (= scale); everything else held fixed |
| scale points | 1× 224, 3× 672, 10× 2240, 30× 6720, 100× 22400 games (log-spaced) |
| substrates | dyadic-tropical (TropicalSGD) **+ dense (Adam) control** at every point |
| seeds | 5 per (point, substrate) — seed variance is first-order (it killed Exp D's false +0.306) |
| held fixed | sims=100, AZ true-rules tree, games/iter=16, train_steps/iter=40, unroll=3, latent=64, τ 1→0.3, recipe=ce — scale **only** #iters |
| **metric** | **head-to-head**: trained MCTS agent vs its own *byte-identical untrained-init* agent, n=200, alternating colors, random openings. >0.5 = training added strength on identical search. (Chosen over lift-vs-opponent, which swings ±0.4 on baseline noise and manufactured the earlier "catastrophe".) |
| diagnostics | mode_frac, prior_entropy, value_pred_L1 vs value_target_std (the collapse mechanism) |

**Why these choices (audited):** sims is *held* at 100 (varying it reintroduces the starved-search confound
that faked the earlier collapse); iters scale with games so the train:data ratio / buffer freshness stays
constant; the dense control means a flat tropical curve is provably the *MuZero wall*, not a tropical defect
(and dense-rises-while-tropical-doesn't would be the first real substrate gap — it would flip the story).

### Pre-registered decision rule (no cherry-picking)
- **GREEN-LIGHT** — tropical 5-seed-mean H2H monotone in log-volume **and** 100× lower-CI (mean − 1.96·SE) > 0.50 → run the gated **1,000×** confirmation point; gate 10,000× / chess behind that.
- **PARTIAL** — no CI crossing but regression slope on log-volume > 0 (p<0.05) → run 1,000× as tiebreaker.
- **HARD STOP** (clean negative) — curve flat (every point within ±2·SE of 0.50) and slope not significant → do **not** spend on 1,000×/10,000×/chess; ship the forward kernel on its inference merits; do **not** gate FPGA on a strength win. (Scope the claim to *affordable scale*, not "can never play well.")
- Fallback before declaring STOP: a small entropy/target-temp sweep at the 100× point only (collapse is the known nulling mechanism).

## Cost (corrected — the audit caught a real bug)

My first model was **2–6× optimistic**: `games_per_iter=16` caps useful workers at 16, so a 96-vCPU box
idles 80 cores. Corrected (right-size to ~16 vCPU, or pack ~6 concurrent 16-worker runs on one 96-vCPU spot box):

| experiment | cost | wall |
|---|---|---|
| **Connect4 curve, 50 runs → 100×** (the decisive spend) | **~$6–7 OD / ~$2 spot** | ~2.5 h packed / ~15 h on 16 vCPU |
| gated 1,000× confirmation point | ~$21 OD / ~$7 spot | ~5 h |
| 10,000× (~2.2M games, ~real-MuZero scale) | ~$205 OD / ~$68 spot | ~48 h |
| chess 100× (gated follow-up; 10–50×/game) | ~$45–220 OD / ~$15–75 spot | — |

**Required infra fixes before running** (audited): right-size/pack the box (don't rent 96 vCPU for one
16-worker run); add per-eval `best_state` checkpoint to S3 (`BoardMuZero.run()` keeps it in memory only — a
spot kill loses the whole run; fine to just re-run the cheap points, reckless for long chess runs).
**Cheapest decisive spend: ~$2 spot** for the Connect4 slope.

## Can it run natively on FPGA?

**Inference: yes, on silicon today. Training: no — and it's the wrong goal for the strength question.**

| loop component | status |
|---|---|
| net-eval **forward** (repr+pred) | ✅ **on F2 silicon** (AFI, 9.22 ms/inf chess, DSP=0, 14% SLR) |
| net-eval **dynamics** g(h,a)→(h′,r) | ⚠️ HLS + CSIM/ctypes bit-exact (this session); no AFI; needs fixed-point activation scaling |
| **MCTS** select/expand/backup | ❌ 100% Python/CPU; on-fabric = unbuilt **research** (URAM tree + PUCT sqrt/mul/div) |
| **learner** train-step | ❌ research-track: DSP=0 but **184% of one SLR + timing miss**, never routed; no KernelSpec registered |
| replay / orchestration | ❌ CPU; on-fabric unbuilt |

Effort: dynamics AFI = **days**; learner-on-silicon = **weeks–months** (area/partition rewrite + unsolved
zero-DSP magnitude-aware optimizer, H7); on-fabric MCTS = **months/research**.

**Why FPGA-native training isn't worth chasing for strength:** Connect4 CPU training is ~$2; training is
net-neutral at fair search; and self-play is **CPU-tree-bound** (the reference GPU sat 93% idle), so a faster
on-fabric *net-eval* doesn't attack the bottleneck — and on CPU the dyadic substrate is actually 3.5× *slower*
than dense (unoptimized quantize). An FPGA-native learner would accelerate a loop whose learned prior adds nothing.

**Where FPGA-native genuinely pays off:** (1) **deployment** — low-latency, sub-watt multiply-free *inference*
of the trained actor (shippable on its own merits); (2) **massively-parallel on-fabric actors** — *only if*
MCTS-on-fabric is built (a separate research bet), which is the one path that removes the CPU-tree bottleneck
and could beat the GPU actor via linear kernel replication.

## Recommendation

**GO** on the ~$2-spot Connect4 H2H-vs-scale curve — it's a decisive, pre-registered slope, and ~$2 buys an
honest answer to "budget or ceiling." Expect a flat curve (a clean negative) as the most likely result.
**NO-GO** on building FPGA-native training to chase strength — keep training on CPU; reserve FPGA for
inference deployment, and treat parallel on-fabric actors (MCTS-on-fabric) as a separate research bet gated
behind a strength green-light. Gate 1,000×/10,000×/chess strictly behind each prior tier's rule.
