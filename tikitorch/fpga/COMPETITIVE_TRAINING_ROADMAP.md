# Roadmap: competitive Connect4 / chess MuZero that trains on FPGA

How to get from the honest Task #3 null (all arms 0.00 vs negamax-d2+ on Connect4; the chess agent
below random) to a *competitive* agent using closure-oriented dyadic recipes and FPGA kernels. Produced by a
4-proposal adversarial design panel whose claims were re-verified against the repo artifacts (the
verification corrected several over-optimistic numbers — see "Honest hardware status"). Companion to
`FPGA_BUILD_NOTES.md` (silicon flow) and `results/board_scaling/TASK3_BENCHMARK.md` (the measured null).

## The reframe: the binding risk is the OPTIMIZER, not throughput

The obvious story is "we were search/data-starved (48 sims, ~320 games) — scale it up." That is **true
and necessary**, but the panel surfaced a deeper, evidence-backed risk that must be settled first:

- `results/bakeoff_closed_optimizers.md`: the software-closure-oriented optimizer class (SignSGD / TropicalSGD /
  PureTropicalSGD) is **strong on max-structured targets** (running_max ≈ 0.99) but **weak on
  value-like / DP targets** (min_coin_change **0.00–0.22**, sort **0.12** — vs AdamW **0.70 / 0.97**).
- A MuZero **value head is value-like, not max-structured**, and the Connect4 null was produced **with
  exactly that software-closed optimizer**.

So "just add search + volume" may be insufficient if the multiply-free optimizer cannot train a
competitive value/policy net at all. That question is the first gate (M0), not an afterthought.

## Two senses of "trains on FPGA"

1. **QAT actor-farm (credible, near-term):** FPGA runs the *forward-kernel actors* (the genuinely
   measured, DSP=0, latency-bounded primitive); the learner is **host AdamW on a float master with
   quantization-aware training** — forward through the *same* `nearest_pow2` the kernel uses, STE back,
   then export the bit-exact `(sign,exp)` schedule (`fpga.dyadic_export.verify_schedule`) to the actors
   each cycle. The data-generating policy and the deployed arithmetic are the FPGA's; only the Adam
   moment update is host. No train/deploy cliff; sidesteps the optimizer risk.
2. **Fully on-fabric multiply-free learner (the pure thesis, gated):** the `forge_train_step_muzero_train`
   kernel does fwd+bwd+SignSGD on-chip. This is the flagship claim — but it is **gated** on (a) M0
   proving the software-closed optimizer can train a value head, and (b) M2 turning the kernel into a
   latency-bounded, dynamics-complete learner. Until both, it is a hypothesis, not a deliverable.

## Staged plan

| stage | where | goal | success gate |
|---|---|---|---|
| **M0** optimizer gate | CPU/GPU | Connect4 at scale (200 sims, ~25k games): param-matched **software-closed TropicalSGD vs host AdamW**; port the n-step value bootstrap into `BoardReplay` (it currently discards root values) | GO/NO-GO for the on-fabric-learner thesis: does the audited optimizer match AdamW on win-rate vs negamax-d2/d3? |
| **M1** Connect4 strength | CPU/GPU | sims 48→200–400, games →25k–200k (parallel actors, disjoint seeds), categorical value head, entropy/KL trust-region vs policy collapse, gated Reanalyze | **beat negamax-d2 (score ≥ 0.5), non-zero vs d3** — proves starvation (not capacity) was the blocker; first statistically-powered dyadic-vs-dense-vs-maxplus comparison at volume |
| **M2** make the learner real | HLS | bound `nearest_pow2` to a fixed ≤15-iter loop; add **K-step dynamics unroll + reward head + masked-policy CE**; re-C-synth and report the *actual* bounded latency; register `MuZeroKernelConfig` so `KernelCache.match()` binds it (no silent CPU fallback) | a latency-**bounded** csynth report for the full learner |
| **M3** QAT actor-farm | FPGA + host | host-AdamW+QAT loop feeding the measured forward-kernel actors; demonstrate no train/deploy accuracy cliff on the M1 net | the credible "FPGA-in-the-loop training" deliverable, independent of the optimizer ablation |
| **M4** chess re-architecture | CPU/GPU | AlphaZero **8×8×73=4672** action planes (replaces the 4096 `from*64+to` queen-collapse); **structured/factored** policy head; **legal-mask the policy CE in training**; categorical value + continuous bounded adjudication + n-step bootstrap; ply cap →150; **endgame curriculum first** (KQ-vs-K) | flip all three scaffold metrics positive (beat random / 1-ply material / untrained control), then beat depth-1/2 negamax on the curriculum |
| **M5** on-fabric learner + AFI | FPGA | (gated on M0=GO **and** M2 done) PureTropicalSGD-vs-AdamW learning ablation on the real value/policy target; if it matches, build the AFI and run on F2 | mirror the LM acceptance gate: bound, **dsp==0**, params-evolved, CPU-vs-board bit-match at τ=0 |

The near-term, defensible answer to "competitive **and** trains with FPGA" is **M1 + M3**: cure the
measured Connect4 starvation, deployed as a QAT actor-farm. M5 (pure on-fabric learner) is the stretch.

### M0 outcome (2026-06-10) — partial GO, plus a found+fixed eval bug
- **Optimizer is NOT the blocker:** across 8 Connect4 runs (dyadic/dense × TropicalSGD/AdamW, 2 seeds),
  the multiply-free optimizer is **bit-identical to AdamW**. No detectable penalty for going on-fabric →
  the NO-GO/QAT-forced case is ruled out (QAT stays the documented *fallback*, not triggered).
- **A scaled run (200 sims / 2000 games) did not move the score** → suspicious invariance led to finding
  a **deterministic-eval artifact**: the gauntlet replayed ~2 games. Fixed (`games.random_opening` +
  `BoardMuZero.evaluate` random openings). Under the working eval, training gives a real depth-1 gain
  (0.57→0.72) but **does not beat depth-2** (genuine ceiling, not eval, not optimizer, not the volume tried).
- **Verdict: not yet a full GO** (nothing beats depth-2), but the on-fabric path is alive. The binding
  blocker is now **training dynamics** (policy learns weak-opponent play, not depth-2 tactics) → the M1
  levers (anti-collapse entropy/KL, n-step value bootstrap, more search), NOT more raw games. See
  `results/board_scaling/TASK3_BENCHMARK.md` (UPDATE block) + `connect4_eval_diagnostic.json`.

## Concrete fixes the panel identified (all PyTorch-validatable, no silicon)

- **Value signal (the chess killer):** games must *resolve* or be adjudicated. Continuous bounded
  material+positional adjudication squashed to [−1,1] (two-hot via a categorical head) instead of the
  hard 3-way sign; n-step + learned-value bootstrap; MuZero **Reanalyze** (gated on eval-not-improving,
  always blended with the exact terminal signal so it can't launder a collapsed net's errors).
- **Policy collapse (chess, 4096-way):** the `from*64+to` encoding conflates moves and drops
  underpromotions, poisoning visit-count targets. Use the 4672-plane encoding + a structured head +
  legal-masked CE in *training* (not just at the MCTS root) so the head never spends capacity on ~4000
  illegal moves. NOTE: softmax/log/KL here are host-side transcendentals — the chess training loop is
  **not** fully multiply-free; state that plainly.
- **Search:** 48/12 → 200–400 sims. With legal masking, chess effective branching is ~30–40, so 200+ is
  adequate. This is the single biggest lever on target quality.
- **Capacity:** do **not** inflate the net for Connect4 (Task #3 confirms it is not the limiter); keep a
  param-matched **dense control** in every run so a real dyadic-vs-dense delta can finally be measured.

## Honest hardware status (re-audited; corrects earlier over-claims)

- **Forward kernel:** the chess configuration is real and scoped — DSP=0, routed timing met, and the
  DDR AFI runs 4000/4000 invocations on F2 at 9.22 ms/inference. This is a fixed-point forward
  primitive, not an entire actor or accuracy-equivalence result.
- **Training kernel:** DSP=0 in its C-synthesis report, but it uses 184% of one SLR's LUT and misses
  the 4 ns C-synth timing target (5.427 ns). It is **not AFI-viable** and does not implement the whole
  Python training loop; M2 remains the work needed to define and close a usable learner.
- **Throughput scope:** forward invocation throughput is measured for the chess DDR AFI. Training
  throughput remains C-synth-derived, and neither number is a games/s result for MCTS, replay, or
  orchestration. Do not extrapolate full actors without measuring those boundaries.
- **MuZero-on-silicon is component-scoped:** the forward DDR kernel is routed and exercised on F2;
  no complete MuZero actor/learner or training-kernel AFI is recorded.
- **MCTS-on-fabric does not exist:** `mcts.py` is a serial host tree; the URAM flat-tensor layout is a
  named design target. Near-term "FPGA actors" = the eval primitive + host-driven batched leaf-eval, not
  a self-contained on-fabric actor.

**Bottom line:** the software interfaces scale structurally, but these results do not isolate a single
bottleneck. The levers are
(1) settle whether the multiply-free optimizer can train a value head (M0), (2) cure starvation on
Connect4 (M1), (3) ship the QAT actor-farm as the honest FPGA-training story (M3), and treat the fully
on-fabric learner and competitive chess as gated stretches.
