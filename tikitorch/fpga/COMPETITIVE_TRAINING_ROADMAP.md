# Roadmap: competitive Connect4 / chess MuZero that trains on FPGA

How to get from the honest Task #3 null (all arms 0.00 vs negamax-d2+ on Connect4; the chess agent
below random) to a *competitive* agent trained in the multiplier-free / zero-DSP regime. Produced by a
4-proposal adversarial design panel whose claims were re-verified against the repo artifacts (the
verification corrected several over-optimistic numbers — see "Honest hardware status"). Companion to
`FPGA_BUILD_NOTES.md` (silicon flow) and `results/board_scaling/TASK3_BENCHMARK.md` (the measured null).

## The reframe: the binding risk is the OPTIMIZER, not throughput

The obvious story is "we were search/data-starved (48 sims, ~320 games) — scale it up." That is **true
and necessary**, but the panel surfaced a deeper, evidence-backed risk that must be settled first:

- `results/bakeoff_closed_optimizers.md`: the only zero-DSP optimizer class (SignSGD / TropicalSGD /
  PureTropicalSGD) is **strong on max-structured targets** (running_max ≈ 0.99) but **weak on
  value-like / DP targets** (min_coin_change **0.00–0.22**, sort **0.12** — vs AdamW **0.70 / 0.97**).
- A MuZero **value head is value-like, not max-structured**, and the Connect4 null was produced **with
  exactly that closed optimizer**.

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
   proving the closed optimizer can train a value head, and (b) M2 turning the kernel into a
   latency-bounded, dynamics-complete learner. Until both, it is a hypothesis, not a deliverable.

## Staged plan

| stage | where | goal | success gate |
|---|---|---|---|
| **M0** optimizer gate | CPU/GPU | Connect4 at scale (200 sims, ~25k games): param-matched **closed-TropicalSGD vs host-AdamW**; port the n-step value bootstrap into `BoardReplay` (it currently discards root values) | GO/NO-GO for the on-fabric-learner thesis: does the closed optimizer match AdamW on win-rate vs negamax-d2/d3? |
| **M1** Connect4 strength | CPU/GPU | sims 48→200–400, games →25k–200k (parallel actors, disjoint seeds), categorical value head, entropy/KL trust-region vs policy collapse, gated Reanalyze | **beat negamax-d2 (score ≥ 0.5), non-zero vs d3** — proves starvation (not capacity) was the blocker; first statistically-powered dyadic-vs-dense-vs-maxplus comparison at volume |
| **M2** make the learner real | HLS | bound `nearest_pow2` to a fixed ≤15-iter loop; add **K-step dynamics unroll + reward head + masked-policy CE**; re-C-synth and report the *actual* bounded latency; register `MuZeroKernelConfig` so `KernelCache.match()` binds it (no silent CPU fallback) | a latency-**bounded** csynth report for the full learner |
| **M3** QAT actor-farm | FPGA + host | host-AdamW+QAT loop feeding the measured forward-kernel actors; demonstrate no train/deploy accuracy cliff on the M1 net | the credible "FPGA-in-the-loop training" deliverable, independent of the optimizer ablation |
| **M4** chess re-architecture | CPU/GPU | AlphaZero **8×8×73=4672** action planes (replaces the 4096 `from*64+to` queen-collapse); **structured/factored** policy head; **legal-mask the policy CE in training**; categorical value + continuous bounded adjudication + n-step bootstrap; ply cap →150; **endgame curriculum first** (KQ-vs-K) | flip all three scaffold metrics positive (beat random / 1-ply material / untrained control), then beat depth-1/2 negamax on the curriculum |
| **M5** on-fabric learner + AFI | FPGA | (gated on M0=GO **and** M2 done) PureTropicalSGD-vs-AdamW learning ablation on the real value/policy target; if it matches, build the AFI and run on F2 | mirror the LM acceptance gate: bound, **dsp==0**, params-evolved, CPU-vs-board bit-match at τ=0 |

The near-term, defensible answer to "competitive **and** trains with FPGA" is **M1 + M3**: cure the
measured Connect4 starvation, deployed as a QAT actor-farm. M5 (pure on-fabric learner) is the stretch.

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

- **Forward kernel:** real — DSP=0, latency bounded 25,539 cycles, 2.920 ns. A complete actor eval primitive.
- **Training kernel:** DSP=0 ✓, but HLS could **not bound its latency** (`?`; data-dependent `nearest_pow2`
  loop) and it is **initial-inference only** (no dynamics unroll / reward head). A zero-DSP primitive,
  **not** a finished learner — M2 is the work to make it one.
- **Throughput numbers are unvalidated:** neither kernel has any PIPELINE/UNROLL pragma (so "Nx unroll"
  is untested and the forward kernel is AXI/latency-bound at ~2.14 cyc/MAC, ~1 ns slack); per-FPGA
  game-rate figures substitute the tiny gridworld-scale kernel for the (never-synthesized) Connect4/chess
  nets. Treat all sizing as a target, not a result, until the actual nets are synthesized.
- **MuZero-on-silicon is projected:** the only kernel ever P&R'd / run on F2 is the char-LM, and it
  **failed post-route timing** (WNS −1.557 ns). P&R closure for any MuZero kernel is a genuine open risk.
- **MCTS-on-fabric does not exist:** `mcts.py` is a serial host tree; the URAM flat-tensor layout is a
  named design target. Near-term "FPGA actors" = the eval primitive + host-driven batched leaf-eval, not
  a self-contained on-fabric actor.

**Bottom line:** the substrate scales *structurally* and is not the bottleneck; the levers are
(1) settle whether the multiply-free optimizer can train a value head (M0), (2) cure starvation on
Connect4 (M1), (3) ship the QAT actor-farm as the honest FPGA-training story (M3), and treat the fully
on-fabric learner and competitive chess as gated stretches.
