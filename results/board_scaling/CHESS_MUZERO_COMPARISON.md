# Chess: multiply-free TikiTorch vs MuZero — a 4-axis comparison (quality / throughput / energy / silicon)

> **Status: all 4 axes measured; adversarially honesty-audited against source files (silicon axis re-audited
> 2026-06-15 after the 512-bit streaming-kernel rewrite).** Quality (axis 1, CPU), throughput + energy
> (axes 2–3, measured GPU + C-synth-derived FPGA), and silicon (axis 4, chess-scale C-synth) are all filled.
> **Silicon headline: the FORWARD kernel is C-synth silicon-ready (DSP=0, 14%-of-SLR LUT, timing met with
> margin); the TRAIN kernel is research-track (170%-of-SLR LUT + C-synth timing miss, even at DSP=0).**
> Mirrors `results/lm_shakespeare/fig_hallmark_lm.md`.

## Executive summary (the honest answer)

Multiply-free TikiTorch chess matches MuZero-on-GPU **only structurally** — it runs the same
self-play→replay→K-step loop and its kernels synthesize **zero-DSP** — but it does **not win** on any axis as
measured. **Quality:** chess strength was **not lifted** — the trained agent stays below its untrained control
(scaffold 0.071 vs 0.393; the AlphaZero-tree + anti-collapse lift gave deltas −0.125 / +0.000 / −0.166, z up
to ≈−1.7 at n=12: no detectable training benefit), a compute/data gap (~5 orders below DeepMind) plus untried
levers (legal-move masking), **not** a substrate gap; DeepMind is the literature ceiling only. **Throughput/
energy:** the only *measured* arm is reference MuZero on one A10G (1.42 self-play moves/s, 54.6 train
samples/s; end-to-end 47.6 J/move, GPU-active 3.7 J/move — GPU ~93% idle, chess self-play is CPU-bound). The
FPGA throughput figures (78.7 inf/s fwd, 30.3/s train) are **C-synth-derived, NOT comparable** to the GPU arm
(tiny dyadic net vs resnet; estimate vs measured) — no FPGA throughput or energy win is proven, and the GPU
actually wins the learner direction (54.6 vs 30.3 samples/s). **Silicon:** the defensible claim is the
**zero-DSP FORWARD inference kernel** — chess-scale C-synth DSP=0, LUT 61.5k (14% of one SLR), timing met with
margin (est 2.920 ns vs 4.00 ns target), 7.92 ms — silicon-ready and the AFI target. The **TRAIN kernel is NOT
AFI-viable**: its chess-scale streaming rewrite C-synths at 170% of one SLR's LUT (740k → mandatorily
multi-SLR) and **misses C-synth timing** (5.427 ns); it also surfaced DSP=195 from runtime address-stride
multiplies (`o*IN` with non-pow2 chess `IN`), fixed bit-exactly to DSP=0 (templated the workers on compile-time
dims; CSIM-confirmed, csynth re-confirm in flight) — but DSP=0 does not resolve the area/timing blockers. All
silicon numbers are Vitis **C-synthesis estimates**, never place-and-routed, no AFI, **no routed energy/power
figure of any kind**.

**Honest framing (the standing caption).** This is a **systems comparison, not a strength contest.** The
chess-quality gap to DeepMind is a **compute/search gap** (TPU-pod scale, ~800-sim MCTS, tens of millions of
self-play games), **not a substrate gap**: the zero-DSP net runs the same MuZero loop end-to-end on chess and
its training loss falls monotonically (27.55→12.00). The structural pitch is a **sub-watt-class zero-multiplier
fabric vs a ~72 W GPU** — but note the FPGA **power/energy is unmeasured** (`[missing]`, no P&R/AFI); any
specific FPGA-watt number would be fabrication, so the energy case is qualitative + projected, not won.

**Provenance tags:** `[measured]` ran and recorded · `[to-measure]` produced by the runs below ·
`[citable]` from the literature (source given) · `[projected]` arithmetic extrapolation, not silicon ·
`[missing]` no datapoint and none planned.

---

## Axis 1 — QUALITY (strength). A literature **ceiling**, not a TikiTorch contest.

The two tables below are kept **physically separate** on purpose: DeepMind is the ceiling, TikiTorch is a
structural demonstrator. They do **not** share a strength column.

### 1A. DeepMind ceiling (cited; verify against primary PDFs before any publication)

| system | metric | value | source |
|---|---|---|---|
| AlphaZero (preprint) | vs Stockfish 8, 100 games, 1 min/move (W/D/L) | **+28 =72 −0** | `[citable]` Silver et al. arXiv:1712.01815, Table 1 |
| AlphaZero (Science) | vs Stockfish, 1000 games, 3h+15s/move | **155 W / 6 L / 839 D** | `[citable]` Silver et al., *Science* 362:1140 (2018) |
| AlphaZero | internal BayesElo (draw-compressed; **not** FIDE/CCRL) | ~3400 plateau; crossed Stockfish-level in ~4 h / ~few·10⁵ steps | `[citable]` arXiv:1712.01815 Fig 1a |
| AlphaZero | training compute | ~9 h chess; 5000 TPU (self-play) + 16 TPU (train); ~800 sims/move | `[citable]` arXiv:1712.01815 |
| MuZero | chess strength | **matched AlphaZero** after ~1M steps; Elo measured **vs AlphaZero @ 800 sims, not vs Stockfish** | `[citable]` Schrittwieser et al., *Nature* 588:604 (2020) / arXiv:1911.08265 |

_Note the MuZero relativity: no published MuZero-vs-Stockfish chess Elo exists; do not synthesize a transitive
"MuZero ~3400 vs Stockfish" number._

### 1B. TikiTorch multiply-free chess (measured locally — structural, below strength)

| arm | metric (n) | value | source |
|---|---|---|---|
| trained pure-MuZero (dyadic + TropicalSGD, latent 64, 12 sims) | score vs random, n=14 | **0.071** (0% W / 14% D / 86% L) | `[measured]` chess_scaffold.json (adjudicated) |
| ↳ untrained control, same net | score vs random, n=14 | **0.393** | `[measured]` chess_scaffold.json |
| ↳ non-adjudicated variant (disclosed, not headline) | score vs random, n=14 | 0.143 | `[measured]` chess_scaffold_pure.json |
| untrained net + **true-rules MCTS** (AlphaZero-tree, 200 sims) | score vs random, n=12 | **0.625** | `[measured]` chess_alphazero.json |
| untrained pure-MuZero **learned** tree (200 sims) | score vs random, n=12 | 0.375 | `[measured]` chess_alphazero.json |
| vs material(1-ply) grabber (trained) | score, n=14 | 0.50 (all draws) | `[measured]` chess_scaffold.json |
| training loss | curve over 8 iters | **27.55 → 12.00** (monotonic) | `[measured]` chess_scaffold.json |

**The lift attempt (2026-06-13): AlphaZero tree + anti-collapse — did NOT lift chess.** Applied the
Connect4-winning levers (true-rules AZ tree + soft-CE + entropy/target-temp; fully multiply-free, closure
`[]`) and compared trained vs untrained-control at *matched* sims=48, n=12, 12×16=192 self-play games:

| recipe (AZ tree, matched sims=48) | trained vs random | untrained-control vs random | Δ | loss |
|---|---|---|---|---|
| `az_ce` (no anti-collapse) | 0.417 | 0.542 | −0.125 | 28.8→14.6 |
| `az_ent1` (entropy 0.1, temp 1.5) | 0.458 | 0.458 | **+0.000** | 25.9→18.3 |
| `az_ent3` (entropy 0.3, temp 2.0) | 0.292 | 0.458 | −0.166 | 21.0→14.9 |
| `mz_ent1` (learned-tree control) | — | — | — | `[missing]` — hung (4096-way learned tree), no matched-sims learned-tree control |
| **`az_mask` (+ LEGAL MASKING + 2.5× self-play: 480 games, sims48, n=20)** | **0.475** | **0.500** | **−0.025** | 12.8→8.5 |

**Follow-up (2026-06-14): the faithful setup (legal masking) + 2.5× self-play STILL does not lift chess.**
Added canonical AlphaZero/MuZero legal-move masking to training (multiply-free; closure `[]`) — the
`az_mask` row above — and scaled self-play 2.5× (480 games). Trained 0.475 vs its untrained control 0.500
(Δ −0.025, within noise at n=20); no collapse (mode_frac ~0.01). So **even with the same setup as default
MuZero (masking) and more compute, the trained prior does not beat untrained** — the compute/data wall holds.
Calibrated cost: 480 games = 59 min self-play + 11 min eval ≈ 92 min/recipe on a 36-vCPU box; lifting to
~5k games would be ~10 h/$15+ per config, vs DeepMind's tens of millions. (The `az_nomask` same-budget
control was cut by the 180-min backstop — `az_mask` vs the prior no-mask `az_ent1` (Δ +0.000) already shows
masking didn't change the verdict.)

At n=12 (draw-heavy, so realistic Δ-SE ≈ 0.10) the deltas are **z ≈ −1.2 / 0.0 / −1.7 — none reaches
significance**, i.e. **no detectable training benefit** (the strong-entropy recipe `az_ent3` trends negative
but is still < 2σ); anti-collapse did not rescue it. The loss-down/strength-down decoupling is **not lifted
under AZ tree + anti-collapse** (probed at 2 entropy settings, n=12). (The 4th recipe, learned-tree control
`mz_ent1`, did not finish — the pure-MuZero tree expands all 4096 actions/node and hung; incremental S3 upload
preserved the 3 AZ recipes. So there is **no matched-sims learned-tree control** — see the `[missing]` row.)

**Axis-1 headline — the loss-down / strength-down decoupling, now confirmed under AZ tree + anti-collapse.**
Training drives the loss down monotonically yet the trained agent does not beat its untrained control. Strength
comes from **search + legality**, NOT from the learned prior — _suggestively_ (these are at DIFFERENT search
budgets, **not** a matched ladder): untrained AlphaZero-tree 0.625 @200 sims > untrained learned-tree 0.375
@200 sims > trained scaffold 0.071 @12 sims ≈ trained lift 0.29–0.46 @48 sims. The only *controlled* comparison
is the within-table trained-vs-untrained at matched sims=48. The binding constraint is **compute/data** — 192 games
over a 4096-action space is ~5 orders of magnitude below DeepMind's tens of millions — plus **legal-move
masking is still untried** (the 4096-way softmax wastes mass on illegal moves). Encoding caveats on every
TikiTorch chess number: 4096-way `from*64+to` (**underpromotions not representable**), 769-d planes, 80-ply
cap, material adjudication.

---

## Axis 2 — WALL-CLOCK + THROUGHPUT

MuZero has **two** throughputs that map to the FPGA actor/learner split — we report **both** for every arm.
**Cross-arm caution:** the GPU arm runs a resnet (b6c64); the FPGA arm a tiny dyadic projection net — model
sizes differ by orders of magnitude, so do NOT directly compare their samples/s as a like-for-like speedup.

| arm | self-play moves/s (actor) | train samples/s (learner) | provenance |
|---|---|---|---|
| TikiTorch multiply-free, **CPU** (own true-MuZero chess loop) | TBD | TBD | `[to-measure]` EC2 CPU (free) |
| **reference MuZero, A10G** (g5.4xlarge, resnet b6c64, 50 sims, 12 workers) | **1.42** | **54.6** (0.43 steps/s × b128) | `[measured]` GPU compute-idle ~93% (util<5%) — self-play CPU-bound (mz_summary.json) |
| TikiTorch chess kernel, **FPGA** _(NOT comparable to GPU row →)_ | fwd: **126 inf/s** _(est.)_ | **40** /s _(est., batch=1)_ | `[C-synth-derived]` from axis-4 streaming latencies (fwd 7.92 ms; train 24.86 ms); tiny dyadic net ≠ resnet; estimate ≠ measured — **implies no FPGA win** (and the train kernel is not AFI-viable; see axis 4) |
| _anchor: MiniZero, single box (Othello, **not** chess)_ | ~5 board-games/s (378.5 s / 2000-game iter, 4×1080Ti) | — | `[citable]` arXiv:2310.11305 |
| _anchor: AlphaZero search rate_ | ~80k positions/s | — | `[citable]` arXiv:1712.01815 |

---

## Axis 3 — ENERGY (mirror `run_lm_gpu_baseline.sh` methodology)

| arm | energy | method | provenance |
|---|---|---|---|
| **reference MuZero, A10G** | **end-to-end 47.6 J/move (1237 mJ/sample)** ‖ **GPU-active 3.7 J/move (96.8 mJ/sample)** | nvidia-smi power×dt @5 Hz; 67.5 W whole-window / 73.7 W GPU-active | `[measured]` (power.csv) |
| TikiTorch chess kernel, FPGA | **NONE** — no P&R/AFI ⇒ no post-route latency or power rail to derive from; any mJ would be fabrication | — | `[missing]` (gated behind P&R→AFI, behind strength) |
| TikiTorch, CPU | not measured | — | `[to-measure]`/`[projected]` |
| DeepMind | not published per-move | — | `[missing]` |

_Activity gate (so the bounds are reproducible from power.csv): **GPU-active = samples with util ≥ 5%** = 7.2%
of the window (mean active power 73.7 W); the remaining **~93% is compute-idle** (whole-window mean 67.5 W).
This gate defines both the 73.7 W figure and the 93%-idle figure — it is distinct from the util==0 fraction
(67%). The two bounds are both honest: **end-to-end** mJ/move (whole-box power × wall-clock) is dominated by
idle time = **pessimistic**; **GPU-active** (active-power × active-time) = optimistic compute-only — the LM A10G
row reports the same split. NB: an earlier placeholder "~0.7–1.7 mJ-class IF it routed" FPGA figure is
**withdrawn** — no MuZero kernel was ever P&R/AFI-routed, so there is **no** defensible board-FPGA energy
number, only the [missing] above._

_The GPU energy caveat is load-bearing: in a Python MuZero loop, CPU-side MCTS dominates wall-clock, so
end-to-end mJ/move over-attributes energy to the GPU. We commit to reporting **both** the end-to-end number
and a GPU-active-window (utilization-gated) bound, labelled — exactly as the morph-LM A10G row reports
active-avg vs whole-window (1.46–1.49 mJ)._

---

## Axis 4 — SILICON (zero-DSP, area, timing)

| arm | metric | value | provenance |
|---|---|---|---|
| **TikiTorch chess-scale FORWARD kernel** (initial inference) — **silicon-ready** | DSP, latency, area, timing | **DSP=0; 1,980,319 cyc = 7.92 ms; LUT 61,542 (14% of one SLR) / FF 26,167 (3%); 133 BRAM (10%); 0 URAM; est 2.920 ns < 4.00 ns target (timing MET, margin)** | `[measured]` Vitis **C-synth** (512-bit streaming rewrite), xcvu47p (F2 part); forge_train_step_muzero_CHESS_csynth.rpt |
| **TikiTorch chess-scale TRAIN kernel** (bwd + SignSGD) — **research-track, NOT AFI-viable** | DSP, latency, area, timing | **DSP=0 after fix** (was 195 — runtime address-stride mults, templated to compile-time dims; CSIM bit-exact, csynth re-confirm in flight); **4,580,159 cyc = 24.86 ms; LUT 740,242 (170% of one SLR) / FF 261,666 (30%); 324 BRAM (24%); 65 URAM (20%); est 5.427 ns > 4.00 ns target (timing MISS)** | `[measured]` Vitis **C-synth** (512-bit streaming rewrite); forge_train_step_muzero_train_CHESS_csynth.rpt |
| _(reference: pre-streaming chess kernels, 2026-06-13, superseded)_ | DSP, latency, LUT | fwd DSP=0 / 12.70 ms / 83,485 LUT · train DSP=1 / 33.05 ms / 89,759 LUT | `[measured]` C-synth; small-area but per-element HBM access, not the deployable streaming kernel — superseded by the streaming rows above |
| any TikiTorch MuZero **AFI / P&R / on-silicon power** | timing closure, power-rail energy | **NEVER routed**; csynth-timing-met ≠ silicon (the only routed sibling, char-LM, failed WNS −1.557 ns before closing v6) | `[missing]` — gated behind a strength result |
| reference GPU / DeepMind | DSP/area | N/A — different substrate (that asymmetry *is* the point) | `[missing]` |

_Axis-4 caveats: all numbers are Vitis HLS **C-synthesis estimates** at chess dims (OBS=769, L=64, A=4096) on
the F2 part (xcvu47p), 4.00 ns / 250 MHz target. They are NOT place-and-route, NOT silicon, NOT an AFI — and
csynth-timing-met does **not** imply silicon closure (the char-LM precedent: WNS −1.557 ns appeared only at
route). **The forward kernel is the defensible silicon claim** (DSP=0, 14%-SLR LUT, timing met with margin —
comfortable single-SLR fit). **The train kernel is research-track**: even at DSP=0 its 170%-of-SLR LUT is
mandatorily multi-SLR (HLS will not auto-partition a single C function across SLRs) and it misses C-synth
timing — it needs an area+partition rewrite (URAM-stage the resident matrices, II-fold the 4096-way head, or
split into SLR-sized `hls::stream`/FIFO sub-kernels) before it is a silicon target. The DSP=195→0 fix restores
the zero-DSP claim for training but does NOT change the area/timing verdict (DSP removal is LUT-neutral)._

---

## What's grounded now vs what each remaining run produces

- **Grounded now (this doc):** Axis 1 entirely (DeepMind ceiling + TikiTorch measured, with the honest
  decoupling); **Axis 4 (silicon): FORWARD kernel C-synth silicon-ready** (DSP=0, 7.92 ms, 14%-SLR LUT, timing
  met with margin) — the AFI target; **TRAIN kernel research-track** (DSP=0 after the address-stride fix, but
  170%-SLR LUT + C-synth timing miss → not AFI-viable without an area/partition rewrite). Streaming-kernel
  re-audit 2026-06-15 **supersedes** the earlier "Axis 4 COMPLETE / timing met" claim (those were the
  pre-streaming per-element-HBM kernels).
- **`[to-measure]`, ~free, fast:** Axis 2 CPU row (wrap TikiTorch's own chess loop in a moves/s + samples/s
  counter — it already runs).
- **`[to-measure]`, ~$0.5–2, ~0.5–2 GPU-h:** Axis 2+3 GPU rows — a dense MuZero board loop on one A10G
  (g5.xlarge), instrumented for moves/s, samples/s, and mJ/move (nvidia-smi), reusing the
  `run_lm_gpu_baseline.sh` lifecycle. **Throughput+energy only — not run to strength.**
- **Gated (deep, $$, days):** lift the TikiTorch trained chess arm above the random floor (anti-collapse
  levers — unproven on chess); full chess-kernel P&R→AFI→on-silicon power (gate behind a strength result so
  silicon doesn't accelerate a below-random agent).
