# Chess: multiply-free TikiTorch vs MuZero — a 4-axis comparison (quality / throughput / energy / silicon)

> **Status: grounded half populated; systems axes are a labelled scaffold.** Axis 1 (quality) and the
> *today* row of Axis 4 (silicon) are filled from verified sources. Axes 2–3 (throughput, energy) and the
> chess-scale rows of Axis 4 are `[to-measure]`/`[projected]` until the runs in "Remaining work" land. This
> mirrors `results/lm_shakespeare/fig_hallmark_lm.md` and its measured-vs-projected discipline.

**Honest framing (the standing caption).** This is a **systems comparison, not a strength contest.** The
chess-quality gap between TikiTorch's multiply-free arm and DeepMind is a **compute/search gap** (TPU-pod
scale, ~800-sim MCTS, tens of millions of self-play games), **not a substrate gap**: the zero-DSP
dyadic-tropical net runs the *same* self-play→replay→K-step MuZero loop end-to-end on chess and its training
loss falls monotonically (27.55→12.00), so the substrate scales structurally. But the trained agent scores
**0.071 vs a random mover — *worse* than its own untrained control (0.393)** — a loss-down/strength-down
decoupling we state plainly (search starvation over a 4096-way action space). DeepMind chess is cited **only
as the quality ceiling.** The reference MuZero-on-GPU run exists **only** to produce a measured
throughput+energy datapoint on one GPU — it is **not** run to strength. The real, fair comparison lives on
the **systems axes** (throughput, energy/move, silicon: DSP=0, area, timing) — where a ~3 W zero-multiplier
fabric can be argued against a ~72 W GPU per frame, exactly as the morph-LM figure argues for LMs.

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

**Axis-1 headline — the loss-down / strength-down decoupling:** training drives the loss down monotonically
yet the trained agent is *weaker than its untrained control* vs a random mover. Strength here comes from
**search + legality** (untrained AlphaZero-tree 0.625 > untrained learned-tree 0.375 > trained pure-MuZero
0.071), not from the learned prior. Encoding caveats baked into every TikiTorch chess number: 4096-way
`from*64+to` actions (**underpromotions not representable**), 769-d planes, 80-ply cap, material adjudication.

---

## Axis 2 — WALL-CLOCK + THROUGHPUT

MuZero has **two** throughputs that map to the FPGA actor/learner split — we report **both** for every arm.
**Cross-arm caution:** the GPU arm runs a resnet (b6c64); the FPGA arm a tiny dyadic projection net — model
sizes differ by orders of magnitude, so do NOT directly compare their samples/s as a like-for-like speedup.

| arm | self-play moves/s (actor) | train samples/s (learner) | provenance |
|---|---|---|---|
| TikiTorch multiply-free, **CPU** (own true-MuZero chess loop) | TBD | TBD | `[to-measure]` EC2 CPU (free) |
| **reference MuZero, A10G** (g5.4xlarge, resnet b6c64, 50 sims, 12 workers) | **1.42** | **54.6** (0.43 steps/s × b128) | `[measured]` GPU idle 93% — self-play is CPU-bound (mz_summary.json) |
| TikiTorch chess kernel, **FPGA** (serial, single instance, no PIPELINE) | fwd: **78.7 inf/s** (12.70 ms) | **30.3** /s (33.05 ms, batch=1) | `[C-synth-derived]` from the axis-4 latencies; NOT silicon, NOT pipelined |
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

_The two GPU bounds are both load-bearing and honest: the **GPU was idle 93% of the window** (chess self-play
is CPU-bound), so **end-to-end** mJ/move (whole-box power × wall-clock) is dominated by idle time and is the
**pessimistic** bound; **GPU-active** (active-power × active-time) is the optimistic compute-only bound — the
LM A10G row reports the same active-avg vs whole-window split. NB: I earlier (axis-3 placeholder) floated a
"~0.7–1.7 mJ-class IF it routed" FPGA figure; that is withdrawn — no MuZero kernel was ever P&R/AFI-routed, so
there is **no** defensible board-FPGA energy number, only the [missing] above._

_The GPU energy caveat is load-bearing: in a Python MuZero loop, CPU-side MCTS dominates wall-clock, so
end-to-end mJ/move over-attributes energy to the GPU. We commit to reporting **both** the end-to-end number
and a GPU-active-window (utilization-gated) bound, labelled — exactly as the morph-LM A10G row reports
active-avg vs whole-window (1.46–1.49 mJ)._

---

## Axis 4 — SILICON (zero-DSP, area, timing)

| arm | metric | value | provenance |
|---|---|---|---|
| **TikiTorch chess-scale FORWARD kernel** (initial inference) | DSP, latency, area @250 MHz | **DSP=0; 3,173,888 cyc = 12.70 ms; LUT 83,485 (6%) / FF 46,558 (1%); 9 BRAM; 0 URAM** | `[measured]` Vitis **C-synth**, xcvu47p (F2 part); forge_train_step_muzero_CHESS_csynth.rpt |
| **TikiTorch chess-scale TRAIN kernel** (bwd + SignSGD) | bounded train-step latency, area | **DSP=1 (~0 of 9024); 8,262,161 cyc = 33.05 ms; LUT 89,759 (6%) / FF 39,544 (1%); 16 BRAM; 0 URAM** | `[measured]` Vitis **C-synth**; the `?` is now RESOLVED (bounded the `nearest_pow2` loop, roadmap M2); forge_train_step_muzero_train_CHESS_csynth.rpt |
| _(reference: gridworld-scale forward, prior)_ | DSP, latency | DSP=0; 25,539 cyc = 0.102 ms; LUT 109,165 | `[measured]` C-synth (small net; superseded by the chess-scale rows above) |
| any TikiTorch MuZero **AFI / P&R / on-silicon power** | timing closure, power-rail energy | **NEVER routed**; csynth-timing-met ≠ silicon (the only routed sibling, char-LM, failed WNS −1.557 ns before closing v6) | `[missing]` — gated behind a strength result |
| reference GPU / DeepMind | DSP/area | N/A — different substrate (that asymmetry *is* the point) | `[missing]` |

_Axis-4 caveats: both numbers are Vitis HLS **C-synthesis estimates** at chess dims (OBS=769, L=64, A=4096),
**serial (no PIPELINE pragma)**, on the F2 part at 250 MHz. They are NOT place-and-route, NOT silicon, NOT an
AFI — csynth timing-met does **not** imply silicon closure (the char-LM precedent). The ~zero-DSP result
(forward DSP=0; train DSP=1/9024) is the load-bearing, defensible claim; the latency/area are estimates._

---

## What's grounded now vs what each remaining run produces

- **Grounded now (this doc):** Axis 1 entirely (DeepMind ceiling + TikiTorch measured, with the honest
  decoupling); **Axis 4 (silicon) COMPLETE at chess scale** — forward (12.70 ms, DSP=0) + train (33.05 ms,
  DSP=1/9024), both C-synth on the F2 part, timing met (DONE 2026-06-13).
- **`[to-measure]`, ~free, fast:** Axis 2 CPU row (wrap TikiTorch's own chess loop in a moves/s + samples/s
  counter — it already runs).
- **`[to-measure]`, ~$0.5–2, ~0.5–2 GPU-h:** Axis 2+3 GPU rows — a dense MuZero board loop on one A10G
  (g5.xlarge), instrumented for moves/s, samples/s, and mJ/move (nvidia-smi), reusing the
  `run_lm_gpu_baseline.sh` lifecycle. **Throughput+energy only — not run to strength.**
- **Gated (deep, $$, days):** lift the TikiTorch trained chess arm above the random floor (anti-collapse
  levers — unproven on chess); full chess-kernel P&R→AFI→on-silicon power (gate behind a strength result so
  silicon doesn't accelerate a below-random agent).
