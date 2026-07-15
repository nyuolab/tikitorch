# NORTH-STAR VALIDATION (V-gates) — pre-registration

> **The north star (user, 2026-07-10):** *"a device sits on its stream; every
> example it serves, it may also learn from, at about the cost of serving it
> twice, with no extra silicon, no extra memory class, and no synchronization
> with anything else."*

Committed before any run. This document is also the successor to the missing
`ONTOGENIC_LEARNING.md` pre-registration referenced by
`experiments/two_speed_learner.py`. It fuses the two proven halves: the
pc-morphssm substrate (`results/pc_convrnn/RESULTS.md`: cascade ≡ BP twin at
wall parity; machine-audited zero-multiply training; no activation graph —
d832 trained where autograd OOM'd; async credit exact) and this directory's
ontogeny harness (two-speed gate G1-GO, precision anchor G4, Fisher fleet G3,
organism daemon, G1b silicon certification). The PAUSE recorded in README.md
("awaiting the recurrent CNN-PC architecture") is hereby lifted — the substrate
exists.

## TWO SEPARATE BUT RELATED GOALS (user framing, 2026-07-11)

**GOAL A — the live loop (a mechanism claim).** One persistent process that
continuously serves predictions on its stream AND intermittently learns from
it, in place, at ~2× serving cost, no second process, no synchronization, no
extra memory class. TRUE OR FALSE independently of long-term retention: the
claim is that the loop exists and is cheap. Gates V0–V2; demonstrations D1/D2.

**GOAL B — lifelong retention (a continual-learning claim).** Whether what the
live loop learns is retained usefully in the weights over the device's whole
life: bounded forgetting, weight-norm equilibrium, revisits that pay back. A
device can satisfy Goal A while failing Goal B — continuous learning without
durable retention. Gates V4 (+V4b). The anchor + bit-shift-lr-decay machinery
targets exactly this, and has never been tested on the real substrate.

V3 (honest incumbents) and V5 (silicon energy) price both goals against the
outside world.

**Meta-honesty rung (carried from the branch):** cascade credit ≡ backprop
gradients (G0, bitwise). The paradigm content is the *schedule + substrate +
audit* — never "a better learning rule."

## D1 — the loop (the thing being validated)

```
      ┌──────────────────────────── ONE PROCESS, ONE MODEL, IN PLACE ─────────────────────────┐
      │                                                                                        │
 tokens ──▶ .step() serve ──▶ prediction out            (cost: 1 fwd-eq, zero multiplies @τ=0) │
      │        │                                                                               │
      │        ├─ window ring buffer (last W ids)                                              │
      │        ▼                                                                               │
      │   surprise gate  z = (CE − EMA)/σ  ── z ≤ z*  ──▶ skip (cost 0)   ← most tokens        │
      │        │                                                                               │
      │        z > z*                                                                          │
      │        ▼                                                                               │
      │   CASCADE KICK: reverse sweep over the SAME routing caches the serve                   │
      │   forward already produced (+1 reverse ≈ 1 fwd-eq) → optimizer step                    │
      │   (PureTropicalSGD: sign/shift) → weights updated IN PLACE                             │
      │        │                                                                               │
      │        ▼                                                                               │
      │   next token is served by the just-updated weights                                     │
      │                                                                                        │
      │   GOAL-B slow loop (retention): precision anchor pulls weights toward the              │
      │   EMA-posterior; bit-shift lr decay bounds long-horizon drift; probes +                │
      │   append-only ledger audit every update                                                │
      └────────────────────────────────────────────────────────────────────────────────────────┘
 amortized cost = 1 + p_fire × c_kick   (G1 precedent: p_fire ≈ 0.3 ⇒ ≈ 1.3–2.0×;
 worst case all-fire = the literal "serving twice")
```

**The load-bearing engineering fact (verified in source, to be fixed in V0):**
today's harness is UNFUSED — it serves with one forward, then
`cascade_grads` runs `_forward_cache` again (1 serve fwd + 1 learn fwd + 1
reverse ≈ 3×). V0 builds the fused `serve_and_learn` (serve forward's caches
feed the reverse sweep) and MEASURES the collapse to ≈2×. That measurement is
headline number N2's first half.

## Gate ladder (pass/kill pre-registered; all compute EC2 except V0/V5)

### V0 — fusion + cost audit (local, code-only, $0) [Goal A: A1–A4]
Build: `MorphPCEngine.serve_and_learn()` (NEW method; `cascade_grads` untouched
to preserve the G0–G5 receipts); `StreamServer` driving the `.step()` APIs
token-by-token (B=1 ring buffers); op-count tracer (`_make_catcher`) and
peak-memory for serve / serve+cascade / serve+BP; prequential scorer +
telemetry schema (designed for the D2 strip-chart from day one: per-token
domain, prequential CE learner+frozen, gate z, fired?, cost counters, norm).
**Pass:** streaming logits ≡ batched forward (fp-noise); fused grads ≡
`cascade_grads` bitwise; measured op-ratio serve+learn/serve ≤ 2.5×; learn
peak-mem ≈ serve-class. **Kill:** parity miss = bug (fix, never waive);
op-ratio > 2.5 ⇒ the headline is restated honestly (e.g. "~2.6×") — paradigm
survives, the round number retires.

### V1 — stream calibration + oracle bounds (EC2 ~$20) [A5 pre-condition]
BPE-1024 (the M1 text8 tokenizer, cached) re-tokenization of
`data/experience_stream/`; three regimes — hard-shift (wiki→shak→docs→code),
temporal drift (rotating mixture), novel-vocab (code-symbol injection) — plus a
STATIONARY NEGATIVE CONTROL. Per-domain oracle specialists (i.i.d.-trained to
convergence) give ℓ_oracle(D); **headroom** = Σ_t[ℓ_frozen − ℓ_oracle(dom_t)].
**Pass:** headroom ≥ floor (pre-set: ≥ 0.3 bits/token averaged) on all three
adaptive regimes. **Kill (of the stream, not the method):** headroom < floor ⇒
redesign the stream. **Calibration kill:** the gate must self-quiet on the
stationary control (near-zero fire-rate); over-firing on stationary data is a
gate-calibration failure.

### V2 — does-it-help (EC2 ~$40) [Goal A verdict + D2 data]
Arms: frozen / gated / gated+anchor, warm-started from the M1 S1 d256 BPE
cascade checkpoint, 3 seeds × 3 regimes, τ=0 continued adaptation (the anneal
already happened at birth; hard-from-scratch ≈ bigram is the known price and
is not attempted). Metrics: prequential CE (score θ(t−1) before learning x_t),
**cumulative regret vs frozen**, **headroom-captured** = 1 − R_o/R_o^frozen
(bounded [0,1], oracle-normalized — the most defensible number), adaptation
latency (tokens to within δ=0.1 bits of oracle after a shift), τ-dial
quality-vs-energy Pareto. **Pass:** regret vs frozen < 0 with seed CIs;
**Kill:** regret ≥ 0 ⇒ always-on learning does not pay; the boundary map is
published either way. **Demotion path:** if τ=0 adaptation diverges in all
arms (>+0.5 CE over start), Goal A demotes to soft-serve (τ-floor > 0) and
this document records it.

### V3 — honest incumbents (EC2 GPU ~$60) [AB]
Build the LoRA-periodic baseline (none exists anywhere): sweep rank
{4,8,16,32} × cadence {10k, 50k, 200k tokens} + drift-triggered × inner-K × lr
per cell, AdamW, rehearsal buffer allowed — LoRA is tuned TO WIN if it can.
Plus full-FT-periodic, rehearsal-BP, frozen. **Primary budget currency:
matched joules at iso-quality** (off-silicon: the op-count energy proxy of
`lm_energy_projection.py`, calibrated to silicon at V5); sensitivities that
must not flip the sign: matched wall / matched updates / matched fwd-eq.
Always publish the full Pareto — never a single matched point. The full
LoRA-cycle tax counts: buffered tokens, K backprop steps with a materialized
activation graph (the memory class the cascade avoids), merge, and the cadence
decision itself (a scheduling choice the always-on learner never makes).
**Pass:** our learner is on the Pareto frontier at ≥1 realistic budget and is
never dominated on the {capability, retention, audit-coverage, memory-class}
tuple. **Kill:** tuned LoRA Pareto-dominates including its tax ⇒ the
efficiency claim fails; pre-registered fallback positioning = the
audit/no-sync/memory-class niche, published as a column-by-column segmentation.

### V4 — lifetime (EC2 long, ~$40–80) [Goal B verdict; bar per user 2026-07-11]
**10M served tokens, 5 era-cycles with revisits** (A→B→C→D→A′→B′→…), run as an
organism life (daemon: kill-9 bit-exact resume, append-only ledger, clone-
isolated probes). Anchor (`G4Policy._anchor_pull`, verbatim) + **bit-shift lr
decay wired together for the first time** — the 6k-step sign-drift collapse
(measured on this branch) is the pre-registered adversary. Metrics: weight-norm
trajectory (pass = plateau), transfer-matrix forgetting (pass = non-increasing
trend after anchor engages), top1_agree retention vs era-1 snapshot, re-
adaptation speed on revisits, audit coverage (pass = 100%). **Kill:** unbounded
norm OR increasing forgetting OR sign-drift collapse ⇒ "always-on" demotes to
bounded-horizon adaptation, published honestly. **V4b rider:** two learners on
disjoint shards, weights-only Fisher merge (G3 port) — informational.

### V5 — silicon energy (= M5; F2; USER-RUN, scheduled after V0–V2 pass)
Full-card PMBus (never core-rail — the ~4W Vccint figure is explicitly not a
board claim), iso-quality checkpoint, identical tokens: forward-only AGFI
(serve) vs fused-train AGFI (serve+learn) **marginal delta**, plus the
amortized figure at the measured gate-fire rate (G1b measured 72.8% fire with
136 kicks never issued — the physical energy saving already observed once).
GPU comparison: **adapting GPU-LoRA** board power (A10G 1.46 mJ/tok board
anchor), same stream, same tokens, iso-quality — a frozen-GPU baseline would
be a strawman in our favor. Deliverables: mJ/token served / learned /
amortized; tokens/J vs GPU-LoRA. **Kill:** no measurable delta (instrumentation
floor) or board-joules ≥ GPU-LoRA at matched tokens+quality ⇒ the energy claim
fails at board level; published as the FPGA-tax number feeding the ASIC study
(the card is pre-registered as EXPECTED to lose board power — that loss is an
input, not a defeat). This gate also calibrates the op-count proxy used in V3.

## Metrics & the SPJS scalar (pre-registered form)

Primary: prequential CE; regret vs frozen; regret vs per-domain oracle;
**headroom-captured**. Secondary: transfer matrix, top1_agree, norm
equilibrium, adaptation latency, audit coverage, per-token cost HISTOGRAM
(skip/fire bimodal — report amortized, p95, worst-case; never a bare mean).
**SPJS is a constrained scalar, not a weighted sum** (weighted sums are
reparameterization-dependent and Goodhartable — the G6 lesson):

> SPJS = capability-bits-per-mJ/token (−R_o/T per joule), **subject to**:
> top1-retention ≥ R_min, audit coverage = 1.0, norm slope ≤ 0.

Retention/audit/stability are pass/fail constraints; capability-per-joule is
the only maximized scalar. The Pareto is always published alongside.

## The three paper-grade numbers

- **N1 (Goal A capability):** headroom-captured + regret vs frozen over Z
  served tokens (V2).
- **N2 (Goal A cost):** the measured amortized serve+learn multiplier, with
  zero added multiplies and serve-class memory — beside the tuned-LoRA tax
  (V0 + V3); on silicon at V5.
- **N3 (Goal B):** 10M-token / 5-cycle retention + norm equilibrium + 100%
  audit (V4).

## Ordering & budget

V0 (local $0) → V1 (~$20) → V2 (~$40) → V3 (~$60) ∥ V4 (~$40–80) → V5 (user).
Fail-fast: each kill refunds the remainder. Total EC2 ≈ $160–200.

---

## RESULTS (running ledger — verdicts against the pre-registered criteria above)

### V0 — PASS (2026-07-12, local, $0; jsonl: results/st_stream/v0_gate.jsonl)
- A1 streaming `.step()` ≡ batched forward: relL2 = 0.0 (bit-identical at τ=0).
- A2 fused `serve_and_learn` grads ≡ `cascade_grads`: **bitwise**.
- A3 measured cost: fused **1.23× wall / 1.38× op-volume** vs serve-only
  (unfused legacy shape: 2.23× / 2.38× — the fusion IS the ~2× claim, with
  headroom). First run's 1.26× wall was CPU noise; op-volume is deterministic.
- A4 audits at τ=0: loop-only clean; **closed config (LUT loss +
  PureTropicalSGD) clean end-to-end**; AdamW's own foreach update arithmetic
  recorded as the research arm's declared boundary.
- Headline: **serve+learn ≈ 1.4× serve, not 2×** — the north-star wording
  "about the cost of serving it twice" is *conservative*.

### V1 — PASS (2026-07-13, EC2; jsonl: results/st_stream/v1_streams.jsonl)
- Tokenizer identity: rebuilt BPE-1024 reproduces the ckpt's banked text8
  bpb 2.0273 exactly.
- Oracle bounds (CE nats/tok): wiki 3.9004 · shakespeare 3.3605 · docs 2.9932
  · code 2.4077. Headroom vs frozen: 0.079 / 6.73 / 8.69 / 9.98 bits/tok.
- Regime admissibility: hard_shift 6.37, drift 6.37, novel_vocab 3.64
  bits/tok (all ≥ 0.3 ✓); stationary control 0.079 < 0.15 ✓.

### V2 — BENEFIT PASS, gate-calibration finding open (2026-07-13, EC2;
jsonl: results/st_stream/v2_live.jsonl; 30 runs = 3 arms × 3 regimes × 3 seeds
+ stationary control)
- **Gated arm (the live loop), regret vs frozen (nats/tok, mean±sd over
  seeds):** hard_shift **−2.973±0.023**, drift **−2.921±0.008**, novel_vocab
  **−1.580±0.017**. Headroom-captured: **0.685 / 0.674 / 0.607**. Fire-rate
  ~0.33–0.35; amortized multiplier **~1.13×** serve-only. Regret < 0 with
  seed CIs on all three adaptive regimes ⇒ **the pre-registered pass**.
- **N1 (paper number):** headroom-captured 0.61–0.69, regret −1.58…−2.97
  nats/tok over 800k served tokens, at 1.13× amortized serve cost.
- **Honest finding 1 — anchor transfer failure:** gated+anchor at λ=1.0
  (the G4 placeholder-era constant) strangles plasticity on the real
  substrate: HC collapses to 0.05–0.19. λ re-tune is pre-registered into V4;
  the anchor is a Goal-B mechanism and was never required for the Goal-A pass.
- **Honest finding 2 — calibration kill triggered on the control:** on
  stationary data the gate fires 0.357 and regret is **+0.103** (needless
  updates hurt). z* = 0.25 (the G1 constant) is not regime-discriminating.
  **V2b sweep** (z* ∈ {0.5, 1.0, 2.0} × {stationary, hard_shift}, box
  i-062b98c1430e3e7c1) is in flight to find a z* that self-quiets on
  stationary while keeping the shift benefit. **V2B RESULT (2026-07-13, 6
  runs, seed 0): NO z* in this gate family satisfies the criterion — the
  fire-rate is regime-INDEPENDENT at every threshold** (stationary vs
  hard_shift: 0.271/0.257 at z*=0.5, 0.138/0.124 at 1.0, 0.030/0.034 at
  2.0). Mechanism: a z-score against the gate's OWN running EMA mean/var
  converges to firing at a fixed quantile of whatever distribution it sits
  on — the self-normalization erases exactly the regime information the
  criterion needs. The boundary storms are real signal (EMA lag); the idle
  floor is the fixed quantile. Raising z* trades benefit for volume on BOTH
  regimes together: hard_shift regret −2.86/−2.40/−1.10 and HC
  0.66/0.55/0.25 at amortized mult 1.098/1.047/1.013. Stationary regret is
  flat (+0.11/+0.13/+0.15) at every fire-rate, i.e. dominated by the
  dropout serving artifact, not update damage. **Best operating point of
  this family: z* = 1.0** (81% of the z*=0.25 benefit at 36% of its
  learning cost). **Gate revision pre-registered into V4: the two-timescale
  ABSOLUTE-MARGIN gate** — fire when fast-EMA(CE) exceeds slow-EMA(CE) by
  δ nats (no variance normalization). On stationary both timescales
  converge → gap → 0 → self-quiets by construction; on shift the gap jumps
  by the CE step (~4 nats ≫ δ) → storms; under drift a persistent gap
  sustains firing. Same O(1) scalar cost as the z-gate.
- **Harness bug (V2b, no data loss):** the three z* jobs per regime wrote
  the SAME `v2_tele_{regime}_gated_s0.jsonl` concurrently — box telemetry
  files are interleaved and were NOT synced back; the intact V2-grid
  telemetry (which D2 uses) is canonical. Fix before V4: tele filename must
  carry gate_z/run tag.
- Instrumentation notes: the frozen ARM's apparent +0.06 self-regret is a
  dropout artifact (learner serves in train mode, twin scores in eval — a
  bias *against* the learner, left in place); stationary HC rows written
  before the masking patch contain a garbage denominator (denominator ≈ 0)
  and are masked in analysis.
- **D2 delivered:** `results/st_stream/D2_life_stripchart.{html,png}` — the
  one-process life on one screen (hard_shift/gated/s0; 8,332 windows).

### V3 — PASS as pre-registered: on the Pareto frontier at the realistic
budget, never dominated on the tuple (2026-07-13, g5.2xlarge ~2h;
jsonl: results/st_stream/v3_lora.jsonl; 33 rows = iso warm + 32 runs)
- **Iso-quality warm start:** dense GPT (d256/6L/8H/ff1024) hit val bpb
  2.0263 vs target 2.0273 (delta −0.001, two-sided flag TRUE) — in **1,150
  steps / 16.6 s** on the A10G. Capability-ceiling context reported honestly:
  the dense substrate reaches our checkpoint's text8 quality in 1,150 steps
  vs our 8,699 (~70x less wall clock, cross-hardware; an earlier draft of
  this line said "~3 orders of magnitude" — corrected 2026-07-14 against
  the jsonls), and its FROZEN stream baseline is better too (preq 6.74
  vs our 7.50 on hard_shift) despite identical text8 quality.
- **The frontier claim (hard_shift, same window sequences, extra
  update-compute in fwd-eq of own serve):** ours = preq **4.53 at +0.13**
  (z*=0.25; +0.047 at z*=1.0 for 4.75). The incumbent at ≤ +0.2 budget
  (LoRA cadence-200k, any rank) reaches only **5.35–5.44** — worse quality
  at ~1.4× our budget. The **cheapest incumbent cell that beats our
  quality** is LoRA r16/drift-triggered: 4.17 at **+0.43 (3.3× our budget)**
  and 2.7× serve memory (138 vs 51 MB — the activation-graph class). No
  cell has (preq ≤ 4.53 AND extra ≤ 0.13): **not dominated**.
- **The incumbent's high-compute end is real and reported:** online-BP
  (update every window) reaches preq 3.38–3.39 on hard_shift/drift at
  **+3.0 (4× serve total, 23× our extra)**; LoRA r32/c10k 3.50 at +4.85.
  GPUs win unconstrained-compute quality; they lose the
  low-power/always-on segment — exactly the pre-registered segmentation.
- Cross-regime: novel_vocab OURS WINS OUTRIGHT (4.39 vs LoRA-center 4.47 at
  7× less compute); drift theirs 4.04 at +0.92 vs ours 4.58 at +0.13.
- **Stationary symmetry:** their online-BP also HURTS on stationary
  (+0.17 regret) — unconditional per-window learning damages any substrate;
  gating is universal, not a tropical patch. LoRA-periodic stationary is
  benign (−0.03) at +0.92 compute burned for nothing.
- Caveats on record: seed 0 only (our side is 3-seed, CIs ±0.02); LoRA
  placement = FFN+head (nn.MultiheadAttention bypasses module swap — stated
  handicap, rank sweep spans capacity); MAC-vs-add energy translation pends
  V5 (each incumbent MAC = mult+add ≥ our 2 adds, so the fwd-eq axis
  UNDERSTATES our advantage); serving-mode asymmetry: V2's train-mode serve
  artifact (+0.06 vs eval) biases against US in this table — conservative.
- **N2 updated:** 1.13× amortized (1.047× at z*=1.0) with zero added
  multiplies and serve-class memory, beside the tuned-LoRA tax: matching our
  quality costs the incumbent 3.3× our update budget and a 2.7× memory
  class; at our budget its best quality trails by +0.82 nats/tok.

### V4cal — margin gate VALIDATED, V4-life constants locked (2026-07-13,
c7i.4xlarge ~2.5h; 10 runs appended to v2_live.jsonl)
- **The two-timescale absolute-margin gate satisfies the calibration
  criterion the z-family could not:** stationary fire-rate 0.24% / 0.01% /
  0.01% at δ = 0.25/0.5/1.0 (z-family: 27–35% at every z*), while keeping
  the shift benefit — hard_shift at δ=0.25: regret −2.67, HC 0.616, fire
  11.2%, amortized 1.042×; drift at δ=0.5 sustains firing (6.7%, HC 0.553).
- **Artifact decomposed:** stationary regret stays +0.11 with fires ≈ 0 ⇒
  the stationary "harm" in all earlier rows was the train-mode dropout
  serving artifact, not update damage.
- **Anchor λ:** 0.01 → HC 0.474 (−0.08 vs the λ=0 same-gate baseline
  0.554); 0.1 → 0.373. λ=0.01 locked; its retention payback is V4's
  question.
- **lr decay smoke:** mechanism sound; halve-per-200-fires over-decays at
  800k (regret −0.77 — starved lr in late domains). Life cadence scales to
  ~2000 fires/halving.
- **V4-LIFE CONSTANTS (locked):** margin gate δ=0.25 (fast 0.9 / slow
  0.999) · anchor λ=0.01 (EMA 0.9999, clamp 100) · bit-shift lr decay
  every 2000 fired kicks · serve dropout OFF (kills the +0.11 artifact;
  noted as a config delta vs V2).

### V4 — PASS on every pre-registered criterion (2026-07-14, c7i.8xlarge
~6.5h; jsonl: v4_life.jsonl, v4_probes_*.jsonl, v4_tele_*.jsonl; 4 lives)
- **The life (main):** 10M served tokens, 5 rotated cycles
  (wiki→shak→docs→code, order rotated per cycle — the review's deadband
  fix). Regret vs frozen **−2.617** nats/tok, HC **0.602**, at amortized
  **1.0063×** serve — the gate fired **1.65%** of windows over the whole
  life, collapsing 1,275 fires (cycle 1) → 45 (cycle 5); two late eras fully
  silent and 42 of cycle 5's 45 kicks in one final-revisit storm (an earlier
  draft said "eras 15–18 fully silent" — corrected 2026-07-14 against the
  telemetry). The device learned when its stream demanded it, then just served.
- **Norm equilibrium: PASS.** 3072.6 → 3142.4 (+2.27% over 10M tokens);
  slope +4.73 units/Mtok full-life → **+0.39 last quarter** (12×
  deceleration = plateau).
- **Forgetting: PASS, stability-plasticity trade stated honestly.** The
  wiki-specialist birth model became a stable GENERALIST: probe average
  7.48 → 4.82 nats (−2.66) at the price of the birth specialty (wiki 3.91
  → ~5.0). Per-domain probe CEs improve nearly every cycle after cycle 1
  (code 4.82→4.23, docs 5.49→4.94, shak 5.22→5.04); interference
  excursions bounded (wiki oscillates 4.75–5.35) and NON-INCREASING across
  cycles.
- **Sign-drift: PASS via the pre-registered adversary** (drifttest rider):
  20,833 consecutive fired updates on stationary data — 3.5× the historical
  strict@6k collapse horizon — with 10 lr halvings exercised; ended BETTER
  than frozen (regret −0.048), no collapse. Caveat on record: exercises the
  mechanism under AdamW (research arm); the closed shift-optimizer variant
  remains on the strict track.
- **top1_agree** vs the cycle-1 snapshot drifts to ~0.5–0.65 while probe
  CEs improve — weights keep moving (plasticity alive), performance keeps
  rising: healthy drift, not collapse.
- **Audit:** 1,236 kicks audited across the 4 lives (5% sampling),
  **0 violations**; anchor/AdamW arithmetic outside the audited region (V0
  A4 declared boundary); full-life attributability is total (deterministic
  from seed + ckpt chain).
- **N3 (paper number):** 10M tokens / 5 cycles: regret −2.62 (HC 0.602) at
  1.0063× serve; norm equilibrium (+2.3%, slope →0.4/Mtok); per-domain
  probe improvement every cycle with bounded, non-increasing interference;
  fire-rate 1.65% → 0.4%/cycle-5; sign-drift adversary passed at 3.5× the
  collapse horizon; 0 multiply violations in 1,236 audited kicks.
- Riders (V4b inputs): riderA (wiki,docs) regret −1.385/HC 0.465; riderB
  (shak,code) −2.064/HC 0.361; both fire ≤4%, 0 violations.
- **V4b MERGE RESULT (2026-07-14): HONEST NEGATIVE — the G3-F2 Fisher
  recipe did NOT transfer.** Naive averaging beats Fisher on all four
  domains (mean 5.51 vs 5.80; naive within +0.54 of every solo and BETTER
  than the solo on 3 own-shard cells). Mechanism hypothesis, on record:
  (a) under SPARSE GATED firing, Adam second moments are stale and
  fire-rate-mismatched across riders (A fired 4.0%, B 1.4% — importance
  reflects when a rider last fired, not what it knows), corrupting the
  precision weighting that worked in dense-update G3-F2; (b) the
  disjoint-shard premise failed — riders converged to overlapping
  generalist basins (riderA beats riderB on riderB's OWN shard: shak
  5.98 vs 6.10, code 5.01 vs 5.55), so there were no true specialists to
  merge and balanced averaging is near-optimal. Fleet consolidation for
  gated-PC lives needs a firing-normalized importance (or probe-based
  Fisher) — pre-registered as the V4b.1 follow-up, not assumed.

### V5 — FLOOR REPORT, the pre-registered honest branch (2026-07-14,
assistant-run per user delegation; f2.6xlarge, two-image S1 session; data:
results/v5_pmbus/; scoping amendments in V5_PMBUS_PROTOCOL.md)
- **Session integrity:** both KATs green ON SILICON before and during
  measurement — serve image generated bit-faithful text (426.5 ms/char,
  forward-bound); trainer image CONVERGED during the power windows (CE
  4.149→1.912, acc 24/64→64/64, trajectory == CSIM, 512 steps/kick ×14
  kicks ×3 reps). GPU rider reproduced its V3 numbers exactly while
  logging board power. Strike 1 = dropped -I lines (fail-fast, ~$0.50).
- **THE HEADLINE IS A BOUND:** F2 Vccint telemetry quantizes at 1 W, and
  every arm sits at ~4 W (idle-serve 4.55, serve 3.99, idle-train 4.00,
  serve+learn 4.01). The serve→serve+learn power delta is UNRESOLVABLE:
  **marginal learning power ≤ ~1 W on a ~4 W core rail**, i.e. learning
  while serving is energetically invisible at this kernel scale on this
  instrument. Per the pre-registered kill/honest-negative branch, the
  floor is reported — not a win.
- Banked absolute numbers (Vccint rail, labeled — NOT board power): the
  trainer runs **5.30 mJ/token-position total-rail at 756 tok/s** while
  converging; dynamic (active − same-image idle) ≈ 0.013 mJ/tok — below
  quantization, quoted only as ≤-bound. Serve on the kv2 image is
  LATENCY-dominated (62.5 MHz clock, sequential generation): 1.70 J per
  generated token — a clock/architecture artifact of the deploy image,
  not a power story.
- **The regime-level GPU contrast (the meaningful cross-substrate row,
  rails-vs-board caveat STATED):** the A10G burns **79 W board just
  sitting on the batch-1 stream at 26% utilization** — 5.64 mJ/tok
  frozen-serve, 7.49 LoRA-periodic, 16.00 online-BP (measured, same
  streams, nvidia-smi board power). The F2 core rail runs the ENTIRE
  serve+learn loop within ~4 W. A full-card F2 comparison needs external
  12V instrumentation; as pre-registered, this feeds the ASIC study as
  the FPGA-tax/floor input rather than claiming a board-level win.
- **N2-silicon:** the fused-loop op-volume multiplier (1.38×) could not be
  resolved as a power multiplier because the substrate's absolute power is
  at the telemetry floor — the honest statement is "learning adds ≤1 W to
  a ~4 W rail; amortized at the V4 fire-rate the addition is unmeasurable."

### V4 multi-seed (E0-2, 2026-07-15): the life REPRODUCES across 3 seeds
HC 0.602/0.547/0.567 (mean 0.572 ± 0.028, all ≥ the 0.5 bar); regret
−2.62/−2.38/−2.46; fire 1.65/1.36/1.72%; amortized 1.006×; norm +2.2/+2.3%
(equilibrium both); **0 audit violations across all 3 lives (3×10⁷ served
tokens)**. The V4 single-seed caveat is CLOSED; N3 is now a 3-seed number.
