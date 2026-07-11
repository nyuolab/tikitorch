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
