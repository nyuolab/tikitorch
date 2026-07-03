# G2 — the rule-level question (pre-registered spec, v1)

**Status:** spec'd 2026-07-03, not yet running. **Owner:** this instance (local + GPU rental;
no fabric time). **Pre-registration discipline:** the go/no-go below is frozen before any run;
changes require a logged amendment in this file.

## 1. Question and stakes

G1/G1.2 proved the **schedule** (surprise-gated wake + surprise-weighted sleep) beats
matched-budget rehearsal backprop. G1.1 showed the **rule** (iterative settled PC as the sleep
step) runs stably but adds nothing over cascade-sleep at 0.5M-param scale. Every remaining
paradigm-*specific* claim — and the strategy panels flagged this as the program's single most
dangerous assumption — rests on whether the **non-converged PC regime** (settle activity
before weights: prospective configuration, Song et al. Nat. Neurosci. 2024) earns measurable
advantages at LM scale against *modern* SGD countermeasures, at **matched compute**.

- **GO** ⇒ the rule is load-bearing → it enters the paper as a first-class claim, and the
  fabric's iterative/relaxation mode becomes a hardware priority.
- **NO-GO** ⇒ the paradigm honestly contracts to schedule + systems (gating, audit, elasticity,
  energy) — still the G1 result, published with this negative alongside.

## 2. Hypotheses (each independently scored)

| id | hypothesis | primary metric |
|---|---|---|
| H1 | Prospective/iterative PC **interferes less** on sequential domain streams than FLOP-matched BP | forgetting matrix; per-update damage to prior-domain loss |
| H2 | It **adapts faster in the low-data regime** (few-shot domain entry) | adaptation AUC over first k batches of an unseen domain |
| H3 | It **retains plasticity longer** on long/cyclic streams (the Dohare disease) | plasticity-vs-time curve over N domain revisits |
| H4 | Prospective-config **sleep** beats cascade sleep inside the two-speed learner at scale (G1.1 re-asked in the right regime) | G1 Pareto with FLOP-matched sleep budgets |

## 3. Design constraints (lessons already paid for)

1. **FLOP-matched, not step-matched.** Settling costs 5–15× per update; every comparison
   equalizes total forward-equivalents (report step-matched as secondary). Anything else is
   the trivial criticism that kills the paper.
2. **Baselines are 2025–26 countermeasures, not vanilla FT:** (a) G1-style rehearsal (best
   config: 25% domain-balanced), (b) a sparse/memory-subset finetuning proxy (update only
   embedding+last-block — the strongest published forgetting-reduction family), (c) optional
   TTT-style fast-weights baseline if time allows.
3. **Stabilized iterative settings from G1.1** (settle=8, lr_slow=1e-4, γ=0.0625, snapshot
   divergence guard) are the starting point; a small γ/settle sweep is *budgeted in the pilot
   only* — no post-hoc tuning at the decision scale.
4. **≥2 seeds at the decision scale, 3 where cheap.** All events to JSONL; frozen corpora.

## 4. Scale ladder

| rung | model | where | cost | purpose |
|---|---|---|---|---|
| **G2a (pilot)** | the G1 family scaled: DyadicPCNet context-LM, d≈1024, depth 8–16 (~5–15M params) — settling machinery already exists and is measured | local (CPU/MPS) | $0 | find the regime: γ/settle sweep, FLOP-matching harness, H1–H3 first read |
| **G2b (decision)** | PC transformer 30M, BPE (port the worktree `pc_xformer_charlm.py` recipe; implement iterative settling for the transformer stack) | 1 rented GPU, 1–2 days | ~$50–100 | the pre-registered go/no-go |
| **G2c (optional)** | 124M | GPU | ~$300 | only if G2b GO and the effect grows or holds with scale |

## 5. Streams and probes

- **Sequential stream (H1):** the frozen G1 corpora scaled up (wiki base → shakespeare → docs
  → code), plus one novel held-back domain for H2.
- **Few-shot probe (H2):** enter an unseen domain with k ∈ {64, 256, 1024} batches; score
  adaptation AUC and final bpc.
- **Cyclic long-stream (H3):** loop the 3 stream domains ×N revisits at fixed budget/revisit;
  fit the plasticity-decay curve per arm (the continual-backprop protocol, on our substrate).

## 6. Pre-registered go/no-go (frozen 2026-07-03)

- **GO:** at G2b (30M, ≥2 seeds), the prospective/iterative arm beats the FLOP-matched BP
  control on H1 (less forgetting at ≥90% relative plasticity) **or** H3 (strictly slower
  plasticity decay), with the corresponding two-speed variant (H4) at least matching G1's
  schedule-only Pareto point.
- **PARTIAL:** wins only at G2a scale, or only on H2/side-capabilities → report as
  scale-limited; rule enters the paper as an open direction, not a claim.
- **NO-GO:** no hypothesis wins at G2b → rule-level claims are dropped from the paper;
  the negative is published in the honesty ledger with the full sweep logs.

## 7. Execution checklist

- [x] G2a harness: `experiments/g2_rule_probe.py` — h1/h2/h3 probes, FLOP-matched `bp_flop`
      control, iterative full-wake arm grammar `iter_s{N}[_g{X}][_lr{k}]`, divergence guard.
      Runs on MPS (10× CPU, measured). *(2026-07-03)*
- [x] G2a sweep, Phase A.1 (`g2a_h1_sweep.jsonl`, seed 0, 1037 s): **the `infer`-path settle
      (fixed-step energy descent on states) fails at width 768 in every configuration tried.**
      s=8/lr=1e-4: numerically finite but destructive (final bpc 10.8–26.9, model destroyed);
      s∈{16,24,40} × γ∈{0.0625,0.125} × lr∈{1e-3,1e-4}: **1800/1800 updates diverged** to NaN
      (guard rolled all back; arm ≡ frozen). The width-64 MNIST recipe (s=40, γ=0.125) does
      not transfer; PC settling conditioning degrades with width — consistent with the µPC
      line's motivation. → Pilot finding P1, goes in the paper's honesty ledger regardless of
      the eventual verdict.
- [x] Phase A.2 (`g2a_h1_sweep2.jsonl`, seed 0, 396 s): **the error-relaxation path works.**
      Sync sanity passed exactly — relax\_p100\_i24 reproduces bp bit-for-bit
      (3.509/4.047/3.445/2.702), validating the machinery. Async non-converged regime is
      stable at width 768: relax\_p50\_i24 = 3.527/3.998/3.474/2.728 (0 divergences),
      p25 slightly behind, lr4 variant trades plasticity for retention (wiki 2.969).
      Small-γ infer last-checks: dead (γ=0.02 → bpc 11–21; γ=0.01 → bpc 400+ — worse, not
      better). **LOCKED for decision runs: async error-relaxation, participation 0.5,
      n\_iter 24 (=3×depth), lr 1e-3 (+ the lr4 variant as the retention-flavored point).**
- [x] Phase B complete (`g2a_{h1,h2,h3}_phaseB.jsonl`, `g2a_h1_bplr4.jsonl`, 2 seeds each):
      **the stable non-converged form (async error-relaxation, p=0.5) is behaviorally
      equivalent to backprop on all three hypotheses at 4.1M.**
      H1: relax 3.388/+0.740/1.00 vs bp 3.399/+0.750/1.00 (indistinguishable); the FLOP-rich
      bp control *hurts* retention (+1.041). H2: few-shot curves identical (2.670 vs 2.671 at
      k=1024). H3: plasticity decay identical (66.5% vs 67.3%). Attribution control: the lr4
      retention profile is pure learning-rate — bp\_lr4 3.690/−0.097/+0.450 ≡
      relax\_lr4 3.696/−0.090/+0.453. **Meanwhile the G1 schedule (ts) replicated at 9×
      scale:** best H1 Pareto point again (3.386/+0.634 at fewer updates), and on H3 cyclic
      streams it dominated final quality on every domain (wiki 3.666 vs 4.107) — it converts
      relearning into retention.

## 8. G2a verdict and amendment (logged 2026-07-03)

**Pilot verdict: no rule-level advantage exists in any regime we can currently run.** The
hypothesis-carrying regime (prospective configuration / settle-before-update) fails to
*condition* at width 768 (P1) — it cannot be tested at 30M without µPC-style reparameterization
— and the regime that does run (async error-relaxation) is exactly BP-equivalent (P2–P4).

**Amendment A1:** G2b's GPU spend is **paused** under the program's own rule (every dollar
follows a passed gate): its premise — a candidate regime carrying H1–H4 — did not survive the
pilot. G2b is re-gated on a conditioning fix for prospective-configuration at width
(µPC/Depth-µP-style rescaling), filed as research-track. Rule-level claims remain **out of the
paper** (they were never in). Program priority moves to **G1b** (fabric certification) and
**G3** (fleet merged-beats-solo), where the schedule — which has now won at 0.5M and 4.1M, on
sequential and cyclic streams — keeps compounding.

- [ ] (re-gated) G2b: conditioning fix for prospective PC at width, then the 30M rung.
- Amendments log: **A1** above (2026-07-03) · **A2** below (2026-07-03).

## 9. Amendment A2 — the U-gate: conditioning probe for train≡inference (pre-registered)

**Why (mission sharpening, user 2026-07-03):** the paradigm's definitional end-state is
**no difference between training and inference** — one settling computation, target clamped
(learn) or free (infer). That is the prospective/settling regime, which G2a-P1 found fails to
*condition* at width 768 (plain fixed-step energy descent on states: destroys or diverges in
every config). The µPC line exists precisely because of this; the fix-shape is
reparameterized/normalized settling dynamics.

**Probe (single seed, local MPS, ~15 min):** implement RMS-normalized settling —
`state -= γ · g / (RMS(g) + ε)` per layer (bounded per-element steps regardless of width) —
as a standalone function (library untouched). Matrix: {plain γ=0.0625 (known-broken control),
µ-settle γ∈{0.0625, 0.125}} × settle∈{16, 32} at h768/d8.
Stage 1 (stability): 100 full-wake settle-then-update kicks at lr 1e-4 on the Shakespeare
stream → finite? base-domain destruction (wiki drift)?
Stage 2 (only survivors): 600-update single-domain stream vs a cascade-600 reference.

**Pre-registered read-out:** a µ-settle config that (i) stays finite, (ii) wiki drift < 0.5
bpc, and (iii) lands within 0.15 bpc of cascade on the adapted domain ⇒ **the conditioning
hypothesis is confirmed and G2b re-opens** with that parameterization (train≡inference becomes
testable at scale). All µ configs failing (i)/(ii) ⇒ the unification pillar stays gated on
deeper reparameterization (Depth-µP), honestly reported.

**RESULT (`ugate_probe.jsonl`, 2026-07-03): "not yet" — stability solved, quality gated.**
Plain settle (control): NaN immediately, reproducing G2a-P1 ✓. µ-settle (RMS-normalized):
**every config finite through stage 1** with modest drift (+0.17…+0.26 over 100 full-wake
kicks) — the divergence problem is FIXED by per-layer gradient normalization. But stage-2
learning quality is far from cascade (+1.60…+1.80 bpc at 600 kicks; one config, γ0.0625/s32,
degraded numerically without NaN). Criteria (i)(ii) pass for 3 configs; (iii) fails for all.
**Conclusion: normalization conditions the *dynamics* but not the *credit assignment* at this
width — the unification pillar stays gated on principled reparameterization (Depth-µP proper:
width-scaled weights/activities, not just normalized steps). That is the next U-gate iteration,
research-track.**
