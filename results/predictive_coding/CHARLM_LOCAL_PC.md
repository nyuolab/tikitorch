# Local tropical PC trains a char-LM at backprop quality (multiplier-free, async/fabric form)

**Date:** 2026-06-20 · **Code:** `scripts/pc_charlm.py` (reuses `DyadicPCNet`
`error_relax_grads` / `train_step_relax`) · **Data:** tiny-shakespeare (char).

## The question (the architecture gate, done right)

The scaling thesis needs **local** PC — the iterative error-relaxation in its **async /
partial / stale** form (neighbour-local, no global backward sweep, the parallelizable
*fabric* carrier) — **not** the cascade (which is an ordered single-pass backward = the
Z-IL backprop limit). The gate: *does local tropical PC train a competitive sequence
model to backprop quality, multiplier-free, when adequately trained?* MLP/conv
**classification** was already shown (flat to depth 100); this is the first **sequence /
LM** test of the local form.

Model: a feedforward **context** char-LM (Bengio-style) = `DyadicPCNet` residual stack
mapping a C-char one-hot context → next char. Multiplier-free (±2^k `DyadicLinear`,
tropical max-residual). This isolates the **credit-assignment** question (local PC vs
backprop on the *same* architecture); it is deliberately not a transformer (the min-plus
*recurrence* we first tried is numerically unstable — see below).

## Result — local PC matches backprop (identical setup, best-val early-stopping)

Same model / data / regularisation; each arm at its proper learning rate; report each
arm's best validation bits-per-char. **Headline: full data (~1M chars), ctx16, held-out
*future* text** (`charlm_localpc.jsonl`):

| arm | credit assignment | val bpc |
|---|---|---|
| bp_float | float backprop (the ceiling) | 2.483 |
| bp_dyadic | backprop (real-multiply backward), ±2^k forward | 2.573 |
| relax (sync local PC) | local, ordered (== cascade gradient) | 2.541 |
| **relax_async (async/fabric local PC)** | **neighbour-local, 50% participation + stale + random-order** | **2.540** |
| *bigram floor* | — | 3.581 |

**The async/fabric form — partial participation, stale reads, no global update order,
fully multiplier-free — trains the LM to 2.540, *matching* backprop credit and the sync
form, and slightly beating bp_dyadic (2.573) on the *same* ±2^k forward.** The +0.057 vs
bp_float is the price of the multiplier-free *forward* (bp_dyadic carries it too), **not**
a credit-assignment gap: with the forward held fixed, local async PC = backprop. This is
the carrier the fabric thesis needs, working on a sequence task, not just MLP
classification.

Confirmed independently on a same-setup **ctx12 / 300k** comparison (the lr sweep,
`_charlm_lr_diag/`): bp_dyadic 2.370, relax 2.327, **relax_async 2.341** — local PC
matches/beats backprop there too.

## The learning-rate lesson (and why an honest answer needed two artifacts ruled out)

Getting here was a cautionary tale about letting artifacts decide a result — in **both**
directions:

1. **Undertraining → false positive.** A 3-epoch smoke showed local-PC-async 2.667 vs
   backprop 2.633 (+0.034) — a "perfect match." But at 3 epochs the gap had not opened.
2. **Adequate training → apparent gap + instability.** Trained adequately at the
   backprop learning rate (1.5e-3), local PC peaked at **epoch 2 (2.68)** then **degraded**
   monotonically, trailing bp_dyadic — looking like a real gap *and* an instability.
3. **The gap was a learning-rate artifact → real match.** The fully-multiplier-free
   updates (operand-quantized weight grad, ±2^k) have a **larger effective step**, so at
   the backprop lr they overshoot. At **lr ≈ 5e-4 (~3× smaller)** local PC descends
   **smoothly** to 2.327/2.341, matching backprop. Measured lr sweep (ctx12/300k,
   `_charlm_lr_diag/`): relax 1.5e-3 → early-peak 2.68-then-degrade; 5e-4 → 2.327
   (smooth, best); 2e-4 → 2.400 (smooth, still descending at ep13 = slightly too low).

**Takeaway (reusable):** multiplier-free local PC trains to backprop quality, but needs a
~3× smaller learning rate than backprop. Mis-tuning it produces an early-peak-then-degrade
that masquerades as a fundamental gap. Both the undertraining (false-positive) and the
mis-tuned-lr (false-negative) had to be ruled out before the honest answer emerged.

## Honest scope / limitations

- **Weak architecture.** This is a feedforward context-MLP LM (~2.34 bpc here), not a
  competitive LM — the in-tree tropical conv `MorphSSM` reaches 1.93 nats. The result is
  "local PC matches backprop **on the same architecture**" (a credit-assignment claim),
  not "a competitive LM trained by local PC." A stronger arch (tropical conv / attention)
  is the next step.
- **The min-plus *recurrence* is the wrong carrier.** `TropicalSequenceModel`
  (h_t = min(W_h⊕h, W_x⊕x)) is numerically unstable for LM — the state drifts to its
  running extremum (measured: h mean −81, → 16.5 bpc, worse than random). We use the
  stable feedforward context model instead.
- **Single seed.** The match is within ~0.03 bpc; this is "comparable," not a robust
  "beats."
- **Async is simulated.** Participation/staleness/random-order are emulated in one
  process — the parallelism is structural, not wall-clock-measured, and each async step
  pays the relaxation round-tax (`n_iter = 3·depth`; see `staleness_frontier`).

## What it means for the thesis

The fabric scale-out thesis requires that the **local, async, multiplier-free** PC form
can train real models at backprop quality. This is the first evidence it does so on a
**sequence** task (not just classification): the async/partial/stale/any-order form holds
backprop quality on a char-LM. Open work toward "outscale transformers": (1) a **stronger
architecture** (tropical conv / the LLM-instance's mult-free transformer) trained by local
PC; (2) **measured** parallelism economics (round-tax vs the avoided global ops) on the
LM; (3) multi-seed robustness.

## Reproduce

```
PYTHONPATH=. python scripts/pc_charlm.py --context 16 --depth 4 --epochs 14 \
    --lr 1.5e-3 --pc-lr 5e-4 --wd 2e-4 --drop 0.1     # bp arms @1.5e-3, local-PC @5e-4
```
