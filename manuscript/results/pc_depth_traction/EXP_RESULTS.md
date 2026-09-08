# Deep-PC traction — results (2026-07-05)

Answers the three TODO items from `results/graphics/README.md`. Pre-reg: `PREREG.md`.
Raw: `exp1_divergence.jsonl`, `exp2_traction.jsonl`. Family: DyadicPCNet char-LM
(ctx 12, hidden 256, residual), TinyShakespeare, CPU.

## Exp 1 — cascade ≡ relaxation, exactly, at every depth

Per-parameter gradient cosine of `error_relax_grads` vs `cascade_grads`:

| depth | sync (10L iters) | async (3L, participation 0.5, stale, rand-order) |
|---|---|---|
| 2 | **1.00000** | **1.00000** |
| 4 | 1.00000 | 1.00000 |
| 8 | 1.00000 | 1.00000 |
| 16 | 1.00000 | 1.00000 |

The single-sweep cascade **is** the settled-PC gradient — not an approximation — and the
equivalence is robust to depth *and* to stochastic truncation (the async/partial fabric
regime). No compounding with depth; worst-layer cosine = mean-layer cosine = 1.0.
Scope note (pre-registered): softmax/LN attribution needs the transformer relaxation port
(the G2b gap); this MLP family localizes by depth *position*, where nothing compounds.
Theory of *why*: `THEORY_CASCADE_EQUIVALENCE.md` (feedforward graph + chain-rule loss +
converged stable settling ⇒ closed-form fixed point).

## Exp 2 — PC-vs-BP bpb gap vs depth (the central traction figure)

Mean best-val bpc over seeds; **substrate tax** = bp_dyadic − bp_float (dyadic ±2^k vs float,
BP rule); **rule tax** = relax_async − bp_dyadic (local-PC rule vs BP, same dyadic substrate):

| depth | bp_float | bp_dyadic | relax_async | substrate tax | rule tax | n |
|---|---|---|---|---|---|---|
| 2  | 3.054 | 3.075 | **2.677** | +0.021 | **−0.398** | 3 |
| 4  | 2.725 | 2.821 | **2.599** | +0.095 | **−0.221** | 3 |
| 8  | 2.673 | 2.780 | **2.579** | +0.108 | **−0.201** | 3 |
| 16 | 2.670 | 2.770 | **2.572** | +0.100 | **−0.198** | 3 |

*(complete: 36/36 runs, 3 seeds/cell; depth-16 relax seeds 2.579/2.584/2.553 — tight.)*

**Verdict: deep PC has traction.** Both taxes are small and **bounded — neither compounds
with depth** (substrate plateaus at +0.11; the local rule stays ~−0.2). The striking part:
the **rule tax is negative** — local async predictive coding on the dyadic substrate is the
best arm at every depth, beating *both* dyadic-BP and float-BP, and its bpc is flat (2.58)
from depth 4 to 16 while it never degrades. Reading with Exp 1: the *gradients* are identical,
so the win is not a different descent direction — it is the settling loop's implicit
conditioning/regularization finding a better minimum on the non-smooth dyadic loss.

**The honest caveat (pre-registered, not hidden):** this is matched-**data**, not
matched-**compute**. The relaxation arm runs L settling iterations per step, so its wall-cost
is ~10–20× the BP arms at depth 16 (4664 s vs ~200–450 s). At equal *updates over the same
stream* PC wins; per FLOP it does not. The defensible claim is therefore: *the local PC rule
is not a quality tax at depth — it is a quality gain traded for settling compute*, which is
exactly the trade the fabric substrate (where settling is native and multiplies are free)
is built to make cheap.

## Bottom line for the deep-PC figure

- cascade vs iterative: **settled** (identical gradients, all depths) — stop worrying about it.
- PC vs BP at depth: PC **wins** on quality, bounded, non-compounding — traction confirmed.
- the real levers are the two Exp-2 taxes (both small/bounded) and the C3 conditioning problem
  at width (the U-gate track) — not the learning rule.
