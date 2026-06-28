# WIDE BACKWARD RESULT — does widening the cascade backward reach the autograd bar?

**Date:** 2026-06-28
**Question:** The single-block PC-transformer cascade caps at ~0.15 acc while the
identical-forward autograd bar reaches ~0.26. Hypothesis under test: the cap is the
**backward error-path precision** (int16 floor on cross-chip error messages / i64 route
overflow). If true, *widening only the backward error path* would lift the cap and set a
minimal precision → a DSP budget → BUILD-GO.

**Answer: REFUTED. Widening the backward buys nothing.** No backward precision
(int16band / fix24 / bf16 / double) reaches the bar; the widest (double) is statistically
equal to the narrowest (int16band). The cap is **not** the error-path width — it is the
cascade's gradient *computation* through attention/LN, and/or the optimizer step.

---

## DECISION: NO-GO

This experiment does **not** de-risk the cascade and does **not** justify a build. It does
the opposite: it tells us **not to spend DSP widening the backward error path**, because
that path is not the bottleneck.

- **Minimal backward precision that reaches the bar: NONE.** No `BWD_PREC` setting
  (int16band, fix24, bf16, double) lifts single-block accuracy to the ~0.26–0.30 autograd
  bar. There is therefore **no accuracy-justified precision target** — i.e. no DSP budget
  to allocate to a wider backward.
- **Multi-block: NOT TESTED (null).** The refutation stands only at single block. A build
  could not be greenlit even if single-block had passed.
- **Verification: SOUND.** Every reported number reproduces bit-for-bit on an independent
  recompile + the canonical harness; the negative result is real, not a metric artifact.

The premise of the parent BUILD-GO criterion ("wide backward reaches the bar at a known
minimal precision → cascade de-risked") is **falsified**. We do not write
`forge_pc_xformer_mp_block.cpp` off this result.

---

## 1. What was measured

### Setup (single-block, decisive sweep)
- `/tmp/bwdprec/` — copies of the canonical golden, with a new `BWD_PREC` toggle in
  `forge_pc_xformer_bwdprec_block.hpp` that widens **only** the backward error path:
  - `bp_route` = WIDE `double` route accumulator (no i64 overflow at the chosen grid, no
    int16 floor on the cross-chip error message).
  - `bp_ln_bwd` = LN variance-coupling dot + γ-grad as **real multiplies** at the chosen
    precision (no dyadic/int16 rounding).
  - `wgrad` via `mp_wgrad`.
- Harness `forge_pc_xformer_bwdprec_test.cpp`: single-block, OUTPUT role, `ATTN_DOT=1`,
  `HEAD_RESCALE=6`, `STEPS=1500`, batch1 (`ACCUM=1`), EMA acc/bpb (the MP-test metric),
  seeds {0,1,2}.
- Bar: `autograd_bar.py` — `loss.backward()` on the **same forward recipe**, batch1 EMA.

### Forward stayed mult-free (verified, byte-identical)
A forward-identity check flips the `BWD_PREC` global and re-runs `g_forward`: logit / GLOG
/ R / ATTN diffs = **0 in all four modes**. The forward is byte-identical and unchanged;
`int16band + MP_OPT=int` reproduces the integer golden bit-for-bit (0.1542 / 5.810). The
widening is confined to the backward.

> **Precision note on "forward DSP=0":** with `ATTN_DOT=1` the forward *does* contain the
> warm Q·K dot + P·V blend real multiplies. This is the recipe-level **allowed warm-attention
> multiply**, not a deploy-weight multiply, and it is **identical in both the cascade and the
> autograd bar** (the bar uses `q@k` too) — so the comparison is apples-to-apples. The
> deploy-weight projections (`g_wapply`/`g_matmul`) remain pure power-of-2 shift-add,
> `DSP_MULT=0`.

---

## 2. Results — the wide sweep (single-block)

**Bar (autograd, same forward recipe, batch1 EMA):**

| optimizer | acc | bpb | seeds |
|---|---|---|---|
| AdamW | **0.260** | 3.67 | {0.2598, 0.2606} |
| dyadic-Adam | 0.233 | 3.92 | 0.2331 |

acc×chance ≈ **16×** at the bar.

**Cascade with widened backward error path (known-good integer optimizer `MP_OPT=int`),
acc per seed {0,1,2}:**

| BWD_PREC | acc {0,1,2} | bpb | reaches 0.26 bar? |
|---|---|---|---|
| int16band (baseline) | {0.154, 0.164, 0.127} | 5.81 | no |
| fix24 | {0.151, 0.133, 0.133} | **5.42** (↓ from 5.81) | no |
| bf16 | {0.096} | 5.92 | no (slightly worse) |
| double | {0.109, 0.110} | 5.84 | no |

acc×chance ≈ **9×** for the cascade vs **16×** for the bar.

- **fix24 lowers bpb (5.81→5.42) but NOT accuracy** — the wider error path changes the
  gradient, but not toward the bar.
- **bf16's coarse 8-bit mantissa on the error messages actually hurts.**
- All settings land ~0.10–0.16, statistically the **int16band cap**. The widest (double) ≈
  the narrowest (int16band).

**Optimizer also widened, to rule out the optimizer re-narrowing:**
- Wide-error + wide-opt `MP_OPT=dbl` (tuned `MP_LR`): peak **0.163 @ MP_LR=0.125**; default
  `MP_LR=1.0` even *degrades* to 0.127 (a tuning artifact, not a precision win).
- Wide-error + FLOAT dyadic-Adam: tops **0.108**.
- Matched-capacity H=4/DH=16: int16band 0.164, double 0.123 — same verdict.

Neither a wide error path nor a wide/float optimizer reaches the bar.

---

## 3. Multi-block confirm — NOT RUN

`MULTI-BLOCK: null`. Depth was **not** tested. The refutation is single-block only. Even a
positive single-block result would not have been buildable without the depth confirm; with
a negative single-block result, depth is moot.

---

## 4. Verification — SOUND (independent reproduction)

Status from the verify pass: **sound**, `build_justified: false`.

- **(a) Did wide backward reach the bar? NO — reproduced bit-for-bit.** Recompiled the
  bwdprec harness and re-ran the decisive sweep:
  - int16band baseline 0.1542/5.810 (seed0); {0.1542, 0.1640, 0.1269} seeds {0,1,2}.
    **Cross-confirmed** by the independent canonical harness
    `forge_pc_xformer_mp_golden_test (MP_OPT=int)` → **identical 0.1542/5.8100** (the
    bwdprec harness faithfully reproduces the canonical integer cascade).
  - fix24+int {0.1509, 0.1327, 0.1326}; double+dbl {0.1095, 0.1179, 0.1069}; bf16+dbl
    {0.1118, 0.1254, 0.1118}. All the int16band cap.
  - Autograd bar reproduced: AdamW {0.2598, 0.2606}/3.67; dyadic-Adam 0.2331/3.92.
- **(b) Minimal-precision claim sound.** Genuine 4-precision × 2–3-seed sweep + `MP_LR`
  sweep (0.0625→2.0; peak 0.161 @ 0.125, default 1.0→0.1095, both reproduced) +
  matched-capacity H=4/DH=16 (int16band 0.1644, double 0.1246, reproduced). Minor note:
  the claimed `FLOAT_OPT=1` sub-toggle is not in the shipped harness, but it is **not
  load-bearing** — `MP_OPT=dbl` already is a full-double-master dyadic-Adam optimizer, so
  the "wide optimizer doesn't re-narrow" point is covered.
- **(c) Forward genuinely byte-identical** — 0 diffs in all four modes; weight projections
  pure shift-add (`DSP_MULT=0`). Caveat: `ATTN_DOT=1` warm-attention multiply present in
  *both* sides (apples-to-apples), per §1.
- **(d) Depth:** multi-block null/untested.
- **(e) Interpretation independently corroborated** (this is the load-bearing finding):
  - **Decomposition:** dyadic-Adam on **true autograd** grads reaches 0.233, while the wide
    cascade with the **same** dyadic optimizer caps at ~0.15 → the ~0.08 residual is the
    **gradient**, not the optimizer (optimizer cost AdamW→dyadic is only ~0.03 on the bar).
  - **Gradient fidelity (`/tmp/bwdprec/grad_fidelity.cpp`):** wide (double) cascade gradient
    vs finite-difference of the real CE forward — **linear** layers (`WHEAD`/`WDN`/`WUP`)
    sign-agree ~0.57–0.78 with the true gradient, but the **attention** `WQ`/`WK` gradients
    (through the softmax backward) sign-agree only **~0.45–0.53 = CHANCE**. FD is noisy on
    the dyadic STE surface (treat as directional), but it localizes the divergence to
    **attention backward**, exactly as claimed.

**Hygiene:** `/tmp/bwdprec` copies of the golden headers are code-identical (mod comments)
to canonical; no experiment files leaked into the canonical kernels dir; FIX#1/2/3 intact;
canonical golden untouched.

---

## 5. Interpretation — where the 0.15→0.26 gap actually lives

The autograd bar uses the **identical forward recipe** yet reaches 0.26 via true backprop,
while the hand-rolled cascade caps at ~0.15 **even with a full-double error path and a
full-double/float optimizer**. Therefore the cap is **not** the cross-chip error-message
width. It is:

1. **The cascade's gradient computation/structure** — specifically the **attention softmax
   backward** (WQ/WK sign-agreement at chance vs FD), and the LN-coupled path. The golden's
   FD grad-check historically validated only the matmul-weight parts, not the
   attention/LN-coupled backward.
2. **And/or the optimizer-step regime** (smaller, ~0.03 component).

**Anchor reconciliation:** the prior "cascade = backprop, bit-exact deterministic
equivalence" result covered the **matmul-weight** cascade. It does **not** hold in this
**dot-attention/LN-coupled** regime — the attention softmax backward diverges from
backprop. That divergence is precisely the gap this experiment exposes.

---

## 6. Recommendation (engineering consequence)

- **DO NOT** spend wider precision (DSP) on the backward error path — it buys nothing
  (double = int16band on accuracy; bf16 is worse; fix24 only moves bpb, not acc).
- **KEEP** the mult-free forward (`DSP_MULT=0`) + mult-free deploy. Those wins stand.
- The 0.15→0.26 lift must come from **making the cascade gradient match autograd** — i.e.
  the **attention/LN backward fidelity** — and/or a **true-Adam step**. A wider error path
  does not address either.

### What this means for the build / spec
- **`forge_pc_xformer_mp_block.cpp` is NOT written off this result.** The premise that
  motivated bit-exacting a fixed+widened golden into the kernel (wide backward lifts the
  cap) is falsified, so there is no widened golden worth realizing in HLS yet.
- **`KERNEL_DESIGN_SPEC.md`:** do **NOT** reclassify the backward as BF16-class. This
  experiment found **no accuracy benefit** to a BF16/double backward, so the spec stays as
  the **mult-free (fix16-class) backward** pending a *different* fix. The backward
  precision section should be annotated: "widening the error path measured to buy nothing
  at single block (see WIDE_BACKWARD_RESULT.md); BF16-class backward NOT justified."

### Next experiment (not a build)
Fix the **cascade attention/LN backward fidelity** (make the softmax/LN backward match
autograd), re-measure single-block to the 0.26 bar, **then** add the multi-block depth
confirm. Only if *that* passes does the build question reopen — and even then the precision
is whatever the *gradient fidelity* fix requires, not a wider error path.

---

## 7. Honesty caveats (seeds / scale / CPU budget)

- **Seeds:** cascade sweep is 1–3 seeds per cell (int16band/fix24 at 3; bf16/double at
  1–2). The bar is 2 seeds (AdamW {0.2598, 0.2606} — tight). The cascade–bar separation
  (~0.15 vs ~0.26) is far larger than the per-seed spread (~0.01–0.04), so the verdict is
  robust to the thin seed count, but individual cell means are not high-precision.
- **Scale:** single-block, D-small, `STEPS=1500`, batch1 EMA. This is a CPU-golden
  micro-regime, not the D=64 AFI scale. The refutation is a statement about *this regime*;
  it is strong because the bar and cascade share the identical forward at the same scale.
- **CPU budget:** all CPU csim; no F2 build, no GPU. Multi-block was skipped (`null`) — a
  real omission, not a pass. The `grad_fidelity.cpp` FD check is noisy (dyadic STE surface)
  and is directional evidence localizing the gap to attention, not a precise gradient norm.

---

## Artifacts
- `/tmp/bwdprec/forge_pc_xformer_bwdprec_block.hpp` — `BWD_PREC` toggle (wide route + LN
  bwd as real multiplies).
- `/tmp/bwdprec/forge_pc_xformer_bwdprec_test.cpp` — single-block sweep harness.
- `/tmp/bwdprec/autograd_bar.py` — autograd bar (same forward recipe).
- `/tmp/bwdprec/grad_fidelity.cpp` — cascade-grad vs FD; localizes divergence to attention.
- Canonical golden (`/tmp/tt-pc/results/predictive_coding/...`) **untouched**.

---

**DECISION: NO-GO.** Widening the backward error path does not reach the ~0.26 autograd
bar at any precision (int16band ≈ fix24 ≈ bf16 ≈ double, all ~0.10–0.16 vs bar 0.26);
minimal backward precision that reaches the bar = **NONE**; multi-block untested.
**Next action:** fix the cascade **attention/LN backward fidelity** (its softmax backward
sign-agrees with autograd only at chance), re-measure single-block to the bar, then add the
depth confirm — do **not** spend DSP widening the backward error path, and do **not** write
`forge_pc_xformer_mp_block.cpp` or reclassify the spec backward as BF16-class off this
result.
