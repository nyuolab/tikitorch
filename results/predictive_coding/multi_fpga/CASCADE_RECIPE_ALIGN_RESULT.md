# Cascade Recipe-Alignment Result — Decisive Gap Diagnosis

**Date:** 2026-06-27
**Scope:** Single-block (NBLK=1) cascade vs. autograd-batch1 bar, on the canonical
`forge_pc_xformer_*` golden tree (`/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/`).
**Verdict status:** SOUND-WITH-CAVEATS (independently reproduced from scratch on the canonical tree).

---

## DECISION: STILL-BLOCKED (recipe lever) → REROUTE to the BACKWARD lever

Recipe alignment does **NOT** close the cascade-vs-autograd gap (0.13–0.15 → 0.25–0.30).
The gap survives **every** recipe lever tried. This is the *anomalous* branch the test was
designed to catch, and it correctly localizes the true divergence: it is **STE-vs-true-quantized
gradient**, not the optimizer recipe and not the forward substrate.

**BUILD is NOT justified on the premise that recipe alignment lets the cascade climb to the bar.**
That premise (the prior commit faf1141 hypothesis) is refuted by this result. The single-block,
no-overflow-confound design isolates the cause cleanly.

---

## 1. The golden fix (in place + verified on the canonical tree)

The three localized depth-fixes were carried from `/tmp/hr_work/mb_golden_block_FIX.hpp` into the
**canonical** golden tree as guarded changes that preserve single-block bit-exactness with the
production kernel. No LLM-instance pattern file was touched — only the canonical
`forge_pc_xformer_*` golden/config tree.

**`forge_pc_xformer_golden_block.hpp`** (2 edits):
- **FIX#1** — `g_backward` attention bwd, DOT path only: promote the score-gradient products to
  `__int128` before the `>> X_ABITS` truncation:
  `gQ[...]+= (i64)(((__int128)gs*a.K[...])>>X_ABITS);` (and the symmetric `gK`/`a.Q`).
  The L1 else-branch (the production path) is **byte-unchanged**.
- **FIX#3** — `g_forward` after the head matmul: guarded `HEAD_RESCALE` line, default 6 (= C_SMBITS),
  `-1` reproduces the old forward.

**`forge_pc_xformer_mp_golden_block.hpp`** (1 edit):
- **FIX#1** — same `__int128` promotion in `mp_backward`'s DOT branch.

**FIX#2** (inter-block `g_rescale`) was already present in the production kernel.

**Deterministic verification of the fix (anchors):**
- `gradcheck1` exhibits the int64 overflow the fix corrects: HIDDEN block-0
  `max|gs*K| = 6.47e20  >>  2^63 = 9.22e18`. i64-wrap vs i128 sign-agreement is only
  **0.5504 / 0.7068** → the i128 promotion is *necessary and correct*.
- `gradcheck2` on the canonical tree: the multi-block cascade backward's hidden-block WQ/WK is
  **BIT-EXACT** to an independent `__int128` reference (`signagree=1.0000`, `BITEXACT=YES`,
  all hidden blocks). RESULT line: *"hidden WQ/WK ARE bit-exact to the __int128 reference
  (header carries FIX#1+FIX#2)."*
- The 3 fixes are genuinely mult-free / DSP=0 (arithmetic right-shift; wider accumulator before
  truncation).
- The canonical golden's `ALIGN_*` recipe toggles are **absent** (0 occurrences) — recipe-alignment
  edits were correctly confined to `/tmp/gate` copies. *(Confirmed: `__int128` ×6 and `HEAD_RESCALE`
  ×2 present in the canonical golden_block; `ALIGN_RECIPE/LN/EMB/CLAMP` = 0 in the canonical mp
  golden.)*

---

## 2. The decisive recipe-alignment test

**Setup:** single-block (NBLK=1, OUTPUT role, ATTN_DOT=1, HEAD_RESCALE=6, ACCUM=1, ~1200 steps,
SEED=1), MP_OPT ∈ {fix16, dbl}. All recipe changes are **mult-free / kernel-implementable** and live
only in `/tmp/gate` copies, behind env toggles `ALIGN_RECIPE / ALIGN_LN / ALIGN_EMB / ALIGN_CLAMP`:
- **(a)** dyadic-Adam on LayerNorm gamma/beta (were FROZEN);
- **(b)** dyadic-Adam on EMB/POS (instead of SignSGD ±1);
- **(c)** drop the int16 master clamp on the fix16 shadow `Wf` (keep the fix16 per-tensor pow2 grid).

**Result — the gap does not close:**

| Configuration | fix16 acc | dbl acc |
|---|---|---|
| **autograd-batch1 reference** (L=1,B=1, same substrate, dyadic-Adam, float master) | **0.272** | — |
| cascade baseline (handicapped recipe) | 0.140 | — |
| cascade STABLE-aligned (learn LN; keep clamp; EMB still SignSGD) | 0.145 | 0.101 |
| cascade LITERAL-aligned (a + b + c) | **0.003 (COLLAPSE)** | 0.105 |

3000-step run **plateaus at ~0.15** (not slow convergence).

**Per-change ablation (why literal alignment fails / is neutral):**
- **(c) drop-clamp** — CATASTROPHIC in fix16 (0.003): one runaway `Wf` coarsens the shared per-tensor
  pow2 grid. In dbl it is a literal no-op (master never reaches ±32767). ⇒ the fix16 per-tensor grid
  *is* the stable master; dropping the clamp actively harms and should not be done.
- **(b) EMB/POS dyadic-Adam** — HURTS (fix16 0.091, dbl 0.126) vs SignSGD 0.140: the pow2 step
  overshoots the small EMB table.
- **(a) learn LN gamma/beta** — roughly NEUTRAL (0.145 vs 0.140); gamma is dyadic-applied, so learning
  only moves it among powers of two.
- Swept `MP_LR {0.25..2}`, `HEAD_RESCALE {-1..12}`, `C_SMBITS {6..14}`, `WFRAC {0..4}`, and the
  float-master optimizer (`FLOAT_OPT FLR {0.003..0.03}`): **none exceed ~0.15.** The cascade is robustly
  walled at ~0.14–0.15.

---

## 3. The decisive divergence (root cause)

Measured with the `pc_xformer_ablate.py` harness — the project's declared Z-IL "faithful stand-in"
for the cascade backward (L=1, B=1, 1200 steps, SAME forward substrate):

| Backward / master | held-out acc |
|---|---|
| float-softmax + float-master | 0.30 |
| precise-softmax + float-master | 0.30 |
| int-LUT-softmax + float-master | 0.33 |
| precise-softmax + int16-master (QMASTER=11) | 0.25 |
| **true integer cascade backward** (same forward) | **~0.15** |

⇒ The full cascade FORWARD substrate (precise/int-LUT softmax, int16 master, fdyad-LN, dyadic weights,
i16 acts) trains to **0.25–0.33 under autograd**, but the **same forward under the true integer cascade
backward caps at ~0.15.**

**ROOT CAUSE — STE vs TRUE-QUANTIZED-GRADIENT, not the recipe and not the forward:**
The "faithful stand-in" (and the autograd bar) use **straight-through estimators**: `DyadSTE`/`QAct`
return `g` unchanged, so the float gradient flows *around* the quantizers. The integer cascade computes
the **true** gradient of the heavily-quantized forward — propagating through `g_route` on dyadic-rounded
weights and against floor-shifted (`g_floorshr`) cached activations — so quantization dead-zones /
floor-rounding **attenuate the signal at every layer.**

The frame's "bit-exact to backprop" is real but **narrower than stated**: `hr_gradcheck` proves the
cascade integer backward == a double-precision reference of the cascade's *own* integer ops (relerr 0);
it does *not* prove the integer gradient == the float/STE backprop gradient. Those are **different
functions**, and the STE gradient is a materially stronger training signal. That difference — not the
optimizer recipe — is the whole 0.13 → 0.25 gap. (This refines the earlier "cascade backward == backprop"
claim: the c2dee53 / c2-grad-check held only under `WGRAD_MODE=1` full-precision operands; the production
fix16/int cascade is a different, weaker function than STE/float backprop.)

---

## 4. Independent verification (reproduced from scratch on the canonical tree)

All reruns on own builds from the canonical tree + `/tmp` harnesses (L=1, B=1, 1200 steps unless noted,
HR=6, ATTN_DOT=1, SEED=1):

- **STE-autograd bar** (precise-softmax + float master, dyadic-Adam, full mult-free substrate):
  acc **0.297**; with int16 master Q11: **0.250**; with float softmax: **0.281** → matches the test's 0.25–0.33.
- **Integer cascade** (`mp_golden_test`, single block): fix16 **0.146**, dbl **0.122**; 3000-step dbl
  PLATEAUS at ~0.13–0.15 → matches the test's 0.14–0.15.
- **Recipe-aligned cascade** (`ALIGN_RECIPE=1`): fix16 COLLAPSES to **0.035** (test said ~0.003 — same
  catastrophic direction), dbl **0.102** (test said ~0.105 — match); baseline (no align) fix16 0.135
  → the gap survives every recipe lever; the "NO" verdict is robust, not cherry-picked.

**Soundness checklist:**
- **(a) Fair target:** yes, modulo a metric mismatch worth flagging — the autograd bar reports HELD-OUT
  acc while the cascade harness reports an EMA of single-batch TRAIN acc. But the ~2× gap (0.27–0.30 vs
  0.12–0.15) far exceeds that noise, and the comparison holds the SAME forward substrate constant
  (dyadic wts, int16 acts, fdyad-LN, warm dot attention, HEAD_RESCALE=6).
- **(b) Reached the bar?** No — reproduced as collapse / no-lift across fix16 and dbl.
- **(c) Mult-free?** Yes — dyadic-Adam (`sign(m)·2^floor(log2(|m|/v))`) on LN/EMB and the drop-clamp are
  pure shift/compare; no float master was snuck in. The `dbl` mode IS a double-precision shadow but is
  explicitly the cascade's diagnostic FLOAT CEILING (integer forward kept; only optimizer/wgrad double),
  NOT a buildable kernel claim. Its purpose — prove the cap is not optimizer-precision — is valid: dbl
  still caps at 0.12.
- **(d) Multi-block grad-check:** PASSES post-fix (gradcheck2 bit-exact, §1).
- **(e) BUILD on recipe-alignment premise:** the test's own conclusion — *"do not BUILD on the premise
  recipe-alignment closes the gap; the lever is the BACKWARD"* — is correct.

---

## 5. Honest caveats (CPU budget + experiment design)

- **STE-vs-true-gradient root cause is an INFERENCE** from the dbl-cascade-vs-STE-bar gap, not an
  independently built controlled A/B. The file `/tmp/gate/ablate_noste.py` (which by name should
  implement the true-gradient ablate) is a **byte-identical copy** of the canonical STE ablate
  (empty diff) — the intended true-gradient harness was never actually implemented;
  `/tmp/gate/ablate2.txt` / `ablate_matrix.txt` contain only labels with empty result fields. The
  attribution is well-evidenced and almost certainly right, but has not been nailed by its own
  dedicated harness.
- **Single-seed, L=1 / B=1, noisy single-batch train metrics.** Direction is robust across reruns;
  exact values jitter (e.g. literal-align fix16 0.035 reproduced vs 0.003 reported).
- **CPU budget:** the full batched-bar sweep is ~13 h serial and was only partially demonstrated. The
  **deterministic grad-check + the autograd-recipe result carry the unblocking logic** — they do not
  depend on the long batched run.

---

## 6. Next action (the real lever = the BACKWARD)

Do **not** spend more effort on optimizer/recipe levers. To reach the bar, attack the backward:
1. **(i)** Make the on-fabric backward **STE-like through the quantizers** — skip the floor/dyad in the
   *bwd* path, keep wide cached activations (so the gradient is not floor-/dead-zone-attenuated); **or**
2. **(ii)** Reduce forward quantization aggression in the bwd-relevant tensors.

Build the dedicated **non-STE vs STE true-gradient A/B harness** (the missing `ablate_noste.py`) to
nail the attribution, then prototype an STE-through-quantizer cascade backward at NBLK=1 and confirm it
climbs toward the 0.27–0.30 bar before committing to `forge_pc_xformer_mp_block.cpp`.

---

**DECISION: STILL-BLOCKED** — recipe alignment does not close the 0.13→0.25 gap; the divergence is
STE-vs-true-quantized-gradient in the backward, not the recipe.
**Next action:** implement an STE-through-quantizer cascade backward (skip floor/dyad in the bwd path,
keep wide cached activations) at NBLK=1 and confirm it climbs to the 0.27–0.30 bar — plus build the
dedicated non-STE vs STE A/B harness to nail the attribution — *before* writing the kernel.
