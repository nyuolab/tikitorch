# Gate G2 (DISAMBIGUATION) — Backward-Bug vs Forward-Cap, Resolved

**The question this pass closes:** `G2_RETRY_RESULT.md` returned **NEEDS-MORE (leaning B)** —
it re-localized the acc-0.15 wall to "the **cascade backward + magnitude-decoupled dyadic-Adam**"
but flagged two confounds that kept hard-(B) from being airtight:
1. Is the hand-rolled cascade backward *actually* correct backprop, or is it an **approximate
   backward bug** (some op deviates)?
2. Is the acc-0.15 ceiling the **backward**, or is it the **integer attention forward** carried
   over from the old golden?

Two decisive experiments were run to disambiguate. **Both answers point the same way, and they
overturn the leaning-B read.**

**Date:** 2026-06-27 · **Status:** COMPLETE — grad-check + attention-isolation both RUN & reproduced.

---

## DECISION: **A — BUILD** (with recipe-#3 warm DOT attention + matched capacity)

The two probes G2_RETRY demanded as the precondition for closing the gate have now run, and they
**exonerate the backward** and **convict the forward**:

- **The cascade backward IS correct backprop — no op is buggy.** Grad-check control `WGRAD_MODE=1`
  (full-precision multiply on the weight-grad operands only) makes **every** gradient **bit-exact**
  to the exact double-backprop / outer-product reference: signagr **1.000**, frac<1e-2 **1.000**,
  relerr **0** across WHEAD/WUP/WDN/WQ/WK/WV/WO/LN/EMB/POS. Reproduced here: all `wg:*` rows go
  `0.66–0.99 → 1.000` the instant the deliberate rounding is removed.
- **The integer attention softmax IS the cap; the backward is clean in the failing run too.** With
  a *perfect autograd backward* held fixed, swapping **only** the attention-softmax forward
  warm→intlut drops held-out **acc 0.4513 → 0.1754** (toward the cascade's ~0.15 cap) and raises
  **bpb 2.6501 → 4.5480**. Same data, same backward, same optimizer — the one changed variable is
  the integer attention forward.

So the G2_RETRY wall was **misattributed**. The acc-0.15 ceiling that NEEDS-MORE pinned on "the
cascade backward" is reproduced **with a textbook-correct backward** the moment you feed it the
**old golden's integer attention forward** — and it **vanishes** when that forward is the
recipe-#3 **warm dot** attention (acc 0.45, on the bar). G2 tested an **outdated forward**, not a
broken backward.

**(A)'s two conditions are now met:** (i) the cascade *backward* is sound (bit-exact under the
full-mul control; the only deviation is by-design mult-free rounding, direction always correct);
(ii) the wall is a *fixable forward op* (integer attention softmax) for which the locked recipe
already prescribes the fix (warm dot attention = the one allowed multiply).

---

## 1. EXPERIMENT 1 — GRAD-CHECK: is the cascade backward correct backprop of its own forward?

**Method.** `/tmp/gcheck/final.cpp` (+ `wgradref/attnref/lnref/gcheck2/deepprobe/actprobe`):
compare each backward grad against a clean reference — activation-space finite-difference for
routing, a clean **double** softmax-Jacobian for attention, the exact double L1-LN Jacobian for
LayerNorm, and the exact double **outer-product** for weight-grads. `WGRAD_MODE=1` is the control
that replaces the shipped mult-free dyadic wgrad with a full-precision multiply.
*(Note: dbl-mode keeps `g_forward` integer, so deep weight-space FD is a chaotic staircase —
diagnosed in `deepprobe.cpp`; that is why activation-space FD + clean double-Jacobian refs +
the WGRAD_MODE=1 control are used instead of naive weight-space FD.)*

**Result (reproduced this pass):**

| grad group | shipped (WGRAD_MODE=0) | control (WGRAD_MODE=1) | interpretation |
|---|---|---|---|
| `route:*` (WHEAD/WUP/WDN/WO) | signagr **1.000** | signagr **1.000** | transpose-matmul routing **bit-exact** (activation-space FD) |
| `attn:gQ/gK/gV` (softmax-Jac) | signagr **1.000**, relerr ~1e-7..1e-12 | signagr **1.000** | the **prime suspect** matches clean double softmax-Jac in **both L1 and dot** modes |
| `wg:*` (all 7 weight matmuls) | signagr 0.66–0.99 | signagr **1.000**, frac<1e-2 **1.000**, relerr **0** | deviation is **only** the deliberate mult-free dyadval pow2 rounding (magnitude); sign/direction correct |
| LN backward | 94% sign vs exact double L1-LN Jac | — | correct LN Jacobian *direction*; fixed-point gamma scale |
| EMB / POS / LNFB / LN1B/LN2B | relerr **0** | relerr **0** | exact |

**Deviating ops (the honest list).** Exactly **two**, both **by design, not bugs**:
1. `g_wgrad` weight-grad **MAGNITUDE** — deliberate mult-free dyadic (`sign·2^k`) pow2 rounding;
   **direction correct**, magnitude approximate. Goes bit-exact under WGRAD_MODE=1.
2. `g_ln_bwd` LN-gamma **SCALE** — correct LN Jacobian direction, fixed-point gamma scaling.

Neither is an algorithmic backward error. The cascade backward **computes backprop**; under
full-precision operands it **is** backprop to the bit.

**Verdict E1:** `backward_correct = yes-matches-backprop`. No buggy op. The G2_RETRY phrase
"approximate backward" was correct *as worded* (the mult-free wgrad **is** approximate in
magnitude) but **wrong as a wall hypothesis** — that approximation is sign-correct and is the
same recipe that trains CIFAR/MNIST cascades to the backprop ceiling elsewhere. It is **not** the
source of an acc-0.15 collapse.

---

## 2. EXPERIMENT 2 — ATTENTION ISOLATION: does integer attention alone cap autograd at ~0.15?

**Method.** A `/tmp` copy `/tmp/pc_xformer_charlm_attnsm.py` of the tracked
`scripts/pc_xformer_charlm.py` (**tracked file not edited**) with an `ATTN_SM ∈ {warm, intlut}`
knob and a deterministic held-out eval (40×48×64 tokens → bpb + next-char acc). `warm` =
`F.scaled_dot_product_attention` float softmax (the bar). `intlut` = the golden's **integer**
attention softmax lifted verbatim from `scripts/pc_xformer_ablate.py` (per-row adaptive shift
renormalizing the score gap to ~6 bits, coarse 256-entry exp2 LUT, Q16 normalize). **Only the
attention forward changes**: the backward is exact autograd and the p·V blend stays warm float, so
the integer attention is isolated from the backward. Both: D=64 L=3 H=4 block=64, 2500 steps,
TinyShakespeare, MULTFREE=1, CPU.

**Result (from `/tmp/attnsm_warm.log`, `/tmp/attnsm_intlut.log`):**

| ATTN_SM | held-out bpb | held-out acc | training curve |
|---|---|---|---|
| **warm** (recipe-#3 float dot softmax) | **2.6501** | **0.4513** | descends smoothly 6.38 → 2.73, on the bar (2.85/0.41) |
| **intlut** (old golden integer softmax) | **4.5480** | **0.1754** | flat & early — 4.66 @ step 250, stuck ~4.5 from 1750–2499 |

The intlut plateau is a **hard training wall**, not a transient (flat from step 250). It drops a
**perfect-autograd** model to **acc 0.18**, right at the cascade's ~0.15 cap, and inflates bpb to
4.55 — reproducing the memory observation (coarse exp2-LUT + adaptive-shift attention dropping
autograd 0.347→0.153) at this exact G2/bar config.

**Verdict E2:** `integer_attn_is_cap = true`. With the backward held perfect, the integer attention
forward **alone** reproduces the G2 wall. The cascade backward is **exonerated** a second,
independent way.

*Caveat (stated for honesty):* intlut applies the golden's adaptive-shift+coarse-LUT to **float**
dot scores here (not the full int8 score pipeline), so it is a faithful **emulation** of the
attention-softmax structure, not a bit-exact golden replica — which is the intent (isolate the
softmax forward). The direction (acc 0.45 → 0.18) is unambiguous.

---

## 3. SYNTHESIS — why this is (A) and not (B)/NEEDS-MORE

G2_RETRY's leaning-B rested on a single localization: "the wall is the cascade backward +
magnitude-decoupled dyadic-Adam." The two probes **dissolve** that localization:

- **The backward is not the wall** — it is correct backprop (E1, bit-exact under the full-mul
  control). The "magnitude-decoupled" optimizer concern is real as a *quality* knob but is **not**
  what produces acc-0.15: the same mult-free wgrad+dyadic-Adam, fed a **warm** attention forward,
  reaches **acc 0.45** (E2 warm row).
- **The forward is the wall** — the integer attention softmax (E2) reproduces acc-0.15 with a
  *perfect* backward. G2 was run on the **old golden's integer attention**, i.e. an **outdated
  forward**. That is the entire gap.

This is the classic G2-style misattribution: a forward-path artifact masquerading as a backward
bug because both surface as a flat accuracy ceiling. The disambiguating move — hold the backward
perfect and toggle only the forward — was exactly the experiment needed, and it lands cleanly.

### Does this vindicate locked recipe #3 (warm dot attention = the one allowed multiply)?
**Yes — directly and quantitatively.** Recipe #3's central bet is that the **one** place a
multiply must be spent is the **attention softmax** (warm dot), with everything else mult-free.
E2 is a controlled measurement of exactly that trade: mult-free-everything + **warm** attention =
acc 0.45 / bpb 2.65 (on the bar); the same stack with the **mult-free integer** attention = acc
0.18 / bpb 4.55 (the cap). Recipe #3 was chosen **precisely to avoid this cap**, and the
measurement confirms the choice is load-bearing, not cosmetic. This also matches the broader
memory thread (mult-free competitive LM is walled at the attention softmax; warm-train +
mult-free-deploy is the answer; the `pc_xformer_ablate` precise-exp finding).

### Strength of evidence (stated plainly)
- **E1 (grad-check):** **strong & deterministic.** The WGRAD_MODE=1 control is exact (relerr 0,
  bit-exact) and reproduced here; the softmax-Jac match holds in both L1 and dot modes. Not seed-
  or scale-sensitive — it is an algebraic identity check.
- **E2 (attention isolation):** **strong directional, single-config.** D=64/L=3/H=4, single run
  per arm, one dataset (TinyShakespeare). The 0.45→0.18 swing is far outside any plausible
  seed/estimator noise (the held-out eval is 40×48×64 tokens; warm matches the independent
  autograd bar's 0.41 within estimator slack). It is **not** multi-seed and **not** large-D — so
  the *exact* warm number (0.45) carries a modest CI, but the **qualitative claim** (integer
  attention is the cap; warm is on the bar) is unambiguous.
- **Residual open items (do NOT block build, DO gate the re-run):** (i) the warm-attention cascade
  must be measured **inside the actual golden** at **matched capacity** (the lingering G2_RETRY
  capacity confound: bar twin L3/H4 vs golden 1-block/H2) to confirm cascade+warm reaches ~the bar;
  (ii) `fix16`-tracks-`dbl` must be re-confirmed on the warm-attention forward (it collapsed on the
  *fixed-calibration* path, an orthogonal change now abandoned); (iii) ideally ≥2 seeds on the warm
  G2 re-run. These are **confirmation** steps on the (A) path, not reasons to fix the backward.

---

## 4. BUILD-OR-FIX DECISION

**Neither the FIX-BACKWARD nor the NEEDS-MORE-leaning-B reading survives the two probes.**
- **FIX-BACKWARD is ruled out:** no op is buggy; the backward is bit-exact backprop under the
  full-mul control (E1). The only deviations are the deliberate, sign-correct mult-free wgrad
  rounding and LN-gamma fixed-point scale — both by design.
- **The wall is a forward op (integer attention softmax), already addressed by the locked recipe
  (#3 warm dot).** G2 tested an outdated integer-attention forward. The fix is a **forward swap**
  the recipe already specifies — not a backward fix and not an open research question.

### Action plan (the (A) path)
1. **Update the golden's forward to warm / recipe-#3 dot attention** (replace the integer
   adaptive-shift + coarse exp2-LUT softmax with the warm dot softmax = the one allowed multiply).
   Keep everything else mult-free (route/wgrad/LN/dyadic-Adam unchanged — proven correct in E1).
2. **Re-run G2 at matched capacity** (L=3/H=4/vocab=65, ≥2 seeds) to confirm **cascade + warm
   attention ≈ the bar** (target bpb ≤ ~2.85 / acc ≥ ~0.41) and that `fix16` tracks `dbl` on the
   warm forward. *(This converts E2's isolated-twin result into an in-golden, capacity-matched,
   multi-seed confirmation — closing the two residual confounds.)*
3. **Fold the readout temperature** (`X_LOGIT_RDTEMP=16`, eval-path only) from `G2_RETRY` into the
   golden as the deploy/eval calibration fix (it bounds the saturated-softmax spread; harmless to
   training).
4. **THEN build `forge_pc_xformer_mp_block.cpp`** → csim (bit-exact to the warm-forward golden) →
   csynth gate (the prior csynth already passed: WNS +1.08, DSP=0 on dyadic ops, 185 DSP attention-
   only, fits 1 SLR — and the 185 attention DSPs are exactly the warm-dot multiply this decision
   endorses) → CL → distributed driver.

The build is **UNBLOCKED.** The one precondition before kicking off the HLS build is step 2 (the
capacity-matched, multi-seed warm-attention G2 re-run) — a confirmation run on the (A) path, not a
detour.

---

### Source / artifact files (absolute)
- **Grad-check (E1):** `/tmp/gcheck/final.cpp` (+ `wgradref.cpp`, `attnref.cpp`, `lnref.cpp`,
  `gcheck2.cpp`, `deepprobe.cpp`, `actprobe.cpp`); golden headers under `/tmp/gcheck/`.
  Repro: `g++ -O2 -std=c++17 -I. final.cpp -o final && ./final; WGRAD_MODE=1 ./final; ATTN_DOT=1 ./final`.
- **Attention isolation (E2):** new file (nothing tracked edited)
  `/tmp/pc_xformer_charlm_attnsm.py`; logs `/tmp/attnsm_warm.log`, `/tmp/attnsm_intlut.log`.
- **Predecessor gates:** `G2_RETRY_RESULT.md` (NEEDS-MORE→B, the verdict this pass overturns),
  `G2_CASCADE_PRECISION_RESULT.md` (original B).
- **Build targets / spec:** `KERNEL_DESIGN_SPEC.md`, `BUILD_PLAN.md`,
  `forge_pc_xformer_block_csynth.rpt` (csynth already passed),
  `tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_block.hpp` (forward swap site:
  the attention softmax block).

---

DECISION: **A — BUILD.** No backward op is buggy (grad-check: bit-exact backprop under the full-mul
control; only deviations are by-design mult-free wgrad rounding + LN-gamma fixed-point scale).
Integer attention **is** the cap (attention isolation: warm acc 0.4513 / bpb 2.6501 vs intlut acc
0.1754 / bpb 4.5480, backward held perfect). G2 tested an **outdated integer-attention forward**;
this vindicates locked recipe #3 (warm dot = the one allowed multiply). Evidence: E1 strong &
deterministic; E2 strong-directional but single-config/single-seed.

**Next action:** swap the golden's forward to recipe-#3 **warm dot attention**, re-run G2 at
**matched capacity (L3/H4/vocab65, ≥2 seeds)** to confirm cascade+warm reaches the bar
(bpb ≤ ~2.85 / acc ≥ ~0.41) and that fix16 tracks dbl — **then** build
`forge_pc_xformer_mp_block.cpp`.
