# Gate G2 (RETRY) — Cascade Backward vs the Autograd Bar

**Question the retry was built to answer:** The first G2 pass returned **(B) STOP — fix the
cascade**: the hand-rolled PC-cascade char-LM capped at bpb ≈ 5.39 / acc ≈ 0.15 (dbl, zero
format loss), ~3.4 bpb above a *prior* autograd reference, and never fell toward it. Two gaps
left that verdict "strong but not airtight": (1) no **same-harness autograd twin** to anchor the
bar, and (2) the diagnosis pointed at a frozen final-LN gamma but had not been mechanistically
nailed. This retry closes both, then **fixes → re-gates → verifies** — and this time the fix
sweep and verification **were executed**.

**Date:** 2026-06-27 · **Status:** COMPLETE — bar + diagnosis + fix sweep + verify all RUN.

## DECISION: **NEEDS-MORE** (evidence now leans back toward **B** — the cascade *backward* is the wall)

Everything was run this time. The bar is anchored, the diagnosis is mechanistically resolved
(with a load-bearing premise correction), **and the fix sweep + two independent verification
passes are complete.** The result is decisive and it is *not* (A):

- **No trainable calibration fix approached the bar.** The full ranked plan (fixed pre-CE logit
  temperature, fixed/capped CE-softmax temperature, head weight-decay, un-frozen LN gamma, lower
  fixed gamma-init) was swept. **Every one is a no-op or HURTS** — best calibration-corrected bpb
  stays at ~4.75–4.84, accuracy stays pinned at the cascade's structural **0.13–0.15**, ~2.0 bpb
  and ~3× accuracy short of the bar (bpb 2.85 / acc 0.41).
- **The ONE change that does real work is a deploy-time readout temperature** (a single mult-free
  `LG >>= 16` applied **only** when emitting probabilities, not in the cached forward that seeds
  the gradient). It recovers the model's *true* bpb from a saturated-softmax **readout artifact**
  (33.97 → 4.84 held-out, effective spread 2.7M → ~40) — but it does **not change training** and
  therefore **cannot move accuracy.** It cures a measurement pathology, not the model.
- So why not (B)? Because the residual gap is now precisely localized to the **cascade backward +
  normalized dyadic-Adam** (a feature/optimizer-quality ceiling, not calibration), and that is a
  *new, concrete, deeper finding* with a stated next hypothesis — but it is single-seed, small-D,
  and `fix16` does **not** track `dbl` on the fixed cascade (it collapses). The honest call is
  **NEEDS-MORE on the path to (B)**: the calibration hypothesis is refuted, the wall is
  re-localized, and the next probe (below) must run before we close the gate as a hard (B).

**Do NOT proceed to `forge_pc_xformer_mp_block.cpp` / csim / csynth.** Neither (A) condition holds:
(i) fixed-dbl bpb does **not** approach 2.85 (caps ~4.84, acc ~0.15); (ii) `fix16` does **not**
track `dbl` (opt 5.64 vs 4.83, acc collapses to 0.037). The winning readout-temperature fix is
worth folding in as a **deploy/eval-path** calibration fix, but it is not a build trigger.

---

## 1. THE AUTOGRAD BAR — established (same config, instrumented twin)

Built an instrumented `/tmp` copy (`/tmp/autograd_bar_charlm.py`; the tracked
`scripts/pc_xformer_charlm.py` was **never edited**) at the golden cascade's *exact* config:
XF_D=64, XF_LAYER=3, XF_HEAD=4, XF_BLOCK=64, XF_BATCH=48, XF_STEPS=2500, XF_WARMUP=120, real
TinyShakespeare (vocab 65), CPU, OMP_NUM_THREADS=8. Held-out eval over 40 val batches.

| twin | repr | EVAL bpb | EVAL acc | EVAL logit-spread |
|------|------|---------|---------|------|
| **MULTFREE=1** (mult-free fwd + autograd bwd — closest architectural twin to the golden cascade) | int/dyadic fwd, float bwd | **2.8492** | **0.4117** | 21.9 |
| MULTFREE=0 (full float — absolute ceiling) | float | 2.4230 | 0.5011 | 17.2 |

**chance = log2(65) ≈ 6.02.** Logs: `/tmp/autograd_mf1.log`, `/tmp/autograd_mf0.log`.

**The bar the cascade must approach: bpb ≤ 2.85, acc ≥ 0.41** (the MULTFREE=1 twin — same
mult-free forward as the kernel), with the float ceiling at bpb 2.42 / acc 0.50.

**Anchoring the prior G2 verdict:** the golden cascade (dbl) capped at **bpb 5.39 / acc 0.15**.
Against this same-config twin that is **~2.5 bpb above its own architectural twin** and ~2.97 bpb
above the float ceiling. So the cascade is far below what its architecture supports — exactly what
the first pass concluded, now anchored by a twin at the *identical* recipe. The single biggest
caveat from G2-v1 ("no same-harness autograd twin") is **closed.**

### The key spread evidence (the original hypothesis — now tested and refined)
- **Autograd MULTFREE=1:** logit spread rises 3.7 → 24.5 by step 1000, then **settles back** and
  oscillates ~20–24 (final 22.0, eval 21.9). **Bounded, self-regulating.**
- **Autograd MULTFREE=0 (float):** spread saturates smoothly to ~17.4 (eval 17.2).
- **Golden cascade (dbl):** spread runs away **28k → 430k → 1.3M → 2.7M → 5.2M** — ~5 orders of
  magnitude looser than either autograd twin.

The autograd twin keeps logits bounded via a **learnable LN-gamma** + **dyadic-Adam calibration**;
the cascade's **frozen** gamma supplies no pushback and the logits explode. This framed the
hypothesis: *the runaway is a calibration failure, not a representational ceiling.* **The fix
sweep below tested that hypothesis directly — and the calibration half is REFUTED.**

---

## 2. THE DIAGNOSIS — resolved, with a load-bearing premise correction

The G2 premise ("frozen fine-dyadic gamma can't go below 1.0 → can't calibrate") is **two-thirds
correct, one-third wrong**, and the wrong third changes the fix.

**Q1 — gamma freeze (CORRECT).** The LNF gamma/beta grad **is** computed (`g_ln_bwd`,
`forge_pc_xformer_golden_block.hpp:152`; MP fork `forge_pc_xformer_mp_golden_block.hpp:140`) into
`g.LNFG`/`g.LNFB`, then **never read** by `g_step`/`mp_step` — a **deliberate** freeze (commented
at `golden_block.hpp:236–238`: SignSGD ±1 on gamma drifts monotonically → LN stops bounding →
logits explode). The grads are written then discarded. *Subtle correction:* LN re-normalizes RP
every step, so frozen gamma keeps the LNF **output** bounded; the explosion is **downstream** —
**WHEAD magnitude grows** (it *is* trained), and gamma=64 (`<<6`) pre-amplifies XF before the
growing WHEAD → unbounded logit spread. "Frozen gamma" is a contributor, not the single lever.

**Q2 — the CE backward is NOT exactly `softmax − onehot` (the mechanism is real; the prescribed
fix is BACKWARDS — see §3).** It is an **adaptive-temperature exp2-LUT softmax minus onehot** in
Q16 (`golden_block.hpp:84–94`). Line 87 `smsh = g_normshift(mx − mn, C_SMBITS)` is a **per-token
adaptive temperature** that renormalizes the logit gap to ~6 bits **before** the exp. As the spread
grows 28k → 5.2M, `smsh` grows with it, so the softmax always sees a ~64-wide gap and **never
reflects the over-confidence** — the calibrating signal is divided out before it enters the
gradient. **This description is mechanically correct.** But the fix sweep proved that *forcing* a
fixed/capped temperature (so the softmax can saturate) makes things **worse**, not better: the
adaptive temperature is **load-bearing** (it keeps the CE gradient soft despite runaway logits;
saturating it reproduces the hard-argmax/MSE-top-error collapse from memory).

**Q3 — premise error: gamma is NOT fine-dyadic 1.0–1.75.** In `g_layernorm`
(`forge_pc_xformer_golden.hpp:114–115`), gamma `gcode[d]` is decoded by `g_dyad` to a **bare
power-of-2 exponent** and applied as a **pure shift** `g = r << ge`; init `LNFG=64 = 2^6 → <<6`.
The fine-dyadic `(nq/4)·2^e`, 1.0–1.75 format is applied **only to the `1/meanabs` normalizer**
(`golden.hpp:110/113`) — that is the "X_MBITS" object the premise conflated with gamma. So gamma
**can** go below 1.0 (64→32→16…) — but only in coarse octaves with monotonic-drift instability.
The sweep tested both un-freezing gamma and lowering its fixed init; **both hurt** (§3).

---

## 3. THE FIX SWEEP — **RUN** (all in /tmp copies; NO tracked file edited)

Implemented a toggle fork over `forge_pc_xformer_mp_golden_block.hpp`
(`/tmp/g2fix/g2_fix_block.hpp` + harness `g2_fix_test.cpp` + mechanism probe `probe_T.cpp`).
Baseline reproduced **bit-for-bit** (dbl `final_bpb=5.3934`, all toggles off). All numbers are
MODE=dbl (the cascade's **float ceiling** — zero format loss) unless noted, TinyShakespeare D=64
SEQ=64 2500 steps, same seed/order as G2. `bpb` below = the readout/`opt`-style number.

| combo | bpb | acc | verdict |
|---|---|---|---|
| **baseline (no fix)** | 4.7456 | 0.1502 | reference |
| FIX_TEMP=4 / =6 (pre-CE pow2 logit shift) | 4.746 / 4.7456 | 0.1502 / 0.1502 | **no-op** |
| FIX_TEMP=8 | 4.74 | 0.1343 | slightly worse |
| FIX_HEADNORM=6 / =8 (per-token head-input max-shift) | 4.75 / 4.75 | 0.1422 / 0.1366 | no-op/worse |
| **FIX_CE temp=6** (PRIORITY — fixed CE softmax temp, alone) | **13.68** | 0.1059 | **HURTS BADLY** |
| FIX_CE temp=8 | 13.7 | 0.0962 | hurts badly |
| FIX_CE_CAP=8 / =10 (cap adaptive smsh) | 13.7 / 12.4 | 0.0962 / 0.072 | hurts badly |
| FIX_TEMP=6 + FIX_CE temp=6 | 5.1537 | 0.0787 | worse |
| FIX_TEMP=8 + FIX_CE temp=6 | 4.8267 | 0.1303 | flat/worse |
| FIX_TEMP=8 + FIX_CE temp=8 | 4.84 | 0.1417 | flat |
| FIX_HEADNORM=8 + FIX_CE temp=6 | 13.9 | 0.0964 | hurts badly |
| FIX_WD_ALL=8 / =6 (proportional decay) | 4.7509 / 4.75 | 0.1502 / 0.1387 | no-op |
| **FIX_LNG** (un-freeze final-LN gamma) | 4.8919 | 0.0924 | **worse (spread → ~1e8)** |
| FIX_LNG_ALL (un-freeze all LN gamma) | 4.901 | 0.0857 | worse |
| LNFG_INIT=16 / =8 (lower fixed gamma <<4 / <<3) | 4.7428 / 4.8148 | 0.1398 / 0.1306 | no-op/worse |
| baseline 5000 steps (2× train) | 4.7394 | 0.1448 | acc plateaus |
| FIX_TEMP=8+CE6 5000 steps | 4.9062 | 0.114 | no help from more steps |
| **WINNING: deploy pow2 READOUT T=2^16** (held-out 60 batches) | **4.8388** | 0.1479 | recovers readout artifact only |

**The winning move — a deploy/eval-only readout temperature.** A single mult-free
`a.LG[i] >>= 16` applied **only when emitting probabilities/bpb**, NOT to the `GLOG` that seeds the
backward. It recovers the true bpb from a saturated-softmax readout artifact (**33.97 → 4.84**,
effective spread **2.7M → ~40**) and is the only change that does real work. It leaves training
untouched, so accuracy stays at the cascade's structural **0.148**.

**Everything trainable is a no-op or hurts.** The adaptive CE temperature divides any *constant*
logit rescale right back out of the gradient (so FIX_TEMP / FIX_HEADNORM / FIX_WD are no-ops), and
the spread is **structural** (gamma=64 × D-fold accumulation, not weight-growth) so weight decay
can't bound it. Forcing a fixed/capped CE temperature saturates the softmax into a hard
argmax-vs-onehot signal → the degenerate collapse (bpb 12–14, acc 0.07–0.10). Un-freezing gamma
lets it drift monotonically UP → spread explodes ~1e8 and acc drops to 0.086–0.092.

**reaches_bar: NO** — `best_bpb ≈ 4.84`, `best_acc ≈ 0.15`; the bar is bpb 2.85 / acc 0.41.

---

## 4. THE RE-GATE — **RUN** (four-mode intent; the decisive measurements are dbl + fix16)

The A-vs-B gate requires two measured conditions. **Both FAIL on the fixed cascade:**

- **(i) Does fixed-dbl approach the bar?** **NO.** The calibration-corrected (`opt`, best single
  pow2 readout temperature = the bpb the model would reach *if perfectly calibrated*) caps at
  **4.75–4.84**, still **~2.0 bpb ABOVE the 2.85 bar.** The residual gap is **accuracy** (0.148 vs
  0.41), which is argmax-based and therefore **scale/temperature-INVARIANT** — no readout fix can
  move it. A 5000-step (2×) run plateaus at acc 0.145. The accuracy wall is **structural.**
- **(ii) Does fix16 track dbl?** **NO.** On the *fixed* cascade (TEMP=8 CE6), fix16 does **not**
  track dbl: opt **5.64 vs 4.83**, acc **collapses to 0.0365** (2.3× chance), spread NOT bounded
  (654k vs dbl 11k). The earlier "fix16 within ~0.2 bpb of dbl" was measured on the **UNFIXED**
  cascade and does **not** carry over. fix16-tracks-dbl is therefore **NOT established.**

Spread is genuinely bounded by the winning readout fix (2.7M → ~40), so the **spread runaway is
solved** — but bounding the spread does not lift accuracy. The gate stays closed; the evidence
points at (B).

---

## 5. THE VERIFICATION — **RUN** (two independent passes, both `sound-with-caveats`)

Two independent audits rebuilt and re-ran the harness locally (g++ -O2, MODE=dbl, 2500 steps,
against `~/.cache/tiur/tinyshakespeare.txt`). Both returned **sound-with-caveats** and **agree the
decision must NOT flip to (A).**

**Pass 1 — the fix is principled, mult-free, kernel-clean (not a metric hack):**
- Forward stays **byte-identical / mult-free**: the temp op is a pure arithmetic right-shift on
  int64 logits (`g2_fix_block.hpp:106`); toggle OFF reproduces baseline bit-for-bit
  (`final_bpb=5.3934`, `final_ref=33.6108`). The single p·V attention multiply is untouched.
- The logit-spread bound is **real, not metric clipping**: the T=1 readout bpb of 33.6 is a
  *saturation artifact* (eval softmax one-hot on a 2.7M-wide logit); `probe_T.cpp` (60 held-out
  batches, independent) confirms best pow2 readout T=2^16 drops bpb 27.4 → 5.15 and bounds
  effective spread ~1.07M → ~16. The recovery calibrates a genuine logit; it is not a clip.
- Every op maps to the int/dyadic datapath (T=2^16 = single shift; FIX_LNG clamped `<<3..<<8`).
- **CAVEAT 1:** "gradient unchanged" is precise only for the **pure deploy-time** readout. The
  in-place FIX_TEMP toggle applied during training *does* change dynamics (spread 1.7M → ~50, acc
  0.1346 → 0.1328) but acc stays statistically ~0.13–0.15 and `final_opt` is identical → the temp
  recovers only the readout artifact, not quality. To ship as the gradient-free fix, the kernel
  must apply the shift **only on the readout/eval path**, not in the cached forward that seeds GLOG.
- **CAVEAT 2:** the win is a **readout/calibration** fix, not an accuracy fix; the ~2.0 bpb residual
  and 0.15 acc ceiling are the cascade's structural limit, untouched.

**Pass 2 — the build decision should NOT flip to (A); evidence leans (B):**
- Independently reproduced: bar 2.8492 / acc 0.4117 (real, `/tmp/autograd_mf1.log`, the MULTFREE=1
  twin); baseline dbl bit-exact (`glog=5.3934`, acc 0.1502, `ref`(T=1)=33.61, `opt`=4.7456); the
  4.8388 "fixed-cascade" number sits in the FIX_TEMP=8/CE6-family cluster (4.8258–4.8488).
- The load-bearing point: **`opt`-bpb IS the best achievable bpb under any single readout
  temperature**, so the residual gap is **NOT calibration** (already temperature-optimal) — it is
  **feature/backward quality**, which no temperature fix recovers.
- Confirmed the negative on the third honesty axis: **fix16 does NOT track dbl** on the fixed
  cascade (opt 5.64, acc collapses to 0.0365, spread not bounded).
- **CAVEAT (real soft spot):** config asymmetry not fully disclosed — the autograd bar twin is
  D=64 **/ L=3 / H=4 / vocab=65**, but the golden cascade is a **single block, H=2, FFN=256,
  vocab=64**. The vocab diff is flagged; the **layer/head capacity diff is not.** A 1-block/2-head
  model has a higher intrinsic floor, so *part* of the 2.85→4.8 gap may be **capacity**, not
  cascade-vs-autograd. The gap (and acc 0.41 vs 0.13) is too large to be capacity alone, but a
  **same-capacity twin** would make the anchored-bar claim airtight. **This is the one experiment
  that must run before closing the gate as a hard (B).**
- 4.8388 lives only in `/tmp/g2fix` scratch (sweeps print to stdout) — reproducible, not archived.
  Everything is single-seed.

---

## 6. THE DEEPER FINDING (why this is NEEDS-MORE → B, not A)

The original hypothesis was **calibration**: bounded logits via a learnable/adaptive normalizer
would close the ~2.5 bpb gap. **That hypothesis is now refuted by measurement.** The deeper finding:

1. **The spread runaway is real but is a READOUT pathology, not the wall.** A mult-free deploy-time
   readout temperature fully bounds it (2.7M → ~40) and recovers the true bpb (34 → 4.84). After
   that, the cascade's *calibration-corrected* bpb is ~4.75–4.84 — its honest deployable number.
2. **The residual gap is ACCURACY, and accuracy is temperature-INVARIANT.** The cascade is pinned
   at acc ~0.13–0.15 vs the twin's 0.41. No trainable fix and no amount of extra training moves it.
   This is a **feature-quality ceiling of the hand-rolled cascade backward + normalized dyadic-Adam**,
   not a calibration artifact.
3. **Mechanism (the next hypothesis to test):** the dyadic-Adam step is
   `sign(m)·2^floor(log2(|m|/v))` — a **normalized ~±1 step decoupled from gradient magnitude.** So
   even when CE screams "this logit is 1e6 too big," the per-weight corrective is ~±1/step while
   WHEAD×XF (gamma=64) grows the logit faster → the spread runs away **regardless of any
   calibration knob.** The wall is the **magnitude-decoupled optimizer step + the cascade's
   approximate backward**, not the softmax. **Next probe:** measure whether a magnitude-aware (not
   fully sign-normalized) dyadic step, and/or a closer-to-exact cascade backward, lifts the **0.15
   accuracy** ceiling toward the twin's 0.41 — at a **same-capacity** twin (L=3/H=4) to remove the
   capacity confound from §5.

So: **NEEDS-MORE** because (a) the gate's two conditions both FAIL, pointing at (B), but (b) the
capacity-asymmetry confound and single-seed scale mean the hard-(B) "cascade backward is the wall"
claim needs the same-capacity + magnitude-aware-step probe to be airtight.

---

## 7. WHAT TO FOLD INTO THE KERNEL NOW (the readout fix only — NOT a build trigger)

The sweep produced **one** kernel-implementable deliverable: the **deploy/eval readout
temperature.** Fold it into the canonical golden as a **readout-path** change (NOT training):

**Into `tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_block.hpp` (and the shared
`forge_pc_xformer_golden_block.hpp`):**
- In `g_forward`'s CE block, **after** `g_matmul(a.XF, w.WHEAD, a.LG, ...)` (`golden_block.hpp:81`),
  **only on the EVAL/READOUT distribution path** (the emitted probabilities/bpb), apply:
  ```
  for (int i=0;i<X_SEQ*X_VOCAB;i++) a.LG[i] >>= X_LOGIT_RDTEMP;   // X_LOGIT_RDTEMP=16, pure pow2 shift
  ```
  **Apply ONLY to the readout** — NOT to the `GLOG` that seeds the backward. The training gradient
  must stay on the un-tempered logits via the existing **adaptive-temperature CE**, which is
  **load-bearing** (a fixed/capped CE temperature collapses the model — see §3).

**Into `KERNEL_DESIGN_SPEC.md`:** add config const `X_LOGIT_RDTEMP` (pow2 shift integer, =16,
readout/eval-path ONLY); document that (a) gamma (`LNFG`) is a **pure pow2 shift**, not fine-dyadic;
(b) LN gammas/betas are **frozen** (un-freezing hurts — measured); (c) the CE-head adaptive
temperature `smsh = g_normshift(mx−mn, ...)` is **load-bearing and must be kept** (do NOT replace
with a fixed/capped temperature); (d) the readout temperature is a **calibration/deploy** fix that
recovers true bpb (34 → 4.84) but does **not** lift the cascade to the bar.

**DO NOT** add: fixed/capped CE temperature, un-frozen LN gamma, head weight-decay, lowered gamma
init, or an in-the-cached-forward logit shift — all measured no-op-or-worse.

---

## 8. HONEST CAVEATS

- **Single seed, small scale** (D=64, SEQ/BLOCK=64, CPU). The fixed-cascade sweep is single-seed;
  the hard-(B) verdict needs a multi-seed confirmation before any build commitment is foreclosed.
- **Capacity-asymmetry confound (the real soft spot):** bar twin = L=3/H=4/vocab=65; golden
  cascade = 1-block/H=2/FFN=256/vocab=64. Part of the 2.85→4.8 gap *may* be capacity, not
  cascade-vs-autograd. The 0.41-vs-0.13 acc gap is too large to be capacity alone, but a
  **same-capacity twin** is the one experiment that would make the verdict airtight. **Run it next.**
- **bpb-vs-accuracy:** the raw bpb is **inflated by the saturated-softmax readout** (can exceed
  chance even when argmax-acc > chance). The load-bearing, scale-invariant signal is the
  **argmax-accuracy plateau (~0.15 = ~9× chance)** vs the twin's 0.41 — that is what no fix moves.
  Use the calibration-corrected `opt`-bpb, never the raw T=1 bpb, for any comparison.
- **fix16 vs dbl:** earlier "within ~0.2 bpb" was on the **unfixed** cascade and does **not** carry
  to the fixed one (acc collapses to 0.037). fix16-tracks-dbl is **not established** for the fix.
- **The winning fix is a calibration/deploy fix, not an accuracy fix.** It bounds the spread and
  recovers true bpb; it does **not** make the integer cascade match backprop.
- **Provenance:** all work in `/tmp` copies; `git status` clean for `forge_kernels/` (no tracked
  file edited). 4.8388 lives in `/tmp/g2fix` scratch — reproducible, not archived/committed.

---

### Source / artifact files (absolute)
- Bar twin script (instrumented `/tmp` copy; tracked script untouched): `/tmp/autograd_bar_charlm.py`
- Bar logs: `/tmp/autograd_mf1.log` (MULTFREE=1, the bar), `/tmp/autograd_mf0.log` (float ceiling)
- Fix harness (all `/tmp` copies, no tracked edits): `/tmp/g2fix/g2_fix_block.hpp` (toggle fork
  over `forge_pc_xformer_mp_golden_block.hpp`), `/tmp/g2fix/g2_fix_test.cpp` (harness),
  `/tmp/g2fix/probe_T.cpp` (readout-temperature mechanism probe)
- Prior G2 verdict (this retry's predecessor): `/tmp/tt-pc/results/predictive_coding/multi_fpga/G2_CASCADE_PRECISION_RESULT.md`
- Kernel headers (diagnosis sites + transcription targets):
  - `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_golden_block.hpp` — gamma grad L152; CE adaptive-temp softmax L84–94 (smsh L87); g_step freeze L231–243 (comment L236–238); g_adam decay L226; head matmul L81.
  - `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_golden.hpp` — gamma pow2-shift apply L114–115; fine-dyadic invq (the 1.0–1.75 object) L41–60 used only at L110/113; g_normshift L37.
  - `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_block.hpp` — mp_step freeze L229–239; LNF grad computed-then-dropped L140.
  - `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_config.hpp` — X_MBITS (fine-dyadic normalizer, NOT gamma) L28–30; adaptive-softmax rationale L40–54; LR/WD shifts L57–66.
- Spec to update with the readout fix: `/tmp/tt-pc/results/predictive_coding/multi_fpga/KERNEL_DESIGN_SPEC.md`

---

DECISION: **NEEDS-MORE** (leaning **B**) — bar anchored (twin bpb 2.85 / acc 0.41), diagnosis
resolved, **fix sweep + two verifications RUN.** No trainable calibration fix approaches the bar;
the only real win is a deploy-time readout temperature that bounds the runaway spread (2.7M → ~40)
and recovers true bpb (34 → 4.84) but **cannot lift the structural acc-0.15 ceiling** (twin 0.41).
fix16 does NOT track dbl. The wall is re-localized to the **cascade backward + magnitude-decoupled
dyadic-Adam**, not calibration. (A) is NOT earned; hard-(B) needs the same-capacity + magnitude-
aware-step probe (capacity-asymmetry confound, single-seed).

**Next action:** run a **same-capacity** (L=3/H=4/vocab=65) cascade-vs-autograd-twin comparison
with a **magnitude-aware** (non-sign-normalized) dyadic optimizer step to test whether the
acc-0.15 ceiling is the cascade backward or the optimizer step — then close the gate as hard (B).
Fold ONLY the readout temperature (`X_LOGIT_RDTEMP=16`, eval-path) into the golden + spec now; do
NOT build `forge_pc_xformer_mp_block.cpp`.
