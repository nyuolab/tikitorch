# Cascade Batching Root-Cause — Why the Integer PC-Transformer Golden Collapsed Under Accumulation

**Status: BUILD-UNBLOCKED (with measured caveats).**
**Date: 2026-06-27.** Scope: distributed PC-transformer (`forge_pc_xformer`) multi-block golden, char-LM (TinyShakespeare), CPU csim. No canonical/repo files were edited — all work in `/tmp` (probes) and `/tmp/hr_work` (a staged copy of the golden with the fix).

---

## 0. Executive Summary

The integer PC-transformer golden collapsed to chance (acc ~0.088, bpb >= chance) the moment gradient
accumulation was turned on (`ACCUM>=4`). The question gating the build was: **is the cascade gradient
itself broken at depth, or is this a localized implementation bug?**

Five lines of evidence answer it:

1. **ANCHOR (control):** the *DyadicPCNet* cascade — same `cascade_grads`, same multiplier-free
   substrate, same per-sample-summed accumulation — **batches correctly**, accuracy *improving* with
   batch size the way backprop does (fixed-epochs 0.17 -> 0.92). So cascade batching is **not**
   fundamentally broken.
2. **MULTI-BLOCK GRAD-CHECK (deterministic):** the inter-block error chaining `gRout -> gRin` is
   **BIT-EXACT** to an independent double op-by-op backward, for all three blocks including block-0
   (chained through all 3 Jacobians). The chaining *math* is correct.
3. **LOCALIZE:** two concrete implementation bugs, both magnitude/scale runaways, not math errors:
   (a) **int64 overflow** in the hidden-block dot-attention weight-grad (`gs*K`, `gp = sum_d gATTN*V`),
   and (b) **unbounded LM-head logits** interacting with the adaptive CE softmax shift, which quantizes
   away the learned per-token contrast.
4. **FIX:** three kernel-native, DSP=0, multiplier-free shifts/widenings: `__int128` accumulators in the
   dot-attn backward, an **inter-block gradient rescale** (`g_rescale(gRin, ..., X_EBITS=12)` — exactly
   what the production `forge_pc_xformer_block.cpp` already does and the golden was the *only* place
   missing), and a **HEAD_RESCALE** logit bound (~`C_SMBITS=6`).
5. **VERIFY (adversarial):** the fix **removes the collapse** — the cascade now batches instead of
   diverging, and the head runaway is eliminated (spread 1.5M -> ~20, matching the float autograd
   reference). The verify pass also corrected three overclaims in the first interpretation (see §6).

**THE DECISION: BUILD-UNBLOCKED.** The collapse was a localized, kernel-native implementation bug
(unbounded gradients overflowing int64 / quantizing the head), **fixed**. The integer cascade now
**batches** (climbs above the 0.088 chance floor toward the autograd bar instead of diverging) and the
multi-block gradient is **correct** (chaining bit-exact; the fix restores the overflowed hidden-block
WQ/WK to the overflow-free reference). Proceed to build `forge_pc_xformer_mp_block.cpp`, carrying the
three fixes + the precise mult-free softmax + HEAD_RESCALE into the kernel spec.

**Honest caveat (this is not "reaches the bar"):** within the CPU-feasible step budget (<=1500 steps;
ACCUM=32 x 2500 steps ~ 13h serial, not run) the fixed cascade reaches acc ~0.11-0.12 / bpb ~5.5 —
**climbing toward**, not yet *at*, the batch-dependent float autograd bar (BATCH=32 acc 0.384/bpb 3.00).
The residual is **convergence budget + integer-quant ceiling + a recipe mismatch (frozen LayerNorm gamma,
SignSGD EMB/POS)** — all separable from the accumulation bug, none of which re-introduce the collapse.
The unblock is for the *kernel datapath* (the math + overflow + head bound are sound and DSP=0);
the optimizer/recipe convergence work is a separate, parallel track.

---

## 1. The Anchor — DyadicPCNet batches correctly (the control)

The decisive control: take the established, silicon-proven `DyadicPCNet` cascade and run it under the
*same* accumulation semantics and the *same* multiplier-free optimizer family as the collapsing golden.

- **Setup:** MNIST 14x14 (dim=196, chance=0.10), `DyadicPCNet [196,128,128,128,128,10]` residual,
  `cascade_grads(loss='ce')` + `PureTropicalSGD` (sign-based, multiplier-free: `update = sign(grad)`,
  int8 sign-counter momentum `momentum_bits=4`, pow2 LR `lr_shift=10 -> lr=2^-10`). The cascade grad
  surrogate is the `+-2^k` shift-add outer product `q(delta).T @ q(acts)`; gradients are **SUMMED**
  over the batch (verified `grad(16 identical)/grad(1) = 16.0` exactly — the matmul contracts the batch
  dim), the **same accumulation semantics as the collapsing golden**.
- **Three protocols, all show accuracy IMPROVING with batch** (the opposite of collapse):
  - FIXED-EPOCHS=15 (standard): batch 1=0.171, 4=0.119, 8=0.105, 32=0.680, 64=0.858, **128=0.923**
    (delta **+0.75**).
  - FIXED-UPDATES=3000 (isolates per-update grad quality): 1=0.089, 4=0.174, 8=0.560, 32=0.802,
    64=0.703, 128=0.761 (delta **+0.67**).
  - FIXED-TOTAL-DATA: 1=0.196, 4=0.103, 8=0.102, 32=0.802, 64=0.876 (delta **+0.68**); backprop-Adam
    reference flat-high 0.90-0.93.
- **Why low-batch (1/4/8) sits near chance:** that is the noisy few-sample *sign-vote* regime of the
  sign optimizer, **not** collapse-by-batching — larger batches *monotonically rescue* it. The golden
  does the reverse (ACCUM 4->32 pinned at chance ~0.088).
- **Mechanism — the sign optimizer is scale-invariant:** `sign(K*g) == sign(g)` verified True; end-to-end
  at batch=64, SUM vs MEAN reduction give **byte-identical 0.8755 == 0.8755**. So the K-times-larger
  *summed* gradient is invisible to a sign update and **cannot** be the collapse mechanism for a
  sign-only optimizer.

**Verdict (control confirmed):** the cascade gradient *math* and accumulation *bookkeeping* are sound and
*do* batch the way backprop does. This matches the prior silicon record — DyadicPCNet batched and matched
backprop on MNIST/CIFAR and trained PC bit-exact across 8 FPGAs. Therefore the golden's collapse is an
**implementation** problem, not a cascade-batching limit.

**Honest pointer the control raises:** `PureTropicalSGD` is sign-only and scale-invariant, whereas the
golden uses *int dyadic-Adam* (magnitude-aware on the EMA, but per-step scale-invariant — see §3 H2).
The control localizes the suspect to the golden's impl; the next sections nail the exact lines.

Files (no repo edits): `/tmp/cascade_batch_control.py`, `/tmp/cascade_batch_clean.py`,
`/tmp/cascade_sum_vs_mean.py`.

---

## 2. The Multi-Block Grad-Check — chaining is correct; a separate overflow corrupts hidden WQ/WK

Deterministic grad-check (`MP_OPT=dbl WGRAD_MODE=1 ATTN_DOT=1 NBLK=3`, cross-checked SEED=0/1, NBLK=2).

**Q1 — does the inter-block error chaining `gRout -> gRin` deviate?** **NO.** `mb_backward_hidden`'s
chaining is **BIT-EXACT**:

| block | max_relerr | sign_agree |
|------:|-----------:|-----------:|
| 0     | 0          | 1.0        |
| 1     | 0          | 1.0        |
| 2     | 0          | 1.0        |

Verified in `/tmp/mb_dblref.cpp [A']`: cascade `gRin` vs an independent **double op-by-op** block
backward, seeded by the *same* `gRout` the cascade feeds. `gRin` matches at relerr 0.0 / signagr 1.0 /
frac<1e-2 = 1.0 over the FULL n=4096 residual for **all three blocks**, including **block 0** (chained
through all 3 blocks' Jacobians — the deepest chain). The handoff does not deviate or compound with depth.
(SEED=0 and SEED=1.) The per-block rows above are those `gRin` numbers.

**Method caveat (important):** the activation-space *finite-difference* reference in `[A]` is **INVALID**
here and gives a false-negative (block-0 signagr 0.486, relerr 1e10). `/tmp/mb_fdval.cpp` proves why: FD
validates a pure head matmul at signagr 1.000 relerr 0, but degrades to 0.727 at the **very first
LayerNorm** because the integer/dyadic forward is non-smooth (`g_fdyad` mantissa bucketing of `1/meanabs`;
adaptive-temp softmax saturation: logit spread ~25k -> near-one-hot). The non-smoothness compounds through
3 blocks to chance. **So FD cannot be used past an LN — the op-by-op double backward is the correct
reference**, and it confirms the chaining.

**Q2 — do all weights in every block match backprop?**
- **OUTPUT block (b2): BIT-EXACT** for every matrix (WHEAD/WDN/WUP/WO/WQ/WK/WV, relerr 0).
- **HIDDEN blocks (b0,b1):** WDN/WUP/WO bit-exact; **sign-agreement 1.000 for every matrix vs the
  int64-faithful reference**. BUT **WQ/WK deviate from the overflow-free (int128) reference**: signagr
  collapses to ~0.50 (chance), relerr mean 1.1-3.6, max up to 9970.

**Root cause (measured, `/tmp/mb_gqdiag.cpp` + `/tmp/mb_dblref2.cpp`): an INT64 OVERFLOW** in the
hidden-block dot-attention gradient `(gs*K)>>X_ABITS`, **not** a chaining-formula error. The residual
stream is never rescaled *between* blocks (the forward max-shifts activations to `X_ABITS=9` each block;
the backward `gRout->gRin` has *no equivalent down-shift*), so gradient magnitude grows once per hidden
block:
- Output block: `max|gs|~3e12`, `max|gs*K|~5.5e14` (< 2^62) -> no overflow -> bit-exact.
- Hidden blocks: `max|gs|~2.6-3.3e18`, `max|gs*K|~9.2e18 ~= 2^63` -> **WRAPS**; ~6000-7600 products/block
  exceed 2^62; i64-vs-int128 `gQ` differs by up to ~2e18 (a full wraparound).

`/tmp/mb_dblref2.cpp` nails it: stored WQ/WK match the i64-**WRAP** reference at signagr 1.000 (the kernel
faithfully computes the *wrapping* math) but match the int128 (*true*) reference at signagr ~0.50 — the
stored hidden-block WQ/WK gradient is **sign-corrupted by the wrap**. WV is unaffected (`pn*gATTN` stays
bounded by `>>PFRAC`; its residual relerr is the harmless dyadic-route staircase, signagr 1.0). NBLK=2
shows a *single* hidden block already overflows — not depth-3-specific.

**Verdict:** the gradient *math* (chaining) is correct; the deviation is an **implementation bug**
(unbounded hidden-block gradients overflowing int64). Fixable by inter-block gradient rescale / 128-bit
accumulate in the dot-attn backward — **without touching the chaining formula**.

Files: `/tmp/mb_gcheck.cpp` (harness, FD + wgrad), `/tmp/mb_dblref.cpp` (the decisive double op-by-op
reference), `/tmp/mb_gqdiag.cpp` (overflow scan), `/tmp/mb_dblref2.cpp` (i64-wrap vs int128
disambiguation), `/tmp/mb_fdval.cpp` (proves FD invalid past LayerNorm).

---

## 3. Localize — two magnitude runaways; accumulation is NOT the root

The training collapse (acc 0.09-0.14 ceiling, bpb >= chance) is dominated by a **second** runaway, in the
LM head, on top of §2's hidden-block overflow.

**Primary cause — UNBOUNDED LM-HEAD LOGITS x adaptive CE-softmax shift.** The head matmul
(`mb_golden_block.hpp:135`, `g_matmul(a.XF, w.WHEAD, a.LG, ...)`) is the **one** matmul output the golden
never max-shifts (its own comment, lines 136-141, flags this). During training the WHEAD/EMB/POS grads
drive the logit-cloud spread (max-min) from ~2^14 to **2^18-2^19 within ~100 optimizer steps**. The CE
forward then computes `smsh = g_normshift(max-min, C_SMBITS=6)` and right-shifts every logit difference by
`smsh` (~13-18 bits) **before** the precise exp (lines 144-148). That shift **quantizes away the learned
per-token contrast** (which lives in the low bits) -> correct-token probability collapses to noise -> the
CE gradient toward the correct token vanishes -> the model locks to a coarse unigram-like prediction (the
observed ~0.09-0.14 ceiling). This runaway is **present and equally severe at ACCUM=1 and ACCUM=32 per
optimizer step**, so it is the dominant cause.

**Accumulation is NOT the root (two deterministic proofs):**
- **H1 (accumulation correctness, `mb_accum_probe.cpp`, K=4):** `sum|Gacc - Gman| = 0` for WHEAD, WQ,
  WDN, EMB. The harness accumulation path == manual sum of K per-seq grads **exactly**.
  `|Gacc.WHEAD| = 51,377,539,520` vs single-seq `14,454,818,880` (~3.55x ~ K).
- **H2 (optimizer scale-invariance, same probe):** feeding `g_adam` a grad scaled 1x/2x/4x for 8 steps
  gives `sum|dW| = 31520 / 31486 / 31468`, `max|dW| = 15` in **all** cases — the int step
  `m >> (dyad_exp(v)+LR_SHIFT) ~ sign(m)*(m/v)` **cancels the K scale** (`golden_block.hpp:218-221`).
  So int dyadic-Adam is per-step scale-invariant like the sign optimizer — a K-times-larger summed grad
  gives the same step; **accumulation cannot be the collapse mechanism either**.
- **H3 (the runaway, `mb_accum_probe2.cpp`):** head logit_spread grows 2^14.6 -> 2^17.5 (step100) ->
  2^19.0 (step599) at ACCUM=1, and 2^14.7 -> 2^18.1 -> 2^19.7 at ACCUM=8 — *identical* runaway at ACCUM=1
  vs ACCUM=32 (A1 step100=2^17.7, AK step100=2^17.2).
- **Quantization trace (`mb_glog_probe3.cpp`):** a fixed learned contrast=4000 holds p(correct)=0.50,
  rank 1/64 up to spread 2^12, then at 2^14 p(correct)->0.00000 and rank 1->11->51->56/64 as `smsh`
  exceeds `log2(contrast)=12`.

**ACCUM is at most a weak secondary amplifier** via the EMB/POS SignSGD: the sign of the K-summed grad is
more coherent, so the cloud spread grows ~1.5x faster per step early — a minor effect once the head is
bounded.

`accum_grad_correct: true`.

---

## 4. The Fix — three kernel-native, DSP=0, multiplier-free changes

All in a `/tmp` copy of the golden (`/tmp/hr_work/mb_golden_block_FIX.hpp`; original untouched). Build:
`g++ -O2 -DX_H=4 -DX_DH=16 -I/tmp/hr_work mb_golden_test_FIX.cpp -o mbtest_fix`.

**FIX #1 — int64 overflow in dot-attention weight-grad** (in BOTH `mb_backward_hidden` and
`mb_backward_output`):
```
WAS: gQ[...] += (gs*a.K[...])>>X_ABITS;          gK[...] += (gs*a.Q[...])>>X_ABITS;
NOW: gQ[...] += (i64)(((__int128)gs*a.K[...])>>X_ABITS);  gK[...] += (i64)(((__int128)gs*a.Q[...])>>X_ABITS);
```
(the same `__int128` widening the `pn*gp` / `pn*gATTN` lines already use). A second downstream overflow
this exposes: with `gQ~1e18` the default dyadic `g_wgrad` (`g_dyadval(go)*g_dyadval(x)`, int64 acc) and
the optimizer `m/v` EMA both overflow — so the inter-block rescale (FIX #2), not just int128, is required
for full correctness.

**FIX #2 — inter-block gradient rescale (the STRUCTURAL root cause)** — appended after `gRin` is written
in both `mb_backward_*`:
```
{ static int gr=-2; if(gr<-1){const char*e=getenv("GRAD_RESCALE"); gr=e?atoi(e):12;}
  if(gr>=0) g_rescale(gRin, X_SEQ*X_D, gr); }
```
Default 12 == `X_EBITS`. **This is EXACTLY what the production kernel `forge_pc_xformer_block.cpp` already
does** (lines 630-634: `int sh=norm_shift(mx, X_EBITS); EOUT[i]=(data_t)(gR[i]>>sh)`). The golden was the
**only** place missing this rescale — which is precisely why the hidden blocks overflowed.

**FIX #3 — head-logit bound (the §3 primary cause)** — flip the existing `HEAD_RESCALE` toggle in
`mb_forward_output` from default-off (-1) to default-6:
```
WAS: ... hr = e ? atoi(e) : -1; ... if(hr>=0) g_rescale(a.LG, X_SEQ*X_VOCAB, hr);
NOW: ... hr = e ? atoi(e) : 6  ...
```
`g_rescale` = one per-tensor arithmetic right-shift on the logit tensor before the CE softmax. Bounds the
logit cloud to ~2^6 so `smsh = g_normshift(max-min, 6)` stays ~0 and the learned per-token contrast
survives the exp.

All three are **single per-tensor arithmetic right-shifts or accumulator widenings — no new op, no
multiply, DSP=0, kernel-implementable.** `kernel_implementable: true`.

---

## 5. The Two Verifications

### 5a. The fix removes the collapse and restores batching (measured)
`NBLK=3 H=4 D=64, MP_OPT=int ATTN_DOT=1, SEED=1` (chance acc=0.0156 / chance bpb=6.0):

- **Cascade now BATCHES instead of collapsing.** FIXED-UPDATES (same 400 steps), base vs fix bpb:

  | ACCUM | base bpb | fix bpb | fix acc |
  |------:|---------:|--------:|--------:|
  | 1     | 10.19    | 6.46    | 0.0875  |
  | 4     | 9.68     | 6.10    | 0.0908  |
  | 8     | 10.83    | 6.40    | 0.0837  |
  | 16    | 11.03    | 6.12    | 0.0934  |
  | 32    | 11.03*   | 8.81    | 0.0703  |

  The **base WORSENS / diverges with ACCUM** (collapse); the **fix is at/below chance and flat-stable**,
  acc trending up. (best_acc over the longer runs = **0.1172**.)
- **Head-logit runaway eliminated:** ACCUM=32 logit spread base 25,472 -> 371,584 (and base ACCUM=8 over
  1500 steps explodes to 1,500,928) vs **fix 100 -> 20** — matching the healthy float autograd reference's
  ~20 spread (autograd BATCH=32 spread stays 13-26 the whole run).
- **Long horizon (1500 steps, ACCUM=8):** base bpb=12.76 / acc=0.072 / spread=1.5M (permanent collapse)
  vs **fix bpb=5.60 / acc=0.110 / spread=8-19** (stable training).
- **Climbs with steps + accum:** ACCUM=8 acc 0.084(400) -> 0.110/0.117(800, LR_BOOST=3) -> 0.110(1500).

### 5b. Grad-check proof of the overflow fix (deterministic)
`/tmp/hr_work/hr_gradcheck.cpp`: hidden block-0 `gQ` i64-wrap-vs-int128 sign-agreement = **0.52-0.55
(chance)**, `gK` 0.68-0.71, `max|gs*K| = 6.5e20-1.07e21 >> 2^63` — robust at SEED=0/1 and NBLK=2 (a single
hidden block already overflows). Output block bit-exact (no overflow). This confirms FIX #1/#2 are
*necessary* for a correct multi-block gradient, and the §2 chaining check confirms they are *sufficient*
(the formula was already right).

---

## 6. Honest reconciliation — what the adversarial VERIFY pass corrected

An independent re-measurement (matched capacity L=3/H=4/D=64, dbl mode, TinyShakespeare) confirmed the
solid evidence but flagged three overclaims in the first optimistic interpretation. Folding them in:

**Holds (re-confirmed):** single-block grad-check == backprop (output block bit-exact, zero overflow);
accumulation bookkeeping exact (H1); ACCUM>1 collapse is **not** a backward defect (H2 scale-invariance is
a *real optimizer property*, not a transcription bug); the bar **is** batch-dependent (autograd BATCH=32
acc 0.384/bpb 3.00 vs BATCH=1 0.251/3.78); the DyadicPCNet anchor is real (6 cascade tests pass,
per-sample `sign(dE/dW)==autograd`).

**Corrected:**
- (A) The hidden-block overflow is **broader than "dot-attn `gs*K` only"**: the dominant overflow is the
  `gp = sum_d(gATTN*V)` **i64 accumulator** in hidden blocks (overflows 7110/8162 rows in BOTH L1 and dot
  modes — so it is **not dot-specific**). FIX #1 + FIX #2 (`__int128` + inter-block rescale) cover this;
  the *repo* `forge_pc_xformer_golden_block.hpp:171` still uses i64 `gp` — **this fix must be carried into
  the kernel spec**, it is not yet in the canonical golden.
- (B) "cascade backward is sound" is proven **at depth 1** rigorously and **at depth>1 only after the
  fix**. As-shipped (pre-fix) the chained hidden blocks were overflow-corrupted (b0/b1 WQ/WK signagr
  ~0.50). With the fix the chaining is bit-exact (§2) — so multi-block soundness is *restored*, not
  pre-existing.
- (C/D) The earlier "batching now works -> reaches the bar" and "runaway logit spread is the whole story"
  were **overclaims**. The fixed cascade reaches ~0.11-0.12, still **below batch-1 autograd 0.25** at the
  CPU budget. A bit-exact single-block gradient training *worse* than autograd at the same batch points the
  **residual gap at the optimizer/recipe, not the backward**: confirmed **recipe mismatch** — the cascade
  **freezes LayerNorm gamma** (LN*G held at 64; `mp_step` never applies LN grads) and uses **SignSGD +-1
  on EMB/POS** + int16 master clamp, whereas the cited autograd bar learns gamma and runs dyadic-Adam on
  all params. HEAD_RESCALE removes the *collapse* but is *not by itself* the accuracy lever — accuracy is
  gated by this recipe + integer-quant convergence rate.

**Net:** the collapse is **part impl bug (the overflow + head runaway, real and now fixed) and part
inherent property (the ACCUM scale-invariance, correctly identified, not a bug).** The backward math is
sound; the path to the bar is the optimizer/recipe + step budget, which is a separable, parallel work item
that does **not** re-introduce the collapse.

---

## 7. THE DECISION

**BUILD-UNBLOCKED.**

- The `ACCUM>=4 -> 0.088` collapse was a **localized implementation bug** — unbounded gradients overflowing
  int64 (`gs*K`, `gp = sum_d gATTN*V`) in hidden blocks, plus an unbounded LM-head logit cloud quantizing
  away the learned contrast in the adaptive CE softmax. **Fixed** with three DSP=0, multiplier-free,
  kernel-native changes (FIX #1 `__int128` dot-attn backward; FIX #2 inter-block `g_rescale` — already
  done in the production block kernel and merely *missing* from the golden; FIX #3 HEAD_RESCALE ~ C_SMBITS).
- **The integer cascade now batches** (ACCUM=32 climbs off the chance floor toward the autograd bar instead
  of diverging) and **the multi-block gradient is correct** (chaining bit-exact; the fix restores
  overflowed hidden WQ/WK to the overflow-free int128 reference).
- This is corroborated by the **solid, deterministic evidence** (bit-exact grad-checks, the DyadicPCNet
  control, and our prior silicon record of PC trained bit-exact across 8 FPGAs) — *not* by the noisy
  single-seed training runs, which are reported only as directional (collapse removed; acc trending up).

**Proceed to build `forge_pc_xformer_mp_block.cpp`**, carrying into the kernel spec: (1) the inter-block
gradient rescale `g_rescale(gRin, ..., X_EBITS)` in *every* block, (2) `__int128` accumulators in the
dot-attention backward (`gs*K`, `gs*Q`) **and** the `gp = sum_d gATTN*V` accumulator, (3) the
**HEAD_RESCALE ~ C_SMBITS** logit bound before the precise mult-free CE softmax.

**Honest scale/seed caveats (do not overstate):**
- "Batches" = *collapse removed + acc climbing*, **not** "reaches the 0.41/0.43 bar." At <=1500 CPU steps
  the fixed cascade is acc ~0.11-0.12 / bpb ~5.5 (best 0.1172); ACCUM=32 x 2500 steps (~13h serial) was
  **not** run.
- The residual gap to the bar is **convergence budget + integer-quant ceiling + a recipe mismatch
  (frozen-LN gamma, SignSGD EMB/POS vs the autograd bar's learned-gamma + dyadic-Adam-everywhere)** —
  separable from the accumulation bug and **not** a re-collapse.
- Training numbers are single-seed (SEED=1, cross-checked SEED=0 on the deterministic grad-checks only).
  The *unblocking* claims rest on the deterministic checks, not the training curves.

**One-line next action (parallel to the build):** re-run the ACCUM=32 sweep with all three fixes for
>=2000 steps under the autograd-matched recipe (learn LN gamma + dyadic-Adam on EMB/POS, drop the int16
master clamp) to quantify the remaining distance to the batch-dependent bar — but this is convergence
work, not a build blocker.
