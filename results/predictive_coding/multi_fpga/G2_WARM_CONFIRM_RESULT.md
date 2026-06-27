# G2 Warm-Confirm Gate — Result

**Status: BOTH GATES FAIL → BUILD NOT CONFIRMED.**
**DECISION: NEEDS-MORE — fix the hand-rolled cascade backward (not softmax, not format, not capacity).**

This gate asked whether we may proceed to implement the fix16 mp kernel
(`forge_pc_xformer_mp_block.cpp`) with warm dot attention. The precondition was a
two-part AND:

1. **(G2a)** dbl cascade + warm attention reaches the autograd bar
   (acc ≥ ~0.41 / bpb ≤ ~2.85) at matched capacity (L=3, H=4, D=64), **AND**
2. **(G2b)** fix16 tracks dbl, **AND**
3. **(G2c)** the warm softmax is a mult-free shift-add exp with no DSP escalation.

G2c **passes** (the softmax is fully de-risked, see below). G2a and G2b **fail**.
Because the AND is not satisfied, the build is **not** confirmed. The residual wall
is isolated to the **cascade backward**, not to anything that a kernel-format or
softmax change would fix.

---

## 1. Softmax-precision sub-result (G2c) — PASS

**Question:** which softmax precision reaches the char-LM autograd bar?

| mode   | acc    | bpb    | reaches bar |
|--------|--------|--------|-------------|
| warm (float exp) | 0.4422 | 2.6887 | yes (reference) |
| **precise** (mult-free shift-add ROM exp) | **0.4419** | **2.6871** | **yes** |
| lut8 (coarse adaptive-shift) | 0.164 | 4.6472 | no |

- **2-seed** sweep, D=64 / L=3 / H=4 / block=64, 2500 steps, TinyShakespeare,
  `MULTFREE=1`, CPU, held-out eval (40 batches × 48 × 64 tokens).
- The **precise mult-free exp reaches the warm/float bar essentially exactly**:
  acc gap precise−warm = **0.0003** (0.4419 vs 0.4422), inside the ±0.05 tolerance
  and *inside seed noise* (the two warm seeds themselves spread 0.018 acc). bpb is
  identical (precise 2.687 vs warm 2.689 — precise marginally *lower*).
- The two precise seeds track the two warm seeds **across the entire training curve**
  (e.g. step 1000 bpb precise 2.998/3.086 vs warm 2.992/3.077; step 1750
  2.854/2.767 vs 2.826/2.775).
- **Coarse lut8 caps far below** (acc 0.164, bpb 4.65) and *drifts upward* during
  training (lut8_s1 bpb 4.30→4.75 over steps 250→2499) — the coarse adaptive-shift
  `2^(-g/8)` bucketing actively destabilizes attention. This reproduces the
  pc-xformer-softmax-ceiling memory (~0.153).

**Kernel implication:** keep the **mult-free shift-add ROM exp** — `u = d·log2(e)`
via `x·1.4375 = x + x>>2 + x>>3 + x>>4`, a 256-entry mantissa ROM (`EXPFRAC=8`),
octave via `>>int`. **No true float exp, no DSP escalation for the softmax.** The
only attention multiply that remains is the explicitly-allowed int64 Q·K dot +
p·V blend (~185 DSP). Implementation lifted verbatim from
`scripts/pc_xformer_ablate.py` `SOFTMAX=precise` (`PreciseSoftmax`). No canonical
golden forward was modified.

**Caveat (honestly stated):** small-scale isolation (0.16M param, 2500 steps,
2 seeds). Backward for both int modes is the *exact softmax-of-those-probs gradient*
(autograd) and the p·V blend stays warm, so the *only* changed variable vs warm is
the forward attention map. This strongly de-risks the shift-add ROM exp; a
larger-scale confirm is advisable before final AFI budget lock.

---

## 2. Warm-golden cascade sub-result (G2a / G2b) — FAIL

Cascade + warm attention (dot Q·K + precise exp) + matched capacity (L=3, H=4,
D=64), 2500 steps, TinyShakespeare, multi-seed.

### dbl (the cascade's float ceiling — zero quantization)
| seed | bpb | acc |
|------|-----|-----|
| 0 | 5.3453 | 0.1477 |
| 1 | 5.4198 | 0.1347 |

→ **dbl_reaches_bar = false.** ~0.14 acc vs ≥0.41 bar; ~5.3 bpb vs ≤2.85 bar.
dbl reaches only **~33%** of the bar accuracy.

### fix16
| seed | bpb | acc |
|------|-----|-----|
| 0 | 15.9382 | 0.0009 (DIVERGED — logit spread 333M, prob pinned at floor) |
| 1 | 5.2284 | 0.1525 |

→ **fix16_tracks_dbl = false.** fix16 is unstable (seed0 diverges without
HEAD_RESCALE; with HEAD_RESCALE=10 it gives 0.127/0.044) and at best matches a
*failing* dbl ceiling. Moot anyway, since dbl itself is near chance.

**BOTH GATES FAIL → build of `forge_pc_xformer_mp_block.cpp` is NOT confirmed.**

---

## 3. Triangulation — the residual is the CASCADE BACKWARD

Ablation, all L=3 / H=4 / dot warm attn / precise exp / TinyShakespeare / 2500 steps:

| config | acc | bpb | bar? |
|--------|-----|-----|------|
| autograd, matched (dot/float-LN/float-act/AdamW, BATCH=32) | 0.486 | 2.414 | PASS |
| autograd, **full kernel recipe** (fdyad-LN + i16-act + dyadic-W + dyadic-Adam), BATCH=32 | 0.426 | 2.763 | PASS |
| autograd, full recipe, **BATCH=1** (like cascade ACCUM=1) | 0.19–0.34 (noisy) | — | fail |
| **dbl CASCADE**, warm dot attn, ACCUM=1 | 0.13–0.15 | 5.3–5.4 | FAIL |
| dbl cascade, ACCUM=4 | 0.088 (chance) | — | fail |
| dbl cascade, ACCUM=32 | 0.088 (chance) | — | fail |

What this isolates:
- **Not the integer format** — dbl is the float ceiling and still fails.
- **Not the optimizer** — autograd + dyadic-Adam = 0.481.
- **Not the softmax precision** — precise mult-free exp + warm dot attn used
  throughout; bounding the logit spread via HEAD_RESCALE did not lift acc.
- **Not capacity** — L=3 / H=4 matched the bar twin exactly.
- **Not vocab** — 64 vs 65 is negligible (chance bpb 6.000 vs 6.022; cascade is at
  ~5.3, nowhere near chance).
- **It IS the hand-rolled cascade backward** (`g_route` / `g_wgrad` / `g_ln_bwd` /
  `__int128` attention bwd). Every substrate component is fine under autograd; the
  cascade backward does NOT track autograd at this 3-block/4-head scale — which
  **contradicts the earlier Z-IL "cascade just computes backprop" claim for THIS
  config**. The single-block cascade already caps ~0.12–0.20, so the defect is
  present at depth 1 and compounds with depth.
- The cascade also **cannot batch**: ACCUM>1 collapses to chance (predicts the
  unigram), locking it at the noisy ACCUM=1 ceiling (~0.14) while the autograd bar
  itself is batch-dependent.

**GAP:** ~0.27 acc (0.14 vs 0.41) / ~2.5 bpb (5.3 vs 2.85).

---

## 4. Code changes (all benign; bit-exactness preserved)

- **Canonical (1 line):** wrapped `X_H` in `#ifndef` in
  `forge_pc_xformer_config.hpp` so the bar twin's H=4/DH=16 can be set via
  `-DX_H=4 -DX_DH=16`. Default still `X_H=2`; the int kernel and every existing
  golden compile/run unchanged.
- **New /tmp experiment files** (never touch canonical patterns):
  - `/tmp/pc_xformer_charlm_softmax_precision.py` — softmax-precision sweep
    (extends `/tmp/pc_xformer_charlm_attnsm.py`).
  - `/tmp/mb_golden_block.hpp` — multi-block cascade golden
    (`mb_forward/backward_hidden` + `_output`, verbatim copies of `g_forward`'s
    block portion and `mp_backward`'s body; reuses `g_layernorm/g_matmul/g_route/
    g_ln_bwd/mp_wgrad` + `__int128` attention bwd). Adds the **HEAD_RESCALE** toggle
    (env, default off): the kernel's existing mult-free max-shift on head logits
    `a.LG` before CE, behind `if(hr>=0)`. Kernel-implementable (no new op); does
    NOT alter int bit-exactness when unset.
  - `/tmp/mb_golden_test.cpp` — driver; per-seed init + data order; **NBLK=1
    verified BIT-IDENTICAL** to the original single-block mp golden
    (bpb 5.9156, spread 1110528 match exactly).
- **Warm attention needed no new canonical code:** `ATTN_DOT=1` already exists in
  `forge_pc_xformer_golden_block.hpp` (L57/L61, dot Q·K >> X_ABITS, the 185-DSP
  allowed multiply) and the precise mult-free exp `g_exp_q16` is already the
  golden's default softmax path.

---

## 5. Independent verification — SOUND and HONEST

All five skeptical checks pass; every reported number reproduced to 4 decimals
(g++ -O2, TinyShakespeare, 2500 steps):

- **(a)** Warm attention is genuinely **dot + mult-free precise exp** — no hidden
  float-exp DSP cost. Score is int64 `dot += Q*K; sc = dot>>X_ABITS`; softmax uses
  `g_exp_q16` (ROM), not `std::exp`. The only float exp/pow in the golden are
  host-staged ROM precompute (not datapath) and the `FLOAT_OPT` knob (defaults OFF,
  never touches forward softmax). Warm path confirmed engaged (ATTN_DOT 1 vs 0 give
  different trajectories).
- **(b)** Forward toggle **defaults off; int-kernel bit-exactness intact**.
  `ATTN_DOT=0`, `MP_OPT=MP_INT`, `HEAD_RESCALE=-1` all default safe; only the benign
  `#ifndef X_H` guard changed canonically (default build still reports H=2). NBLK=1
  bit-identical to the single-block mp golden (final_bpb=12.9054, every digit
  matches).
- **(c)** The ≥2 seeds are **real** — seeds vary both init (`srand(7+101*SEED+17*b)`)
  and data order (`srand(123+1009*SEED)`); trajectories differ throughout. dbl
  SEED0 5.3453/0.1477, SEED1 5.4198/0.1347 reproduced exactly.
- **(d)** Capacity is **L3/H4 matching the bar** (`-DX_H=4 -DX_DH=16`, X_D=64,
  H·DH=64=D, NBLK=3). The bar is **reachable**: the float autograd control hits
  held-out acc 0.42 / bpb 2.83 at only 600 CPU steps (2500-step bar 0.486/2.414).
  So the cascade's ~0.14 is a real backward defect, not an unreachable target.
- **(e)** The bpb is **calibrated, not saturated**. `batch_bpb` reconstructs the
  model softmax prob of the target; dbl 5.3–5.4 sits well below the 1/65536 floor
  (16.0). The one saturated number (fix16 SEED0 = 15.9382, spread 333M) is correctly
  flagged as DIVERGENCE / FAIL, not success.
- **Collateral:** HEAD_RESCALE is real and kernel-implementable (existing g_rescale
  max-shift): OFF → head logit spread ~542720 (~2^19, the one unbounded matmul
  output); =10 → spread 1673; defaults off. ACCUM=4 corroborates "cascade cannot
  batch" (~0.10 vs ACCUM=1's ~0.15). Vocab 64-vs-65 caveat honestly disclosed and
  negligible.

---

## 6. Reusable artifacts (fold into spec once cascade backward is fixed)

- **HEAD_RESCALE toggle** — head logits are the one unbounded matmul output; bound
  them with the existing mult-free max-shift before CE. Correct and bit-exact to the
  single-block golden when off.
- **Multi-block decomposition** (`mb_forward/backward_hidden` + `_output`) — correct
  and bit-exact to the single-block golden at NBLK=1.

## 7. Two concrete follow-ups before any kernel build

1. **Per-weight gradient cosine check** — finite-difference / autograd-vs-cascade at
   L=3/H=4 to localize *which* cascade-backward stage (LN-bwd vs route /
   rescale-Jacobian vs attention-bwd) diverges from true backprop as depth grows.
   The defect is present at depth 1 and compounds.
2. **Root-cause the ACCUM>1 collapse** — the cascade's collapse under gradient
   accumulation (→ chance) is itself a bug (likely a unigram sign-bias in the
   accumulated GLOG/EMB grads). Fixing it would let the cascade batch — a
   prerequisite for ever approaching the batch-dependent bar.

---

## Relevant files

- `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_golden_block.hpp`
- `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_block.hpp`
- `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_config.hpp`
- `/tmp/mb_golden_block.hpp`, `/tmp/mb_golden_test.cpp`
- `/tmp/autograd_bar_charlm.py`
- `/tmp/pc_xformer_charlm_softmax_precision.py`, `/tmp/sm_prec_logs/`

---

**DECISION: NEEDS-MORE.** Next action: run the per-weight autograd-vs-cascade
gradient cosine check at L=3/H=4 to localize which cascade-backward stage (LN-bwd /
route-Jacobian / `__int128` attention-bwd) diverges from true backprop, and
root-cause the ACCUM>1 chance-collapse — before any `forge_pc_xformer_mp_block.cpp`
build.
