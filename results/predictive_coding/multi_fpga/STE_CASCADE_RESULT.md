# STE Backward on the Single-Block Transformer Cascade — DECISION: PARTIAL

**Date:** 2026-06-27
**Substrate:** single-block PC transformer cascade (NBLK=1, OUTPUT role), char-LM, CPU golden (`g++ -O2`), batch1
**Canonical tree:** `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/` — **git-clean, 0 files changed** (all experiments in `/tmp/ste/` + `/tmp/ste_verify/`)

---

## TL;DR

Porting the `DyadicPCNet` straight-through-estimator (STE) magnitude-aware dyadic backward into the golden cascade backward **lifts the long-standing ~0.135 accuracy cap to ~0.15** — a **real, reproducible** movement off a cap that had not moved across the entire cascade-precision investigation. But the autograd bar on the *same* single-block substrate is **~0.30**, so STE closes only **~12% of the 0.135→0.30 gap**. The wgrad multiply is **not** the cap; the residual is a structural **rescale floor dead-zone** in the route backward.

**The cascade is NOT yet de-risked.** This is a PARTIAL lift, not a BUILD.

**Mult-free verdict: NOT fully mult-free.** STE adds exactly **one genuine multiply** (the LayerNorm variance-coupling wide dot + its apply). The fully-mult-free training thesis is **not** resurrected on this path; we accept one multiply (the LN dot), and the dot-attention path (recipe #3) already carries multiplies independently of STE.

---

## 1. Template — confirmed and precise

The `DyadicPCNet` backward (`/tmp/tt-pc/tikitorch/nn/predictive_coding.py`; quantizers `dyadic_quantize` L600-609, `DyadicLinear` L612-643 in `tikitorch/nn/__init__.py`) treats **every quantizer as identity in the backward** — a clean straight-through estimator. The signature line (`nn/__init__.py:608-609`):

```python
q = torch.sign(w) * torch.exp2(torch.log2(mag).round())
return w + (q - w).detach()        # forward = q (pow2); backward = identity to wide w
```

The mechanism was confirmed at the bit level: cascade gradient *magnitudes* are sums of powers of two (`0.0039=2⁻⁸`, `0.0078=2⁻⁷`, `0.0156=2⁻⁶`, …) — i.e. quantized operands — **yet the gradient SIGN matches autograd-with-STE exactly at every layer**. That gives a complete, precise template: route the wide (pre-quantization) latent through the backward, keep the operand magnitudes dyadic where the op is mult-free, and only break to a real multiply where two non-pow2 operands must meet.

---

## 2. Dead-zone pinpoint — the prompt's stated root cause was REFINED

A per-op energy/cos probe (`/tmp/deadzone_probe.cpp`), a full-depth cos-to-autograd trace S0…S6 (`/tmp/depth_trace.cpp`), and a direct i64-vs-`__int128`-vs-float overflow check (`/tmp/overflow_check.cpp`) relocated the dominant killer. **It is not a forward-quantizer flat-region dead-zone.** Ranked:

**#1 — DOMINANT: i64 overflow from unbounded gradient growth in the route + attention backward.**
The gradient grows **~512× per route stage** (WO/WUP weights decode to dyad-exp 9-10) with **no inter-stage rescale**, exceeds `2^63`, and the i64 accumulator wraps to noise. Measured at the WO route: `max|gM|=1.8e17`, **9.0%** of dyad-shift terms exceed `2^63`; `cos(i64-route, autograd) = -0.001` (pure noise) vs `cos(__int128-route, autograd) = 0.996`. A **mult-free** per-stage max-shift rescale of `gM` before the route also gives `cos=0.996`. In the depth trace this is the **S4→S5 collapse** (‖g‖ explodes `1.3e18→3.4e20`, cos drops `0.62→0.02`). The attention L1 subgradient `gQ = -gs·sign` independently reaches `max|gQ| ~ 4.9e18 ≈ 2^62` and overflows when routed through WQ. Fix is **mult-free**: `__int128` accumulate + per-stage `g_rescale` (arithmetic shift, DSP_MULT=0).

**#2 — the genuine QUANTIZER dead-zone: LN variance-coupling double-dyad multiply (`g_ln_bwd`).**
The terms `g_dyadval(gc[d])·g_dyadval(chat[d])` and `g_dyadval(chat[d])·g_dyadval(vcm)` snap **both** operands to nearest `±2^k` before multiplying, flooring small `chat`/`vcm` contributions and randomizing the dominant LN gradient direction. Measured: final-LN backward (S1) cos-to-autograd jumps **0.047 → 0.772** when that one dot uses an exact wide multiply (`LN_STE=2`); the alignment then propagates through the FFN (S2 0.80, S3 0.81). This is the baseline S0→S1 collapse (cos `0.86→0.05`). **This is the one site STE cannot keep mult-free.**

**Ruled out (measured, NOT dominant):** #3 dyad weight decode in the route (a *symptom* of #1 overflow; with overflow fixed it already matches autograd at cos 0.996, and wide-weight STE alone moves training only 0.126→0.123); #4 activation `qact`/int16 band (0.0% of X1 clamped at ABITS=9); #5 LN `invq` 1/std rounding (`LN_STE=1` does not recover S1 cos); #6 `g_dyadval` operands in `g_wgrad` (dbl mode already wide and still caps at 0.126).

**Refinement of the prompt's hypotheses:** the prompt's #1 (qact/int16 band) and #3 (g_dyadval wgrad) are measured non-dominant. The real dominant killer is **i64 overflow** (mult-free to fix); the real quantizer dead-zone is the **LN variance-coupling dot** (needs a multiply).

---

## 3. Implementation — STE backward with a toggle

All in `/tmp/ste/`, built `g++ -O2 -I/tmp/ste`. **Forward proven byte-identical under the toggle.**

- `/tmp/ste/ste_golden.hpp` — **byte-identical to canonical** `forge_pc_xformer_golden.hpp` (`diff` exit 0). Proves the forward + all forward quantizers (`g_dyadval`/`g_dyad`/`g_wapply`/`g_applyinvq`/`g_matmul`/`g_layernorm`/`g_rescale`) are untouched.
- `/tmp/ste/ste_golden_block.hpp` — the backward, gated by `STE_BWD`. Only `g_route`, `g_ln_bwd`, and the attention-subgradient site in `g_backward` changed.
- `/tmp/ste/ste_mp_golden_block.hpp` — `mp_backward` (the trainer's path), same attn-subgrad STE site.
- `/tmp/ste/ste_test.cpp` — char-LM trainer. `/tmp/ste/fwd_identity.cpp` — forward-identity proof. `/tmp/ste/ste_depth_trace.cpp` — cos-to-autograd mechanism probe.

**Toggles:** `STE_BWD` (env, default `0` → reproduces the true-quantized-gradient cascade, bit-exact to canonical). `STE_BAND` (default 14) = the per-stage max-shift rescale band. The route uses `g_wapply` (dyad weight = signed bit-shift, `x<<e` / `x>>-e`) accumulated in `__int128` (wider adder, DSP_MULT=0) then `g_rescale` (right shift); the attn-subgrad rescale is `g_rescale`; the gamma-grad `dyadval·dyadval` is `pow2·pow2` = shift-add. **The single non-mult-free site is the LN variance-coupling wide dot**, flagged at the call site.

---

## 4. Verification — PARTIAL LIFT, honestly

Measured per-seed (1500 steps unless noted, NBLK=1, OUTPUT role, ATTN_DOT=1, HEAD_RESCALE=6, batch1/ACCUM=1, `acc_ema`/`bpb_ema`, `STE_BAND=14`); 2-seed means in the table.

| Config | acc | bpb |
|---|---|---|
| **Autograd bar** (dot-attn) | **0.297** | **3.55** |
| Autograd bar (L1-attn) | 0.256 | 3.71 |
| OFF cap (dbl) | 0.135 | 6.10 |
| OFF cap (fix16) | 0.128 | 5.74 |
| **ON STE (dbl)** | **0.151** | 5.56 |
| **ON STE (fix16)** | **0.150** | 5.42 |

STE moves the cap **0.135 → ~0.15** (real; holds to 4000 steps — *not* slow convergence: ON-dbl plateaus 0.157 / bpb 5.4 at 4000). The bpb story corroborates: cascade-ON `bpb_ema ~5.4-5.6` barely under chance (6.0) vs bar 3.5 — **not a metric artifact**. STE closes only **~12%** of the acc gap.

**Hard-constraint sanity (all PASS):**
1. Forward byte-identical under toggle — `fwd_identity` → `LG_diffs=0`, `GLOG_diffs=0`.
2. `STE_BWD=0` **bit-exact** to the canonical git-clean tree (built `forge_pc_xformer_mp_golden_test.cpp` from canonical → identical RESULT line; see `/tmp/ste_verify/canon_mptest`).
3. Canonical `forge_kernels/` git status empty; no LLM-instance files touched.

**Skeptic checks:**
(a) Not slow convergence — 4000-step plateau confirmed. (b) Not attention-mode-specific — with L1 attn (`ATTN_DOT=0`) the bar still hits 0.256/3.71 while cascade-ON caps 0.145/5.86 → the residual gap is **structural to the cascade backward**, not the dot-attn path. (c) Not wgrad precision — in dbl mode `mp_wgrad` already uses full-double operands; `WGRAD_MODE=1` (full-mul) is bit-identical → **wgrad snapping is NOT the cap**. (d) Component attribution (dbl, seed0): ROUTE-only 0.137, LN-only 0.120, BOTH 0.148, OFF 0.127 → **ROUTE overflow/rescale fix is the main lever**, LN-fix marginal. (e) `STE_BAND=14` slightly beats `12` (0.148-0.155 vs 0.133-0.144).

**Root cause of the residual gap (honest ceiling):** the mult-free per-stage max-shift rescale (`g_rescale`, arithmetic right shift) that contains the route overflow introduces a **NEW floor dead-zone on small gradient components** — it bounds magnitude by flooring, so small grads are lost. The dbl cascade has exact wide wgrad yet still caps at ~0.15, confirming the cap is in the **ROUTE/rescale + structural cascade backward**, not the optimizer/wgrad precision. Reaching the full bar needs a **finer multi-bit rescale or per-tensor magnitude normalization that does not floor small components**.

---

## 5. Mult-free verdict (PROMINENT)

**NOT fully mult-free.** STE adds exactly **ONE genuine multiply** on top of baseline:

- **[MULTIPLY #1]** `g_ln_bwd` STE branch: `vc += (__int128)gc[d]·chat[d]` — LN variance-coupling dot, **both operands wide** (not pow2) → genuine int multiply, ~192 dots/seq (3 LNs × 64 tokens).
- **[MULTIPLY #1b]** `g_ln_bwd` STE branch: `gin += … - (__int128)chat[d]·vcm` — the variance-coupling apply, wide×wide → genuine int multiply (same LN, per element).
- (Pre-existing, NOT from STE) `ATTN_DOT=1` dot-attention is recipe #3 and already carries multiplies (`Q·K`, `gs·K`, `gs·Q`, `pn·gp`, …) regardless of STE. In the **L1 path (`ATTN_DOT=0`)** these vanish and STE's **sole** multiply is the LN dot above.

Everything else STE touches is mult-free (route = `g_wapply` signed bit-shift + `__int128` accumulate + `g_rescale` right-shift; attn-subgrad rescale = `g_rescale`; gamma-grad = pow2·pow2 shift-add). **The fully-mult-free training thesis is not resurrected on this path** — we accept the one LN multiply (or a finer multi-term-dyad LN approximation, untried, as the route back to fully-mult-free). The no-global-sync / local-per-chip-backward architecture is **untouched** — all changes are local per-chip backward ops.

---

## 6. Caveats (load-bearing)

- **Single-seed-class / 2-seed means, single-block (NBLK=1), CPU golden, batch1.** Not multi-block, not silicon, not multi-seed-converged. The lift is real but small; on a 0.135→0.30 scale a 0.015 movement is modest and could partly reflect seed noise at the lower bound (mitigated by 2-seed means + 4000-step plateau, but not eliminated).
- The autograd bar itself was still climbing at step 1200 (~0.27→0.30); the bar may be slightly higher than 0.30 with more steps, which would *widen* the residual gap.
- The mult-free rescale floor is a **new** dead-zone introduced by the fix — the next probe must measure it directly, not assume it away.

---

## 7. Recommended next probe (the residual)

The lever is the **route rescale floor**, not the optimizer or wgrad. Order:
1. Replace the single max-shift `g_rescale` with a **finer multi-bit / per-tensor magnitude normalization** that preserves small components (kill the new floor dead-zone) — keep it mult-free if possible.
2. Apply the same `__int128` + per-stage rescale to the **attention-internal** backward (currently only the subgradient site is treated).
3. Decide the LN dot: accept the one multiply (recipe #3-style bargain) **or** prototype a finer multi-term-dyad LN-variance term to stay fully mult-free, and measure the acc cost.

Only after the rescale floor is lifted toward the bar should STE be folded into the canonical golden + multi-block.

---

## DECISION

**DECISION: PARTIAL** — STE lifts the single-block cascade 0.135→~0.15 (real, reproducible, ~12% of the 0.135→0.30 autograd gap) but does **NOT** close it; the cascade is **not yet de-risked**. **Fully mult-free: NO** — STE adds one genuine multiply (the LN variance-coupling wide dot; the dot-attention path carries more independently). **Next action:** replace the route's single max-shift `g_rescale` with a finer multi-bit / per-tensor magnitude normalization that does not floor small gradient components, then re-measure against the ~0.30 bar before any canonical fold or `forge_pc_xformer_mp_block.cpp` build.
