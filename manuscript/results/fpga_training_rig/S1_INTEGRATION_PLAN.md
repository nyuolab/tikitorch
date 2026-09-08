# S1 Integration Plan — minimal-multiply transformer block, end-to-end trainer on AWS F2

**Goal (S1):** ONE transformer block (LN1 → Attn → Res → LN2 → FFN → Res) + token/pos embed + tied
LM-head + CE loss, training **end-to-end** (forward + backward + dyadic-Adam) **entirely on F2**,
**bit-exact** to a CPU golden, **multiplier-free** except the single attention soft-blend (DSP=128).

This plan is grounded in the **measured, csynth-closing** kernels (all 2.920 ns ≤ 4.0 ns on
xcvu47p): `forge_fwd_wide.cpp` (DSP=0), `forge_attn.cpp` (DSP=128), `forge_wgrad_df3.cpp` (DSP=0),
`forge_dyadic_adam.cpp` (DSP=0); the end-to-end on-fabric trainer precedent
`forge_train_step_morph_lm.cpp` (mode dispatch + fused `do_train_run`); the 124M DDR layout
`forge_buffer_gpt2.hpp`; and the CPU-validated algorithm `scripts/charlm_mixer.py --linear mfgrad`
(Gate 3 PASSED: transformer char-LM → gather 2.481 == autograd baseline 2.48).

**Honest scope.** S1 is an **on-chip, small** block (no DDR streaming; that is S2). It does NOT
target the 124M widths. It is the integration + CSIM proof that the four measured kernels compose
into a real transformer trainer. The four kernels are sized for d=768 today; S1 **re-parameterizes
them to d=128** via their existing `#define` config so the whole model fits on-chip for the first
CSIM. The wide/dataflow structure and the dyadic-MAC PE are reused verbatim — only the dimension
defines and a handful of missing element-wise kernels change.

---

## 1. S1 model config (small enough to fit on-chip for first CSIM)

```
S1_D       = 128     # d_model         (= morph_lm ML_D=128; LOG2_D=7 for the shift-mean norm)
S1_H       = 2       # heads           (head_dim = 64 == forge_attn AT_DH; reuse attn verbatim)
S1_DH      = 64      # head dim        (matches forge_attn AT_DH=64 exactly)
S1_FFN     = 512     # FFN inner       (4*D; multiple of 16 for wide bursts)
S1_L_SEQ   = 128     # context         (= forge_attn AT_L=128 exactly)
S1_V       = 128     # vocab           (= morph_lm ML_V=128; small CSIM vocab, tied embed/head)
S1_LAYERS  = 1       # blocks          (1 for first CSIM; 2 once 1-block is bit-exact)
S1_N       = 16      # GEMM microbatch  (= FW_N=PR_N=16; the contraction width baked into kernels)
```

**Why these numbers (each is load-bearing):**
- **D=128, LOG2_D=7.** Power-of-two so the LayerNorm mean is `sum >> 7` (no divider — the morph
  `batch_shift_row` shift trick, `forge_train_step_morph_lm.cpp:185`). Matches morph_lm ML_D so the
  proven embed-gather / residual / batch-shift / CE-LUT / LM-head logic ports with zero arithmetic
  change.
- **head_dim=64, L_SEQ=128.** **Exactly** `forge_attn.cpp`'s `AT_DH=64`, `AT_L=128`. The attention
  kernel is reused with **zero edits** — it already closes at 2.92 ns / DSP=128. With D=128, H=2 →
  H·DH = 128 = D (clean QKV reshape). Two heads = two `forge_attn` invocations over disjoint
  64-col slices of Q/K/V.
- **V=128 (tied embed/head).** Small vocab keeps the CE softmax row, the LM-head max-plus reduction,
  and the embedding table all on-chip and CSIM-fast. Matches morph_lm ML_V=128 so the balanced CE-LUT
  gradient (`forge_ce_lut.hpp`, rows-sum-to-zero) ports verbatim.
- **All widths multiples of 16.** Mandatory for the 512-bit wide bursts in every kernel
  (`max_widen_bitwidth=512`); 128/64/512/128 are all ÷16. Avoids the OFF_CE_LUT alignment class of
  bug (morph bug-(a): unaligned burst read the table shifted on silicon).
- **N=16.** The GEMM contraction width is a **compile-time constant** in all four kernels
  (`FW_N`, `PR_N`). S1 reuses N=16 → kernels reused without re-tiling. Tokens are processed in
  microbatches of 16 positions.
- **1 layer first.** Smallest unit that exercises the full chain. Once bit-exact, bump `S1_LAYERS=2`
  (the charlm_mixer CPU recipe already validated multi-block) — the only change is a layer loop in
  the host schedule + layer-major weight offsets.
- **On-chip total.** Weights ≈ (QKV 3·128·128 + O 128·128 + FFN 2·128·512 + LN 4·128 + tied
  embed/head 128·128) ≈ 213k params × (W+m+v) int32 ≈ 2.6 MB, plus activations/checkpoints/indices
  ≈ a few MB — comfortably inside the ~34 MB on-chip URAM+BRAM. This is **why D=128, not 768**: the
  768 kernels at full width need DDR residency (S2), but at d=128 the whole block + optimizer state
  fits on-chip, so S1 needs **no DDR streaming**.

**Scales to the 124M DDR path later (S2/S4):** the kernels keep their `d=768` structure; S1 only
shrinks the defines. S2 sets D=768/L=12/V=50257, swaps the on-chip weight arrays for
`forge_buffer_gpt2.hpp` layer-major DDR streaming (`layer_W(l)`, int8-dyadic, m+v resident), and adds
checkpoint/recompute. The compute kernels are unchanged between S1 and S4 — only addressing changes.

---

## 2. Missing kernels to author

The four measured kernels cover: forward GEMM, attention, weight-grad GEMM, optimizer. The
end-to-end block needs these **additional** element-wise / GEMM kernels. Each line: one-line spec +
clone source. All multiplier-free except where noted.

| # | kernel | spec (math) | clone from | DSP |
|---|---|---|---|---|
| M1 | `forge_xgrad_wide` (**input-grad GEMM — CRITICAL BLOCKER**) | `gx[n][i] = Σ_o gO[n][o] · q(W[o][i])` — W^T·gO; W dyadic→shift, **gO NOT dyad-quantized** (stays int16/int32 for fidelity, per charlm_mixer line 76 `gx=gO@Wq`). Transposed weight access (i-stationary tiling vs fwd's o-stationary). | `forge_fwd_wide.cpp` (same wide-W-load + dyadic-MAC PE; replace the dyad(x)·decode(W) inner with decode(W)·rawmul-free-shift over the o-reduction, x-tile held on-chip) | 0 |
| M2 | `forge_layernorm` fwd (fine-dyadic LN) | `c = x − mean(x)` (mean=`sum>>7`); `y = c · invq + β` where `invq = round-pow2(1/meanabs(c))` shift-add (mbits-mantissa, charlm `_fine_dyadic_rescale`); learnable dyadic γ optional | `forge_train_step_morph_lm.cpp:174` `batch_shift_row` (extends the mean-subtract with the meanabs scale + β) | 0 |
| M3 | `forge_layernorm_bwd` | backward of M2: grad through (x−mean) is `gR += gbs − (Σgbs>>7)` (morph batch-shift bwd, `forge_train_step_morph_lm.cpp:516-531`); the dyadic invq scale is STE pass-through (forward-quantized, backward-smooth), β-grad = Σ_n g | `forge_train_step_morph_lm.cpp:516` (batch-shift backward; add the scale chain) | 0 |
| M4 | `forge_relu` fwd+bwd (or fused into GEMM) | fwd `y=max(x,0)`, save 1-bit mask; bwd `gx = mask ? gO : 0`. (S1 uses ReLU not GELU — GELU needs a multiply; charlm `--mlp-act relu` is the mult-free deploy stack) | trivial; the gate pattern mirrors morph's max-route mask save (`tap_row`/`proj_row` argmax routes) | 0 |
| M5 | `forge_residual` add (fwd) + split (bwd) | fwd `R = sat16(R + branch)` (morph residual, `:334`); bwd residual split is pass-through: `gR` flows to both branch and skip (morph `:483-498`). Likely **fused** into M2/M3 invocations, not standalone. | `forge_train_step_morph_lm.cpp:334` (fwd add+sat), `:483` (bwd split) | 0 |
| M6 | CE loss + grad — **REUSE, do not author** | balanced LUT CE: `p=ce_lut[(zmax−logit)>>zshift]`, normalize by `>>ceil_log2(S)`, `grad[v]=p[v]−(v==tgt?Σp:0)` (rows sum to zero) | `forge_train_step_morph_lm.cpp:383-426` verbatim + `forge_ce_lut.hpp` (256 Q16) | 0 |
| M7 | LM-head (max-plus) + bwd — **REUSE** | fwd `logits[v]=max_d(W_HEAD[v][d]+R[d])` save argmax d*; bwd sparse scatter `GWHEAD[v*D+d*]+=g; gR[d*]+=g` | `forge_train_step_morph_lm.cpp:344-372` (fwd), `:452-466` (bwd) | 0 |
| M8 | Embed gather fwd + scatter bwd — **REUSE** | fwd `R[d]=EMBED[token*D+d]`; bwd `GEMBED[token*D+d]+=gR[d]` | `forge_train_step_morph_lm.cpp:235-246` (gather), `:535-540` (scatter) | 0 |
| — | bias-grad `gb=Σ_n gO[n][o]` | per-output column sum; **fold into M1 or `forge_wgrad_df3`** (the wgrad kernel already streams gO; emit gb alongside gW) | `forge_wgrad_df3.cpp` (add a Σ_n reduction next to the tile compute) | 0 |
| — | attention backward | gradient through L1-score + soft-blend. **OPEN DESIGN POINT** — see §6 risk. Smallest viable: train-time soft-blend bwd (`gV = Σ_i p[i][j]·gO`, `gp` then `gQ/gK` via `sign(Q−K)` from the L1 path). The blend bwd has a multiply (mirror of the fwd DSP). | `forge_attn.cpp` soft path; charlm `ShiftSoftmaxBlend`/`_hinge_grad` for the Q/K subgradient | 128 (mirrors fwd) |

**Net new authoring effort:** M1 (real GEMM, the critical blocker), M2/M3 (LN fwd/bwd, modest), M4
(ReLU, trivial), attention-backward (the hardest — see §6). M5–M8 are reuse/refactor of morph_lm
logic. M6/M7/M8 are lifted whole.

---

## 3. Integrated top-level train-step ABI + chaining

### ABI (mirror the proven morph_lm mode-dispatcher, `forge_train_step_morph_lm.cpp:835`)

```c
extern "C" int forge_train_step_s1(
    int32_t  buffer[S1_BUFFER_WORDS],   // monolithic on-chip-staged buffer (S1: all regions in one
    int      mode,                      //   alloc; S2: regions map onto forge_buffer_gpt2 DDR base)
    int      arg1, int arg2, int arg3
);
#pragma HLS INTERFACE m_axi      port=buffer offset=slave bundle=gmem \
        depth=S1_BUFFER_WORDS num_write_outstanding=16 num_read_outstanding=16 \
        max_write_burst_length=16 max_read_burst_length=16 max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite  port=mode/arg1/arg2/arg3/buffer/return bundle=control
```

**Modes (identical dispatch idiom to morph_lm):**
- `mode=0 FWD` — embed → [LN1 → attn → res → LN2 → FFN → res]×L → LM-head → CE loss-grad. Saves
  activation checkpoints + argmax routes (ReLU mask, head d*, attn jstar/p). Per-microbatch.
- `mode=1 BWD` — CE-grad → LM-head adjoint → block backward (LN2-bwd, FFN xgrad+wgrad, residual
  split, attn-bwd, LN1-bwd) → embed scatter. Accumulates gW/gb/gEmbed grads.
- `mode=2 OPT` — dyadic-Adam over every weight region (the `forge_dyadic_adam` recipe per region),
  then zero grads.
- `mode=3 TRAIN_RUN` — **the fused on-fabric trainer**: `for step: zero-acts; FWD sweep; zero-grads;
  BWD sweep; OPT`. ONE host kick = n_steps. (Modes 0/1/2 `#if`-guarded out of synthesis exactly like
  morph_lm to avoid duplicate instantiation; CSIM uses 0/1/2 per-step, silicon uses 3 — proven
  bit-exact by construction since mode 3 calls the same do_* functions.)

### Forward chain (one block), reusing the measured kernels

```
R = embed_gather(token)                          # M8 (reuse)
x1 = forge_layernorm(R)            [M2]           # LN1
# --- attention (per head h in 0..1, 64-col slice) ---
Q = forge_fwd_wide(x1, Wq_h); K = forge_fwd_wide(x1, Wk_h); V = forge_fwd_wide(x1, Wv_h)   [measured fwd GEMM]
A_h = forge_attn(Q,K,V, mode=soft)               # MEASURED attn, DSP=128, the ONE multiply
A   = forge_fwd_wide(concat_h(A_h), Wo)          # output proj [measured fwd GEMM]
R = forge_residual(R, A)           [M5]          # res add
x2 = forge_layernorm(R)            [M2]           # LN2
U = forge_fwd_wide(x2, W_up)       [measured]    # FFN up  (D->FFN)
H = forge_relu(U)                  [M4]
Dn= forge_fwd_wide(H, W_down)      [measured]    # FFN down (FFN->D)
R = forge_residual(R, Dn)          [M5]
logits = lm_head_maxplus(R)        [M7 reuse]
grad_logit = ce_lut_grad(logits, target)  [M6 reuse]   # rows sum to zero
```

### Backward chain (reverse), reusing measured + new kernels

```
gR  = lm_head_bwd(grad_logit)            [M7]  # sparse scatter -> GWHEAD, gR
# FFN bwd
gH  = forge_xgrad_wide(gDn, W_down)      [M1]  ; gW_down = forge_wgrad_df3(gDn, H)  [measured]
gU  = forge_relu_bwd(gH, relu_mask)      [M4]
gx2 = forge_xgrad_wide(gU, W_up)         [M1]  ; gW_up   = forge_wgrad_df3(gU, x2)  [measured]
gR += forge_layernorm_bwd(gx2)           [M3]  # + residual split pass-through      [M5]
# attn bwd
gAcat = forge_xgrad_wide(gA, Wo)         [M1]  ; gW_o = forge_wgrad_df3(gA, Acat)   [measured]
gQ,gK,gV = forge_attn_bwd(...)           [attn-bwd, §6]
gx1 += forge_xgrad_wide(gQ,Wq)+...       [M1]  ; gW_q/k/v = forge_wgrad_df3(...)    [measured]
gR += forge_layernorm_bwd(gx1)           [M3]  + residual split                     [M5]
embed_scatter(gR, token)                 [M8]  # GEMBED[token] += gR
```

### Optimizer

`forge_dyadic_adam` recipe (m,v EMA β=0.875, step=`m>>round(log2 v)`, `w−=step>>LR_SHIFT`) applied
per weight region (Wq,Wk,Wv,Wo,W_up,W_down,LN γ/β,EMBED). At S1 widths each region is a contiguous
`[W|grad|m|v]` slice; loop the region-RMW (the `signsgd_region_oc` idiom for on-chip resident state,
or the wide-burst `forge_dyadic_adam` REGION loop for buffer-backed).

### Addressing scheme

- **S1 (on-chip):** all regions in one `buffer` (the morph_lm pattern). Weights resident in
  BRAM/URAM (TT_RESIDENT_W style), grads on-chip (TT_ONCHIP_GRAD → II=1 scatter), activation
  checkpoints + argmax routes per-position in BS_STREAM-style rolling caches. Write resident weights
  back to buffer after the fused loop for host read-back (morph flush-back, gotcha #8).
- **S2 (deferred, DDR-streaming):** map every region onto `forge_buffer_gpt2.hpp` byte offsets
  (`layer_W/M/V(l)`, int8-dyadic packing, OFF_ACTCK checkpoints, OFF_GRADT tile, OFF_CE_LUT 64B
  aligned). The kernels are unchanged; only the offsets become DDR-resident and weights stream
  per-layer. **Explicitly NOT in S1.**

### Kernel-boundary impedance (must handle at integration — from the MAP gotchas)

1. **fwd y (int32 accum) → attn Q/K/V (int16):** LN sits between fwd and attn in the real chain, so
   the int16 cast happens naturally at the LN output. For raw GEMM→attn (no LN) stage, **explicitly
   sat16** before Q/K/V (attn `burst_load` casts to int16). (MAP impedance note + gotcha #2.)
2. **wgrad_df3 go_x packing:** gO and x must be staged **side-by-side, wide-word aligned** in one
   alloc at `OFF_GO_W`/`OFF_X_W`; gW output region must NOT alias the input (DATAFLOW RAW hazard,
   gotcha #12/#14).
3. **gW → dyadic_adam grad input:** `forge_wgrad_df3` writes int32 gW; `forge_dyadic_adam` reads grad
   from its `OFF_G`. Integration maps the gW output offset onto the optimizer's grad offset for that
   region.
4. **exp2 ROMs host-staged:** both `forge_attn`'s softmax LUT (256 Q16) and the CE-LUT
   (`FT_CE_EXP2_Q16`, 256 Q16) are **host-staged at runtime** (BRAM static-init reads all-zero on the
   F2 AFI — morph bug). Stage into 16-word-aligned offsets (gotcha #1, #6, #10).
5. **attn mode read-once:** `buffer[OFF_CFG]` selects soft (train) vs gather (deploy) **once per
   invocation** — train calls soft (mode 0), so set CFG=0 for the S1 trainer (gotcha #13).
6. **dyad round-to-nearest rule must match CPU golden exactly:** `if(msb>=1 && a[msb-1]) msb++`
   (bump-on-bit-below-MSB, NOT round-half-even). The CPU golden MUST use the identical rule or
   gradients diverge bit-for-bit (gotcha #9, #11).

---

## 4. CPU golden reference (for bit-exact CSIM)

**Two-layer reference, both extending what already exists:**

### (a) Algorithm-level golden — `charlm_mixer.py --linear mfgrad` (ALREADY VALIDATED, Gate 3)

This is the **numerical recipe** authority: `_MFGradLinearFn` does exactly `gx=gO@Wq`,
`gW=q(gO)ᵀ@q(x)`, dyadic-Adam int-state. It already trains the S1-shaped transformer to 2.481. Use it
to (i) freeze the S1 config (d=128/H=2/L=1/seq=128/V=128, `--mlp-act relu`, `--norm fine_dyadic
--norm-mbits 2`), (ii) confirm the small config still learns (sanity gate before HLS), and (iii)
**dump a frozen step**: initial weights, tokens/targets, the exp2 LUTs, and the post-step weights →
`s1_golden_step.npz`. This is the *float/algorithm* ground truth.

### (b) Bit-exact C++ golden in the testbench `_tb.cpp` (the CSIM authority)

Follow the `forge_pc_wgrad_tb.cpp` precedent exactly: a standalone C++ reference using **int64
accumulators** mirroring the kernel's integer arithmetic, with a **`ref_dyad()` identical to the
kernel `dyad()`** (priority encoder + bump-on-bit-below-MSB, `forge_pc_wgrad_tb.cpp:29`). The pass
criterion is **`max|kernel − ref| == 0` (BIT-EXACT)** per region (gW, gx, logits, grad_logit,
post-opt W), plus the **int32-headroom check** (peak |acc| < 2^31) at the deployed scaling range
(`|x|<600, |g|<1200` — `forge_pc_wgrad_tb.cpp:44-49`, avoids int48 saturation hiding bugs).

**Plan:** per-kernel `_tb.cpp` for the four measured kernels + each new kernel (M1–M4), then **one
integrated `forge_train_step_s1_tb.cpp`** that runs the full one-step schedule (modes 0/1/2 per
position) and checks the post-step weight CRC against the algorithm golden (a) staged into the
buffer. The integrated golden is a flat int64 re-implementation of the whole chain (the morph
`golden_morph_lm.cpp` pattern: load `buffer_init.bin`, run, FNV-1a weight CRC + `preds16` per step).

Because the CPU golden (a) and the C++ ref (b) and the kernel share the **exact dyad rule and
accumulation order**, the three are bit-exact by construction (the morph precedent: modes 0/1/2 ==
mode 3 by identical accumulation order).

---

## 5. COSIM / CSIM verification plan

The tikitorch workflow uses **g++ standalone C-sim** (the `forge_hls_types.hpp` shim swaps
`ap_int<N>`→`int16/int32` when `__VITIS_HLS__` undefined) as the production CI path, with optional
Vitis `csim_design` (only `synth_pc_wgrad_csim.tcl` does this today).

**Steps:**

1. **Author `_tb.cpp` for each kernel** in `tikitorch/hardware/hls/forge_kernels/`:
   `forge_fwd_wide_tb.cpp`, `forge_attn_tb.cpp`, `forge_wgrad_df3_tb.cpp`,
   `forge_dyadic_adam_tb.cpp` (note: it already has a scalar `#else` fallback to compare against),
   plus the new `forge_xgrad_wide_tb.cpp`, `forge_layernorm_tb.cpp`, `forge_relu_tb.cpp`, and the
   integrated `forge_train_step_s1_tb.cpp`. Each: aligned_alloc(64,…), fill deployed-range operands,
   call kernel, compare element-wise to int64 ref, `return ok?0:1`.

2. **g++ standalone build** (the CI/local path, no Vitis), per the README + ci.yml pattern:
   ```bash
   g++ -O2 -std=c++17 -DCSIM_ONLY -I.. forge_fwd_wide.cpp forge_fwd_wide_tb.cpp -o /tmp/tb_fwd && /tmp/tb_fwd
   # ... repeat for attn / wgrad_df3 / dyadic_adam / xgrad_wide / layernorm / relu / train_step_s1
   ```
   **Attn caveat:** the soft p·V blend is the one multiply → CSIM mode 0 compares to a float64
   reference with small epsilon (softmax is numerically delicate); mode 1 (gather) is bit-exact
   integer (MAP cosim gotcha #4).

3. **Add the testbenches to `.github/workflows/ci.yml` `csim` job** (after the existing
   `forge_train_step` lines) so every push re-runs the bit-exact checks (no Vitis, no F2).

4. **Optional Vitis `csim_design`** for in-HLS validation: clone `synth_pc_wgrad_csim.tcl` →
   `synth_<kernel>_csim.tcl` (`add_files -tb <kernel>_tb.cpp; csim_design; csynth_design`). NOT
   required (csynth already closes), but recommended for `forge_train_step_s1` (the integration) to
   catch fixed-width truncation the g++ shim path might mask. RTL CoSim (`cosim_design`) is NOT used
   anywhere today and is out of scope for S1.

**Runner/tcl to add:** `synth_xgrad_wide.tcl`, `synth_layernorm.tcl`, `synth_relu.tcl`,
`synth_train_step_s1.tcl` (clone `synth_fwd_wide.tcl`; csynth-only), plus the `*_csim.tcl` variants
for the integration. A `run_s1_csim.sh` wrapper that g++-builds + runs all `_tb` and reports
pass/fail (CI-callable).

---

## 6. Build order + gate before AFI

**Phase A — close the missing compute kernels (csynth + bit-exact CSIM). Parallelizable.**
1. **`forge_xgrad_wide` (M1) — FIRST, it is the critical blocker.** Clone `forge_fwd_wide`, transpose
   weight access, no-gO-quantize. csynth (target 2.92 ns, DSP=0, <half-SLR) + `_tb` bit-exact.
   *Gate-2 class is already proven for this PE → low risk, but it is on the critical path so do it
   first.*
2. **`forge_layernorm` fwd/bwd (M2/M3)** + **`forge_relu` (M4)** — small, parallel to M1.
   csynth + `_tb`.
3. **Backfill `_tb.cpp` + CSIM for the four measured kernels** (fwd_wide, attn, wgrad_df3,
   dyadic_adam) — they are area/timing-proven but **not yet cosim'd** (ROADMAP "STILL OPEN (a)").
   Fully parallel with 1–2.

**Phase B — attention backward (the genuine unknown).** Author `forge_attn_bwd` (soft-blend bwd +
L1-score subgradient). csynth + CSIM vs the charlm `ShiftSoftmaxBlend`/hinge reference. **This is the
one piece with no measured precedent** — gate its area/timing before committing to the integration.

**Phase C — integrate.** Author `forge_train_step_s1.cpp` (mode dispatcher + do_fwd/do_bwd/do_opt/
do_train_run) chaining all kernels at D=128. Author the integrated golden + `forge_train_step_s1_tb`.
**Bit-exact CSIM of the full one-step (modes 0/1/2) against the algorithm golden.** Then csynth the
whole block (expect ~1–2 SLR, DSP≈128 from the soft-blend, all else 0).

**Phase D — convergence proof in CSIM/host-emulation.** Run mode-3 fused trainer for N steps in
CSIM, confirm loss drops + matches `charlm_mixer --linear mfgrad` trajectory (the algorithm is
already known to reach 2.481 at this config). This is the "the substrate trains a transformer"
evidence WITHOUT F2 spend.

**Parallelizable:** Phase A items 1/2/3 are independent. Phase B can start in parallel with A (it
only needs `forge_attn`). Phase C blocks on A+B. Phase D blocks on C.

### GATE before AFI (must ALL be green — no F2 spend until then)
- [ ] Every kernel csynths ≤ 4.0 ns, DSP profile as designed (only attn fwd/bwd = 128).
- [ ] Every kernel **bit-exact** in CSIM vs its int64 C++ ref (closes ROADMAP open-item (a)).
- [ ] Integrated `forge_train_step_s1` one-step **bit-exact** to the `charlm_mixer --linear mfgrad`
      algorithm golden (post-step weight CRC match).
- [ ] Mode-3 fused trainer **bit-exact == modes-0/1/2** per-position (by-construction, verified).
- [ ] Full-block area fits ≤ 2 SLR with the soft-blend DSP=128; LUT budget leaves room for the CL
      shell (the morph/muzero F2 precedent).
- [ ] CSIM convergence: N-step loss matches the CPU recipe trajectory.

Only after all gates: build the CL package (clone the `cl_forge_muzero_ddr` precedent at S2; S1 is
on-chip so a simpler PCIM/host-buffer CL like `cl_forge_muzero`), build the AFI, run mode-3 on F2,
compare weight CRC to the CSIM golden.

---

## Honest risk callouts (carried into the structured fields)

- **Attention backward has NO measured precedent** — the four kernels are all *forward/optimizer/
  weight-grad*; the soft-blend + L1-score backward is new HLS. It is the single biggest S1 unknown
  (area, timing, and whether the mult-free Q/K subgradient trains as well in-kernel as in charlm).
- **M1 (input-grad GEMM) is the critical-path blocker** but low-risk (same PE class as the proven
  Gate-2 weight-grad; just transposed + no-quantize).
- **LayerNorm at d=128 with affine** is heavier than morph's bare batch-shift; the meanabs scale +
  dyadic γ STE must match the charlm `_fine_dyadic_rescale` bit-for-bit, and ROADMAP risk #7 flags
  batch-shift-alone may underfit a transformer (S1 uses fine-dyadic LN, not bare batch-shift).
- **Sparse scatter II=4** (LM-head/embed bwd) unless on-chip grad accumulators (TT_ONCHIP_GRAD) —
  affordable at V=128/D=128 (fits on-chip), but is the per-step critical path.
- **dyad round rule + accumulation order** must be identical across CPU golden / C++ ref / kernel or
  bit-exactness fails (the whole verification rests on this).
- **CSIM ≠ silicon for static ROM init** — both exp2 LUTs MUST be host-staged (morph AFI bug); CSIM
  can't catch the BRAM-init-reads-zero failure, so the staging discipline is a manual gate.
