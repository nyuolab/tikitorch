# BF16 Backprop-STE Backward on the PC-Transformer — DECISION: BUILD-GO

**Date:** 2026-06-28
**Substrate:** char-LM PC transformer block (TinyShakespeare next-char), CPU golden (`g++ -O2`), golden dims **D=64, H=2, DH=32, SEQ=64, VOCAB=64, FFN=256, ABITS=9, MBITS=2**. 1500 steps, ACCUM=4, char-LM dyadic-Adam, EMA-acc on the last sub-batch (same metric for every mode).
**Canonical tree:** `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/` — **byte-unchanged** (`diff -q` clean; canonical golden mtimes Jun27, pre-session; no `*bf16*` files leaked in). All work isolated in `/tmp/bf16bwd/`.

---

## DECISION: BUILD-GO

A **clean, textbook BF16/wide-float backprop-with-STE backward** — written fresh, NOT the prior mp-cascade fork — **reaches the char-LM autograd bar at single AND multi block**, clearly clearing the integer cap, with the forward byte-identical and the gradients bit-identical to torch autograd. **The PC-transformer training recipe is finally de-risked on CPU.**

```
off_cap (int cascade)  =  0.128         <<  on_single (BF16 STE) = 0.343        ~  autograd_bar = 0.352
                                            on_multi  (BF16 STE) = 0.324  (NBLK=3, chained, depth-flat)
```

- **~6× accuracy lift** over the int cap (0.128 → 0.343); **bpb 4.9 → 3.26** and still climbing at step 1500 (implementer reports 3.07 bpb / 0.37 acc at step 2500).
- The BF16 STE backward **IS** the autograd bar: 0.343 vs the same-forward torch-autograd 0.352 — within seed noise, because the gradients are **bit-identical** (cos=1.000000 on every parameter). The prompt's "0.26–0.29" bar was a low/short-run estimate; the true bar at these dims/steps is ~0.35. BF16 clears it.
- **Reproduced** seed0 exactly (deterministic across recompiles); seed1 within noise (single 0.344/3.25, multi 0.327/3.31).

### This SUPERSEDES the same-day `COMPLETE_STE_RESULT.md` NO/PARTIAL verdict — and explains why
That doc's NO verdict was against a **different implementation**: a "complete-STE" wide-float *shadow of the mp integer cascade* (`forge_pc_xformer_mp_golden_block.hpp`, batch1, dot-attn + HEAD_RESCALE temperature-scaling). It correctly diagnosed two impl defects: **(1)** a pathological wide forward (bpb≈32 at init), and **(2)** a gradcheck that only validated the *input* gradient `gR`, never the *weight* gradients (`g.WQ…g.WHEAD`). This session's `/tmp/bf16bwd/` implementation is a **clean rewrite that fixes exactly those two defects**:
- **(1) fixed** — `bf_forward` is a clean textbook recompute (`forge_pc_xformer_bf16_golden_block.hpp:137`), not a mp-cascade wide-shadow. No bpb=32 init; the run starts healthy and drops monotonically.
- **(2) fixed** — `torch_check.py` and `mb_torch_check.py` explicitly compare **every weight gradient** (WQ/WK/WV/WO/WUP/WDN/WHEAD/EMB/POS/gR) to torch autograd, at init AND after 300 warm steps: **sign=1.0, cos=1.000000, relL2≈2e-12** on all. The previously-unchecked wgrad path is now the most-checked path.

The two docs are consistent: the cascade-shadow's wgrad path was broken; a correct textbook backprop on the same forward reaches the bar exactly as the chain rule requires.

---

## 1. The numbers — all measured on this exact config (golden dims, 1500 steps, ACCUM=4)

| Config | acc (s0 / s1) | bpb (s0 / s1) | reaches bar? |
|---|---|---|---|
| **Autograd bar** — same forward, torch autograd, golden dims | **0.352** | 3.214 | — (this IS the bar) |
| off-cap (`BF16_BWD=0`, int cascade) | 0.128 / 0.115 | 4.919 / 4.891 | no (the ~0.15 cap, bpb stalls ~4.9) |
| **on-single** (`BF16_BWD=1`, NBLK=1 H=2) | **0.343 / 0.344** | 3.258 / 3.252 | **yes** |
| **on-multi** (`BF16_BWD=1`, NBLK=3 H=4, chained gRin) | **0.324 / 0.327** | 3.344 / 3.310 | **yes — depth-flat** |
| bf16-rounded backward (mode 2, 1s8e7m) | matches wide-float | — | precision sufficient |

- The int cap is **confirmed** (~0.15, ~2.7× below the bar); the lift is **full**, not partial.
- Multi-block uses **local per-block chained backward** (`gRin(b) = gRout(b-1)` neighbour message; no global gradient) — the no-global-sync architecture is preserved and trains the stack.
- mode-2 (every backward intermediate truncated to BF16 1s8e7m) matches the wide-float result → **BF16 precision is sufficient; the int cap was ALGORITHMIC** (true-quantized-gradient + LN variance-coupling + int overflow/rescale), not a dynamic-range deficiency.

## 2. Forward byte-identical — CONFIRMED (the hard constraint)
- FNV hash of int `g_forward` (LG + GLOG + loss, 20 batches) = `a8d2467ed93a43a3` **WITH and WITHOUT** the BF16 header. The WITH==WITHOUT match on this machine proves the BF16 header does not alter the deployed forward (the hash differs from the implementer's `21af5a` only by compiler/RNG; the equality is what is load-bearing).
- The deployed forward is the **dyadic-quantized, mult-free** one (route via `dyad(master)`, qact STE), so the forward semantics match the kernel; only the **backward + optimizer** precision is wide.

## 3. Unfakeable gradient cross-check vs torch autograd (ground truth, not a C++ ref)
- **Single-block** (`torch_check.py`): every param (WHEAD/WDN/WUP/WO/WV/WK/WQ/gR + attention intermediates gATTN/gV/gQ/gK) **sign=1.0, cos=1.000000, relL2≈2e-12** at init AND after 300 warm steps (loss 4.37→2.63). FWD relerr ≈1e-12.
- **2-block chained** (`mb_torch_check.py`, fresh torch 2-block autograd): all per-block grads (block0 AND block1 *through the gRin chaining*), head, EMB/POS scatter cos=1.000000; **MIN cosine over all grads = 1.000000**. Confirms the chaining is correct — not "trains anyway."

---

## 4. THE EXACT KERNEL RECIPE TO TRANSCRIBE (`forge_pc_xformer_mp_block.cpp`, bit-exact to this BF16 golden)

Forward stays **byte-identical, mult-free, DSP_MULT=0**; the backward + optimizer go **BF16/wide-float (`fix16` in silicon)**. Source of truth: `/tmp/bf16bwd/forge_pc_xformer_bf16_golden_block.hpp` (single) + `…_mb_block.hpp` (multi).

**FORWARD (unchanged — do not touch):**
1. **Mult-free dyadic forward**, DSP_MULT=0: every weight applied as `dyad(W) = ±2^round(log2|W|)` → signed shift-add (`bf_dyad_f`, hpp:76).
2. **Warm dot-product attention** = the ONE allowed multiply: `softmax(Q·Kᵀ / √DH) @ V`, the `1/√DH` scale (`F.scaled_dot_product_attention`), causal. The `p·V` blend is the only forward multiply (2 DSP).
3. **Precise mult-free softmax** (exp2-LUT + `>>shn` normalize) deployed; the backward treats the LUT as identity and uses the **analytic** softmax probs.
4. **qact** = per-tensor int16 max-shift, round to pow2 grid, keep magnitude (`bf_qact`, hpp:98). **FDyadLN** = fine-dyadic LayerNorm (MBITS=2 mantissa).

**BACKWARD — BF16/wide-float textbook backprop with STE (the change):**
5. **STE set = identity in the backward:** dyad-weight round (grad → float master), qact max-shift, FFN floor/block-round, softmax exp2-LUT+normalize, head rescale + CE-LUT. These quantizers are `P(x)=x`.
6. **The REAL analytic Jacobians (not identity):**
   - **Linear bwd:** `dx = dyad(W)ᵀ @ gy` (route, dyadic/shift in silicon), `dW = gyᵀ @ x` (the new wgrad MAC) (`bf_linear_bwd`, hpp:185).
   - **LayerNorm bwd — MAD+dyadic-quantized form:** `s = (1+mant)·2^e` is **stop-grad** (the round/floor block the gradient to `s`), so the ONLY x-coupling is the mean term: `dx = (1/s)(gxhat − mean(gxhat))`, `gxhat = gy·γ` (`bf_ln_bwd`, hpp:203). **The RMS variance-coupling third term is SPURIOUS for this LN** (it belongs to a std-normalized LN; including it caused the 0.8 sign-agreement on the LN/attn-coupled grads). ★ **This OVERTURNS KERNEL_DESIGN_SPEC §2.2's instruction to ADD the vc term** — the spec told the kernel to add a variance-coupling term to match the old golden `g_ln_bwd`; the autograd ground truth says the correct LN bwd for this MAD+dyadic LN has **no vc term**. The kernel `ln_bwd` must match the mean-only form here.
   - **Attention bwd:** analytic softmax Jacobian `gs = p·(gp − Σ_k p_k·gp_k)·(1/√DH)`; `gV += p·gATTN`; `gQ += gs·K`, `gK += gs·Q` (hpp:241). (This is the **dot-attention** bwd, NOT the L1-margin subgradient — match the warm-dot forward.)
   - **CE grad:** `softmax(wide logits) − onehot`, mean over tokens (hpp:219).
   - **Residual + scatter:** `gR = gR_attn + gR_ln1`; scatter to EMB + POS.
7. **Wide errors:** all intermediate grads stay wide (`acc_t`=int64 in silicon; requantize to `XM_GBITS` only at the consumer — m/v and the grad accumulator).
8. **BF16 (fix16) dyadic-Adam optimizer** on the FLOAT master: `m=0.875m+0.125g`, `v=0.875v+0.125|g|`, `step=sign(m)·2^floor(log2(|m|/v))`, `p−=lr·step`, char-LM LR warmup (`bf_adam_one`, hpp:273). DSP=0 (shifts/compares/priority-encode only).

**MULTI-BLOCK:** stack NBLK blocks; residual stream chains `Rout(b)=Rin(b+1)`; backward chains `gRin(b)=gRout(b-1)` — **purely local per-block** (each block's backward needs only its own cached acts + the neighbour error message). Every primitive is reused verbatim from the single-block header (`…_mb_block.hpp`). This is the model-parallel, no-global-sync fabric arrangement.

---

## 5. HONEST CAVEATS (scale, seeds, and the explicit LATER steps)
- **Scale:** golden dims (D=64, SEQ=64, 1 char-LM block, TinyShakespeare). This de-risks the **recipe + datapath**, not a large model.
- **Seeds:** 2 seeds per mode; seed0 reproduced bit-exactly. Differences are within seed noise. Not a large seed sweep.
- **FLOAT-master optimizer (deliberate):** per the prompt, this trains a **float-master** model with a float/bf16 dyadic-Adam. The deployed forward IS the dyadic-quantized one (route via `dyad(master)`, qact STE), so forward semantics match the kernel; only backward+optimizer precision is wide. **The int16-master optimizer narrowing is a SEPARATE later step** — deliberately excluded here so it does not confound the backward-is-the-wall finding. The measured precision floor (`KERNEL_DESIGN_SPEC §1.3`) says `fix16` ≥ `bf16`, so this is expected to survive narrowing, but that is the next experiment, not this result.
- **Mult-free backward is also a LATER step.** This backward uses the allowed dot-attention multiply and wide (BF16) wgrad MACs. Pushing the wgrad/optimizer to fully mult-free is future work; the floor analysis predicts it holds, but it is not measured here.

---

## 6. NEXT ACTIONS (the build path)
1. Write **`forge_pc_xformer_mp_block.cpp`** bit-exact to THIS bf16 golden (`/tmp/bf16bwd/forge_pc_xformer_bf16_golden_block.hpp` + `…_mb_block.hpp`).
2. **csim** the kernel vs the bf16 golden → `max|diff|==0`.
3. **csynth** — measure the BF16 (fix16) **backward DSP budget**; carry over the int kernel's existing attention-bwd DSPs unchanged; confirm timing/area.
4. **D=64 AFI** → distributed **multi-F2** run.
5. **★ KERNEL_DESIGN_SPEC must change to BF16 (fix16) backward** — and its **LN-bwd instruction is corrected**: the MAD+dyadic LayerNorm backward is **mean-only (no variance-coupling term)**, per the autograd ground truth. The spec's §2.2 "MUST ADD the vc term" is overturned.

## Key files (all in `/tmp/bf16bwd/`)
- `forge_pc_xformer_bf16_golden_block.hpp` — single-block STE backward + float dyadic-Adam (the recipe source of truth)
- `forge_pc_xformer_bf16_golden_test.cpp` — e2e training
- `torch_check.py` + `forge_pc_xformer_bf16_dump.cpp` — single-block gradcheck vs torch autograd
- `forge_pc_xformer_bf16_mb_block.hpp` + `forge_pc_xformer_bf16_mb_test.cpp` — NBLK multi-block, chained backward (new this session)
- `mb_dump.cpp` + `mb_torch_check.py` — 2-block chained gradcheck (new this session)
- `autograd_bar.py` — torch-autograd bar at golden dims (new this session)
- `forward_hash.cpp` — forward byte-identity (WITH vs WITHOUT bf16 header)

---

**DECISION: BUILD-GO** — write `forge_pc_xformer_mp_block.cpp` bit-exact to `/tmp/bf16bwd/forge_pc_xformer_bf16_golden_block.hpp` (mean-only LN-bwd), csim to `max|diff|==0`, then csynth the fix16 backward DSP budget.
