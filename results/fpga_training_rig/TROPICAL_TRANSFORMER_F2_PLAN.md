# Tropical transformer on F2 — deployment + training plan

**Date:** 2026-06-27. Synthesis of the validated multiplier-free tropical transformer LM recipe
([`../lm_shakespeare/TROPICAL_TRANSFORMER_LM.md`](../lm_shakespeare/TROPICAL_TRANSFORMER_LM.md)) into a concrete
F2 path. Headline measured result: a **fully multiplier-free tropical transformer** (max-plus argmax-gather
attention + dyadic everything) hits **2.276 bpc** on char-LM vs a dense softmax transformer's **2.220** (+0.056;
gap *shrinks* to +0.043 at D=256). The whole deploy datapath is DSP=0.

## The architecture maps onto kernels that ALREADY EXIST at silicon maturity

| stage | op | kernel (status) | DSP |
|---|---|---|---|
| token embed + **positional embed** | int gather + int add | s1 `S1_POS_EMBED` (added, grad-checked, byte-id-off) | 0 |
| Q/K/V/O projections | dyadic GEMM (±2^k shift-add) | `forge_fwd_gemm` / s1 dyadic GEMM (csynth-closed) | 0 |
| **attention** | L1 score + argmax-GATHER | **`forge_attn_maxplus`** (this commit; csynth in flight) | **0** |
| FFN (up/ReLU/down) | dyadic GEMM + ReLU | `forge_fwd_gemm` + `forge_relu` (csynth-closed) | 0 |
| LayerNorm | fine-dyadic (shift mean) | `forge_layernorm` (csynth-closed, bit-exact) | 0 |
| **LM head** | dyadic GEMM | s1 `S1_HEAD_MODE=1` (added, FD-grad-checked) | 0 |
| cross-entropy | balanced exp2-LUT | morph-LM fused CE (on F2, bit-exact) | 0 |

So the **deploy (inference) datapath is fully multiplier-free, DSP=0** — every piece is a csynth-closing kernel,
and the one new piece (`forge_attn_maxplus`) is a *simplification* of the already-closed `forge_attn` (its mode-1
deploy path, with the soft-blend/softmax/exp2 removed). The attention is also **more FPGA-parallel than softmax**:
adds (−|Q−K|) + a comparator/argmax tree (log T) + one gather — no transcendental, no sequential normalize.

## The key design choice: argmax-gather, not softmax, not top-k
- **argmax-GATHER** (`out_i = V_{argmax_j score}`) is the validated deploy op (2.276 bpc). It's the *coherent*
  max-plus op (pick one position, copy its whole V), **not** per-channel max-plus matmul (incoherent → collapse).
- **Multi-head** (H=4) gives H hard picks combined by O → recovers most of softmax's multi-position mixing.
- **top-k is the WRONG deploy** (measured: craters to 3.14/3.94) — training τ-anneals to a peaky top-1, so
  uniform top-k averaging fights it. k=1 argmax-gather is both best *and* simplest for hardware.

## Two routes to "a tropical LLM on F2"

### Route A — DEPLOY mult-free now, warm-train (lowest risk, near-term)
Train the model with the **soft τ-annealed attention** (the one train-time multiply) off-fabric or via the
proven host-optimizer hybrid, then **deploy inference fully mult-free on F2** (argmax-gather + dyadic). This is
the "warm-train + mult-free-deploy" answer the whole repo converged on, now with a *competitive* model. Build:
the s1 forward kernel + `forge_attn_maxplus` mode + `S1_HEAD_MODE=1` + `S1_POS_EMBED=1` → one AFI, run inference.

### Route B — TRAIN on F2 (the flagship, higher risk)
On-fabric training needs the soft attention for the gradient (the s1 `forge_attn` mode-0 soft-blend, DSP for the
attention only) + the dyadic-Adam optimizer (host-optimizer hybrid, already silicon-proven for the s1 backbone).
Open risk: the s1 on-fabric attention backward showed a gradient bias (the earlier "context wall") — the torch
warm-train works, so the fix is to reconcile the kernel's int16 attention adjoint with the torch path (a
backward audit), OR accept warm-train (Route A). The forward/optimizer are proven; the attention backward is the
one piece to harden.

## De-risk order (no AFI until each csynth/CPU gate passes — the s1 discipline)
1. **[in flight]** csynth `forge_attn_maxplus` → confirm DSP=0 + timing (the mult-free attention hardware claim).
2. CPU/csim: assemble the full tropical-transformer forward from the existing kernels at the validated config,
   bit-exact vs the torch reference; confirm the assembled forward reproduces the 2.276 bpc deploy number.
3. Route A AFI: forward-only (inference) tropical transformer → run on F2, measure on-silicon bpc + tok/s.
4. Route B (stretch): the soft-attention backward audit → on-fabric warm-train.

## Honest status
- **Measured:** the recipe (torch, char-LM, D=128→256). The kernel pieces individually csynth-close / are on F2.
- **Projected:** the *assembled* on-silicon bpc (step 2/3) — not yet run. The on-fabric *training* (Route B)
  depends on the attention-backward audit. Scale beyond char-level / longer context is untested.
- The banked, defensible claim: a fully multiplier-free tropical transformer is competitive with dense attention
  on a real char-LM, its entire deploy datapath is DSP=0, and every component is a buildable F2 kernel — so a
  tropical LLM *running* on F2 (Route A) is a near-term, well-scoped build, with on-fabric *training* (Route B)
  the flagship stretch.
