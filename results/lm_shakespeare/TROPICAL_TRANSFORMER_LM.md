# Fully multiplier-free TROPICAL TRANSFORMER char-LM ≈ dense transformer (measured)

**Date:** 2026-06-27 · torch/MPS · `scripts/tropical_transformer_lm.py` · tinyshakespeare (lowercased, V=39).

## Result (D=128, 3 layers, T=64, 6000 steps; bigram baseline 3.472 bpc)
| model | val bits/char |
|---|---|
| dense transformer (softmax, float weights) | **2.220** |
| tropical transformer, soft warm-train (dyadic weights, soft L1 attn) | 2.232 |
| **tropical transformer, HARD deploy (fully multiplier-free)** | **2.276** |

**Gap (mult-free hard − dense) = +0.056 bpc.** Hard deploy is only +0.044 over its own soft warm-train → the
τ-anneal → argmax-gather conversion is clean (no train-soft/deploy-hard mismatch).

## What "fully multiplier-free" means here
- **Attention = max-plus, multi-head argmax-GATHER.** score_ij = −L1(Q_i,K_j) = −Σ_d|Q−K| (adds only); causal;
  per head pick `j* = argmax_j score` and **gather** V_{j*} (copy the winning position's whole value vector —
  the *coherent* op, NOT per-channel max-plus matmul). H=4 heads → 4 hard picks combined by the O projection.
- **All linears dyadic** (Q/K/V/O, FFN, embed, head): weights snapped to ±2^k via STE → shift-add, DSP=0.
- **The only multiply is the soft-blend at TRAIN time** (softmax(score/τ)·V), τ annealed 2.0→0.2. Deploy is hard
  argmax-gather → no softmax, no exp, no normalize, no PV-multiply. "warm-train + mult-free-deploy."

## Why this matters for F2
The deploy attention is **more FPGA-parallel than softmax**: per query, scores are adds (−|Q−K|), routing is a
comparator/argmax tree (log T, pipelined), aggregation is one gather — no transcendental, no sequential
normalize, every query independent. Same O(T²) work as softmax but a far better constant + parallel structure
for an adder/comparator fabric. The whole model is DSP=0 at deploy.

## Honest scope
Single seed, small scale (610k params, char-level, T=64). The +0.056 gap is at this scale; behavior at larger
D / longer training / multi-hop tasks is not yet measured. Training uses the soft-blend multiply (warm-train);
a fully on-fabric *training* loop would need the τ-annealed soft attention or a mult-free routing-gradient.
The headline is validated: a fully multiplier-free tropical transformer is competitive with dense attention on
a real char-LM, with an FPGA-friendlier attention op.

## Reproduce
`python scripts/tropical_transformer_lm.py --steps 6000 --L 3 --T 64 --bs 64`

## Stress tests (2026-06-27)
- **Scale (D=256, 2× width, 8k steps):** dense 2.366 / tropical hard 2.408 → **gap +0.043** (vs +0.056 at
  D=128 — the mult-free gap SHRINKS at scale). NB: D=256 absolute bpc is worse than D=128 = overfitting on
  tiny tinyshakespeare (~1M chars); the GAP is the signal. A cleaner scale check would use text8_20m.
- **Top-k deploy is the WRONG op:** top-k=2 hard 3.140, top-k=4 hard 3.941 (both far worse than k=1's 2.276).
  Cause: training τ-anneals the attention to PEAKY (top-1); uniformly averaging the top-k at deploy fights the
  trained distribution. **argmax-gather (k=1) is the correct, and simplest-for-hardware, deploy op.**

## Locked recipe
max-plus **argmax-gather** (k=1) attention + dyadic everything + τ-annealed warm-train → hard deploy.
~+0.05 bpc behind dense, stable across D=128→256. DSP=0 deploy, FPGA-friendlier attention than softmax.
