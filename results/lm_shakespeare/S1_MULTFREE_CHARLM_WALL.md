# S1 strict-multiplier-free char-LM: the deploy wall (measured 2026-06-28)

Goal this session: train a small tropical LLM and **generate text on real F2 silicon** with the
strict multiplier-free (DSP=0) S1 transformer kernel `forge_train_step_s1`
(fwd-only deploy: dyad pow2 weights, MAD-LayerNorm with a 2-bit-mantissa reciprocal,
L1 argmax-gather attention, int16 activations, dyadic-GEMM head).

## The gate caught a hidden train/deploy gap
The earlier "faithful" torch trainer reported HARD ~2.6–2.95 bpc, but that was a **torch-only**
number. Running the model through the **real kernel forward** (`s1_deploy_eval`, which `#include`s
`forge_train_step_s1.cpp` and calls the actual kernel entry — i.e. exactly what silicon runs):

| model | torch HARD | real kernel |
|---|---|---|
| old faithful qx=1 | 3.37 | 4.34 |
| old faithful qx=0 | 2.60 | 4.33–4.80 |

→ the kernel never reproduced the trained model; deploying any of them = worse than bigram (3.47).
Generating from them on silicon would have produced garbage.

## QAT-through-a-kernel-sim (the right method) + recipe crack
Built `scripts/tropical_llm_kernelsim.py`: a torch forward that mirrors the kernel's integer
pipeline (int mean/meanabs, the `fdyad` 2-bit reciprocal LN, dyad-scaled weights, int16 GEMM+sat16,
L1 attention) with straight-through estimators, and trains QAT *through* it.

Recipe crack: the L1 scores are ~hundreds, so the initial softmax temp was ~100× too hard →
"from-scratch argmax stall". With genuine soft-τ annealing (invtau 0.002→0.2) + **8000 steps**,
the float-equivalent (exact-LN + exact-W + fine-act) model reaches **HARD 2.78 — beats bigram.**
So the architecture + recipe are sound.

## Quant-cost decomposition (8000 steps, clean float-equiv = 2.78)
Adding ONE coarse effect back to the clean model:

| + effect | HARD bpc | Δ | fixable? |
|---|---|---|---|
| + int16 activations | 2.78 | +0.00 | already fine |
| + 2-bit LN mantissa | 3.35 | +0.57 | only via kernel rebuild (MBITS=4) |
| + **dyad pow2 weights** | **3.55** | **+0.77** | **inherent to multiplier-free** |

→ **The dominant wall is the power-of-2 weight constraint itself** (round every weight to a single
±2^k). The 2-bit LN reciprocal is second. Int activations are NOT the problem. Either of dyad-W or
2-bit-LN alone caps you near bigram; you must fix both to get coherence.

## A second, separate gap: the sim is ~0.77 optimistic vs the real kernel for the QAT model
The QAT model is 3.56 in the sim but **4.33 on the real kernel**. (The old model agreed within 0.16,
so the QAT model learned to exploit a sim quirk.) Ruled out: per-term GEMM floor (`xv>>-ew`, only
−0.02), attention tie-break (true argmax-gather, ~0.00). Root cause not yet localized → a faithful
sim needs a per-stage diff against the kernel's saved intermediate buffers (X1/R/logits regions).

## Bottom line
A **strict multiplier-free (pow2-weight) D=64 char-LM is fundamentally bigram-class.** Even a
perfectly faithful QAT pipeline tops out around bigram, because the pow2-weight rounding (+0.77) and
the coarse LN (+0.57) are the ceiling — not the recipe, not the int activations. Coherent text needs:
- **multi-term dyadic weights** (w ≈ ±2^a±2^b, CSD) — still DSP=0 (2 shift-adds), ~halves +0.77; and/or
- **finer LN mantissa** (MBITS 2→4) — removes most of +0.57; and/or
- a **larger model** (more weights → pow2 grid less lossy per output); or
- accept bigram-class for the strict-DSP=0 demo.

All of the multi-term/MBITS options require a kernel change + AFI rebuild.

## Artifacts
- Deploy AFI built + registered: `afi-039ffed701d87d2c4` / `agfi-051979ac735ac66ea`
  (`forge_train_step_s1` fwd-only, S1_GEMM_QX=0 full-activation GEMM, DSP=0, csynth Fmax 342 MHz).
- `scripts/tropical_llm_kernelsim.py` — kernel-sim QAT trainer (+ `--ln_exact/--w_exact/--fine_act`
  ablations, `--soft` p·V mode).
- Gate harness `tikitorch/hardware/hls/forge_kernels/s1_deploy_eval.cpp` (real kernel forward).
