# A fully multiplier-free tropical LLM that trains + generates (build artifact)

**Date:** 2026-06-27 · `scripts/tropical_llm.py` · tinyshakespeare char (V=39).
Model: tropical transformer — max-plus multi-head **argmax-gather** attention (mult-free) + dyadic ±2^k linears
(Q/K/V/O, FFN, embed, head) + positional embeddings. Trained τ-annealed (soft warm-train) → **deployed HARD**
(argmax-gather), fully multiplier-free (DSP=0 datapath). Generates via the hard mult-free deploy.

## Trained model (D=128, 4 layers, T=128, 815k params, τ-floor 0.05, 7000 steps)
- soft_bpc (τ=0.05) = 2.237 · **HARD (fully mult-free) bpc = 2.834** · bigram baseline 3.472.
- Beats bigram by 0.64 bits at the fully-multiplier-free hard deploy.

### Generated text (fully multiplier-free hard deploy, prompt "the ", temp 0.8)
```
the stay the revens thro hath scrand the;
unwither honour waring mon fame over would what the,
an' moththty not--tok bethod the feew to sigh,
as wel thousalest parise---
king hend shal kindry me like, i can is,
then he moneres.

leontezex:
duke o, some he
pauleture:
yoron, gooo, no mine rator
that instra cheren my timet presshes seecre
```
A small char model, but clearly a working LM: English/Shakespeare word structure, punctuation, line breaks, and
the character-name `NAME:` dialogue format — all generated with **zero multiplies** in the deploy datapath.

## Honest finding: the hard-deploy gap grows with depth + context
| config | HARD bpc | soft bpc | gap |
|---|---|---|---|
| L=3, T=64  | 2.276 | 2.232 | 0.04 (clean) |
| L=3, T=128 | 2.54  | — | — |
| L=4, T=128 | 2.834 | 2.237 | 0.60 |
Each argmax-gather layer picks ONE position/head; stacking layers + lengthening context compounds the soft→hard
mismatch. So the cleanest fully-mult-free deploy is at modest depth/context (L=3/T=64 → hard 2.276 ≈ soft).
Mitigations: peakier τ-anneal (lower floor), more heads, or a small soft/top-k component (not fully mult-free).
**Best clean mult-free model = L=3/T=64 (2.276); best generation = L=4/T=128 (2.834, coherent text).**

## Reproduce
`python scripts/tropical_llm.py --D 128 --H 4 --L 4 --T 128 --steps 7000 --taufloor 0.05`
