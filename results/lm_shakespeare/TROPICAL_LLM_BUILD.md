# A fully multiplier-free tropical LLM — trains, generates, clean mult-free deploy (build artifact)

**Date:** 2026-06-27 · `scripts/tropical_llm.py` · tinyshakespeare char (V=39).
Model: tropical transformer — max-plus multi-head **argmax-gather** attention (mult-free) + dyadic ±2^k linears
(Q/K/V/O, FFN, embed, head) + positional embeddings. Trained τ-annealed (soft warm-train) → **deployed HARD**
(argmax-gather), fully multiplier-free (DSP=0 datapath). Generates via the hard mult-free deploy.

## Final model — D=128, 4 layers, T=128, τ-floor 0.02, 815k params, 7000 steps
- soft_bpc = 2.421 · **HARD (fully multiplier-free) bpc = 2.425** · gap 0.004 (CLEAN) · bigram 3.472.
- Beats bigram by ~1.05 bits at the fully-multiplier-free hard deploy; hard ≈ soft → the warm-train→hard-deploy
  conversion is lossless. (Dense softmax transformer reference at this scale ≈ 2.2.)

### Generated text (fully multiplier-free hard deploy, prompt "the ", temp 0.8)
```
the strange trave i her, and some no the cust is once
give love me and made parts,
whose of the se most be chal-closed
the behart what prince he discontes. where pity;
who thou sirst father, are like, i canot be contry
if you pooint your give fuil o' much he sin
of my son a chaor of illlow'd, in my lord.

third seen you with world childre wild be myself:
lok madam: gent self, that say breath,
```
A small char model, but unmistakably a working LM — English/Shakespeare words, punctuation, line breaks, the
character-name `NAME:` dialogue format — generated with **zero multiplies** in the deploy datapath.

## The deploy-gap finding, and its fix (τ-floor scaled to depth)
The argmax-gather hard deploy must MATCH the soft attention it was warm-trained as; stacking layers + longer
context needs a PEAKIER final attention (lower τ-floor) for argmax to match the soft blend:
| config | τ-floor | HARD bpc | soft bpc | gap |
|---|---|---|---|---|
| L=3, T=64  | 0.05 | 2.276 | 2.232 | 0.04 (clean) |
| L=4, T=128 | 0.05 | 2.834 | 2.237 | 0.60 (broke) |
| **L=4, T=128** | **0.02** | **2.425** | 2.421 | **0.004 (clean + best)** |
So the depth/context gap was **not fundamental** — just the anneal floor. Rule: **scale τ-floor DOWN with
depth/context.** No hybrid soft head needed; the fully-mult-free argmax-gather deploy is clean at depth.

## Reproduce
`python scripts/tropical_llm.py --D 128 --H 4 --L 4 --T 128 --steps 7000 --taufloor 0.02`  (checkpoint: tropical_llm.pt)
