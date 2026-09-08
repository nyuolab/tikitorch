# Char-LM bakeoff: dense vs tropical, attention vs morphological mixing

**Questions.** (1) At matched parameters and token budget, what does the
multiplier-free tropical substrate cost on language modeling? (2) Which
tropical architecture salvages the most performance? (3) What does the
hard-deployable sign-based training stack cost on top?

**Protocol.** tiny-shakespeare (1.1 MB, char-level, vocab 65, 90/10 split),
1500 steps × batch 24 × seq 96 ≈ 3.46 M tokens, ~0.51–0.53 M params per arm
(one exception noted), device=mps, canonical-repo import forced via
`PYTHONPATH` (see Provenance). Eval: float CE on 18,432 fixed val tokens,
identical windows for every arm; tropical arms deployed hard (tau=0; final
soft==hard bit-exact in every run). Seeds: headline arms n=3 (seeds 0–2),
others seed 0. Each family's lr was swept (300-step probes + full-run
confirmations; full-run lr sweeps for the fast dense arms).

## Main results (1500 steps; mean ± range/2 where n=3)

| arm | mixer | substrate | params | val CE (nats) | bpc | acc |
|---|---|---|---|---|---|---|
| dense-conv twin (AdamW 3e-3) | dilated conv | real | 515,840 | **1.7019** | 2.455 | 50.2% |
| dense GPT (AdamW 3e-3) | attention | real | 525,312 | 1.7388 | 2.509 | 48.9% |
| dense-conv twin (1e-3, n=3) | dilated conv | real | 515,840 | 1.783 ± .025 | 2.57 | 47.8% |
| dense GPT (1e-3, n=3) | attention | real | 525,312 | 1.837 ± .024 | 2.65 | 45.3% |
| **MorphSSM float (1e-2, n=3)** | morph conv | **tropical** | 512,065 | **1.934 ± .035** | 2.79 | 43.0% |
| MorphSSM-L3 (239 k params) | morph conv | tropical | 238,657 | 1.9388 | 2.80 | 43.1% |
| *interpolated trigram* | — | — | — | *2.1397* | *3.09* | — |
| **MorphSSM closed (SignSGD+tropical_ce)** | morph conv | tropical | 512,065 | **2.1489** | 3.100 | 38.5% |
| hilbert attention | trop. attention | tropical | 524,481 | 2.4498 | 3.534 | 27.7% |
| *count bigram (add-1)* | — | — | — | *2.4819* | *3.58* | *27.0%* |
| hilbert + dyadic-ffn | trop. attention | tropical | 524,481 | 2.4887 | 3.590 | 27.3% |
| dyadic-ffn | trop. attention | tropical | 524,481 | 2.4960 | 3.601 | 27.3% |
| dyadic-full (FFN+head+QKVO) | trop. attention | tropical | 524,481 | 2.5889 | 3.735 | 26.2% |
| attention baseline (pos, 1e-2) | trop. attention | tropical | 524,481 | 2.6095 | 3.765 | 27.1% |
| closed transformer (v2) | trop. attention | tropical | 524,481 | 2.6611 | 3.839 | 27.0% |
| *unigram* | — | — | — | *3.3473* | *4.83* | *14.9%* |
| morphssm-dyadic (COLLAPSED) | morph conv | mixed | 512,065 | 3.3554 | 4.84 | 14.9% |

300-step screens only (never run to 1500; screens demonstrably mis-rank by
~0.08 nats, see Methods note): maxmin 2.823, iter-topk 3.012, topk-proj 3.027.

## Findings

1. **The token mixer, not the FFN/head, was the tropical bottleneck.
   Morphological conv mixing recovers most of the gap.** The v2 tropical
   transformer trailed dense GPT by 0.76 nats; MorphSSM (dilated causal
   max/min convs, fully tropical at deploy) trails the *tuned dense GPT* by
   0.20 nats and beats the trigram table. Every tropical-ATTENTION variant
   tried — scores (dot/hilbert), aggregation (argmax/maxmin/iter-topk),
   projections (dilation/topk/dyadic), FFN (max-plus/dyadic), positions —
   stays in a 2.45–2.74 band around bigram CE (hilbert alone beats bigram,
   by 0.032 nats). The full-run evidence covers 5 attention arms;
   maxmin/iter-topk/topk-proj are excluded at screen level only.

2. **The substrate cost, properly attributed, is ≈ 0.15–0.23 nats
   (0.22–0.33 bpc, 1.16–1.26× per-char perplexity).** The attribution
   control is `DenseConvLM`, a real-arithmetic twin of MorphSSM (same
   d=256/L=7/k=7 shape, standard depthwise conv+GELU+linear): paired by
   seed at matched lr the morphssm−twin gap is 0.141/0.141/0.160 nats
   (mean 0.151 ± 0.010); tuned-vs-tuned (twin@3e-3 n=1 vs morphssm@1e-2
   n=3) it is ≈ 0.23. Without this control the naive "morphssm vs dense
   GPT" gap (0.10) would have over-credited the substrate: roughly a third
   to half of MorphSSM's win over dense GPT is the conv mixer's inductive
   bias — the twin beats dense GPT too (1.702 vs 1.739 tuned).

3. **Tropical amplifies mixer choice.** On the real substrate, conv vs
   attention is worth 0.04 nats at this scale; on the tropical substrate it
   is worth 0.5–0.7. Tropical attention's `max_d(Q+K)` scoring + argmax
   value-copying is the broken piece, not max/min sequence operators per se.
   (Echo of arXiv:2406.02528, which also dropped attention for a
   conv/recurrence-style mixer under a multiplication constraint.)

4. **The max-plus channel mixer is load-bearing — a linear (dyadic OR
   float) replacement collapses MorphSSM.** morphssm-dyadic fell to a
   constant space-predictor (acc pinned at exactly the eval modal-char
   frequency 0.1490 from step 0). Diagnosed by controlled 150-step probes:
   collapse reproduces with stock kaiming dyadic, small-std re-init,
   identity init, plain float nn.Linear (both inits), full nn.LayerNorm,
   and lr 1e-3 — only the max-plus out_proj trains (2.31 vs 3.36 at step
   150). The branch std explodes ~100× under AdamW in every linear arm;
   the max-plus projection is gain-bounded (≈1-Lipschitz in x). Dyadic
   layers DO help in the transformer FFN slot (+0.11 nats); they are not a
   drop-in for tropical projections in residual conv stacks.

5. **Hard-deployable sign-based training costs 0.20 nats on MorphSSM**
   (2.149 vs 1.947; same anneal, SignSGD 2e-3 + tropical_ce) — larger than
   the 0.05 measured on the attention baseline, but the absolute result is
   the best closed LM in the repo by 0.51 nats and sits at trigram level
   (0.009 nats short of the trigram table). Closed-morphssm was the
   steepest-improving arm at step 1499 (−0.011/100 steps), so 0.20 is a
   budget-bound upper estimate. Caveats: SignSGD lr was tuned on the
   transformer, not morphssm; "fully multiplier-free training" still
   requires TropicalLion/PureTropicalSGD + tropical_ce_lut (momentum EMA =
   hole H7, host softmax), which remain unbenchmarked.

6. **Tau-anneal-to-hard works.** Final hard == soft bit-exact everywhere;
   mid-run soft-hard divergence peaks at 0.12 nats (morphssm) and is closed
   smoothly by the 0.5→0 anneal (steps 750–1350). The v1 SignSGD divergence
   episode did not recur on morphssm.

7. **lr lessons.** 300-step probes mis-rank 1500-step outcomes twice over:
   transformer 1e-2's early 0.38-nat lead shrank to 0.02; morphssm 2e-2's
   early 0.11-nat lead inverted to −0.03 (final 1.978 vs 1.948 at 1e-2).
   Screens are for pruning, not for picking lrs or declaring winners.

8. **Depth saturation:** MorphSSM-L3 (3 layers, 239 k params — less than
   half) matches L7 (1.939 vs 1.948). The morph mixer saturates early at
   this task scale; MorphSSM's win over tropical attention is not a depth
   effect.

## Samples (300 chars, temp 0.8)

dense GPT: word-level Shakespeare babble. MorphSSM float: real words and
clause structure ("Will thour none have took purst to daim … and love of not
say"). Closed morphssm: words with more spelling noise. Attention arms:
char-soup with speaker-tag structure only.

## Scope limits

- One dataset (char-level tiny-shakespeare), one scale (~0.5 M params,
  3.46 M tokens), no LR decay, nothing converged (dense improving ~2×
  faster than morphssm at step 1499 — the 0.15–0.23 gap is a fixed-budget
  snapshot and was *widening*, not closing).
- n=3 seeds for dense GPT/dense-conv/morphssm at base lrs; n=1 for tuned-lr
  runs and all other arms. Dense seed spread alone is ~0.05 nats.
- MorphSSM/twin shape (d=256 L=7) differs from the attention arms
  (d=128 L=3); params matched, shape not — conclusion (1)'s causal
  attribution rests on the dense twin, and on L3 ruling out depth.
- No claims about scaling or about arXiv:2406.02528's regime (370M–2.7B
  params, 15–100B tokens, ternary substrate); this is three orders of
  magnitude away on both axes.
- Wall-clock (MPS, no fused tropical kernels): dense ~150–235 k tok/s,
  morphssm ~3.7 k, dyadic-full ~21 k. Implementation artifact; target
  hardware is FPGA.

## Provenance

- **Import provenance:** v1/v2 runs (bakeoff_v1, v2_*) executed the
  pip-installed kanervasforge submodule copy of tikitorch (sys.path quirk);
  code-diff audit found only doc/comment deltas plus block-dropout impl
  (nn.Dropout vs TropicalDropout — identical at eval), so those numbers
  stand. All v3 runs force the canonical checkout via
  `PYTHONPATH=<repository-root>`.
- **Library fix in this round:** `tropical_value_pull_iter_topk` crashed for
  Lv < k (1-token generation prompts); fixed in `tikitorch/functional.py`
  (`rank_bias[:kk]` + backward zero-pad), unit-tested.
- Three adversarial verification passes (multi-agent) ran before
  conclusions were frozen: v1 (eval correctness, fairness, lr probes —
  caught the missing positional signal and untuned lr), v3 (caught the
  missing dense-conv control and dense lr asymmetry; root-caused the
  morphssm-dyadic collapse; slope/wording audits).
- Artifacts: `bakeoff_v1.jsonl` (v1), `v2_*.jsonl` (positional/lr fixes),
  `screen_*.jsonl` (300-step screens), `full_*.jsonl` (v3 fulls),
  `seed*_{dense,denseconv,morphssm}.jsonl` (replicates),
  `dense_lr*.jsonl` / `denseconv_lr*.jsonl` / `probe_*.jsonl` (lr sweeps),
  `full_morphssm_lr2e2.jsonl` (lr confirmation).

---

# Round 4: strict closure + BPE (2026-06-11)

**New standing constraint: strictly closed models only.** "Strict" is
machine-checked, not asserted: `--strict` runs `assert_closed_runtime` at
build (deploy closure at hard tau) and a full `find_training_multiplies`
trace over forward+loss+backward+optimizer for hard-from-scratch training,
waiving only the repo-sanctioned float-sim artifacts (loss-value batch-mean
divide — silicon emits the ±1 gradient; exp2 ROM table build and
log2-as-leading-zero-count in `shift_softmax`). Closed optimizers:
PureTropicalSGD, TropicalLion, SignSGD(momentum=0), CountAgreementSGD.
Closed loss: `tropical_ce_lut` (LUT softmax, shift normalization).

## The gradient-bias collapse (found, root-caused, fixed)

The first full-length strict suite (strictF_*) all collapsed into the
constant modal-token predictor: learn to ~step 300, drift dead by 400–1000
(acc pinned at exactly the modal-char rate 0.1490; BPE arm at modal-token
0.0879 with CE *rising* 6.13→6.50). Diagnostic probes isolated the cause:
`shift_softmax(normalize="shift")` yields sum(p) ∈ [0.5,1], so the
unbalanced CE gradient `(p − onehot)` carries a per-row bias of (s−1)/m —
measured at 1.5e-3..3.0e-2, the same order as the gradient signal — which
sign-based optimizers integrate into a constant-rate drift. Exact-normalize
CE (host or LUT) trains cleanly; LUT quantization itself is harmless;
perceptron loss avoids the bias but is near-useless for LM (3.74 val CE).

**Fix** (`tikitorch/functional.py`, TropicalCrossEntropyFn): scatter
**sum(p)** instead of 1.0 at the target — `grad = p − s·onehot`. Exactly
balanced for any normalization (row sums ≤1.2e-8 vs 1e-2 pre-fix),
sign-equivalent at the logit level, reduces to the old behavior for exact
softmax to within roundoff (≤3e-8), closure-clean (a scatter of a
precomputed sum). Verified by an adversarial review: all 101 relevant tests
pass; regression guard added
(`tests/test_functional.py::test_lut_shift_ce_grad_rows_sum_to_zero`).
Post-fix: zero collapse across 7 full runs; every run ends at its
run-minimum CE (one tau=0.5 transient bump in the anneal arm, recovered by
the anneal). Related: `classification_loss`'s `tropical_ce_lut` now passes
`normalize="shift"` (the old default ran a host reciprocal, contradicting
the "fully multiplier-free" label) — safe only together with the balanced
gradient; the coupling is documented at both sites.

## The strictness ladder (char, MorphSSM d=256 L=7, 1500 steps, hard deploy)

| stage | val CE (nats) | bpb | acc | added cost |
|---|---|---|---|---|
| float (AdamW + exact CE, anneal), n=3 | 1.934 ± .035 | 2.79 | 43.0% | — |
| sign updates (SignSGD mom 0.9 + exact CE, anneal) | 2.149 | 3.10 | 38.5% | +0.215 |
| **fully strict (PureTropicalSGD + LUT-shift CE, anneal)** | **2.231** | **3.219** | **35.9%** | **+0.082** |
| strict, hard-from-scratch (best: TropicalLion) | 2.440 | 3.520 | 29.5% | +0.209 |
| strict, hard-from-scratch (worst: PureTropicalSGD) | 2.476 | 3.572 | 29.7% | +0.245 |

- Total float → strict-hard: **0.51–0.54 nats depending on optimizer**.
- The big surprise: **removing the H7 momentum multiply and the host softmax
  costs only ~0.08 nats** (once the gradient is balanced). The expensive
  step is dropping the tau anneal (+0.21–0.25), and the annealed arm's
  terminal slope was 2–7× steeper than every hard arm's — the gap was
  *widening* at step 1499; nothing in the data supports hard-from-scratch
  closing it within a 2–3× budget.
- Strict-annealed (2.231) beats the bigram table (2.482) comfortably and
  sits 0.09 above trigram (2.140). Strict-hard ≈ bigram.
- **Closed-optimizer choice barely matters with the balanced loss**: all
  four hard arms within a 0.036-nat band — but each arm is n=1 and the
  float arm's n=3 seed range is 0.070, so within-band ordering is not
  resolvable (and the band itself may be seed noise). CountAgreement ran
  with windowed accumulation (N=4) and a CPU frexp fallback (MPS gap).

## BPE under strict closure (vocab 1024, byte-level, train-split-only)

bits-per-byte (the tokenizer-independent metric; val_tokens_per_byte
0.44307, verified self-consistent across every row):

| config | char bpb | BPE bpb |
|---|---|---|
| dense GPT (float ref) | 2.509 | **2.273** |
| dense-conv twin (float ref) | 2.455 | 2.351 |
| MorphSSM float | 2.810 | 2.498 |
| MorphSSM strict annealed | 3.219 | **3.071** |
| MorphSSM strict hard | 3.520–3.572 | 3.267 |

- **BPE helps the tropical side disproportionately**: float substrate gap
  vs the dense twin shrinks from 0.35 (char) to 0.15 bpb (BPE); under
  strict closure BPE wins both regimes like-for-like (anneal 3.071 vs
  3.219; hard 3.267 vs 3.52–3.57). **BPE + strict + annealed = 3.071 bpb
  is the best strictly-closed LM config measured.**
- Note dense-conv loses its edge over dense GPT under BPE (2.351 vs
  2.273) — the conv prior weakens with larger token units; architecture
  conclusions from char do not transfer automatically.
- Caveats: the BPE model is ~2× total params (the vocab-1024 embed+head is
  52% of the model; the *backbone* is matched to 0.2%); a 96-token BPE
  window spans ~2.26× more bytes of context (a legitimate tokenizer
  advantage that bpb absorbs); the BPE strict-hard arm was still improving
  at the steepest terminal rate of all hard arms (gap to BPE float: 0.77
  bpb at this budget — far from converged, NOT a plateau).
- Probes on the strict-hard BPE arm (500 steps): the "vocab-1024 head is
  the bottleneck" hypothesis is NOT supported — a vocab-256 control
  (which degenerates to byte-level: a 256-symbol ByteLevel alphabet
  admits no merges) is *worse* in bpb at matched steps (3.64 vs 3.40),
  and lr_shift=9 (2× step) helps modestly (3.30 at step 500). The arm is
  budget-limited, not head-limited; lr_shift 9 is the better default for
  BPE strict runs.
- BPE strict-hard acc 0.150 = 1.71× modal-token rate — genuine learning
  (the pre-fix run pinned at exactly modal 0.0879: the collapse exemplar).

## Strict-round provenance

- Runs: `strictB_*` (300-step screens), `strictF_*` (PRE-fix, collapsed —
  cite only as the bug exemplar), `diag_*` (loss-conditioning probes),
  `strictG_*` (post-fix fulls), `bpeA_*` (BPE float refs),
  `probe_bpe_*` (lr_shift/vocab probes). BPE tokenizer cache
  `data/bpe_tinyshakespeare_1024.json` verified train-split-only
  (retraining reproduces it bit-exactly; full-text training does not).
- Lesson re-confirmed: 300-step screens mis-rank (they said hard ≥ anneal;
  at 1500 steps anneal wins by 0.21–0.25 — the screen horizon sat inside
  the anneal's soft phase).
- All strict-round numbers are n=1 seed; the ladder's qualitative ordering
  (float < sign < strict-anneal < strict-hard) is large relative to the
  measured n=3 seed spread (0.07), the within-rung orderings are not.

## Repro (key arms)

```bash
export PYTHONPATH="$(git rev-parse --show-toplevel)"
# dense references (tuned)
python scripts/nanogpt_shakespeare.py --runs dense --steps 1500 --device mps --adam-lr 3e-3 --tag dense_lr3e-3
python scripts/nanogpt_shakespeare.py --runs dense-conv --steps 1500 --device mps --adam-lr 3e-3 --tag denseconv_lr3e-3
# headline tropical arm (n=3: --seed 0/1/2)
python scripts/nanogpt_shakespeare.py --runs tropical-float --variant morphssm --steps 1500 --device mps --adam-lr 1e-2 --tag full_morphssm
# hard-deployable closed stack
python scripts/nanogpt_shakespeare.py --runs tropical-closed --variant morphssm --steps 1500 --device mps --tag full_closed_morphssm
# best attention variant
python scripts/nanogpt_shakespeare.py --runs tropical-float --variant hilbert --steps 1500 --device mps --pos-embed --adam-lr 1e-2 --tag full_hilbert
# strict closure (round 4): best strict config (BPE + annealed)
python scripts/nanogpt_shakespeare.py --runs tropical-closed --variant morphssm --tokenizer bpe \
    --opt puretropical --loss tropical_ce_lut --strict --steps 1500 --device mps --tag strictG_bpe_anneal_puretrop
# strict, hard-from-scratch (Level-B training closure, machine-checked)
python scripts/nanogpt_shakespeare.py --runs tropical-closed --variant morphssm \
    --opt tropicallion --loss tropical_ce_lut --strict --tau 0 --no-anneal --steps 1500 --device mps --tag strictG_hard_tropicallion
```
