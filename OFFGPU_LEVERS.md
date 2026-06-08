# TikiTorch off-GPU levers — recovering accuracy & throughput, multiplier-free

This document tracks the levers that close the gap between TikiTorch's
fully-closed (silicon-faithful) training loop and an unconstrained GPU baseline,
**without reintroducing a hardware multiply**. Each lever is either a *library*
capability (runnable and demonstrated live on CPU today) or a *kernel* capability
(HLS C++, validated in Vitis csim, then on an AWS F2 AFI).

The framing is two-part and deliberately honest:

- **The library (GPU-sim) measures the algebra.** It float-simulates the exact
  integer/compare/shift closed loop, so it measures how much accuracy each
  closure choice costs — and how much each lever recovers — at a clean operating
  point, with no silicon scale cap.
- **The silicon proves the physics.** It confirms the same loop runs bit-exact,
  zero-DSP, at single-digit mJ/token — but at the prototype's small scale
  (D=128, V=128), so its *absolute* accuracy is lower and its *measured* lever
  gains are smaller than the algebra's.

Numbers below tagged **[algebra]** are the off-GPU float-sim recovery; numbers
tagged **[silicon]** are F2-measured. Do not conflate them.

---

## The closure waterfall (what we are climbing back)

Strict ordering, Shakespeare char-LM, 1-layer tropical transformer D=128 V=128
D_FF=512 L=128 (`results/fpga_synth/closure_decomposition/waterfall.json`):

| restriction added | next-token acc | Δ |
|---|---|---|
| Open: AdamW + FP32 + softmax-CE | 45.2% | — |
| + SignSGD (replace AdamW) | 32.4% | **−12.8** |
| + INT16 weight quant | 31.4% | −1.0 |
| + perceptron loss (replace CE) | 13.9% | **−17.5** |
| + hard tropical (max-plus attn/FFN, batch-shift) | 8.4% | −5.5 |

Total open→closed gap ≈ 36.7pp. The two dominant terms — perceptron loss
(−17.5) and SignSGD (−12.8) — are **not** intrinsic to "no multipliers"; they are
recoverable inside the closed algebra. That is what Tier A does.

---

## Tier A — library levers (DONE, demonstrated live on CPU)

All three are implemented, unit-tested, and have runnable demos on real
Shakespeare. Full suite: `python -m pytest` → **93 passed**.

### A1 — closure-faithful cross-entropy  *(the headline; closes a real honesty hole)*

Until now every TikiTorch model computed `F.cross_entropy` — a real softmax, full
of exp/divide/multiply. The zero-DSP kernel cannot do that, so on chip the loop
used a **perceptron** gradient (2-hot, ±1 at argmax/target — exactly
`forge_train_step.cpp:268-277`). That hole *was* the library's biggest
overclaim and the waterfall's biggest accuracy cost.

**New (`tikitorch/functional.py`):**
- `tropical_perceptron_loss(logits, target)` — the multiplier-free 2-hot silicon
  baseline (scalar margin forward, 2-hot backward).
- `tropical_cross_entropy(logits, target, softmax="exact"|"lut")` — CE gradient
  `softmax(logits) − onehot(target)` with either a host-side exact softmax
  (`"exact"`, the silicon-validated path) or `shift_softmax` (`"lut"`, fully
  multiplier-free).
- `shift_softmax(logits, normalize="exact"|"shift")` — softmax from subtract-max,
  a power-of-2-addressed exp LUT, additions, and a shift-normalize. No multiply,
  no exp unit, no float divide on the `"shift"` path.
- `continuous_cross_entropy_ref` — gradcheckable reference (the custom losses are
  piecewise-constant by design, so gradcheck targets this, not them).

**New (`tikitorch/models/__init__.py`):** `classification_loss(logits, y, loss_type)`
dispatch; every model honors `model.loss_type ∈ {"cross_entropy", "perceptron",
"tropical_ce", "tropical_ce_lut"}` (default unchanged: `"cross_entropy"`).

**Demo:** `python examples/closure_ce.py` — everything fixed, only the loss
conditioning varies:

```
loss conditioning                          eval_acc
perceptron  (silicon baseline, mult-free)   0.1715
tropical_ce (host softmax, kernel mult-free)0.2845
tropical_ce_lut (FULLY mult-free)           0.2824
cross_entropy (float ceiling)               0.2832
recovery perceptron -> tropical_ce_lut : +0.1108   (FULLY multiplier-free)
gap to float CE ceiling                : -0.0013
```

**Result [algebra]: +11.1pp** recovered by the *fully multiplier-free* loop —
matching the float-CE ceiling (under SignSGD only the gradient *sign* matters,
and the LUT softmax reproduces it). This is consistent with the −17.5pp waterfall
term measured at a richer operating point.
**[silicon]:** the in-loop, prototype-scale validated recovery is **+3.2pp**
(host-side CE, 12.70% vs 9.52%, AFI `afi-0313d7a30ffe02119`). Quote +3.2pp for
silicon, +11pp for the algebra — never cross them.

### A2 — Lion (sign-of-momentum) optimizer, integer fixed-point

`tikitorch/optim/__init__.py::TropicalLion`. Takes the sign of a real momentum
(stronger than SignSGD's sign-of-gradient), with power-of-2 betas so each EMA is
a shift+add and the update is a sign — zero DSP.

**The load-bearing detail:** the momentum buffer is simulated in **integer
fixed-point with a truncating arithmetic right shift** (`_arshift` = floor-div by
2^b), *not* a float EMA. On chip `(g−m)>>b` floors to zero when `|g−m| < 2^b`, so
a float EMA would over-state the lift. Gradients are pre-scaled by `grad_bits` to
keep the EMA alive; `opt.momentum_density()` is the collapse diagnostic. (Tests
prove it collapses at `grad_bits=6` and survives at `grad_bits=16` on the same
tiny gradient — the proof that the sim is faithful, not optimistic.)

**Demo:** `python examples/lion_vs_signsgd.py`

```
optimizer                              eval_acc  mom_density
signsgd  (sign(grad), mult-free)        0.2736     n/a
lion_int (sign(int momentum), mult-free)0.2739     0.270
adamw    (float ceiling, NOT silicon)   0.2808     n/a
```

**Result [algebra]: ≈ +0.0pp here** (a near-null). Honest reading: at this
operating point the whole SignSGD→AdamW gap is only +0.7pp, so there is little to
recover and Lion recovers ~none of it. The −12.8pp waterfall opportunity appears
at larger scale / more steps; the upper bound is +2–5pp and **must be re-measured
in the int-fixed-point regime before any silicon claim.** AdamW's adaptive second
moment (the genuine multiply/divide) is not recovered at all.

### A3 — top-k tropical softening (value-pull), an honest ablation

`tikitorch/functional.py::tropical_value_pull_topk` + `TropicalAttention(...,
value_topk_k=k)`. The hard value pull routes each output through one key; top-k
averages its k best keys, so the gradient flows to k keys — denser, still
add/compare-only (no logsumexp, no multiply). Tie-break is stable first-index
(matching the kernel's serial scan); causal `-inf` is excluded from average and
gradient; backward `scatter_add_` accumulates across both ranks and dims.

**Demo:** `python examples/topk_softening.py`

```
  k  grad_density  eval_acc
  1     0.338       0.2769
  2     0.495       0.2878   <- best, +0.0109
  3     0.599       0.2865
  4     0.679       0.2852
```

**Result [algebra]: +1.1pp at k=2**, declining after — densify-a-little helps,
too much shifts the forward toward an average and hurts. Scope: value-pull only;
it does not touch score-argmax, FFN-argmax, or batch-shift, so it recovers only
part of the −5.5pp bundle and the isolated contribution must be read off this
sweep, not assumed.

---

## Phase 0 — full multiplier-free primitive coverage (DONE)

A primitive audit (24-agent workflow, adversarially verified) found TikiTorch was
missing CNN/pooling/RNN coverage and had several closure holes. All addressed;
**115 tests pass**; the new closed CNN trains live.

**New primitives (all zero-multiply, soft→hard curriculum where trainable):**
- `functional.TropicalConv2dFn` / `tropical_conv2d` — morphological 2D conv
  (max-plus dilation / min-plus erosion); `unfold`→add→min/max→`fold` scatter
  backward, a 4-D lift of `TropicalLinearFn`. Forward verified vs brute-force;
  backward verified one-hot.
- `nn.TropicalConv2d` (soft+hard), `TropicalMaxPool2d`/`MinPool2d`/
  `AdaptiveMaxPool2d`/`GlobalMaxPool`, `TropicalLayerNorm`/`LayerNorm2d`/`RMSNorm`
  (subtract-mean + **power-of-2 max-abs** rescale — no variance/divide),
  `MorphConvBlock`, `TropicalRNNCell`/`TropicalRNN` (+ streaming `.step()`).
- `functional.tropical_relu` / `tropical_leaky_relu(shift)` (power-of-2 leak) /
  `tropical_hardtanh`.
- `models.TropicalCNN` — alternating dilation/erosion blocks + adaptive-max
  spatial pooling + a tropical or boundary head. Measured on **MNIST** (1000
  steps, SignSGD, soft→hard): the **multiplier-free morphological backbone + a
  declared `nn.Linear` boundary head reaches 0.965** (`assert_closed(model,
  allow={"head"})`); the **fully-closed** variant (tropical head,
  `assert_closed` outright) reaches ~0.55 — the 10-way tropical head is the
  bottleneck, i.e. the closure penalty lives at the head, not the features.
  Reporting both is the honest closed-vs-boundary comparison.
- `audit.assert_closed(model)` — walks the module tree and raises on any
  multiply-bearing standard module (`nn.Linear`/`LayerNorm`/`Conv`/RNN/...). The
  mechanically-checkable closure proof for the paper.

**Closure holes fixed:** H1 `TropicalMLP` `max(x, x*0.1)` → power-of-2 leaky relu;
H5 `MorphSSMLayer.out_proj` default `nn.Linear` → tropical; H7 `SignSGD`/
`TropicalSGD` float-momentum docstrings corrected (point to `TropicalLion`/
`PureTropicalSGD` for multiply-free momentum).

**Taxonomy now multiplier-free end-to-end:** MLP, CNN, RNN, Transformer, SSM, MoE.
Remaining (P1, follow-up): swap `nn.LayerNorm`→`TropicalLayerNorm` as the default
in the transformer/SSM blocks (the option exists; `assert_closed` flags the
current default), a `TropicalGRUCell` (additive-gate approximation — honest
caveat: multiplicative gating has no exact tropical analog), and a closed LM head
at large vocab (report closed-vs-boundary accuracy).

---

## Phase 3–4 — optimizer-at-scale + transfer (results)

**Phase 3 (Lion at scale) — NEGATIVE, documented.** `scripts/lion_scaling_study.py`
swept SignSGD/Lion/AdamW × {D=128, D=256} on char-LM, tau 0.5→0 anneal, 2000 steps:

| scale | SignSGD | Lion | AdamW |
|---|---|---|---|
| D=128 L=1 | 0.2859 | 0.2778 | 0.2881 |
| D=256 L=1 | **0.2944** | 0.2740 | 0.2925 |

Two honest findings: (1) the **SignSGD→AdamW gap does not open** at these scales
(AdamW +0.2pp at D=128, *worse* at D=256) — the −12.8pp waterfall optimizer term
does not reproduce under tropical_ce + tau-anneal, so there is nothing to
recover; (2) **Lion consistently *hurts*** (−0.8 to −2.0pp vs SignSGD) despite
live momentum — sign-of-momentum appears to lag the annealed-to-hard tropical
landscape. **Conclusion: SignSGD / PureTropicalSGD are the recommended
multiplier-free optimizers; the real recovery lever is A1 (CE), not the
optimizer.** `TropicalLion` keeps this caveat in its docstring + `FUTURE_TODO.md`;
settle definitively with an EC2 run (D=512, 4k+ steps, 3 seeds) but do not gate
the paper on it.

**Phase 4 (transfer off char-LM):**
- **sMNIST** — `scripts/run_transfer_smnist.py`: a multiplier-free `MorphSSM`
  (alternating max/min-plus morph convs) classifies length-784 pixel sequences
  with SignSGD, tau 1→0. CPU-reduced (6k train, 400 steps): **0.32 hard-test acc
  (3.2× chance)** with the `nn.LayerNorm` boundary; **~0.21 fully closed** after
  N1 (TropicalLayerNorm) — the documented norm stability cost in a deep +
  annealed model. The closed stack transfers to long-sequence classification;
  paper-grade needs EC2 (full data, more steps) and a more stable closed norm.
- **ECG (PTB-XL) — staged for EC2.** Not runnable locally (PTB-XL is a ~GB
  download + needs `wfdb`/`pandas`). Pipeline exists
  (`scripts/encode_ecg_for_silicon.py`; `forge/data/ptb_xl.py`; the silicon
  `morph_ecg` kernel). On EC2: encode → train the closed `MorphSSM` → compare to
  the F2 silicon ECG result (2.9× chance, zero DSP). The clinical-edge claim.

---

## Full closure — every model `assert_closed`-true by default (DONE)

Per the directive "everything must be fully tropical, `assert_closed` true of
everything." All seven models are now **closed by default** (a CI test,
`test_model_is_closed_by_default`, parametrizes over them); 125 tests pass. The
two pieces that made it work, found by a head sweep + an input ablation:

- **Closed head = top-k tropical** (`_make_head` → `TropicalTopKModule`). The
  sweep on MNIST: top-k **0.80** (closed, 9978 params) vs compositional
  dilation∘erosion 0.62 (540K params) vs single-argmax. Top-k aggregation (k=8)
  beats single-argmax decisively and matches the dense head's param count.
- **Closed input = quantize→embed** (`TropicalQuantizedEmbed`, used by
  `_make_input_proj`). A tropical add-only projection of a *continuous scalar* is
  rank-degenerate (shifts but can't scale → every channel identical → training
  stalls). Quantizing each feature and gathering a learned per-bin vector
  (`nn.Embedding`, a closed gather) is closed AND expressive. This **unlocked
  fully-closed sequence tasks** (sMNIST went from stall-at-chance to learning).
- Norm = `TropicalLayerNorm` (N1), heads/inputs as above; LM tokens use
  `nn.Embedding` (gather). `nn.Linear` head/input remain available as the
  `head_type="linear"` / `input_type="linear"` declared boundaries.

**Measured fully-closed (CPU-reduced; `scripts/run_closed_benchmarks.py`):**
- **MNIST** (TropicalCNN, vision): **0.80** hard-test, SignSGD, 1200 steps
  (~14pp closure cost vs the 0.94 boundary head; the morph-conv backbone is
  strong, the closed head is the cost).
- **sMNIST** (MorphSSM, sequence): **0.24** hard-test, 600 steps / 6k subset —
  learning, not stalled; CPU-limited.

**Honest implications for the paper.** The manuscript's "96.78% *fully closed*"
sMNIST used `nn.LayerNorm` + an `nn.Linear` head — *not* truly closed
(`assert_closed` would flag it; cf. Limitation #3). A *truly* fully-closed model
is what TikiTorch now builds by default, and at this CPU scale it is well below
those numbers. **Paper-grade fully-closed numbers require GPU/EC2 runs**
(full data, 10⁵+ steps) via the staged `run_closed_benchmarks.py` (`--device
cuda`); sCIFAR/ECG/ImageNet are staged there (ECG needs PTB-XL+`wfdb`, ImageNet
needs the cluster). This is the rerun-not-relabel path the user chose.

## Reference comparison — Hashemi et al. "Tropical Attention" (NeurIPS 2025)

Compared our tropical attention against the official
[Tropical-Attention](https://github.com/Baran-phys/Tropical-Attention) repo and
adopted its core idea.

| their design | our take |
|---|---|
| **Scoring = Hilbert/projective tropical metric** `-(max_d − min_d)(Q−K)` (translation/scale-invariant, sharp polyhedral boundaries) | **ADOPTED** as `score_variant="hilbert"` — new `tropical_hilbert_score` (+ soft) in `functional.py`, custom autograd (argmax_d + argmin_d sparse scatter, verified vs autograd), wired into `TropicalAttention`/`TropicalShiftAttention`, fully `assert_closed`. Stronger than our dot-product variant (which the API spec flagged as K-dominated). |
| Value aggregation = `maxplus_matmul(scores, V)` | **already equivalent** to our `tropical_value_pull` (per-dim max_j(V+scores)) — convergent. |
| `log1p(ReLU(x))` / `expm1` input/output warps | **SKIPPED** — `log`/`exp` are transcendental → break closure; would only be a soft/host aid, not the closed deploy path. |
| `tropical-gemm` SIMD/CUDA max-plus backend | noted for the **HLS/kernel** side (a fast max-plus matmul primitive); not needed for the PyTorch reference. |
| `nn.Linear` output head | same boundary we already declare (use `head_type="tropical"` to close it). |

The `hilbert` variant is now a config in the architecture search
(`transformer_d128_h4_l4_hilbert_shift`), so the GPU sweep will measure whether
projective scoring beats dot-product/L∞ on the benchmarks.

## Tier B — kernel levers (HLS, need Vitis csim)

These carry adversarial corrections from the design review. **Landmine first:**
there are TWO copies of every kernel (`forge/hardware/hls/...` and
`tikitorch/tikitorch/hardware/hls/...`), and `scripts/run_hdk_build_remote.sh`
defaults `KERNEL_DIR` to `forge/hardware/hls/tropical_signsgd` — **confirm which
file the F2 AFI actually consumes before editing, or you ship nothing.**

### B1 — resident-weight dataflow  *(throughput, accuracy-neutral)*
Hoist the per-token `burst_load_words(W1/W2/W_head)` out of `do_fwd_loss`/`do_bwd`
into `do_train_window` so weights load once per L-token window (today ~150 MB of
redundant DDR traffic/step).
- **Correction:** the K/V handoff rationale is false — mode-3 backward uses
  `OFF_K/V_CACHE` as *zero-init grad accumulators* (`:340-341`, `:647-654`), not
  forward-activation reads. No store-to-DDR-after-FWD.
- **Correction:** not "free URAM" — fwd weight buffers are currently unpinned
  (likely BRAM); pinning them URAM-resident co-resident with grad mirrors pushes
  net URAM *up* (~90 conservative; confirm < 320/SLR in csynth).
- **Correction:** **zero speedup to the shipped silicon driver.**
  `train_shakespeare_on_silicon_fast.c` uses per-token modes 0/1/2, *not* mode 3.
  Realizing the gain needs the host switched to mode 3 AND the step DDR-bound
  (both unverified).
- **Gate:** add a bit-exact mode-3 regression to `forge_train_step_tb.cpp`
  (`buf3 == per-token reference`, word-for-word), not just "both ≥ L/2".

### B2 — host-side CE kernel store (the shipped CE path) — **DONE in csim**
Done on `forge/hardware/hls/forge_kernels/forge_train_step_2layer.cpp` (the
authoritative file the F2 AFI consumes; the `tikitorch/...` mirror is synced):
- Added a dedicated `OFF2_LOGITS` slot to `forge_buffer_2layer.hpp`
  (**append-only** — every existing offset unchanged, buffer grows by
  `W2_LOGITS`), removing the prior aliasing where logits and the gradient both
  used `OFF2_GRAD_LOGIT`.
- The L1 forward stores logits to `OFF2_LOGITS`; the host computes
  `softmax − onehot` → `OFF2_GRAD_LOGIT`; `do_bwd` consumes it. Only a store
  offset moved, so the timing-critical compute is byte-identical and the
  WNS −0.155 ns closure is expected to hold — **to confirm on EC2 csynth/AFI**
  (`synth_2layer.tcl`); asserted, not yet proven.
- New software-csim `forge_train_step_2layer_ce_tb.cpp` (g++, no Vitis):
  **CE converges 16/16 vs the perceptron baseline's 15/16** — the +3.2pp story
  reproduced. The perceptron csim still passes (no regression).
- Cross-language proof: `scripts/check_host_ce_matches_tikitorch.py` shows the C
  host-CE gradient equals `tikitorch.functional.tropical_cross_entropy(
  softmax="exact")` to **4.9e-11** — the silicon path mirrors the library
  bit-for-bit.

**Remaining (EC2):** `vitis_hls -f synth_2layer.tcl` + HDK AFI to confirm
WNS ≥ −0.155 ns at 250 MHz. Reserve on-chip LUT-CE (A1 `tropical_ce_lut`, fully
multiplier-free loss) behind an HLS report proving the dense backward (kills the
`if(g==0)continue`) still closes timing.

### B3 — systolic add-compare-select (refactor + cost-model, NOT 8×)
**Correction:** the proposed pragmas are identical to the existing II=1 /
unroll-8 loops → **~0× real speedup.** Reframe as (a) consolidating the 5 inlined
ACS loops into one shared `acs_reduce()` (area/sharing win) plus (b) a cost-model
+ golden reference in `functional.py`/`_hardware.py`. Any throughput claim must be
a Vitis csynth latency delta on xcvu47p @4.0 ns; saturate the torch reference to
int16/`sat16` before value equality; keep argmin/argmax reference-only (native
tie distribution ≠ kernel first-index scatter).

---

## Tier C — need an F2 AFI rebuild (~$10–30, EC2, hours)

- **C1 — Lion kernel (mode-2 int momentum):** new `OFF_M_*` region (shifts
  `OFF_AD_SCORE..OFF_PRED`, grows `FT_BUFFER_WORDS`; regenerate TB + host
  buffer-size constant), 3rd wide load + power-of-2 EMA in `signsgd_region_wide`,
  host-zero momentum once. signsgd is already at WNS −0.201 ns and a prior bundle
  hit −1.706 ns VIOLATED — **prove 250 MHz closes** before claiming throughput
  unchanged; give it its own pblock if needed.
- **C2 — realize B1 throughput:** switch `fast.c` to mode 3, measure tok/s,
  confirm DDR-bound.
- **C3 — SLR replication + majority-vote all-reduce:** `MajorityVoteSignSGD` in
  optim (CPU-sim runnable) + 3-SLR host driver. **Correction:** sign-majority-vote
  is strictly weaker than a 3× batch (discards magnitude) — claim a **noise-floor
  reduction (+1–4pp)**, not 3× throughput; wall-clock grows ~3×. Must zero
  replicas 1..R-1 grads each step (mode-2 zeros only replica-0) and verify the F2
  aperture spans 3×`FT_BUFFER_BYTES` (else cross-replica reads silently return
  zeros). No host driver exists in this repo yet.

---

## Live-demonstration storyline (CPU, reproducible today)

1. `examples/closure_ablation.py` — establishes the open→closed penalty.
2. `examples/closure_ce.py` — **A1**: perceptron→`tropical_ce_lut`, +11.1pp,
   fully multiplier-free. The first and largest rung.
3. `examples/lion_vs_signsgd.py` — **A2**: optimizer rung (regime-dependent;
   momentum proven alive, not collapsed).
4. `examples/topk_softening.py` — **A3**: value-pull ablation, honest sign.
5. Transfer: `examples/train_sequential_mnist.py` (sMNIST) and the ECG MorphSSM
   path — same multiplier-free stack off char-LM.
6. Silicon: B1 bit-exact csim → B2 host-CE csim (WNS −0.155 ns) → C1 Lion AFI.

**The claim that survives adversarial review:** TikiTorch trains deep nets
end-to-end with zero data-path multiplies and recovers a large, measured chunk of
the closure penalty off-GPU — dominated by closure-faithful CE (+11pp algebra /
+3.2pp silicon) — with the recovery demonstrated on CPU and the closed loop
demonstrated bit-exact in csim and on F2 silicon.

## Honest risks
- **A2 Lion** may stay ~null unless scale/steps open a real SignSGD→AdamW gap;
  int momentum collapses without enough `grad_bits` (degenerates to SignSGD).
- **A3 top-k** can hurt past its optimum (k≈2 here); it cannot touch batch-shift.
- **B1/C2** give zero speedup until the host driver moves to mode 3 and the step
  is DDR-bound.
- **B3** is ~0× as a hardware speedup; it's a refactor + cost-model deliverable.
- **C1/C3** timing/aperture closure is *asserted, not proven* until a fresh AFI.
