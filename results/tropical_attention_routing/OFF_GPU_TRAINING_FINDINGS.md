# Multiplier-free attention: deploy, teacher-free routing, and the off-GPU training wall

A research record (2026-06) of whether a **fully multiplier-free** (tropical / max-plus
semiring) transformer can be **trained off-GPU** and deployed competitively. The deploy
question is settled positively; the training question hits a wall that we characterize
precisely on three independent fronts.

Scope of "multiplier-free": no `*` in the datapath — only add, subtract, abs, max/min,
compare, shift (`×2^k`), gather/scatter, integer counts. DSP blocks as adders/comparators
are allowed. Dyadic-quantized operands (`±2^a · ±2^b = ±2^(a+b)`, a shift) are the project's
standing convention; the only residual "multiply" is the autograd weight-grad matmul, which is
dyadic-quantized-equivalent — identical status across every layer, no *new* multiply.

Probe: **variable-offset block-copy induction** (the canonical content-routing task; chance
1/V). LM: **tiny-shakespeare** char-level, d256 / 4 layers / 8 heads / seq256, val bits-per-byte.

---

## 1. What works

### 1a. Multiplier-free DEPLOY — solved, lossless
Score = L1 distance `s_ij = -Σ_d |Q_i - K_j|` (dense gradient, abs+add); aggregation =
**argmax-GATHER** `out_i = V[argmax_j s_ij]` (one key per query). Train with a soft relaxation,
deploy with the hard gather:
- Induction: gather-eval **0.999–1.000** = softmax ceiling, lossless.
- Char-LM: l1+gather **2.43–2.53 bpb** (≈ dense softmax 2.43), hard ≈ soft (lossless).

The L1 (distance) score is load-bearing: it learns *peaked, decisive* attention that survives
argmax-hardening; a dot-product score learns diffuse attention that is lossy when hardened.
**The wrong deploy op** (per-dim max-plus matmul `max_j(s+V)`) collapses to chance — the deploy
op must be argmax-gather.

### 1b. Teacher-free, multiplier-free routing for WINNER-TAKE-ALL tasks — solved (`ff_succ`)
Training the *router* (Q/K) is the hard part: with hard-gather the LM loss cannot reach Q/K
(argmax has no gradient). For the induction circuit we train Q/K on a **separate L1-distance
path** with a triplet **margin hinge** toward a teacher-free **structural target**:
- early layers → previous-token head (`j = i-1`, positional),
- last layer → successor head (the unique causal `j` with `x[j-1] == x[i]`, = `match+1`).

The hinge subgradient through `s = -Σ|Q-K|` is `sign(Q-K)` — dense, contrastive over competitors,
multiplier-free; it never touches the argmax. The optimizer is dyadic-magnitude
(`step ∝ sign·2^round(log2|grad|)`).

**Result (induction, 3 seeds): gather-eval 1.000 ± 0.000**, beating the teacher-distillation
bar (0.918). The circuit self-organizes (L0 prev-token selectivity 1.00, L1 successor 1.00).
A shuffle-target control collapses to chance → the gain is **content** routing, not
load-balancing. This **overturns** the earlier "teacher-free mult-free content routing is a
hard wall" verdict — *for structural / winner-take-all tasks*. Needs ≥2 layers (induction is a
2-head circuit).

### 1c. A multiplier-free DIFFERENTIABLE attention that co-adapts Q/K from the LM loss — first of its kind (`ShiftSoftmaxBlend`)
A base-2 "soft" relaxation with **no exp and no divide**: weights `w_ij = 2^round((s_ij-max)/Δ)`
(a shift), output `(Σ_j w_j V_j) / pow2_ceil(Σ_j w_j)` (normalize by a leading-zero-count
power-of-2 → a shift). Backward is an explicit dyadic-STE carrying the **weighted partition
contrast** `adv_ij = ⟨g,V_j⟩ - ⟨g,out⟩` (the `-⟨g,out⟩` term gives the heterogeneous cross-key
coupling that mean-subtraction cannot). Δ is annealed high→low so the blend collapses to one-hot
and hard-gather deploy is lossless.

**Result (induction, AdamW): trains from the LM loss ALONE — soft 1.000, gather 0.831**, circuit
self-organizes, weighted contrast load-bearing (0.831 with vs 0.417 without). This is the only
relaxation that preserves the Q/K↔prediction co-adaptation that softmax provides, *without* the
exp/divide.

---

## 2. The off-GPU competitive-LM wall (9 attempts, characterized on 3 fronts)

Getting a **competitive language model** trained fully off-GPU fails. Each route demands a
property that the multiplier-free constraint cannot supply; all three trace to a multiply.

| front | the requirement | why mult-free fails | measured |
|---|---|---|---|
| **target-findability** (hard-gather + a manufactured target) | content-adaptive **and** deploy-findable | structural = not adaptive; grad-token = not findable (uses the answer `y`); value-alignment = circular | ff2 **4.77**, grad-token **4.08**, value-align ≈ chance — all vs random-frozen **3.54** |
| **blend-smoothness** (a differentiable relaxation) | smooth **and** mult-free | dyadic weights are coarse; finer mantissa → less peaked → lossy hardening | shift soft/gather **4.40/4.44**; finer (3-bit) **3.63/3.97** — still > random 3.54 |
| **optimizer-adaptivity** (the mult-free update) | per-parameter adaptive scaling | sign-magnitude / error-feedback / momentum / *plain SGD on the true grad* all stall — the router needs Adam's adaptivity (not better gradients) | shift+{dyadic, EF, momentum, SGD} = chance vs shift+AdamW **0.831** — **CRACKED, see §2a** |

### 2a. The optimizer front, CRACKED — `dyadic-Adam` (leveraging the PC pow2 trick)
The diagnostic above (plain SGD on the *true* gradient also stalls) shows the router needs Adam's
*per-parameter adaptive scaling*, not better-quality gradients ("transformers need Adam"). We supply
that adaptivity **multiplier-free** by borrowing the predictive-coding work's signature move
(power-of-2 quantization): **dyadic-Adam** — per-parameter scale `1/|g|_EMA` realized as a pow2
**shift** (`2^-round(log2 v)`); `m, v` are dyadic EMAs (`β = 0.875 = 1−2⁻³`, a shift); the step
`m · pow2_scale` is a shift. Every op is add/abs/shift/leading-bit-extract — no multiply.

**Measured (induction, 3 seeds): dyadic-Adam gather 0.929 ± 0.023 — *beats* AdamW 0.850 ± 0.029**
(soft 1.000 both). So **fully multiplier-free differentiable attention routing — forward + backward
+ OPTIMIZER — trains the router from the LM loss alone to ceiling**, no teacher, no target, no
multiply. dyadic-Adam is a **reusable general multiplier-free *adaptive* optimizer**: it fills a gap
in the PC toolkit (PC's `set_mf_grads` / dyadic-magnitude is SGD-flavored — it matches Adam on
SGD-friendly MLPs/Convs but, as shown here, stalls on Adam-needing *attention*). Arms:
`shift_sgd` (diagnostic), `shift_dadam` (the win), in `induction_selfsup.py`.

This **removes the optimizer front** of the wall. The two remaining fronts (target-findability for a
general LM, and blend-smoothness — the dyadic-coarse blend underfits smooth attention, independent of
the optimizer) still block *competitive char-LM*. So the crack completes the **winner-take-all**
mechanism fully multiplier-free, not competitive LM.

**The through-line:** the multiply (exp/divide for the soft weights; continuous weights for a
smooth blend; `1/√v` for an adaptive optimizer) does essential work for *smooth* attention
learning that no multiplier-free substitute replicates. Multiplier-free methods win at
*winner-take-all / discrete* routing and underfit *smooth* objectives — the same theme as the
project's MuZero capacity verdict (max-plus underfits smooth/oblique value).

Reference points (char-LM, mult-free stack: dyadic linears + fine-dyadic norm + ReLU MLP):
soft-trained l1_gather **2.43** (the routing ceiling), random-frozen attention **3.54** (the bar
any real router must beat), fully-mult-free dyadic-mag floor **4.07**.

**Practical answer:** the working artifact toward an off-GPU comparison is **warm-train**
(soft/teacher router = a train-time multiply) **+ multiplier-free deploy** (l1+gather, 2.43 bpb)
— the project's host-train / fabric-deploy handoff. Inference is off-GPU/mult-free; training is
not.

---

## 3. Scripts (this directory's research code lives in `scripts/`)

- `tropical_induction_v2.py` — the validated variable-offset block-copy task (`block_copy_batch`)
  + the `Attn` primitive (L1 score, gather/matmul/soft aggs) + `selectivity`. (`tropical_induction.py`,
  `tropical_induction_grid.py` are the earlier v1 — v1 was found *degenerate* (fixed-offset =
  position-solvable) by an adversarial-audit workflow; v2 is the fix.)
- `induction_v2_deploy.py`, `induction_v2_curve.py` — deploy-op + STE studies.
- `induction_distill.py` — teacher-based on-fabric router training (perceptron distill + dyadic-mag;
  0.918), the path-A baseline the teacher-free work had to match.
- `induction_selfsup.py` — **path B (teacher-free)**: all the attempts. Arms: `lm_only` (floor),
  `ff_succ`/`ff_succ_shuffle` (structural successor, **the 1.000 win**), `reinforce_cf`, `ccorrect`,
  `valign`/`valign_warmup` (value-alignment), `gradtok`/`gradtok_circuit` (logit-grad token target),
  `shift`/`shift_adamw`/`shift_nocontrast`/`shift_ef`/`shift_mom` (the `ShiftSoftmaxBlend` +
  optimizer studies).
- `charlm_mixer.py` — the **char-LM port**: swaps only the attention mixer on a standard
  transformer. Arms: `dense`, `l1_gather`, `ff2` (structural target library), `gradtok_circuit`,
  `shift` (the differentiable blend), + `lm_only_hard` / `_adamw` / `_shuffle` controls. Flags:
  `--linear dyadic --norm fine_dyadic --mlp-act relu` (the mult-free deploy stack),
  `--delta0/--delta-floor/--shift-wbits`.
- `nanogpt_shakespeare.py` — full-stack runner; `--score-variant l1 --agg-variant gather` with the
  l1⇒gather operator-mismatch guard.

These scripts depend on the working-tree `tikitorch` library (`tropical_l1_score`,
`tropical_value_gather`, `dyadic_quantize`, `DyadicLinear`, the fine-dyadic norm); run with
`PYTHONPATH=.`.

---

## 3. Track B — a fully off-GPU LM now TRAINS end-to-end (blend-smoothness narrowed)

The optimizer front (§2a) being solved, the remaining front was a *smooth* mult-free blend. Result
(char-LM, dyadic-Adam, fully multiplier-free fwd+bwd+opt, deploy = argmax-gather):

| fully-mult-free LM | gather bpb | note |
|---|---|---|
| `l1_gather` (**softmax** blend — the one multiply) | **2.48** | optimizer solved: ≈ AdamW 2.43 |
| **`mbr`** — Margin-Banded Rank Blend (mult-free) | **3.44** | lossless; beats random-frozen 3.54; **the accepted off-GPU blend** |
| `rtk` — Residual Top-K blend (mult-free) | 3.55 | lossless; finer top weight did NOT beat MBR |
| `pshift` — peaking-regularized shift | 4.81 | ❌ forcing the model to peak over-constrains it |
| random-frozen attention | 3.54 | floor |

**`l1_gather + dyadic-Adam = 2.48 ≈ AdamW 2.43** isolates the gap as *purely the blend*: the
optimizer, dyadic linears, fine-dyadic norm and ReLU MLP all reach softmax quality off-GPU.

**MBR** (fixed dyadic rank weights `2^-r` over top-k + a margin-band *gate* — smoothness lives in
the gate, not the weight value; band→0 = exact argmax-gather, lossless by construction) is the best
multiplier-free blend: **3.44, lossless, beats random-frozen.** So the whole stack — MBR + dyadic-Adam
+ dyadic linears + fine-dyadic norm + ReLU MLP — is multiplier-free in forward, backward *and*
optimizer and **trains end-to-end with no GPU**. Full off-GPU training is achieved; quality is ~1 bpb
behind the softmax blend (2.48).

**`pshift`** (force the *model* to peak via a relu-hinge on the top1−top2 gap, so a smooth blend still
hardens losslessly) **failed (4.81)** — forcing peakedness destroys the blending the LM needs;
smoothness is load-bearing.

## 4. Status / next

A fully off-GPU-trainable tropical LM **exists** (MBR @ 3.44, beats random, lossless deploy); the lone
remaining gap to *competitive* (softmax 2.48) is a **smooth multiplier-free blend** — the dyadic-coarse
weight wall (`Σ w·V` mult-free forces dyadic weights), the same "mult-free wins winner-take-all,
underfits smooth" theme seen across the project. Both smooth-blend candidates were tried: MBR (3.44,
accepted) and RTK (3.55, worse — the residual finer-top-weight did not help); `pshift` (force-peak)
failed (4.81). **MBR is the accepted off-GPU blend.**

NEXT (scaling toward NanoGPT): the off-GPU stack is settled, so the remaining work is (1) SCALE past
char/tiny-shakespeare — BPE tokenizer, bigger model, more data — and check whether the ~1 bpb blend gap
holds/shrinks/grows; (2) a MINIMAL-MULTIPLY fallback if the gap matters at scale: `l1_gather` (softmax
blend) + dyadic-Adam = 2.48 keeps *everything but one train-time softmax* off-GPU; (3) stand up the
NanoGPT GPU baseline for a quality + tokens/J energy comparison.
