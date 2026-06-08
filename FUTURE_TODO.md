# TikiTorch — future TODOs

Known limitations and deferred work, kept honest for the paper.

## Optimizer: TropicalLion is a negative result so far
`tikitorch.optim.TropicalLion` (sign of an integer fixed-point momentum) was
intended to recover part of the SignSGD→AdamW closure gap (−12.8pp in the
char-LM waterfall) while staying multiplier-free. **Measured (Phase 3,
`scripts/lion_scaling_study.py`, char-LM, tau 0.5→0 anneal, 2000 steps):**

| scale | SignSGD | TropicalLion | AdamW |
|---|---|---|---|
| D=128 L=1 | 0.2859 | 0.2778 | 0.2881 |
| D=256 L=1 | 0.2944 | 0.2740 | 0.2925 |

- The SignSGD→AdamW gap **did not open** at these scales (AdamW ≈ SignSGD), so
  there was nothing to recover.
- TropicalLion **hurt** by 0.8–2.0pp despite live momentum (`momentum_density`
  > 0), likely because sign-of-momentum lags the rapidly-changing annealed-to-hard
  landscape.

**Action:** prefer `SignSGD` / `PureTropicalSGD`. Keep `TropicalLion` as a
documented option. **Definitive test (deferred to EC2/GPU):** D=512, ≥4000 steps,
3 seeds, including an open softmax-CE control where the AdamW gap is known to
exist — to determine whether Lion ever helps or should be removed.

## TropicalLayerNorm stability in deep / annealed-to-hard models
N1 closed the `nn.LayerNorm` holes: `TropicalLayerNorm` (subtract-mean +
power-of-2 mean-abs rescale, shift clamped to ±8) is now the default norm in
`MorphSSMLayer`, `TropicalTransformerBlock`, and all sequence models, so they are
multiplier-free out of the box (only `nn.Linear` input/output projections remain
as declared boundaries). **Cost:** without true variance scaling it is less
stable than `nn.LayerNorm` for deep + annealed-to-hard nets — a 2-layer MorphSSM
on sMNIST reaches ~0.21 vs ~0.32 with `nn.LayerNorm` and shows late noise (the
shallow char-LM headline is unaffected: +5pp CE recovery still holds). Pass
`norm_type="layernorm"` for the stable boundary fallback. **Open work:** a
genuinely stable multiplier-free norm — candidates: a learnable per-feature
power-of-2 gain (integer shift, straight-through), an output clamp
(`tropical_hardtanh`), or a running-statistics variant. Until then this is a
real closure-vs-stability tradeoff to state in the paper.

## Fully-closed heads at large vocabulary
The closed (tropical) LM/classification head routes each output through one
argmax (sparse gradient) and underperforms a dense `nn.Linear` head at high
fan-out (e.g. MNIST 10-way: ~0.55 closed vs ~0.96 with a boundary head;
vocab-32k LM untested). Default models keep the `nn.Linear` head as a *declared
DSP boundary*; a stronger fully-closed head (e.g. a wider compositional
dilation∘erosion, or a hierarchical/factorized tropical head) is open work.

## On-chip (Option-1) multiplier-free softmax CE
`tropical_ce_lut` (the fully multiplier-free loss via `shift_softmax`) is
validated on CPU. The on-chip HLS realization (dense backward, LUT exp) must be
shown to still close timing before it replaces the host-side CE on silicon
(see `OFFGPU_LEVERS.md` B2).

## Rigor-audit deferred items (lower severity)
The closure/rigor/docs audit's HIGH + quick-MEDIUM items are fixed (audit
soundness incl. GLU/PReLU/LRN/EmbeddingBag; soft-anneal freeze in
dilation/linear/RNN/transformer-FFN; MoTE all-zero capacity; TropicalSGD/Lion/
PureTropicalSGD docstrings + momentum_bits/decay guards; shift_softmax pow2 step;
streaming `step` residual fold; MorphSSM `set_tau`). Deferred:
- **MoTE expert token-mixing** runs each expert's routed tokens as one sequence,
  mixing across batch elements / non-adjacent positions in `nonzero()` order →
  output is batch/order dependent. Restrict to same-sequence/original order, or
  document. Also: capacity drops by lowest flat index (positional bias) not
  router score; `get_expert_utilization` returns `{}` (implement).
- **`MultiScaleMorphConv1d.step`** missing → `AttributeError` in the streaming
  path when `multi_scale=True` (forward works). Implement or guard.
- **`hardware_cost`** DSP/LUT constants differ across architectures (`//16` vs
  `*3`, LUT `//4` vs `*6+//8`) → non-comparable Fig-6 numbers; unify.
- **`TropicalConv2d`** soft path re-implements unfold/combine inline (drift vs the
  hard `TropicalConv2dFn`); factor a shared helper.
- **`TropicalShiftAttention` `rope_relative`** uses `m*g` (constant-per-position
  → foldable to a precomputed shift table; opt-in, `shift_mode='free'` is add-only).

## TropicalGRUCell (gated recurrence)
Multiplicative gating (`sigmoid ⊙ h`) has no exact tropical analog. An
additive-gate approximation is sketched in the audit but unbuilt; present any
such cell as "tropical gated recurrence," not a multiply-free GRU, and validate
before claiming parity.
