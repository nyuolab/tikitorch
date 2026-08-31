# Architectures

The main model families compose an input representation, tropical or hard-routed
blocks, normalization, and an output head. Exact constructor options differ by class;
use the [models API](../api/models.md) as the source for signatures.

The diagrams label tropical components, not whole-recipe closure. Soft routing, loss,
backward, optimizer, dropout, projections, and compiled implementation must be checked
separately with the [closure workflow](closure.md).

## MorphSSM — morphological state-space sequence model

A causal, streaming sequence model: depthwise **morphological convolutions**
(dilation/erosion with exponentially-spaced dilations) replace attention. Runs
token-by-token via `.step()` with an explicit state buffer. Specific FPGA kernels use
a corresponding streaming form, but the Python model is not itself a hardware build.

```text
tokens ─► Embed ─►┌───────────── MorphSSMLayer × n_layers ─────────────┐─► Tropical norm ─► Head ─► logits
                  │  Tropical norm                                       │                    [tropical
                  │      │                                               │                     or linear]
                  │      ▼                                              │
                  │  DepthwiseMorphConv1d  (max-plus / min-plus, causal)│
                  │      │   (multi-scale dilations 1,2,4,…,max_dilation)│
                  │      ▼                                              │
                  │  + residual                                          │
                  └─────────────────────────────────────────────────────┘
```

Use for: long sequences, streaming/clinical signals (ECG), LM. `kernel_size`,
`max_dilation`, `multi_scale` set the receptive field.

## TropicalTransformer — attention by routing

Replaces softmax attention with **tropical attention**: score queries↔keys by an
L1 / L∞ / Hilbert distance, then **argmax-gather** the values (route to one key per
query) instead of a softmax-weighted mix.

```text
tokens ─► Embed+Pos ─►┌──────────── TropicalTransformerBlock × n_layers ───────────┐─► Norm ─► Head ─► logits
                      │  Tropical norm                                               │            [tropical
                      │      │                                                      │             or linear]
                      │      ▼                                                      │
                      │  TropicalAttention   score = -L1/L∞/Hilbert(Q,K)           │
                      │      │   route = argmax-gather over keys (causal mask)      │
                      │      ▼                                                      │
                      │  + residual ─► TropicalLayerNorm ─► tropical FFN ─► +resid  │
                      └────────────────────────────────────────────────────────────┘
```

Use for: content-routing tasks (induction/copy), LM where attention matters.
`score_variant` picks the distance; `use_shift_attn` selects the dyadic shift attention.

## MoTE — mixture of tropical experts

Shared tropical layers wrap a sparse **mixture of experts**: a `TropicalRouter`
top-k-routes each token to a few tropical experts (transformer or SSM blocks).

```text
tokens ─► Embed ─► shared-pre ─►┌─────────── MoTEBlock ───────────┐─► shared-post ─► Head ─► logits
                  (tropical)    │  TropicalRouter ─► top-k experts │   (tropical)    [tropical or linear]
                                │     ├─ expert_1  (tropical)      │
                                │     ├─ …                         │   + load-balance aux loss
                                │     └─ expert_E  (tropical)      │     (aux_loss_weight)
                                └──────────────────────────────────┘
```

Use for: scaling capacity without scaling per-token compute. `n_experts`,
`capacity_factor`, `expert_type` configure it.

## TropicalCNN — morphological vision

A 2-D morphological feature extractor: each block is a **tropical conv** (learned
structuring element) + power-of-2 norm + power-of-2 leaky-ReLU + max-pool. Alternating
dilation/erosion gives opening/closing structure (more expressive than all-dilation).

```text
image ─►┌──────────── MorphConvBlock × len(channels) ───────────┐─► AdaptiveMaxPool(grid) ─► Head ─► logits
        │  TropicalConv2d   (dilation, then erosion, alternating)│                           [tropical
        │      │                                                 │                            or linear]
        │      ▼                                                 │
        │  TropicalLayerNorm2d ─► TropicalLeakyReLU(2^-k)        │
        │      │                                                 │
        │      ▼                                                 │
        │  TropicalMaxPool2d (stride 2)                          │
        └─────────────────────────────────────────────────────── ┘
```

Use for: image-classification experiments with learned dilation/erosion features.
Compare `head_type="linear"` and `head_type="tropical"` as an explicit ablation rather
than treating the head choice as a performance-neutral replacement. See
[notebook 03](https://github.com/nyuolab/tikitorch/blob/pc-morphssm/notebooks/03_vision_cnn.ipynb).

## Heads and other boundaries

Several composed models take `head_type`:

- `"tropical"` selects a tropical output projection where supported.
- `"linear"` selects a standard `nn.Linear` output boundary.

The registered path can be `head` or `output_head`, depending on the class. Inspect
`model.named_modules()` before declaring an `allow={...}` path. A tropical head is not
enough to establish full closure: input projections, soft routing, normalization,
dropout, loss, backward, and optimizer still belong to the recipe.

See [Choosing a model](choosing-a-model.md) for which to pick, and the
[API reference](../api/models.md) for every constructor argument.
