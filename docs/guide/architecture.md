# Architectures & diagrams

The four headline models in `tikitorch.models` share a shape: a (closed or declared)
**input projection** → a stack of **tropical blocks** → a multiplier-free
**normalization** → a **head** (fully tropical, or a declared `nn.Linear` DSP
boundary). The blocks differ in how they mix across the sequence/space. All support
the `set_tau()` soft→hard curriculum and expose `compute_loss(x, y)`.

Legend: `[closed]` = multiplier-free; `[DSP]` = a declared multiply boundary you opt
into with `assert_closed(model, allow={"head"})`.

## MorphSSM — morphological state-space sequence model

A causal, streaming sequence model: depthwise **morphological convolutions**
(dilation/erosion with exponentially-spaced dilations) replace attention. Runs
token-by-token via `.step()` with an on-chip ring buffer — the form the FPGA kernel
streams.

```text
tokens ─► Embed ─►┌───────────── MorphSSMLayer × n_layers ─────────────┐─► TropicalLayerNorm ─► Head ─► logits
                  │  TropicalLayerNorm                                  │       [closed]        [closed
                  │      │                                              │                        or DSP]
                  │      ▼                                              │
                  │  DepthwiseMorphConv1d  (max-plus / min-plus, causal)│
                  │      │   (multi-scale dilations 1,2,4,…,max_dilation)│
                  │      ▼                                              │
                  │  + residual  (max-plus skip)                        │
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
                      │  TropicalLayerNorm                                          │  [closed]  [closed
                      │      │                                                      │             or DSP]
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
                  (tropical)    │  TropicalRouter ─► top-k experts │   (tropical)    [closed or DSP]
                                │     ├─ expert_1  (tropical)      │
                                │     ├─ …                         │   + load-balance aux loss
                                │     └─ expert_E  (tropical)      │     (aux_loss_weight)
                                └──────────────────────────────────┘
```

Use for: scaling capacity without scaling per-token compute. `n_experts`,
`capacity_factor`, `expert_type` configure it.

## TropicalCNN — multiplier-free vision

A 2-D morphological feature extractor: each block is a **tropical conv** (learned
structuring element) + power-of-2 norm + power-of-2 leaky-ReLU + max-pool. Alternating
dilation/erosion gives opening/closing structure (more expressive than all-dilation).

```text
image ─►┌──────────── MorphConvBlock × len(channels) ───────────┐─► AdaptiveMaxPool(grid) ─► Head ─► logits
        │  TropicalConv2d   (dilation, then erosion, alternating)│        [closed]           [closed
        │      │                                                 │                            or DSP]
        │      ▼                                                 │
        │  TropicalLayerNorm2d ─► TropicalLeakyReLU(2^-k)        │
        │      │                                                 │
        │      ▼                                                 │
        │  TropicalMaxPool2d (stride 2)                          │
        └─────────────────────────────────────────────────────── ┘
```

Use for: image classification under a hardware/energy budget. `head_type="linear"`
+ `allow={"head"}` reaches ~96–97% MNIST with a sign-only optimizer;
`head_type="tropical"` is fully closed (the closure penalty shows at the 10-way head).
See [notebook 03](https://github.com/nyuolab/tikitorch/blob/main/notebooks/03_vision_cnn.ipynb).

## The head: the one place a multiply may hide

Every model takes `head_type`:

- `"tropical"` — a compositional dilation∘erosion head; **`assert_closed(model)` passes
  outright**. Weaker at wide classification heads (that gap *is* the closure penalty).
- `"linear"` — a standard `nn.Linear`; the backbone stays closed and the head is a
  **declared DSP boundary**: `assert_closed(model, allow={"head"})`. Report both numbers.

See [Choosing a model](choosing-a-model.md) for which to pick, and the
[API reference](../api/models.md) for every constructor argument.
