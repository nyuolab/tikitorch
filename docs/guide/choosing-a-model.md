# Choosing a model, optimizer, and loss

A short decision guide. The defaults are good; reach for the alternatives when the
listed condition applies.

## Which model?

| Your task | Model | Why |
|---|---|---|
| Sequence / LM / streaming signals | `MorphSSM` | causal morphological convs, `.step()` streaming, long receptive field |
| LM / tasks where *content routing* matters (induction, copy) | `TropicalTransformer` | tropical attention routes (argmax-gather) by distance |
| Scale capacity at fixed per-token compute | `MoTE` | sparse top-k mixture of tropical experts |
| Image classification | `TropicalCNN` | morphological (dilation/erosion) feature extractor |
| Tabular / vector classification, a simple baseline | `TropicalMLP` / `MorphMLP` | alternating min/max layers |
| Backprop-free / local learning | `TropicalForwardForward`, `DyadicForwardForward`, PC nets | forward-only or predictive-coding training |
| DP / shortest-path / sorting (learn an algorithm) | `algorithmic.TropicalReasoner` | the layer *is* the min-plus recurrence — see [Beyond DL](beyond-deep-learning.md) |

All take `task_type` (`"classification"` | `"lm"` | `"regression"`) and `head_type`
(`"tropical"` fully-closed, or `"linear"` declared-boundary). Start with
`head_type="linear"` to measure the strong number, then quantify the fully-closed cost.

## Which optimizer?

Every optimizer here is multiplier-free (sign-based or power-of-two steps).

| Optimizer | Use when |
|---|---|
| **`SignSGD`** | the default — sign-of-gradient, optional momentum. Robust, simple, what most examples use. |
| `PureTropicalSGD` | you want *strictly* pure (integer-counter momentum, power-of-two lr as a shift, LFSR noise) — the silicon-faithful one. |
| `TropicalAdam` | SignSGD stalls and you want adaptivity — per-parameter power-of-two step via sign-streak Rprop (zero-DSP). |
| `TropicalLion` | you want sign-of-**momentum** (stronger signal than sign-of-gradient), pow2 betas. |
| `FrexpMagnitudeSGD`, `ExponentEMAAdam`, `CountAgreementSGD`, `PerLayerStepSGD` | magnitude-aware closed adaptivity — tier the step by the gradient's frexp exponent or sign-agreement count. Try these to recover the SignSGD-vs-Adam gap. |

Rule of thumb: **start with `SignSGD`**; if it underfits, move to `TropicalAdam` or the
magnitude-aware family. The optimizer is one of the two big closure-penalty terms (see
[Why tropical for ML/DL](why-tropical-ml.md#the-closure-penalty)).

## Which loss?

| Loss | Use when |
|---|---|
| **`nn.TropicalCrossEntropyLoss(softmax="exact")`** | the default — closure-faithful CE, host-conditioned softmax. |
| `nn.TropicalCrossEntropyLoss(softmax="lut")` | you need the loss itself multiplier-free (full closed loop / on-silicon) — power-of-two exp LUT softmax. |
| `nn.TropicalPerceptronLoss` | only as the silicon baseline / ablation — it is the single biggest accuracy cost (~−17pp on hard char-LM). |

The functional forms (`F.tropical_cross_entropy`, `F.tropical_perceptron_loss`) are
identical; or just call `model.compute_loss(x, y)` which uses the model's `loss_type`.

## The `tau` soft→hard recipe

The curriculum is the third closure-penalty lever. For CNN/SSM/Transformer/MoTE:

```python
total = EPOCHS * len(loader); step = 0
for xb, yb in loader:
    model.set_tau(max(tau0 * (1 - step / (warmup_frac * total)), 0.0))  # anneal to 0
    ...
    step += 1
model.set_tau(0.0)        # eval & deploy: the hard, multiplier-free op
```

- `tau0 ≈ 0.5–1.0` to start (smooth, trainable), anneal to `0.0`.
- `warmup_frac ≈ 0.8` — reach hard before training ends so the last steps fine-tune the
  deploy graph.
- **Always `set_tau(0.0)` for eval and FPGA export** — `tau=0` is exactly the graph that
  maps to the zero-DSP kernel.

`TropicalMLP` fixes its softness at construction (`use_softmin`, `temperature`,
`tau_min`) instead of `set_tau`.

## Verify before you ship

```python
from tikitorch.audit import assert_closed, find_closure_violations
assert_closed(model)                 # fully closed, or…
assert_closed(model, allow={"head"}) # …backbone closed, head a declared boundary
find_closure_violations(model)       # list what (if anything) multiplies
```

See [FAQ & troubleshooting](faq.md) when training stalls or `assert_closed` complains.
