# Migrating from PyTorch

TikiTorch mirrors the PyTorch namespace (`tt.nn`, `tt.optim`, `tt.functional`,
`tt.models`) and reuses `nn.Module` wholesale, so most of your muscle memory
carries over: `.parameters()`, `.state_dict()`, `.to(device)`, `.train()/.eval()`,
`nn.Sequential`, and `torch.optim.Optimizer`-style `opt.step(); opt.zero_grad()`
all work unchanged. What changes is the **algebra**: every op reduces to
compare / add / bit-shift, so the whole loop is multiplier-free and the one new
invariant is **closure** — `tikitorch.audit.assert_closed(model)`.

## The swap table

| PyTorch | TikiTorch | Notes |
|---|---|---|
| `nn.Linear` | `nn.TropicalLinear` (erosion) / `nn.TropicalDilation` (dilation) | `y_j = min_i (W_ji + x_i)` / `max_i`; gradient routes to one argmin/argmax |
| `nn.Conv2d` | `nn.TropicalConv2d(..., mode="dilation"|"erosion")` | learned grayscale morphology — add structuring element, reduce by compare |
| `nn.MaxPool2d` | `nn.TropicalMaxPool2d` / `nn.TropicalMinPool2d` | already tropical in PyTorch |
| `nn.LayerNorm` | `nn.TropicalLayerNorm` | subtract-mean + **power-of-2** rescale (no variance/divide) |
| `nn.BatchNorm*` | — (use `nn.TropicalLayerNorm` / `nn.TropicalBatchShift`) | variance + divide cannot be closed |
| `nn.ReLU` | `nn.TropicalReLU` | `max(x, 0)` — already tropical |
| `nn.LeakyReLU(0.01)` | `nn.TropicalLeakyReLU(shift=k)` | slope is a **power of two** `2**-k` (a shift, not a multiply) |
| `nn.Sigmoid/Tanh/GELU` | — | smooth dense activations are incompatible with the semiring |
| `nn.Dropout` | `nn.TropicalDropout` | a `{0,1}` gate, **no** `1/(1-p)` survivor rescale |
| `nn.Embedding` | `nn.Embedding` (reused) | gather-only, already multiply-free |
| `nn.MultiheadAttention` | `nn.TropicalAttention` | L1/L∞/Hilbert scores + argmax-gather routing |
| `nn.CrossEntropyLoss` | `nn.TropicalCrossEntropyLoss(softmax="exact"|"lut")` | `softmax="lut"` keeps the loss multiplier-free |
| `F.cross_entropy` | `F.tropical_cross_entropy` | functional form of the above |
| `torch.matmul` | `F.tropical_matmul` | `C_ij = min_k (A_ik + B_kj)` |
| `optim.SGD` | `optim.SignSGD` | sign-of-gradient; lr can be a power-of-2 shift |
| `optim.Adam` | `optim.TropicalAdam` / `optim.PureTropicalSGD` | per-param power-of-2 step (Rprop-style), zero-DSP |

The clean short names (`TropicalLinear`) and the original research names
(`TropicalLinearModule`) point to the same class — prefer the short alias in new
code. See the full surface in the [API reference](../api/nn.md).

## Your training loop, before and after

=== "PyTorch"

    ```python
    import torch, torch.nn as nn

    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 14 * 14, 10),
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()

    for xb, yb in loader:                       # xb: (B, 1, 28, 28)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
    ```

=== "TikiTorch (closed)"

    ```python
    import torch, tikitorch as tt, tikitorch.optim as optim
    from tikitorch.audit import assert_closed

    model = tt.models.TropicalCNN(in_channels=1, num_classes=10, channels=(16, 32),
                                  head_type="linear", use_soft=True, tau=1.0)
    assert_closed(model, allow={"head"})        # backbone closed; head = declared boundary
    opt = optim.SignSGD(model.parameters(), lr=2e-3, momentum=0.9)
    crit = tt.nn.TropicalCrossEntropyLoss()     # or model.compute_loss(x, y)

    total = EPOCHS * len(loader); step = 0
    for xb, yb in loader:                        # xb: (B, 1, 28, 28)
        model.set_tau(max(1.0 * (1 - step / (0.8 * total)), 0.0))   # soft -> hard
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step(); step += 1
    model.set_tau(0.0)                           # deploy the hard, multiplier-free op
    ```

    Set `head_type="tropical"` for a *fully* closed model — `assert_closed(model)`
    then passes with no allowance (the closure penalty shows up at the 10-way head).

Three new ideas, all covered in the guide:

1. **The soft→hard curriculum.** The hard `min`/`max` has sparse gradients. Train
   with a soft relaxation at temperature `tau > 0` and anneal `tau → 0`; the *same
   weights* drive the smooth-training and the hard-deploy graph. See
   [Foundations](foundations.md#the-softhard-curriculum).
2. **`assert_closed` is your test.** It walks the model and raises on any hardware
   multiply/divide. A `head_type="linear"` classifier is a *declared* boundary —
   allow it explicitly with `assert_closed(model, allow={"head"})`.
3. **Sign-based optimizers.** `SignSGD` uses only the gradient sign; the adaptive
   family (`TropicalAdam`, `PureTropicalSGD`, the magnitude-aware optimizers) keep
   per-parameter steps as power-of-two shifts. See [Choosing a model](choosing-a-model.md).

## What intentionally has no equivalent

Closure removes a few PyTorch tools by construction, not by omission:

- **BatchNorm / variance normalization** — needs a divide. Use `TropicalLayerNorm`
  (mean-subtract + power-of-2 rescale) or `TropicalBatchShift`.
- **Smooth dense activations** (sigmoid, tanh, GELU, SiLU) — not expressible as
  compare/add/shift. The tropical nonlinearity *is* the `min`/`max` routing itself.
- **A free-lunch swap.** Closure costs accuracy — measured, and recoverable with the
  levers above. Read [Why tropical for ML/DL](why-tropical-ml.md#the-closure-penalty)
  for the honest decomposition before you port a model you care about.
