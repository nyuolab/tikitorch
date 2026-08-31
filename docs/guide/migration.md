# Migrating from PyTorch

TikiTorch mirrors familiar PyTorch namespaces (`tt.nn`, `tt.optim`, `tt.functional`,
and `tt.models`) and builds on `torch.nn.Module`. Common conventions such as
`.parameters()`, `.state_dict()`, `.to(device)`, `.train()`, `.eval()`, and
`torch.optim.Optimizer`-style updates carry over.

“PyTorch-shaped” does not mean drop-in compatibility with every operator. The main
change is algebraic: tropical layers select over sums instead of forming dense weighted
sums. A second, optional concern is **closure**. Only an explicitly configured and
audited recipe can be described as multiplier-free; TikiTorch as a whole cannot.

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
| `nn.Dropout` | `nn.TropicalDropout` | Omits survivor rescaling, but the training-time mask still belongs in a full-step audit |
| `nn.Embedding` | `nn.Embedding` (reused) | gather-only, already multiply-free |
| `nn.MultiheadAttention` | `nn.TropicalAttention` | L1/L∞/Hilbert scores + argmax-gather routing |
| `nn.CrossEntropyLoss` | `nn.TropicalCrossEntropyLoss(softmax="exact"|"lut")` | `"exact"` is host-conditioned; `"lut"` is the closure-oriented experimental path |
| `F.cross_entropy` | `F.tropical_cross_entropy` | functional form of the above |
| `torch.matmul` | `F.tropical_matmul` | `C_ij = min_k (A_ik + B_kj)` |
| `optim.SGD` | `optim.SignSGD` | sign-of-gradient; lr can be a power-of-2 shift |
| `optim.Adam` | `optim.TropicalAdam` / `optim.PureTropicalSGD` | Different update rules, not numerically equivalent replacements; closure depends on options |

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

=== "TikiTorch-style"

    ```python
    import torch, tikitorch as tt, tikitorch.optim as optim

    model = tt.models.TropicalCNN(in_channels=1, num_classes=10, channels=(16, 32),
                                  head_type="linear", use_soft=True, tau=1.0)
    opt = optim.SignSGD(model.parameters(), lr=2e-3, momentum=0.9)
    crit = tt.nn.TropicalCrossEntropyLoss()     # exact host-conditioned path

    total = EPOCHS * len(loader); step = 0
    for xb, yb in loader:                        # xb: (B, 1, 28, 28)
        model.set_tau(max(1.0 * (1 - step / (0.8 * total)), 0.0))   # soft -> hard
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step(); step += 1
    model.set_tau(0.0)                           # evaluate the corresponding hard routes
    ```

    This example shows the familiar loop, not a closed training recipe: it has a soft
    path during training, a dense head, exact host-conditioned loss, and floating-point
    momentum. Change and audit those ingredients deliberately if closure is the goal.

Three new ideas, all covered in the guide:

1. **The soft→hard curriculum.** The hard `min`/`max` has sparse selected gradients.
   One available recipe trains with a soft relaxation at `tau > 0` and anneals toward
   `tau = 0`; the same parameters are evaluated through different routing operators. See
   [Foundations](foundations.md#the-softhard-curriculum).
2. **Audit level matters.** `assert_closed` is a structural deny-list. Runtime and
   full-step checks inspect executed operations, while synthesis reports cover the
   physical build. See [Auditing closure](closure.md).
3. **Optimizer names are not certificates.** `SignSGD` has a simple sign-only base
   path, but options such as floating-point momentum change the boundary. Audit the
   concrete configuration. See [Choosing a model](choosing-a-model.md).

## What intentionally has no equivalent

Closure removes a few PyTorch tools by construction, not by omission:

- **BatchNorm / variance normalization** — needs a divide. Use `TropicalLayerNorm`
  (mean-subtract + power-of-2 rescale) or `TropicalBatchShift`.
- **Smooth dense activations** (sigmoid, tanh, GELU, SiLU) — not expressible as
  compare/add/shift. The tropical nonlinearity *is* the `min`/`max` routing itself.
- **A numerically equivalent swap.** Tropical operators and constrained optimizers
  change the hypothesis class and update rule. Establish an ordinary baseline and use
  controlled ablations before drawing conclusions. Read
  [Why tropical for ML/DL](why-tropical-ml.md) for the evidence boundary.
