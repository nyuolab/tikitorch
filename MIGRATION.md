# Migrating from PyTorch to TikiTorch

TikiTorch mirrors familiar PyTorch namespaces (`tt.nn`, `tt.optim`,
`tt.functional`, `tt.models`) and builds on `torch.nn.Module`, so common module,
state-dict, device, and optimizer-loop conventions carry over. It is not a
drop-in implementation of every PyTorch operator, and using a tropical layer
does not make an entire training recipe multiplier-free.

The additional concept is **recipe-scoped closure**. Structural, runtime, and
full-training-step audits answer different questions; none replaces synthesis
evidence for a physical hardware claim.

> **Full guide (with the before/after training loop and the soft→hard curriculum):**
> [docs/guide/migration.md](docs/guide/migration.md) — also rendered on the docs site
> under **Guide → Migrating from PyTorch**.

## Quick swap table

| PyTorch | TikiTorch |
|---|---|
| `nn.Linear` | `nn.TropicalLinear` (erosion) / `nn.TropicalDilation` (dilation) |
| `nn.Conv2d` | `nn.TropicalConv2d(mode="dilation"|"erosion")` |
| `nn.MaxPool2d` | `nn.TropicalMaxPool2d` / `nn.TropicalMinPool2d` |
| `nn.LayerNorm` | `nn.TropicalLayerNorm` (mean-subtract + power-of-2 rescale) |
| `nn.BatchNorm*` | — (use `nn.TropicalLayerNorm` / `nn.TropicalBatchShift`) |
| `nn.ReLU` | `nn.TropicalReLU` |
| `nn.LeakyReLU(0.01)` | `nn.TropicalLeakyReLU(shift=k)` (slope `2**-k`) |
| `nn.Sigmoid/Tanh/GELU` | — (incompatible with the semiring) |
| `nn.Dropout` | `nn.TropicalDropout` ({0,1} gate, no `1/(1-p)` rescale) |
| `nn.MultiheadAttention` | `nn.TropicalAttention` |
| `nn.CrossEntropyLoss` | `nn.TropicalCrossEntropyLoss(softmax="exact"|"lut")` |
| `F.cross_entropy` | `F.tropical_cross_entropy` |
| `torch.matmul` | `F.tropical_matmul` (min-plus) |
| `optim.SGD` | `optim.SignSGD` |
| `optim.Adam` | `optim.TropicalAdam` / `optim.PureTropicalSGD` |

Three recurring ideas are the **soft→hard `tau` curriculum**, explicit
arithmetic boundaries, and sign-/shift-oriented optimizer variants. Treat the
table as an API map, not as proof that every substituted component has the same
numerics, accuracy, gradient density, or hardware cost. See the full guide for
the runnable before/after loop and audit limitations.
