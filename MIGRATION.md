# Migrating from PyTorch to TikiTorch

TikiTorch mirrors the PyTorch namespace (`tt.nn`, `tt.optim`, `tt.functional`,
`tt.models`) and reuses `nn.Module`, so `.parameters()`, `.state_dict()`,
`.to(device)`, `.train()/.eval()`, `nn.Sequential` and `opt.step(); opt.zero_grad()`
all work unchanged. What changes is the **algebra** — every op is compare / add /
bit-shift, so the loop is multiplier-free — and the one new invariant is **closure**:
`tikitorch.audit.assert_closed(model)`.

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

Three new ideas: the **soft→hard `tau` curriculum** (train soft, anneal `tau → 0`,
deploy the hard op), **`assert_closed`** as the zero-multiply test (a `head_type=
"linear"` head is a *declared* boundary — `assert_closed(model, allow={"head"})`),
and **sign-based optimizers**. See the full guide for the details and a runnable
before/after loop.
