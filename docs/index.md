# TikiTorch

TikiTorch is **PyTorch-shaped infrastructure for tropical and hard-routed deep
learning**. It packages max-plus and min-plus operators, morphological layers,
model families, optimizers, recipe-scoped closure audits, exact selected-VJP replay,
and FPGA research paths behind a familiar Python interface.

```bash
python -m pip install -e ".[dev]"
```

```python
import torch
import tikitorch as tt

model = tt.models.MorphSSM(
    d_model=64,
    n_layers=4,
    task_type="classification",
    input_dim=1,
    num_classes=10,
)
optimizer = tt.optim.PureTropicalSGD(model.parameters())

x = torch.randn(8, 784, 1)
y = torch.randint(0, 10, (8,))
loss = model.compute_loss(x, y)
loss.backward()
optimizer.step()
```

This is an ordinary training-loop example. A multiplier-free claim requires a
closure-faithful loss and model configuration plus an audit of the full executed
step. Tropical operators alone do not certify every loss, backward rule, optimizer,
host boundary, or synthesized build.

## Start here

- **[Foundations](guide/foundations.md)** introduces max-plus and min-plus algebra.
- **[Migrating from PyTorch](guide/migration.md)** maps familiar training patterns to
  TikiTorch and explains closure boundaries.
- **[Choosing a model](guide/choosing-a-model.md)** identifies ordinary, dyadic, and
  tropical components in each family.
- **[Architecture](guide/architecture.md)** diagrams the main model families.
- **[Closure audit](api/audit.md)** documents the structural, runtime, and full-step
  checks and their limitations.
- **[Route-certificate autodiff](api/autodiff.md)** documents exact replay for the
  explicitly supported hard operators.
- **[FAQ](guide/faq.md)** covers common numerical and hardware questions.

The API reference is generated from source docstrings. Research claims and submission
gates are governed separately by the repository's
[claim ledger](https://github.com/nyuolab/tikitorch/blob/main/docs/research/CLAIM_LEDGER.md).
