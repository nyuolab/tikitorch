# TikiTorch

TikiTorch is a PyTorch-shaped library for experiments with tropical and hard-routed
deep learning. It provides familiar `nn`, `optim`, `functional`, and `models`
namespaces while reusing PyTorch tensors, autograd, modules, devices, and training-loop
conventions.

```python
import torch
import tikitorch as tt

layer = tt.nn.TropicalLinear(4, 3)
x = torch.randn(8, 4)
y = layer(x)

print(y.shape)  # torch.Size([8, 3])
```

TikiTorch is currently research software at version 0.2. It does not promise complete
PyTorch API compatibility, optimized eager-GPU performance, or multiplier-free
execution for every configuration.

## Choose a path

| Goal | Start here |
| --- | --- |
| Install and run a first model | [Installation](get-started/installation.md) → [Quickstart](get-started/quickstart.md) |
| Port a familiar PyTorch loop | [Migrating from PyTorch](guide/migration.md) |
| Learn the algebra | [Tropical algebra](guide/foundations.md) |
| Select a model, loss, and optimizer | [Choosing a recipe](guide/choosing-a-model.md) |
| Check a multiplier-free claim | [Auditing closure](guide/closure.md) |
| Look up a class or function | [API reference](api/index.md) |
| Reproduce an experiment | [Research scope and evidence](research/index.md) |

## What is in the library?

- Max-plus and min-plus tensor operations and neural-network layers.
- Morphological sequence, vision, attention, and mixture-of-experts models.
- Sign-, shift-, and counter-oriented optimizers alongside PyTorch interoperability.
- Structural, executed-path, and full-training-step closure audits.
- Serialization, profiling, backend dispatch, and opt-in FPGA paths.
- Operator-scoped route-certificate replay for supported hard-routed graphs.

The [quickstart](get-started/quickstart.md) demonstrates the normal library workflow.
The [research section](research/index.md) separately tracks measured results, draft
claims, limitations, paper workspaces, and reproduction artifacts.
