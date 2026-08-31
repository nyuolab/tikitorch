# API reference

The reference pages below are generated from the package's source docstrings. Use the
[user guide](../guide/foundations.md) for concepts and the reference for exact
signatures, arguments, return values, and exported names.

TikiTorch is version 0.2 research software. Public names are organized in a familiar
PyTorch shape, but the project does not yet promise a formal API-stability window or
complete PyTorch compatibility.

## Core modeling and training

| Namespace | Purpose |
| --- | --- |
| [`tikitorch.functional`](functional.md) | Stateless max-plus, min-plus, routing, loss, and autograd operations |
| [`tikitorch.nn`](nn.md) | Layers and loss modules |
| [`tikitorch.models`](models.md) | Composed model families |
| [`tikitorch.optim`](optim.md) | Sign-, shift-, and counter-oriented optimizers |
| [`tikitorch.train`](train.md) | Soft-to-hard training convenience API |

## Runtime and verification

| Namespace | Purpose |
| --- | --- |
| [`tikitorch.audit`](audit.md) | Structural and executed-path closure diagnostics |
| [`tikitorch.backend`](backend.md) | Logical devices and backend dispatch |
| [`tikitorch.serialization`](serialization.md) | TikiTorch model bundles |
| [`tikitorch.profile`](profile.md) | Recipe-oriented operation and hardware-cost summaries |

## Specialized APIs

| Namespace | Purpose |
| --- | --- |
| [`tikitorch.autodiff`](autodiff.md) | Selected-VJP certificates for a declared hard-operator set |
| [`tikitorch.fpga`](fpga.md) | Optional FPGA integration and kernel runners |
| [`tikitorch.forward_forward`](forward_forward.md) | Layer-local learning experiments |
| [`tikitorch.planning`](planning.md) | Tropical dynamic-programming and planning primitives |

## Mathematical utilities

The [`classical`](classical.md), [`geometry`](geometry.md), [`prune`](prune.md), and
[`algorithmic`](algorithmic.md) modules contain specialized tropical-learning and
analysis tools.

## Top-level conveniences

`import tikitorch as tt` exposes common namespaces and selected convenience names such
as `tt.MorphSSM`, `tt.PureTropicalSGD`, `tt.assert_closed`, `tt.save`, and `tt.load`.
Prefer namespace-qualified names in reusable code when ownership matters, for example
`tt.models.MorphSSM` or `tt.optim.PureTropicalSGD`.

The internal `tikitorch.research` package and repository scripts are not presented as
stable public APIs. Research reproduction belongs in the
[research section](../research/index.md).
