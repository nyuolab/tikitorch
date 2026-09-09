# TikiTorch

TikiTorch is an independent companion for tropical and hard-routed research in the
PyTorch ecosystem. It provides familiar `nn`, `optim`, `functional`, and `models`
namespaces while reusing PyTorch tensors, autograd, modules, devices, and training-loop
conventions. “PyTorch-shaped” describes that interoperability, not affiliation or
endorsement by the PyTorch project.

```python
import torch
import tikitorch as tt

layer = tt.nn.TropicalLinear(4, 3)
x = torch.randn(8, 4)
y = layer(x)

print(y.shape)  # torch.Size([8, 3])
```

The [quickstart](get-started/quickstart.md) continues from that familiar layer API to
TikiTorch's distinctive workflow: audit a concrete hard path, pack its route choices,
and replay a selected VJP from the route record rather than retained activations. The
same copy/paste workflow is runnable as `python -m examples.closure_workflow`.

The distinguishing feature is the combination rather than any single layer:

| PyTorch provides | TikiTorch adds |
| --- | --- |
| General tensors, autograd, modules, optimizers, and CPU/GPU execution | Hard and relaxed max-plus, min-plus, morphological, dyadic, and routed models |
| A mature general-purpose training workflow | Structural, executed-forward, and full-training-step closure audits for a concrete recipe |
| Ordinary saved tensors and autograd graphs | Packed forward-route records and selected-VJP replay for supported hard operators |
| Extensible software and compiler foundations | Experimental local-learning APIs and repository HLS/F2 paths with build-specific evidence |

This lets a study connect representation, optimization, route semantics, and deployment
boundaries while continuing to use PyTorch as the underlying tensor and autograd system.

TikiTorch is currently research software at version 0.2.0. It does not promise complete
PyTorch API compatibility, optimized eager-GPU performance, or multiplier-free
execution for every configuration.

## Choose a path

| Goal | Start here |
| --- | --- |
| Install and run a first model | [Installation](get-started/installation.md) → [Quickstart](get-started/quickstart.md) |
| Port a familiar PyTorch loop | [Migrating from PyTorch](guide/migration.md) |
| Learn the algebra | [Tropical algebra](guide/foundations.md) |
| Evaluate the opportunities and trade-offs | [Why tropical for ML/DL?](guide/why-tropical-ml.md) |
| Select a model, loss, and optimizer | [Choosing a recipe](guide/choosing-a-model.md) |
| Check a multiplier-free claim | [Auditing closure](guide/closure.md) |
| Look up a class or function | [API reference](api/index.md) |
| Reproduce an experiment | [Research scope and evidence](research/index.md) |
| Join or review the project | [Shared collaboration handoff](https://github.com/nyuolab/tikitorch/blob/main/manuscript/README.md) |

## What is in the library?

- Max-plus and min-plus tensor operations and neural-network layers.
- Morphological sequence, vision, attention, and mixture-of-experts models.
- Sign-, shift-, and counter-oriented optimizers alongside PyTorch interoperability.
- Structural, executed-path, and full-training-step closure audits.
- Forward-Forward and predictive-coding experiments with local objectives and updates.
- Serialization, profiling, backend dispatch, and opt-in FPGA paths.
- Operator-scoped route-certificate replay for supported hard-routed graphs.

The route-certificate and hardware paths are evidence-bounded research capabilities.
Exact replay is conditional on the registered operators and deterministic tie rules;
software closure reports cover only the configured and executed recipe; and current F2
results do not establish a general speed, accuracy, or energy advantage over PyTorch.

The [quickstart](get-started/quickstart.md) demonstrates the normal library workflow.
The [research section](research/index.md) separately tracks measured results, draft
claims, limitations, paper workspaces, and reproduction artifacts.
