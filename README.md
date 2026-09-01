# TikiTorch

[![CI](https://github.com/nyuolab/tikitorch/actions/workflows/ci.yml/badge.svg)](https://github.com/nyuolab/tikitorch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**A PyTorch-shaped foundation for tropical deep learning.**

TikiTorch brings max-plus, min-plus, morphological, and hard-routed learning to a
familiar interface:

```python
import tikitorch as tt

model = tt.models.MorphSSM(
    d_model=64,
    n_layers=4,
    task_type="classification",
    input_dim=1,
    num_classes=10,
)
optimizer = tt.optim.PureTropicalSGD(model.parameters())
```

The project is organized as a reusable library first. It includes differentiable
tropical operators, `nn.Module` layers, model families, optimizers, closure audits,
exact hard-route replay, and an FPGA backend. Research artifacts and hardware kernels
live beside the library, but do not change the semantics of its public API.

## What tropical means here

Tropical algebra replaces ordinary addition and multiplication with operations such as
`max` and `+`, or `min` and `+`. This gives neural operators whose forward dataflow can
be expressed with comparisons, additions, routing, and data movement.

That does **not** imply that every TikiTorch program, loss, gradient, or host boundary is
automatically multiplier-free. Closure is a property of a concrete recipe. TikiTorch
therefore exposes static, runtime, and full-training-step audits so a hardware claim can
be tested rather than inferred from a model name.

## Install

```bash
git clone --branch pc-morphssm --single-branch https://github.com/nyuolab/tikitorch.git
cd tikitorch
python -m pip install -e ".[dev]"
```

TikiTorch requires Python 3.9 or newer, PyTorch 2.0 or newer, and NumPy. The project is
licensed under GPL-3.0-or-later. Version `0.2.0` is currently a development branch,
not a published PyPI release.

## Library map

| Namespace | Role |
|---|---|
| `tikitorch.functional` | Stateless max-plus, min-plus, morphological, and routing operators |
| `tikitorch.nn` | PyTorch-style tropical layers and blocks |
| `tikitorch.models` | MorphSSM, tropical transformer, MoTE, MLP, and Forward-Forward families |
| `tikitorch.optim` | Sign-, shift-, and counter-based optimizers |
| `tikitorch.audit` | Static, runtime, and full-step closure checks |
| `tikitorch.autodiff` | Packed forward-route certificates and exact selected VJP replay |
| `tikitorch.backend` | Logical devices and execution dispatch |
| `tikitorch.fpga` | Local and AWS F2 execution bindings |
| `tikitorch.hardware` | HLS kernels, testbenches, and build material |

The package is intentionally shaped like PyTorch: use `tt.nn`, `tt.optim`,
`tt.models`, `tt.save`, `tt.load`, `tt.device`, and `tt.to` where their documented
semantics apply.

## API maturity

TikiTorch uses maturity labels to distinguish reusable library surface from active
research:

| Level | Meaning | Current examples |
|---|---|---|
| **Public** | Import-tested API intended for ordinary use; changes require docs, tests, and a changelog entry | `functional`, core `nn`, `models`, `optim`, serialization, logical devices |
| **Experimental** | Usable and tested, but schemas or performance may change before the first stable release | `autodiff`, FPGA bindings, planning, profiling |
| **Research artifact** | Evidence, runners, and manuscripts; not a compatibility-guaranteed package API | `scripts/`, `results/`, `papers/`, `manuscript/` |

“Public” does not yet mean semantically frozen: the project has not cut its first
stable release. Research modules under `tikitorch.research` are explicitly unstable.

## A complete training step

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
optimizer.zero_grad()
```

This example uses the model's default classification loss. Treat it as a library
quickstart, not as a closure certificate. To make a multiplier-free training claim,
choose a closure-faithful loss and hardened model, then audit the entire step:

```python
report = tt.closure_report(model, x, y, loss_fn, optimizer)
tt.assert_training_closed(model, x, y, loss_fn, optimizer)
```

See [Migrating from PyTorch](MIGRATION.md) for the recipe boundary and
[the audit API](docs/api/audit.md) for supported checks and limitations.

## Core capabilities

### Tropical operators and layers

TikiTorch supplies hard and relaxed max-plus/min-plus linear maps, morphological
convolutions, hard attention and value routing, normalizations, losses, and classical
tropical algorithms. Hard operators use deterministic routing conventions documented
and tested in the package; relaxed operators remain useful during optimization but are
not automatically closure-clean.

### Model families

- `MorphSSM` uses causal morphological sequence mixing and supports token-wise
  streaming.
- `TropicalTransformer` exposes tropical distance and hard-routing attention variants.
- `MoTE` provides sparse routing over tropical experts.
- `TropicalMLP`, `MorphMLP`, and `TropicalSequenceModel` are compact baselines.
- `TropicalForwardForward` and `DyadicForwardForward` support layer-local,
  forward-only learning experiments.

The [model-selection guide](docs/guide/choosing-a-model.md) states which boundaries are
tropical, dyadic, or ordinary PyTorch for each family.

### Closure auditing

Three levels answer different questions:

1. `tt.assert_closed(model)` inspects declared module structure.
2. `tt.assert_closed_runtime(model, example_inputs)` observes executed operators.
3. `tt.assert_training_closed(...)` observes a concrete forward, loss, backward, and
   optimizer step.

Runtime auditing is empirical operator interception, not a formal proof about unexecuted
branches or external kernels. For hardware reporting, pair it with synthesis reports and
the exact workload configuration.

### Forward-route certificates

`tikitorch.autodiff` records the route choices and minimal local values needed to replay
selected vector-Jacobian products in supported hard-routed operators. The replay is exact
under the recorded tie rules and operator assumptions. It does not create a new gradient,
eliminate every model boundary, or certify arbitrary PyTorch programs.

The reference implementation, tests, and benchmark entry points are under
[`scripts/route_certificates`](scripts/route_certificates), with retained evidence under
[`results/route_certificates`](results/route_certificates).

### FPGA path

The repository includes HLS kernels and host bindings for multiplier-free inference and
learning experiments on AWS F2 hardware. The committed evidence includes bit-exact
silicon execution of fused training kernels and a small held-out MNIST learning result.
Final comparative claims still require matched baselines, timing-closed builds, and
instrumented power measurements; the paper draft tracks those gates explicitly.

Start with [the FPGA build guide](tikitorch/hardware/BUILD_GUIDE.md) and
[`examples/fpga_quickstart.py`](examples/fpga_quickstart.py).

## Documentation

- [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md)
- [Tutorials](docs/tutorials/index.md)
- [User guide](docs/guide/foundations.md)
- [Closure auditing](docs/guide/closure.md)
- [API reference](docs/api/index.md)
- [Research scope and evidence](docs/research/index.md)
- [Runnable examples](examples/README.md)
- [Notebooks](notebooks/README.md)

Build the documentation locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build
```

## Research artifacts

Two focused manuscripts are maintained independently of the future ontogenic-agent
programme:

- [Root route-certificate manuscript handoff](manuscript/README.md)
- [Multiplier-free learning on FPGA](papers/multiplier_free_fpga/README.md)
- [Forward traces as exact differentiation certificates in hard-routed networks](papers/route_certificates/README.md)

Their READMEs distinguish measured results from experiments still required for
submission. Historical exploratory results under `results/` are evidence records, not
blanket claims about the current library.

## Development

```bash
python -m ruff check .
pytest -q
```

Run learning-heavy tests explicitly with `pytest -m slow`. Hardware CSIM and silicon
workflows have additional toolchain requirements documented with their kernels.

Before opening a change, keep public imports PyTorch-shaped, add an exactness test for
new hard-routing semantics, and state any ordinary-arithmetic boundary in the API docs.

## Citation and versioning

The current development version is `0.2.0`. See [CITATION.cff](CITATION.cff),
[CHANGELOG.md](CHANGELOG.md), and the paper-specific bibliographies for citation
metadata. A release tag and archival DOI are required before either manuscript is
submitted.
