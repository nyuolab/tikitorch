# TikiTorch

[![CI](https://github.com/nyuolab/tikitorch/actions/workflows/ci.yml/badge.svg)](https://github.com/nyuolab/tikitorch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20--%203.12-blue.svg)](https://github.com/nyuolab/tikitorch/blob/main/pyproject.toml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/nyuolab/tikitorch/blob/main/LICENSE)

**An independent companion for tropical and hard-routed research in the PyTorch
ecosystem.**

TikiTorch keeps the tensors, autograd, modules, devices, and training-loop conventions
that make PyTorch useful, then adds an unusual combination: hard and relaxed tropical
models, three-level full-recipe closure auditing, replayable forward-route certificates,
experimental local-learning APIs, and build-specific HLS/AWS F2 paths. “PyTorch-shaped”
describes that interoperability and familiarity; TikiTorch is not affiliated with or
endorsed by the PyTorch project.

| PyTorch provides | TikiTorch adds |
|---|---|
| General tensors, autograd, `nn.Module`, optimizers, and CPU/GPU execution | Max-plus, min-plus, morphological, dyadic, and hard-routing operators and models |
| A mature general-purpose training ecosystem | Soft-to-hard experiments with structural, executed-forward, and full-training-step closure audits |
| Ordinary saved tensors and autograd graphs | Packed route records and exact selected-VJP replay for a declared hard-operator set |
| Extensible software and compiler foundations | Experimental local learning plus repository HLS kernels, host bindings, and configuration-specific F2 evidence |

The result is a place to study a representation, its learning rule, and its deployment
boundary together without presenting TikiTorch as a replacement for PyTorch.

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
git clone https://github.com/nyuolab/tikitorch.git
cd tikitorch
python -m pip install -e ".[dev]"
```

TikiTorch requires Python 3.9 or newer, PyTorch 2.0 or newer, and NumPy. The project is
licensed under GPL-3.0-or-later. Version `0.2.0` is the current source version, not yet a
published PyPI release.

## Library map

| Namespace | Role |
|---|---|
| `tikitorch.functional` | Stateless max-plus, min-plus, morphological, and routing operators |
| `tikitorch.nn` | PyTorch-style tropical layers, blocks, and experimental predictive-coding components |
| `tikitorch.models` | MorphSSM, tropical transformer, MoTE, MLP, and Forward-Forward families |
| `tikitorch.optim` | Sign-, shift-, and counter-based optimizers |
| `tikitorch.audit` | Static, runtime, and full-step closure checks |
| `tikitorch.autodiff` | Packed forward-route certificates and exact selected VJP replay |
| `tikitorch.backend` | Logical devices and execution dispatch |
| `tikitorch.fpga` | Optional local host-reference bridge and configuration-specific AWS F2 bindings |
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
| **Experimental** | Usable and tested, but schemas or performance may change before the first stable release | `autodiff`, local-learning APIs, FPGA bindings, planning, profiling |
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

See [Migrating from PyTorch](https://github.com/nyuolab/tikitorch/blob/main/MIGRATION.md)
for the recipe boundary and
[the audit API](https://github.com/nyuolab/tikitorch/blob/main/docs/api/audit.md) for
supported checks and limitations.

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

The [model-selection guide](https://github.com/nyuolab/tikitorch/blob/main/docs/guide/choosing-a-model.md)
states which boundaries are tropical, dyadic, or ordinary PyTorch for each family.

### Local-learning experiments

`TropicalForwardForward`, `DyadicForwardForward`, `TropicalPCNet`, and `DyadicPCNet`
provide experimental alternatives to a conventional global `loss.backward()` workflow.
They make local objectives, routed errors, and constrained updates available for direct
comparison; they do not by themselves establish an accuracy, biological-plausibility,
or hardware advantage.

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

Draft measurements in three hard-routed MorphSSM configurations show portable
certificates 1.97–2.96× smaller than a checkpoint-retained-state comparator at a
common supplied-output-cotangent boundary. At the current loss-inclusive boundary,
which retains logits and targets on both sides, those ratios are 1.69–2.79×. This is
a retained-state representation result, not yet a process-peak-memory, traffic,
latency, or energy advantage.

The reference implementation, tests, and benchmark entry points are under
[`scripts/route_certificates`](https://github.com/nyuolab/tikitorch/tree/main/scripts/route_certificates),
with retained evidence under
[`results/route_certificates`](https://github.com/nyuolab/tikitorch/tree/main/results/route_certificates).

### FPGA path

The repository includes HLS kernels and host bindings for multiplier-free inference and
learning experiments on AWS F2 hardware. In one configuration-specific 30-update D512
run, the archived resident accuracy, weight checksum, and 16 sampled predictions match
the C-simulation record bit for bit; unlogged internal signals were not exhaustively
compared. A separate two-layer 16-by-16 MNIST run whose fabric-only updates reached
74.75% test accuracy was evaluated at 74.13% through an independent exported-weight
path. That 0.62-point difference remains unresolved because identical inputs and
preprocessing were not frozen. These establish narrow execution and feasibility
results, not competitive speed, energy efficiency, state-of-the-art accuracy, or broad
learning quality. Matched baselines, timing-closed headline builds, and instrumented
full-card power remain open gates tracked by the paper draft.

Start with [the FPGA build guide](https://github.com/nyuolab/tikitorch/blob/main/tikitorch/hardware/BUILD_GUIDE.md)
and
[`examples/fpga_quickstart.py`](https://github.com/nyuolab/tikitorch/blob/main/examples/fpga_quickstart.py).

## Documentation

- [Installation](https://github.com/nyuolab/tikitorch/blob/main/docs/get-started/installation.md)
- [Quickstart](https://github.com/nyuolab/tikitorch/blob/main/docs/get-started/quickstart.md)
- [Tutorials](https://github.com/nyuolab/tikitorch/blob/main/docs/tutorials/index.md)
- [User guide](https://github.com/nyuolab/tikitorch/blob/main/docs/guide/foundations.md)
- [Closure auditing](https://github.com/nyuolab/tikitorch/blob/main/docs/guide/closure.md)
- [API reference](https://github.com/nyuolab/tikitorch/blob/main/docs/api/index.md)
- [Research scope and evidence](https://github.com/nyuolab/tikitorch/blob/main/docs/research/index.md)
- [Runnable examples](https://github.com/nyuolab/tikitorch/blob/main/examples/README.md)
- [Notebooks](https://github.com/nyuolab/tikitorch/blob/main/notebooks/README.md)

Build the documentation locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build
```

## Research artifacts

Two focused manuscripts are maintained independently of the future ontogenic-agent
programme. The
[joint discussion handoff](https://github.com/nyuolab/tikitorch/blob/main/papers/XIANG_HANDOFF.md)
gives Xiang a single review entry point:

- [Forward traces as exact differentiation certificates in hard-routed networks](https://github.com/nyuolab/tikitorch/blob/main/papers/route_certificates/README.md)
  ([root build handoff](https://github.com/nyuolab/tikitorch/blob/main/manuscript/README.md))
- [Multiplier-free learning on FPGA](https://github.com/nyuolab/tikitorch/blob/main/papers/multiplier_free_fpga/README.md)

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

Project expectations and release gates are documented in
[CONTRIBUTING.md](https://github.com/nyuolab/tikitorch/blob/main/CONTRIBUTING.md),
[GOVERNANCE.md](https://github.com/nyuolab/tikitorch/blob/main/GOVERNANCE.md),
[CODE_OF_CONDUCT.md](https://github.com/nyuolab/tikitorch/blob/main/CODE_OF_CONDUCT.md),
[SECURITY.md](https://github.com/nyuolab/tikitorch/blob/main/SECURITY.md), and
[RELEASING.md](https://github.com/nyuolab/tikitorch/blob/main/RELEASING.md). The current tree is release-candidate preparation;
creating a public tag or publishing a package remains a separate maintainer decision.

## Citation and versioning

The current development version is `0.2.0`. See
[CITATION.cff](https://github.com/nyuolab/tikitorch/blob/main/CITATION.cff),
[CHANGELOG.md](https://github.com/nyuolab/tikitorch/blob/main/CHANGELOG.md), and the
paper-specific bibliographies for citation metadata. A release tag and archival DOI are
required before either manuscript is submitted.
