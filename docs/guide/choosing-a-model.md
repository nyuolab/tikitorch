# Choosing a model and recipe

A TikiTorch experiment is more than a model class. Choose the model, projections,
normalization, routing temperature, loss, reverse rule, optimizer, and deployment
boundary together.

## Start with the task

| Task | Starting model | Main reason to try it |
| --- | --- | --- |
| Sequence classification, streaming signals, or character modeling | `MorphSSM` | Causal morphological convolution and a streaming `step()` interface |
| Content-routing sequence tasks | `TropicalTransformer` | Distance scoring followed by hard or softened routing |
| Sparse expert experiments | `MoTE` / `MoTEModel` | Top-k expert routing |
| Image classification | `TropicalCNN` | Learned morphological dilation and erosion blocks |
| Small vector or tabular baseline | `TropicalMLP` / `MorphMLP` | Compact alternating min/max layers |
| Local, forward-only learning | `TropicalForwardForward` / `DyadicForwardForward` | A distinct training API without global `loss.backward()` |
| Dynamic programming or planning | `algorithmic` / `planning` | Operations that directly express min-plus or max-plus recurrences |

Check the [`models` API](../api/models.md) for the task types and constructor arguments
actually supported by each class. Not every model accepts the same options.

## Decide what boundary you are studying

Use ordinary PyTorch components when the scientific question is representation or
learning behavior and multiplier-free closure is not the dependent variable. Use a
closed-oriented recipe only when that systems constraint matters, and record every
exception.

Common recipe switches include:

| Switch | Ordinary or soft path | Closure-oriented path |
| --- | --- | --- |
| Input/output projection | `input_type="linear"`, `head_type="linear"` | tropical projection/head where supported |
| Routing | `tau > 0` soft relaxation | `tau == 0` hard selection |
| Normalization | standard PyTorch normalization | a supported tropical/dyadic normalization with compatible dimensions |
| Dropout | training-time dropout | `dropout=0.0` for the audited step |
| Loss | ordinary task loss | an explicitly audited loss path |
| Optimizer | any PyTorch optimizer | a concrete sign/dyadic configuration that passes the training-step audit |

A dense head can be a useful declared boundary. Report it as such and do not describe
the resulting whole model as closed.

## Choose an optimizer

All TikiTorch optimizers implement the `torch.optim.Optimizer` protocol, but closure
depends on options and the executed path.

| Optimizer | When to consider it | Important boundary |
| --- | --- | --- |
| `SignSGD` / `TropicalSGD` | Simple sign updates | Optional floating-point momentum and weight decay are not the strict path |
| `PureTropicalSGD` | Integer-counter momentum and power-of-two step experiments | Keep `adaptive=False`; audit noise and decay options used by the recipe |
| `TropicalLion` | Sign-of-momentum experiments | Verify the configured integer/dyadic path with `assert_training_closed` |
| `TropicalAdam` | Experimental adaptive update | Do not infer closure from the class name; audit the chosen settings |
| `FrexpMagnitudeSGD`, `ExponentEMAAdam`, `CountAgreementSGD`, `PerLayerStepSGD` | Magnitude-aware or layerwise experiments | Specialized research optimizers; compare against a simple baseline and audit separately |

For a new problem, begin with an ordinary optimizer to establish learnability, then add
the constrained optimizer as a controlled change. This separates a model failure from
an update-rule failure.

## Choose a loss

| Loss | Role |
| --- | --- |
| `torch.nn.CrossEntropyLoss` or a model's default `compute_loss` | Ordinary learnability baseline; do not assume closure |
| `nn.TropicalCrossEntropyLoss(softmax="exact")` | Host-conditioned gradient path; the exact softmax is an explicit ordinary boundary |
| `nn.TropicalCrossEntropyLoss(softmax="lut")` | LUT-conditioned experimental path; validate the full executed step and compiled implementation |
| `nn.TropicalPerceptronLoss` | Simple routed-gradient baseline used by some hardware experiments |

Loss-value computation, loss gradient, averaging, and placement can have different
boundaries. State which object the compiled kernel implements.

## Use soft routing deliberately

Several models expose `set_tau`:

```python
for step, (x, target) in enumerate(loader):
    progress = step / max(1, len(loader) - 1)
    model.set_tau(max(1.0 - progress, 0.0))

    optimizer.zero_grad()
    loss = model.compute_loss(x, target)
    loss.backward()
    optimizer.step()

model.set_tau(0.0)
```

This is a template, not a universal schedule. Soft routing can make optimization
easier, while hard routing defines the selected-path graph used by a closure-oriented
deployment. Evaluate both and record when hardening occurs.

## Verify the exact recipe

```python
import tikitorch as tt

report = tt.closure_report(
    model,
    example_inputs=x,
    targets=target,
    loss_fn=loss_fn,
    optimizer=optimizer,
)
```

Follow the [closure guide](closure.md) before making a multiplier-free claim. A passing
software report still needs build-specific synthesis and routed evidence for a physical
hardware claim.
