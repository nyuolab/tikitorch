# Quickstart

This page shows the ordinary TikiTorch workflow: construct a PyTorch-compatible model,
run autograd, update parameters, and then inspect the executed recipe separately.

## Train one step

```python
import torch
import tikitorch as tt

model = tt.models.MorphSSM(
    d_model=64,
    n_layers=4,
    dropout=0.0,
    task_type="classification",
    input_dim=1,
    num_classes=10,
)
optimizer = tt.optim.PureTropicalSGD(model.parameters())

x = torch.rand(8, 784, 1)  # default quantized input embedding expects [0, 1]
target = torch.randint(0, 10, (8,))

optimizer.zero_grad()
loss = model.compute_loss(x, target)
loss.backward()
optimizer.step()

print(loss.detach().item(), model.param_count)
```

`MorphSSM` is a `torch.nn.Module`; standard calls such as `.parameters()`, `.to(...)`,
`.train()`, `.eval()`, and `.state_dict()` work normally. TikiTorch optimizers follow
the `torch.optim.Optimizer` protocol.

The default input embedding bins values over `[0, 1]`, clips outside values,
and does not differentiate through bin selection. Normalize inputs deliberately
or choose an explicit linear input embedding. Multilabel tasks need an explicit
loss such as `BCEWithLogitsLoss`, not this single-label `compute_loss` default.

This sample demonstrates interoperability, not physical multiplier-free closure.
`compute_loss` uses the model's task loss, and a default model can contain soft-routing
or other recipe-dependent boundaries.

## Audit and replay one hard route

The smallest end-to-end demonstration of TikiTorch's unusual tooling is a hard
min-plus operator whose executed path can be audited and whose winner choices can be
packed for selected-VJP replay:

```python
layer = tt.nn.TropicalLinear(4, 3, use_softmin=False, tau=0.0)
x = torch.tensor([[1.0, -2.0, 0.5, 3.0]])

report = tt.closure_report(layer, x)
assert report["inference_closed"]
output, route = tt.autodiff.min_plus(x, layer.weight, layer.bias)

tape = tt.autodiff.PackedRouteTape()
tape.append("winner", route, cardinality=x.shape[-1])
replayed_dx, _, _ = tt.autodiff.min_plus_vjp(
    torch.ones_like(output), tape.read("winner"), x.shape[-1]
)

print(report["inference_closed"], tape.nbytes, replayed_dx)
```

The complete deterministic example is
[`examples/closure_workflow.py`](https://github.com/nyuolab/tikitorch/blob/main/examples/closure_workflow.py).
This is an operator-scoped certificate for the supplied output cotangent, not a claim
about arbitrary PyTorch graphs or physical hardware.

## Inspect closure at the right level

Closure belongs to an executed recipe, not to the package name. Begin with a report:

```python
report = tt.closure_report(model, example_inputs=x)

print(report["structural"])
print(report["inference"])
print(report["inference_closed"])
```

The report temporarily hardens models exposing `set_tau` by default and inspects the
representative input path supplied here. Model/optimizer state and PyTorch RNG state
are restored after auditing. It does not inspect unexecuted branches, undo external
side effects in user code, or prove anything about a synthesized build.

For a full training-step audit, pass an explicit loss function and optimizer:

```python
loss_fn = tt.nn.TropicalPerceptronLoss()
training_report = tt.closure_report(
    model,
    example_inputs=x,
    targets=target,
    loss_fn=loss_fn,
    optimizer=optimizer,
)

print(training_report["training"])
print(training_report["training_closed"])
```

Treat the returned violations as diagnostics. Do not change a scientific claim merely
to make a checker pass: first define the intended model, loss, reverse rule, optimizer,
input path, and deployment boundary. The [closure guide](../guide/closure.md) explains
the three software audit levels and the separate synthesis boundary.

## Save and restore

PyTorch state dictionaries remain available directly:

```python
torch.save(model.state_dict(), "morphssm.pt")
model.load_state_dict(torch.load("morphssm.pt", weights_only=True))
```

TikiTorch also provides [`tt.save` and `tt.load`](../api/serialization.md) for its own
model bundle format. Bundles are decoded with `weights_only=True`; built-in
`tikitorch.models` classes can reconstruct automatically. For a custom class, pass an
already constructed `model=` (recommended). The opt-in `trusted=True` custom-class
reconstruction can execute checkpoint-selected imports and constructors and must never
be used with an untrusted artifact.

## Continue learning

- Follow a [notebook tutorial](../guide/notebooks.md).
- Browse the [runnable examples](../tutorials/examples.md).
- Use the [model and recipe guide](../guide/choosing-a-model.md).
- Look up exact signatures in the [API reference](../api/index.md).
