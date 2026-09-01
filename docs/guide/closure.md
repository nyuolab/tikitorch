# Auditing multiplier-free closure

Tropical operators are built from additions and selections, but that fact alone does
not certify a model, training step, or hardware image. Closure is a property of a
concrete recipe and boundary:

```text
model + loss + reverse rule + optimizer + executed inputs + compiled build
```

TikiTorch provides three software checks. HLS and routed implementation reports are a
separate fourth boundary.

## 1. Structural inspection

```python
from tikitorch.audit import assert_closed, find_closure_violations

violations = find_closure_violations(model)
assert_closed(model)
```

Structural inspection walks registered modules and rejects known multiply-bearing
PyTorch components. It is fast and useful in unit tests. It cannot see arithmetic
written directly inside a custom `forward` method.

Use `allow` only to declare a deliberate boundary, not to hide one:

```python
assert_closed(model, allow={"output_head"})
```

The resulting claim is “closed except for the declared output head,” not “fully
closed.” Module paths must match `model.named_modules()`.

## 2. Executed forward-path tracing

```python
from tikitorch.audit import assert_closed_runtime, find_runtime_multiplies

hits = find_runtime_multiplies(model, example_input)
assert_closed_runtime(model, example_input)
```

Runtime tracing observes PyTorch operations on the supplied input. It can catch raw
multiply, divide, and transcendental operations that structural inspection misses.
It covers only the branch, shapes, dtypes, device, and mode that actually execute.

By default, compatible models are hardened through `set_tau(0.0)` during this check.
That is useful for inspecting a deploy graph, but it is not evidence that the soft
training graph is closed.

## 3. Full training-step tracing

```python
from tikitorch.audit import assert_training_closed, find_training_multiplies

hits = find_training_multiplies(
    model,
    (example_input,),
    target,
    loss_fn,
    optimizer=optimizer,
)

assert_training_closed(
    model,
    (example_input,),
    target,
    loss_fn,
    optimizer=optimizer,
)
```

This check executes forward, loss, backward, and optionally `optimizer.step()`. It is
the appropriate software test for a learning-loop claim. Ordinary cross-entropy,
soft routing, dropout rescaling, dense heads, ordinary normalization, and adaptive
optimizer paths can introduce operations outside the closed subset.

[`closure_report`](../api/audit.md) gathers the available checks into one dictionary:

```python
report = tt.closure_report(
    model,
    example_inputs=example_input,
    targets=target,
    loss_fn=loss_fn,
    optimizer=optimizer,
)
```

## 4. Compiled implementation

A passing software audit does not prove a routed FPGA or ASIC implementation has no
multipliers. Conversely, a DSP block can be configured as an adder rather than a
multiplier. Physical claims must bind the exact source revision and configuration to:

- HLS operator and resource reports;
- final routed resource and timing reports;
- bit-exact or tolerance-scoped known-answer tests;
- device, toolchain, clocks, and build flags; and
- measured workload boundaries for latency, throughput, or power claims.

The research [claim ledger](../research/index.md) records which build-specific results
meet those requirements and which remain preliminary.

## A useful test pattern

For a component intended to support a closed recipe, test both structure and an
executed path:

```python
def test_hard_model_recipe_is_closed():
    model = make_model(dropout=0.0)
    model.set_tau(0.0)
    x, target = representative_batch()

    tt.assert_closed(model)
    tt.assert_closed_runtime(model, x)

    loss_fn = tt.nn.TropicalPerceptronLoss()
    optimizer = tt.optim.PureTropicalSGD(model.parameters())
    tt.assert_training_closed(model, (x,), target, loss_fn, optimizer)
```

Choose representative batches that exercise every branch relevant to the claim. Keep
ordinary correctness tests as well: closure does not imply numerical correctness,
learning quality, timing closure, or useful performance.
