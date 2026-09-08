# Auditing multiplier-free closure

Tropical operators are built from additions and selections, but that fact alone does
not certify a model, training step, or hardware image. Closure is a property of a
concrete recipe and boundary:

```text
model + loss + reverse rule + optimizer + executed inputs + compiled build
```

TikiTorch provides three software checks. HLS and routed implementation reports are a
separate fourth boundary.

Keep diagnostics inside an audited region tensor-valued. Forward-Forward trainers
accept `host_metrics=False` at construction or `train_step`; `TropicalPCNet.infer`
and `train_step` accept the same option. Default host-float metrics remain available
for compatibility, but scalar extraction is an explicit unknown boundary under
the strict auditor. PC tensor diagnostics expose raw winner-change counts and
total counts; compute a reporting fraction outside the audited update.

Likewise, the coin-change planning helper uses amount-dependent Python loops.
Its scalar extraction makes runtime certification inconclusive even though its
algorithmic recurrence is min-plus. Do not suppress this finding to obtain a
closure boolean.

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

Certification is allow-list based. Every dispatched operation has one of three
outcomes:

- **safe** — the versioned registry explicitly recognizes a data-movement,
  compare/select, additive-reduction, integer/bit, or provenance-checked dyadic-shift
  operation;
- **unsafe** — recognized multiply/divide, transcendental, normalization, loss,
  linalg, attention, or other arithmetic; or
- **unknown** — the operation is not yet classified, so the result is
  `inconclusive` and fails closed.

`find_runtime_multiplies` retains its historical name and `(category, op_name)`
return shape. It now includes `("unknown-operation", op_name)` blockers. An empty
list means that every operation observed on that eager path was explicitly classified
safe; it is not a claim about unexecuted branches or compiler-generated code.

Registration is namespace- and schema-aware. A custom operator called `sum`, for
example, remains unknown even though `aten::sum` is registered safe. Overloads with
arithmetic options are checked at the call site: non-dyadic `index_add` scaling,
multiplicative scatter reduction, and frequency-scaled embedding gradients are
blockers rather than inheriting the verdict of their additive forms.

Uniform random-bit generation is an explicit software-audit primitive; mapping it
through a dyadic range remains shift/add arithmetic. This does not certify a physical
random-number generator. Gaussian generation (`randn` and `normal`) and non-dyadic
uniform range maps are arithmetic boundaries and are rejected.

By default, compatible models are hardened through `set_tau(0.0)` during this check.
That is useful for inspecting a deploy graph, but it is not evidence that the soft
training graph is closed. Hardening and evaluation mode are temporary: model state and
PyTorch RNG state are restored when the audit ends.

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
optimizer paths can introduce operations outside the closed subset. Parameters,
gradients, buffers, optimizer state, module mode, and PyTorch RNG state are restored
after the check. External side effects in user callables cannot be rolled back.
`find_training_multiplies` likewise includes unknown-operation blockers rather than
silently treating a new fused or special operator as safe.

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

The report is versioned (`tikitorch.closure-report.v2`, audit semantics
`strict-eager-v3`). For each executed boundary, inspect both the tri-state status and
coverage metadata:

```python
assert report["inference_status"] in {"closed", "violated", "inconclusive"}
print(report["inference_unknown"])
print(report["coverage"]["inference"])
```

Coverage records the eager mode, whether `harden_tau` was requested and applied,
observed tensor dtypes/devices, operator-call counts, and the safe, unsafe, and unknown
operator names. The compatibility booleans `inference_closed`, `training_closed`, and
`training_zero_dsp` are `True` only for status `closed`; both a known violation and an
unknown operation make them `False`. `training_zero_dsp` is a legacy name for this
software result and does not report synthesized DSP utilization.

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
