# `tikitorch.audit`

Structural inspection, executed forward-path tracing, and full training-step tracing
answer different questions. Read [Auditing multiplier-free closure](../guide/closure.md)
before using these functions as evidence. No software audit proves the resource mapping
of a compiled hardware build.

Executed-path checks conservatively classify fused PyTorch losses, normalizations,
activations, reductions, attention kernels, and dense or elementwise arithmetic. A
runtime tensor is not treated as a shift constant merely because its current values
happen to be powers of two; shift exemptions require traced static provenance.
The operator registry is explicit and fail-closed: a dispatched operation that is not
registered safe or unsafe is reported as `unknown-operation`, yields an
`inconclusive` status, and prevents the compatibility `*_closed` booleans from becoming
true.

Registry decisions are ATen-namespace and overload specific. Optional scales and
reduction modes are inspected rather than being admitted by basename alone, and an
opaque custom operator never inherits the verdict of a same-named ATen operator.
Uniform random-bit generation is treated as a declared primitive (not a hardware RNG
certificate); Gaussian generation and non-dyadic range mapping are rejected as
arithmetic boundaries.

`closure_report` currently emits schema `tikitorch.closure-report.v2` under audit
semantics `strict-eager-v3`. Its per-boundary `coverage` entries record mode,
`harden_tau`, tensor dtypes/devices, observed call counts, and safe/unsafe/unknown
operator names. These fields should be archived with any claim-bearing software audit;
different audit-semantics versions are not interchangeable evidence.

Audits execute the supplied model and loss. TikiTorch restores module mode and simple
scalar configuration attributes, parameters and gradients, buffers, optimizer state,
and PyTorch RNG state afterward, including when execution raises. It cannot undo
arbitrary external side effects performed by user code, such as file or network writes,
so audit only trusted callables.

::: tikitorch.audit
