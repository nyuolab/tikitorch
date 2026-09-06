# `tikitorch.audit`

Structural inspection, executed forward-path tracing, and full training-step tracing
answer different questions. Read [Auditing multiplier-free closure](../guide/closure.md)
before using these functions as evidence. No software audit proves the resource mapping
of a compiled hardware build.

Executed-path checks conservatively classify fused PyTorch losses, normalizations,
activations, reductions, attention kernels, and dense or elementwise arithmetic. A
runtime tensor is not treated as a shift constant merely because its current values
happen to be powers of two; shift exemptions require traced static provenance.

Audits execute the supplied model and loss. TikiTorch restores module mode and simple
scalar configuration attributes, parameters and gradients, buffers, optimizer state,
and PyTorch RNG state afterward, including when execution raises. It cannot undo
arbitrary external side effects performed by user code, such as file or network writes,
so audit only trusted callables.

::: tikitorch.audit
