# `tikitorch.autodiff`

Reference operators, packed route tapes, and exact selected-VJP replay for the
explicitly supported hard-routed graph.

The certificate records deterministic route choices plus operator-specific auxiliary
fields. Replay also requires the declared output cotangent or loss boundary. This API
does not define a new gradient and does not cover arbitrary PyTorch operators.

`DeviceRouteTape` keeps per-field packed byte tensors on the execution device;
`PackedRouteTape` is the portable bit-contiguous serialized format. Device fields
have at most seven padding bits each, and `nbytes` counts those bytes honestly.
Both preserve configuration fingerprints across clones and serialization.

`morphssm_forward_tape(model, ids)` streams a hard MorphSSM forward into a tape.
`morphssm_vjp(tape, model, output_cotangent)` is loss-independent.
`route_logits(model, ids)` integrates the same recorded backward with PyTorch
autograd. See the [route-training guide](../guide/route-training.md) for supported
configurations, first-order derivative limits, and benchmarking.

::: tikitorch.autodiff
