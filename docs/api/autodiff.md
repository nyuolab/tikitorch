# `tikitorch.autodiff`

Reference operators, packed route tapes, and exact selected-VJP replay for the
explicitly supported hard-routed graph.

The certificate records deterministic route choices plus operator-specific auxiliary
fields. Replay also requires the declared output cotangent or loss boundary. This API
does not define a new gradient and does not cover arbitrary PyTorch operators.

::: tikitorch.autodiff
