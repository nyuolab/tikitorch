# `tikitorch.backend`

Logical device selection and backend dispatch. Backend availability and fallback are
runtime-specific; inspect the selected backend when validating deployment behavior.

Public tropical linear, dilation, soft-linear, and top-k modules consult the operation
registry. Resolution order is an explicit module-tree binding made with `tt.to`, then a
non-CPU `backend.default_device(...)`, then the input tensor's native PyTorch device.
An inferred CUDA/MPS tensor without an extension kernel uses the generic PyTorch
reference silently and stays on that device. An explicitly selected unavailable logical
backend emits `BackendFallbackWarning` in eager execution; select the reference device
before `torch.compile` if such a fallback is intended.

```python
with tt.backend.default_device("my_accelerator"):
    output = tt.nn.TropicalLinear(32, 64)(x)

layer = tt.to(tt.nn.TropicalLinear(32, 64), "my_accelerator")
output = layer(x)
```

A backend that returns operations must declare
`BackendCapabilities(op_dispatch=True)`. Re-registering a backend name atomically
replaces its exact operation set, so removed operations cannot remain stale. Route-
returning methods additionally require either the canonical reference implementation or
an explicit paired value/route contract; a value-only custom dilation is rejected rather
than silently recording a route from different math.

::: tikitorch.backend
