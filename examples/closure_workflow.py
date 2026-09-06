"""Tiny hard-route workflow: run, audit, pack, and replay.

Run from a source checkout with::

    python -m examples.closure_workflow

The certificate in this example covers one registered min-plus operator and one
declared output cotangent. It is not a certificate for arbitrary PyTorch code or
a physical hardware build.
"""

import torch

import tikitorch as tt


layer = tt.nn.TropicalLinear(4, 3, use_softmin=False, tau=0.0)
with torch.no_grad():
    layer.weight.copy_(
        torch.tensor(
            [
                [0.0, 2.0, 4.0, 6.0],
                [-2.0, 4.0, 0.0, 4.0],
                [2.0, 3.0, -3.0, 1.0],
            ]
        )
    )
    layer.bias.zero_()

x = torch.tensor([[1.0, -2.0, 0.5, 3.0]], requires_grad=True)
report = tt.closure_report(layer, x)

# Record the hard min-plus winners, then pack the three 4-way choices into one byte.
output, route = tt.autodiff.min_plus(x, layer.weight, layer.bias)
torch.testing.assert_close(output, layer(x))
tape = tt.autodiff.PackedRouteTape()
tape.append("winner", route, cardinality=x.shape[-1])

# Replay the selected VJP from the route and output cotangent, without activations.
output_cotangent = torch.tensor([[1.0, -2.0, 3.0]])
replayed_dx, _, _ = tt.autodiff.min_plus_vjp(
    output_cotangent,
    tape.read("winner"),
    input_features=x.shape[-1],
)
reference_dx, = torch.autograd.grad(output, x, output_cotangent)
torch.testing.assert_close(replayed_dx, reference_dx, rtol=0, atol=0)

print("executed path closed:", report["inference_closed"])
print("route:", route.tolist(), "packed bytes:", tape.nbytes)
print("replayed input VJP:", replayed_dx.tolist())
