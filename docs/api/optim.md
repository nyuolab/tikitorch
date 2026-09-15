# `tikitorch.optim`

Optimizer closure depends on constructor options and the executed update path. Use the
full-step audit described in [Auditing closure](../guide/closure.md); a class name is not
a certificate.

::: tikitorch.optim

## Experimental integer error feedback

`tikitorch.optim.fixed_point_ef_step` is a functional CPU integer reference,
not a `torch.optim.Optimizer` or an alias for `EFTropicalSGD`. It requires int16
weights, int32 raw gradients/remainders and a fixed integer shift. See the
[carry/resume example and contract](../guide/route-training.md#integer-error-feedback-contract).
