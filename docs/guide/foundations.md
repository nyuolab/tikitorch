# Foundations: the tropical semiring

A *semiring* is arithmetic with the two basic operations swapped out. The
**tropical** (or min-plus / max-plus) semiring keeps real numbers but redefines
"multiply" and "add":

| | ordinary | tropical (min-plus) | tropical (max-plus) |
|---|---|---|---|
| "multiply" ⊗ | `a · b` | `a + b` | `a + b` |
| "add" ⊕ | `a + b` | `min(a, b)` | `max(a, b)` |
| ⊗ identity (1) | `1` | `0` | `0` |
| ⊕ identity (0) | `0` | `+∞` | `-∞` |

It is a genuine semiring: ⊗ distributes over ⊕, both are associative and
commutative, and the identities behave. There is no additive *inverse* (you can't
"subtract" a `min`), which is exactly why the algebra is order-based rather than
cancellative — and why it maps to comparators instead of multipliers.

## A dot product with no multiplies

The ordinary inner product `Σ_i a_i b_i` becomes, term by term, `min_i (a_i + b_i)`
(or `max_i`). So the tropical analogue of a matrix–vector product `y = Wx` is

```
y_j = min_i (W_ji + x_i)        # min-plus "erosion"
y_j = max_i (W_ji + x_i)        # max-plus "dilation"
```

No multiplications appear in this hard operator — only additions and comparisons. In `tikitorch` these are
`tikitorch.functional.tropical_linear` / `tropical_dilation`, and the
`nn.TropicalLinear` / `nn.TropicalDilation` modules. (See the runnable derivation in
[notebook 02 — *The tropical semiring, by hand*](https://github.com/nyuolab/tikitorch/blob/main/notebooks/02_tropical_math.ipynb).)

## Erosion, dilation, and morphology

Those two ops are precisely **grayscale mathematical morphology**: a max-plus
convolution of a signal with a structuring element *dilates* (spreads peaks); a
min-plus convolution *erodes* (carves valleys). A tropical conv layer
(`nn.TropicalConv2d`) is a *learned* structuring element — opening/closing
structure that is far more expressive than dilation alone. This is the lens
[notebook 03 — *A morphological CNN*](https://github.com/nyuolab/tikitorch/blob/main/notebooks/03_vision_cnn.ipynb)
uses to build a vision model.

## Sparse, routed gradients

Each tropical output selects **one** input — its argmin (or argmax). So the backward
pass routes the gradient through that single winning index: a one-hot scatter rather
than a dense Jacobian. That sparsity gives the operator an add/compare hardware mapping
and a different inductive bias from dense nets — closer to *routing* than to *mixing*.
A complete backward path can contain other parameter-gradient rules, so audit the full
recipe separately. Sparse routing is also why the following technique can be useful.

## The soft→hard curriculum

A hard `min`/`max` is piecewise-constant in its argument: its gradient is sparse and
flat between switches, which can stall training from scratch. TikiTorch trains with a
**soft relaxation** at temperature `tau`:

```
softmin(z, tau) = -tau · logsumexp(-z / tau)
```

which is smooth (dense gradients) for `tau > 0` and **converges to the hard min as
`tau → 0`**. The recipe — *Maslov dequantization* — is: train soft, anneal `tau → 0`,
then evaluate the corresponding hard add/compare operator. The **same weights** drive both graphs;
`tau` only changes how the forward pass routes during training. Models expose
`model.set_tau(value)` and you schedule it per step (CNN, SSM, Transformer, MoTE all
support it; `TropicalMLP` fixes its softness at construction).

```python
import torch, tikitorch.functional as F
z = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
for tau in [2.0, 1.0, 0.5, 0.1, 0.01]:
    print(tau, F.tropical_softmin(z, tau=tau).item())   # -> converges to min = -1.0
```

Soft relaxation contains transcendental operations and is not a multiplier-free path.
Hardening a tropical operator also does not certify the surrounding loss, optimizer, or
compiled build; see [Auditing closure](closure.md).

## Where to go next

- **[Why tropical for ML/DL](why-tropical-ml.md)** — hardware motivation, routing
  inductive bias, and the evidence boundary.
- **[Tropical geometry of neural nets](tropical-geometry.md)** — ReLU/morph nets *are*
  tropical rational maps; decision regions and expressivity.
- **[Beyond deep learning](beyond-deep-learning.md)** — the same `min`-plus op as
  shortest-path DP, clustering, and SVM.
