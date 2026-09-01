# Tropical geometry of neural networks

One of the deepest reasons tropical algebra belongs in deep learning: a feed-forward
network with piecewise-linear activations (ReLU, max-pooling, maxout, and the
morphological ops here) **is** a tropical object. This page is the conceptual
companion to
[notebook 05 — *Tropical geometry*](https://github.com/nyuolab/tikitorch/blob/pc-morphssm/notebooks/05_tropical_geometry.ipynb);
the runnable code lives there.

## ReLU networks are tropical rational maps

A **tropical polynomial** is a max of affine pieces, `max_i (a_i·x + b_i)` under the
relevant coefficient convention — a convex, piecewise-linear function. Results in the
tropical-neural-network literature represent broad classes of piecewise-linear network
maps as differences of such convex functions (tropical rational maps), subject to the
paper's weight and coefficient assumptions. This connection motivates the utilities
here; it does not make every arbitrary PyTorch model a TikiTorch model.

- The network's **input space is partitioned into linear regions** — a polyhedral
  complex — and within each region the map is a single affine function.
- The **decision boundary** of a classifier is where two tropical rational pieces
  cross: a union of polytope faces, not a smooth surface.
- **Depth vs width** trade off in how many linear regions they can create — the
  classic measure of expressivity.

## Seeing it: linear regions and decision regions

`tikitorch.geometry` exposes the tools to *measure* this directly:

```python
from tikitorch.geometry import (
    count_linear_regions, decision_regions_2d, jacobian_signature, newton_polytope,
)

# How many distinct affine pieces does this net realize over a sampled region?
n_regions = count_linear_regions(model, points)        # points: (N, in_dim)

# The label map over a 2-D slice — a polytope arrangement you can imshow().
grid = decision_regions_2d(model, xrange=(-3, 3), yrange=(-3, 3), n=200)

# Which linear piece is active at x? (its rounded input-gradient = the region id)
sig = jacobian_signature(model, x)
```

`count_linear_regions` collects distinct rounded `jacobian_signature`s over sampled
points. It estimates signature diversity on that sample; disconnected regions can
share a Jacobian, and finite sampling can miss regions. Treat the result as a diagnostic,
not an exact global region count.

## Newton polytopes

A tropical polynomial corresponds to the **Newton polytope** of its (ordinary)
polynomial twin: the convex hull of the exponent vectors of its monomials. The
polytope's geometry (vertices, faces, the regular subdivision induced by the
coefficients) *is* the function's piecewise-linear structure — vertices ↔ linear
pieces. `newton_polytope(exponents)` returns this hull so you can connect a small
tropical layer's weights to the shape of the function it computes.

```python
poly = newton_polytope(exponents)     # convex hull of monomial exponent vectors
```

## Why this is useful, not just pretty

- **Expressivity bounds.** Counting linear regions is the standard yardstick for "how
  complex a function can this architecture represent"; tropical geometry gives the
  cleanest derivations, and `count_linear_regions` lets you check them on real models.
- **Local analysis.** Inside a verified affine region, properties such as local
  monotonicity or a local linear bound can be analyzed from that affine map. The
  sampling utilities on this page do not certify a global robustness radius or exhaust
  every region.
- **Pruning & structure.** The Newton-polytope view explains which monomials/units are
  redundant (interior points contribute no new vertex), connecting to
  `tikitorch.prune` (`tropical_prune`, `tropical_sparsity`).

## Next

- **[Beyond deep learning](beyond-deep-learning.md)** — the same algebra as
  shortest-path DP, clustering and SVM.
- **[Foundations](foundations.md)** — if the semiring itself is new, start there.
