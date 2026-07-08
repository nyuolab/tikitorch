# Tropical geometry of neural networks

One of the deepest reasons tropical algebra belongs in deep learning: a feed-forward
network with piecewise-linear activations (ReLU, max-pooling, maxout, and the
morphological ops here) **is** a tropical object. This page is the conceptual
companion to
[notebook 05 — *Tropical geometry*](https://github.com/nyuolab/tikitorch/blob/main/notebooks/05_tropical_geometry.ipynb);
the runnable code lives there.

## ReLU networks are tropical rational maps

A **tropical polynomial** is a max of affine pieces, `max_i (a_i·x + b_i)` with
integer slopes — a convex, piecewise-linear function. A ReLU network's output, per
coordinate, is a **difference of two tropical polynomials** (a *tropical rational
map*): `f(x) = p(x) − q(x)`, each of `p, q` convex piecewise-linear. This is an exact
equivalence, not an analogy (Zhang–Naumann–Schmidt-Hieber and the Maragos line of
work). Consequences that matter for ML:

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

`count_linear_regions` works by collecting the distinct `jacobian_signature`s (the
rounded local linear map) over the sampled points — two points share a region iff they
share a Jacobian. Watching `n_regions` grow as you add depth/width is the expressivity
story made empirical.

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
- **Certification.** Because regions are polytopes and the map is affine inside each,
  properties (monotonicity, Lipschitz bounds, robustness radii) can be reasoned about
  *exactly* per region — smooth nets only approximate this.
- **Pruning & structure.** The Newton-polytope view explains which monomials/units are
  redundant (interior points contribute no new vertex), connecting to
  `tikitorch.prune` (`tropical_prune`, `tropical_sparsity`).

## Next

- **[Beyond deep learning](beyond-deep-learning.md)** — the same algebra as
  shortest-path DP, clustering and SVM.
- **[Foundations](foundations.md)** — if the semiring itself is new, start there.
