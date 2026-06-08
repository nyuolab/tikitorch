"""tikitorch.geometry — tropical-geometry tools for tropical/PWL networks.

Operationalizes *Tropical Geometry of Deep Neural Networks* (Zhang, Naitzat,
Lim, ICML 2018): a tropical/ReLU network is a piecewise-linear (tropical
rational) map whose **linear regions** correspond to vertices of the polytopes of
its tropical polynomials, and whose **decision boundary** is a tropical
hypersurface. These utilities measure those objects for any TikiTorch (or PWL)
model.

    from tikitorch.geometry import count_linear_regions, decision_regions_2d, newton_polytope
    n = count_linear_regions(model, points)          # distinct affine pieces (sampled)
    grid = decision_regions_2d(model, (-2, 2), (-2, 2))  # class map -> tropical hypersurface
    hull = newton_polytope(monomial_exponents)       # tropical Newton polytope vertices
"""
import torch

__all__ = ["count_linear_regions", "decision_regions_2d", "newton_polytope",
           "jacobian_signature"]


@torch.no_grad()
def _unused():
    pass


def jacobian_signature(model, x, round_decimals: int = 4):
    """Gradient of the summed output w.r.t. one input point.

    A PWL map is affine on each linear region, so this gradient is constant
    within a region and changes across regions -> it is a region fingerprint.
    """
    x = x.detach().clone().requires_grad_(True)
    y = model(x.unsqueeze(0)).sum()
    (g,) = torch.autograd.grad(y, x)
    g = g.reshape(-1)
    # Scale-invariant signature: normalize by max-abs so a region is identified by
    # its gradient *direction* (robust to tiny magnitudes). An all-zero gradient
    # (a collapsed / dead-cell model) hashes to the zero signature -> one region,
    # which is itself a useful diagnostic.
    scale = g.abs().max()
    if scale > 0:
        g = g / scale
    return tuple(torch.round(g, decimals=round_decimals).tolist())


def count_linear_regions(model, points, round_decimals: int = 4):
    """Count distinct linear regions a PWL model exhibits over ``points``.

    ``points``: ``(N, *input_shape)`` tensor of sample inputs. Returns the number
    of distinct affine pieces seen (a sampling lower bound on the true region
    count). Each region has a constant input->output Jacobian; we hash the
    summed-output gradient per point and count uniques. Deeper tropical nets
    create exponentially more regions (Zhang et al., Thm. on depth).
    """
    model_was_training = model.training
    model.eval()
    sigs = set()
    for i in range(points.shape[0]):
        sigs.add(jacobian_signature(model, points[i], round_decimals))
    if model_was_training:
        model.train()
    return len(sigs)


@torch.no_grad()
def decision_regions_2d(model, xrange, yrange, n: int = 100, device="cpu"):
    """Predicted-class map over a 2D input grid (the model's tropical hypersurface).

    Returns ``(grid_preds[n,n], xs[n], ys[n])``. The boundaries between regions of
    ``grid_preds`` are the network's decision boundary = a tropical hypersurface.
    The model must accept ``(B, 2)`` inputs and return ``(B, C)`` logits.
    """
    xs = torch.linspace(xrange[0], xrange[1], n)
    ys = torch.linspace(yrange[0], yrange[1], n)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(device)
    model.eval()
    preds = model(pts).argmax(dim=-1).reshape(n, n).cpu()
    return preds, xs, ys


def newton_polytope(exponents):
    """Vertices of the tropical Newton polytope = convex hull of monomial exponents.

    A tropical polynomial ``p(x) = max_k (c_k + <a_k, x>)`` has Newton polytope
    ``conv{a_k}``; its vertices are the monomials that are "active" somewhere (the
    linear pieces of ``p``). ``exponents``: ``(K, d)`` array-like of the ``a_k``.
    Returns the vertex exponents (uses scipy's ConvexHull when available; falls
    back to the input points for d<=1 or if scipy is absent).
    """
    pts = torch.as_tensor(exponents, dtype=torch.float64)
    if pts.dim() != 2 or pts.shape[0] <= 2 or pts.shape[1] <= 1:
        return pts
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        import warnings
        warnings.warn(
            "newton_polytope: scipy is not installed, so all input points are "
            "returned instead of the true Newton-polytope vertices. Install scipy "
            "(`pip install scipy` or `pip install tikitorch[dev]`) for correct results.",
            RuntimeWarning, stacklevel=2,
        )
        return pts
    try:
        hull = ConvexHull(pts.numpy())
        return pts[list(hull.vertices)]
    except Exception:
        # Degenerate/low-rank point sets that qhull can't process: fall back to
        # the input points (a superset of the true vertices).
        return pts
