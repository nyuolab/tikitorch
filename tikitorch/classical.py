"""tikitorch.classical — classical tropical machine-learning methods.

Beyond deep learning, the tropical (max-plus) semiring has a full classical ML
toolkit: the tropical (Hilbert projective) metric, tropical clustering, tropical
classifiers, and tropical dimensionality reduction, all operating on the tropical
projective torus R^d / R(1,...,1). (Yoshida et al.; Maragos, *Tropical Geometry
for ML*, ICASSP 2024; the TML package, arXiv:2309.01082.) TikiTorch ports them to
a PyTorch-native, differentiable-where-possible API so the deep and classical
tropical worlds live in one library.

    from tikitorch.classical import tropical_distance, tropical_kmeans, TropicalNearestCentroid, tropical_pca
"""
import torch

__all__ = ["tropical_distance", "tropical_projective_normalize",
           "tropical_kmeans", "TropicalNearestCentroid", "TropicalSVM",
           "tropical_pca"]


def tropical_projective_normalize(X):
    """Canonicalize points on the tropical projective torus (subtract coord 0).

    Points equal up to adding a constant (lambda * (1,...,1)) are identified; we
    pick the representative with first coordinate 0.
    """
    return X - X[..., :1]


def tropical_distance(X, Y=None):
    """Pairwise tropical (Hilbert projective) distance.

    ``d_tr(x, y) = max_k(x_k - y_k) - min_k(x_k - y_k)``. ``X: (n, d)``,
    ``Y: (m, d)`` (default Y=X) -> ``(n, m)``. Translation-invariant (a metric on
    the tropical projective torus); multiplier-free.
    """
    if Y is None:
        Y = X
    diff = X.unsqueeze(1) - Y.unsqueeze(0)          # (n, m, d)
    return diff.max(dim=-1).values - diff.min(dim=-1).values


def tropical_kmeans(X, k, n_iter: int = 50, seed: int = 0):
    """Tropical k-means over the projective torus (Lloyd with the tropical metric).

    Assigns by :func:`tropical_distance`; updates each centroid to the mean of its
    members on the projective torus (a tractable tropical representative).
    Returns ``(labels:(n,), centroids:(k,d))``.
    """
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    Xn = tropical_projective_normalize(X)
    centroids = Xn[torch.randperm(n, generator=g)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(n_iter):
        D = tropical_distance(Xn, centroids)        # (n, k)
        new_labels = D.argmin(dim=-1)
        if torch.equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            members = Xn[labels == c]
            if members.numel():
                centroids[c] = tropical_projective_normalize(members.mean(dim=0, keepdim=True))[0]
    return labels, centroids


class TropicalNearestCentroid:
    """Supervised tropical classifier: nearest per-class centroid in the tropical metric.

    A clean, correct tropical classifier (the tropical-metric analogue of nearest
    centroid). For the *max-margin* tropical SVM (an LP over tropical hyperplanes;
    Tang & Yoshida), see TROPICAL_DL_ROADMAP.md — it is the documented next
    extension; this centroid classifier is the shipped, tested baseline.
    """

    def __init__(self):
        self.centroids = None
        self.classes = None

    def fit(self, X, y):
        Xn = tropical_projective_normalize(X)
        self.classes = torch.unique(y)
        self.centroids = torch.stack([
            tropical_projective_normalize(Xn[y == c].mean(dim=0, keepdim=True))[0]
            for c in self.classes])
        return self

    def predict(self, X):
        Xn = tropical_projective_normalize(X)
        D = tropical_distance(Xn, self.centroids)
        return self.classes[D.argmin(dim=-1)]

    def score(self, X, y):
        return (self.predict(X) == y).float().mean().item()


class TropicalSVM:
    """Max-margin tropical classifier (a learned apex per class).

    Upgrades :class:`TropicalNearestCentroid` from *fixed* class means to
    *max-margin* apices. Each class ``c`` gets an apex ``omega_c`` on the tropical
    projective torus; prediction is ``argmin_c d_tr(x, omega_c)`` in the tropical
    (Hilbert projective) metric, and the apices are placed to maximize the minimum
    **tropical margin**

        ``m_i = d_tr(x_i, nearest-wrong-apex) - d_tr(x_i, own-apex)``,

    i.e. each point should sit much closer to its own class apex than to any
    other. We optimize the soft-margin (hinge) objective
    ``mean_i softplus(margin - m_i)`` by gradient descent on the apices.

    The exact max-margin tropical hyperplane is a linear program over tropical
    hyperplanes (Tang, Wang & Yoshida, *Tropical Support Vector Machines*, 2020;
    Gärtner & Jaggi). This is a differentiable subgradient realization of that
    same max-margin objective, and — like the LP version — **inference is
    multiplier-free**: :func:`tropical_distance` is a max minus a min of
    coordinate differences (comparisons and additions only), so a fitted
    ``TropicalSVM`` is ``assert_closed``-compatible at predict time.
    """

    def __init__(self, margin: float = 1.0, n_iter: int = 300, lr: float = 0.1,
                 seed: int = 0):
        self.margin = margin
        self.n_iter = n_iter
        self.lr = lr
        self.seed = seed
        self.apices = None        # (C, d)
        self.classes = None       # (C,)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        Xn = tropical_projective_normalize(X)
        self.classes = torch.unique(y)
        C = len(self.classes)
        # warm-start apices at class means on the projective torus.
        apex = torch.stack([
            tropical_projective_normalize(Xn[y == c].mean(dim=0, keepdim=True))[0]
            for c in self.classes]).clone()
        apex.requires_grad_(True)
        # one-hot of the true-class index for masking the own/other distances.
        y_idx = torch.stack([(y == c) for c in self.classes], dim=1).float()  # (n, C)
        opt = torch.optim.Adam([apex], lr=self.lr)
        for _ in range(self.n_iter):
            opt.zero_grad()
            D = tropical_distance(Xn, apex)                 # (n, C)
            own = (D * y_idx).sum(dim=1)                    # own-class distance
            wrong = D + y_idx * 1e9                         # mask out own class
            other = wrong.min(dim=1).values                 # nearest wrong apex
            m = other - own                                 # per-point margin
            loss = torch.nn.functional.softplus(self.margin - m).mean()
            loss.backward()
            opt.step()
        self.apices = tropical_projective_normalize(apex.detach())
        return self

    def decision_function(self, X):
        """Negative tropical distance to each class apex (higher = more likely)."""
        Xn = tropical_projective_normalize(X)
        return -tropical_distance(Xn, self.apices)          # (n, C)

    def predict(self, X):
        Xn = tropical_projective_normalize(X)
        D = tropical_distance(Xn, self.apices)
        return self.classes[D.argmin(dim=-1)]

    def margins(self, X, y):
        """Per-point tropical margin under the fitted apices."""
        Xn = tropical_projective_normalize(X)
        D = tropical_distance(Xn, self.apices)
        y_idx = torch.stack([(y == c) for c in self.classes], dim=1).float()
        own = (D * y_idx).sum(dim=1)
        other = (D + y_idx * 1e9).min(dim=1).values
        return other - own

    def score(self, X, y):
        return (self.predict(X) == y).float().mean().item()


def tropical_pca(X, k: int = 2, n_iter: int = 300, lr: float = 0.1, seed: int = 0):
    """Tropical dimensionality reduction via tropical-metric MDS.

    Embeds points into ``k`` dims so Euclidean distances in the embedding match
    the pairwise tropical (Hilbert) distances (stress minimization by gradient
    descent). A tractable, differentiable stand-in for the tropical-polytope PCA
    of Yoshida et al.; useful for visualizing data on the tropical torus.
    Returns the ``(n, k)`` embedding.
    """
    g = torch.Generator().manual_seed(seed)
    D = tropical_distance(X).detach()
    n = X.shape[0]
    Z = torch.randn(n, k, generator=g) * 0.1
    Z.requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        d_emb = torch.cdist(Z, Z)
        loss = ((d_emb - D) ** 2).mean()
        loss.backward()
        opt.step()
    return Z.detach()
