"""Tests for tikitorch.classical — classical tropical ML methods."""
import torch
from tikitorch.classical import (tropical_distance, tropical_kmeans,
                                  TropicalNearestCentroid, TropicalSVM,
                                  tropical_pca)


def test_tropical_distance_metric_properties():
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    D = tropical_distance(X)
    assert torch.allclose(torch.diagonal(D), torch.zeros(5), atol=1e-6)   # d(x,x)=0
    assert torch.allclose(D, D.t(), atol=1e-6)                            # symmetric
    assert (D >= -1e-6).all()                                             # nonnegative
    # translation-invariant: add c*(1..1) to a row
    X2 = X.clone(); X2[0] += 3.0
    assert torch.allclose(tropical_distance(X2)[0], D[0], atol=1e-5)


def _two_tropical_clusters(n=40, d=5, sep=8.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, d, generator=g) * 0.3
    b = torch.randn(n, d, generator=g) * 0.3
    b[:, 1:] += sep                       # shift later coords -> large tropical gap
    X = torch.cat([a, b]); y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    return X, y


def test_tropical_kmeans_recovers_clusters():
    X, y = _two_tropical_clusters()
    labels, centroids = tropical_kmeans(X, k=2, seed=0)
    # purity: each predicted cluster is mostly one true class
    purity = max((labels[:40] == labels[0]).float().mean().item(),
                 (labels[40:] == labels[40]).float().mean().item())
    assert centroids.shape == (2, 5)
    assert ((labels[:40].float().mean() - labels[40:].float().mean()).abs() > 0.5)
    assert purity > 0.8, f"clusters not recovered cleanly (purity={purity:.3f})"


def test_tropical_nearest_centroid_separates():
    X, y = _two_tropical_clusters()
    clf = TropicalNearestCentroid().fit(X, y)
    assert clf.score(X, y) > 0.9


def _three_tropical_clusters(n=60, d=5, sep=6.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, d, generator=g) * 0.4
    b = torch.randn(n, d, generator=g) * 0.4; b[:, 1:] += sep
    c = torch.randn(n, d, generator=g) * 0.4; c[:, 2:] -= sep
    X = torch.cat([a, b, c])
    y = torch.cat([torch.zeros(n), torch.ones(n), 2 * torch.ones(n)]).long()
    return X, y


def test_tropical_svm_separates_and_has_positive_margin():
    X, y = _three_tropical_clusters()
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(1))
    tr, te = perm[:120], perm[120:]
    svm = TropicalSVM(margin=2.0, n_iter=200, lr=0.1).fit(X[tr], y[tr])
    assert svm.score(X[te], y[te]) > 0.9
    assert svm.margins(X[tr], y[tr]).min().item() > 0.0     # all points correct side
    assert svm.decision_function(X[te]).shape == (len(te), 3)


def test_tropical_svm_matches_or_beats_nearest_centroid():
    # max-margin apex placement should be at least as good as fixed class means.
    X, y = _three_tropical_clusters(sep=4.0)
    svm = TropicalSVM(margin=1.0, n_iter=200, lr=0.1).fit(X, y)
    nc = TropicalNearestCentroid().fit(X, y)
    assert svm.score(X, y) >= nc.score(X, y) - 1e-6


def test_tropical_pca_shape_and_preserves_distance():
    torch.manual_seed(0)
    X = torch.randn(20, 6)
    Z = tropical_pca(X, k=2, n_iter=200)
    assert Z.shape == (20, 2)
    # embedding distances correlate with tropical distances
    D = tropical_distance(X); Demb = torch.cdist(Z, Z)
    iu = torch.triu_indices(20, 20, 1)
    corr = torch.corrcoef(torch.stack([D[iu[0], iu[1]], Demb[iu[0], iu[1]]]))[0, 1]
    assert corr > 0.6
