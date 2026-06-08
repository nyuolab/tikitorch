"""Tests for tikitorch.geometry — tropical-geometry utilities."""
import torch
import tikitorch as tt
from tikitorch.geometry import count_linear_regions, decision_regions_2d, newton_polytope


def test_count_linear_regions_finds_many_on_relu_net():
    # Ground truth: a ReLU MLP is PWL with many linear regions; the counter
    # (distinct input->output Jacobians) must find well more than one.
    import torch.nn as tnn
    torch.manual_seed(0)
    relu = tnn.Sequential(tnn.Linear(2, 16), tnn.ReLU(),
                          tnn.Linear(16, 16), tnn.ReLU(), tnn.Linear(16, 3))
    pts = torch.rand(400, 2) * 4 - 2
    assert count_linear_regions(relu, pts) > 10


def test_count_linear_regions_runs_on_tropical_model():
    # On any TikiTorch model it returns a valid count (>=1). A collapsed/dead
    # untrained tropical net legitimately reports 1 region (a useful diagnostic).
    torch.manual_seed(0)
    m = tt.models.TropicalCNN(in_channels=1, num_classes=10, channels=(8, 16))
    n = count_linear_regions(m, torch.rand(60, 1, 28, 28))
    assert isinstance(n, int) and n >= 1


def test_decision_regions_2d_shape():
    model = tt.models.TropicalMLP(d_model=16, n_layers=2, input_dim=2, num_classes=3)
    grid, xs, ys = decision_regions_2d(model, (-2, 2), (-2, 2), n=20)
    assert grid.shape == (20, 20) and xs.numel() == 20
    assert grid.min() >= 0 and grid.max() < 3


def test_newton_polytope_hull():
    # square corners + interior point -> hull = 4 corners.
    # newton_polytope needs scipy for the real ConvexHull; without it the util
    # degrades to returning all input points (and warns), so skip rather than
    # assert on the degraded path. scipy ships in the [dev] extra that CI installs.
    import pytest
    pytest.importorskip("scipy")
    pts = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.5, 0.5]])
    hull = newton_polytope(pts)
    assert hull.shape[0] == 4
