"""Tests for tikitorch.prune — tropical structured pruning."""
import torch
import tikitorch as tt
from tikitorch.prune import tropical_prune, tropical_sparsity
from tikitorch.audit import assert_closed


def test_tropical_prune_runs_and_preserves_closure():
    m = tt.models.TropicalCNN(in_channels=1, num_classes=10, channels=(8, 16))
    assert tropical_sparsity(m) == 0.0
    pruned, total = tropical_prune(m, sparsity=0.5)
    assert pruned > 0 and 0.4 < tropical_sparsity(m) < 0.6
    # still runs and is still multiplier-free after pruning
    out = m(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10) and torch.isfinite(out).all()
    assert_closed(m)


def test_prune_inert_taps_are_infinite():
    import tikitorch.nn as nn
    conv = nn.TropicalConv2d(2, 4, 3, mode="dilation")
    tropical_prune(conv, sparsity=0.5)
    # dilation prunes to -inf (absorbing element of max-plus)
    assert torch.isneginf(conv.weight).any()
