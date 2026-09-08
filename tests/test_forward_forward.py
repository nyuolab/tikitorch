"""Tests for Tropical Forward-Forward (``tikitorch.forward_forward``).

The headline assertion is :func:`test_train_step_is_multiplier_free` — the WHOLE
forward-only TFF train step (two forward passes, the local hinge loss, its
backward, and the ``PureTropicalSGD`` update) has no registered arithmetic
blockers or unknown operations in its audited software trace. This is not a
physical-resource certificate. The rest pin the goodness functions, the tropical (subtract-max)
normalization, the negative-data generators, and that the local rule actually
separates positive from negative goodness on a learnable signal.
"""
from copy import deepcopy

import pytest
import torch

import tikitorch as tt
from tikitorch.forward_forward import (
    goodness_topk, goodness_max, tropical_max_normalize,
    tropical_ff_hinge_loss, tropical_ff_logistic_loss,
    TropicalFFLayer, TropicalFFNet, TropicalFFTrainer,
    overlay_label, make_negative_labels, scramble_features,
)

from conftest import requires_optim, requires_model


# ---------------------------------------------------------------------------
# Goodness functions — multiplier-free, correct values
# ---------------------------------------------------------------------------
def test_goodness_topk_is_sum_of_topk():
    h = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
    assert torch.allclose(goodness_topk(h, k=2), torch.tensor([9.0]))   # 5 + 4
    assert torch.allclose(goodness_topk(h, k=3), torch.tensor([12.0]))  # 5 + 4 + 3
    # k=1 reduces to the max
    assert torch.allclose(goodness_topk(h, k=1), goodness_max(h))
    # k is clamped to the feature width (no error past the width)
    assert torch.allclose(goodness_topk(h, k=99), h.sum(dim=-1))


def test_goodness_max():
    h = torch.tensor([[5.0, 1.0, 4.0], [-2.0, -9.0, -3.0]])
    assert torch.allclose(goodness_max(h), torch.tensor([5.0, -2.0]))


def test_goodness_functions_are_closed():
    h = torch.randn(8, 32)
    with tt.assert_no_multiply():
        goodness_topk(h, k=4).sum()
        goodness_max(h).sum()


# ---------------------------------------------------------------------------
# Tropical normalization — subtract the max (NOT L2)
# ---------------------------------------------------------------------------
def test_tropical_max_normalize_pins_max_to_zero():
    h = torch.randn(7, 13)
    n = tropical_max_normalize(h)
    assert torch.allclose(n.max(dim=-1).values, torch.zeros(7), atol=1e-6)
    # it is a pure shift: every row moved by a constant (pairwise gaps preserved)
    shift = (h - n)
    assert torch.allclose(shift, shift[:, :1].expand_as(shift), atol=1e-6)


def test_tropical_max_normalize_is_closed():
    h = torch.randn(8, 16)
    with tt.assert_no_multiply():
        tropical_max_normalize(h).sum()


# ---------------------------------------------------------------------------
# Negative-data generators
# ---------------------------------------------------------------------------
def test_overlay_label_writes_one_hot():
    x = torch.rand(5, 784)
    y = torch.tensor([0, 3, 7, 9, 1])
    xo = overlay_label(x, y, num_classes=10, on=1.0, off=0.0)
    assert torch.equal(xo[:, :10].argmax(dim=1), y)         # label position is the max
    assert torch.allclose(xo[:, 10:], x[:, 10:])            # pixels past the overlay untouched
    assert not xo.data_ptr() == x.data_ptr()                # a copy, input unmodified


def test_make_negative_labels_never_equal_true():
    y = torch.randint(0, 10, (1000,))
    yn = make_negative_labels(y, num_classes=10)
    assert (yn != y).all()
    assert int(yn.min()) >= 0 and int(yn.max()) < 10


def test_negative_labels_conditional_wrap_matches_modulo_and_audits():
    labels = torch.arange(10).repeat(8)
    expected_generator = torch.Generator().manual_seed(19)
    offset = torch.randint(1, 10, labels.shape, generator=expected_generator)
    expected = (labels + offset) % 10
    with tt.assert_no_multiply():
        actual = make_negative_labels(labels, 10, generator=torch.Generator().manual_seed(19))
    assert torch.equal(actual, expected)


def test_scramble_preserves_marginals_breaks_joint():
    x = torch.rand(64, 40)
    xs = scramble_features(x)
    # per-feature (column) multiset is unchanged -> sorted columns match exactly
    assert torch.allclose(x.sort(dim=0).values, xs.sort(dim=0).values)
    # but the joint is broken (the tensor differs)
    assert not torch.allclose(x, xs)


# ---------------------------------------------------------------------------
# Layer / net plumbing
# ---------------------------------------------------------------------------
def test_layer_forward_shape_and_relu():
    layer = TropicalFFLayer(20, 12, use_relu=True, goodness="topk", k=4)
    h = layer(torch.randn(6, 20))
    assert h.shape == (6, 12)
    assert (h >= 0).all()                                    # tropical ReLU = max(.,0)
    assert layer.goodness(h).shape == (6,)


def test_net_accumulated_goodness_shape_and_skip_first():
    net = TropicalFFNet([20, 16, 16, 16], num_classes=10, k=4)
    x = torch.randn(5, 20)
    g_all = net.accumulated_goodness(x, skip_first=False)
    g_skip = net.accumulated_goodness(x, skip_first=True)
    assert g_all.shape == (5,) and g_skip.shape == (5,)
    # skipping the first layer changes the score (it drops a non-negative term)
    assert not torch.allclose(g_all, g_skip)


# ---------------------------------------------------------------------------
# THE headline closure assertion: the full train step is multiplier-free.
# ---------------------------------------------------------------------------
@requires_optim("PureTropicalSGD")
def test_train_step_is_multiplier_free():
    torch.manual_seed(0)
    net = TropicalFFNet([784, 64, 64, 64], num_classes=10, goodness="topk", k=8)
    trainer = TropicalFFTrainer(net, lr_shift=4, momentum_bits=4, margin=1.0, host_metrics=False)
    # The local loss takes a mean, so a power-of-two batch makes that fixed
    # reduction a shift. A non-dyadic extent is correctly rejected by the audit.
    x = torch.rand(16, 784)
    y = torch.randint(0, 10, (16,))
    x_pos = overlay_label(x, y, 10)
    x_neg = overlay_label(x, make_negative_labels(y, 10), 10)

    # The deliverable: the WHOLE forward-only update — two forwards, the local
    # hinge loss, its backward, and the sign-only optimizer step — runs with no
    # registered multiply/divide/transcendental or unknown executed op. Physical
    # hardware claims require a separate configuration-matched build report.
    with tt.assert_no_multiply():
        trainer.train_step(x_pos, x_neg)


@requires_optim("PureTropicalSGD")
def test_train_step_closed_for_max_goodness():
    torch.manual_seed(0)
    net = TropicalFFNet([64, 32, 32], num_classes=8, goodness="max")
    trainer = TropicalFFTrainer(net, lr_shift=5, momentum_bits=0, theta="adaptive", host_metrics=False)
    x = torch.rand(16, 64)
    y = torch.randint(0, 8, (16,))
    xp = overlay_label(x, y, 8)
    xn = overlay_label(x, make_negative_labels(y, 8), 8)
    with tt.assert_no_multiply():
        trainer.train_step(xp, xn)


def test_adaptive_theta_and_tensor_diagnostics_never_roundtrip_through_python():
    net = TropicalFFNet([16, 8], num_classes=2, k=2)
    host_net = deepcopy(net)
    tensor_trainer = TropicalFFTrainer(net, host_metrics=False)
    host_trainer = TropicalFFTrainer(host_net)
    positive, negative = torch.rand(8, 16), torch.rand(8, 16)
    with tt.assert_no_multiply():
        theta = tensor_trainer._theta(torch.rand(8), torch.rand(8))
        tensor_metrics = tensor_trainer.train_step(positive, negative)
    assert isinstance(theta, torch.Tensor) and theta.ndim == 0 and not theta.requires_grad
    host_metrics = host_trainer.train_step(positive, negative)
    for name, values in tensor_metrics.items():
        assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in values)
        assert [float(value) for value in values] == host_metrics[name]
    for tensor_parameter, host_parameter in zip(net.parameters(), host_net.parameters()):
        torch.testing.assert_close(tensor_parameter, host_parameter, rtol=0, atol=0)
    with pytest.raises(tt.ClosureViolation, match="_local_scalar_dense"):
        with tt.assert_no_multiply():
            tensor_trainer.train_step(positive, negative, host_metrics=True)


@pytest.mark.parametrize("loss_fn", [tropical_ff_hinge_loss, tropical_ff_logistic_loss])
def test_loss_accepts_detached_tensor_threshold_without_host_conversion(loss_fn):
    h = torch.randn(8, 8, requires_grad=True)
    baseline = h.detach().clone().requires_grad_()
    positive = torch.arange(8) < 4
    theta = torch.tensor(0.375, requires_grad=True)
    expected = loss_fn(baseline, positive, theta=0.375, k=2)
    expected.backward()
    with tt.assert_no_multiply(allow_transcendental=loss_fn is tropical_ff_logistic_loss):
        actual = loss_fn(h, positive, theta=theta, k=2)
        actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(h.grad, baseline.grad, rtol=0, atol=0)
    assert theta.grad is None


def test_hinge_loss_gradient_is_sign_gated():
    """The loss backward is exactly ``{-1,0,+1}`` scattered into the top-k
    positions — sign-valued, no magnitude — so it feeds the SignSGD path."""
    torch.manual_seed(0)
    h = torch.randn(6, 10, requires_grad=True)
    is_pos = torch.tensor([1, 1, 1, 0, 0, 0]).bool()
    # huge margin -> every sample is active (gate open for all)
    loss = tropical_ff_hinge_loss(h, is_pos, theta=0.0, margin=1e6, goodness="topk", k=3)
    loss.backward()
    g = h.grad
    # exactly k non-zeros per row, all in {-1, 0, +1}
    assert (g != 0).sum(dim=1).tolist() == [3, 3, 3, 3, 3, 3]
    assert set(g.unique().tolist()) <= {-1.0, 0.0, 1.0}
    # positives pushed up (grad -1), negatives pushed down (grad +1)
    assert (g[is_pos].sum(dim=1) == -3).all()
    assert (g[~is_pos].sum(dim=1) == 3).all()


def test_logistic_loss_per_weight_update_is_closed_but_coeff_is_host():
    """The logistic FF loss keeps the per-weight update multiply-free (scatter of
    a host-computed coefficient) but the coefficient itself is a host sigmoid — so
    it needs ``allow_transcendental=True`` (unlike the fully-closed hinge loss)."""
    torch.manual_seed(0)
    h = torch.randn(8, 16, requires_grad=True)
    is_pos = torch.cat([torch.ones(4), torch.zeros(4)]).bool()
    # the per-weight backward (scatter) is multiply-free once the coeff exists...
    with tt.assert_no_multiply(allow_transcendental=True):
        loss = tropical_ff_logistic_loss(h, is_pos, theta=0.0, goodness="topk", k=3)
        loss.backward()
    # ...and the gradient is still top-k sparse
    assert (h.grad != 0).sum(dim=1).tolist() == [3] * 8


# ---------------------------------------------------------------------------
# The local rule actually learns: positive/negative goodness separates.
# ---------------------------------------------------------------------------
@requires_optim("PureTropicalSGD")
def test_local_rule_separates_pos_neg_goodness():
    """On a planted signal — positives carry a high-value block in columns [:8],
    negatives an equal block in the DISJOINT columns [8:16] (so the goodness is
    symmetric at init) — a single TFF layer must learn to route the positive
    pattern to high goodness and the negative one to low. That gap *opening* is
    the whole point of the local objective."""
    torch.manual_seed(0)
    D = 64
    net = TropicalFFNet([D, 64], num_classes=2, goodness="topk", k=8)
    trainer = TropicalFFTrainer(net, lr_shift=3, momentum_bits=4, margin=1.0)

    def batch(n=128):
        base = torch.rand(n, D) * 0.1
        pos = base.clone(); pos[:, :8] += 3.0                # high block at [:8]
        neg = base.clone(); neg[:, 8:16] += 3.0              # high block at [8:16] (disjoint)
        return pos, neg

    sep0 = sep = None
    for step in range(150):
        out = trainer.train_step(*batch())
        sep = out["g_pos"][0] - out["g_neg"][0]
        if sep0 is None:
            sep0 = sep
    assert abs(sep0) < 0.5            # symmetric at init (nothing to read off yet)
    assert sep > 2.0                  # training opened a clear positive gap
    assert sep > sep0 + 1.0


# ---------------------------------------------------------------------------
# The formal model (tikitorch.models.TropicalForwardForward)
# ---------------------------------------------------------------------------
@requires_model("TropicalForwardForward")
def test_model_is_drop_in_classifier_and_closed():
    import tikitorch as tt
    torch.manual_seed(0)
    m = tt.models.TropicalForwardForward(d_model=64, n_layers=3, input_dim=784, num_classes=10)
    assert m.param_count > 0 and m.task_type == "classification"
    tt.assert_closed(m)                                  # no multiply-bearing torch modules
    x = torch.rand(8, 784)
    assert m(x).shape == (8, 10)                         # goodness logits -> drop-in classifier
    assert m(x).argmax(1).shape == (8,)


@requires_model("TropicalForwardForward")
@requires_optim("PureTropicalSGD")
def test_model_forward_forward_training_is_multiplier_free():
    import tikitorch as tt
    torch.manual_seed(0)
    m = tt.models.TropicalForwardForward(d_model=64, n_layers=3, num_classes=10)
    trainer = m.make_trainer(lr_shift=4, margin=2.0, host_metrics=False)
    x, y = torch.rand(16, 784), torch.randint(0, 10, (16,))
    with tt.assert_no_multiply():
        m.forward_forward_step(x, y, trainer)


# ---------------------------------------------------------------------------
# The dyadic substrate (closure-clean, dense ±2^k — the high-accuracy variant)
# ---------------------------------------------------------------------------
def test_dyadic_layer_forward_and_substrate_validation():
    import pytest as _pytest
    layer = TropicalFFLayer(20, 12, substrate="dyadic", goodness="topk", k=4)
    assert layer.substrate == "dyadic"
    h = layer(torch.randn(5, 20))
    assert h.shape == (5, 12) and (h >= 0).all()          # relu
    with _pytest.raises(ValueError):
        TropicalFFLayer(8, 8, substrate="quadratic")       # only maxplus/dyadic


@requires_optim("PureTropicalSGD")
def test_dyadic_ff_train_step_runs_and_separates():
    """The dyadic substrate uses the SAME forward-only local rule; one step must
    run and (being dense/additive) open a positive goodness gap on a planted signal."""
    torch.manual_seed(0)
    D = 64
    net = TropicalFFNet([D, 64], num_classes=2, substrate="dyadic", goodness="topk", k=8)
    trainer = TropicalFFTrainer(net, lr_shift=6, momentum_bits=0, margin=2.0)
    sep0 = sep = None
    for _ in range(150):
        base = torch.rand(128, D) * 0.1
        pos = base.clone(); pos[:, :8] += 3.0
        neg = base.clone(); neg[:, 8:16] += 3.0
        out = trainer.train_step(pos, neg)
        sep = out["g_pos"][0] - out["g_neg"][0]
        if sep0 is None:
            sep0 = sep
    assert sep > sep0                                      # learned to separate


@requires_model("DyadicForwardForward")
def test_dyadic_model_is_structurally_closed_drop_in_classifier():
    """DyadicForwardForward passes the STRUCTURAL closure scan (DyadicLinear is a
    declared _zero_dsp_layer). Its forward is zero-DSP as a static weight-value
    property (verify_schedule), NOT via the eager op trace — so unlike the max-plus
    model it does NOT pass the runtime assert_no_multiply (the documented caveat)."""
    import tikitorch as tt
    torch.manual_seed(0)
    m = tt.models.DyadicForwardForward(d_model=64, n_layers=3, num_classes=10)
    assert m.substrate == "dyadic"
    tt.assert_closed(m)                                    # structural scan passes
    assert m(torch.rand(8, 784)).shape == (8, 10)          # drop-in classifier


@requires_model("TropicalForwardForward")
def test_model_compute_loss_is_a_non_ff_backprop_baseline():
    """compute_loss is CE on the goodness logits — differentiable (a documented
    NON-forward-forward global-backprop path), distinct from the local FF rule."""
    import tikitorch as tt
    torch.manual_seed(0)
    m = tt.models.TropicalForwardForward(d_model=32, n_layers=2, num_classes=10)
    x, y = torch.rand(6, 784), torch.randint(0, 10, (6,))
    loss = m.compute_loss(x, y)
    loss.backward()                                      # builds a graph (grad-capable forward)
    assert torch.isfinite(loss)
