"""Forward/backward smoke tests for every tikitorch.models architecture.

For each model and each supported ``task_type`` we check:
  * forward produces a finite tensor of the expected shape,
  * compute_loss is a finite scalar and backward populates gradients,
  * param_count is a positive int.

All tests are skip-guarded behind ``requires_model`` so the suite stays green
until the Phase-2 models module lands, then activates automatically (the
Phase-3 verification path).
"""
import pytest
import torch

import tikitorch.models as models
from conftest import requires_model


def _assert_trains_one_step(model, x, y, expected_out_shape=None):
    """Forward + compute_loss + backward must work and stay finite."""
    out = model(x)
    assert torch.isfinite(out).all()
    if expected_out_shape is not None:
        assert out.shape == expected_out_shape
    loss = model.compute_loss(x, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    model.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)
    assert isinstance(model.param_count, int) and model.param_count > 0


# ---------------------------------------------------------------------------
# TropicalMLP
# ---------------------------------------------------------------------------


@requires_model("TropicalMLP")
def test_tropical_mlp_classification():
    m = models.TropicalMLP(d_model=32, n_layers=2, input_dim=20, num_classes=5,
                           task_type="classification")
    x = torch.randn(8, 20)
    y = torch.randint(0, 5, (8,))
    _assert_trains_one_step(m, x, y, expected_out_shape=(8, 5))


@requires_model("TropicalMLP")
def test_tropical_mlp_soft_hard_switch_and_sparsity():
    m = models.TropicalMLP(d_model=16, n_layers=2, input_dim=10, num_classes=4)
    x = torch.randn(6, 10)
    y = torch.randint(0, 4, (6,))
    m.compute_loss(x, y).backward()
    summ = m.get_gradient_sparsity_summary()
    assert "grad_sparsity/tropical_mean" in summ
    # switch to hard min for deployment must not error and stays finite.
    m.switch_to_hard_min()
    assert torch.isfinite(m(x)).all()
    m.switch_to_soft_min()
    m.set_temperature(0.1)


@requires_model("TropicalMLP")
def test_tropical_mlp_hardware_cost():
    m = models.TropicalMLP(d_model=16, n_layers=2, input_dim=10, num_classes=4)
    if hasattr(m, "hardware_cost"):
        prof = m.hardware_cost(batch_size=1)
        assert hasattr(prof, "multiplications")
        # alias must point at the same method.
        assert m.get_hardware_cost_profile(batch_size=1).multiplications == prof.multiplications


# ---------------------------------------------------------------------------
# TropicalSequenceModel
# ---------------------------------------------------------------------------


@requires_model("TropicalSequenceModel")
def test_tropical_sequence_model():
    m = models.TropicalSequenceModel(d_model=32, n_layers=2, input_dim=1,
                                     seq_len=24, num_classes=5,
                                     task_type="classification")
    x = torch.randn(4, 24, 1)
    y = torch.randint(0, 5, (4,))
    _assert_trains_one_step(m, x, y, expected_out_shape=(4, 5))


# ---------------------------------------------------------------------------
# MorphSSM — all three task types
# ---------------------------------------------------------------------------


@requires_model("MorphSSM")
def test_morphssm_classification():
    m = models.MorphSSM(d_model=24, n_layers=2, input_dim=1, num_classes=6,
                        task_type="classification")
    x = torch.randn(4, 32, 1)
    y = torch.randint(0, 6, (4,))
    _assert_trains_one_step(m, x, y, expected_out_shape=(4, 6))


@requires_model("MorphSSM")
def test_morphssm_lm():
    m = models.MorphSSM(d_model=24, n_layers=2, vocab_size=37, task_type="lm")
    x = torch.randint(0, 37, (3, 16))
    y = torch.randint(0, 37, (3, 16))
    out = m(x)
    assert out.shape == (3, 16, 37)
    loss = m.compute_loss(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.param_count > 0


@requires_model("MorphSSM")
def test_morphssm_regression_uses_plain_mse():
    m = models.MorphSSM(d_model=24, n_layers=2, input_dim=1, num_classes=1,
                        task_type="regression")
    x = torch.randn(3, 20, 1)
    y = torch.randn(3, 20, 1)
    out = m(x)
    assert torch.isfinite(out).all()
    loss = m.compute_loss(x, y)   # must be plain MSE, no GenerationLoss
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()


@requires_model("MorphSSM")
def test_morphssm_default_n_layers_is_8():
    # API_SPEC: preserve forge default n_layers=8 for MorphSSM.
    import inspect
    sig = inspect.signature(models.MorphSSM.__init__)
    assert sig.parameters["n_layers"].default == 8


# ---------------------------------------------------------------------------
# MorphMLP
# ---------------------------------------------------------------------------


@requires_model("MorphMLP")
def test_morphmlp_classification():
    m = models.MorphMLP(d_model=32, n_layers=2, input_dim=20, num_classes=5)
    x = torch.randn(8, 20)
    y = torch.randint(0, 5, (8,))
    _assert_trains_one_step(m, x, y, expected_out_shape=(8, 5))


# ---------------------------------------------------------------------------
# TropicalTransformer
# ---------------------------------------------------------------------------


@requires_model("TropicalTransformer")
def test_tropical_transformer_classification():
    m = models.TropicalTransformer(d_model=32, n_heads=4, n_layers=2,
                                   input_dim=1, num_classes=5,
                                   max_seq_len=32, task_type="classification")
    x = torch.randn(3, 16, 1)
    y = torch.randint(0, 5, (3,))
    _assert_trains_one_step(m, x, y, expected_out_shape=(3, 5))


@requires_model("TropicalTransformer")
def test_tropical_transformer_lm_and_set_tau():
    m = models.TropicalTransformer(d_model=32, n_heads=4, n_layers=2,
                                   vocab_size=29, max_seq_len=32, task_type="lm")
    x = torch.randint(0, 29, (2, 12))
    y = torch.randint(0, 29, (2, 12))
    out = m(x)
    assert out.shape == (2, 12, 29)
    m.compute_loss(x, y).backward()
    m.set_tau(0.0)  # cascade to all blocks; must not error


@requires_model("TropicalTransformer")
def test_tropical_transformer_d_ff_default():
    # d_ff=None -> d_model*4
    m = models.TropicalTransformer(d_model=16, n_heads=4, n_layers=1, d_ff=None,
                                   input_dim=1, num_classes=3, max_seq_len=16)
    x = torch.randn(2, 8, 1)
    assert torch.isfinite(m(x)).all()


# ---------------------------------------------------------------------------
# MoTEModel (+ alias MoTE)
# ---------------------------------------------------------------------------


@requires_model("MoTEModel")
def test_mote_classification_and_aux_loss():
    m = models.MoTEModel(d_model=32, n_experts=4, expert_type="transformer",
                         n_heads=4, input_dim=1, num_classes=5,
                         max_seq_len=32, task_type="classification")
    x = torch.randn(3, 16, 1)
    y = torch.randint(0, 5, (3,))
    out = m(x)
    assert out.shape == (3, 5)
    loss = m.compute_loss(x, y)   # task loss + aux_loss_weight * aux_loss
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    assert m.param_count > 0
    # get_expert_utilization must exist and be dict-like.
    assert isinstance(m.get_expert_utilization(), dict)


@requires_model("MoTE")
def test_mote_alias_identical():
    assert models.MoTE is models.MoTEModel


@requires_model("TropicalCNN")
def test_tropical_cnn_classification():
    model = models.TropicalCNN(in_channels=1, num_classes=10, channels=(8, 16))
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    _assert_trains_one_step(model, x, y, expected_out_shape=(4, 10))
    assert isinstance(model.param_count, int) and model.param_count > 0


@requires_model("TropicalCNN")
def test_tropical_cnn_is_fully_closed():
    # The default (tropical head) CNN must be multiplier-free; the linear-head
    # variant must NOT be (it is the declared DSP boundary).
    from tikitorch.audit import assert_closed, find_closure_violations
    assert_closed(models.TropicalCNN(channels=(8, 16)))  # passes
    linear = models.TropicalCNN(channels=(8, 16), head_type="linear")
    assert find_closure_violations(linear) == [("head", "Linear")]
    with pytest.raises(AssertionError):
        assert_closed(linear)
    assert_closed(linear, allow={"head"})  # declared boundary ok


@requires_model("TropicalCNN")
def test_tropical_cnn_3channel_input():
    model = models.TropicalCNN(in_channels=3, num_classes=4, channels=(8,))
    x = torch.randn(2, 3, 16, 16)
    _assert_trains_one_step(model, x, torch.randint(0, 4, (2,)), expected_out_shape=(2, 4))


@requires_model("MoTEModel")
def test_mote_small_shape_not_all_zero():
    # Regression: capacity_factor*B*L/n_experts < 1 must not zero the whole block.
    import torch
    m = models.MoTEModel(d_model=16, n_experts=8, expert_type="transformer",
                         task_type="classification", input_dim=1, num_classes=4,
                         max_seq_len=8, n_shared_pre=1, n_shared_post=1)
    x = torch.randn(2, 4, 1)   # B*L=8, /8 experts *1.5 -> capacity rounds small
    out = m(x)
    assert torch.isfinite(out).all() and out.abs().sum() > 0


# ---------------------------------------------------------------------------
# Closure CI: every model in its DEFAULT config must be fully multiplier-free.
# ---------------------------------------------------------------------------

_CLOSED_MODELS = {
    "TropicalMLP": dict(d_model=32, n_layers=2, input_dim=16, num_classes=10),
    "MorphMLP": dict(d_model=32, n_layers=2, input_dim=16, num_classes=10),
    "TropicalSequenceModel": dict(d_model=32, n_layers=2, input_dim=1, seq_len=20, num_classes=10),
    "MorphSSM": dict(d_model=32, n_layers=2, task_type="classification", input_dim=1, num_classes=10),
    "TropicalCNN": dict(in_channels=1, num_classes=10, channels=(8, 16)),
    "TropicalTransformer": dict(d_model=32, n_heads=2, n_layers=2, task_type="lm",
                                vocab_size=128, max_seq_len=16),
    "MoTEModel": dict(d_model=32, n_experts=4, task_type="classification", input_dim=1,
                      num_classes=10, max_seq_len=16, n_shared_pre=1, n_shared_post=1),
}


@pytest.mark.parametrize("name,kw", list(_CLOSED_MODELS.items()))
def test_model_is_closed_by_default(name, kw):
    pytest.importorskip("tikitorch.models")
    from tikitorch.audit import assert_closed, find_closure_violations
    if not hasattr(models, name):
        pytest.skip(f"{name} not present")
    m = getattr(models, name)(**kw)
    violations = find_closure_violations(m)
    assert violations == [], f"{name} not closed by default: {violations}"
    assert_closed(m)
