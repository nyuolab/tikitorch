"""Tests for tikitorch.optim — the multiplier-free optimizers.

Each optimizer must (a) take a single .step() without error on a parameter that
has a gradient, and (b) reduce a simple convex toy loss over a handful of steps.
All tests are skip-guarded behind ``requires_optim`` so the suite stays green
until the Phase-2 optimizer module lands, then activates automatically.
"""
import pytest
import torch

import tikitorch.optim as optim
from conftest import requires_optim


def _toy_problem():
    """Minimize ||W - target||^2. Returns (param, target, loss_fn)."""
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(16, 8))
    target = torch.randn(16, 8)

    def loss_fn():
        return (W - target).pow(2).mean()

    return W, target, loss_fn


def _run_descent(make_opt, steps=200):
    W, target, loss_fn = _toy_problem()
    opt = make_opt([W])
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# One-step smoke: a single step changes params and does not NaN
# ---------------------------------------------------------------------------


@requires_optim("SignSGD")
def test_signsgd_one_step():
    W = torch.nn.Parameter(torch.randn(8, 4))
    before = W.detach().clone()
    opt = optim.SignSGD([W], lr=1e-2)
    (W.sum()).backward()
    opt.step()
    assert torch.isfinite(W).all()
    assert not torch.equal(W.detach(), before)


@requires_optim("TropicalSGD")
def test_tropicalsgd_one_step():
    W = torch.nn.Parameter(torch.randn(8, 4))
    before = W.detach().clone()
    opt = optim.TropicalSGD([W], lr=1e-2)
    (W.pow(2).sum()).backward()
    opt.step()
    assert torch.isfinite(W).all()
    assert not torch.equal(W.detach(), before)


@requires_optim("PureTropicalSGD")
def test_puretropicalsgd_one_step():
    W = torch.nn.Parameter(torch.randn(8, 4))
    before = W.detach().clone()
    opt = optim.PureTropicalSGD([W], lr_shift=6)
    (W.pow(2).sum()).backward()
    opt.step()
    assert torch.isfinite(W).all()
    assert not torch.equal(W.detach(), before)


@requires_optim("PureTropicalSGD")
def test_puretropicalsgd_lr_is_power_of_two():
    # Documented contract: lr = 2^-lr_shift (bit-shift on FPGA).
    opt = optim.PureTropicalSGD([torch.nn.Parameter(torch.zeros(2))], lr_shift=10)
    assert opt.param_groups[0]["lr"] == pytest.approx(2.0 ** -10)


# ---------------------------------------------------------------------------
# Convergence: each optimizer reduces the toy loss substantially
# ---------------------------------------------------------------------------


@requires_optim("SignSGD")
def test_signsgd_reduces_loss():
    losses = _run_descent(lambda p: optim.SignSGD(p, lr=2e-2), steps=300)
    assert losses[-1] < losses[0] * 0.5


@requires_optim("TropicalSGD")
def test_tropicalsgd_reduces_loss():
    losses = _run_descent(lambda p: optim.TropicalSGD(p, lr=2e-2), steps=300)
    assert losses[-1] < losses[0] * 0.5


@requires_optim("PureTropicalSGD")
def test_puretropicalsgd_reduces_loss():
    losses = _run_descent(lambda p: optim.PureTropicalSGD(p, lr_shift=5), steps=400)
    assert losses[-1] < losses[0] * 0.7


@requires_optim("PureTropicalSGD")
def test_puretropicalsgd_with_momentum_and_noise():
    # Exercise the integer-counter momentum and LFSR noise code paths.
    losses = _run_descent(
        lambda p: optim.PureTropicalSGD(p, lr_shift=5, momentum_bits=4,
                                        noise_scale=1e-3),
        steps=400,
    )
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0]


@requires_optim("PureTropicalSGD")
def test_puretropicalsgd_adaptive_path():
    # Exercise the sign-L1 adaptive (per-parameter power-of-2) rate path.
    losses = _run_descent(
        lambda p: optim.PureTropicalSGD(p, lr_shift=5, adaptive=True,
                                        adaptive_bits=3),
        steps=400,
    )
    assert losses[-1] < losses[0]


@requires_optim("PureTropicalSGD")
def test_optimizer_no_grad_param_is_safe():
    # A param with grad=None must be skipped, not crash.
    W = torch.nn.Parameter(torch.randn(4))
    opt = optim.PureTropicalSGD([W], lr_shift=6)
    opt.step()  # no backward called -> grad is None
    assert torch.isfinite(W).all()


# ---------------------------------------------------------------------------
# TropicalLion — Lion with integer fixed-point momentum
# ---------------------------------------------------------------------------


@requires_optim("TropicalLion")
def test_lion_one_step():
    W = torch.nn.Parameter(torch.randn(8, 4))
    before = W.detach().clone()
    opt = optim.TropicalLion([W], lr_shift=6)
    (W.pow(2).sum()).backward()
    opt.step()
    assert torch.isfinite(W).all()
    assert not torch.equal(W.detach(), before)


@requires_optim("TropicalLion")
def test_lion_lr_is_power_of_two():
    opt = optim.TropicalLion([torch.nn.Parameter(torch.zeros(2))], lr_shift=9)
    assert opt.param_groups[0]["lr"] == pytest.approx(2.0 ** -9)


@requires_optim("TropicalLion")
def test_lion_reduces_loss():
    losses = _run_descent(
        lambda p: optim.TropicalLion(p, lr_shift=5, beta1_bits=1, beta2_bits=3,
                                     grad_bits=16),
        steps=400,
    )
    assert losses[-1] < losses[0] * 0.7


@requires_optim("TropicalLion")
def test_lion_update_is_sign_valued():
    # The weight delta each step must be lr * {-1, 0, +1} (sign-only, no multiply).
    W = torch.nn.Parameter(torch.randn(64))
    opt = optim.TropicalLion([W], lr_shift=8, grad_bits=16)
    before = W.detach().clone()
    (W.pow(2).sum()).backward()
    opt.step()
    lr = 2.0 ** -8
    step = (W.detach() - before) / lr
    assert torch.all((step.abs() - step.abs().round()).abs() < 1e-5)
    assert step.abs().max() <= 1.0 + 1e-5


@requires_optim("TropicalLion")
def test_lion_momentum_collapses_on_tiny_grad_without_prescale():
    # Silicon-fidelity guard: with too few grad_bits the INTEGER EMA truncates to
    # zero on small gradients (Lion degenerates to SignSGD). A float EMA would
    # NOT show this; modeling it is the whole point.
    W = torch.nn.Parameter(torch.zeros(1000))
    opt = optim.TropicalLion([W], lr_shift=8, beta2_bits=3, grad_bits=6)
    for _ in range(5):
        opt.zero_grad()
        W.grad = torch.full_like(W, 2e-4)
        opt.step()
    assert opt.momentum_density() < 0.05  # collapsed


@requires_optim("TropicalLion")
def test_lion_momentum_survives_with_prescale():
    # With enough grad_bits the integer momentum stays alive on the same tiny grad.
    W = torch.nn.Parameter(torch.zeros(1000))
    opt = optim.TropicalLion([W], lr_shift=8, beta2_bits=3, grad_bits=16)
    for _ in range(5):
        opt.zero_grad()
        W.grad = torch.full_like(W, 2e-4)
        opt.step()
    assert opt.momentum_density() > 0.5  # alive


@requires_optim("TropicalLion")
def test_lion_no_grad_param_is_safe():
    W = torch.nn.Parameter(torch.randn(4))
    opt = optim.TropicalLion([W], lr_shift=6)
    opt.step()
    assert torch.isfinite(W).all()


# --- TropicalAdam: closed adaptive (sign-streak Rprop) optimizer --------------

@requires_optim("TropicalAdam")
def test_tropical_adam_reduces_loss():
    W, target, loss_fn = _toy_problem()
    opt = optim.TropicalAdam([W], step_init=2 ** -2, streak=2)
    first = loss_fn().item()
    for _ in range(50):
        opt.zero_grad(); loss_fn().backward(); opt.step()
    assert loss_fn().item() < first * 0.2          # converges well


@requires_optim("TropicalAdam")
def test_tropical_adam_training_step_is_multiplier_free():
    # The whole update must be zero-DSP: sign + compare + power-of-two shift only.
    import tikitorch.algorithmic as A
    import tikitorch.functional as F
    from tikitorch.audit import find_training_multiplies
    x, y = A.gen_running_max(6, 12, n_symbols=12, seed=0)
    C = A.get_task("running_max").n_symbols
    m = A.build_reasoner("running_max", d_model=32, n_layers=1, attention="tropical",
                         agg_variant="argmax", pos_mode="none")
    loss = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, C), yy.reshape(-1))
    opt = optim.TropicalAdam(m.parameters())
    find_training_multiplies(m, (x,), y, loss, optimizer=opt)         # build state
    hits = find_training_multiplies(m, (x,), y, loss, optimizer=opt)  # streak/flip active
    bad = [h for h in hits if h[0] != "elementwise-divide"]           # loss /n exempt
    assert not bad, f"TropicalAdam update has DSP-critical ops: {bad}"


@requires_optim("TropicalAdam")
def test_tropical_adam_step_is_power_of_two_and_adapts():
    import math
    W = torch.nn.Parameter(torch.zeros(8))
    opt = optim.TropicalAdam([W], step_init=2 ** -4, streak=2)
    # consistent-direction gradient -> step should ACCELERATE (grow) after the streak.
    for _ in range(6):
        opt.zero_grad(); W.grad = torch.ones_like(W); opt.step()
    grown = opt.step_density()
    assert grown > 2 ** -4                          # accelerated past the init step
    st = opt.state[W]["step"]
    l = torch.log2(st[st > 0])
    assert torch.allclose(l, l.round(), atol=1e-5)  # every step stays an exact power of two
    # oscillating gradient -> step should DECELERATE (shrink).
    for i in range(8):
        opt.zero_grad(); W.grad = torch.ones_like(W) * (1.0 if i % 2 == 0 else -1.0); opt.step()
    assert opt.step_density() < grown               # flips shrank the step


# --- Closed adaptive optimizers (Phase A): magnitude/agreement-aware, zero-DSP ---
#
# FrexpMagnitudeSGD, CountAgreementSGD, ExponentEMAAdam, PerLayerStepSGD recover a
# magnitude / dense-signal analogue WITHOUT leaving the multiplier-free regime: the
# step magnitude comes from frexp (the closed exponent tier) gathered through a
# constant power-of-two table -- never ``2**tier`` / ``ldexp`` (which lower to pow).

_NEW_CLOSED = ["FrexpMagnitudeSGD", "CountAgreementSGD", "ExponentEMAAdam", "PerLayerStepSGD"]


def _build_closed_opt(name, params, base_shift=2):
    return getattr(optim, name)(params, base_shift=base_shift)


@pytest.mark.parametrize("name", _NEW_CLOSED)
def test_new_closed_optimizer_reduces_loss(name):
    if not hasattr(optim, name):
        pytest.skip(f"{name} not available")
    W, target, loss_fn = _toy_problem()
    opt = _build_closed_opt(name, [W], base_shift=2)
    first = loss_fn().item()
    for _ in range(120):
        opt.zero_grad(); loss_fn().backward(); opt.step()
    assert torch.isfinite(W).all()
    assert loss_fn().item() < first * 0.5           # meaningfully descends the toy


@pytest.mark.parametrize("name", _NEW_CLOSED)
def test_new_closed_optimizer_training_step_is_multiplier_free(name):
    # The full fwd+loss+backward+optimizer step must be zero-DSP. Run twice: the
    # adaptive state (EMAs/counters/streaks) only activates on the second step.
    if not hasattr(optim, name):
        pytest.skip(f"{name} not available")
    import tikitorch.algorithmic as A
    import tikitorch.functional as F
    from tikitorch.audit import find_training_multiplies
    x, y = A.gen_running_max(6, 12, n_symbols=12, seed=0)
    C = A.get_task("running_max").n_symbols
    m = A.build_reasoner("running_max", d_model=32, n_layers=1, attention="tropical",
                         agg_variant="argmax", pos_mode="none")
    loss = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, C), yy.reshape(-1))
    opt = _build_closed_opt(name, m.parameters(), base_shift=6)
    find_training_multiplies(m, (x,), y, loss, optimizer=opt)         # build state
    hits = find_training_multiplies(m, (x,), y, loss, optimizer=opt)  # adaptive active
    bad = [h for h in hits if h[0] != "elementwise-divide"]           # loss /n exempt
    assert not bad, f"{name} update has DSP-critical ops: {bad}"


@pytest.mark.parametrize("name", _NEW_CLOSED)
def test_new_closed_optimizer_step_stays_power_of_two(name):
    # Every INDIVIDUAL applied step magnitude must be an exact power of two (a
    # bit-shift). The step may change between steps (Rprop accel), so the cumulative
    # sum need not be a power of two -- check each step's delta in isolation.
    if not hasattr(optim, name):
        pytest.skip(f"{name} not available")
    W = torch.nn.Parameter(torch.zeros(32))
    opt = _build_closed_opt(name, [W], base_shift=4)
    moved = False
    for _ in range(5):
        before = W.detach().clone()
        opt.zero_grad(); W.grad = torch.ones_like(W); opt.step()
        delta = (W.detach() - before).abs()
        nz = delta[delta > 0]
        if nz.numel() == 0:
            continue
        moved = True
        l = torch.log2(nz)
        assert torch.allclose(l, l.round(), atol=1e-5)   # exact powers of two only
    assert moved                                          # it actually updated


@requires_optim("FrexpMagnitudeSGD")
def test_frexp_magnitude_scales_step_with_relative_gradient_spike():
    # FrexpMagnitudeSGD tracks each param's typical exponent with an integer EMA
    # (the Adam-like normalizer: a CONSTANT grad -> tier 0 -> base step regardless of
    # absolute size). A sudden magnitude SPIKE above the running EMA must widen the
    # step (bigger tier -> smaller shift -> bigger 2^-shift) vs the steady step.
    W = torch.nn.Parameter(torch.zeros(64))
    opt = optim.FrexpMagnitudeSGD([W], base_shift=8, tier_span=4)
    for _ in range(8):                               # warm up EMA on small, steady grads
        opt.zero_grad(); W.grad = torch.full_like(W, 1.0 / 64); opt.step()
    b0 = W.detach().clone()
    opt.zero_grad(); W.grad = torch.full_like(W, 1.0 / 64); opt.step()
    steady = (W.detach() - b0).abs().max().item()   # steady-state step size
    b1 = W.detach().clone()
    opt.zero_grad(); W.grad = torch.full_like(W, 256.0); opt.step()   # huge spike
    spike = (W.detach() - b1).abs().max().item()
    assert spike > steady                            # spike above EMA -> wider step


@requires_optim("CountAgreementSGD")
def test_count_agreement_accumulate_hook_pools_signs():
    # The accumulate() hook reads sign(grad) into int counters and zeroes grad;
    # consistent agreement over a window must drive a (zero-DSP) update.
    W = torch.nn.Parameter(torch.zeros(16))
    opt = optim.CountAgreementSGD([W], base_shift=4)
    before = W.detach().clone()
    opt.reset_accum()
    for _ in range(5):                               # 5 microbatches, all grad>0
        opt.zero_grad(); W.grad = torch.ones_like(W); opt.accumulate()
        assert torch.count_nonzero(W.grad) == 0      # accumulate zeroed the grad
    opt.step()
    assert (W.detach() < before).all()               # net positive grad -> param decreased


@requires_optim("PerLayerStepSGD")
def test_perlayer_step_is_single_shared_scalar():
    # One shared power-of-two step magnitude across the whole tensor (per-element
    # direction only) -> all nonzero |updates| are identical.
    W = torch.nn.Parameter(torch.zeros(8, 4))
    opt = optim.PerLayerStepSGD([W], base_shift=4)
    before = W.detach().clone()
    W.grad = torch.randn(8, 4); opt.zero_grad(); W.grad = torch.randn(8, 4)
    opt.step()
    delta = (W.detach() - before).abs()
    nz = delta[delta > 0]
    assert nz.numel() > 0
    assert torch.allclose(nz, nz.flatten()[0].expand_as(nz))  # one shared step magnitude
