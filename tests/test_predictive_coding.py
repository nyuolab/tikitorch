"""Tests for tropical predictive coding (TropicalPCLayer / TropicalPCNet).

Predictive coding here is a local, backprop-free learning rule. The two
properties that matter:

  1. **Scoped closure** — selected tropical and dyadic recipes expose explicit
     add/compare/scatter/shift boundaries. Tests name the exact path they cover;
     the nudge solver's exact-D division and ordinary optimizers remain open.
  2. **It learns** — the local rule actually reduces prediction error and lifts
     classification accuracy well above chance, including at depth, where the
     unit-gain 0/1 Jacobian is supposed to keep error from vanishing.
"""
import pytest
import torch

import tikitorch as tt
# Import from the submodule directly so this test does not depend on the
# tikitorch.nn package re-export (the convenience export in nn/__init__.py).
from tikitorch.nn.predictive_coding import (
    TropicalPCLayer,
    TropicalPCNet,
    DyadicPCNet,
    SparseCodingPC,
)


def _blobs(n_per, n_classes, d_in, seed=0, spread=0.6):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, d_in, generator=g) * 2.0
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(means[c] + spread * torch.randn(n_per, d_in, generator=g))
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    X, Y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], Y[perm]


# ---------------------------------------------------------------------------
def test_layer_shapes_and_kinds():
    for kind in ("dilation", "erosion"):
        layer = TropicalPCLayer(6, 5, kind=kind)
        mu, idx = layer.predict(torch.randn(3, 6))
        assert mu.shape == (3, 5)
        assert idx.shape == (3, 5)
        assert idx.max() < 6 and idx.min() >= 0


def test_net_alternates_kinds():
    net = TropicalPCNet([8, 16, 16, 4])
    kinds = [l.kind for l in net.layers]
    assert kinds == ["dilation", "erosion", "dilation"]


def test_pc_step_is_closure_clean():
    """One settle + local update must execute no hardware multiply."""
    net = TropicalPCNet([20, 32, 32, 4], gamma_shift=2, lr_shift=6)
    x = torch.randn(16, 20)
    y = torch.randint(0, 4, (16,))
    target = net._make_target(y, 4, ref=x)
    with tt.assert_no_multiply():
        out = net.infer(x, target=target, n_steps=6, init="forward")
        net._apply_local_updates(out["states"])


def test_sign_mode_also_closure_clean():
    net = TropicalPCNet([20, 32, 4], weight_update="sign", lr_shift=5)
    x = torch.randn(8, 20)
    target = net._make_target(torch.randint(0, 4, (8,)), 4, ref=x)
    with tt.assert_no_multiply():
        out = net.infer(x, target=target, n_steps=4, init="forward")
        net._apply_local_updates(out["states"])


def test_forward_init_zeros_interior_error():
    """Forward-init seeds states from the feed-forward pass, so with no clamp
    every interior error is EXACTLY 0 — and zero-init is not, proving it's the
    seeding (not a trivially-small metric) doing the work."""
    net = TropicalPCNet([12, 16, 16, 5])
    x = torch.randn(7, 12)
    fwd = net.infer(x, target=None, n_steps=1, init="forward")
    assert fwd["trace"]["energy"][0] == 0.0          # exact, by construction
    zero = net.infer(x, target=None, n_steps=1, init="zero")
    assert zero["trace"]["energy"][0] > 1.0          # contrast: zero-init is far off


def test_pc_learns_blobs():
    """The local rule lifts accuracy far above chance on a separable task."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 20, seed=1)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = TropicalPCNet([20, 32, 32, 4], gamma_shift=2, lr_shift=5)
    for _ in range(6):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], n_steps=8, init="forward")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.85, f"PC failed to learn separable blobs: acc={acc:.3f}"


def test_deep_pc_trains_with_identity_init():
    """The depth claim: a deep (10-edge) net trains with identity init.

    Identity-initialised square interior layers keep the deep representation
    from collapsing and route error straight down the diagonal — accuracy stays
    flat with depth. (A uniform-init baseline collapses to chance by ~depth 8;
    see test_deep_baseline_collapses for the contrast.)
    """
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 24, seed=2)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    dims = [24] * 10 + [4]                       # 10 edges, square interior
    net = TropicalPCNet(dims, gamma_shift=2, lr_shift=5, init_identity=True)
    for _ in range(8):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], n_steps=10, init="forward")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.8, f"deep PC failed to learn with identity init: acc={acc:.3f}"


def test_deep_baseline_collapses():
    """Documents the failure mode identity init fixes: deep uniform-init PC sits
    near chance (error never reaches the early layers)."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 24, seed=2)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    dims = [24] * 10 + [4]
    net = TropicalPCNet(dims, gamma_shift=2, lr_shift=5, init_identity=False)
    for _ in range(8):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], n_steps=10, init="forward")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc < 0.5, f"expected deep baseline to underperform, got {acc:.3f}"


# ---------------------------------------------------------------------------
# Dyadic PC — the variant that trains deep and matches backprop
# ---------------------------------------------------------------------------
def test_dyadic_pc_trains_deep():
    """Dyadic (±2^k shift-add) PC trains a DEEP (6-edge) net — past where pure
    max-plus PC collapses. Local rule, autograd-driven settling."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 24, seed=3)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([24] * 6 + [4], gamma=0.1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(8):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_steps=30)
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.8, f"deep dyadic PC failed to learn: acc={acc:.3f}"


def test_dyadic_pc_layer_deploys_quantized_weights():
    """Closure for dyadic is a weight-VALUE property: the DEPLOYED forward must
    use ±2^k weights (shift+add), not the raw latent weights. Checking
    dyadic_quantize(w) alone is vacuous (it's a power of two by construction), so
    exercise the layer's forward: it must equal F.linear with quantized weights
    and DIFFER from F.linear with the raw (non-pow2) latent weights."""
    import torch.nn.functional as F
    from tikitorch.nn import dyadic_quantize
    net = DyadicPCNet([16, 16, 4])
    x = torch.randn(5, 16)
    layer = net.layers[0]
    deployed = layer(x)
    # the deployed map uses quantized (±2^k) weights ...
    assert torch.allclose(deployed, F.linear(x, dyadic_quantize(layer.weight), layer.bias))
    # ... and is genuinely different from the raw-weight matmul (so a regression
    # that bypassed the quantizer would be caught).
    assert not torch.allclose(deployed, F.linear(x, layer.weight, layer.bias))
    # and the deployed weights really are exact powers of two
    q = dyadic_quantize(layer.weight).abs().clamp_min(1e-12)
    log2 = torch.log2(q)
    assert torch.allclose(log2, log2.round(), atol=1e-5)


def test_dyadic_sign_product_identity_and_ternary():
    """The multiplier-free weight update: (1) operands are TERNARY ({-1,0,+1}) so
    the sign-product matmul is a sign-XOR/popcount add-tree on fabric (closure by
    operand-value, not op-trace); (2) per-sample (batch=1) the sign-product
    equals sign(autograd dE/dW) exactly — the hardware rule sign(dE/dW) =
    sign(mu-x) ⊕ sign(x)."""
    torch.manual_seed(0)
    net = DyadicPCNet([12, 12, 4], gamma=0.125)
    x0 = torch.randn(1, 12)
    y = torch.randint(0, 4, (1,))
    target = net._make_target(y, ref=x0)
    states = net.infer(x0, target, n_steps=15)

    # (1) ternary operands inside set_sign_grads
    for l, layer in enumerate(net.layers):
        pre = layer(states[l])
        interior = l < net.depth - 1
        mu = torch.relu(pre) if interior else pre
        mask = (pre > 0) if interior else torch.ones_like(pre, dtype=torch.bool)
        sa = torch.sign(mu - states[l + 1]) * mask
        sb = torch.sign(states[l])
        assert torch.equal(sa, sa.round()) and sa.abs().max() <= 1
        assert torch.equal(sb, sb.round()) and sb.abs().max() <= 1

    # (2) per-sample identity vs autograd
    net.zero_grad(set_to_none=True)
    net.energy(states).backward()
    autograd_grads = [layer.weight.grad.clone() for layer in net.layers]
    net.set_sign_grads(states)
    for g_auto, layer in zip(autograd_grads, net.layers):
        g_sign = layer.weight.grad
        nz = g_auto.abs() > 1e-8
        assert torch.equal(torch.sign(g_auto[nz]), torch.sign(g_sign[nz])), \
            "per-sample sign(dE/dW) must equal sign(mu-x)⊕sign(x)"


def test_dyadic_pc_dyadic_gradient_path_matches_well():
    """The dyadic local-gradient path trains well despite quantized operands.

    The non-dyadic gamma in this fixture intentionally leaves the complete step
    open; this test covers learning behavior, not full-recipe closure.
    """
    from tikitorch.optim import PureTropicalSGD
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=4)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16, 16, 16, 4], gamma=0.1)
    opt = PureTropicalSGD(net.parameters(), lr_shift=10, momentum_bits=0)
    for _ in range(15):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_steps=20,
                           multiplier_free=True, mf_mode="dyadic")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.85, f"dyadic multiplier-free PC underperformed: acc={acc:.3f}"


def test_dyadic_pc_residual_forward_identity():
    """The skip is exactly `max(relu(Wx), x)` on square interior layers and absent
    on the readout (the `l < depth-1` guard)."""
    net = DyadicPCNet([16, 16, 16, 4], residual=True)
    x = torch.randn(5, 16)
    pre0 = net.layers[0](x)
    assert torch.allclose(net.f(0, x), torch.maximum(torch.relu(pre0), x))   # interior: skip
    h = torch.randn(5, 16)
    assert torch.allclose(net.f(2, h), net.layers[2](h))                     # readout: no skip


def test_dyadic_pc_residual_gate_matches_autograd():
    """The gate-aware multiplier-free update must reproduce autograd's grad SIGN
    EVERYWHERE (incl. zeros) at batch=1 — this is what protects the max-gate. A
    nonzero-filtered check would miss a gate REMOVAL (autograd is 0 on rows where
    the skip won; dropping the gate makes the surrogate nonzero exactly there),
    so we compare the full sign pattern, and confirm the gate is non-trivial."""
    torch.manual_seed(0)
    net = DyadicPCNet([16, 16, 16, 4], residual=True)
    x1 = torch.randn(1, 16)
    target = net._make_target(torch.randint(0, 4, (1,)), ref=x1)
    states = net.infer(x1, target, n_steps=15)

    # gate is non-trivial: on a skip layer the x-branch wins for some units, not all
    pre0 = net.layers[0](states[0])
    gate0 = torch.relu(pre0) >= states[0]
    assert gate0.any() and (~gate0).any(), "max-gate should be non-trivial"

    net.zero_grad(set_to_none=True)
    net.energy(states).backward()
    g_auto = [layer.weight.grad.clone() for layer in net.layers]
    net.set_mf_grads(states, mode="sign")            # exact per-sample sign rule
    for ga, layer in zip(g_auto, net.layers):
        assert torch.equal(torch.sign(ga), torch.sign(layer.weight.grad)), \
            "gate-aware mf grad sign must match autograd on ALL entries"

    # residual-branch operands are ternary (closure-by-operand-value)
    mu0 = torch.maximum(torch.relu(pre0), states[0])
    mask0 = (pre0 > 0) & gate0
    sa = torch.sign((mu0 - states[1]) * mask0)
    sb = torch.sign(states[0])
    assert torch.equal(sa, sa.round()) and sa.abs().max() <= 1
    assert torch.equal(sb, sb.round()) and sb.abs().max() <= 1


def test_dyadic_pc_tropical_residual_learns():
    """Residual PC with dyadic local gradients and a closed optimizer learns.

    This is a learning smoke for the mapped recipe, not a physical-resource
    certificate or an executed-op closure assertion.
    """
    from tikitorch.optim import PureTropicalSGD
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=5)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16, 16, 16, 4], residual=True)
    opt = PureTropicalSGD(net.parameters(), lr_shift=10, momentum_bits=0)
    for _ in range(15):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_steps=20,
                           multiplier_free=True, mf_mode="dyadic")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.8, f"residual dyadic PC failed to learn: acc={acc:.3f}"


def test_dyadic_pc_cascade_matches_autograd_sign():
    """The single-pass cached-routing cascade equals autograd backprop in grad
    SIGN at batch=1 — verifying the routing math incl. the residual skip-highway
    and the Wᵀ error propagation. (Cascade error-prop uses the same ±2^k weight
    as autograd's STE, so signs match exactly.)"""
    torch.manual_seed(0)
    net = DyadicPCNet([12, 12, 12, 4], residual=True)
    x = torch.randn(1, 12)
    target = net._make_target(torch.randint(0, 4, (1,)), ref=x)
    # autograd backprop on the same L2 energy (0.5||logits - target||^2)
    net.zero_grad(set_to_none=True)
    h = x
    for l in range(net.depth):
        h = net.f(l, h)
    (0.5 * ((h - target) ** 2).sum()).backward()
    g_auto = [layer.weight.grad.clone() for layer in net.layers]
    # cascade backward (sign mode → exact per-sample sign); loss="mse" to match the
    # L2-energy autograd reference above (loss="ce" is the classification default).
    net.cascade_grads(x, target, mode="sign", loss="mse")
    for ga, layer in zip(g_auto, net.layers):
        assert torch.equal(torch.sign(ga), torch.sign(layer.weight.grad)), \
            "cascade grad sign must match autograd backprop (routing/skip-highway)"


def test_dyadic_pc_cascade_ce_matches_autograd_sign():
    """With loss="ce" (the classification default) the cascade equals autograd
    softmax-cross-entropy in grad SIGN at batch=1 — this is what lets the cascade
    match backprop at depth (MSE trails)."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    net = DyadicPCNet([12, 12, 12, 4], residual=True)
    x = torch.randn(1, 12)
    y = torch.randint(0, 4, (1,))
    target = net._make_target(y, ref=x)
    net.zero_grad(set_to_none=True)
    h = x
    for l in range(net.depth):
        h = net.f(l, h)
    F.cross_entropy(h, y).backward()                     # autograd CE on the dyadic forward
    g_auto = [layer.weight.grad.clone() for layer in net.layers]
    net.cascade_grads(x, target, mode="sign", loss="ce")
    for ga, layer in zip(g_auto, net.layers):
        assert torch.equal(torch.sign(ga), torch.sign(layer.weight.grad)), \
            "cascade(loss='ce') grad sign must match autograd cross-entropy"


def test_dyadic_pc_cascade_trains_deep():
    """The cascade trains a deep residual net without settling.

    Internal credit/gradient routing and the optimizer are shift-add oriented;
    the default CE loss remains an explicit open boundary. This is a fast
    12-edge learning smoke, not a full-recipe closure assertion.
    """
    from tikitorch.optim import PureTropicalSGD
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=6)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16] + [16] * 11 + [4], residual=True)        # 12 edges
    # The conservative 2^-10 step is stable across the supported PyTorch/BLAS
    # matrix; 2^-9 crosses hard routing boundaries differently on older stacks.
    opt = PureTropicalSGD(net.parameters(), lr_shift=10, momentum_bits=0)
    for _ in range(20):
        for i in range(0, 900 - 32, 32):
            net.train_step_cascade(Xtr[i:i + 32], Ytr[i:i + 32], opt)
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.8, f"cascade deep PC failed to learn: acc={acc:.3f}"


def test_dyadic_pc_gamma_default_is_shift_and_settles():
    """The default settling step is gamma=2^-3=0.125 (a right shift), and it
    actually descends the L2 energy (a diverging step, e.g. 0.5, would not)."""
    assert DyadicPCNet([8, 8, 4]).gamma == 0.125
    torch.manual_seed(0)
    net = DyadicPCNet([12, 12, 12, 4])                # default gamma
    x = torch.randn(8, 12)
    target = net._make_target(torch.randint(0, 4, (8,)), ref=x)
    # one long settle: final energy well below the (forward-init) start
    states = [s.detach() for s in net.feedforward(x)]
    states[-1] = target
    e_start = net.energy(states).item()
    settled = net.infer(x, target, n_steps=40)
    e_end = net.energy(settled).item()
    assert e_end < e_start, f"settling did not reduce energy: {e_start:.2f} -> {e_end:.2f}"


def test_dyadic_pc_multiplier_free_sign_learns():
    """The pure ±1 sign-product path still learns (above chance), just lags the
    dyadic mode."""
    from tikitorch.optim import PureTropicalSGD
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=4)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16, 16, 16, 4], gamma=0.1)
    opt = PureTropicalSGD(net.parameters(), lr_shift=10, momentum_bits=0)
    for _ in range(15):
        for i in range(0, 900 - 32, 32):
            net.train_step(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_steps=20,
                           multiplier_free=True, mf_mode="sign")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.5, f"sign-product multiplier-free PC did not learn: acc={acc:.3f}"


# ---------------------------------------------------------------------------
# Nudged ITERATIVE settling — the fully-local / async / distributed regime
# ---------------------------------------------------------------------------
def test_dyadic_pc_nudge_settling_reduces_augmented_objective():
    """Nudged settling lowers its augmented objective from feed-forward init."""
    torch.manual_seed(0)
    net = DyadicPCNet([12] + [12] * 4 + [4], residual=True)
    x = torch.randn(8, 12)
    target = net._make_target(torch.randint(0, 4, (8,)), ref=x)
    beta = 0.5
    ff = [s.detach() for s in net.feedforward(x)]

    def nudged_objective(states):
        target_penalty = 0.5 * beta * ((states[-1] - target) ** 2).sum()
        return net.energy(states) + target_penalty

    e0 = nudged_objective(ff).item()
    settled = net.settle_nudge(x, target, n_sweeps=20, beta=beta)
    e1 = nudged_objective(settled).item()
    assert e1 < e0, f"nudged settling did not reduce its objective: {e0:.2f} -> {e1:.2f}"


def test_dyadic_pc_nudge_applies_gamma_to_each_relaxation_update():
    """A one-edge sweep follows the documented damped fixed-point recurrence."""
    gamma, beta = 0.125, 0.5
    net = DyadicPCNet([2, 2], gamma=gamma)
    x = torch.tensor([[0.5, -1.0]])
    target = net._make_target(torch.tensor([1]), ref=x)
    initial = net.feedforward(x)[-1]

    # A depth-one sweep visits the sole free state twice (forward, then reverse).
    equilibrium = (initial + beta * target) / (1.0 + beta)
    expected = equilibrium + (1.0 - gamma) ** 2 * (initial - equilibrium)
    settled = net.settle_nudge(x, target, n_sweeps=1, beta=beta)[-1]

    torch.testing.assert_close(settled, expected)


@pytest.mark.parametrize("gamma", [True, "0.125", 0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_dyadic_pc_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        DyadicPCNet([2, 2], gamma=gamma)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_sweeps": True}, "n_sweeps"),
        ({"n_sweeps": -1}, "n_sweeps"),
        ({"n_sweeps": 1.5}, "n_sweeps"),
        ({"beta": True}, "beta"),
        ({"beta": -0.1}, "beta"),
        ({"beta": float("nan")}, "beta"),
        ({"async_frac": True}, "async_frac"),
        ({"async_frac": -0.1}, "async_frac"),
        ({"async_frac": 1.1}, "async_frac"),
        ({"async_frac": float("inf")}, "async_frac"),
    ],
)
def test_dyadic_pc_nudge_rejects_invalid_controls(kwargs, match):
    net = DyadicPCNet([2, 2])
    x = torch.tensor([[0.5, -1.0]])
    target = net._make_target(torch.tensor([1]), ref=x)
    with pytest.raises(ValueError, match=match):
        net.settle_nudge(x, target, **kwargs)


def test_dyadic_pc_nudge_trains():
    """Nudged iterative settling (no global backward pass) trains via the local
    multiplier-free gradient surrogate; Adam remains an explicit open boundary."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=7)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16] + [16] * 5 + [4], residual=True)        # 6-edge deep
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(12):
        for i in range(0, 900 - 32, 32):
            net.train_step_nudge(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_sweeps=12, beta=0.5)
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.8, f"nudged iterative PC failed to learn: acc={acc:.3f}"


def test_dyadic_pc_async_robust():
    """Trains under the distributed conditions the cascade fundamentally can't take:
    partial participation + stale (lagged) reads + random update order — no global
    backward, no global schedule."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=7)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16] + [16] * 5 + [4], residual=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(15):
        for i in range(0, 900 - 32, 32):
            net.train_step_nudge(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_sweeps=20,
                                 beta=0.5, async_frac=0.5, stale=True, rand_order=True)
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.6, f"async/stale/rand-order nudged PC failed to learn: acc={acc:.3f}"


# ---------------------------------------------------------------------------
# Error-relaxation — the async/distributed regime that works AT SCALE
# (the archived unit-step state-settling run diverged as weights grew; corrected
# gamma-damped scale results are pending. Error-relaxation is triangular/nilpotent.)
# ---------------------------------------------------------------------------
def test_dyadic_pc_error_relax_sync_matches_cascade():
    """Synchronous error-relaxation converges in one in-place pass to EXACTLY the
    cascade gradient (it is the iterative solve of the same triangular system)."""
    torch.manual_seed(0)
    net = DyadicPCNet([16] + [16] * 5 + [4], residual=True)
    x = torch.randn(8, 16)
    target = net._make_target(torch.randint(0, 4, (8,)), ref=x)
    net.cascade_grads(x, target, mode="dyadic")
    g_casc = [layer.weight.grad.clone() for layer in net.layers]
    net.error_relax_grads(x, target, mode="dyadic")           # sync default
    for gc, layer in zip(g_casc, net.layers):
        assert torch.equal(gc, layer.weight.grad), \
            "sync error-relaxation must equal the cascade gradient exactly"


def test_dyadic_pc_error_relax_trains():
    """Schedule-tolerant iterative error-relaxation learns."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=7)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16] + [16] * 5 + [4], residual=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(12):
        for i in range(0, 900 - 32, 32):
            net.train_step_relax(Xtr[i:i + 32], Ytr[i:i + 32], opt)
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.8, f"error-relaxation PC failed to learn: acc={acc:.3f}"


def test_dyadic_pc_error_relax_async_robust():
    """Error-relaxation trains under the distributed conditions: partial node
    participation + stale reads + random order, unlike the archived unit-step
    state-settling run; corrected-gamma scale behavior is not yet measured."""
    torch.manual_seed(0)
    X, Y = _blobs(300, 4, 16, seed=7)
    Xtr, Ytr, Xte, Yte = X[:900], Y[:900], X[900:], Y[900:]
    net = DyadicPCNet([16] + [16] * 5 + [4], residual=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(15):
        for i in range(0, 900 - 32, 32):
            net.train_step_relax(Xtr[i:i + 32], Ytr[i:i + 32], opt, n_iter=24,
                                 participation=0.5, stale=True, rand_order=True, gen=gen)
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    assert acc > 0.6, f"async error-relaxation failed to learn: acc={acc:.3f}"


# ---------------------------------------------------------------------------
# Generative PC — dyadic sparse coding (inference backprop can't do)
# ---------------------------------------------------------------------------
def _sparse_data(n, n_pix, n_atoms_true, k, gen, seed_atoms=0):
    """Images = sum of k random nonneg dictionary atoms (a sparse generative model)."""
    ga = torch.Generator().manual_seed(seed_atoms)
    Dtrue = torch.rand(n_pix, n_atoms_true, generator=ga)            # nonneg atoms
    X = torch.zeros(n, n_pix)
    for i in range(n):
        idx = torch.randperm(n_atoms_true, generator=gen)[:k]
        coef = torch.rand(k, generator=gen)
        X[i] = (Dtrue[:, idx] * coef).sum(1)
    return X / X.max()


def test_sparse_coding_pc_completes_better_than_mean_fill():
    """Dyadic sparse-coding PC (generative, no backprop) infers the code from
    OBSERVED pixels and in-paints the rest — recovering masked pixels far better than
    a mean-fill baseline. This is the inference capability a feed-forward net lacks."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    Xtr = _sparse_data(2000, 64, 20, 4, g)
    Xte = _sparse_data(200, 64, 20, 4, g)
    model = SparseCodingPC(n_pixels=64, n_atoms=48, n_active=6)
    for _ in range(40):
        for i in range(0, Xtr.shape[0] - 64, 64):
            model.train_step(Xtr[i:i + 64], lr_dict=0.1, n_steps=20)
    # mask 50% of pixels at random, complete from the observed half
    mask = (torch.rand(Xte.shape, generator=g) > 0.5).float()
    filled = model.complete(Xte, mask, n_steps=100)
    miss = (1.0 - mask)
    pc_mse = ((filled - Xte) ** 2 * miss).sum() / miss.sum()
    meanfill_mse = ((Xtr.mean(0) - Xte) ** 2 * miss).sum() / miss.sum()
    assert pc_mse < 0.6 * meanfill_mse, \
        f"sparse-coding PC did not complete well: {pc_mse:.4f} vs mean-fill {meanfill_mse:.4f}"


def test_sparse_coding_pc_code_is_k_sparse():
    """The inferred code is exactly k-active (top-k winner-take-all)."""
    torch.manual_seed(0)
    model = SparseCodingPC(n_pixels=64, n_atoms=48, n_active=6)
    x = torch.rand(8, 64)
    s, _ = model.infer(x, n_steps=10)
    assert ((s != 0).sum(1) <= 6).all(), "code exceeds n_active nonzeros"
