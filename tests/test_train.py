"""Tests for tikitorch.train.Trainer — the closure-aware training loop.

The Trainer owns the two things the closure claim hangs on: driving the tropical
relaxation ``tau`` to exactly 0 at deploy, and gating that deploy on
``assert_closed`` / ``assert_closed_runtime``. These tests verify (a) it trains a
tiny closed TropicalMLP and the loss drops, (b) the tau anneal actually reaches 0
at the deploy endpoint, (c) the closure gate PASSES for a closed model and (d)
RAISES for one whose head is a standard ``nn.Linear``. Guarded behind
``requires_train`` so the suite stays green until the module lands.
"""
import pytest
import torch
import torch.nn as tnn

import tikitorch as tt
import tikitorch.optim as optim
from tikitorch.train import Trainer, _tau_at


def _attr_present(module_name, attr):
    try:
        mod = __import__(module_name, fromlist=[attr])
    except Exception:
        return False
    return hasattr(mod, attr)


requires_train = pytest.mark.skipif(
    not _attr_present("tikitorch.train", "Trainer"),
    reason="tikitorch.train.Trainer not implemented yet",
)
requires_model = pytest.mark.skipif(
    not _attr_present("tikitorch.models", "TropicalMLP"),
    reason="tikitorch.models.TropicalMLP not implemented yet",
)


# ---------------------------------------------------------------------------
# Toy data + models
# ---------------------------------------------------------------------------

def _toy_classification(n=480, d=12, n_cls=3, seed=0):
    """A max-plus-friendly toy: each class is a distinct hot-feature prototype.

    ``class c`` puts its mass on feature ``c`` (plus a little noise), so the label
    is recoverable by a single max-plus comparison -- the kind of separation a
    tropical (compare/add) MLP head can actually fit, unlike a dense linear
    boundary. (The closed TropicalMLP's add-only scalar projection is
    rank-degenerate on arbitrary linear tasks; this is the geometry it solves.)
    """
    g = torch.Generator().manual_seed(seed)
    proto = torch.eye(n_cls, d)                        # class c -> feature c hot
    y = torch.randint(0, n_cls, (n,), generator=g)
    x = proto[y] + 0.05 * torch.rand(n, d, generator=g)
    return x, y


def _closed_mlp(d=12, n_cls=3):
    # one tropical layer + tropical top-k head: tiny, and it cleanly fits the
    # prototype toy (the 2-layer alternating min/max composition stalls on it).
    return tt.models.TropicalMLP(d_model=32, n_layers=1, input_dim=d,
                                 num_classes=n_cls, dropout=0.0)


class _LinearHeadModel(tnn.Module):
    """A model with a standard nn.Linear head — intentionally NOT closed.

    It exposes ``compute_loss`` and a ``set_tau`` no-op so the Trainer drives it
    like any tropical model, but the ``nn.Linear`` head means ``assert_closed``
    must flag it at the deploy gate.
    """

    def __init__(self, d=12, n_cls=3):
        super().__init__()
        self.head = tnn.Linear(d, n_cls)              # the DSP boundary

    def set_tau(self, tau):                           # accept the anneal, no-op
        pass

    def forward(self, x):
        return self.head(x)

    def compute_loss(self, x, y):
        return tnn.functional.cross_entropy(self.forward(x), y)


# ---------------------------------------------------------------------------
# tau schedule
# ---------------------------------------------------------------------------

@requires_train
def test_tau_schedule_anneals_to_floor():
    # During training tau ramps from tau_max to the floor tau_min (NOT exactly 0 --
    # sign-only optimizers collapse on the fully-hard landscape); the exact tau=0
    # deploy graph is pinned by Trainer.fit after the loop (tested separately).
    steps = 50
    taus = [_tau_at(s, steps, schedule="linear", tau_max=1.0, tau_min=0.0,
                    hard_frac=0.8) for s in range(steps)]
    assert taus[0] == pytest.approx(1.0)             # starts soft
    # monotone non-increasing for the linear schedule
    assert all(b <= a + 1e-9 for a, b in zip(taus, taus[1:]))
    # the tail (last 20%) sits at the floor (here tau_min=0.0)
    assert all(t == pytest.approx(0.0) for t in taus[int(0.8 * (steps - 1)) + 1:])
    # with a positive floor the tail holds at exactly that floor
    floored = [_tau_at(s, steps, schedule="linear", tau_max=1.0, tau_min=0.05,
                       hard_frac=0.8) for s in range(steps)]
    assert floored[-1] == pytest.approx(0.05)


@requires_train
def test_tau_schedule_variants_ramp_down():
    for sched in ("linear", "cosine", "constant"):
        t0 = _tau_at(0, 40, schedule=sched, tau_max=1.0, tau_min=0.0, hard_frac=0.8)
        tN = _tau_at(39, 40, schedule=sched, tau_max=1.0, tau_min=0.0, hard_frac=0.8)
        assert tN == pytest.approx(0.0)              # floored to tau_min
        assert t0 > 0.0


@requires_train
def test_tau_schedule_rejects_unknown():
    with pytest.raises(ValueError, match="tau_schedule"):
        _tau_at(0, 10, schedule="bogus", tau_max=1.0, tau_min=0.0, hard_frac=0.8)


# ---------------------------------------------------------------------------
# Training: loss decreases on a closed TropicalMLP
# ---------------------------------------------------------------------------

@requires_train
@requires_model
def test_trainer_reduces_loss_on_closed_mlp():
    torch.manual_seed(0)
    X, y = _toy_classification()
    model = _closed_mlp()
    opt = optim.SignSGD(model.parameters(), lr=3e-2, momentum=0.9)
    g = torch.Generator().manual_seed(0)
    trainer = Trainer(model, opt, steps=400, batch_size=64, generator=g)
    trainer.fit((X, y))
    # loss trace recorded, and the late loss is well below the early loss
    assert len(trainer.history) == 400
    early = sum(trainer.history[:20]) / 20
    late = sum(trainer.history[-40:]) / 40           # avg over the hard-tau tail
    assert late < early * 0.75, f"loss did not drop: early={early:.3f} late={late:.3f}"
    assert min(trainer.history) < early * 0.7


@requires_train
@requires_model
def test_trainer_evaluate_learns_above_chance():
    torch.manual_seed(0)
    X, y = _toy_classification()
    Xte, yte = _toy_classification(n=256, seed=99)
    model = _closed_mlp()
    opt = optim.SignSGD(model.parameters(), lr=3e-2, momentum=0.9)
    g = torch.Generator().manual_seed(0)
    trainer = Trainer(model, opt, steps=400, batch_size=64, generator=g)
    trainer.fit((X, y))
    res = trainer.evaluate((Xte, yte))
    assert "loss" in res and "accuracy" in res
    assert res["accuracy"] > 0.6                      # chance = 1/3


@requires_train
@requires_model
def test_trainer_accepts_iterable_of_batches():
    torch.manual_seed(0)
    X, y = _toy_classification(n=256)
    batches = [(X[i:i + 64], y[i:i + 64]) for i in range(0, 256, 64)]
    model = _closed_mlp()
    opt = optim.SignSGD(model.parameters(), lr=2e-2, momentum=0.9)
    trainer = Trainer(model, opt, steps=40)
    trainer.fit(batches)                              # cycles the 4 batches
    assert len(trainer.history) == 40


@requires_train
@requires_model
def test_trainer_loss_fn_path_when_no_compute_loss():
    # A model without compute_loss must train through a provided loss_fn.
    torch.manual_seed(0)
    X, y = _toy_classification()

    class _NoLossMLP(tt.models.TropicalMLP):
        compute_loss = None                          # hide compute_loss

    model = _NoLossMLP(d_model=32, n_layers=1, input_dim=12, num_classes=3,
                       dropout=0.0)
    opt = optim.SignSGD(model.parameters(), lr=3e-2, momentum=0.9)

    def loss_fn(pred, target):
        return tnn.functional.cross_entropy(pred, target)

    g = torch.Generator().manual_seed(0)
    trainer = Trainer(model, opt, steps=400, batch_size=64, loss_fn=loss_fn,
                      generator=g)
    trainer.fit((X, y))
    early = sum(trainer.history[:20]) / 20
    assert min(trainer.history) < early * 0.75


# ---------------------------------------------------------------------------
# The closure-on-deploy gate
# ---------------------------------------------------------------------------

@requires_train
@requires_model
def test_deploy_gate_passes_for_closed_model():
    torch.manual_seed(0)
    X, y = _toy_classification()
    model = _closed_mlp()
    opt = optim.SignSGD(model.parameters(), lr=2e-2, momentum=0.9)
    trainer = Trainer(model, opt, steps=30, batch_size=64,
                      assert_closed_on_deploy=True)
    # must NOT raise: a closed model passes the structural + runtime gate.
    trainer.fit((X, y))


@requires_train
def test_deploy_gate_raises_for_linear_head():
    torch.manual_seed(0)
    X, y = _toy_classification()
    model = _LinearHeadModel(d=12, n_cls=3)
    opt = optim.SignSGD(model.parameters(), lr=2e-2)
    trainer = Trainer(model, opt, steps=20, batch_size=64,
                      assert_closed_on_deploy=True)
    with pytest.raises(AssertionError, match="multiplier-free|Linear"):
        trainer.fit((X, y))


@requires_train
def test_deploy_gate_can_be_disabled():
    # With the gate off, an unclosed model trains without raising (caller's risk).
    torch.manual_seed(0)
    X, y = _toy_classification()
    model = _LinearHeadModel(d=12, n_cls=3)
    opt = optim.SignSGD(model.parameters(), lr=2e-2)
    trainer = Trainer(model, opt, steps=20, batch_size=64,
                      assert_closed_on_deploy=False)
    trainer.fit((X, y))                              # no raise


@requires_train
def test_deploy_gate_allow_declares_dsp_boundary():
    # allow={'head'} declares the Linear an INTENTIONAL DSP boundary -> no raise.
    torch.manual_seed(0)
    X, y = _toy_classification()
    model = _LinearHeadModel(d=12, n_cls=3)
    opt = optim.SignSGD(model.parameters(), lr=2e-2)
    trainer = Trainer(model, opt, steps=20, batch_size=64,
                      assert_closed_on_deploy=True, closed_allow={"head"})
    trainer.fit((X, y))                              # no raise


@requires_train
@requires_model
def test_model_left_in_eval_at_hard_tau():
    # After fit, the model is in eval() AND pinned to the hard tau=0 deploy graph.
    torch.manual_seed(0)
    X, y = _toy_classification()
    model = _closed_mlp()
    opt = optim.SignSGD(model.parameters(), lr=2e-2)
    trainer = Trainer(model, opt, steps=20, batch_size=64)
    trainer.fit((X, y))
    assert not model.training
    # every tropical *submodule* is hardened: tau==0 and the soft relaxation off.
    # (The model-level use_softmin flag is just the constructor record; the layers
    # are what define the deploy graph, and those are what _set_tau drives.)
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "tau"):
            assert m.tau == 0.0
        if hasattr(m, "use_softmin"):
            assert m.use_softmin is False
