"""The historical LM runner must not bypass the canonical fail-closed auditor."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tikitorch.models
from tikitorch.research import nanogpt


class ProbeData:
    def __init__(self):
        self.requested = None

    def batch(self, split, batch, seq_len, gen):
        self.requested = (split, batch, seq_len)
        return torch.ones(batch, seq_len), torch.zeros(batch, seq_len, dtype=torch.long)


class ProbeModel(nn.Module):
    def __init__(self, *, tau=0.0, operation="add"):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.register_buffer("counter", torch.tensor(0))
        self.tau = tau
        self.operation = operation

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        self.counter.add_(1)
        # Deliberately consume RNG to exercise audit rollback, without arithmetic.
        torch.rand(())
        value = x + self.weight
        if self.tau > 0:
            value = value.exp()
        if self.training:
            if self.operation == "divide":
                value = value / 3.0
            elif self.operation == "transcendental":
                value = value.exp()
            elif self.operation == "unknown":
                # Keep the unknown operation separate from its backward, so
                # this is inconclusive rather than a known log violation too.
                torch.special.entr(value.detach())
        return torch.stack((value, -value), dim=-1)


def arguments(**overrides):
    settings = dict(batch_size=2, seq_len=4, device="cpu", tau=0.0, tau_floor=0.0,
                    anneal=False, sign_lr=0.25)
    settings.update(overrides)
    return SimpleNamespace(**settings)


@pytest.fixture
def sum_loss(monkeypatch):
    # Isolate runner policy from the implementation of a particular loss.
    monkeypatch.setattr(tikitorch.models, "classification_loss",
                        lambda logits, targets, loss_type: logits[:, 0].sum())


def test_strict_gate_returns_versioned_scoped_reports_and_restores_state(
        monkeypatch, capsys, sum_loss):
    model = ProbeModel().eval()
    model.weight.grad = torch.tensor(7.0)
    original_grad = model.weight.grad
    optimizer = nanogpt.make_optimizer("signsgd0", model, arguments())
    optimizer.state[model.weight]["sentinel"] = torch.tensor(19)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    monkeypatch.setattr(nanogpt, "make_optimizer", lambda *args: optimizer)
    parameters_before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    data = ProbeData()

    reports = nanogpt.strict_gate(model, data, arguments(), "signsgd0", "perceptron")

    assert data.requested == ("train", 2, 4)
    assert reports["deployment_candidate"]["audit_semantics"] == "strict-eager-v3"
    assert reports["current_training"]["schema_version"] == "tikitorch.closure-report.v2"
    coverage = reports["current_training"]["coverage"]["training"]
    assert coverage["certified"] is True
    assert coverage["harden_tau"] is False
    assert coverage["devices"] == ["cpu"]
    assert model.training is False
    assert model.tau == 0.0
    assert model.weight.grad is original_grad
    assert model.weight.grad.item() == 7.0
    for name, value in parameters_before.items():
        assert torch.equal(model.state_dict()[name], value)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert optimizer.state_dict() == optimizer_before
    output = capsys.readouterr().out
    assert "deployment-candidate: closed" in output
    assert "current-training: closed" in output
    assert "physical zero-DSP behavior are not certified" in output
    assert "training-closure OK" not in output


@pytest.mark.parametrize(
    "operation, category, status",
    [("divide", "elementwise-divide", "violated"),
     ("transcendental", "transcendental", "violated"),
     ("unknown", "unknown-operation", "inconclusive")],
)
def test_strict_gate_never_waives_training_blockers(
        operation, category, status, capsys, sum_loss):
    model = ProbeModel(operation=operation)
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(SystemExit, match=category):
        nanogpt.strict_gate(model, ProbeData(), arguments(), "signsgd0", "tropical_ce_lut")
    assert model.training is True
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value)
    assert torch.equal(torch.get_rng_state(), rng_before)
    output = capsys.readouterr().out
    assert "deployment-candidate: closed" in output
    assert f"current-training: {status}" in output
    assert "supplied eager step certified" not in output


def test_strict_gate_does_not_harden_soft_training(capsys, sum_loss):
    model = ProbeModel(tau=0.5)
    with pytest.raises(SystemExit, match="transcendental"):
        nanogpt.strict_gate(model, ProbeData(), arguments(tau=0.5, anneal=True),
                            "signsgd0", "perceptron")
    assert model.tau == 0.5
    output = capsys.readouterr().out
    assert "deployment-candidate: closed" in output
    assert "current-training: violated" in output


def test_strict_gate_rejects_anneal_even_when_initial_step_is_closed(sum_loss):
    with pytest.raises(SystemExit, match="supports only --tau 0 --no-anneal"):
        nanogpt.strict_gate(ProbeModel(), ProbeData(), arguments(anneal=True),
                            "signsgd0", "perceptron")


@pytest.mark.parametrize("floor", [0.25, -0.25])
def test_strict_gate_rejects_nonzero_temperature_floor(floor, sum_loss):
    data = ProbeData()
    with pytest.raises(SystemExit, match="requires --tau-floor 0"):
        nanogpt.strict_gate(ProbeModel(), data, arguments(tau_floor=floor),
                            "signsgd0", "perceptron")
    assert data.requested is None


def test_hard_morphssm_builder_and_runner_reach_real_strict_gate(monkeypatch, tmp_path):
    class TokenData(ProbeData):
        vocab = 8

        def batch(self, split, batch, seq_len, gen):
            x, y = super().batch(split, batch, seq_len, gen)
            return x.long(), y

    args = arguments(
        strict=True, seed=0, variant="morphssm", morph_d=8, morph_layers=1,
        morph_multi_scale=False, morph_group_size=1, dropout=0.0,
        opt="signsgd0", loss="perceptron", steps=1,
    )
    data = TokenData()
    built = nanogpt.build_model("tropical-closed", data, args)
    assert all(m.tau == 0.0 for m in built.modules() if hasattr(m, "tau"))

    real_gate = nanogpt.strict_gate
    observed = {}

    class GateComplete(Exception):
        pass

    def stop_after_real_gate(model, *gate_args):
        assert all(m.tau == 0.0 for m in model.modules() if hasattr(m, "tau"))
        observed.update(real_gate(model, *gate_args))
        raise GateComplete

    monkeypatch.setattr(nanogpt, "strict_gate", stop_after_real_gate)
    with pytest.raises(GateComplete):
        nanogpt.train_run("tropical-closed", data, args, tmp_path / "unused.jsonl")
    assert observed["current_training"]["training_closed"] is True
    assert not (tmp_path / "unused.jsonl").exists()


@pytest.mark.parametrize("tau, floor", [(0.5, 0.0), (0.0, 0.25)])
def test_runner_initializes_requested_soft_temperature_before_gate(
        tau, floor, monkeypatch, tmp_path):
    args = arguments(
        strict=True, seed=0, opt="signsgd0", loss="perceptron", steps=1,
        tau=tau, tau_floor=floor,
    )
    model = ProbeModel()
    monkeypatch.setattr(nanogpt, "build_model", lambda *args: model)

    class GateReached(Exception):
        pass

    def stop_at_gate(observed_model, *gate_args):
        assert observed_model.tau == max(tau, floor)
        raise GateReached

    monkeypatch.setattr(nanogpt, "strict_gate", stop_at_gate)
    with pytest.raises(GateReached):
        nanogpt.train_run("tropical-closed", ProbeData(), args, tmp_path / "unused.jsonl")


def test_strict_gate_uses_actual_batch_extent_for_mean_reduction():
    data = ProbeData()
    with pytest.raises(SystemExit, match="elementwise-divide"):
        nanogpt.strict_gate(ProbeModel(), data, arguments(batch_size=3, seq_len=3),
                            "signsgd0", "perceptron")
    assert data.requested == ("train", 3, 3)


def test_strict_gate_rejects_non_dyadic_optimizer_step(sum_loss):
    with pytest.raises(SystemExit, match="scalar-multiply"):
        nanogpt.strict_gate(ProbeModel(), ProbeData(), arguments(sign_lr=0.1),
                            "signsgd0", "perceptron")


def test_strict_gate_does_not_certify_untraced_accumulation_window(sum_loss):
    with pytest.raises(SystemExit, match="accumulate/step/reset window"):
        nanogpt.strict_gate(ProbeModel(), ProbeData(), arguments(),
                            "countagree", "perceptron")


@pytest.mark.parametrize("kind", ["dense", "dense-conv"])
def test_strict_flag_is_not_silently_ignored_for_dense_controls(kind, tmp_path):
    with pytest.raises(SystemExit, match="cannot certify a dense control"):
        nanogpt.train_run(kind, ProbeData(), arguments(strict=True), tmp_path / "run.jsonl")
