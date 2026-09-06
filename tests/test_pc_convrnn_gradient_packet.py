import argparse
import json

import pytest
import torch

from tikitorch.nn import TropicalTopKModule
from tikitorch.research.pc_convrnn import (
    ConvRNNLM,
    MorphPCEngine,
    RouteCertificate,
    run_one,
)


def _model():
    torch.manual_seed(7)
    model = ConvRNNLM(
        vocab=17,
        d_model=8,
        n_layers=2,
        kernel_size=3,
        dropout=0.0,
        tau=0.0,
        recurrent=True,
    )
    model.eval()
    model.set_tau(0.0)
    return model


class _TinyData:
    vocab = 17

    def batch(self, _split, batch_size, seq_len, gen):
        x = torch.randint(0, self.vocab, (batch_size, seq_len), generator=gen)
        y = torch.randint(0, self.vocab, (batch_size, seq_len), generator=gen)
        return x, y


def test_strict_run_audit_records_every_hit_and_fails_closed(tmp_path):
    args = argparse.Namespace(
        seed=0,
        device="cpu",
        d_model=8,
        n_layers=1,
        kernel=3,
        dropout=0.0,
        tau=0.0,
        lam_min=2 ** -6,
        opt="adamw",
        loss="ce",
        lr_shift=10,
        adam_lr=1e-3,
        pc_lr=1e-3,
        strict_audit=True,
        anneal=False,
        out=tmp_path,
        seq_len=6,
    )

    with pytest.raises(RuntimeError, match="strict closure audit failed"):
        run_one("convrnn", "cascade", _TinyData(), args, tmp_path / "run.jsonl")

    audit_lines = (tmp_path / "g5_audit.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in audit_lines]
    assert len(rows) == 1
    assert rows[0]["audit_semantics"] == "strict-eager-v2"
    assert rows[0]["training_closed"] is False
    assert rows[0]["violations"]
    assert rows[0]["waived"] == {}


def test_compute_grads_matches_cascade_and_does_not_step():
    model = _model()
    engine = MorphPCEngine(model)
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    y = torch.tensor([[2, 3, 4, 5, 6, 7]])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    logits_ref = engine.cascade_grads(x, y)
    grads_ref = {n: p.grad.detach().clone()
                 for n, p in model.named_parameters() if p.grad is not None}

    packet = engine.compute_grads(x, y)
    assert torch.equal(packet.logits, logits_ref)
    assert packet.grads.keys() == grads_ref.keys()
    for name in grads_ref:
        assert torch.equal(packet.grads[name], grads_ref[name]), name
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name


def test_packet_support_contains_every_nonzero_trunk_gradient():
    model = _model()
    packet = MorphPCEngine(model).compute_grads(
        torch.tensor([[1, 1, 2, 3, 5, 8]]),
        torch.tensor([[1, 2, 3, 5, 8, 13]]),
    )
    assert set(packet.route_support) == {
        "layers.0.morph_conv.weight",
        "layers.0.out_proj.weight",
        "layers.1.morph_conv.weight",
        "layers.1.out_proj.weight",
    }
    for name, selected in packet.route_support.items():
        assert selected.dtype == torch.bool
        assert selected.shape == packet.grads[name].shape
        assert torch.all(selected | (packet.grads[name] == 0)), name


def test_target_free_certificate_has_exact_counts_and_no_hidden_mutation():
    model = _model()
    engine = MorphPCEngine(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor([[1, 1, 2, 3, 5, 8], [13, 8, 5, 3, 2, 1]])
    y = torch.tensor([[1, 2, 3, 5, 8, 13], [8, 5, 3, 2, 1, 0]])
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7.0)
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    grads_before = {
        name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer_before = optimizer.state_dict()
    rng_before = torch.random.get_rng_state().clone()

    certificate = engine.route_certificate(x)

    assert isinstance(certificate, RouteCertificate)
    assert certificate.route_counts is not None
    assert certificate.route_counts.keys() == certificate.route_support.keys()
    for name, selected in certificate.route_support.items():
        count = certificate.route_counts[name]
        assert count.dtype == torch.int64
        assert count.shape == selected.shape
        assert torch.equal(selected, count > 0), name
        assert int(count.sum()) == x.shape[0] * x.shape[1] * model.embed.embedding_dim
        assert int(count.max()) > 1

    _, caches, _ = engine._forward_cache(x)
    for layer_index, (layer, cache) in enumerate(zip(model.layers, caches)):
        conv_route = cache["conv_route"]
        expected_conv = torch.stack(
            [
                (conv_route == kernel_index).sum(dim=(0, 2), dtype=torch.int64)
                for kernel_index in range(layer.morph_conv.kernel_size)
            ],
            dim=1,
        )
        conv_name = f"layers.{layer_index}.morph_conv.weight"
        assert torch.equal(certificate.route_counts[conv_name], expected_conv)

        proj_route = cache["proj_route"]
        out_features, in_features = layer.out_proj.weight.shape
        expected_proj = torch.zeros(
            out_features, in_features, dtype=torch.int64, device=proj_route.device
        )
        expected_proj.scatter_add_(
            1, proj_route.transpose(0, 1), torch.ones_like(proj_route).transpose(0, 1)
        )
        proj_name = f"layers.{layer_index}.out_proj.weight"
        assert torch.equal(certificate.route_counts[proj_name], expected_proj)
    assert all(
        torch.equal(parameter, parameters_before[name])
        for name, parameter in model.named_parameters()
    )
    assert all(
        torch.equal(parameter.grad, grads_before[name])
        for name, parameter in model.named_parameters()
    )
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    packet = engine.compute_grads(x, y)
    assert packet.route_counts is not None
    assert torch.equal(packet.logits, certificate.logits)
    for name in certificate.route_support:
        assert torch.equal(packet.route_support[name], certificate.route_support[name])
        assert torch.equal(packet.route_counts[name], certificate.route_counts[name])


def test_target_free_certificate_consumes_only_normal_forward_rng():
    torch.manual_seed(11)
    model = ConvRNNLM(
        vocab=17,
        d_model=8,
        n_layers=2,
        kernel_size=3,
        dropout=0.25,
        tau=0.0,
        recurrent=True,
    )
    model.train()
    model.set_tau(0.0)
    engine = MorphPCEngine(model)
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    rng_before = torch.random.get_rng_state().clone()

    certificate = engine.route_certificate(x)
    rng_after_certificate = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng_before)
    logits, _, _ = engine._forward_cache(x)
    rng_after_forward = torch.random.get_rng_state().clone()

    assert torch.equal(certificate.logits, logits)
    assert torch.equal(rng_after_certificate, rng_after_forward)


def test_certificate_action_logits_match_existing_acting_probe_in_eval():
    model = _model()
    model.action_head = TropicalTopKModule(
        model.embed.embedding_dim, out_features=5, k=3
    )
    model.eval()
    engine = MorphPCEngine(model)
    x = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    rng_before = torch.random.get_rng_state().clone()

    certificate = engine.route_certificate(x)
    rng_after_certificate = torch.random.get_rng_state().clone()
    probe_logits = engine.act_probe_logits(x)
    rng_after_probe = torch.random.get_rng_state().clone()

    assert certificate.action_logits is not None
    assert certificate.action_logits.shape == (2, 6, 5)
    assert torch.equal(certificate.action_logits, probe_logits)
    assert torch.equal(rng_before, rng_after_certificate)
    assert torch.equal(rng_before, rng_after_probe)

    legacy_construction = RouteCertificate(
        logits=certificate.logits,
        route_support=certificate.route_support,
    )
    assert legacy_construction.route_counts is None
    assert legacy_construction.action_logits is None


def test_soft_route_certificate_uses_legacy_support_without_integer_counts():
    model = _model()
    model.set_tau(0.25)
    certificate = MorphPCEngine(model).route_certificate(
        torch.tensor([[1, 2, 3, 4, 5, 6]])
    )
    assert certificate.route_counts is None
    assert certificate.route_support
    assert all(mask.dtype == torch.bool for mask in certificate.route_support.values())


def test_compute_grads_replaces_stale_gradients_by_default():
    model = _model()
    for p in model.parameters():
        p.grad = torch.full_like(p, 123.0)
    packet = MorphPCEngine(model).compute_grads(
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[1, 2, 3, 4]]),
    )
    for name, param in model.named_parameters():
        if name in packet.grads:
            assert torch.equal(param.grad, packet.grads[name])
            assert not torch.all(param.grad == 123.0)


def test_install_grads_is_validated_and_does_not_step():
    model = _model()
    engine = MorphPCEngine(model)
    packet = engine.compute_grads(
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[1, 2, 3, 4]]),
    )
    before = {name: parameter.detach().clone()
              for name, parameter in model.named_parameters()}
    doubled = {name: gradient * 2 for name, gradient in packet.grads.items()}
    engine.install_grads(doubled)
    assert all(torch.equal(parameter.grad, doubled[name])
               for name, parameter in model.named_parameters())
    assert all(torch.equal(parameter, before[name])
               for name, parameter in model.named_parameters())

    incomplete = dict(doubled)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(KeyError, match="incomplete"):
        engine.install_grads(incomplete)

    bad = dict(doubled)
    name = next(iter(bad))
    bad[name] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        engine.install_grads(bad)
