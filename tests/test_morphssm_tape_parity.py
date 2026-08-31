"""Exp-1 item 4 — the ACTUAL-MODEL bit-parity gate.

Three-way parameter-gradient equality on a real MorphSSM at tau=0:
  framework autograd  ≡  MorphPCEngine.cascade_grads  ≡  replay(pack(cache))
Autograd equality holds when no route sits at a tie (generic random inputs);
the exact-tie fixture pins the DOCUMENTED property that replay ≡ cascade
always (both consume the recorded decisions) even where autograd's
subgradient choice may differ. Fingerprint refusal is exercised end-to-end.
"""
import argparse

import pytest
import torch

from tikitorch.research.pc_convrnn import MorphPCEngine, build_model

from tikitorch.autodiff.morphssm_tape import (
    deserialize,
    model_fingerprint,
    pack_engine_cache,
    replay,
    serialize,
)

D, L, V, T, B = 16, 2, 23, 40, 2


def make_model(seed=0, dropout=0.0, dtype=torch.float64):
    torch.manual_seed(seed)
    ns = argparse.Namespace(d_model=D, n_layers=L, kernel=7, dropout=dropout,
                            tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", V, ns)
    model.set_tau(0.0)
    model = model.to(dtype)
    model.eval()
    return model


def batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    targets = torch.randint(0, V, (B, T), generator=g)
    return ids, targets


def autograd_grads(model, ids, targets):
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V).float(), targets.reshape(-1))
    loss.backward()
    out = {n: p.grad.detach().clone() for n, p in model.named_parameters()
           if p.grad is not None}
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return out


def cascade_grads_dict(model, ids, targets):
    engine = MorphPCEngine(model)
    for p in model.parameters():
        p.grad = None
    engine.cascade_grads(ids, targets)
    out = {n: p.grad.detach().clone() for n, p in model.named_parameters()
           if p.grad is not None}
    for p in model.parameters():
        p.grad = None
    return out


def replay_grads_dict(model, ids, targets):
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    tape = pack_engine_cache(caches, top, model, ids)
    blob = serialize(tape, model)
    back = deserialize(blob, model)
    return replay(back, model, logits, targets), tape


def test_three_way_parity_generic_inputs():
    model = make_model()
    ids, targets = batch()
    ag = autograd_grads(model, ids, targets)
    cg = cascade_grads_dict(model, ids, targets)
    rg, _ = replay_grads_dict(model, ids, targets)
    assert set(cg) == set(rg)
    for n in cg:
        # replay ≡ cascade: BIT-EXACT always (same decisions, same kernels)
        assert torch.equal(cg[n], rg[n]), f"replay!=cascade at {n}"
    for n in ag:
        # autograd ≡ cascade away from ties, at the honest tolerance:
        # `_top_error` computes the softmax at FLOAT32 by design (the
        # declared host boundary, pc_convrnn:503) then sweeps in the model
        # dtype, while autograd backpropagates THROUGH the fp32 cast —
        # two correct computations that differ at fp32-epsilon scale
        # (per-element ~1.5e-9, compounding over R=80-position sums to
        # ~1e-8 on beta/embed accumulations; measured max 1.04e-8). Documented
        # in AUDIT_FINDINGS; the tight 1e-14 convention applies only to
        # full-precision-matched pipelines.
        assert torch.allclose(ag[n], cg[n], rtol=1e-6, atol=1e-7), \
            f"autograd!=cascade at {n}: " \
            f"{(ag[n] - cg[n]).abs().max().item():.3e}"


def test_exact_tie_fixture_replay_still_equals_cascade():
    # Zero embeddings + zero weights => ties EVERYWHERE. The documented
    # property: replay ≡ cascade regardless (recorded decisions), while
    # autograd's tie subgradients are a separate category (not asserted).
    model = make_model(seed=7)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    ids, targets = batch(seed=2)
    cg = cascade_grads_dict(model, ids, targets)
    rg, _ = replay_grads_dict(model, ids, targets)
    for n in cg:
        assert torch.equal(cg[n], rg[n]), f"tie divergence at {n}"


def test_dropout_train_mode_masks_packed():
    model = make_model(seed=3, dropout=0.3)
    model.train()
    ids, targets = batch(seed=4)
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    if caches[0]["drop_mask"] is None and top["mask0"] is None:
        pytest.skip("model config produced no dropout masks")
    tape = pack_engine_cache(caches, top, model, ids)
    names = {f.name for f in tape.fields}
    assert any(n.endswith("drop_gate") for n in names)


def test_fingerprint_refuses_mismatched_model():
    model = make_model()
    ids, targets = batch()
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    blob = serialize(pack_engine_cache(caches, top, model, ids), model)
    other = make_model(seed=9)
    fp = model_fingerprint(other)
    fp["d_model"] = D * 2
    from tikitorch.autodiff import PackedRouteTape
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        PackedRouteTape.from_bytes(blob, expected_fingerprint=fp)


def test_soft_phase_refused():
    model = make_model()
    model.set_tau(0.5)
    ids, _ = batch()
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    with pytest.raises((AssertionError, ValueError)):
        pack_engine_cache(caches, top, model, ids)


def test_tape_bytes_match_audit_accounting():
    # The saved-state claim's own arithmetic at this small config:
    # per layer: conv B*D*T*3b + proj (B*T)*D*7b(bits for D=16 -> 4b)
    # ... assert only the MEASURED payload equals the field-table sum.
    model = make_model()
    ids, targets = batch()
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    tape = pack_engine_cache(caches, top, model, ids)
    total_bits = sum(f.bit_length for f in tape.fields)
    assert tape.bit_length == total_bits
    assert len(tape.payload()) == (total_bits + 7) // 8
