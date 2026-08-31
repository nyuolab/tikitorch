"""Bridge reverse actions (ontogenic_v2 Exp-1 item 1): pow2 norm, gate mux,
embedding gather, frozen-gate leaky scan — each checked against framework
autograd on an equivalent differentiable forward, plus the ldexp/multiply
bit-identity and route-data-only sufficiency."""
import pytest
import torch

from tikitorch.autodiff import (
    embedding_gather_vjp,
    gate_vjp,
    leaky_scan_vjp,
    pow2_norm_vjp,
)


def test_pow2_norm_vjp_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(3, 7, 16, dtype=torch.float64, requires_grad=True)
    beta = torch.randn(16, dtype=torch.float64, requires_grad=True)
    shift = torch.randint(-8, 9, (3, 7, 1))
    y = (x - x.mean(dim=-1, keepdim=True)) * torch.exp2(-shift.double()) + beta
    g = torch.randn_like(y)
    y.backward(g)
    e_in, beta_grad = pow2_norm_vjp(g, shift)
    assert torch.allclose(e_in, x.grad, rtol=0, atol=0)
    assert torch.allclose(beta_grad, beta.grad, rtol=0, atol=0)


def test_pow2_norm_ldexp_bit_identical_to_multiply():
    torch.manual_seed(1)
    e = torch.randn(4, 9, 8, dtype=torch.float32)
    shift = torch.randint(-8, 9, (4, 9, 1))
    f = torch.exp2(-shift.float())
    ec_mult = e * f
    ec_ldexp = torch.ldexp(e, -shift)
    assert torch.equal(ec_mult, ec_ldexp)


def test_pow2_norm_rejects_float_shift():
    with pytest.raises(TypeError):
        pow2_norm_vjp(torch.randn(2, 3, 4), torch.randn(2, 3, 1))


def test_pow2_norm_beta_grad_optional():
    e = torch.randn(2, 3, 4)
    _, bg = pow2_norm_vjp(e, torch.zeros(2, 3, 1, dtype=torch.int64),
                          want_beta_grad=False)
    assert bg is None


def test_gate_vjp_matches_autograd():
    torch.manual_seed(2)
    x = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    mask = (torch.rand(5, 6) > 0.4)
    y = torch.where(mask, x, torch.zeros((), dtype=torch.float64))
    g = torch.randn_like(y)
    y.backward(g)
    assert torch.equal(gate_vjp(g, mask), x.grad)


def test_gate_vjp_shape_mismatch():
    with pytest.raises(ValueError):
        gate_vjp(torch.randn(2, 3), torch.ones(3, 2))


def test_embedding_gather_vjp_matches_autograd():
    torch.manual_seed(3)
    W = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(0, 11, (4, 9))
    y = W[ids]
    g = torch.randn_like(y)
    y.backward(g)
    assert torch.allclose(embedding_gather_vjp(g, ids, 11), W.grad,
                          rtol=0, atol=0)


def test_embedding_gather_rejects_out_of_range():
    with pytest.raises(ValueError):
        embedding_gather_vjp(torch.randn(2, 4), torch.tensor([0, 9]), 5)


def test_leaky_scan_vjp_matches_autograd_when_routes_fixed():
    # Forward h_t = max(u_t, h_{t-1} - lam) with WIDE margins so no route is
    # near a tie: frozen-gate reverse must equal framework autograd exactly.
    torch.manual_seed(4)
    B, C, T = 2, 3, 6
    u = (torch.randn(B, C, T, dtype=torch.float64) * 4).requires_grad_(True)
    lam = torch.full((C,), 0.25, dtype=torch.float64, requires_grad=True)
    h = torch.zeros(B, C, dtype=torch.float64)
    hs, gates = [], []
    for t in range(T):
        cand = h - lam.unsqueeze(0)
        gates.append(cand > u[:, :, t])
        h = torch.maximum(cand, u[:, :, t])
        hs.append(h)
    H = torch.stack(hs, dim=-1)
    g = torch.randn_like(H)
    H.backward(g)
    e_u, lam_grad = leaky_scan_vjp(g, torch.stack(gates, dim=-1))
    assert torch.allclose(e_u, u.grad, rtol=0, atol=1e-15)
    assert torch.allclose(lam_grad, lam.grad, rtol=0, atol=1e-15)


def test_leaky_scan_lam_grad_is_negative_count_accumulation():
    # A pure carry chain: gate always True after t=0 -> lam_grad = -sum of
    # propagated adjoint along the carry path.
    B, C, T = 1, 1, 4
    g = torch.ones(B, C, T)
    gates = torch.tensor([[[False, True, True, True]]])
    e_u, lam_grad = leaky_scan_vjp(g, gates)
    # reverse: t3 tot=1 (carry), t2 tot=2 (carry), t1 tot=3 (carry),
    # t0 tot=4 (emit) => lam_grad = -(1+2+3) = -6, e_u=[4,0,0,0]
    assert float(lam_grad) == -6.0
    assert e_u.flatten().tolist() == [4.0, 0.0, 0.0, 0.0]


def test_leaky_scan_requires_bool_gates():
    with pytest.raises(TypeError):
        leaky_scan_vjp(torch.randn(1, 1, 3), torch.ones(1, 1, 3))


# ---- format v2: signed fields + model fingerprint (Exp-1 item 2) -----------
from tikitorch.autodiff import PackedRouteTape


def test_signed_field_round_trip():
    tape = PackedRouteTape()
    shifts = torch.randint(-8, 9, (4, 7, 1))
    f = tape.append_signed("f1_shift", shifts, lo=-8, hi=8)
    assert f.bit_width == 5 and f.value_offset == -8       # 17 codes
    assert torch.equal(tape.read("f1_shift"), shifts)
    blob = tape.to_bytes()
    back = PackedRouteTape.from_bytes(blob)
    assert torch.equal(back.read("f1_shift"), shifts)


def test_signed_field_range_rejection():
    tape = PackedRouteTape()
    with pytest.raises(ValueError):
        tape.append_signed("bad", torch.tensor([-9]), lo=-8, hi=8)


def test_v1_records_stay_byte_identical():
    a, b = PackedRouteTape(), PackedRouteTape()
    for t in (a, b):
        t.append("winners", torch.tensor([3, 1, 2]), cardinality=7)
        t.append_mask("gate", torch.tensor([1, 0, 1]))
    blob = a.to_bytes()
    assert blob == b.to_bytes()
    import json as _json
    hdr = _json.loads(blob[8:8 + int.from_bytes(blob[4:8], "little")])
    assert hdr["version"] == 1 and "value_offset" not in hdr["fields"][0]


def test_fingerprint_binding_and_refusal():
    fp = dict(arch="morphssm", d_model=128, n_layers=4, K=7, k=8,
              vocab=38, tau=0.0, tie_rule="model-produced", dtype="float32")
    tape = PackedRouteTape()
    tape.append("w", torch.tensor([1, 2]), cardinality=7)
    blob = tape.to_bytes(fingerprint=fp)
    PackedRouteTape.from_bytes(blob, expected_fingerprint=fp)   # accepts
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        PackedRouteTape.from_bytes(
            blob, expected_fingerprint=dict(fp, d_model=256))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        # unfingerprinted record refused when caller demands binding
        PackedRouteTape.from_bytes(tape.to_bytes(), expected_fingerprint=fp)


def test_signed_serialization_is_v2_and_validates_stored_domain():
    tape = PackedRouteTape()
    tape.append_signed("s", torch.tensor([-8, 0, 8]), lo=-8, hi=8)
    blob = tape.to_bytes()
    import json as _json
    hdr = _json.loads(blob[8:8 + int.from_bytes(blob[4:8], "little")])
    assert hdr["version"] == 2
    assert hdr["fields"][0]["value_offset"] == -8
    back = PackedRouteTape.from_bytes(blob)                 # revalidates
    assert back.read("s").tolist() == [-8, 0, 8]


def test_vectorized_pack_matches_loop_oracle_fuzz():
    # Exp-1 item 5: the vectorized packer must be BYTE-EXACT vs the retained
    # per-element loops across random widths/shapes/offsets (mixed fields
    # force unaligned boundaries).
    from tikitorch.autodiff.packed_tape import (
        PackedRouteTape, _read_unsigned)
    g = torch.Generator().manual_seed(99)
    for trial in range(20):
        tape = PackedRouteTape()
        specs = []
        for i in range(int(torch.randint(1, 6, (1,), generator=g))):
            w = int(torch.randint(1, 12, (1,), generator=g))
            n = int(torch.randint(1, 400, (1,), generator=g))
            vals = torch.randint(0, 2 ** w, (n,), generator=g)
            tape.append(f"f{trial}_{i}", vals, bit_width=w)
            specs.append((f"f{trial}_{i}", vals, w))
        for name, vals, w in specs:
            field = tape._fields[name]
            # oracle: per-element loop reads over the vectorized payload
            loop_vals = [
                _read_unsigned(tape._payload,
                               field.bit_offset + k * w, w)
                for k in range(field.count)]
            assert loop_vals == vals.tolist(), f"{name} mismatch"
            assert torch.equal(tape.read(name), vals)
