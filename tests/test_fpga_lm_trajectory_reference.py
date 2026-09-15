"""Synthetic pure-forward/reverse checks; no corpus or trajectory experiment."""
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from manuscript.tools import fpga_lm_float_reference as old  # noqa: E402
from manuscript.tools import fpga_lm_numeric_diagnostic as numeric  # noqa: E402
from manuscript.tools import fpga_lm_trajectory_reference as ref  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def parameters(d=4, v=3, n=4, k=7, integer=False, seed=91):
    gen = torch.Generator().manual_seed(seed)
    shapes = {"EMBED": (v, d), "MH": (n, d, k), "MP": (n, d, d),
              "MB": (n, d), "W_HEAD": (v, d)}
    return {name: torch.randint(-8, 9, shape, generator=gen).double() / (1 if integer else 8)
            for name, shape in shapes.items()}


def scalar_forward(weights, tokens, use_context):
    """Independent Python list/scalar forward, including explicit first ties."""
    p = {key: value.tolist() for key, value in weights.items()}
    d, vocab, n, k = len(p["EMBED"][0]), len(p["EMBED"]), len(p["MH"]), len(p["MH"][0][0])
    length = len(tokens)
    history = [[None] * length for _ in range(n)]
    taps, projection = ([[None] * length for _ in range(n)] for _ in range(2))
    logits, head = [], []
    for position, token in enumerate(tokens):
        residual = list(p["EMBED"][token])
        for layer in range(n):
            center = math.floor(sum(residual) / d)
            history[layer][position] = [value - center for value in residual]
            conv, tap_row = [], []
            for feature in range(d):
                candidates = [(tap, history[layer][position - (tap << min(layer, 6))][feature]
                               + p["MH"][layer][feature][tap]) for tap in range(k if use_context else 1)
                              if position >= (tap << min(layer, 6))]
                best = (max if layer % 2 == 0 else min)(value for tap, value in candidates)
                tap_row.append(next(tap for tap, value in candidates if value == best))
                conv.append(best)
            taps[layer][position] = tap_row
            projected, proj_row = [], []
            for output in range(d):
                values = [conv[feature] + p["MP"][layer][output][feature] for feature in range(d)]
                best = max(values)
                proj_row.append(values.index(best))
                projected.append(best + p["MB"][layer][output])
            projection[layer][position] = proj_row
            residual = [a + b for a, b in zip(residual, projected)]
        scores, routes = [], []
        for target in range(vocab):
            values = [residual[feature] + p["W_HEAD"][target][feature] for feature in range(d)]
            best = max(values)
            scores.append(best)
            routes.append(values.index(best))
        logits.append(scores)
        head.append(routes)
    return logits, {"AD_HEAD": head, "PROJ_IDX": projection, "TAP_IDX": taps}


@pytest.mark.parametrize("d,n,k,length", [(1, 1, 1, 1), (2, 2, 3, 7), (4, 4, 7, 13), (8, 8, 2, 17)])
@pytest.mark.parametrize("use_context", [True, False])
def test_fractional_forward_exact_scalar_and_chunk_parity(d, n, k, length, use_context):
    p = parameters(d=d, n=n, k=k)
    tokens = torch.tensor([(i * 2 + i // 3) % 3 for i in range(length)])
    expected, routes = scalar_forward(p, tokens.tolist(), use_context)
    for chunk in (1, 3, 64):
        result = ref.forward(p, tokens, use_context=use_context, chunk_size=chunk)
        assert result["logits"].tolist() == expected
        assert {key: value.tolist() for key, value in result["routes"].items()} == routes
        assert result["mode"] == "continuous_floor" and result["schema"] != old.SCHEMA
        assert not result["logits"].requires_grad and result["native_execution"] is False


@pytest.mark.parametrize("d,length", [(2, 5), (16, 13), (512, 2)])
@pytest.mark.parametrize("use_context", [True, False])
def test_integer_state_matches_old_forward_all_routes_and_replay(d, length, use_context):
    p = parameters(d=d, v=128 if d == 512 else 3, integer=True)
    tokens = torch.tensor([i % 3 for i in range(length)])
    actual = ref.forward(p, tokens, use_context=use_context)
    expected = old.forward(p, tokens, mode="integer_anchor", use_context=use_context)
    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["centering_offsets"], expected["centering_offsets"])
    for key in actual["routes"]:
        assert torch.equal(actual["routes"][key], expected["routes"][key])
    tape = ref.reference_base(actual)
    assert tape["mode"] == "fixed_routes"
    assert tape["tape_origin"] == {"schema": ref.SCHEMA, "forward_mode": ref.MODE,
                                   "parameter_sha256": ref.parameter_digest(p)}
    replay = old.forward(p, tokens, mode="fixed_routes", base=tape)
    assert torch.equal(actual["logits"], replay["logits"])


def diagonal():
    p = {key: torch.zeros_like(value) for key, value in parameters(d=2, v=2, n=1, k=1).items()}
    p["MP"].fill_(-8192)
    p["MP"][0, torch.arange(2), torch.arange(2)] = 0
    p["W_HEAD"].fill_(-8192)
    p["W_HEAD"][torch.arange(2), torch.arange(2)] = 0
    return p


def test_fractional_tape_is_affine_adapter_not_integer_or_floor_model():
    p = diagonal()
    p["EMBED"][0] = torch.tensor([.25, 1.0])
    tokens = torch.tensor([0])
    result = ref.forward(p, tokens)
    assert result["logits"].tolist() == [[.5, 2.0]]
    assert result["centering_offsets"].tolist() == [[.625]]
    tape = ref.reference_base(result)
    assert torch.equal(old.forward(p, tokens, mode="fixed_routes", base=tape)["logits"], result["logits"])
    changed = {key: value.clone() for key, value in p.items()}
    changed["EMBED"][0, 0] += .5
    affine = old.forward(changed, tokens, mode="fixed_routes", base=tape)["logits"]
    dynamic = ref.forward(changed, tokens)["logits"]
    assert not torch.equal(affine, dynamic)
    with pytest.raises(ValueError):
        old.forward(p, tokens, mode="integer_anchor")


def test_ties_and_causal_dilations_select_first_valid_candidate():
    p = {key: torch.zeros_like(value) for key, value in parameters(d=2, v=2, n=2).items()}
    tokens = torch.tensor([0, 1, 0])
    tied = ref.forward(p, tokens)
    assert all(not bool(values.any()) for values in tied["routes"].values())
    p["MH"][0, :, 1] = 20.25
    p["MH"][1, :, 1] = -20.25
    p["MH"][0, :, 6], p["MH"][1, :, 6] = 1000, -1000
    result = ref.forward(p, tokens)
    assert result["routes"]["TAP_IDX"][0].tolist() == [[0, 0], [1, 1], [1, 1]]
    assert result["routes"]["TAP_IDX"][1].tolist() == [[0, 0], [0, 0], [1, 1]]
    assert not ref.forward(p, tokens, use_context=False)["routes"]["TAP_IDX"].any()


def test_negative_fractional_center_uses_floor_not_truncation():
    p = diagonal()
    p["EMBED"][0] = torch.tensor([-.25, 0])
    actual = ref.forward(p, torch.tensor([0]))
    assert actual["centering_offsets"].tolist() == [[.875]]
    assert actual["logits"].tolist() == [[.5, 1.0]]


@pytest.mark.parametrize("variant", ref.VARIANTS)
def test_all_head_vjps_use_common_raw_equivalent_units(variant):
    p = parameters()
    tokens, targets = torch.tensor([0, 1, 2, 1, 0]), [1, 2, 1, 0, 2]
    result = ref.forward(p, tokens)
    actual = ref.gradient(p, tokens, targets, result, variant=variant)
    head = numeric.loss_head(result["logits"], targets, "variable_mass" if variant == "legacy" else variant)
    if variant in ("legacy", "variable_mass"):
        cotangent = head["native_gradient"].double()
    else:
        cotangent = head["supplied_gradient"] * (65536 / numeric.BETA)
    callback = (lambda g: g - torch.floor(g.sum() / g.numel())) if variant == "legacy" else None
    expected = old.manual_vjp(p, tokens, cotangent, ref.reference_base(result), center_vjp=callback)
    assert actual["gradient_units"] == ref.GRADIENT_UNITS
    assert actual["native_integer_gradient"] is (variant == "legacy")
    for name in ref.WEIGHTS:
        torch.testing.assert_close(actual["gradients"][name], expected[name], rtol=0, atol=1e-10)
    if variant == "legacy":
        assert all(torch.equal(g, g.trunc()) for g in actual["gradients"].values())


def test_synthetic_full_d16_integer_head_reverse_equals_frozen_observer():
    from manuscript.tools.observe_fpga_lm_routes import observe
    from scripts import fpga_lm_pretrain as pilot
    shape = pilot.layout(16)
    p = parameters(d=16, v=128, n=shape.ML_N_LAYERS, k=shape.ML_K, integer=True)
    tokens = torch.tensor([(7 * i + i // 5) % 128 for i in range(128)])
    targets = [(11 * i + 3) % 128 for i in range(128)]
    flat = {name: values.reshape(-1).to(torch.int64).tolist() for name, values in p.items()}
    expected = observe(flat, tokens.tolist(), targets, shape, zshift=2, include_parameter_gradients=True)
    result = ref.forward(p, tokens)
    actual = ref.gradient(p, tokens, targets, result)
    assert actual["head"]["predictions"].tolist() == expected["predictions"]
    assert actual["head"]["native_gradient"].reshape(-1).tolist() == expected["gradients"]
    for name in result["routes"]:
        assert result["routes"][name].reshape(-1).tolist() == expected["native_routes"][name]
    for name in ref.WEIGHTS:
        assert actual["gradients"][name].reshape(-1).tolist() == expected["parameter_gradients"][name]


def test_real_center_vjp_matches_affine_autograd_not_floor_derivative_claim():
    p = parameters()
    tokens, targets = torch.tensor([0, 1, 2]), [1, 2, 0]
    result = ref.forward(p, tokens)
    actual = ref.gradient(p, tokens, targets, result, variant="exact_ce")
    differentiable = {key: value.clone().requires_grad_(True) for key, value in p.items()}
    replay = old.forward(differentiable, tokens, mode="fixed_routes", base=ref.reference_base(result))
    loss = numeric.loss_head(replay["logits"], targets)["loss"]
    expected = torch.autograd.grad(loss, tuple(differentiable.values()))
    for (name, _), derivative in zip(differentiable.items(), expected):
        torch.testing.assert_close(actual["gradients"][name] * (numeric.BETA / 65536), derivative, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("stage", ["EMBED", "centering", "temporal", "projection", "residual", "head"])
def test_would_clip_guards_all_weights_and_selected_intermediates(stage):
    p = {key: torch.zeros_like(value) for key, value in parameters(d=2, v=2, n=1, k=1).items()}
    if stage == "EMBED":
        p["EMBED"][0, 0] = 32767.125
    elif stage == "centering":
        p["EMBED"][0] = torch.tensor([-32768, 32767])
    elif stage == "temporal":
        p["EMBED"][0] = torch.tensor([-1, 1])
        p["MH"].fill_(32767)
    elif stage == "projection":
        p["EMBED"][0] = torch.tensor([-1, 1])
        p["MB"].fill_(32767)
    elif stage == "residual":
        p["EMBED"].fill_(32767)
        p["MB"].fill_(1)
    else:
        p["EMBED"].fill_(1)
        p["W_HEAD"].fill_(32767)
    with pytest.raises(ValueError, match=stage):
        ref.forward(p, torch.tensor([0]))


@pytest.mark.parametrize("region", ref.WEIGHTS)
def test_all_weight_regions_bounded_before_forward(region):
    p = parameters()
    p[region].reshape(-1)[0] = -32768.125
    with pytest.raises(ValueError, match=region):
        ref.forward(p, torch.tensor([0]))


@pytest.mark.parametrize("change", ["width", "dtype", "nan", "missing", "tokens", "bool_context", "float_chunk"])
def test_forward_rejects_invalid_geometry_dtype_and_configuration(change):
    p, tokens, kwargs = parameters(), torch.tensor([0]), {}
    if change == "width":
        p = parameters(d=3)
    elif change == "dtype":
        p["EMBED"] = p["EMBED"].float()
    elif change == "nan":
        p["MH"].reshape(-1)[0] = float("nan")
    elif change == "missing":
        del p["MB"]
    elif change == "tokens":
        tokens = tokens.double()
    elif change == "bool_context":
        kwargs["use_context"] = 1
    else:
        kwargs["chunk_size"] = 1.0
    with pytest.raises(ValueError):
        ref.forward(p, tokens, **kwargs)


@pytest.mark.parametrize("edit", ["schema", "mode", "native", "autograd", "saturation", "bool_saturation", "geometry",
                                  "tokens", "logits", "offset", "past_tap", "hash", "context"])
def test_reference_base_rejects_malformed_or_misleading_tape(edit):
    result = ref.forward(parameters(), torch.tensor([0, 1]))
    if edit == "schema":
        result["schema"] = old.SCHEMA
    elif edit == "mode":
        result["mode"] = "integer_anchor"
    elif edit == "native":
        result["native_execution"] = True
    elif edit == "autograd":
        result["autograd_forward"] = True
    elif edit == "saturation":
        result["saturation_events"] = 1
    elif edit == "bool_saturation":
        result["saturation_events"] = False
    elif edit == "geometry":
        result["geometry"]["D"] = 4.0
    elif edit == "tokens":
        result["tokens"][0] = 3
    elif edit == "logits":
        result["logits"][0, 0] = float("inf")
    elif edit == "offset":
        result["centering_offsets"][0, 0] = -0.5
    elif edit == "past_tap":
        result["routes"]["TAP_IDX"][0, 0, 0] = 1
    elif edit == "hash":
        result["parameter_sha256"] = "x" * 64
    else:
        result["use_context"] = 1
    with pytest.raises(ValueError):
        ref.reference_base(result)


@pytest.mark.parametrize("change", ["weights", "tokens"])
def test_gradient_rejects_stale_forward_binding(change):
    p, tokens = parameters(), torch.tensor([0, 1])
    result = ref.forward(p, tokens)
    if change == "weights":
        p["MP"][0, 0, 0] += .125
    else:
        tokens[0] = 1
    with pytest.raises(ValueError, match="differs"):
        ref.gradient(p, tokens, [1, 2], result)


def test_forward_tape_adapter_and_reverse_do_not_mutate_inputs():
    p, tokens = parameters(), torch.tensor([0, 1, 2])
    for value in p.values():
        value.requires_grad_(True)
    before = {key: value.detach().clone() for key, value in p.items()}
    result = ref.forward(p, tokens)
    snapshot = copy.deepcopy(result)
    tape = ref.reference_base(result)
    tape["routes"]["AD_HEAD"].zero_()
    tape["centering_offsets"].zero_()
    tape["tokens"].zero_()
    ref.gradient(p, tokens, [1, 2, 0], result)
    for key in p:
        assert torch.equal(p[key], before[key]) and p[key].grad is None
    for key in result["routes"]:
        assert torch.equal(result["routes"][key], snapshot["routes"][key])
    assert torch.equal(result["tokens"], snapshot["tokens"])
    assert torch.equal(result["centering_offsets"], snapshot["centering_offsets"])


def test_checked_scatter_rejects_transient_overflow_without_partial_publication():
    target = np.zeros(2, dtype=np.int64)
    with pytest.raises(OverflowError, match="ordered prefix"):
        ref._scatter(target, [0, 0, 0], [2**31 - 1, 1, -1], "test")
    np.testing.assert_array_equal(target, [0, 0])
    ref._scatter(target, [1, 0, 1, 0], [7, -3, -2, 4], "test")
    np.testing.assert_array_equal(target, [1, 5])


def test_raw_reverse_rejects_head_scatter_overflow_and_noninteger_cotangent():
    p = {key: torch.zeros_like(value) for key, value in parameters(d=2, v=3, n=1, k=1).items()}
    tokens = torch.tensor([0])
    result = ref.forward(p, tokens)
    with pytest.raises(OverflowError, match="head scatter"):
        ref.raw_legacy_vjp(p, tokens, torch.tensor([[2**31 - 1, 1, -1]], dtype=torch.float64), result)
    with pytest.raises(ValueError, match="exact-integer"):
        ref.raw_legacy_vjp(p, tokens, torch.tensor([[.5, 0, 0]], dtype=torch.float64), result)
    with pytest.raises(OverflowError, match="exceeds int32"):
        ref.raw_legacy_vjp(p, tokens, torch.tensor([[2**31, 0, 0]], dtype=torch.float64), result)


def test_dyadic_fractional_parameter_and_raw_gradient_arithmetic():
    p, tokens = parameters(), torch.tensor([0, 1, 2])
    p["EMBED"][0, 0] += 1 / (1 << 18)
    result = ref.forward(p, tokens)
    assert torch.equal(result["logits"] * (1 << 18), (result["logits"] * (1 << 18)).round())
    gradient = ref.gradient(p, tokens, [1, 2, 0], result)["gradients"]
    assert all(torch.equal(value, value.round()) for value in gradient.values())


def test_source_identity_and_explicit_scope():
    identity = ref.source_identity()
    assert set(identity) == {"manuscript/tools/fpga_lm_trajectory_reference.py",
                             "manuscript/tools/fpga_lm_float_reference.py",
                             "manuscript/tools/fpga_lm_numeric_diagnostic.py"}
    root = Path(__file__).resolve().parents[1]
    for relative, expected in identity.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    assert ref.recipe()["native_execution"] is False and ref.recipe()["training_performed"] is False
    with pytest.raises(ValueError):
        ref.gradient(parameters(), torch.tensor([0]), [0], {}, variant="unknown")
