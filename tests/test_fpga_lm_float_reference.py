"""Synthetic math tests only: no corpus, scientific training or cloud use."""

import math

import pytest
import torch

from manuscript.tools import fpga_lm_float_reference as ref


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def weights(seed=719, *, d=16, v=4, n=4, k=7, integer=False, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    dimensions = {"EMBED": (v, d), "MH": (n, d, k), "MP": (n, d, d),
                  "MB": (n, d), "W_HEAD": (v, d)}
    return {name: (torch.randint(-4, 5, shape, generator=generator).to(torch.float64) if integer
                   else torch.randn(shape, generator=generator, dtype=torch.float64)).requires_grad_(requires_grad)
            for name, shape in dimensions.items()}


def scalar_forward(parameters, tokens, integer, use_context=True):
    """Independent scalar/list interpreter, no torch operations or shared routing."""
    p = {name: value.detach().tolist() for name, value in parameters.items()}
    ids = tokens.tolist()
    v, d, n, k, t = len(p["EMBED"]), len(p["EMBED"][0]), len(p["MH"]), len(p["MH"][0][0]), len(ids)
    history = [[None] * t for _ in range(n)]
    taps, projections = ([[None] * t for _ in range(n)] for _ in range(2))
    head, logits = [], []
    for position, token in enumerate(ids):
        residual = list(p["EMBED"][token])
        for layer in range(n):
            mean = sum(residual) / d
            if integer:
                mean = math.floor(mean)
            history[layer][position] = [value - mean for value in residual]
            convolution, selected = [], []
            for feature in range(d):
                scores, valid = [], []
                for tap in range(k if use_context else 1):
                    time = position - tap * (1 << min(layer, 6))
                    if time >= 0:
                        scores.append(history[layer][time][feature] + p["MH"][layer][feature][tap])
                        valid.append(tap)
                best = max(scores) if layer % 2 == 0 else min(scores)
                convolution.append(best)
                selected.append(valid[scores.index(best)])
            taps[layer][position] = selected
            projected, selected = [], []
            for output in range(d):
                scores = [p["MP"][layer][output][feature] + convolution[feature] for feature in range(d)]
                best = max(scores)
                projected.append(best + p["MB"][layer][output])
                selected.append(scores.index(best))
            projections[layer][position] = selected
            residual = [left + right for left, right in zip(residual, projected)]
        scores, selected = [], []
        for output in range(v):
            row = [p["W_HEAD"][output][feature] + residual[feature] for feature in range(d)]
            scores.append(max(row))
            selected.append(row.index(max(row)))
        logits.append(scores)
        head.append(selected)
    return logits, {"AD_HEAD": head, "PROJ_IDX": projections, "TAP_IDX": taps}


@pytest.mark.parametrize("mode", ["smooth", "integer_anchor"])
@pytest.mark.parametrize("use_context", [True, False])
def test_complete_forward_and_routes_equal_independent_scalar_interpreter(mode, use_context):
    p = weights(integer=True)
    tokens = torch.tensor([0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 3])
    result = ref.forward(p, tokens, mode=mode, use_context=use_context, chunk_size=3)
    expected, routes = scalar_forward(p, tokens, mode == "integer_anchor", use_context)
    assert torch.allclose(result["logits"], torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    for name, values in routes.items():
        assert torch.equal(result["routes"][name], torch.tensor(values))
    assert all(value.dtype == torch.int64 for value in result["routes"].values())
    assert result["geometry"] == {"D": 16, "V": 4, "N": 4, "K": 7, "T": 12}
    assert result["use_context"] is use_context


def test_all_ties_choose_first_valid_index_and_no_alternative_has_infinite_margin():
    p = {name: torch.zeros_like(value) for name, value in weights().items()}
    tokens = torch.tensor([0, 1, 2, 3] * 3)
    result = ref.forward(p, tokens)
    assert not bool(result["logits"].any())
    for values in result["routes"].values():
        assert not bool(values.any())
    for name, margin in result["margins"].items():
        single = result["candidate_counts"][name] == 1
        assert torch.equal(torch.isinf(margin), single)
        assert not bool(margin[~single].any())
    replay = ref.forward(p, tokens, mode="fixed_routes", base=result)
    assert replay["margins"] is None


def test_causal_padding_is_excluded_and_layers_alternate_max_min_dilations():
    p = {name: torch.zeros_like(value) for name, value in weights(d=2, v=2, n=2).items()}
    p["MH"][0, :, 1] = 20
    p["MH"][1, :, 1] = -20
    p["MH"][0, :, 6], p["MH"][1, :, 6] = 1000, -1000
    result = ref.forward(p, torch.tensor([0, 1, 0]), mode="integer_anchor")
    taps = result["routes"]["TAP_IDX"]
    assert taps[0].tolist() == [[0, 0], [1, 1], [1, 1]]
    assert taps[1].tolist() == [[0, 0], [0, 0], [1, 1]]
    assert result["candidate_counts"]["TAP_IDX"][:, :, 0].tolist() == [[1, 2, 3], [1, 1, 2]]
    disabled = ref.forward(p, torch.tensor([0, 1, 0]), mode="integer_anchor", use_context=False)
    assert not bool(disabled["routes"]["TAP_IDX"].any())
    assert not torch.equal(disabled["logits"], result["logits"])


def diagonal_case(n=2):
    p = {name: torch.zeros_like(value) for name, value in weights(d=2, v=2, n=n, k=1).items()}
    p["MP"].fill_(-100)
    p["MP"][:, 0, 0], p["MP"][:, 1, 1] = 0, 0
    p["W_HEAD"].fill_(-100)
    p["W_HEAD"][0, 0], p["W_HEAD"][1, 1] = 0, 0
    return p


def test_residual_and_mean_centering_have_hand_derived_logits_and_vjp():
    p = diagonal_case()
    p["EMBED"][0] = torch.tensor([1., 3.])
    p["MB"].fill_(1)
    tokens = torch.tensor([0])
    base = ref.forward(p, tokens)
    assert base["logits"].tolist() == [[0., 8.]]
    # Two residual blocks give J=(I+C)^2=4I-3P, where P is mean projection.
    gradient = ref.manual_vjp(p, tokens, torch.tensor([[2., -1.]], dtype=torch.float64), base)
    assert gradient["EMBED"][0].tolist() == [6.5, -5.5]
    assert gradient["EMBED"][1].tolist() == [0., 0.]
    assert gradient["W_HEAD"].tolist() == [[2., 0.], [0., -1.]]


@pytest.mark.parametrize("mode", ["smooth", "integer_anchor"])
def test_arbitrary_cotangent_manual_reverse_equals_autograd_for_all_regions(mode):
    p = weights(integer=(mode == "integer_anchor"), requires_grad=True)
    tokens = torch.tensor([0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 3])
    base = ref.forward(p, tokens, mode=mode)
    generator = torch.Generator().manual_seed(592)
    cotangent = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    replay = ref.forward(p, tokens, mode="fixed_routes", base=base)
    assert torch.allclose(replay["logits"], base["logits"], atol=1e-12, rtol=1e-12)
    expected = torch.autograd.grad(replay["logits"], tuple(p.values()), cotangent, retain_graph=True)
    direct = torch.autograd.grad(base["logits"], tuple(p.values()), cotangent)
    manual = ref.manual_vjp(p, tokens, cotangent, base)
    for name, a, b in zip(p, expected, direct):
        assert torch.allclose(manual[name], a, atol=1e-11, rtol=1e-11), name
        assert torch.allclose(a, b, atol=1e-11, rtol=1e-11), name
        assert not manual[name].requires_grad


def test_fixed_routes_are_jointly_affine_in_every_parameter_even_across_dynamic_route_cells():
    p, other = weights(), weights(seed=331)
    tokens = torch.tensor([0, 2, 1, 3, 1, 0, 2, 2, 3, 0, 1, 3])
    base = ref.forward(p, tokens)
    mixed = {name: .3 * p[name] + .7 * other[name] for name in p}
    a = ref.forward(p, tokens, mode="fixed_routes", base=base)["logits"]
    b = ref.forward(other, tokens, mode="fixed_routes", base=base)["logits"]
    c = ref.forward(mixed, tokens, mode="fixed_routes", base=base)["logits"]
    assert torch.allclose(c, .3 * a + .7 * b, atol=2e-13, rtol=2e-13)
    assert not torch.equal(ref.forward(other, tokens)["routes"]["PROJ_IDX"], base["routes"]["PROJ_IDX"])


def test_fixed_route_ce_directional_derivative_matches_manual_vjp():
    p, direction = weights(), weights(seed=331)
    tokens, targets = torch.tensor([0, 2, 1, 3, 0]), torch.tensor([2, 1, 3, 0, 1])
    base = ref.forward(p, tokens)
    logits = base["logits"]
    cotangent = ref.BETA * (torch.softmax(ref.BETA * logits, -1) - torch.nn.functional.one_hot(targets, 4)) / len(tokens)
    gradient = ref.manual_vjp(p, tokens, cotangent, base)
    expected = sum(float((gradient[name] * direction[name]).sum()) for name in p)
    step = 1e-4
    values = []
    for sign in (-1, 1):
        altered = {name: p[name] + sign * step * direction[name] for name in p}
        values.append(float(ref.loss(ref.forward(altered, tokens, mode="fixed_routes", base=base)["logits"], targets)))
    assert (values[1] - values[0]) / (2 * step) == pytest.approx(expected, abs=2e-10, rel=2e-8)


def test_exact_ce_temperature_and_common_shift_null_direction():
    assert ref.BETA == math.log(2) / 64
    p = weights(requires_grad=True)
    tokens, targets = torch.tensor([0, 2, 1, 3]), torch.tensor([2, 1, 3, 0])
    result = ref.forward(p, tokens)
    actual = ref.loss(result["logits"], targets)
    expected = (torch.logsumexp(ref.BETA * result["logits"], -1)
                - ref.BETA * result["logits"][torch.arange(4), targets]).mean()
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    altered = {name: value.detach().clone() for name, value in p.items()}
    altered["W_HEAD"] += 23
    shifted = ref.forward(altered, tokens)
    assert torch.allclose(shifted["logits"], result["logits"] + 23, atol=1e-12, rtol=1e-12)
    assert torch.allclose(ref.loss(shifted["logits"], targets), actual, atol=1e-14, rtol=1e-14)
    gradient = torch.autograd.grad(actual, p["W_HEAD"])[0]
    assert abs(float(gradient.sum())) < 1e-16


def test_integer_anchor_offsets_and_fixed_integer_separate_mean_rounding_from_routes():
    p = diagonal_case(n=1)
    p["EMBED"][0] = torch.tensor([0., 1.])
    tokens = torch.tensor([0])
    base = ref.forward(p, tokens, mode="integer_anchor")
    assert base["logits"].tolist() == [[0., 2.]]
    assert base["centering_offsets"].tolist() == [[.5]]
    smooth = ref.forward(p, tokens)
    assert smooth["logits"].tolist() == [[-.5, 1.5]]
    p["EMBED"][0, 0] = 1
    affine = ref.forward(p, tokens, mode="fixed_routes", base=base)
    integer = ref.forward(p, tokens, mode="fixed_integer", base=base)
    dynamic = ref.forward(p, tokens, mode="integer_anchor")
    assert affine["logits"].tolist() == [[1.5, 1.5]]
    assert integer["logits"].tolist() == dynamic["logits"].tolist() == [[1., 1.]]
    assert integer["centering_offsets"].tolist() == [[0.]]
    assert integer["margins"] is None


def test_chunking_does_not_change_dynamic_routes_values_or_gradients():
    p = weights(requires_grad=True)
    tokens = torch.tensor([0, 1, 2, 3])
    results = [ref.forward(p, tokens, chunk_size=chunk) for chunk in (1, 3, 64)]
    gradients = [torch.autograd.grad(result["logits"].sum(), tuple(p.values()), retain_graph=True) for result in results]
    for result, gradient in zip(results[1:], gradients[1:]):
        assert torch.equal(result["logits"], results[0]["logits"])
        for name in result["routes"]:
            assert torch.equal(result["routes"][name], results[0]["routes"][name])
            assert torch.equal(result["margins"][name], results[0]["margins"][name])
        for a, b in zip(gradient, gradients[0]):
            assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)


def test_d512_affine_replay_uses_only_supplied_routes_and_no_candidate_selector(monkeypatch):
    p = weights(d=512, v=128, n=4, integer=True)
    tokens = torch.tensor([0, 1])
    base = ref.forward(p, tokens, mode="integer_anchor", chunk_size=32)
    monkeypatch.setattr(ref, "_select", lambda *args, **kwargs: pytest.fail("fixed replay materialized candidates"))
    replay = ref.forward(p, tokens, mode="fixed_routes", base=base)
    assert torch.equal(replay["logits"], base["logits"])
    gradient = ref.manual_vjp(p, tokens, torch.ones(2, 128, dtype=torch.float64), base)
    assert sum(value.numel() for value in gradient.values()) == 1_196_032


def test_synthetic_integer_forward_and_raw_legacy_vjp_match_frozen_host_oracles():
    from manuscript.tools import diagnose_fpga_lm_calibration as calibration
    from manuscript.tools.observe_fpga_lm_routes import observe
    from scripts import fpga_lm_pretrain as pilot
    shape = pilot.layout(16)
    p = weights(d=16, v=shape.ML_V, n=shape.ML_N_LAYERS, k=shape.ML_K, integer=True)
    flat = {name: value.reshape(-1).to(torch.int64).tolist() for name, value in p.items()}
    tokens = torch.tensor([(index * 7 + index // 3) % shape.ML_V for index in range(shape.ML_L_SEQ)])
    targets = tokens.roll(-1)
    original = observe(flat, tokens.tolist(), targets.tolist(), shape, zshift=2, include_parameter_gradients=True)
    independent_logits = calibration.integer_logits(flat, tokens.tolist(), shape)
    base = ref.forward(p, tokens, mode="integer_anchor")
    assert base["logits"].tolist() == independent_logits["logits"]
    assert base["logits"].argmax(-1).tolist() == original["predictions"]
    for name, value in base["routes"].items():
        assert value.reshape(-1).tolist() == original["native_routes"][name]
    raw = torch.tensor(original["gradients"], dtype=torch.float64).reshape(shape.ML_L_SEQ, shape.ML_V)
    gradient = ref.manual_vjp(p, tokens, raw, base, center_vjp=lambda g: g - torch.floor(g.sum() / shape.ML_D))
    for name, value in gradient.items():
        assert value.reshape(-1).tolist() == original["parameter_gradients"][name], name
    assert original["saturation_events"] == 0


@pytest.mark.parametrize("stage", ["centering", "temporal", "projection", "residual", "head"])
def test_integer_anchor_fails_closed_on_every_would_clip_boundary(stage):
    p = {name: torch.zeros_like(value) for name, value in weights(d=2, v=2, n=1, k=1).items()}
    if stage == "centering":
        p["EMBED"][0] = torch.tensor([-32768., 32767.])
    elif stage == "temporal":
        p["EMBED"][0] = torch.tensor([-1., 1.])
        p["MH"].fill_(32767)
    elif stage == "projection":
        p["EMBED"][0] = torch.tensor([-1., 1.])
        p["MB"].fill_(32767)
    elif stage == "residual":
        p["EMBED"].fill_(32767)
        p["MB"].fill_(1)
    else:
        p["EMBED"].fill_(1)
        p["W_HEAD"].fill_(32767)
    with pytest.raises(ValueError, match=stage):
        ref.forward(p, torch.tensor([0]), mode="integer_anchor")


@pytest.mark.parametrize("mode", ["integer_anchor", "fixed_integer"])
def test_integer_modes_reject_float_weights_but_fixed_affine_allows_perturbations(mode):
    p = weights(integer=True)
    tokens = torch.tensor([0, 1])
    base = ref.forward(p, tokens, mode="integer_anchor")
    p["MP"][0, 0, 0] += .25
    with pytest.raises(ValueError):
        ref.forward(p, tokens, mode=mode, base=base if mode == "fixed_integer" else None)
    assert torch.isfinite(ref.forward(p, tokens, mode="fixed_routes", base=base)["logits"]).all()


@pytest.mark.parametrize("mutation", ["float32", "nan", "missing", "shape", "float_tokens", "bad_token"])
def test_malformed_parameters_and_tokens_fail_before_computation(mutation):
    p, tokens = weights(), torch.tensor([0, 1])
    if mutation == "float32":
        p["MP"] = p["MP"].float()
    elif mutation == "nan":
        p["MH"][0, 0, 0] = float("nan")
    elif mutation == "missing":
        del p["MB"]
    elif mutation == "shape":
        p["MP"] = p["MP"][:, :-1]
    elif mutation == "float_tokens":
        tokens = tokens.double()
    else:
        tokens[0] = 4
    with pytest.raises(ValueError):
        ref.forward(p, tokens)


@pytest.mark.parametrize("mutation", ["tokens", "geometry", "head", "projection", "past_tap", "route_float", "offset",
                                       "negative_offset", "smooth_offset", "mode", "context"])
def test_fixed_route_base_is_bound_to_complete_valid_geometry_and_causality(mutation):
    p, tokens = weights(), torch.tensor([0, 1, 2])
    base = ref.forward(p, tokens)
    if mutation == "tokens":
        base["tokens"][0] = 1
    elif mutation == "geometry":
        base["geometry"]["D"] = 16.0
    elif mutation == "head":
        base["routes"]["AD_HEAD"][0, 0] = 16
    elif mutation == "projection":
        base["routes"]["PROJ_IDX"][0, 0, 0] = -1
    elif mutation == "past_tap":
        base["routes"]["TAP_IDX"][0, 0, 0] = 1
    elif mutation == "route_float":
        base["routes"]["AD_HEAD"] = base["routes"]["AD_HEAD"].double()
    elif mutation == "offset":
        base["centering_offsets"][0, 0] = float("nan")
    elif mutation == "negative_offset":
        base["centering_offsets"][0, 0] = -.5
    elif mutation == "smooth_offset":
        base["centering_offsets"][0, 0] = .5
    elif mutation == "mode":
        base["mode"] = "unknown"
    else:
        base["use_context"] = 1
    with pytest.raises(ValueError):
        ref.forward(p, tokens, mode="fixed_routes", base=base)


def test_reference_and_manual_reverse_never_mutate_inputs_or_base():
    p, tokens = weights(), torch.tensor([0, 1, 2])
    snapshots = {name: value.clone() for name, value in p.items()}
    base = ref.forward(p, tokens)
    saved_routes = {name: value.clone() for name, value in base["routes"].items()}
    saved_offsets, ids = base["centering_offsets"].clone(), tokens.clone()
    cotangent = torch.ones(3, 4, dtype=torch.float64)
    ref.manual_vjp(p, tokens, cotangent, base, center_vjp=lambda incoming: incoming.sub_(incoming.mean()))
    ref.forward(p, tokens, mode="fixed_routes", base=base)
    assert torch.equal(cotangent, torch.ones_like(cotangent))
    assert torch.equal(tokens, ids) and torch.equal(base["centering_offsets"], saved_offsets)
    assert all(torch.equal(p[name], value) for name, value in snapshots.items())
    assert all(torch.equal(base["routes"][name], value) for name, value in saved_routes.items())


@pytest.mark.parametrize("bad", [lambda g: g.float(), lambda g: g[:-1], lambda g: g * float("nan")])
def test_manual_conditioner_override_requires_complete_finite_fp64_vectors(bad):
    p, tokens = weights(), torch.tensor([0, 1])
    base = ref.forward(p, tokens)
    with pytest.raises(ValueError):
        ref.manual_vjp(p, tokens, torch.ones(2, 4, dtype=torch.float64), base, center_vjp=bad)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_loss_temperature_is_explicit_positive_finite_number(value):
    with pytest.raises(ValueError):
        ref.loss(torch.zeros(2, 4, dtype=torch.float64), torch.tensor([0, 1]), beta=value)


def test_source_identity_is_local_and_complete():
    from pathlib import Path
    import hashlib
    path = Path(ref.__file__).resolve()
    assert ref.source_identity() == {"manuscript/tools/fpga_lm_float_reference.py": hashlib.sha256(path.read_bytes()).hexdigest()}
