"""Tiny synthetic exact-CE fitting contracts; never a corpus study."""
from __future__ import annotations

import importlib
import math

import pytest
import torch


@pytest.fixture
def helper():
    return importlib.import_module("manuscript.tools.fpga_lm_exact_fit")


def weights(seed=7):
    generator = torch.Generator().manual_seed(seed)
    shapes = {"EMBED": (4, 4), "MH": (2, 4, 3), "MP": (2, 4, 4),
              "MB": (2, 4), "W_HEAD": (4, 4)}
    return {name: torch.randn(shape, dtype=torch.float64, generator=generator) * .2
            for name, shape in shapes.items()}


def copies(mapping):
    return {key: value.detach().clone() for key, value in mapping.items()}


def assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].shape == expected[name].shape
        assert torch.equal(actual[name], expected[name]), name


def literal_forward(parameters, tokens, context=True):
    """Separate full-matrix autograd oracle with real, not floor, centering."""
    layers, _, kernel = parameters["MH"].shape
    histories = [[] for _ in range(layers)]
    output = []
    for position, token in enumerate(tokens):
        residual = parameters["EMBED"][token]
        for layer in range(layers):
            histories[layer].append(residual - residual.mean())
            dilation = 2 ** min(layer, 6)
            count = min(kernel, position // dilation + 1) if context else 1
            candidates = torch.stack([
                histories[layer][position - tap * dilation] + parameters["MH"][layer, :, tap]
                for tap in range(count)], dim=-1)
            convolution = (candidates.max(dim=-1).values if layer % 2 == 0
                           else candidates.min(dim=-1).values)
            projection = (parameters["MP"][layer] + convolution[None, :]).max(dim=-1).values
            residual = residual + projection + parameters["MB"][layer]
        output.append((parameters["W_HEAD"] + residual[None, :]).max(dim=-1).values)
    return torch.stack(output)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("context", [False, True])
def test_smooth_forward_matches_separate_full_matrix_oracle(helper, independent, context):
    parameters = weights()
    tokens = torch.tensor([0, 1, 3, 2, 0], dtype=torch.int64)
    before = copies(parameters)
    actual = helper.forward(parameters, tokens, context=context, independent=independent)
    expected = literal_forward(parameters, tokens, context=context)
    torch.testing.assert_close(actual["logits"], expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(actual["centering_offsets"], torch.zeros_like(actual["centering_offsets"]))
    assert not actual["logits"].requires_grad
    assert_same(parameters, before)


@pytest.mark.parametrize("independent", [False, True])
def test_gradient_is_exact_mean_ce_derivative_without_second_beta(helper, independent):
    parameters = weights()
    tokens = torch.tensor([0, 1, 3, 2, 0], dtype=torch.int64)
    targets = torch.tensor([1, 3, 2, 0, 2], dtype=torch.int64)
    base = helper.forward(parameters, tokens)
    actual = helper.gradient(parameters, tokens, targets, base, len(tokens), independent=independent)
    leaves = {name: value.clone().requires_grad_(True) for name, value in parameters.items()}
    loss = torch.nn.functional.cross_entropy(literal_forward(leaves, tokens), targets)
    expected = torch.autograd.grad(loss, tuple(leaves.values()))
    for name, value in zip(leaves, expected):
        torch.testing.assert_close(actual[name], value, atol=1e-12, rtol=1e-11)
        assert actual[name].dtype == torch.float64 and not actual[name].requires_grad
    unreduced = helper.gradient(parameters, tokens, targets, base, 1, independent=independent)
    for name in parameters:
        torch.testing.assert_close(unreduced[name], actual[name] * len(tokens), atol=1e-12, rtol=1e-11)


def test_dimensionless_scaling_preserves_true_probabilities_and_homogeneity(helper):
    parameters = weights()
    before = copies(parameters)
    beta = math.log(2.0) / 64.0
    scaled = helper.dimensionless(parameters)
    for name in parameters:
        assert torch.equal(scaled[name], parameters[name] * beta)
        assert scaled[name].data_ptr() != parameters[name].data_ptr()
    tokens = torch.tensor([0, 1, 3, 2], dtype=torch.int64)
    raw = literal_forward(parameters, tokens)
    logits = helper.forward(scaled, tokens)["logits"]
    torch.testing.assert_close(logits, raw * beta, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(logits.softmax(-1), (raw * beta).softmax(-1), atol=1e-14, rtol=1e-12)
    assert_same(parameters, before)


def test_independent_full_routes_preserve_first_ties(helper):
    parameters = {name: torch.zeros_like(value) for name, value in weights().items()}
    tokens = torch.tensor([0, 1, 3, 2], dtype=torch.int64)
    produced = helper.trace(helper.forward(parameters, tokens))
    checked = helper.trace(helper.forward(parameters, tokens, independent=True))
    assert_same(produced, checked)
    for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX"):
        assert produced[name].dtype == torch.int64
        assert torch.count_nonzero(produced[name]) == 0


def test_tropical_causality_and_disabled_context(helper):
    parameters = weights()
    tokens = torch.tensor([0, 1, 3, 2, 0], dtype=torch.int64)
    changed = tokens.clone()
    changed[-1] = 2
    for independent in (False, True):
        baseline = helper.forward(parameters, tokens, independent=independent)
        modified = helper.forward(parameters, changed, independent=independent)
        assert torch.equal(baseline["logits"][:-1], modified["logits"][:-1])
        disabled = helper.forward(parameters, tokens, context=False, independent=independent)
        separate = torch.cat([helper.forward(parameters, token.reshape(1), independent=independent)["logits"]
                              for token in tokens])
        assert torch.equal(disabled["logits"], separate)
        assert torch.count_nonzero(disabled["routes"]["TAP_IDX"]) == 0


@pytest.mark.parametrize("tokens,targets,vocab,expected", [
    ([[0, 0, 1, 1]], [[0, 0, 1, 1]], 2, 0.0),
    ([[0, 0, 1, 1]], [[0, 1, 0, 1]], 2, math.log(2.0)),
    ([[0, 0, 0, 0]], [[0, 0, 0, 1]], 2, -.75 * math.log(.75) - .25 * math.log(.25)),
    ([[0, 1], [0, 1]], [[1, 0], [1, 0]], 3, 0.0),
])
def test_empirical_bigram_floor_exact_known_counts(helper, tokens, targets, vocab, expected):
    actual = helper.empirical_bigram_floor(torch.tensor(tokens), torch.tensor(targets), vocab)
    assert actual == pytest.approx(expected, abs=1e-14)


def test_empirical_bigram_counts_do_not_invent_cross_window_pairs(helper):
    # The supplied per-position pairs are all deterministic, regardless of the
    # accidental last/first token adjacency between these independent windows.
    tokens = torch.tensor([[0, 0], [1, 1]], dtype=torch.int64)
    targets = torch.tensor([[1, 1], [0, 0]], dtype=torch.int64)
    assert helper.empirical_bigram_floor(tokens, targets, 2) == 0.0


def test_eager_adam_matches_explicit_bias_corrected_recurrence(helper):
    parameter = torch.tensor([.5, -1.5], dtype=torch.float64)
    expected = parameter.clone()
    first = torch.zeros_like(parameter)
    second = torch.zeros_like(parameter)
    optimizer = helper.make_optimizer([parameter])
    assert type(optimizer) is torch.optim.Adam
    group = optimizer.param_groups[0]
    assert group["lr"] == .001 and group["betas"] == (.9, .999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0
    assert group["foreach"] is False and group["fused"] is False
    for step, values in enumerate(([.3, -.6], [-.1, .5], [.7, -.2]), start=1):
        gradient = torch.tensor(values, dtype=torch.float64)
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient.square()
        expected = expected - .001 * (first / (1 - .9 ** step)) / (
            torch.sqrt(second / (1 - .999 ** step)) + 1e-8)
        parameter.grad = gradient.clone()
        optimizer.step()
        torch.testing.assert_close(parameter, expected, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], first, atol=1e-15, rtol=1e-13)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], second, atol=1e-15, rtol=1e-13)
        assert optimizer.state[parameter]["step"].item() == step


@pytest.mark.parametrize("case", ["empty", "duplicate", "dtype", "nonfinite", "nonleaf"])
def test_adam_constructor_rejects_invalid_parameters_without_mutation(helper, case):
    value = torch.tensor([1., 2.], dtype=torch.float64)
    if case == "empty":
        parameters = []
    elif case == "duplicate":
        parameters = [value, value]
    elif case == "dtype":
        parameters = [value.float()]
    elif case == "nonfinite":
        parameters = [torch.tensor([float("nan")], dtype=torch.float64)]
    else:
        parameters = [value.clone().requires_grad_(True) * 2]
    before = value.clone()
    with pytest.raises(ValueError):
        helper.make_optimizer(parameters)
    assert torch.equal(value, before)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0])
def test_mean_loss_divisor_is_positive_literal_integer(helper, bad):
    parameters = weights()
    tokens = torch.tensor([0, 1], dtype=torch.int64)
    base = helper.forward(parameters, tokens)
    with pytest.raises(ValueError):
        helper.gradient(parameters, tokens, tokens, base, bad)


@pytest.mark.parametrize("case", ["missing_region", "shape", "dtype", "nonfinite", "attached"])
def test_dimensionless_mapping_rejects_invalid_geometry_and_values(helper, case):
    parameters = weights()
    if case == "missing_region":
        del parameters["MB"]
    elif case == "shape":
        parameters["W_HEAD"] = parameters["W_HEAD"][:2]
    elif case == "dtype":
        parameters["MB"] = parameters["MB"].float()
    elif case == "nonfinite":
        parameters["MB"][0, 0] = float("inf")
    else:
        parameters["MB"].requires_grad_(True)
    with pytest.raises(ValueError):
        helper.dimensionless(parameters)


def test_gradient_rejects_stale_parameter_or_token_tape(helper):
    parameters = weights()
    tokens = torch.tensor([0, 1, 2], dtype=torch.int64)
    base = helper.forward(parameters, tokens)
    changed = copies(parameters)
    changed["MB"][0, 0] += .01
    with pytest.raises(ValueError):
        helper.gradient(changed, tokens, tokens, base, 3)
    with pytest.raises(ValueError):
        helper.gradient(parameters, tokens.flip(0), tokens, base, 3)


@pytest.mark.parametrize("case", ["float_tokens", "float_targets", "shape", "negative", "large", "empty"])
def test_empirical_floor_rejects_invalid_pair_arrays(helper, case):
    tokens = torch.tensor([[0, 1]], dtype=torch.int64)
    targets = torch.tensor([[1, 0]], dtype=torch.int64)
    if case == "float_tokens":
        tokens = tokens.double()
    elif case == "float_targets":
        targets = targets.double()
    elif case == "shape":
        targets = targets.reshape(-1)
    elif case == "negative":
        tokens[0, 0] = -1
    elif case == "large":
        targets[0, 0] = 2
    else:
        tokens = targets = torch.empty((0, 2), dtype=torch.int64)
    with pytest.raises(ValueError):
        helper.empirical_bigram_floor(tokens, targets, 2)


def test_gru_initialization_is_deterministic_without_changing_global_rng(helper):
    before = torch.random.get_rng_state().clone()
    model = helper.make_gru(seed=9, vocab=4, embedding=3, hidden=5)
    assert torch.equal(torch.random.get_rng_state(), before)
    same = helper.make_gru(seed=9, vocab=4, embedding=3, hidden=5)
    other = helper.make_gru(seed=10, vocab=4, embedding=3, hidden=5)
    assert_same(model.state_dict(), same.state_dict())
    assert any(not torch.equal(value, other.state_dict()[name])
               for name, value in model.state_dict().items() if not name.startswith("head."))
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu" and parameter.dtype == torch.float64
        if name.startswith("head."):
            assert torch.count_nonzero(parameter) == 0
    assert not any("prior" in name for name, _ in model.named_parameters())
    assert not any("prior" in name for name, _ in model.named_buffers())


@pytest.mark.parametrize("independent", [False, True])
def test_zero_correction_head_preserves_prior_and_first_body_gradient_is_zero(helper, independent):
    model = helper.make_gru(seed=2, vocab=4, embedding=3, hidden=5)
    tokens = torch.tensor([[0, 1, 2, 1], [3, 0, 2, 3]], dtype=torch.int64)
    prior = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10
    targets = (tokens + 1) % 4
    original_prior = prior.clone()
    logits = model(tokens, prior, independent=independent)
    assert torch.equal(logits, prior[tokens])
    assert torch.equal(logits.softmax(-1), prior[tokens].softmax(-1))
    torch.nn.functional.cross_entropy(logits.reshape(-1, 4), targets.flatten()).backward()
    assert torch.equal(prior, original_prior) and prior.grad is None
    assert any(torch.count_nonzero(parameter.grad) > 0
               for name, parameter in model.named_parameters() if name.startswith("head."))
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if not name.startswith("head."):
            assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("context", [False, True])
def test_gru_builtin_and_manual_recurrence_have_same_nonzero_body_gradients(helper, context):
    model = helper.make_gru(seed=5, vocab=4, embedding=3, hidden=5)
    with torch.no_grad():
        model.head.weight.copy_(torch.arange(20, dtype=torch.float64).reshape(4, 5) / 37)
    tokens = torch.tensor([[0, 1, 2, 1], [3, 0, 2, 3]], dtype=torch.int64)
    prior = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10
    targets = (tokens + 1) % 4
    logits = model(tokens, prior, context=context)
    oracle = model(tokens, prior, context=context, independent=True)
    torch.testing.assert_close(logits, oracle, atol=1e-13, rtol=1e-12)
    named = tuple(model.named_parameters())
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), targets.flatten())
    reference_loss = torch.nn.functional.cross_entropy(oracle.reshape(-1, 4), targets.flatten())
    actual = torch.autograd.grad(loss, tuple(parameter for _, parameter in named))
    expected = torch.autograd.grad(reference_loss, tuple(parameter for _, parameter in named))
    assert any(torch.count_nonzero(gradient) for (name, _), gradient in zip(named, actual)
               if not name.startswith("head."))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-11)


def test_gru_context_off_resets_each_position_and_batch_rows_are_independent(helper):
    model = helper.make_gru(seed=5, vocab=4, embedding=3, hidden=5)
    with torch.no_grad():
        model.head.weight.fill_(.25)
    tokens = torch.tensor([[0, 1, 2], [3, 1, 0]], dtype=torch.int64)
    prior = torch.zeros((4, 4), dtype=torch.float64)
    full = model(tokens, prior)
    for index, row in enumerate(tokens):
        torch.testing.assert_close(full[index], model(row[None, :], prior)[0], atol=1e-14, rtol=1e-13)
    isolated = torch.cat([model(tokens[:, index:index + 1], prior)
                          for index in range(tokens.shape[1])], dim=1)
    torch.testing.assert_close(model(tokens, prior, context=False), isolated, atol=1e-14, rtol=1e-13)


@pytest.mark.parametrize("case", ["attached_prior", "float_tokens", "prior_shape", "prior_dtype", "nonfinite_prior", "bad_context"])
def test_gru_forward_rejects_invalid_prior_and_tokens(helper, case):
    model = helper.make_gru(seed=0, vocab=4, embedding=3, hidden=5)
    tokens = torch.tensor([[0, 1]], dtype=torch.int64)
    prior = torch.zeros((4, 4), dtype=torch.float64)
    context = True
    if case == "attached_prior":
        prior.requires_grad_(True)
    elif case == "float_tokens":
        tokens = tokens.double()
    elif case == "prior_shape":
        prior = prior[:2]
    elif case == "prior_dtype":
        prior = prior.float()
    elif case == "nonfinite_prior":
        prior[0, 0] = float("nan")
    else:
        context = 1
    with pytest.raises(ValueError):
        model(tokens, prior, context=context)
