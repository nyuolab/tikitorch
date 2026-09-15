"""Synthetic/offline contracts for the fixed update-scale screen.

No native executable, scientific training, actual corpus or reserved cohort is
used. Lifecycle tests substitute fake execution while retaining real formats.
"""
import copy
import importlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.test_fpga_lm_update_scale")


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def evaluation(nll=2.90, context=0.04, positions=8192):
    return {"lut_normalized_nll": nll, "context_disabled_nll": nll + context,
            "saturation_events": 0, "native_reference_equal": True, "native_routes_equal": True,
            "route_diagnostics": {"positions": positions, "nonzero_logit_gradient_positions": positions,
                "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
                "head_winners_ge16_positions": positions, "backward_int32_overflow_detected": False}}


@pytest.fixture
def passing(study):
    cells = [{"arm": arm, "seed": seed, "k": k, "complete": True,
              "evaluation": evaluation(2.94 if k == 18 else 2.90)}
             for arm in study.ARMS for seed in study.SEEDS for k in study.K_VALUES]
    initial = {arm: evaluation(3.1, context=0) for arm in study.ARMS}
    controls = {"stronger": {"unigram": {"nll": 3.2}, "bigram": {"nll": 3.0}}}
    return cells, initial, controls


def candidate(cells, *, arm="diverse", seed=0, k=20):
    return next(row for row in cells if (row["arm"], row["seed"], row["k"]) == (arm, seed, k))


def test_predeclared_recipe_is_paired_bounded_and_does_not_promote(study):
    assert study.ARMS == ("historical", "diverse")
    assert study.SEEDS == (0, 1, 2)
    assert study.K_VALUES == (18, 20)
    assert (study.WIDTH, study.STEPS, study.CHECKPOINTS) == (512, 1024, (0, 256, 1024))
    assert study.VALIDATION == tuple(range(88, 152))
    assert study.RESERVED == tuple(range(152, 864))
    assert study.PARITY_STEPS == (1, 257)
    recipe = study.recipe()
    assert recipe["selection_checkpoint"] == 1024
    assert recipe["primary_initialization"] == "diverse"
    assert "not a novel" in recipe["prior_counterevidence"]
    assert "no control retraining" in recipe["control_policy"]
    assert recipe["new_native_updates"] == 6144
    assert recipe["new_gradient_pair_exposures"] == 786432
    assert recipe["inherited_control_gradient_pair_exposures"] == 786432
    assert recipe["summed_inherited_fit_pair_exposures"] == 1572864
    assert recipe["summed_model_history_pair_exposures"] == 3145728
    assert recipe["new_fit_pair_exposures"] == 0
    assert recipe["distinct_training_pairs"] == 131072
    for key in ("reserved_cohort_evaluated", "heldout_test_evaluated", "paid_execution_authorized",
                "physical_fpga_execution", "statistical_significance_claim", "adaptive_extension"):
        assert recipe[key] is False


def test_recipe_retains_old_mechanism_and_stronger_quality_thresholds(study):
    thresholds = study.recipe()["thresholds"]
    assert thresholds == {"mean_paired_improvement_nats": 0.01, "minimum_paired_improvement_each_seed": 0.0,
        "context_nats_each_seed": 0.02, "context_growth_nats_each_seed": 0.02,
        "unigram_advantage_nats_each_seed": 0.05, "mean_bigram_advantage_nats": 0.02,
        "minimum_bigram_advantage_each_seed": 0.0, "maximum_zero_head_vjp_fraction": 0.25,
        "minimum_diverse_head_fraction": 0.75}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_order_is_exact_old_permutation_and_does_not_consume_global_rng(study, seed):
    expected = list(range(1024))
    random.Random(seed).shuffle(expected)
    before = random.getstate()
    assert study.recipe()["training_orders"][str(seed)] == expected
    assert random.getstate() == before
    assert sorted(expected) == list(range(1024))


@pytest.mark.parametrize("arm,seed,k", [("other", 0, 18), ("diverse", True, 18), ("diverse", 0.0, 18),
                                       ("diverse", 3, 18), ("diverse", 0, 19), ("diverse", 0, 20.0)])
def test_undeclared_cells_rejected(study, arm, seed, k):
    with pytest.raises(ValueError):
        study.cell_name(arm, seed, k)


def test_primary_pass_only_opens_separate_confirmation_not_promotion(study, passing):
    result = study.gate_results(*passing)
    assert result["primary_screen_pass"] is True
    assert result["historical_is_fallback_candidate"] is False
    assert result["confirmation_required_before_recipe_promotion"] is True
    assert result["recipe_promoted"] is False
    assert result["paid_execution_authorized"] is False


def test_historical_success_cannot_rescue_failed_primary(study, passing):
    cells, initial, controls = passing
    for seed in (0, 1, 2):
        candidate(cells, seed=seed)["evaluation"] = evaluation(2.94)
    result = study.gate_results(cells, initial, controls)
    assert result["by_initialization"]["historical"]["all_screen_criteria"] is True
    assert result["primary_screen_pass"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "incomplete", "truthy_complete", "bool_seed", "float_k"])
def test_all_twelve_literal_complete_ordered_cells_required(study, passing, mutation):
    cells, initial, controls = passing
    if mutation == "missing":
        cells.pop()
    elif mutation == "duplicate":
        cells[-1] = copy.deepcopy(cells[-2])
    elif mutation == "reordered":
        cells[0], cells[1] = cells[1], cells[0]
    elif mutation in ("incomplete", "truthy_complete"):
        cells[-1]["complete"] = False if mutation == "incomplete" else 1
    elif mutation == "bool_seed":
        cells[0]["seed"] = False
    else:
        cells[0]["k"] = 18.0
    with pytest.raises(ValueError):
        study.gate_results(cells, initial, controls)


def test_mean_improvement_does_not_hide_an_order_regression(study, passing):
    cells, initial, controls = passing
    for seed, nll in enumerate((2.80, 2.95, 2.80)):
        candidate(cells, seed=seed)["evaluation"] = evaluation(nll)
    result = study.gate_results(cells, initial, controls)
    assert result["by_initialization"]["diverse"]["mean_paired_improvement_nats"] > 0.01
    assert result["primary_screen_pass"] is False


def test_paired_improvement_must_reach_declared_mean(study, passing):
    cells, initial, controls = passing
    for seed in (0, 1, 2):
        candidate(cells, seed=seed)["evaluation"] = evaluation(2.935)
    assert study.gate_results(cells, initial, controls)["primary_screen_pass"] is False


def test_stronger_bigram_must_not_be_lost_by_one_seed(study, passing):
    cells, initial, controls = passing
    for seed, nll in enumerate((2.8, 3.001, 2.8)):
        candidate(cells, seed=seed)["evaluation"] = evaluation(nll)
        candidate(cells, seed=seed, k=18)["evaluation"] = evaluation(nll + 0.03)
    result = study.gate_results(cells, initial, controls)
    assert result["by_initialization"]["diverse"]["mean_bigram_advantage_nats"] > 0.02
    assert result["primary_screen_pass"] is False


@pytest.mark.parametrize("field,value", [("saturation_events", 1), ("saturation_events", False),
    ("native_reference_equal", False), ("native_reference_equal", 1), ("native_routes_equal", False)])
def test_any_numerical_mechanism_failure_blocks(study, passing, field, value):
    cells, initial, controls = passing
    candidate(cells)["evaluation"][field] = value
    assert study.gate_results(cells, initial, controls)["primary_screen_pass"] is False


@pytest.mark.parametrize("field,value", [("nonzero_logit_gradient_positions", 0),
    ("exact_zero_head_input_vjp_on_nonzero_gradient_positions", 2049),
    ("head_winners_ge16_positions", 6143), ("backward_int32_overflow_detected", True),
    ("backward_int32_overflow_detected", 0)])
def test_exact_mechanism_thresholds_are_not_weakened(study, passing, field, value):
    cells, initial, controls = passing
    candidate(cells)["evaluation"]["route_diagnostics"][field] = value
    assert study.gate_results(cells, initial, controls)["primary_screen_pass"] is False


@pytest.mark.parametrize("field,value", [("positions", 2048), ("positions", 8192.0),
    ("nonzero_logit_gradient_positions", True), ("nonzero_logit_gradient_positions", 8193),
    ("exact_zero_head_input_vjp_on_nonzero_gradient_positions", -1), ("head_winners_ge16_positions", 8193)])
def test_mechanism_count_types_bounds_and_64_window_coverage(study, passing, field, value):
    cells, initial, controls = passing
    candidate(cells)["evaluation"]["route_diagnostics"][field] = value
    with pytest.raises(ValueError):
        study.gate_results(cells, initial, controls)


@pytest.mark.parametrize("failure", ["final_context", "growth", "unigram"])
def test_each_context_requirement_is_retained(study, passing, failure):
    cells, initial, controls = passing
    if failure == "final_context":
        candidate(cells)["evaluation"] = evaluation(2.90, context=0.019)
    elif failure == "growth":
        initial["diverse"] = evaluation(3.1, context=0.025)
    else:
        controls["stronger"]["unigram"]["nll"] = 2.94
    assert study.gate_results(cells, initial, controls)["primary_screen_pass"] is False


@pytest.mark.parametrize("field,value", [("lut_normalized_nll", True), ("context_disabled_nll", False),
    ("lut_normalized_nll", float("nan")), ("lut_normalized_nll", float("inf")),
    ("lut_normalized_nll", -0.1)])
def test_gate_losses_are_finite_nonnegative_numbers_not_booleans(study, passing, field, value):
    cells, initial, controls = passing
    candidate(cells)["evaluation"][field] = value
    with pytest.raises(ValueError):
        study.gate_results(cells, initial, controls)


def payload(study, shape, *, k=18, step=0, remainder=0, value=3, loss_kind=1, zshift=2):
    source = study.native.source_digest(study.native.source_identity())
    words = [int(source[i:i + 8], 16) for i in range(0, 64, 8)]
    words = [x if x < 2**31 else x - 2**32 for x in words]
    count = sum(getattr(shape, "WORDS_" + name) for name in study.pilot.WEIGHTS)
    header = [study.native.MAGIC, 1, shape.ML_D, shape.ML_BUFFER_WORDS, count, k,
              loss_kind, zshift, step, 0, shape.WORDS_CE_LUT, 0, *words]
    lut = [round(65536 * 2.0 ** (-i / 16)) for i in range(256)]
    return study.native._pack(header + lut + [value] * count + [remainder] * count)


@pytest.mark.parametrize("k", [18, 20])
def test_initial_buffer_preserves_full_weights_lut_and_cfg_without_mutation(study, k):
    shape = study.pilot.layout(16)
    original = payload(study, shape)
    result = study.initial_buffer(original, shape, k)
    for name in study.pilot.WEIGHTS:
        start, words = getattr(shape, "OFF_" + name), getattr(shape, "WORDS_" + name)
        assert result[start:start + words] == [3] * words
    assert result[shape.OFF_CFG:shape.OFF_CFG + 2] == [1, 2]
    assert result[shape.OFF_CE_LUT:shape.OFF_CE_LUT + 256] == [round(65536 * 2.0 ** (-i / 16)) for i in range(256)]
    assert original == payload(study, shape)


@pytest.mark.parametrize("change", [{"step": 1}, {"k": 20}, {"remainder": 1}, {"loss_kind": 0}, {"zshift": 4}])
def test_init_only_accepts_original_k18_zero_state(study, change):
    shape = study.pilot.layout(16)
    with pytest.raises(ValueError):
        study.initial_buffer(payload(study, shape, **change), shape, 20)


@pytest.mark.parametrize("k", [True, 18.0, 19, 21])
def test_initial_shift_is_strict_declared_integer(study, k):
    shape = study.pilot.layout(16)
    with pytest.raises(ValueError):
        study.initial_buffer(payload(study, shape), shape, k)


@pytest.fixture
def synthetic_corpus(study, tmp_path):
    rng = random.Random(2817)
    raw = bytes(rng.randrange(128) for _ in range(200_000))
    path = tmp_path / "synthetic.txt"
    path.write_bytes(raw)
    _, dataset = study.pilot.corpus_plan(path, 1024, 152)
    return raw, dataset


def baseline_oracle(raw, dataset, passes, validation):
    uni, bi = [1] * 128, [[1] * 128 for _ in range(128)]
    for index in range(1024):
        begin = dataset["splits"]["train"]["start"] + 129 * index
        for offset in range(begin, begin + 128):
            uni[raw[offset + 1]] += passes
            bi[raw[offset]][raw[offset + 1]] += passes
    result = {}
    for name, counts in (("unigram", [uni] * 128), ("bigram", bi)):
        totals = list(map(sum, counts))
        losses = []
        for index in validation:
            start = dataset["splits"]["validation"]["start"] + 129 * index
            losses.extend(-math.log(counts[raw[j]][raw[j + 1]] / totals[raw[j]]) for j in range(start, start + 128))
        result[name] = sum(losses) / len(losses)
    return result


def test_one_pass_and_replay_baselines_use_exact_exposure_budgets(study, synthetic_corpus):
    raw, dataset = synthetic_corpus
    actual = study.baselines(raw, dataset)
    for passes in (1, 2):
        expected = baseline_oracle(raw, dataset, passes, range(88, 152))
        assert actual[str(passes)]["exposures"] == passes * 131072
        assert actual[str(passes)]["distinct_pairs"] == 131072
        assert actual[str(passes)]["alpha"] == 1
        for name in ("unigram", "bigram"):
            assert actual[str(passes)][name]["nll"] == pytest.approx(expected[name], abs=1e-12)
            assert [x["window_index"] for x in actual[str(passes)][name]["per_window"]] == list(range(88, 152))
    for name in ("unigram", "bigram"):
        assert actual["stronger"][name]["nll"] == min(actual[str(p)][name]["nll"] for p in (1, 2))


def test_baselines_do_not_use_reserved_test_or_unselected_training_bytes(study, synthetic_corpus):
    raw, dataset = synthetic_corpus
    changed = bytearray(raw)
    stop = 1024 * 129
    changed[stop:dataset["splits"]["train"]["stop"]] = b"!" * (dataset["splits"]["train"]["stop"] - stop)
    reserve = dataset["splits"]["validation"]["start"] + 152 * 129
    changed[reserve:] = b"?" * (len(changed) - reserve)
    assert study.baselines(bytes(changed), dataset) == study.baselines(raw, dataset)


@pytest.fixture
def training_log(study, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "STEPS", 3)
    order = [2, 0, 1]
    logs, payloads = [], {"predictions": b"", "gradients": b""}
    for step, index in enumerate(order, 1):
        row = {"step": step, "window_index": index}
        for name, words in (("predictions", 128), ("gradients", 16384)):
            data = np.full(words, step, dtype="<i4").tobytes()
            payloads[name] += data
            row[name + "_sha256"] = study.pilot.digest(data)
        logs.append(row)
    (tmp_path / "training.jsonl").write_text("".join(json.dumps(x) + "\n" for x in logs))
    for name, data in payloads.items():
        (tmp_path / ("training_" + name + ".i32")).write_bytes(data)
    return SimpleNamespace(root=tmp_path, order=order, logs=logs)


def test_every_training_array_chunk_is_hashed_against_exact_order(study, training_log):
    study.verify_training(training_log.root, training_log.order)


@pytest.mark.parametrize("mutation", ["missing_line", "extra_line", "wrong_order", "bool_step", "wrong_hash",
                                       "short_predictions", "short_gradients", "changed_prediction", "changed_gradient"])
def test_training_coverage_or_hash_tampering_is_rejected(study, training_log, mutation):
    case = training_log
    if mutation in ("missing_line", "extra_line", "wrong_order", "bool_step", "wrong_hash"):
        if mutation == "missing_line":
            case.logs.pop()
        elif mutation == "extra_line":
            case.logs.append(case.logs[-1])
        elif mutation == "wrong_order":
            case.logs[1]["window_index"] = 1
        elif mutation == "bool_step":
            case.logs[0]["step"] = True
        else:
            case.logs[1]["gradients_sha256"] = "0" * 64
        (case.root / "training.jsonl").write_text("".join(json.dumps(x) + "\n" for x in case.logs))
    else:
        name = "predictions" if "prediction" in mutation else "gradients"
        path = case.root / ("training_" + name + ".i32")
        data = path.read_bytes()
        path.write_bytes(data[:-4] if mutation.startswith("short") else bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError):
        study.verify_training(case.root, case.order)


def observed_fixture(weights, tokens, targets, shape, **kwargs):
    result = {"predictions": list(tokens), "gradients": [0] * 16384,
        "native_routes": {name: [0] * getattr(shape, "WORDS_" + name) for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")},
        "lut_normalized_nll": 2.9 + weights["W_HEAD"][0] / 100, "accuracy": .25,
        "saturation_events": 0, "rounded_zero_target_mass": 0,
        "route_diagnostics": {"positions": 128, "nonzero_logit_gradient_positions": 128,
            "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0, "head_winners_ge16_positions": 128,
            "single_shared_head_winner_positions": 0, "nonzero_embedding_input_vjp_positions": 128,
            "max_abs_head_input_vjp": 3, "max_abs_embedding_input_vjp": 3,
            "backward_int32_overflow_detected": False, "head_winner_counts": [16] * 128}}
    if kwargs.get("include_parameter_gradients"):
        result["parameter_gradients"] = {name: [0] * len(value) for name, value in weights.items()}
        result["parameter_gradients"]["W_HEAD"][0] = 1 << 20
    return result


@pytest.fixture
def fake_evaluation(study, synthetic_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "VALIDATION", (88, 89))
    shape = study.pilot.layout(16)
    start = payload(study, shape)
    controls = SimpleNamespace(wrong_array=False, wrong_step=False, mutate=False, import_mutate=False)

    class Endpoint:
        # Deliberately no k attribute, matching the actual frozen native API.
        def __init__(self):
            self.current = start
            self.evaluated = 0

        def export_state(self):
            return self.current

        def import_state(self, value):
            self.current = payload(study, shape, value=9) if controls.import_mutate else value

        def evaluate(self, tokens, targets):
            self.evaluated += 1
            state = study.native.decode_state(self.current, shape)
            row = observed_fixture(state["weights"], tokens, targets, shape)
            result = {key: row[key] for key in ("predictions", "gradients", "native_routes")}
            result["step"] = 1 if controls.wrong_step else state["step"]
            if controls.wrong_array:
                result["predictions"][0] ^= 1
            if controls.mutate:
                self.current = payload(study, shape, value=9)
            return result

    monkeypatch.setattr(study.old, "_checked_observer", observed_fixture)

    def probe(weights, tokens, targets, layout):
        result = observed_fixture(weights, tokens, targets, layout)
        result["lut_normalized_nll"] += .04
        return study.screen._scalar_result(result)

    monkeypatch.setattr(study.old, "_probe", probe)
    return SimpleNamespace(shape=shape, endpoint=Endpoint(), controls=controls, root=tmp_path,
                           raw=synthetic_corpus[0], dataset=synthetic_corpus[1], payload=start)


def run_evaluation(study, case):
    return study.evaluate(case.endpoint, case.root, case.raw, case.dataset, case.shape, 0, 18)


def test_evaluation_full_arrays_aggregate_and_import_invariance(study, fake_evaluation):
    case = fake_evaluation
    result = run_evaluation(study, case)
    assert result["lut_normalized_nll"] == pytest.approx(2.93)
    assert result["context_disabled_nll"] == pytest.approx(2.97)
    assert result["route_diagnostics"]["positions"] == 256
    assert result["full_state_evaluation_invariant"] is True
    assert result["full_state_import_equal"] is True
    assert case.endpoint.evaluated == 2
    assert case.endpoint.export_state() == case.payload
    assert study.verify_evaluation(case.root, case.payload, case.raw, case.dataset, case.shape, 0, 18, True) == result


@pytest.mark.parametrize("failure", ["wrong_array", "wrong_step", "mutate", "import_mutate"])
def test_evaluation_mismatch_or_state_mutation_fails(study, fake_evaluation, failure):
    case = fake_evaluation
    setattr(case.controls, failure, True)
    with pytest.raises(ValueError):
        run_evaluation(study, case)


def test_evaluation_integrity_mode_does_not_recompute_model(study, fake_evaluation, monkeypatch):
    case = fake_evaluation
    result = run_evaluation(study, case)

    def forbidden(*args, **kwargs):
        raise AssertionError("no forward computation allowed in integrity mode")

    monkeypatch.setattr(study.old, "_checked_observer", forbidden)
    monkeypatch.setattr(study.old, "_probe", forbidden)
    assert study.verify_evaluation(case.root, case.payload, case.raw, case.dataset, case.shape, 0, 18, False) == result


@pytest.mark.parametrize("tamper", ["window", "missing_key", "short_array", "float_array", "matrix_array", "wrong_prediction"])
def test_evaluation_arrays_and_window_binding_fail_closed(study, fake_evaluation, tamper):
    case = fake_evaluation
    run_evaluation(study, case)
    if tamper == "window":
        path = case.root / "eval_088.json"
        row = study.read(path)
        row["window_index"] = 89
        write_json(path, row)
    else:
        path = case.root / "eval_088.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        if tamper == "missing_key":
            arrays.pop("TAP_IDX")
        elif tamper == "short_array":
            arrays["gradients"] = arrays["gradients"][:-1]
        elif tamper == "float_array":
            arrays["predictions"] = arrays["predictions"].astype(np.float64)
        elif tamper == "matrix_array":
            arrays["predictions"] = arrays["predictions"].reshape(1, -1)
        else:
            arrays["predictions"][0] ^= 1
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError):
        study.verify_evaluation(case.root, case.payload, case.raw, case.dataset, case.shape, 0, 18, tamper == "wrong_prediction")


def test_evaluation_publication_is_append_only(study, fake_evaluation):
    run_evaluation(study, fake_evaluation)
    original = (fake_evaluation.root / "eval_088.npz").read_bytes()
    with pytest.raises(FileExistsError):
        run_evaluation(study, fake_evaluation)
    assert (fake_evaluation.root / "eval_088.npz").read_bytes() == original


def encode_current(study, shape, state):
    base = list(study.native._unpack(payload(study, shape, k=state["k"], step=state["step"])))
    values = base[:20 + 256]
    for section in ("weights", "remainders"):
        for name in study.pilot.WEIGHTS:
            values.extend(state[section][name])
    return study.native._pack(values)


@pytest.fixture
def fake_run(study, synthetic_corpus, tmp_path, monkeypatch):
    """Twelve reduced cells with fake native arithmetic and actual state/IO ABI."""
    monkeypatch.setattr(study, "WIDTH", 16)
    monkeypatch.setattr(study, "STEPS", 4)
    monkeypatch.setattr(study, "CHECKPOINTS", (0, 2, 4))
    monkeypatch.setattr(study, "PARITY_STEPS", (1, 3))
    monkeypatch.setattr(study, "VALIDATION", (88, 89))
    monkeypatch.setattr(study.old, "training_order", lambda seed: [*range(seed, 4), *range(seed)])
    monkeypatch.setattr(study.old, "_checked_observer", observed_fixture)
    monkeypatch.setattr(study.old, "observe", observed_fixture)

    def probe(weights, tokens, targets, shape):
        result = observed_fixture(weights, tokens, targets, shape)
        result["lut_normalized_nll"] += .04
        return study.screen._scalar_result(result)

    monkeypatch.setattr(study.old, "_probe", probe)
    shape = study.pilot.layout(16)
    history = tmp_path / "old_complete"
    history.mkdir()
    executable = history / "fake_native"
    executable.write_bytes(b"fake binary; never executed")
    build = history / "build.json"
    write_json(build, {"synthetic": True})
    sources = {"test_fixture": "frozen"}
    write_json(history / "manifest.json", {"schema": "synthetic-history"})
    public = tmp_path / "old_record.json"
    write_json(public, {"complete": True, "synthetic": True})
    native_identity = {"schema": study.native.SCHEMA, "executable_sha256": study.pilot.digest(executable.read_bytes()),
        "build_sha256": study.pilot.digest(build.read_bytes()), "source_hashes": study.native.source_identity(),
        "layout": study.pilot.layout_values(shape), "cpu_only": True, "existing_afi_compatible": False}
    monkeypatch.setattr(study.old, "_native_identity", lambda *args: native_identity)
    states, paths = {}, [Path(synthetic_corpus[1]["path"]), executable, build, history / "manifest.json", public]
    for arm in study.ARMS:
        for seed in study.SEEDS:
            name = study.cell_name(arm, seed, 18)
            directory = history / name
            directory.mkdir()
            states[name] = {}
            for step in study.CHECKPOINTS:
                path = directory / f"state_{step:06d}.i32"
                path.write_bytes(payload(study, shape, step=step))
                states[name][str(step)] = str(path)
                paths.append(path)
    binding = {"ancestry": {"prior_run": str(history), "prior_record": str(public)},
        "inputs": [study.identity(path) for path in paths], "states": states,
        "dataset": synthetic_corpus[1], "executable": {"path": str(executable)}}
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(sources))
    monkeypatch.setattr(study, "bind_inputs", lambda config: (copy.deepcopy(binding), synthetic_corpus[0]))
    controls = SimpleNamespace(endpoints=[], train_counts={18: 0, 20: 0}, fail_train=False,
                               wrong_native_array=False, closed=0)
    native_type = study.native.EFSGD

    class Endpoint:
        def __init__(self, executable, layout, buffer, k, **kwargs):
            self.shape, self.identity, self.closed = layout, native_identity, False
            self.sources = study.native.source_identity()
            self.state = study.native.decode_state(payload(study, shape, k=k), shape)
            for name in study.pilot.WEIGHTS:
                start, words = getattr(shape, "OFF_" + name), getattr(shape, "WORDS_" + name)
                self.state["weights"][name] = list(buffer[start:start + words])
            Path(kwargs["log_path"]).write_bytes(b"fake native log")
            controls.endpoints.append(self)

        def export_state(self):
            return encode_current(study, shape, self.state)

        def import_state(self, value):
            self.state = study.native.decode_state(value, shape)

        def evaluate(self, tokens, targets, **kwargs):
            row = observed_fixture(self.state["weights"], tokens, targets, shape)
            return {"step": self.state["step"], **{key: row[key] for key in ("predictions", "gradients", "native_routes")}}

        def train(self, tokens, targets, **kwargs):
            if controls.fail_train:
                raise RuntimeError("synthetic native failure")
            result = self.evaluate(tokens, targets)
            if controls.wrong_native_array:
                result["predictions"][0] ^= 1
            gradients = observed_fixture(self.state["weights"], tokens, targets, shape,
                                         include_parameter_gradients=True)["parameter_gradients"]
            for name in study.pilot.WEIGHTS:
                w, r = study.fixed_point_ef_step(torch.tensor(self.state["weights"][name], dtype=torch.int16),
                    torch.tensor(self.state["remainders"][name], dtype=torch.int32),
                    torch.tensor(gradients[name], dtype=torch.int32), k=self.state["k"])
                self.state["weights"][name], self.state["remainders"][name] = w.tolist(), r.tolist()
            self.state["step"] += 1
            controls.train_counts[self.state["k"]] += 1
            result["step"] = self.state["step"]
            return result

        save_checkpoint = native_type.save_checkpoint

        def close(self):
            if not self.closed:
                controls.closed += 1
                self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(study.native, "EFSGD", Endpoint)
    plan = tmp_path / "plan.md"
    plan.write_text("Frozen synthetic unit fixture, not a research plan\n")
    monkeypatch.setattr(study, "PLAN", plan)
    config = SimpleNamespace(output=tmp_path / "new_run", record=tmp_path / "new_record.json",
        prior_run=history, prior_record=public, corpus=Path(synthetic_corpus[1]["path"]))
    return SimpleNamespace(config=config, controls=controls, binding=binding, shape=shape, sources=sources,
                           history=history, raw=synthetic_corpus[0], dataset=synthetic_corpus[1])


def test_all_twelve_fake_cells_only_six_train_and_all_native_handles_close(study, fake_run):
    case = fake_run
    result = study.run(case.config)
    assert result["complete"] is True
    assert len(result["cells"]) == 12
    assert case.controls.train_counts == {18: 0, 20: 24}
    assert len(case.controls.endpoints) == case.controls.closed == 14
    assert result["reserved_cohort_evaluated"] is False
    assert result["gates"]["recipe_promoted"] is False
    assert study.check(case.config.output, case.config.record, recompute=True) == result
    assert not list(case.config.output.rglob("eval_152*"))


def test_execution_failure_is_retained_without_public_success(study, fake_run):
    case = fake_run
    case.controls.fail_train = True
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        study.run(case.config)
    assert study.read(case.config.output / "failure.json")["complete"] is False
    assert not case.config.record.exists()
    assert all(endpoint.closed for endpoint in case.controls.endpoints)


def test_parity_native_array_mismatch_prevents_publication(study, fake_run):
    fake_run.controls.wrong_native_array = True
    with pytest.raises(ValueError):
        study.run(fake_run.config)
    assert not fake_run.config.record.exists()
    assert (fake_run.config.output / "failure.json").is_file()


@pytest.fixture
def completed_run(study, fake_run):
    study.run(fake_run.config)
    return fake_run


def validate_completed(study, case, recompute=False):
    root = case.config.output
    return study.validate_summary(root, study.read(root / "manifest.json"),
                                  study.read(root / "summary.json"), recompute)


@pytest.mark.parametrize("field", ["schema", "file", "bytes", "sha256", "step", "k", "config", "identity", "regions", "format"])
def test_full_checkpoint_metadata_is_bound(study, completed_run, field):
    path = completed_run.config.output / "historical_seed_0_k18/state_000004.json"
    row = study.read(path)
    row[field] = None
    write_json(path, row)
    with pytest.raises(ValueError):
        validate_completed(study, completed_run)


@pytest.mark.parametrize("tamper", ["missing_region", "short_gradient", "missing_route", "short_route", "extra_output", "wrong_prediction"])
def test_saved_parity_full_array_coverage_without_recompute(study, completed_run, tamper):
    root = completed_run.config.output / "historical_seed_0_k20"
    kind = "gradients" if tamper in ("missing_region", "short_gradient") else "outputs"
    path = root / f"parity_0001_{kind}.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if tamper == "missing_region":
        arrays.pop("MH")
    elif tamper == "short_gradient":
        arrays["MH"] = arrays["MH"][:-1]
    elif tamper == "missing_route":
        arrays.pop("TAP_IDX")
    elif tamper == "short_route":
        arrays["TAP_IDX"] = arrays["TAP_IDX"][:-1]
    elif tamper == "extra_output":
        arrays["extra"] = np.zeros(1, dtype="<i4")
    else:
        arrays["predictions"][0] ^= 1
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises((ValueError, TypeError)):
        validate_completed(study, completed_run)


@pytest.mark.parametrize("directory", ["initial_historical", "historical_seed_0_k18", "diverse_seed_0_k20"])
@pytest.mark.parametrize("field,value", [("native_reference_equal", 1), ("native_routes_equal", False),
                                        ("saturation_events", False), ("backward_int32_overflow_detected", 0)])
def test_every_initial_control_candidate_window_requires_literal_guards(study, completed_run, directory, field, value):
    path = completed_run.config.output / directory / "eval_088.json"
    row = study.read(path)
    if field == "backward_int32_overflow_detected":
        row["route_diagnostics"][field] = value
    else:
        row[field] = value
    write_json(path, row)
    with pytest.raises(ValueError):
        validate_completed(study, completed_run)


@pytest.mark.parametrize("tamper", ["source", "omitted_input", "redirected_state", "input_bytes", "recipe"])
def test_frozen_reconstructs_complete_ancestry_and_actual_input_bytes(study, fake_run, tamper):
    case = fake_run
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(),
                "source_hashes": copy.deepcopy(case.sources), "binding": copy.deepcopy(case.binding)}
    study.frozen(manifest)
    if tamper == "source":
        manifest["source_hashes"]["test_fixture"] = "changed"
    elif tamper == "omitted_input":
        manifest["binding"]["inputs"].pop()
    elif tamper == "redirected_state":
        states = manifest["binding"]["states"]
        states["historical_seed_0_k18"]["0"] = states["historical_seed_1_k18"]["0"]
    elif tamper == "input_bytes":
        Path(case.binding["executable"]["path"]).write_bytes(b"changed never-executed fixture")
    else:
        manifest["recipe"]["paid_execution_authorized"] = True
    with pytest.raises(ValueError):
        study.frozen(manifest)


@pytest.mark.parametrize("tamper", ["public", "extra_artifact", "changed_artifact", "seal", "failure"])
def test_check_binds_public_receipt_complete_tree_and_seal(study, completed_run, tamper):
    case = completed_run
    if tamper == "public":
        row = study.read(case.config.record)
        row["paid_execution_authorized"] = True
        write_json(case.config.record, row)
    elif tamper == "extra_artifact":
        (case.config.output / "extra.txt").write_text("unsealed")
    elif tamper == "changed_artifact":
        (case.config.output / "declaration.md").write_text("changed declaration")
    elif tamper == "seal":
        write_json(case.config.output / "artifact_manifest.json", [])
    else:
        write_json(case.config.output / "failure.json", {"complete": False})
        write_json(case.config.output / "artifact_manifest.json", study.ancestry.artifact_manifest(case.config.output))
    with pytest.raises(ValueError):
        study.check(case.config.output, case.config.record)


@pytest.mark.parametrize("when", ["before_link", "after_link", "foreign_target"])
def test_atomic_publication_failures_retract_only_owned_receipt(study, fake_run, monkeypatch, when):
    case = fake_run
    original_link = study.os.link
    previous = study.signal.getsignal(study.signal.SIGALRM)

    def interrupted_link(source, target):
        if when == "after_link":
            original_link(source, target)
            raise KeyboardInterrupt("synthetic interrupt after atomic link")
        if when == "foreign_target":
            Path(target).write_bytes(b"unrelated concurrent receipt")
            original_link(source, target)
        raise OSError("synthetic link failure before publication")

    monkeypatch.setattr(study.os, "link", interrupted_link)
    with pytest.raises((OSError, KeyboardInterrupt)):
        study.run(case.config)
    assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    assert study.read(case.config.output / "failure.json")["complete"] is False
    assert all(endpoint.closed for endpoint in case.controls.endpoints)
    if when == "foreign_target":
        assert case.config.record.read_bytes() == b"unrelated concurrent receipt"
    else:
        assert not case.config.record.exists()
    assert (case.config.output / "rejected_public.json").exists() is (when == "after_link")
    assert not list(case.config.record.parent.glob(".tt-update-scale-*"))


def test_failed_final_verification_retracts_owned_publication(study, fake_run, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("synthetic final verification failure")

    monkeypatch.setattr(study, "check", reject)
    with pytest.raises(ValueError, match="synthetic final verification failure"):
        study.run(fake_run.config)
    assert not fake_run.config.record.exists()
    assert (fake_run.config.output / "rejected_public.json").is_file()
    assert (fake_run.config.output / "failure.json").is_file()


def test_permanent_staging_cleanup_failure_still_restores_alarm(study, fake_run, monkeypatch):
    previous = study.signal.getsignal(study.signal.SIGALRM)
    original_unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if path.name == "receipt.json" and path.parent.name.startswith(".tt-update-scale-"):
            raise OSError("synthetic permanent staging cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(OSError, match="synthetic permanent staging cleanup failure"):
        study.run(fake_run.config)
    assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    assert not fake_run.config.record.exists()
    assert (fake_run.config.output / "failure.json").is_file()


def test_signal_setup_failure_is_retained_before_native_creation(study, fake_run, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic signal setup failure")

    monkeypatch.setattr(study.signal, "getsignal", fail)
    with pytest.raises(RuntimeError, match="synthetic signal setup failure"):
        study.run(fake_run.config)
    assert (fake_run.config.output / "failure.json").is_file()
    assert not fake_run.config.record.exists()
    assert not fake_run.controls.endpoints


@pytest.mark.parametrize("existing", ["output", "record"])
def test_existing_paths_rejected_without_overwrite_or_execution(study, fake_run, existing):
    path = getattr(fake_run.config, existing)
    path.write_bytes(b"existing user data")
    with pytest.raises((FileExistsError, ValueError)):
        study.run(fake_run.config)
    assert path.read_bytes() == b"existing user data"
    assert not fake_run.controls.endpoints
