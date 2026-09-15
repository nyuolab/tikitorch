"""Offline confidence diagnostics: exact integer recipe, no training or cloud."""
import copy
import json
import math

import numpy as np
import pytest

from manuscript.tools import diagnose_fpga_lm_calibration as diagnostic


def scalar_score(logits, targets, zshift):
    """Scalar oracle, independently spelling out the LUT/balancing arithmetic."""
    table = [round(65536 * 2 ** (-index / 16)) for index in range(256)]
    predictions, gradients, losses, confidences = [], [], [], []
    zero, at_last, above_last = [], [], []
    for row, target in zip(logits, targets):
        winner = max(range(len(row)), key=lambda index: row[index])
        bins = [(row[winner] - value) // (2 ** zshift) for value in row]
        energies = [table[min(index, 255)] for index in bins]
        total = sum(energies)
        divisor = 1
        while total // divisor > 65536:
            divisor *= 2
        rounded = [energy // divisor for energy in energies]
        zero.append(rounded[target] == 0)
        rounded[target] -= sum(rounded)
        predictions.append(winner)
        gradients.extend(rounded)
        losses.append(math.log(total) - math.log(energies[target]))
        confidences.append(energies[winner] / total)
        at_last.append(bins[target] >= 255)
        above_last.append(bins[target] > 255)
    return {
        "predictions": predictions, "gradients": gradients,
        "lut_normalized_nll": sum(losses) / len(losses),
        "accuracy": sum(p == t for p, t in zip(predictions, targets)) / len(targets),
        "rounded_zero_target_mass": sum(zero), "token_nll": losses,
        "maximum_probabilities": confidences, "rounded_zero_target_mask": zero,
        "target_index_at_or_above_last_bin_mask": at_last,
        "target_index_above_last_bin_mask": above_last}


@pytest.mark.parametrize("shift", range(5))
def test_scores_match_independent_scalar_lut_oracle(shift):
    logits = [[0, -255, -256, -32768], [10, 10, -3, -9], [-2, -1, 0, 1]]
    targets = [2, 1, 3]
    actual = diagnostic.score_logits(logits, targets, shift)
    expected = scalar_score(logits, targets, shift)
    for key, value in expected.items():
        if key in ("token_nll", "maximum_probabilities", "lut_normalized_nll"):
            assert actual[key] == pytest.approx(value, abs=1e-12, rel=0)
        else:
            assert actual[key] == value
    assert all(sum(actual["gradients"][offset:offset + 4]) == 0 for offset in range(0, 12, 4))


def test_uniform_distribution_ties_and_balancing():
    actual = diagnostic.score_logits([[7] * 128, [-9] * 128], [0, 127], 0)
    assert actual["predictions"] == [0, 0]
    assert actual["accuracy"] == 0.5
    assert actual["lut_normalized_nll"] == pytest.approx(math.log(128))
    assert actual["maximum_probabilities"] == [1 / 128, 1 / 128]
    assert actual["gradients"][:128] == [-65024] + [512] * 127
    assert actual["gradients"][128:] == [512] * 127 + [-65024]
    assert actual["rounded_zero_target_mass"] == 0


def test_zero_rounded_target_mass_does_not_mean_zero_loss_or_target_gradient():
    logits = [[0] * 127 + [-256]]
    result = diagnostic.score_logits(logits, [127], 0)
    assert result["rounded_zero_target_mass"] == 1
    assert result["gradients"][-1] < 0
    assert sum(result["gradients"]) == 0
    assert math.isfinite(result["lut_normalized_nll"])
    assert result["lut_normalized_nll"] > 10


def test_last_bin_boundary_is_distinct_from_actual_index_clamping():
    result = diagnostic.score_logits([[0, -254], [0, -255], [0, -256]], [1, 1, 1], 0)
    assert result["target_index_at_or_above_last_bin_mask"] == [False, True, True]
    assert result["target_index_above_last_bin_mask"] == [False, False, True]


def test_evaluation_scale_changes_loss_not_argmax_and_never_mutates_logits():
    logits, targets = [[0, -30, -255, -256], [3, 4, 4, -100]], [3, 1]
    original = copy.deepcopy(logits)
    results = [diagnostic.score_logits(logits, targets, shift) for shift in range(5)]
    assert all(result["predictions"] == [0, 1] for result in results)
    assert all(result["accuracy"] == 0.5 for result in results)
    assert len({result["lut_normalized_nll"] for result in results}) > 1
    assert logits == original


@pytest.mark.parametrize("shift", [True, False, -1, 31, 0.0, "0", None])
def test_invalid_scale_fails_closed(shift):
    with pytest.raises(ValueError):
        diagnostic.score_logits([[0, -1]], [1], shift)


@pytest.mark.parametrize("logits,targets", [
    ([], []), ([[]], [0]), ([[0, 1], [2]], [0, 0]),
    ([[0, 1]], []), ([[0, 1]], [0, 1]), ([[0, 1]], [-1]),
    ([[0, 1]], [2]), ([[0, 1]], [True]), ([[0, 1]], [1.0]),
    ([[0, 1.0]], [0]), ([[True, False]], [0]),
    ([[0, float("nan")]], [0]), ([[0, float("inf")]], [0]),
    ([[0, 32768]], [0]), ([[-32769, 0]], [0])])
def test_invalid_logits_or_targets_fail_closed(logits, targets):
    with pytest.raises(ValueError):
        diagnostic.score_logits(logits, targets, 0)


@pytest.mark.parametrize("shift", range(5))
def test_extracted_logits_match_frozen_integer_reference(shift):
    shape = diagnostic.pilot.layout(16)
    initial = shape.build_init(1)
    weights = {name: initial[getattr(shape, "OFF_" + name):
                             getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
               for name in diagnostic.pilot.WEIGHTS}
    tokens = list(range(128))
    targets = tokens[1:] + [0]
    extracted = diagnostic.integer_logits(weights, tokens, shape)
    scored = diagnostic.score_logits(extracted["logits"], targets, shift)
    frozen = diagnostic.pilot.integer_forward(weights, tokens, targets, shape, zshift=shift)
    assert scored["predictions"] == extracted["predictions"] == frozen["predictions"]
    assert scored["gradients"] == frozen["gradients"]
    assert scored["lut_normalized_nll"] == pytest.approx(frozen["lut_normalized_nll"], abs=1e-12)
    assert scored["rounded_zero_target_mass"] == frozen["rounded_zero_target_mass"]
    assert extracted["saturation_events"] == frozen["saturation_events"]


def test_score_accepts_exact_numpy_integers_without_float_conversion():
    result = diagnostic.score_logits(np.array([[0, -300]], dtype=np.int64),
                                     np.array([1], dtype=np.int64), 0)
    assert result["predictions"] == [0]
    assert result["target_index_above_last_bin_mask"] == [True]


def test_executable_identity_binds_new_helper_and_frozen_dependency_chain():
    identity = diagnostic.source_identity()
    for relative in ("manuscript/tools/diagnose_fpga_lm_calibration.py",
                     "manuscript/tools/audit_fpga_lm_budget.py",
                     "manuscript/tools/extend_fpga_lm_budget.py",
                     "scripts/fpga_lm_pretrain.py"):
        assert identity[relative] == diagnostic.pilot.digest((diagnostic.REPO / relative).read_bytes())


def summary_row(losses, correct, zero, at_last, above_last):
    return {"token_nll": losses, "maximum_probabilities": [0.5] * len(losses),
            "correct_mask": correct, "rounded_zero_target_mask": zero,
            "target_index_at_or_above_last_bin_mask": at_last,
            "target_index_above_last_bin_mask": above_last}


def test_pooled_quantiles_and_tail_contributions_use_tokens_not_window_means():
    windows = [summary_row([1.0, 2.0, 3.0], [True, False, False],
                           [False, False, False], [False, False, True], [False, False, False]),
               summary_row([14.0], [False], [True], [True], [True])]
    result = diagnostic.summarize_scores(windows)
    assert result["tokens"] == 4
    assert result["total_nll"] == 20
    assert result["lut_normalized_nll"] == 5
    assert result["accuracy"] == 0.25
    assert result["mean_maximum_probability"] == 0.5
    assert result["token_nll_quantiles"] == pytest.approx({
        "minimum": 1, "median": 2.5, "p90": 10.7, "p95": 12.35,
        "p99": 13.67, "maximum": 14})
    groups = result["loss_contributions"]
    assert groups["rounded_zero_target"] == {
        "count": 1, "total_nll": 14, "mean_nll": 14, "fraction_of_total_nll": 0.7,
        "nll_contribution_per_token": 3.5}
    assert groups["target_index_at_or_above_last_bin"] == {
        "count": 2, "total_nll": 17, "mean_nll": 8.5, "fraction_of_total_nll": 0.85,
        "nll_contribution_per_token": 4.25}
    assert groups["target_index_above_last_bin"] == groups["rounded_zero_target"]
    # These tails overlap; they are not an additive partition of total loss.
    assert groups["correct"]["total_nll"] + groups["incorrect"]["total_nll"] == 20


def test_empty_subgroup_and_zero_total_loss_have_explicit_null_statistics():
    result = diagnostic.summarize_scores([
        summary_row([0.0], [True], [False], [False], [False])])
    assert result["loss_contributions"]["rounded_zero_target"] == {
        "count": 0, "total_nll": 0, "mean_nll": None, "fraction_of_total_nll": None,
        "nll_contribution_per_token": 0}
    assert result["loss_contributions"]["correct"] == {
        "count": 1, "total_nll": 0, "mean_nll": 0, "fraction_of_total_nll": None,
        "nll_contribution_per_token": 0}


@pytest.mark.parametrize("field,value", [
    ("token_nll", []), ("token_nll", [float("nan")]),
    ("token_nll", [-1.0]), ("token_nll", [True]),
    ("maximum_probabilities", [0]), ("maximum_probabilities", [1.1]),
    ("maximum_probabilities", [float("inf")]),
    ("correct_mask", [1]), ("rounded_zero_target_mask", [0]),
    ("target_index_at_or_above_last_bin_mask", [1.0]),
    ("target_index_above_last_bin_mask", [True])])
def test_invalid_pooled_score_metadata_is_rejected(field, value):
    row = summary_row([1.0], [True], [False], [False], [False])
    row[field] = value
    with pytest.raises(ValueError):
        diagnostic.summarize_scores([row])


def test_empty_summary_is_not_a_successful_diagnostic():
    with pytest.raises(ValueError):
        diagnostic.summarize_scores([])


def test_diagnostic_grid_has_no_selection_training_or_gate_override():
    recipe = diagnostic.recipe()
    assert recipe["arms"] == ["anchors", "anchors_fine"]
    assert recipe["seeds"] == [0, 1, 2]
    assert recipe["checkpoints"] == [256, 1024]
    assert recipe["evaluation_lut_shifts"] == [0, 1, 2, 3, 4]
    assert recipe["validation_indices"] == list(range(24, 40))
    for name in ("training", "native_execution", "cloud_operations",
                 "heldout_test_evaluated", "historical_gates_modified"):
        assert recipe[name] is False


@pytest.mark.parametrize("payload", ['{"complete": true, "complete": false}',
                                   '{"value": NaN}', '{"nested": {"x": 1, "x": 2}}'])
def test_ambiguous_or_nonfinite_input_records_fail_closed(tmp_path, payload):
    path = tmp_path / "receipt.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        diagnostic._read_record(path)


@pytest.fixture
def canonical_case(tmp_path, monkeypatch):
    logits, targets = [[0, -255, -256, -300]], [2]
    score = diagnostic.score_logits(logits, targets, 0)
    extracted = {"logits": logits, "predictions": score["predictions"], "saturation_events": 0}
    scalars = {name: score[name] for name in ("lut_normalized_nll", "accuracy", "rounded_zero_target_mass")}
    scalars["saturation_events"] = 0
    reference = {**scalars, "predictions": score["predictions"], "gradients": score["gradients"]}
    monkeypatch.setattr(diagnostic.pilot, "integer_forward", lambda *a, **k: copy.deepcopy(reference))
    row = {**scalars, "native_arrays": {}}
    for name in ("predictions", "gradients"):
        path = tmp_path / (name + ".i32")
        path.write_bytes(diagnostic.screen._i32(score[name]))
        row["native_arrays"][name] = {"file": path.name}
    return [tmp_path, row, {}, [0], targets, None, extracted, score, 0]


def test_canonical_shift_checks_actual_saved_native_arrays(canonical_case):
    diagnostic._canonical_check(*canonical_case)


@pytest.mark.parametrize("change", ["predictions", "gradients", "nll", "argmax", "reference"])
def test_canonical_check_rejects_native_metric_or_reference_mismatch(canonical_case, change):
    path, row, _, _, _, _, extracted, score, _ = canonical_case
    if change in ("predictions", "gradients"):
        (path / (change + ".i32")).write_bytes(diagnostic.screen._i32([123]))
    elif change == "nll":
        row["lut_normalized_nll"] += 0.1
    elif change == "argmax":
        extracted["predictions"] = [1]
    else:
        score["gradients"] = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        diagnostic._canonical_check(*canonical_case)


@pytest.fixture
def recheck_case(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(b"receipt")
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"fixed data")
    (root / "manifest.json").write_bytes(b"manifest")
    (root / "artifact_manifest.json").write_bytes(b"artifacts")
    calls = []
    monkeypatch.setattr(diagnostic, "source_identity", lambda: {"source": "original"})
    monkeypatch.setattr(diagnostic.experiment, "_verify_artifacts", lambda *a: calls.append(a))
    binding = {"run_directory": str(tmp_path / "historical"), "identity": "fixed"}
    monkeypatch.setattr(diagnostic.experiment, "v1_binding", lambda *a: copy.deepcopy(binding))
    context = {"root": root, "audit_path": audit_path, "audit_bytes": b"receipt",
               "artifact_bytes": b"artifacts", "manifest_bytes": b"manifest",
               "frozen_sources": {"source": "original"},
               "manifest": {"v1_evidence": binding},
               "dataset": {"path": str(corpus), "sha256": diagnostic.pilot.digest(b"fixed data")}}
    return context, calls


def test_final_recheck_calls_full_artifact_and_historical_validators(recheck_case):
    context, calls = recheck_case
    diagnostic._recheck_inputs(context)
    assert calls == [(context["root"], diagnostic.experiment.SCHEMA)]


@pytest.mark.parametrize("change", ["source", "receipt", "manifest", "artifacts", "v1", "corpus"])
def test_late_input_changes_cannot_be_blessed(recheck_case, monkeypatch, change):
    context, _ = recheck_case
    if change == "source":
        monkeypatch.setattr(diagnostic, "source_identity", lambda: {"source": "changed"})
    elif change == "receipt":
        context["audit_path"].write_bytes(b"changed")
    elif change in ("manifest", "artifacts"):
        name = "manifest.json" if change == "manifest" else "artifact_manifest.json"
        (context["root"] / name).write_bytes(b"changed")
    elif change == "v1":
        monkeypatch.setattr(diagnostic.experiment, "v1_binding", lambda *a: {"changed": True})
    else:
        from pathlib import Path
        Path(context["dataset"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        diagnostic._recheck_inputs(context)


@pytest.mark.parametrize("destination", ["existing", "run", "historical"])
def test_cli_rejects_overwrite_or_output_within_either_immutable_run(tmp_path, monkeypatch, destination):
    root, historical = tmp_path / "run", tmp_path / "v1"
    root.mkdir()
    historical.mkdir()
    (root / "manifest.json").write_text(json.dumps({"v1_evidence": {"run_directory": str(historical)}}))
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"preserve")
    output = {"existing": existing, "run": root / "new.json", "historical": historical / "new.json"}[destination]
    calls = []
    monkeypatch.setattr(diagnostic, "diagnostic", lambda *a: calls.append(a))
    with pytest.raises(SystemExit) as raised:
        diagnostic.main(["--run", str(root), "--audit-record", str(existing), "--output", str(output)])
    assert raised.value.code == 2
    assert calls == []
    assert existing.read_bytes() == b"preserve"
