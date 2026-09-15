"""Exact scalar controls for the fixed-gradient diagnostic, not training runs."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from manuscript.tools import diagnose_fpga_lm_sign_bias as diagnostic


@pytest.fixture
def shape():
    return SimpleNamespace(ML_D=16, ML_V=4, ML_N_LAYERS=4, ML_K=3, ML_L_SEQ=8, ML_LOG2_D=4)


@pytest.fixture
def construction(shape):
    counts = [[8, 4, 2, 1], [1, 2, 4, 8], [4, 8, 1, 2], [2, 1, 8, 4]]
    weights, _ = diagnostic.capacity.construct_weights(counts, shape, 2)
    return counts, weights


def test_fixed_mass_formula_matches_scalar_position_scatter(shape, construction):
    _, weights = construction
    probabilities, totals = diagnostic.row_masses(weights, shape)
    assert probabilities.tolist() == [[32768, 16384, 8192, 4096], [4096, 8192, 16384, 32768],
                                      [16384, 32768, 4096, 8192], [8192, 4096, 32768, 16384]]
    assert totals.tolist() == [61440] * 4
    tokens, targets = [0, 0, 1, 2, 3, 0, 2, 1], [0, 1, 2, 3, 0, 2, 1, 0]
    result = diagnostic.window_gradients(tokens, targets, probabilities, totals)
    embedding = [[0] * 4 for _ in range(4)]
    logits = []
    for token, target in zip(tokens, targets):
        row = [int(probabilities[token, v]) - (int(totals[token]) if v == target else 0) for v in range(4)]
        logits.append(row)
        for v, value in enumerate(row):
            embedding[token][v] += value
    assert result["logits"].tolist() == logits
    assert result["embedding"].tolist() == embedding
    assert result["head"].tolist() == [sum(row[v] for row in embedding) for v in range(4)]
    assert result["occurrences"].tolist() == [3, 2, 2, 1]
    assert not result["embedding"].sum(axis=1).any()


@pytest.mark.parametrize("bad", [True, -1, 4, 1.0, "1"])
def test_nonliteral_or_out_of_vocabulary_tokens_fail(shape, construction, bad):
    _, weights = construction
    probabilities, totals = diagnostic.row_masses(weights, shape)
    with pytest.raises(ValueError):
        diagnostic.window_gradients([bad], [0], probabilities, totals)


@pytest.mark.parametrize("change", ["float", "bad_sum", "negative", "bad_shape", "too_large"])
def test_invalid_probability_mass_state_fails(change):
    probabilities = np.array([[1, 1], [1, 1]])
    totals = np.array([2, 2])
    if change == "float":
        probabilities = probabilities.astype(float)
    elif change == "bad_sum":
        totals[0] = 3
    elif change == "negative":
        probabilities[0, 0] = -1
    elif change == "bad_shape":
        probabilities = probabilities[:, :1]
    else:
        totals[0] = 65537
    with pytest.raises(ValueError):
        diagnostic.window_gradients([0], [0], probabilities, totals)


@pytest.mark.parametrize("value,quotient,remainder", [(3, 1, 1), (-3, -1, -1), (1, 0, 1), (-1, 0, -1), (0, 0, 0)])
def test_signed_division_is_truncation_toward_zero(value, quotient, remainder):
    q, r = diagnostic.remainder_step(np.array([0]), np.array([value]), 1)
    assert q.tolist() == [quotient]
    assert r.tolist() == [remainder]


@pytest.mark.parametrize("k", [True, 1.0, -1, 31])
def test_invalid_update_shift_rejected(k):
    with pytest.raises(ValueError):
        diagnostic.remainder_step(np.array([0]), np.array([1]), k)


def test_noncanonical_remainder_and_int32_overflow_fail_closed():
    with pytest.raises(ValueError, match="canonical"):
        diagnostic.remainder_step(np.array([2]), np.array([1]), 1)
    with pytest.raises(ValueError, match="overflow"):
        diagnostic.remainder_step(np.array([1]), np.array([(1 << 31) - 1]), 1)
    with pytest.raises(ValueError):
        diagnostic.remainder_step(np.array([0.0]), np.array([1]), 1)


def test_error_feedback_telescope_does_not_claim_truncation_of_grand_total():
    accounting = diagnostic.GradientAccounting([0], threshold=0, shifts=(1,))
    # Signed remainder state can make the total step differ by one from
    # truncating the final aggregate. The telescoping identity is the invariant.
    accounting.add(np.array([3]))
    accounting.add(np.array([-2]))
    result = accounting.finish()
    candidate = result["fixed_gradient_error_feedback"]["1"]
    assert result["sum_gradient"] == [1]
    assert candidate["sum_steps"] == [1]  # trunc(1/2) would instead be0
    assert candidate["remainder"] == [-1]
    assert candidate["exact_telescoping_identity"] is True


def test_native_deadband_has_nonzero_drift_at_exact_zero_aggregate_gradient():
    accounting = diagnostic.GradientAccounting([0])
    # One class of the native uniform128-way gradient; a balanced finite target
    # stream has127 non-target occurrences then one target occurrence.
    for _ in range(127):
        accounting.add(np.array([512]))
    accounting.add(np.array([-65024]))
    result = accounting.finish()
    assert result["sum_gradient"] == [0]
    assert result["sum_thresholded_sign"] == [-1]
    assert result["positive_active_windows"] == [0]
    assert result["negative_active_windows"] == [1]
    assert result["deadband_nonzero_gradient_windows"] == [127]
    assert result["nonzero_sign_drift_on_exact_zero_sum_gradient_coordinates"] == 1
    assert result["mean_gradient_sign_drift_cosine"] is None
    for candidate in result["fixed_gradient_error_feedback"].values():
        assert candidate["sum_steps"] == candidate["remainder"] == [0]


def test_threshold_comparison_is_strict_and_mean_drift_can_oppose_gradient():
    accounting = diagnostic.GradientAccounting([0])
    for value in (3000, 3000, -10000, 2048, -2048):
        accounting.add(np.array([value]))
    result = accounting.finish()
    assert result["sum_gradient"] == [-4000]
    assert result["sum_thresholded_sign"] == [1]
    assert result["deadband_nonzero_gradient_windows"] == [2]
    assert result["opposite_nonzero_mean_gradient_and_sign_drift_coordinates"] == 1
    assert result["mean_gradient_sign_drift_cosine"] == -1


def test_virtual_bound_crossing_is_reported_without_altering_initial_weights():
    initial = np.array([-32768])
    accounting = diagnostic.GradientAccounting(initial, shifts=(0,))
    accounting.add(np.array([1]))
    result = accounting.finish()
    assert initial.tolist() == [-32768]
    assert result["fixed_gradient_error_feedback"]["0"]["virtual_int16_exceed_windows"] == 1


def test_empty_gradient_stream_is_not_evidence():
    with pytest.raises(ValueError):
        diagnostic.GradientAccounting([0]).finish()


def test_independent_all_parameter_formula_matches_frozen_observer(shape, construction):
    counts, weights = construction
    result = diagnostic.observer_check(weights, counts, [0, 0, 1, 2, 3, 0, 2, 1],
                                       [0, 1, 2, 3, 0, 2, 1, 0], shape)
    assert result["all_parameter_gradients_equal"] is True
    assert result["shared_projection_routes_equal"] is True
    assert result["native_execution"] is False
    assert result["parameter_gradient_entries_checked"] == sum(map(len, weights.values()))
    assert set(result["parameter_gradient_hashes"]) == set(weights)


def test_observer_check_requires_the_exact_constructed_checkpoint(shape, construction):
    counts, weights = construction
    weights["MP"][0] = 1
    with pytest.raises(ValueError, match="exact construction"):
        diagnostic.observer_check(weights, counts, [0] * 8, [0] * 8, shape)


def test_recipe_is_training_only_fixed_gradient_and_retains_all_candidates():
    recipe = diagnostic.recipe()
    assert recipe["training_indices"] == list(range(1024))
    assert recipe["validation_indices_used_for_optimizer_selection"] == []
    assert recipe["observer_check_windows"] == [0, 1, 2]
    assert recipe["update_shifts"] == [16, 18, 20]
    assert recipe["threshold"] == 2048
    assert recipe["gradient_training"] is recipe["native_execution"] is recipe["optimizer_selected"] is False
    assert diagnostic.source_identity().items() >= diagnostic.capacity.source_identity().items()


@pytest.mark.parametrize("destination", ["existing", "inside_capacity", "inside_other_run"])
def test_output_namespace_is_checked_before_any_diagnostic_execution(tmp_path, monkeypatch, destination):
    root = tmp_path / "capacity"
    root.mkdir()
    diagnostic.pilot.write_json(root / "manifest.json", {"schema": diagnostic.capacity.SCHEMA})
    other = tmp_path / "scales"
    other.mkdir()
    diagnostic.pilot.write_json(other / "manifest.json", {"schema": "tikitorch.fpga_lm_training_scales.v1"})
    if destination == "existing":
        output = tmp_path / "existing.json"
        output.write_text("immutable")
    else:
        output = (root if destination == "inside_capacity" else other) / "new.json"
    calls = []
    monkeypatch.setattr(diagnostic, "diagnose", lambda *args: calls.append(args))
    with pytest.raises((ValueError, FileExistsError)):
        diagnostic.main(["--run", str(root), "--record", str(tmp_path / "record"), "--output", str(output)])
    assert not calls


def test_input_or_source_mutation_prevents_publication(tmp_path, monkeypatch):
    record = tmp_path / "record.json"
    record.write_bytes(b"original")
    context = {"manifest": {}, "library": {}, "record_path": record, "record_bytes": b"original",
               "source_hashes": diagnostic.source_identity(), "root": tmp_path, "artifact_bytes": b"seal",
               "plan_bytes": diagnostic.capacity.PLAN.read_bytes()}
    monkeypatch.setattr(diagnostic.capacity, "_frozen", lambda *args: None)
    monkeypatch.setattr(diagnostic, "_seal", lambda root: (b"seal", {}))
    diagnostic._recheck(context)
    changed = copy.deepcopy(context)
    changed["source_hashes"] = {}
    with pytest.raises(ValueError):
        diagnostic._recheck(changed)
    record.write_bytes(b"changed")
    with pytest.raises(ValueError):
        diagnostic._recheck(context)
