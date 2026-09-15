"""Offline audit contracts for new training and same-cohort historical controls."""
import copy
import json

import pytest

from manuscript.tools import audit_fpga_lm_scales as audit


@pytest.fixture
def metadata():
    manifest = {"schema": audit.experiment.SCHEMA, "backend": "csim"}
    summary = {"schema": audit.experiment.SCHEMA, "complete": True, "failures": [],
               "declared_workers": 12, "completed_workers": 12,
               "heldout_test_evaluated": False, "hardware_execution": False,
               "speed_advantage_claim": False, "statistical_significance_claim": False}
    return manifest, summary


def test_all_twelve_training_and_control_cells_are_required(metadata):
    audit._verify_root_status(*metadata)


@pytest.mark.parametrize("name,value", [
    ("schema", "old"), ("complete", False), ("complete", 1),
    ("failures", [{"error": "retained"}]), ("failures", None),
    ("declared_workers", 6), ("declared_workers", 12.0), ("declared_workers", True),
    ("completed_workers", 11), ("completed_workers", 6), ("completed_workers", 12.0),
    ("heldout_test_evaluated", True), ("hardware_execution", True),
    ("speed_advantage_claim", True), ("statistical_significance_claim", True)])
def test_missing_controls_failures_and_expanded_claims_fail_closed(metadata, name, value):
    manifest, summary = metadata
    summary[name] = value
    with pytest.raises(ValueError, match="all twelve"):
        audit._verify_root_status(manifest, summary)


@pytest.mark.parametrize("name,value", [("schema", "old"), ("backend", "pci")])
def test_previous_studies_and_hardware_records_are_not_the_new_protocol(metadata, name, value):
    manifest, summary = metadata
    manifest[name] = value
    with pytest.raises(ValueError):
        audit._verify_root_status(manifest, summary)


@pytest.fixture
def controls():
    return {str(step): {"training_pairs": step * 128, "train_indices": list(range(step)),
                        "validation_indices": list(range(40, 56)), "alpha": 1, "vocabulary": 128}
            for step in (256, 1024)}


def test_initial_evaluation_has_no_fitted_baseline(controls):
    assert audit._controls_for_step(0, controls) is None


@pytest.mark.parametrize("step,pairs", [(256, 32768), (1024, 131072)])
def test_each_checkpoint_uses_its_own_supervised_budget(controls, step, pairs):
    result = audit._controls_for_step(step, controls)
    assert result is controls[str(step)]
    assert result["training_pairs"] == pairs


@pytest.mark.parametrize("step", [True, False, 256.0, "256", -1, 512, 2048])
def test_undeclared_checkpoint_is_rejected(controls, step):
    with pytest.raises(ValueError):
        audit._controls_for_step(step, controls)


@pytest.mark.parametrize("field,value", [
    ("training_pairs", 131072), ("training_pairs", 32768.0), ("training_pairs", True),
    ("train_indices", list(range(1024))), ("train_indices", list(range(1, 257))),
    ("validation_indices", list(range(24, 40))), ("validation_indices", list(range(8, 24))),
    ("validation_indices", [float(i) for i in range(40, 56)]),
    ("alpha", True), ("alpha", 1.0), ("vocabulary", 128.0)])
def test_controls_cannot_substitute_old_cohort_or_extra_training_data(controls, field, value):
    controls["256"][field] = value
    with pytest.raises(ValueError):
        audit._controls_for_step(256, controls)


@pytest.mark.parametrize("arm", ["anchors_shift1", "anchors_shift2", "anchors", "anchors_fine"])
def test_training_work_is_distinct_from_control_checkpoint_history(arm):
    training = arm.startswith("anchors_shift")
    worker = {"steps": 1024, "training_pairs": 131072, "training_performed": training,
              "training_updates_performed": 1024 if training else 0}
    audit._verify_training_scope(worker, arm)
    worker["training_updates_performed"] = 0 if training else 1024
    with pytest.raises(ValueError):
        audit._verify_training_scope(worker, arm)


@pytest.mark.parametrize("field,value", [
    ("training_performed", 1), ("training_performed", False),
    ("training_updates_performed", 1024.0), ("training_updates_performed", True),
    ("steps", 0), ("steps", 1024.0), ("training_pairs", 0), ("training_pairs", 131072.0)])
def test_nonliteral_or_incomplete_training_scope_is_rejected(field, value):
    worker = {"steps": 1024, "training_pairs": 131072, "training_performed": True,
              "training_updates_performed": 1024}
    worker[field] = value
    with pytest.raises(ValueError):
        audit._verify_training_scope(worker, "anchors_shift1")


def test_unknown_arm_cannot_be_treated_as_a_no_training_control():
    with pytest.raises(ValueError):
        audit._verify_training_scope({}, "unknown")


@pytest.fixture
def artifact_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "payload.i32").write_bytes(b"\1\0\0\0")
    audit.pilot.write_json(root / "artifact_manifest.json", audit.experiment._artifact_manifest(root))
    return root


def test_before_after_artifacts_retain_manifest_bytes(artifact_root):
    payload, declared = audit._verify_artifacts(artifact_root)
    assert len(declared["files"]) == 1
    assert audit._verify_artifacts(artifact_root, payload) == (payload, declared)


@pytest.mark.parametrize("change", ["payload", "extra", "symlink", "manifest_reformat", "schema"])
def test_changed_artifacts_or_schema_cannot_pass_final_recheck(artifact_root, change):
    payload, _ = audit._verify_artifacts(artifact_root)
    if change == "payload":
        (artifact_root / "payload.i32").write_bytes(b"\2\0\0\0")
    elif change == "extra":
        (artifact_root / "failure.json").write_text('{"complete":false}')
    elif change == "symlink":
        (artifact_root / "alias").symlink_to(artifact_root / "payload.i32")
    else:
        path = artifact_root / "artifact_manifest.json"
        record = json.loads(path.read_text())
        if change == "schema":
            record["schema"] = "previous-budget-schema"
        path.write_text(json.dumps(record, separators=(",", ":")))
    with pytest.raises(ValueError):
        audit._verify_artifacts(artifact_root, payload)


def test_budget_binding_uses_actual_run_audit_and_new_dataset(monkeypatch):
    calls = []
    monkeypatch.setattr(audit.experiment, "budget_binding", lambda *args: calls.append(args) or {"checked": True})
    manifest = {"budget_evidence": {"run_directory": "run", "audit_record": {"file": "audit"}},
                "dataset": {"sha256": "fixed"}}
    assert audit._budget_binding(manifest) == {"checked": True}
    assert calls == [("run", "audit", {"sha256": "fixed"})]


def test_all_frozen_dependencies_and_new_auditor_are_source_bound():
    identity = audit.source_identity()
    for name in ("manuscript/tools/audit_fpga_lm_scales.py", "manuscript/tools/train_fpga_lm_scales.py",
                 "manuscript/tools/diagnose_fpga_lm_calibration.py", "manuscript/tools/audit_fpga_lm_budget.py",
                 "manuscript/tools/extend_fpga_lm_budget.py", "manuscript/tools/audit_fpga_lm_head_screen.py",
                 "manuscript/tools/screen_fpga_lm_heads.py", "manuscript/tools/observe_fpga_lm_routes.py",
                 "scripts/fpga_lm_pretrain.py", "tikitorch/fpga/bridge.cpp"):
        assert identity[name] == audit.pilot.digest((audit.REPO / name).read_bytes())


def test_quality_thresholds_are_not_relaxed_for_intermediate_scales():
    assert audit.experiment.gate_arm is audit.screen.gate_arm
    assert audit.experiment.recipe()["gates"] == audit.screen.recipe()["gates"]


@pytest.fixture
def synthetic_evaluation(tmp_path, monkeypatch):
    """Injected helper fixture, not native execution or research evidence."""
    routes = {"positions": 128, "nonzero_logit_gradient_positions": 128,
              "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
              "head_winners_ge16_positions": 128, "single_shared_head_winner_positions": 0,
              "nonzero_embedding_input_vjp_positions": 128, "max_abs_head_input_vjp": 12,
              "max_abs_embedding_input_vjp": 12, "backward_int32_overflow_detected": False,
              "head_winner_counts": [128] * 128}
    observed = {"predictions": [0] * 128, "gradients": ([-127] + [1] * 127) * 128,
                "lut_normalized_nll": 3.0, "accuracy": 0.25,
                "saturation_events": 0, "rounded_zero_target_mass": 0,
                "route_diagnostics": routes,
                "native_routes": {"AD_HEAD": [0, 1], "PROJ_IDX": [0, 1, 2], "TAP_IDX": [0, 1, 2, 3]}}
    calls = []

    def window(raw, dataset, split, index):
        calls.append((split, index))
        return list(range(128)), list(range(128))

    monkeypatch.setattr(audit.pilot, "window", window)
    monkeypatch.setattr(audit, "observe", lambda *a, **k: copy.deepcopy(observed))
    monkeypatch.setattr(audit.pilot, "integer_forward", lambda *a, **k: copy.deepcopy(observed))
    arrays = {}
    for name in ("predictions", "gradients"):
        path = tmp_path / (name + ".i32")
        path.write_bytes(audit.screen._i32(observed[name]))
        arrays[name] = {"file": path.name}
    digests = {name: {"words": len(values), "sha256": audit.pilot.digest(audit.screen._i32(values))}
               for name, values in observed["native_routes"].items()}
    probes = {name: audit.screen._input_probe(observed, observed)
              for name in ("context_disabled", "all_space", "reversed_tokens")}
    rows = [{"window_index": index, **copy.deepcopy(audit.screen._scalar_result(observed)),
             "native_arrays": copy.deepcopy(arrays), "native_route_hashes": copy.deepcopy(digests),
             "probes": copy.deepcopy(probes), "baseline_minus_model_nll": {},
             "native_reference_equal": True, "native_routes_equal": True}
            for index in range(40, 56)]
    evaluation = audit.screen._aggregate(rows)
    return (tmp_path, evaluation, {}, b"unused", {}, None, 1, None), calls


def test_recomputation_checks_all_new_cohort_outputs_and_counts(synthetic_evaluation):
    args, calls = synthetic_evaluation
    assert audit._recompute_evaluation(*args) == {
        "predictions": 2048, "gradient_entries": 262144, "route_digest_entries": 144}
    assert calls == [("validation", index) for index in range(40, 56)]


@pytest.mark.parametrize("change", ["cohort", "float_cohort", "gradient", "prediction", "route", "probe", "baseline"])
def test_recomputation_rejects_inconsistent_native_or_counterfactual_records(synthetic_evaluation, change):
    args, _ = synthetic_evaluation
    root, evaluation = args[:2]
    row = evaluation["per_window"][0]
    if change == "cohort":
        row["window_index"] = 24
    elif change == "float_cohort":
        row["window_index"] = 40.0
    elif change in ("gradient", "prediction"):
        name = "gradients" if change == "gradient" else "predictions"
        (root / (name + ".i32")).write_bytes(audit.screen._i32([1]))
    elif change == "route":
        row["native_route_hashes"]["AD_HEAD"]["sha256"] = "changed"
    elif change == "probe":
        row["probes"]["context_disabled"]["lut_normalized_nll"] += 1
    else:
        row["baseline_minus_model_nll"] = {"bigram": 0.2}
    with pytest.raises(ValueError):
        audit._recompute_evaluation(*args)


@pytest.mark.parametrize("destination", ["existing", "run", "budget", "v1"])
def test_audit_output_cannot_mutate_any_immutable_reference(tmp_path, monkeypatch, destination):
    roots = {name: tmp_path / name for name in ("run", "budget", "v1")}
    for path in roots.values():
        path.mkdir()
    (roots["run"] / "manifest.json").write_text(json.dumps({"budget_evidence": {"run_directory": str(roots["budget"])}}))
    (roots["budget"] / "manifest.json").write_text(json.dumps({"v1_evidence": {"run_directory": str(roots["v1"])}}))
    existing = tmp_path / "audit.json"
    existing.write_bytes(b"preserve")
    output = existing if destination == "existing" else roots[destination] / "new.json"
    calls = []
    monkeypatch.setattr(audit, "audit", lambda *a: calls.append(a))
    with pytest.raises(SystemExit) as raised:
        audit.main(["--run", str(roots["run"]), "--output", str(output)])
    assert raised.value.code == 2
    assert calls == []
    assert existing.read_bytes() == b"preserve"
