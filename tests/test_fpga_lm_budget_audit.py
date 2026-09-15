"""Offline evidence guards for the declared matched-budget CPU extension."""
import copy
import json

import pytest

from manuscript.tools import audit_fpga_lm_budget as audit


@pytest.fixture
def metadata():
    manifest = {"schema": audit.experiment.SCHEMA, "backend": "csim"}
    summary = {"schema": audit.experiment.SCHEMA, "complete": True, "failures": [],
               "declared_workers": 6, "completed_workers": 6,
               "heldout_test_evaluated": False, "hardware_execution": False,
               "speed_advantage_claim": False, "statistical_significance_claim": False}
    return manifest, summary


def test_complete_six_cell_cpu_metadata_is_accepted(metadata):
    audit._verify_root_status(*metadata)


@pytest.mark.parametrize("key,value", [
    ("schema", "wrong"), ("complete", False), ("complete", 1),
    ("failures", [{"error": "retained failure"}]), ("failures", None),
    ("declared_workers", 12), ("declared_workers", 6.0),
    ("completed_workers", 5), ("completed_workers", 6.0), ("completed_workers", True),
    ("heldout_test_evaluated", True), ("hardware_execution", True),
    ("speed_advantage_claim", True), ("statistical_significance_claim", True),
])
def test_incomplete_or_expanded_claims_cannot_pass_root_gate(metadata, key, value):
    manifest, summary = metadata
    summary[key] = value
    with pytest.raises(ValueError, match="six declared CPU"):
        audit._verify_root_status(manifest, summary)


@pytest.mark.parametrize("key,value", [("schema", "old-v1-schema"), ("backend", "pci")])
def test_extension_requires_its_own_protocol_and_explicit_csim_backend(metadata, key, value):
    manifest, summary = metadata
    manifest[key] = value
    with pytest.raises(ValueError, match="six declared CPU"):
        audit._verify_root_status(manifest, summary)


@pytest.fixture
def controls():
    return {str(step): {"training_pairs": step * 128, "train_indices": list(range(step)),
                        "validation_indices": list(range(24, 40)), "alpha": 1, "vocabulary": 128}
            for step in (256, 1024)}


def test_initial_control_has_no_fitted_baseline_comparison(controls):
    assert audit._controls_for_step(0, controls) is None


@pytest.mark.parametrize("step,pairs", [(256, 32768), (1024, 131072)])
def test_control_fit_budget_is_selected_at_each_trained_checkpoint(controls, step, pairs):
    selected = audit._controls_for_step(step, controls)
    assert selected is controls[str(step)]
    assert selected["training_pairs"] == pairs
    assert selected["train_indices"] == list(range(step))
    assert selected["validation_indices"] == list(range(24, 40))


@pytest.mark.parametrize("step", [True, False, 256.0, "256", 1, 512, 2048, -1])
def test_undeclared_or_noninteger_checkpoint_is_rejected(controls, step):
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        audit._controls_for_step(step, controls)


@pytest.mark.parametrize("key,value", [
    ("training_pairs", 131072), ("training_pairs", 32768.0), ("training_pairs", True),
    ("train_indices", list(range(1024))), ("train_indices", list(range(1, 257))),
    ("train_indices", [float(i) for i in range(256)]),
    ("train_indices", [False] + list(range(1, 256))),
    ("validation_indices", list(range(8, 24))), ("validation_indices", list(range(24, 39))),
    ("validation_indices", [float(i) for i in range(24, 40)]),
    ("alpha", 2), ("alpha", True), ("alpha", 1.0),
    ("vocabulary", 127), ("vocabulary", 128.0),
])
def test_unmatched_budget_cohort_or_metadata_type_is_rejected(controls, key, value):
    controls["256"][key] = value
    with pytest.raises(ValueError):
        audit._controls_for_step(256, controls)


def test_final_controls_cannot_reuse_the_256_window_fit(controls):
    controls["1024"] = controls["256"]
    with pytest.raises(ValueError):
        audit._controls_for_step(1024, controls)


def test_executable_identity_includes_every_reused_implementation_dependency():
    identity = audit.source_identity()
    for relative in (
            "manuscript/tools/audit_fpga_lm_budget.py",
            "manuscript/tools/extend_fpga_lm_budget.py",
            "manuscript/tools/audit_fpga_lm_head_screen.py",
            "manuscript/tools/screen_fpga_lm_heads.py",
            "manuscript/tools/observe_fpga_lm_routes.py",
            "scripts/fpga_lm_pretrain.py", "scripts/fpga_lm_layout.cpp",
            "tikitorch/fpga/bridge.cpp", "tikitorch/fpga/silicon/gen_buffer_init.py",
            "tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp",
            "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp",
            "tikitorch/hardware/hls/forge_kernels/forge_ce_lut.hpp",
            "tikitorch/hardware/hls/forge_hls_types.hpp"):
        assert identity[relative] == audit.pilot.digest((audit.REPO / relative).read_bytes())


def test_declared_evaluations_and_final_quality_thresholds_are_not_relaxed():
    assert audit.EVALUATIONS == ((0, "initial"), (256, "intermediate"), (1024, "final"))
    assert audit.experiment.gate_arm is audit.screen.gate_arm
    assert audit.experiment.recipe()["gates"] == audit.screen.recipe()["gates"]
    assert audit.experiment.recipe()["gate_float_boundary_tolerance_nats"] == 1e-12


@pytest.fixture
def artifact_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "payload.i32").write_bytes(b"\1\0\0\0")
    audit.pilot.write_json(root / "artifact_manifest.json", audit.experiment._artifact_manifest(root))
    return root


def test_artifact_recheck_retains_exact_original_manifest_bytes(artifact_root):
    original, declared = audit._verify_artifacts(artifact_root)
    assert original == (artifact_root / "artifact_manifest.json").read_bytes()
    assert len(declared["files"]) == 1
    assert audit._verify_artifacts(artifact_root, original) == (original, declared)


def test_extension_artifacts_cannot_be_mislabeled_as_the_prior_screen(artifact_root):
    path = artifact_root / "artifact_manifest.json"
    record = json.loads(path.read_text())
    record["schema"] = audit.screen.SCHEMA
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="extension artifacts.schema"):
        audit._verify_artifacts(artifact_root)


@pytest.mark.parametrize("mutation", ["payload", "extra_file", "symlink", "manifest_reformat"])
def test_late_artifact_change_cannot_be_blessed_by_final_recheck(artifact_root, mutation):
    original, _ = audit._verify_artifacts(artifact_root)
    if mutation == "payload":
        (artifact_root / "payload.i32").write_bytes(b"\2\0\0\0")
    elif mutation == "extra_file":
        (artifact_root / "late_failure.json").write_text('{"complete": false}')
    elif mutation == "symlink":
        (artifact_root / "alias").symlink_to(artifact_root / "payload.i32")
    else:
        path = artifact_root / "artifact_manifest.json"
        path.write_text(json.dumps(json.loads(path.read_text()), separators=(",", ":")))
    with pytest.raises(ValueError):
        audit._verify_artifacts(artifact_root, original)


def test_library_validation_reuses_frozen_binary_source_layout_gate(monkeypatch, tmp_path):
    calls = []

    def checked(root, library, manifest):
        calls.append((root, library, manifest))
        return {"verified": True}

    monkeypatch.setattr(audit.prior, "_verify_library", checked)
    assert audit._verify_library(tmp_path, {"sha256": "example"}, {"layout": [512]}) == {"verified": True}
    assert calls == [(tmp_path, {"sha256": "example"}, {"layout": [512]})]


@pytest.fixture
def fake_evaluation(tmp_path, monkeypatch):
    """Synthetic helper fixture, not native or scientific evidence."""
    diagnostic = {
        "positions": 128, "nonzero_logit_gradient_positions": 128,
        "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 128,
        "head_winners_ge16_positions": 0, "single_shared_head_winner_positions": 128,
        "nonzero_embedding_input_vjp_positions": 0, "max_abs_head_input_vjp": 0,
        "max_abs_embedding_input_vjp": 0, "backward_int32_overflow_detected": False,
        "head_winner_counts": [1] * 128}
    observed = {"predictions": [0] * 128, "gradients": ([-127] + [1] * 127) * 128,
                "lut_normalized_nll": 3.0, "accuracy": 0.25,
                "saturation_events": 0, "rounded_zero_target_mass": 0,
                "route_diagnostics": diagnostic,
                "native_routes": {"AD_HEAD": [0, 1], "PROJ_IDX": [0, 1, 2],
                                  "TAP_IDX": [0, 1, 2, 3]}}
    calls = []

    def window(raw, dataset, split, index):
        calls.append((split, index))
        return list(range(128)), list(range(128))

    monkeypatch.setattr(audit.pilot, "window", window)
    monkeypatch.setattr(audit, "observe", lambda *args, **kwargs: copy.deepcopy(observed))
    monkeypatch.setattr(audit.pilot, "integer_forward", lambda *args, **kwargs: copy.deepcopy(observed))
    arrays = {}
    for name in ("predictions", "gradients"):
        path = tmp_path / (name + ".i32")
        path.write_bytes(audit.screen._i32(observed[name]))
        arrays[name] = {"file": path.name}
    routes = {name: {"words": len(values), "sha256": audit.pilot.digest(audit.screen._i32(values))}
              for name, values in observed["native_routes"].items()}
    probes = {name: audit.screen._input_probe(observed, observed)
              for name in ("context_disabled", "all_space", "reversed_tokens")}
    rows = [{"window_index": index, **copy.deepcopy(audit.screen._scalar_result(observed)),
             "native_arrays": copy.deepcopy(arrays), "native_route_hashes": copy.deepcopy(routes),
             "probes": copy.deepcopy(probes), "baseline_minus_model_nll": {},
             "native_reference_equal": True, "native_routes_equal": True}
            for index in range(24, 40)]
    evaluation = audit.screen._aggregate(rows)
    args = (tmp_path, evaluation, {}, b"unused", {}, None, 4, None)
    return args, calls


def test_recomputation_checks_all_16_new_windows_and_counts_actual_arrays(fake_evaluation):
    args, calls = fake_evaluation
    assert audit._recompute_evaluation(*args) == {
        "predictions": 2048, "gradient_entries": 262144, "route_digest_entries": 144}
    assert calls == [("validation", index) for index in range(24, 40)]


@pytest.mark.parametrize("indices", [list(range(8, 24)), list(range(24, 39)),
                                   list(range(24, 39)) + [38], list(range(39, 23, -1))])
def test_recomputation_rejects_old_incomplete_duplicated_or_reordered_cohort(fake_evaluation, indices):
    args, calls = fake_evaluation
    rows = args[1]["per_window"]
    args[1]["per_window"] = [dict(rows[0], window_index=index) for index in indices]
    with pytest.raises(ValueError, match="validation cohort"):
        audit._recompute_evaluation(*args)
    assert calls == []


@pytest.mark.parametrize("region", ["predictions", "gradients"])
def test_recomputation_rejects_retained_native_array_mismatch(fake_evaluation, region):
    args, _ = fake_evaluation
    path = args[0] / (region + ".i32")
    payload = path.read_bytes()
    path.write_bytes(b"\1\0\0\0" + payload[4:])
    with pytest.raises(ValueError, match="saved native validation output"):
        audit._recompute_evaluation(*args)


def test_recomputation_rejects_route_digest_mismatch(fake_evaluation):
    args, _ = fake_evaluation
    args[1]["per_window"][0]["native_route_hashes"]["AD_HEAD"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native route digest"):
        audit._recompute_evaluation(*args)


def test_initial_evaluation_cannot_claim_trained_baseline_advantage(fake_evaluation):
    args, _ = fake_evaluation
    args[1]["per_window"][0]["baseline_minus_model_nll"] = {"unigram": 0.5}
    with pytest.raises(ValueError, match="checkpoint-specific baseline"):
        audit._recompute_evaluation(*args)


def test_checkpoint_baseline_delta_is_recomputed_from_that_fit(fake_evaluation):
    args, _ = fake_evaluation
    fitted = {"unigram": {"per_window": [{"nll": 4.0}] * 16},
              "bigram": {"per_window": [{"nll": 2.0}] * 16}}
    for row in args[1]["per_window"]:
        row["baseline_minus_model_nll"] = {"unigram": 1.0, "bigram": -1.0}
    assert audit._recompute_evaluation(*args[:-1], fitted)["predictions"] == 2048
    args[1]["per_window"][-1]["baseline_minus_model_nll"]["bigram"] = 1.0
    with pytest.raises(ValueError, match="checkpoint-specific baseline"):
        audit._recompute_evaluation(*args[:-1], fitted)


def test_existing_audit_output_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    output.write_bytes(b"retained evidence")
    monkeypatch.setattr(audit, "audit", lambda path: pytest.fail("must not start recomputation"))
    with pytest.raises(SystemExit) as error:
        audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"retained evidence"


def test_output_cannot_be_created_in_immutable_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(run, target_is_directory=True)
    monkeypatch.setattr(audit, "audit", lambda path: pytest.fail("must not start recomputation"))
    with pytest.raises(SystemExit):
        audit.main(["--run", str(run), "--output", str(alias / "audit.json")])
    assert not list(run.iterdir())


def test_failure_never_publishes_partial_audit(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"

    def failed(path):
        raise ValueError("mismatch")

    monkeypatch.setattr(audit, "audit", failed)
    with pytest.raises(ValueError, match="mismatch"):
        audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert not output.exists()


def test_output_creation_is_exclusive_after_recomputation(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"

    def competitor(path):
        output.write_bytes(b"other audit won the race")
        return {"complete": True}

    monkeypatch.setattr(audit, "audit", competitor)
    with pytest.raises(FileExistsError):
        audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert output.read_bytes() == b"other audit won the race"


def test_successful_audit_is_written_once(tmp_path, monkeypatch, capsys):
    output = tmp_path / "audit.json"
    record = {"complete": True, "schema": "test-only-not-scientific-evidence"}
    monkeypatch.setattr(audit, "audit", lambda path: record)
    audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert json.loads(output.read_text()) == record
    assert json.loads(capsys.readouterr().out) == {"complete": True, "output": str(output)}
