"""Cheap evidence-auditor contracts; no training, native loading or cloud access."""
import copy
import json
import math
import struct
from types import SimpleNamespace

import pytest

from manuscript.tools import audit_fpga_lm_head_screen as audit


@pytest.mark.parametrize("actual,expected", [
    ({"nested": [1, True, None, "x", 1.25]}, {"nested": [1, True, None, "x", 1.25]}),
    ({"nll": 1.0 + 5e-13}, {"nll": 1.0}),
    ({"nll": 1}, {"nll": 1.0}),
])
def test_recursive_comparison_accepts_only_declared_log_rounding(actual, expected):
    audit._assert_equal(actual, expected)


@pytest.mark.parametrize("actual,expected", [
    ({"v": True}, {"v": 1}), ({"v": 1}, {"v": True}),
    ({"v": 1.0}, {"v": 1}), ({"v": True}, {"v": 1.0}),
    ({"v": [1]}, {"v": [1, 2]}), ({"v": (1, 2)}, {"v": [1, 2]}),
    ({"v": None}, {"v": "None"}), ({"v": 1, "extra": 2}, {"v": 1}),
    ({"v": 1.0 + 2e-12}, {"v": 1.0}),
    ({"v": float("nan")}, {"v": 1.0}),
    ({"v": float("inf")}, {"v": 1.0}),
])
def test_recursive_comparison_rejects_type_structure_and_numeric_changes(actual, expected):
    with pytest.raises(ValueError, match="record"):
        audit._assert_equal(actual, expected)


@pytest.mark.parametrize("spelling", ["NaN", "Infinity", "-Infinity"])
def test_json_reader_rejects_nested_nonfinite_numbers(tmp_path, spelling):
    path = tmp_path / "record.json"
    path.write_text('{"rows":[{"loss":' + spelling + '}]}')
    with pytest.raises(ValueError, match="nonfinite"):
        audit._read(path)


def test_payload_unpacking_is_explicit_signed_little_endian(tmp_path):
    path = tmp_path / "weights.i32"
    path.write_bytes(bytes.fromhex("0000008001000000ffffffffffffff7f"))
    assert audit._unpack(path) == [-(1 << 31), 1, -1, (1 << 31) - 1]


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_partial_word_payload_cannot_be_reinterpreted(tmp_path, size):
    path = tmp_path / "partial.i32"
    path.write_bytes(b"\0" * size)
    with pytest.raises(ValueError, match="incomplete int32"):
        audit._unpack(path)


@pytest.mark.parametrize("words", [0, 9, 11])
def test_checkpoint_regions_require_exact_complete_payload(tmp_path, words):
    path = tmp_path / "weights.i32"
    path.write_bytes(struct.pack("<%di" % words, *range(words)))
    shape = SimpleNamespace(**{"WORDS_" + name: 2 for name in audit.pilot.WEIGHTS})
    with pytest.raises(ValueError, match="number of weights"):
        audit._weights(tmp_path, {"file": path.name}, shape)


def test_checkpoint_region_order_matches_frozen_weight_serialization(tmp_path):
    path = tmp_path / "weights.i32"
    path.write_bytes(struct.pack("<10i", *range(10)))
    shape = SimpleNamespace(**{"WORDS_" + name: 2 for name in audit.pilot.WEIGHTS})
    assert audit._weights(tmp_path, {"file": path.name}, shape) == {
        name: [2 * i, 2 * i + 1] for i, name in enumerate(audit.pilot.WEIGHTS)}


def test_compaction_sums_counts_but_maximizes_extrema_without_losing_windows():
    rows = []
    for index, scale in ((8, 2), (9, 3)):
        rows.append({"window_index": index, "lut_normalized_nll": float(scale),
                     "accuracy": 0.5, "baseline_minus_model_nll": {"unigram": -1.0},
                     "probes": {"context_disabled": {"lut_normalized_nll": scale + 0.1}},
                     "route_diagnostics": {"projection_layers": [
                         {"layer": layer, "positions": 128,
                          "nonzero_incoming_vjp_positions": scale,
                          "projection_incoming_l1": 10 * scale,
                          "max_abs_layer_input_vjp": 100 * scale}
                         for layer in range(4)]}})
    evaluation = {"step": 256, "checkpoint": {"sha256": "example"},
                  "lut_normalized_nll": 2.5, "accuracy": 0.5,
                  "context_disabled_nll": 2.6, "saturation_events": 0,
                  "rounded_zero_target_mass": 0, "native_reference_equal": True,
                  "native_routes_equal": True,
                  "route_diagnostics": {"positions": 256, "head_winner_counts": [1, 16, 128]},
                  "per_window": rows}
    original = copy.deepcopy(evaluation)
    result = audit._compact_evaluation(evaluation)
    assert evaluation == original
    diagnostic = result["route_diagnostics"]
    assert "head_winner_counts" not in diagnostic
    assert diagnostic["head_winners_min"] == 1
    assert diagnostic["head_winners_max"] == 128
    assert diagnostic["positions"] == 256
    assert diagnostic["projection_layers"] == [
        {"layer": layer, "positions": 256, "nonzero_incoming_vjp_positions": 5,
         "projection_incoming_l1": 50, "max_abs_layer_input_vjp": 300}
        for layer in range(4)]
    assert [row["window_index"] for row in result["per_window"]] == [8, 9]
    assert result["per_window"][0]["input_sensitivity_probes"] == rows[0]["probes"]


@pytest.fixture
def early_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    audit.pilot.write_json(root / "manifest.json", {"schema": audit.screen.SCHEMA, "backend": "csim"})
    audit.pilot.write_json(root / "summary.json", {
        "schema": audit.screen.SCHEMA, "complete": True, "failures": [],
        "declared_workers": 12, "completed_workers": 12,
        "hardware_execution": False, "heldout_test_evaluated": False,
        "speed_advantage_claim": False, "statistical_significance_claim": False})
    return root


@pytest.mark.parametrize("key,value", [
    ("schema", "not-the-screen-schema"), ("complete", False), ("complete", 1),
    ("failures", [{"arm": "reference", "seed": 0}]),
    ("completed_workers", 11), ("completed_workers", 12.0), ("completed_workers", True),
    ("declared_workers", 13), ("declared_workers", 12.0), ("hardware_execution", True),
    ("heldout_test_evaluated", True), ("speed_advantage_claim", True),
])
def test_incomplete_or_wrong_scope_summary_is_rejected_before_payload_work(early_run, key, value):
    path = early_run / "summary.json"
    summary = json.loads(path.read_text())
    summary[key] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        audit.audit(early_run)


def test_wrong_manifest_schema_is_rejected_before_payload_work(early_run):
    (early_run / "manifest.json").write_text('{"schema":"wrong"}')
    with pytest.raises(ValueError):
        audit.audit(early_run)


def test_valid_root_metadata_reaches_explicit_status_gate(early_run):
    # This positive case keeps the malformed-metadata cases from passing merely
    # because the fixture is already invalid in an unrelated field.
    audit._verify_root_status(audit._read(early_run / "manifest.json"),
                              audit._read(early_run / "summary.json"))


@pytest.mark.parametrize("name", ["failure.json", "cleanup_failure.json"])
def test_root_failure_marker_cannot_be_hidden_by_complete_summary(early_run, name):
    (early_run / name).write_text('{"error":"retained failure"}')
    with pytest.raises(ValueError, match="failed screen"):
        audit.audit(early_run)


@pytest.fixture
def library_fixture(tmp_path, monkeypatch):
    root = tmp_path / "run"
    build_directory = root / "build"
    build_directory.mkdir(parents=True)
    binary = build_directory / "libmorphlm.so"
    binary.write_bytes(b"not executable: fixture bytes are hashed but never loaded")
    library = {"path": str(binary), "sha256": audit.pilot.digest(binary.read_bytes())}
    sources = {"frozen/example.cpp": "a" * 64}
    manifest = {"layout": [1, 512, 128]}
    build = {"library_sha256": library["sha256"], "source_hashes": sources,
             "layout": manifest["layout"]}
    audit.pilot.write_json(build_directory / "build.json", build)
    monkeypatch.setattr(audit.pilot, "source_identity", lambda: sources)
    return root, binary, library, manifest, build


def test_actual_library_payload_and_build_identity_are_checked_without_loading(library_fixture):
    root, _, library, manifest, build = library_fixture
    assert audit._verify_library(root, library, manifest) == build


def test_changed_library_payload_rejected(library_fixture):
    root, binary, library, manifest, _ = library_fixture
    binary.write_bytes(binary.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        audit._verify_library(root, library, manifest)


@pytest.mark.parametrize("key,value", [
    ("library_sha256", "b" * 64), ("source_hashes", {"other.cpp": "c" * 64}),
    ("layout", [1, 16, 128]),
])
def test_stale_build_metadata_rejected(library_fixture, key, value):
    root, _, library, manifest, build = library_fixture
    build[key] = value
    (root / "build/build.json").write_text(json.dumps(build))
    with pytest.raises(ValueError, match="identity differs"):
        audit._verify_library(root, library, manifest)


def test_library_declaration_cannot_use_arbitrary_binary_name(library_fixture):
    root, _, library, manifest, _ = library_fixture
    library["path"] = "/outside/surprise.so"
    with pytest.raises(ValueError, match="unsupported built library"):
        audit._verify_library(root, library, manifest)


def test_library_symlink_is_not_accepted_even_with_matching_bytes(library_fixture):
    root, binary, library, manifest, _ = library_fixture
    target = root / "retained-real-binary"
    binary.rename(target)
    binary.symlink_to(target)
    with pytest.raises(ValueError, match="identity differs"):
        audit._verify_library(root, library, manifest)


def test_existing_output_is_not_overwritten_or_audited(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = tmp_path / "audit.json"
    output.write_bytes(b"existing evidence\n")
    monkeypatch.setattr(audit, "audit", lambda path: pytest.fail("must not begin recomputation"))
    with pytest.raises(SystemExit) as error:
        audit.main(["--run", str(run), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"existing evidence\n"


@pytest.mark.parametrize("symlink", [False, True])
def test_output_cannot_modify_immutable_run_including_via_symlink(tmp_path, monkeypatch, symlink):
    run = tmp_path / "run"
    run.mkdir()
    if symlink:
        alias = tmp_path / "alias"
        alias.symlink_to(run, target_is_directory=True)
    else:
        alias = run
    output = alias / "audit.json"
    monkeypatch.setattr(audit, "audit", lambda path: pytest.fail("must not begin recomputation"))
    with pytest.raises(SystemExit) as error:
        audit.main(["--run", str(run), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()
    assert not list(run.iterdir())


def test_failed_audit_never_publishes_partial_evidence(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"

    def failure(path):
        raise ValueError("deliberate mismatch")

    monkeypatch.setattr(audit, "audit", failure)
    with pytest.raises(ValueError, match="deliberate mismatch"):
        audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert not output.exists()


def test_output_creation_is_exclusive_even_after_precheck(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"

    def competing_writer(path):
        output.write_bytes(b"another completed audit\n")
        return {"complete": True}

    monkeypatch.setattr(audit, "audit", competing_writer)
    with pytest.raises(FileExistsError):
        audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert output.read_bytes() == b"another completed audit\n"


def test_successful_cli_writes_finite_record_once(tmp_path, monkeypatch, capsys):
    output = tmp_path / "audit.json"
    result = {"schema": "example-test-only", "complete": True, "nll": math.log(128)}
    monkeypatch.setattr(audit, "audit", lambda path: result)
    audit.main(["--run", str(tmp_path / "run"), "--output", str(output)])
    assert json.loads(output.read_text()) == result
    assert json.loads(capsys.readouterr().out) == {"complete": True, "output": str(output)}
