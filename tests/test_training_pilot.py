"""The pilot table is complete, source-bound, reproducible, and narrowly scoped."""
from copy import deepcopy
import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from manuscript.tools import summarize_training_pilot as pilot
from manuscript.tools import historical_cpu_evidence as historical


@pytest.fixture
def record():
    return pilot.read_record()


@pytest.fixture
def historical_root():
    with historical.verified_workspace() as root:
        yield root


def test_pinned_manifest_and_generated_table_match_historical_sources(historical_root):
    assert hashlib.sha256(pilot.DEFAULT_RECORD.read_bytes()).hexdigest() == pilot.SOURCE_SHA256
    assert hashlib.sha256(pilot.FAILURE_RECORD.read_bytes()).hexdigest() == pilot.FAILURE_SHA256
    assert pilot.generate(root=historical_root) == pilot.GENERATED_TABLE.read_text()
    result = subprocess.run([sys.executable, "-S", str(pilot.REPO / "manuscript/tools/historical_cpu_evidence.py"),
                             "--check"], cwd=pilot.REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_changed_raw_manifest_is_not_silently_reduced(tmp_path, record):
    raw = tmp_path / "manifest.json"
    raw.write_text(json.dumps(record))  # Even semantically identical reformatting changes raw identity.
    with pytest.raises(ValueError, match="digest mismatch"):
        pilot.read_record(raw)


def test_negative_followup_is_pinned_source_checked_and_excluded_from_table(tmp_path, historical_root):
    failure = pilot.read_record(pilot.FAILURE_RECORD, pilot.FAILURE_SHA256)
    pilot.check_sources(failure, historical_root)
    pilot.check_failure(failure)
    assert "5.600" not in pilot.generate(root=historical_root)  # No crashed worker duration as step latency.
    changed = tmp_path / "failure.json"
    changed.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="digest mismatch"):
        pilot.generate(root=historical_root, failure_path=changed)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(incident_status="completed"), "inconclusive"),
    (lambda d: d["metrics"]["worker_outcome"].update(returncode=0), "negative worker"),
    (lambda d: d["metrics"]["row"].update(status="ok"), "successful"),
    (lambda d: d["metrics"]["row"].update(median_ms=1), "successful"),
    (lambda d: d["config"]["config"].update(layers=1), "configuration"),
])
def test_failure_cannot_be_reinterpreted_as_success(mutation, match):
    failure = pilot.read_record(pilot.FAILURE_RECORD, pilot.FAILURE_SHA256)
    mutation(failure)
    with pytest.raises(ValueError, match=match):
        pilot.check_failure(failure)


def test_reducer_preserves_all_cells_and_derives_metrics(record):
    rows = pilot.selected_rows(record)
    assert [(row["arm"], row["checkpoint_policy"]) for row in rows] == list(pilot.CELLS)
    text = pilot.render(rows)
    assert len(rows) == 10
    assert "Packed routes & -- & 0.375 & 0.006 & 4603.1" in text
    assert "Compiled checkpoint & selective & 0.349 & 0.011 & 6929.8" in text
    assert "tiny single-seed cpu pilot" in text.lower()
    assert "\\textbf" not in text and "$\\times$" not in text
    record["metrics"]["summaries"][0]["packed_speedup_vs_best_baseline"] = 12345
    record["metrics"]["summaries"][0]["best_baseline_arm"] = "packed"
    assert pilot.render(pilot.selected_rows(record)) == text
    record["metrics"]["rows"].reverse()
    assert pilot.render(pilot.selected_rows(record)) == text


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(incident_status="inconclusive"), "incomplete"),
    (lambda d: d["config"]["cells"][0].update(adamw_eps=1e-8), "configuration"),
    (lambda d: d["metrics"]["rows"].pop(), "incomplete pilot cells"),
    (lambda d: d["metrics"]["rows"].__setitem__(1, deepcopy(d["metrics"]["rows"][0])), "duplicate"),
    (lambda d: d["metrics"]["checks"].update(sources_unchanged=False), "sources changed"),
    (lambda d: d["metrics"]["checks"]["learning"]["optimizer"].update(eps=1e-8), "optimizer"),
    (lambda d: d["metrics"]["checks"]["learning"]["models"]["route_morphssm"]["after"].update(accuracy=0.5), "learning gate"),
    (lambda d: d["metrics"]["summaries"][0].update(missing_arms=["packed"]), "summary"),
])
def test_incomplete_or_changed_protocol_is_rejected(record, mutation, match):
    mutation(record)
    with pytest.raises(ValueError, match=match):
        pilot.selected_rows(record)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(median_ms=float("nan")), "median"),
    (lambda r: r.update(median_ms=123), "timing summary"),
    (lambda r: r.update(mad_ms=123), "timing summary"),
    (lambda r: r["samples_ms"].pop(), "sample count"),
    (lambda r: r["samples_ms"].__setitem__(0, float("inf")), "timing sample"),
    (lambda r: r.update(startup_and_first_update_ms=-1), "startup"),
    (lambda r: r.update(update_count=7), "update count"),
    (lambda r: r.update(batch_sha256="a" * 64), "unmatched pilot"),
    (lambda r: r.update(status="parity_failed"), "failed pilot cell"),
    (lambda r: r["worker_outcome"].update(returncode=1), "worker"),
    (lambda r: r["worker_outcome"].update(stop_reason="CELL_TIMEOUT"), "worker"),
    (lambda r: r["parity"].update(compared_measured_state=False), "measured-state"),
    (lambda r: r["parity"].update(atol=0.05), "measured-state"),
    (lambda r: r["parity"]["parameters"].pop(), "parameter parity"),
    (lambda r: r["parity"]["parameters"][0].update(update_max_abs=1), "parity metric"),
    (lambda r: r["model_context"].update(causal_receptive_field=12), "causal context"),
    (lambda r: r.update(cpu_tensor_peak_bytes=100), "tensor-peak"),
    (lambda r: r.update(compile_fullgraph=True), "scope"),
])
def test_bad_cell_metrics_cannot_become_a_table(record, mutation, match):
    mutation(record["metrics"]["rows"][0])
    with pytest.raises(ValueError, match=match):
        pilot.selected_rows(record)


def test_sources_are_relocatable_without_reading_original_paths(tmp_path, record, historical_root):
    old_root = str(pilot.REPO)
    for item in record["inputs"]:
        relative = item["path"].removeprefix(old_root + "/")
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(historical_root / relative, target)
        item["path"] = "/unavailable/original-checkout/" + relative
    pilot.check_sources(record, tmp_path)
    changed = tmp_path / pilot.HELPERS[0]
    changed.write_bytes(changed.read_bytes() + b"\n# source drift\n")
    with pytest.raises(ValueError, match="stale current source"):
        pilot.check_sources(record, tmp_path)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["inputs"].pop(), "incomplete source"),
    (lambda d: d["inputs"].append(deepcopy(d["inputs"][-1])), "duplicate source"),
    (lambda d: d["inputs"][1].update(path="/outside/actual_model_benchmark.py"), "outside original root"),
    (lambda d: d["inputs"][1].update(path="/old/root/../actual_model_benchmark.py"), "unsafe"),
    (lambda d: d["inputs"][1].update(sha256="a" * 64), "stale current source"),
])
def test_source_omission_duplication_escape_and_drift_fail(record, mutation, match, historical_root):
    mutation(record)
    with pytest.raises(ValueError, match=match):
        pilot.check_sources(record, historical_root)


def test_check_mode_does_not_rewrite_stale_table(tmp_path, monkeypatch):
    output = tmp_path / "training_pilot.tex"
    output.write_text("stale\n")
    monkeypatch.setattr(pilot, "GENERATED_TABLE", output)
    with pytest.raises(SystemExit) as error:
        pilot.main(["--check"])
    assert error.value.code == 1
    assert output.read_text() == "stale\n"
