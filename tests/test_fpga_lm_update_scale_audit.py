"""Pure synthetic audit orchestration tests; no corpus or model execution."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from manuscript.tools import audit_fpga_lm_update_scale as audit


@pytest.fixture(autouse=True)
def prohibit_science(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unmocked model/corpus/native execution")
    monkeypatch.setattr(audit.study.native, "EFSGD", forbidden)
    monkeypatch.setattr(audit.study.old, "observe", forbidden)
    monkeypatch.setattr(audit.study, "verify_evaluation", forbidden)
    monkeypatch.setattr(audit.study.pilot, "corpus_plan", forbidden)
    monkeypatch.setattr(audit.study.pilot, "window", forbidden)


def test_fixed_complete_protocol_and_source_binding():
    assert len(audit.expected_cells()) == 12
    assert len(audit.expected_rows()) == 14
    assert len(audit.study.VALIDATION) * 14 == 896
    assert len(audit.study.PARITY_STEPS) * 6 == 12
    sources = audit.source_identity()
    assert sources["manuscript/tools/test_fpga_lm_update_scale.py"] == audit.RUNNER_SHA
    assert "manuscript/tools/audit_fpga_lm_update_scale.py" in sources
    assert "tests/test_fpga_lm_update_scale_audit.py" in sources


def test_readiness_requires_closed_complete_json(tmp_path):
    directory = tmp_path / audit.study.cell_name("historical", 0, 18)
    directory.mkdir()
    path = directory / "summary.json"
    assert audit.ready(tmp_path, "historical", 0, 18) is False
    path.write_text('{"complete": true,')
    assert audit.ready(tmp_path, "historical", 0, 18) is False
    path.write_text('{"complete": 1}')
    assert audit.ready(tmp_path, "historical", 0, 18) is False
    path.write_text('{"complete": true}')
    assert audit.ready(tmp_path, "historical", 0, 18) is True


@pytest.mark.parametrize("kind", ["source", "manifest", "failure"])
def test_guard_rejects_source_input_or_terminal_failure(tmp_path, monkeypatch, kind):
    (tmp_path / "manifest.json").write_bytes(b"original")
    monkeypatch.setattr(audit, "source_identity", lambda: {"source": "original"})
    if kind == "source":
        monkeypatch.setattr(audit, "source_identity", lambda: {"source": "changed"})
    elif kind == "manifest":
        (tmp_path / "manifest.json").write_bytes(b"changed")
    else:
        (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError):
        audit.guard(tmp_path, b"original", {"source": "original"})


def fixture_rows(monkeypatch):
    monkeypatch.setattr(audit, "directory_hashes", lambda path: [{"file": "saved", "sha256": "pin"}])
    rows = {}
    for arm in audit.study.ARMS:
        name = "initial_" + arm
        rows[name] = {"schema": audit.SCHEMA, "name": name, "arm": arm, "seed": None, "k": 18,
                      "evaluation_step": 0, "validation_indices": list(audit.study.VALIDATION),
                      "evaluation": {"nll": 1}, "parity_anchors": [],
                      "directory_artifacts": audit.directory_hashes(None)}
    for arm, seed, k in audit.expected_cells():
        name = audit.study.cell_name(arm, seed, k)
        anchors = [{"step": step, "window_index": audit.study.old.training_order(seed)[step - 1],
                    "raw_parameter_gradients_equal": True, "native_output_arrays_equal": True,
                    "full_weights_remainders_equal": True, "parameter_coordinates": 1196032}
                   for step in audit.study.PARITY_STEPS] if k == 20 else []
        rows[name] = {"schema": audit.SCHEMA, "name": name, "arm": arm, "seed": seed, "k": k,
                      "evaluation_step": 1024, "validation_indices": list(audit.study.VALIDATION),
                      "evaluation": {"nll": 1}, "parity_anchors": anchors,
                      "directory_artifacts": audit.directory_hashes(None)}
    return rows


def test_all14_evaluations_and12_anchors_are_required(monkeypatch):
    rows = fixture_rows(monkeypatch)
    audit.validate_rows(Path("unused"), rows)
    for name in rows:
        missing = {key: value for key, value in rows.items() if key != name}
        with pytest.raises(ValueError):
            audit.validate_rows(Path("unused"), missing)


@pytest.mark.parametrize("mutation", ["anchor", "flag", "window", "cohort", "directory", "initial_arm"])
def test_resealed_audit_coverage_and_guards_fail(monkeypatch, mutation):
    rows = fixture_rows(monkeypatch)
    row = rows["diverse_seed_1_k20"]
    if mutation == "anchor":
        row["parity_anchors"].pop()
    elif mutation == "flag":
        row["parity_anchors"][0]["full_weights_remainders_equal"] = 1
    elif mutation == "window":
        row["parity_anchors"][0]["window_index"] += 1
    elif mutation == "cohort":
        row["validation_indices"] = list(range(152, 216))
    elif mutation == "directory":
        row["directory_artifacts"] = []
    else:
        rows["initial_historical"]["arm"] = "diverse"
    with pytest.raises(ValueError):
        audit.validate_rows(Path("unused"), rows)


def test_final_aggregate_must_match_parent(monkeypatch):
    rows = fixture_rows(monkeypatch)
    parent = {"initial": {arm: {"nll": 1} for arm in audit.study.ARMS},
              "cells": [{"arm": arm, "seed": seed, "k": k, "evaluation": {"nll": 1}}
                        for arm, seed, k in audit.expected_cells()]}
    audit.validate_rows(Path("unused"), rows, parent)
    parent["cells"][-1]["evaluation"]["nll"] = 2
    with pytest.raises(ValueError):
        audit.validate_rows(Path("unused"), rows, parent)


def test_completed_evaluation_recomputes_exact_cohort_once(tmp_path, monkeypatch):
    directory = tmp_path / "historical_seed_0_k18"
    directory.mkdir()
    (directory / "state_001024.i32").write_bytes(b"fixed-state")
    audit.study.pilot.write_json(directory / "summary.json", {"complete": True, "evaluation": {"nll": 1}})
    calls = []
    def recompute(root, payload, raw, dataset, shape, step, k, enabled):
        calls.append((root, payload, raw, dataset, step, k, enabled))
        return {"nll": 1}
    monkeypatch.setattr(audit.study, "verify_evaluation", recompute)
    result = audit.audit_evaluation(tmp_path, {"binding": {"dataset": "split"}}, b"raw", None,
                                    "historical", 0, 18)
    assert calls == [(directory, b"fixed-state", b"raw", "split", 1024, 18, True)]
    assert result["parity_anchors"] == []


def test_evaluation_payload_mutation_is_retained_as_failure(tmp_path, monkeypatch):
    directory = tmp_path / "historical_seed_0_k18"
    directory.mkdir()
    checkpoint = directory / "state_001024.i32"
    checkpoint.write_bytes(b"fixed-state")
    audit.study.pilot.write_json(directory / "summary.json", {"complete": True, "evaluation": {"nll": 1}})
    def changed(*args):
        checkpoint.write_bytes(b"changed")
        return {"nll": 1}
    monkeypatch.setattr(audit.study, "verify_evaluation", changed)
    with pytest.raises(ValueError):
        audit.audit_evaluation(tmp_path, {"binding": {"dataset": "split"}}, b"raw", None,
                               "historical", 0, 18)


def test_synthetic_parity_rederives_all_regions_and_uses_public_step(tmp_path, monkeypatch):
    names = audit.study.pilot.WEIGHTS
    before = {"weights": {name: [0, 10] for name in names},
              "remainders": {name: [1, -2] for name in names}}
    after = {"weights": {name: [-1, 11] for name in names},
             "remainders": {name: [4, -2] for name in names}}
    gradients = {name: [1048579, -1048576] for name in names}
    outputs = {"predictions": [1], "gradients": [-4, 4], "AD_HEAD": [0], "PROJ_IDX": [1], "TAP_IDX": [0]}
    observed = {"parameter_gradients": gradients, "predictions": outputs["predictions"],
                "gradients": outputs["gradients"], "native_routes": {name: outputs[name] for name in
                ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}, "saturation_events": 0,
                "route_diagnostics": {"backward_int32_overflow_detected": False}}
    for filename, payload in (("state_000000.i32", b"before"), ("parity_0001_before.i32", b"before"),
                              ("parity_0001_after.i32", b"after")):
        (tmp_path / filename).write_bytes(payload)
    audit.study.save_arrays(tmp_path / "parity_0001_gradients.npz", gradients)
    audit.study.save_arrays(tmp_path / "parity_0001_outputs.npz", outputs)
    for name in ("predictions", "gradients"):
        (tmp_path / f"training_{name}.i32").write_bytes(audit.study.screen._i32(outputs[name]))
    monkeypatch.setattr(audit.study, "state_config", lambda payload, *args: copy.deepcopy(
        before if payload == b"before" else after))
    monkeypatch.setattr(audit.study.pilot, "window", lambda *args: ([0], [1]))
    monkeypatch.setattr(audit.study.old, "observe", lambda *args, **kwargs: copy.deepcopy(observed))
    shape = SimpleNamespace(WORDS_PRED=1, WORDS_GRAD_LOGIT=2, **{"WORDS_" + name: 2 for name in names})
    result = audit.audit_parity(tmp_path, b"raw", {}, shape, 0, 1)
    assert result["parameter_coordinates"] == 2 * len(names)
    assert result["full_weights_remainders_equal"] is True
    after["remainders"][names[-1]][-1] += 1
    with pytest.raises(ValueError):
        audit.audit_parity(tmp_path, b"raw", {}, shape, 0, 1)


@pytest.mark.parametrize("nested", ["same", "child", "ancestor"])
def test_cannot_mutate_scientific_output_tree(tmp_path, nested):
    root = tmp_path / "run"
    root.mkdir()
    output = {"same": root, "child": root / "audit", "ancestor": tmp_path}[nested]
    with pytest.raises(ValueError, match="outside"):
        audit.run(SimpleNamespace(run=root, record=tmp_path / "public.json", output=output, resume=False))
    assert list(root.iterdir()) == []
