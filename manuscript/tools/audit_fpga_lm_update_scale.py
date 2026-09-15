"""Incremental, read-only replay of the frozen update-scale study's readouts.

Only completed cells are opened. This repeats the declared 896 development
windows and 12 saved training-gradient anchors, never an optimizer trajectory,
reserved cohort, final test, native executable, or model-selection experiment.
Audit receipts live outside the immutable scientific run and may be resumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import test_fpga_lm_update_scale as study  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_update_scale.incremental_audit.v1"
RUNNER_SHA = "5d5bd0bf540e6cd147a79218b50537fafb3ddf596d9ef05836daf8eafed9cdef"
TESTS_SHA = "fdde0bcd0f9dde5b7174198c6917740e40b8c335cf59ab03db41ba34d12fb7e8"
DEADLINE_S, POLL_S = 7200, 45


def source_identity():
    result = study.source_identity()
    study.equal(result["manuscript/tools/test_fpga_lm_update_scale.py"], RUNNER_SHA,
                "exact assigned scientific runner")
    study.equal(result["tests/test_fpga_lm_update_scale.py"], TESTS_SHA,
                "exact assigned scientific tests")
    for name in ("manuscript/tools/audit_fpga_lm_update_scale.py",
                 "tests/test_fpga_lm_update_scale_audit.py"):
        result[name] = study.pilot.digest((REPO / name).read_bytes())
    return result


def directory_hashes(path):
    return study.ancestry.artifact_manifest(Path(path))


def expected_cells():
    return [(arm, seed, k) for arm in study.ARMS for seed in study.SEEDS for k in study.K_VALUES]


def expected_rows():
    return ["initial_" + arm for arm in study.ARMS] + [
        study.cell_name(*cell) for cell in expected_cells()]


def ready(root, arm, seed, k):
    path = root / study.cell_name(arm, seed, k) / "summary.json"
    if not path.is_file():
        return False
    try:
        return study.read(path).get("complete") is True
    except json.JSONDecodeError:
        # The producer creates its final summary exclusively, then closes it.
        # A partially written JSON document is not a completed-cell signal.
        return False


def guard(root, manifest_bytes, sources):
    study.equal(source_identity(), sources, "audit/scientific sources unchanged")
    study.equal((root / "manifest.json").read_bytes(), manifest_bytes, "run manifest unchanged")
    if (root / "failure.json").exists():
        raise ValueError("scientific run failed; no complete independent audit")


def audit_parity(directory, raw, dataset, shape, seed, step):
    """Recompute one already-charged window; compare all G/output/state bytes."""
    if type(step) is not int or step not in study.PARITY_STEPS:
        raise ValueError("undeclared parity anchor")
    order = study.old.training_order(seed)
    before_bytes = (directory / f"parity_{step:04d}_before.i32").read_bytes()
    study.equal(before_bytes, (directory / f"state_{step-1:06d}.i32").read_bytes(),
                "parity anchor is complete saved checkpoint")
    before = study.state_config(before_bytes, shape, step - 1, 20)
    after = study.state_config((directory / f"parity_{step:04d}_after.i32").read_bytes(),
                               shape, step, 20)
    tokens, targets = study.pilot.window(raw, dataset, "train", order[step - 1])
    observed = study.old.observe(before["weights"], tokens, targets, shape, zshift=2,
                                 include_parameter_gradients=True)
    if (type(observed["saturation_events"]) is not int or observed["saturation_events"] != 0
            or observed["route_diagnostics"]["backward_int32_overflow_detected"] is not False):
        raise ValueError("parity-anchor clipping or overflow")
    gradients = study.load_arrays(directory / f"parity_{step:04d}_gradients.npz")
    outputs = study.load_arrays(directory / f"parity_{step:04d}_outputs.npz")
    study.equal(gradients, observed["parameter_gradients"], "all recomputed raw parameter gradients")
    study.equal(outputs, {"predictions": observed["predictions"], "gradients": observed["gradients"],
                          **observed["native_routes"]}, "all recomputed parity output arrays")
    for name, words in (("predictions", shape.WORDS_PRED), ("gradients", shape.WORDS_GRAD_LOGIT)):
        with (directory / f"training_{name}.i32").open("rb") as stream:
            stream.seek((step - 1) * words * 4)
            study.equal(stream.read(words * 4), study.screen._i32(outputs[name]),
                        "parity outputs match recorded training arrays")
    for name in study.pilot.WEIGHTS:
        weights, remainder = study.fixed_point_ef_step(
            study.torch.tensor(before["weights"][name], dtype=study.torch.int16),
            study.torch.tensor(before["remainders"][name], dtype=study.torch.int32),
            study.torch.tensor(gradients[name], dtype=study.torch.int32), k=20)
        study.equal([weights.tolist(), remainder.tolist()],
                    [after["weights"][name], after["remainders"][name]],
                    "all public-primitive versus saved native weights/remainders")
    return {"step": step, "window_index": order[step - 1], "raw_parameter_gradients_equal": True,
            "native_output_arrays_equal": True, "full_weights_remainders_equal": True,
            "parameter_coordinates": sum(getattr(shape, "WORDS_" + name) for name in study.pilot.WEIGHTS)}


def audit_evaluation(root, manifest, raw, shape, arm, seed=None, k=18):
    initial = seed is None
    name = "initial_" + arm if initial else study.cell_name(arm, seed, k)
    directory = root / name
    if not ready(root, arm, 0 if initial else seed, 18 if initial else k):
        raise ValueError("cell is not completed")
    before = directory_hashes(directory)
    step = 0 if initial else study.STEPS
    if initial:
        path = Path(manifest["binding"]["states"][study.cell_name(arm, 0, 18)]["0"])
        payload = path.read_bytes()
        identities = {row["path"]: row for row in manifest["binding"]["inputs"]}
        study.equal(study.identity(path), identities[str(path)], "bound initial-state bytes")
    else:
        payload = (directory / f"state_{step:06d}.i32").read_bytes()
    result = study.verify_evaluation(directory, payload, raw, manifest["binding"]["dataset"],
                                     shape, step, k, True)
    if not initial:
        study.equal(study.read(directory / "summary.json")["evaluation"], result,
                    "independent recomputed cell aggregate")
    anchors = []
    if not initial and k == 20:
        for anchor in study.PARITY_STEPS:
            anchors.append(audit_parity(directory, raw, manifest["binding"]["dataset"],
                                        shape, seed, anchor))
    study.equal(directory_hashes(directory), before, "completed directory unchanged through replay")
    return {"schema": SCHEMA, "name": name, "arm": arm, "seed": seed, "k": k,
            "evaluation_step": step, "validation_indices": list(study.VALIDATION),
            "evaluation": result, "parity_anchors": anchors, "directory_artifacts": before}


def validate_rows(root, rows, parent=None):
    study.equal(sorted(rows), sorted(expected_rows()), "all14 independent evaluations required")
    anchors = 0
    for name in expected_rows():
        row = rows[name]
        study.equal(row["schema"], SCHEMA, "audit row schema")
        study.equal(row["name"], name, "exact row identity")
        study.equal(row["validation_indices"], list(study.VALIDATION), "fixed development cohort")
        study.equal(directory_hashes(root / name), row["directory_artifacts"],
                    "audited directory remains byte-identical")
        initial = name.startswith("initial_")
        if initial:
            study.equal(name, "initial_" + row["arm"], "exact initial arm")
            study.equal([row["seed"], row["k"], row["evaluation_step"]], [None, 18, 0], "initial row")
            expected_anchors = []
            if parent is not None:
                study.equal(row["evaluation"], parent["initial"][row["arm"]], "initial aggregate bound at seal")
        else:
            study.equal(name, study.cell_name(row["arm"], row["seed"], row["k"]), "cell identity")
            study.equal(row["evaluation_step"], study.STEPS, "final-only evaluation")
            expected_anchors = list(study.PARITY_STEPS) if row["k"] == 20 else []
            if parent is not None:
                matches = [cell for cell in parent["cells"] if
                           (cell["arm"], cell["seed"], cell["k"]) == (row["arm"], row["seed"], row["k"])]
                study.equal(len(matches), 1, "one matching sealed cell")
                study.equal(row["evaluation"], matches[0]["evaluation"], "sealed versus independent evaluation")
        study.equal([item["step"] for item in row["parity_anchors"]], expected_anchors, "all12 saved anchors")
        for item in row["parity_anchors"]:
            for flag in ("raw_parameter_gradients_equal", "native_output_arrays_equal", "full_weights_remainders_equal"):
                if item[flag] is not True:
                    raise ValueError("failed parity anchor")
            study.equal(item["window_index"], study.old.training_order(row["seed"])[item["step"] - 1],
                        "actual shuffled parity window")
            anchors += 1
    study.equal(anchors, 12, "all12 independently recomputed anchors")


def run(config):
    root = Path(config.run).resolve(strict=True)
    record = Path(config.record).resolve()
    output = Path(config.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("audit output must be outside the scientific run")
    if record == output or output in record.parents:
        raise ValueError("scientific receipt must be outside audit output")
    if output.exists() and not config.resume:
        raise FileExistsError("audit output already exists; explicit --resume required")
    if not output.exists():
        study.old.prior.capacity._new_external_path(output)
    sources = source_identity()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = study.read(root / "manifest.json")
    study.equal(manifest["run_directory"], str(root), "exact scientific run directory")
    study.frozen(manifest)
    raw, dataset = study.pilot.corpus_plan(manifest["binding"]["dataset"]["path"],
                                         study.STEPS, max(study.VALIDATION) + 1)
    study.equal(dataset, manifest["binding"]["dataset"], "bound corpus bytes and splits")
    audit_manifest = {"schema": SCHEMA, "run_directory": str(root), "public_record": str(record),
                      "source_hashes": sources, "run_manifest_sha256": study.pilot.digest(manifest_bytes),
                      "validation_indices": list(study.VALIDATION), "parity_steps": list(study.PARITY_STEPS),
                      "deadline_s": DEADLINE_S, "poll_s": POLL_S, "native_execution": False,
                      "new_training_updates": 0, "reserved_or_final_test_evaluated": False}
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        study.pilot.write_json(output / "manifest.json", audit_manifest)
    else:
        study.equal(study.read(output / "manifest.json"), audit_manifest, "resume exact audit protocol/source")
    audit_manifest_bytes = (output / "manifest.json").read_bytes()
    if (output / "summary.json").exists():
        raise FileExistsError("audit is already complete; its receipts remain unchanged")
    if (output / "failure.json").exists():
        raise ValueError("failed audit retained; choose a new output, no automatic retry")
    rows = {}
    previous, installed = None, False
    try:
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("audit two-hour deadline")))
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        study.torch.set_num_threads(1)
        shape = study.pilot.layout(study.WIDTH)
        for path in output.glob("row_*.json"):
            value = study.read(path)
            if value["name"] not in expected_rows() or path.name != "row_" + value["name"] + ".json":
                raise ValueError("unknown resume row")
            study.equal(directory_hashes(root / value["name"]), value["directory_artifacts"], "unchanged resume payloads")
            rows[value["name"]] = value
        while True:
            guard(root, manifest_bytes, sources)
            for arm in study.ARMS:
                name = "initial_" + arm
                if name not in rows and ready(root, arm, 0, 18):
                    rows[name] = audit_evaluation(root, manifest, raw, shape, arm)
                    guard(root, manifest_bytes, sources)
                    study.pilot.write_json(output / ("row_" + name + ".json"), rows[name])
                    print(f"AUDITED {name}: 64 windows", flush=True)
            for arm, seed, k in expected_cells():
                name = study.cell_name(arm, seed, k)
                if name not in rows and ready(root, arm, seed, k):
                    rows[name] = audit_evaluation(root, manifest, raw, shape, arm, seed, k)
                    guard(root, manifest_bytes, sources)
                    study.pilot.write_json(output / ("row_" + name + ".json"), rows[name])
                    print(f"AUDITED {name}: NLL {rows[name]['evaluation']['lut_normalized_nll']:.9f}; "
                          f"{len(rows[name]['parity_anchors'])} anchors", flush=True)
            if len(rows) == len(expected_rows()) and record.is_file():
                # This is the ordinary integrity replay, not a second expensive
                # numerical readout replay. The producer may still be closing
                # its public receipt; malformed partial JSON is not completion.
                try:
                    study.read(record)
                except json.JSONDecodeError:
                    pass
                else:
                    public_bytes = record.read_bytes()
                    parent = study.check(root, record, recompute=False)
                    validate_rows(root, rows, parent)
                    guard(root, manifest_bytes, sources)
                    study.equal((output / "manifest.json").read_bytes(), audit_manifest_bytes,
                                "audit manifest unchanged")
                    for name, row in rows.items():
                        study.equal(study.read(output / ("row_" + name + ".json")), row,
                                    "saved audit rows equal independently computed results")
                    study.equal(record.read_bytes(), public_bytes, "scientific public receipt unchanged")
                    result = {"schema": SCHEMA, "complete": True, "run_directory": str(root),
                              "public_record_sha256": study.pilot.digest(public_bytes),
                              "audit_manifest_sha256": study.pilot.digest(audit_manifest_bytes),
                              "source_hashes": sources, "evaluations_recomputed": 14,
                              "validation_windows_recomputed": 896, "parity_anchors_recomputed": 12,
                              "new_training_updates": 0, "native_execution": False,
                              "reserved_or_final_test_evaluated": False, "gates": parent["gates"],
                              "row_sha256": {name: study.pilot.digest((output / ("row_" + name + ".json")).read_bytes())
                                             for name in expected_rows()}}
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                    installed = False
                    study.pilot.write_json(output / "summary.json", result)
                    print(json.dumps({key: result[key] for key in ("complete", "evaluations_recomputed",
                          "validation_windows_recomputed", "parity_anchors_recomputed")}), flush=True)
                    return result
            print(f"WAITING: {len(rows)}/14 readouts audited", flush=True)
            time.sleep(POLL_S)
    except BaseException as error:
        if not (output / "failure.json").exists():
            study.pilot.write_json(output / "failure.json", {"complete": False, "type": type(error).__name__,
                                   "error": str(error), "completed_rows": sorted(rows)})
        raise
    finally:
        if installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
