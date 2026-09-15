"""Controlled algebra, receipts and lifecycle checks; no research study execution."""
import copy
import signal
import subprocess
from types import SimpleNamespace

import pytest

from manuscript.tools import check_fpga_lm_bigram_capacity as capacity
from manuscript.tools.observe_fpga_lm_routes import observe


@pytest.fixture
def shape():
    value = SimpleNamespace(ML_D=16, ML_V=4, ML_N_LAYERS=4, ML_K=3,
                            ML_L_SEQ=8, ML_BATCH=1, ML_LOG2_D=4)
    counts = {"EMBED": 64, "MH": 4 * 16 * 3, "MP": 4 * 16 * 16, "MB": 64, "W_HEAD": 64,
              "TAP_IDX": 4 * 8 * 16, "PROJ_IDX": 4 * 8 * 16, "AD_HEAD": 8 * 4,
              "BS_STREAM": 4 * 8 * 16, "TOKENS": 8, "TARGETS": 8,
              "GRAD_LOGIT": 8 * 4, "PRED": 8, "CFG": 2, "CE_LUT": 256}
    offset = 0
    for name, count in counts.items():
        setattr(value, "WORDS_" + name, count)
        setattr(value, "OFF_" + name, offset)
        offset += count
    value.ML_BUFFER_WORDS = offset
    value.ML_BUFFER_BYTES = offset * 4
    value.build_init = lambda seed: [0] * offset
    return value


@pytest.fixture
def counts():
    # Ratios are exact powers of two, so log-rounding has no numerical ambiguity.
    return [[8, 4, 2, 1], [1, 2, 4, 8], [4, 8, 1, 2], [2, 1, 8, 4]]


@pytest.mark.parametrize("shift", capacity.SHIFTS)
def test_exact_embedding_head_and_zero_projection_construction(shape, counts, shift):
    weights, facts = capacity.construct_weights(counts, shape, shift)
    assert facts["anchors"] == [0, 5, 10, 15]
    assert facts["maximum_unclipped_lut_index"] == 48
    assert facts["minimum_head_margin_bound"] == 8192 - (48 << shift)
    assert facts["count_pairs_excluding_pseudocounts"] == 44
    for name in ("MH", "MP", "MB"):
        assert not any(weights[name])
    assert [weights["EMBED"][index] for index in facts["anchors"]] == [0, -16 << shift, -32 << shift, -48 << shift]
    for v, anchor in enumerate(facts["anchors"]):
        assert weights["W_HEAD"][v * shape.ML_D:(v + 1) * shape.ML_D] == [
            0 if d == anchor else -8192 for d in range(shape.ML_D)]
    assert all(weights["EMBED"][t * shape.ML_D + d] == 0
               for t in range(shape.ML_V) for d in range(shape.ML_D) if d not in facts["anchors"])


def test_declared_nearest_rounding_uses_ties_upward(shape, counts, monkeypatch):
    monkeypatch.setattr(capacity.math, "log2", lambda ratio: 2.5 / 16 if ratio != 1 else 0)
    weights, facts = capacity.construct_weights(counts, shape, 0)
    assert facts["maximum_unclipped_lut_index"] == 3
    assert weights["EMBED"][5] == -3


@pytest.mark.parametrize("shift", [True, False, 3, 5, -1, 1.0, "1"])
def test_only_exact_declared_shifts_are_allowed(shape, counts, shift):
    with pytest.raises(ValueError):
        capacity.construct_weights(counts, shape, shift)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "1", None])
def test_count_fitting_requires_positive_literal_integer_counts(shape, counts, bad):
    counts[0][1] = bad
    with pytest.raises(ValueError):
        capacity.construct_weights(counts, shape, 0)


def test_no_saturation_and_strict_anchor_bound_is_checked_before_use(shape, counts):
    counts[0][0] = 1 << 100
    with pytest.raises(ValueError, match="bound"):
        capacity.construct_weights(counts, shape, 4)
    shape.ML_D = 2
    with pytest.raises(ValueError, match="width"):
        capacity.construct_weights(counts, shape, 0)


def test_constructed_logits_and_native_surrogate_are_exactly_scale_invariant(shape, counts):
    tokens, targets = [0, 1, 2, 3] * 2, [1, 2, 3, 0] * 2
    rows = []
    for shift in capacity.SHIFTS:
        weights, _ = capacity.construct_weights(counts, shape, shift)
        row = capacity.analyze_window(weights, counts, tokens, targets, shape, shift)
        assert row["saturation_events"] == 0
        assert row["context_distribution_equal"] is True
        assert row["logit_differences_equal_constructed_embedding"] is True
        assert row["head_routes"] == [0, 5, 10, 15] * 8
        rows.append(row)
    assert all(row["scores"] == rows[0]["scores"] for row in rows)
    assert rows[0]["scores"]["predictions"] == [0, 3, 1, 2] * 2
    assert rows[0]["scores"]["gradients"][:4] == [32768, -45056, 8192, 4096]


def test_log_quantization_can_change_top1_without_saturation(shape):
    counts = [[100, 101, 1, 1]] * 4
    weights, _ = capacity.construct_weights(counts, shape, 0)
    row = capacity.analyze_window(weights, counts, [0] * 8, [1] * 8, shape, 0)
    assert row["scores"]["predictions"] == [0] * 8
    assert row["quantized_top1_changes_from_floating_counts"] == 8
    assert row["saturation_events"] == 0


def test_shared_projection_cancels_but_residual_embedding_and_head_gradients_survive(shape, counts):
    weights, _ = capacity.construct_weights(counts, shape, 2)
    result = observe(weights, [0] * 8, [1] * 8, shape, zshift=2, include_parameter_gradients=True)
    gradients = result["parameter_gradients"]
    assert not any(gradients["MH"])
    assert any(gradients["EMBED"]) and any(gradients["W_HEAD"])
    assert any(gradients["MP"]) and any(gradients["MB"])
    assert sum(gradients["EMBED"]) == sum(gradients["MB"]) == 0
    for layer in result["route_diagnostics"]["projection_layers"]:
        assert layer["single_projection_winner_positions"] == 8
        assert layer["zero_projection_input_vjp_on_nonzero_incoming_positions"] == 8
        assert layer["nonzero_temporal_carry_positions"] == 0
        assert layer["nonzero_centering_remainder_positions"] == 0


def test_recipe_does_not_disguise_host_fitting_as_gradient_or_fpga_learning():
    recipe = capacity.recipe()
    assert recipe["host_count_fitting"] is True
    assert recipe["gradient_training"] is False
    assert recipe["native_training_updates"] == 0
    assert recipe["native_learning_rate"] == 0
    assert recipe["fitted_supervised_pairs"] == 131072
    assert recipe["shifts"] == [0, 1, 2, 4]
    assert recipe["validation_indices"] == list(range(40, 56))
    assert recipe["training_indices"] == list(range(1024))
    assert recipe["physical_fpga_execution"] is recipe["cloud_operations"] is recipe["heldout_test_evaluated"] is False
    assert capacity.source_identity().items() >= capacity.calibration.source_identity().items()
    assert "manuscript/tools/check_fpga_lm_bigram_capacity.py" in capacity.source_identity()


@pytest.fixture
def worker(tmp_path, monkeypatch, shape):
    # Tiny controlled bridge, not a native research run. Four rows each add32768
    # observations, preserving the real fixed fit-count invariant.
    counts = [[32769, 1, 1, 1] for _ in range(4)]
    monkeypatch.setattr(capacity, "VALIDATION_INDICES", (40, 41))
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"fixture only")
    parent = tmp_path / "parent"
    parent.mkdir()
    library_path = parent / "library.so"
    library_path.write_bytes(b"fixture library")
    library = {"path": str(library_path), "sha256": capacity.pilot.digest(library_path.read_bytes())}
    manifest = {"schema": capacity.SCHEMA, "recipe": capacity.recipe(), "backend": "csim",
                "dataset": {"path": str(corpus), "sha256": capacity.pilot.digest(corpus.read_bytes())},
                "layout": capacity.pilot.layout_values(shape), "source_hashes": capacity.source_identity(),
                "plan_sha256": "fixture"}
    controls = {"bigram_pair_counts": counts, "unigram": {"nll": 1.0, "accuracy": 0.5},
                "bigram": {"nll": 1.0, "accuracy": 0.5}}
    for name, value in (("manifest.json", manifest), ("counts.json", controls), ("library.json", library)):
        capacity.pilot.write_json(parent / name, value)
    monkeypatch.setattr(capacity, "_common", lambda path: (corpus.read_bytes(), shape, copy.deepcopy(controls), copy.deepcopy(manifest)))
    monkeypatch.setattr(capacity.pilot, "window", lambda *args: ([0, 1, 2, 3] * 2, [1, 0, 2, 0] * 2))
    monkeypatch.setattr(capacity.audit, "_verify_library", lambda *args: {})

    class Bridge:
        instances = []

        def __init__(self, *args):
            self.buffer = [0] * shape.ML_BUFFER_WORDS
            self.calls = []
            self.closed = False
            self.instances.append(self)

        def write(self, offset, values):
            self.buffer[offset:offset + len(values)] = values

        def read(self, offset, count):
            return self.buffer[offset:offset + count]

        def step(self, lr, threshold):
            self.calls.append((lr, threshold))
            weights = {name: self.read(getattr(shape, "OFF_" + name), getattr(shape, "WORDS_" + name))
                       for name in capacity.pilot.WEIGHTS}
            row = capacity.analyze_window(weights, counts, self.read(shape.OFF_TOKENS, 8),
                                          self.read(shape.OFF_TARGETS, 8), shape, self.buffer[shape.OFF_CFG + 1])
            for offset, values in ((shape.OFF_PRED, row["scores"]["predictions"]),
                                   (shape.OFF_GRAD_LOGIT, row["scores"]["gradients"]),
                                   (shape.OFF_AD_HEAD, row["head_routes"])):
                self.write(offset, values)

        def close(self):
            self.closed = True

    config = SimpleNamespace(corpus=corpus, parent=parent, output=parent / "shift_2", zshift=2)
    return SimpleNamespace(config=config, bridge=Bridge, manifest=manifest, controls=controls,
                           raw=corpus.read_bytes(), shape=shape, library=library)


def test_worker_uses_lr0_checks_full_checkpoint_and_recomputes_every_saved_array(worker):
    result = capacity.run_worker(worker.config, worker.bridge)
    assert result["complete"] is True and result["native_training_updates"] == 0
    assert result["weights_unchanged"] is True
    assert worker.bridge.instances[0].calls == [(0, 2048), (0, 2048)]
    assert worker.bridge.instances[0].closed is True
    verified = capacity.verify_worker(worker.config.output, worker.manifest, worker.controls,
                                      worker.raw, worker.shape, 2, worker.library)
    assert verified == result


@pytest.mark.parametrize("failure", ["native_gradients", "head_routes", "weights", "sources", "close"])
def test_worker_failure_never_publishes_success_and_retains_failure(worker, monkeypatch, failure):
    base = worker.bridge

    class Broken(base):
        def step(self, lr, threshold):
            super().step(lr, threshold)
            if failure == "native_gradients":
                self.buffer[worker.shape.OFF_GRAD_LOGIT] += 1
            elif failure == "head_routes":
                self.buffer[worker.shape.OFF_AD_HEAD] += 1
            elif failure == "weights":
                self.buffer[worker.shape.OFF_W_HEAD] += 1

        def close(self):
            super().close()
            if failure == "close":
                raise RuntimeError("fixture close failure")

    if failure == "sources":
        monkeypatch.setattr(capacity, "_frozen", lambda *args: (_ for _ in ()).throw(ValueError("source changed")))
    with pytest.raises((ValueError, RuntimeError)):
        capacity.run_worker(worker.config, Broken)
    assert (worker.config.output / "failure.json").is_file()
    assert not (worker.config.output / "summary.json").exists()


@pytest.mark.parametrize("artifact", ["checkpoint_000000.i32", "window_040_gradients.i32", "window_041_head_routes.i32"])
def test_sealed_checkpoint_or_native_array_mutation_is_rejected(worker, artifact):
    capacity.run_worker(worker.config, worker.bridge)
    path = worker.config.output / artifact
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        capacity.verify_worker(worker.config.output, worker.manifest, worker.controls,
                               worker.raw, worker.shape, 2, worker.library)


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("fixture", 240), KeyboardInterrupt()])
def test_bounded_worker_timeout_reaps_only_its_owned_process_group(tmp_path, monkeypatch, failure):
    calls = []

    class Process:
        pid = 123456
        returncode = -9

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                raise failure
            return -9

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(capacity.subprocess, "Popen", popen)
    monkeypatch.setattr(capacity.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            capacity._run_process(["fixture"], tmp_path / "log")
    else:
        assert capacity._run_process(["fixture"], tmp_path / "log") == {"returncode": -9, "timed_out": True}
    assert (123456, signal.SIGKILL) in calls
    assert ("wait", 240) in calls and ("wait", 10) in calls


def test_existing_output_is_never_overwritten(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("immutable")
    with pytest.raises(FileExistsError):
        capacity.run(SimpleNamespace(output=target, corpus=tmp_path / "missing"))
    assert marker.read_text() == "immutable"


def test_cli_does_not_allow_an_undeclared_single_cell_public_run(tmp_path):
    with pytest.raises(SystemExit):
        capacity.main(["--corpus", str(tmp_path / "corpus"), "--output", str(tmp_path / "output"), "--zshift", "2"])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("schema", ["tikitorch.fpga_lm_head_screen.v1", "tikitorch.fpga_lm_budget.v1",
                                   "tikitorch.fpga_lm_training_scales.v1", capacity.SCHEMA])
def test_new_outputs_never_nest_inside_live_or_historical_lm_runs(tmp_path, schema):
    root = tmp_path / "run"
    root.mkdir()
    capacity.pilot.write_json(root / "manifest.json", {"schema": schema})
    output = root / "nested" / "new_run"
    with pytest.raises(ValueError, match="nested"):
        capacity.run(SimpleNamespace(output=output, corpus=tmp_path / "missing"))
    assert not output.parent.exists()


@pytest.mark.parametrize("change", ["sibling", "outside", "boolean_shift", "wrong_protocol", "stale_source"])
def test_worker_cannot_create_unowned_or_unauthorized_output(worker, monkeypatch, tmp_path, change):
    config = copy.copy(worker.config)
    if change == "sibling":
        config.output = config.parent / "shift_1"
    elif change == "outside":
        config.output = tmp_path / "unowned"
    elif change == "boolean_shift":
        config.zshift = True
    else:
        value = capacity._read(config.parent / "manifest.json")
        if change == "wrong_protocol":
            value["schema"] = "tikitorch.fpga_lm_training_scales.v1"
        else:
            value["source_hashes"] = {}
        (config.parent / "manifest.json").write_text(capacity.json.dumps(value))
    with pytest.raises(ValueError):
        capacity.run_worker(config, worker.bridge)
    assert not config.output.exists()
    assert not worker.bridge.instances


@pytest.mark.parametrize("destination", ["existing", "inside_new_run", "inside_old_run"])
def test_record_location_is_checked_before_starting_any_study(tmp_path, monkeypatch, destination):
    output = tmp_path / "new_run"
    old = tmp_path / "old_run"
    old.mkdir()
    capacity.pilot.write_json(old / "manifest.json", {"schema": "tikitorch.fpga_lm_training_scales.v1"})
    if destination == "existing":
        record = tmp_path / "existing.json"
        record.write_text("immutable")
    elif destination == "inside_new_run":
        record = output / "record.json"
    else:
        record = old / "record.json"
    calls = []
    monkeypatch.setattr(capacity, "run", lambda config: calls.append(config))
    with pytest.raises((ValueError, FileExistsError, SystemExit)):
        capacity.main(["--corpus", str(tmp_path / "missing"), "--output", str(output), "--record", str(record)])
    assert not calls and not output.exists()


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    corpus = tmp_path / "corpus"
    corpus.write_bytes(b"fixture")
    binary = root / "library.so"
    binary.write_bytes(b"fixture library")
    manifest = {"schema": capacity.SCHEMA, "source_hashes": capacity.source_identity(),
                "dataset": {"path": str(corpus), "sha256": capacity.pilot.digest(corpus.read_bytes())}}
    library = {"path": str(binary), "sha256": capacity.pilot.digest(binary.read_bytes())}
    cell = {"schema": capacity.SCHEMA, "complete": True, "host_count_fitting": True,
            "gradient_training": False, "native_training_updates": 0, "windows": 16, "positions": 2048,
            "weights_unchanged": True, "native_reference_equal": True, "saturation_events": 0,
            "context_distribution_equal": True}
    result = {"schema": capacity.SCHEMA, "complete": True, "failures": [], "host_count_fitting": True,
              "gradient_training": False, "native_training_updates": 0, "fitted_supervised_pairs": 131072,
              "distinct_validation_positions": 2048, "declared_cells": 4, "completed_cells": 4,
              "all_scale_outputs_equal": True, "physical_fpga_execution": False, "cloud_operations": False,
              "quality_gate_candidate": False, "historical_gates_modified": False, "heldout_test_evaluated": False,
              "cells": [{**cell, "zshift": shift} for shift in capacity.SHIFTS], "recipe": capacity.recipe(),
              "source_hashes": manifest["source_hashes"]}
    for name, value in (("manifest.json", manifest), ("library.json", library), ("summary.json", result)):
        capacity.pilot.write_json(root / name, value)
    capacity.pilot.write_json(root / "artifact_manifest.json", capacity._artifacts(root))
    return root, result, tmp_path / "record.json"


def test_compact_record_binds_seal_and_distinguishes_fitting_from_updates(publication):
    root, result, record = publication
    value = capacity.publish_record(record, root, result)
    assert value["run_directory"] == str(root)
    assert value["host_count_fitting"] is True
    assert value["native_training_updates"] == 0
    assert value["checkpoint_validation_windows"] == 64
    assert value["distinct_validation_positions"] == 2048
    assert value["saved_native_predictions_recomputed"] == 8192
    assert value["saved_native_gradient_entries_recomputed"] == 1048576
    assert value["saved_native_head_route_entries_recomputed"] == 1048576
    assert value["artifact_manifest_sha256"] == capacity.pilot.digest((root / "artifact_manifest.json").read_bytes())
    assert value["quality_gate_candidate"] is False
    with pytest.raises(FileExistsError):
        capacity.publish_record(record, root, result)


@pytest.mark.parametrize("name,value", [("complete", False), ("native_training_updates", False),
                                       ("completed_cells", 3), ("completed_cells", 4.0),
                                       ("host_count_fitting", False), ("quality_gate_candidate", True),
                                       ("cloud_operations", True), ("gradient_training", True)])
def test_invalid_or_partial_study_cannot_publish(publication, name, value):
    root, result, record = publication
    result[name] = value
    with pytest.raises(ValueError):
        capacity.publish_record(record, root, result)
    assert not record.exists()


@pytest.mark.parametrize("change", ["payload", "new_file", "source", "corpus", "cell_count"])
def test_publication_requires_current_sealed_artifacts_sources_and_full_coverage(publication, monkeypatch, change):
    root, result, record = publication
    if change == "payload":
        (root / "library.so").write_bytes(b"changed")
    elif change == "new_file":
        (root / "unexpected").write_text("late write")
    elif change == "source":
        monkeypatch.setattr(capacity, "source_identity", lambda: {})
    elif change == "corpus":
        (root.parent / "corpus").write_text("changed")
    else:
        result["cells"] = result["cells"][:3]
    with pytest.raises(ValueError):
        capacity.publish_record(record, root, result)
    assert not record.exists()


def test_failed_cli_study_keeps_raw_failure_without_publishing(tmp_path, monkeypatch):
    output, record = tmp_path / "output", tmp_path / "record.json"
    monkeypatch.setattr(capacity, "run", lambda config: {"complete": False})
    assert capacity.main(["--corpus", str(tmp_path / "missing"), "--output", str(output), "--record", str(record)]) == 1
    assert not record.exists()


@pytest.mark.parametrize("failed_shift", [None, 1])
def test_parent_attempts_all_four_cells_and_never_promotes_partial_completion(worker, tmp_path, monkeypatch, failed_shift):
    attempts = []

    def build(directory, width):
        directory.mkdir()
        binary = directory / "libmorphlm.so"
        binary.write_bytes(b"fixture compiled library")
        capacity.pilot.write_json(directory / "build.json", {"fixture": True})
        return binary

    def process(command, log):
        log.write_text("fixture process")
        shift = int(command[command.index("--zshift") + 1])
        attempts.append(shift)
        if shift == failed_shift:
            return {"returncode": 1, "timed_out": False}
        config = SimpleNamespace(corpus=worker.config.corpus, zshift=shift,
                                 parent=capacity.Path(command[command.index("--parent") + 1]),
                                 output=capacity.Path(command[command.index("--output") + 1]))
        capacity.run_worker(config, worker.bridge)
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(capacity.pilot, "build_csim", build)
    monkeypatch.setattr(capacity, "_run_process", process)
    output = tmp_path / "new_parent"
    result = capacity.run(SimpleNamespace(corpus=worker.config.corpus, output=output))
    assert attempts == [0, 1, 2, 4]
    assert result["completed_cells"] == (4 if failed_shift is None else 3)
    assert result["complete"] is (failed_shift is None)
    assert result["all_scale_outputs_equal"] is (failed_shift is None)
    assert result["gradient_training"] is False and result["quality_gate_candidate"] is False
    assert result["host_count_fitting"] is True and result["cloud_operations"] is False
    assert (output / "artifact_manifest.json").is_file()
    assert (output / "failure.json").exists() is (failed_shift is not None)
