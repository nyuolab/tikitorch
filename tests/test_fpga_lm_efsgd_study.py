"""Offline contracts for the separately declared integer optimizer experiment.

These tests never launch a native process, run a study, or contact AWS.
"""

import importlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.train_fpga_lm_efsgd")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(81029)
    raw = bytes(rng.randrange(128) for _ in range(190_000))
    path = tmp_path / "corpus.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 1024, 56)
    return raw, dataset


def exposure_oracle(raw, dataset, indices, validation):
    """Independent byte slicing: no production window/count/baseline helpers."""
    uni = [1] * 128
    bi = [[1] * 128 for _ in range(128)]
    for index in indices:
        start = dataset["splits"]["train"]["start"] + index * 129
        for position in range(128):
            token, target = raw[start + position:start + position + 2]
            uni[target] += 1
            bi[token][target] += 1
    results = {}
    for name, counts in (("unigram", [uni] * 128), ("bigram", bi)):
        totals = [sum(row) for row in counts]
        predictions = [max(range(128), key=row.__getitem__) for row in counts]
        losses, correct = [], 0
        for index in validation:
            start = dataset["splits"]["validation"]["start"] + index * 129
            for position in range(128):
                token, target = raw[start + position:start + position + 2]
                losses.append(-math.log(counts[token][target] / totals[token]))
                correct += predictions[token] == target
        results[name] = {"nll": sum(losses) / len(losses), "accuracy": correct / len(losses)}
    return results


def test_protocol_is_four_exploratory_cells_not_a_hardware_or_three_seed_gate(study):
    assert study.ARMS == ("anchors", "count_fitted")
    assert study.K_VALUES == (18, 20)
    assert (study.WIDTH, study.ZSHIFT, study.STEPS) == (512, 2, 1024)
    assert study.CHECKPOINTS == (0, 256, 1024)
    assert study.VALIDATION == tuple(range(40, 56))
    assert study.WORKER_TIMEOUT_S == 1200
    recipe = study.recipe()
    assert recipe["arms"] == ["anchors", "count_fitted"]
    assert recipe["k_values"] == [18, 20] and recipe["seed"] == 0
    assert recipe["training_indices"] == list(range(1024))
    assert recipe["validation_indices"] == list(range(40, 56))
    assert recipe["checkpoints"] == [0, 256, 1024]
    assert "trunc_toward_zero" in recipe["optimizer"]
    assert "not fresh holdout" in recipe["evaluation"]
    assert "no three-seed gate" in recipe["scope"]


def test_source_identity_covers_native_optimizer_and_frozen_reference(study):
    sources = study.source_identity()
    for dependency in (study.capacity, study.optimizer):
        for name, digest in dependency.source_identity().items():
            assert sources[name] == digest
    own = Path(study.__file__).resolve()
    assert sources[own.relative_to(pilot.REPO).as_posix()] == pilot.digest(own.read_bytes())


def test_real_initializers_use_the_declared_anchored_seed_and_exact_count_fit(study, corpus):
    raw, dataset = corpus
    shape = pilot.layout(512)
    anchored = study.screen.initialize_buffer(shape, 0, "anchors")
    anchored[shape.OFF_CFG + 1] = 2
    assert study.initial_buffer(shape, raw, dataset, "anchors") == anchored
    counts = [[1] * 128 for _ in range(128)]
    start = dataset["splits"]["train"]["start"]
    for index in range(1024):
        for offset in range(start + index * 129, start + index * 129 + 128):
            counts[raw[offset]][raw[offset + 1]] += 1
    weights, _ = study.capacity.construct_weights(counts, shape, 2)
    fitted = study.initial_buffer(shape, raw, dataset, "count_fitted")
    expected = list(anchored)
    for name, values in weights.items():
        offset = getattr(shape, "OFF_" + name)
        expected[offset:offset + len(values)] = values
    assert fitted == expected
    changed = bytearray(raw)
    split = dataset["splits"]["validation"]["start"]
    changed[split:] = b"?" * (len(changed) - split)
    assert study.initial_buffer(shape, bytes(changed), dataset, "count_fitted") == fitted


@pytest.mark.parametrize("arm,step", [(a, s) for a in ("anchors", "count_fitted") for s in (0, 256, 1024)])
def test_baselines_charge_actual_fitting_and_replay_exposures(study, corpus, arm, step):
    raw, dataset = corpus
    result = study.exposure_baseline(raw, dataset, arm, step)
    indices = (list(range(1024)) if arm == "count_fitted" else []) + list(range(step))
    if not indices:
        assert result is None
        return
    expected = exposure_oracle(raw, dataset, indices, range(40, 56))
    assert result["supervised_exposures"] == len(indices) * 128
    assert result["distinct_supervised_pairs"] == len(set(indices)) * 128
    assert result["alpha"] == 1
    for name in ("unigram", "bigram"):
        assert result[name]["nll"] == pytest.approx(expected[name]["nll"], abs=1e-12)
        assert result[name]["accuracy"] == expected[name]["accuracy"]


def test_fitted_checkpoint256_is_full_fit_plus_prefix_not_twice_prefix(study, corpus):
    raw, dataset = corpus
    result = study.exposure_baseline(raw, dataset, "count_fitted", 256)
    wrong = exposure_oracle(raw, dataset, list(range(256)) * 2, range(40, 56))
    assert result["supervised_exposures"] == 163840
    assert result["distinct_supervised_pairs"] == 131072
    assert abs(result["bigram"]["nll"] - wrong["bigram"]["nll"]) > 1e-6


@pytest.mark.parametrize("arm,step", [("anchors", 256), ("count_fitted", 0), ("count_fitted", 256)])
def test_baselines_never_fit_validation_test_or_cross_window_pairs(study, corpus, arm, step):
    raw, dataset = corpus
    changed = bytearray(raw)
    stop = 1024 if arm == "count_fitted" else step
    train = dataset["splits"]["train"]
    changed[train["start"] + stop * 129:train["stop"]] = b"!" * (train["stop"] - train["start"] - stop * 129)
    validation = dataset["splits"]["validation"]["start"]
    changed[validation:validation + 40 * 129] = b"?" * (40 * 129)
    test = dataset["splits"]["test"]["start"]
    changed[test:] = b"~" * (len(changed) - test)
    assert study.exposure_baseline(bytes(changed), dataset, arm, step) == \
        study.exposure_baseline(raw, dataset, arm, step)


@pytest.mark.parametrize("arm,step", [("other", 0), ("anchors", True), ("anchors", 0.0), ("anchors", 255)])
def test_undeclared_baseline_recipes_are_rejected(study, corpus, arm, step):
    with pytest.raises(ValueError, match="undeclared"):
        study.exposure_baseline(*corpus, arm, step)


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired, KeyboardInterrupt, RuntimeError])
def test_owned_worker_process_group_is_reaped_on_timeout_or_interruption(study, tmp_path, monkeypatch, failure):
    class Process:
        pid, returncode = 21897, -9

        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                if failure is subprocess.TimeoutExpired:
                    raise failure(["unit-only"], timeout)
                raise failure("unit-only")
            return self.returncode

    process, launches, kills = Process(), [], []

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return process

    monkeypatch.setattr(study.subprocess, "Popen", launch)
    monkeypatch.setattr(study.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    if failure is subprocess.TimeoutExpired:
        result = study._run_process(["unit-only"], tmp_path / "worker.log")
        assert result["timed_out"] is True
        assert result["timeout_s"] == 1200 and result["returncode"] == -9
    else:
        with pytest.raises(failure):
            study._run_process(["unit-only"], tmp_path / "worker.log")
    assert launches[0][1]["start_new_session"] is True
    assert process.waits == [1200, 10]
    assert kills == [(process.pid, signal.SIGKILL)] * 2


def test_process_logs_are_append_only(study, tmp_path, monkeypatch):
    path = tmp_path / "prior.log"
    path.write_text("prior failure\n")
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        study._run_process(["unit-only"], path)
    assert path.read_text() == "prior failure\n"


@pytest.fixture
def fake_worker(study, tmp_path, monkeypatch, corpus):
    """Actual full-state binary ABI and artifact paths; no native calls."""
    monkeypatch.setattr(study, "WIDTH", 16)
    monkeypatch.setattr(study, "STEPS", 4)
    monkeypatch.setattr(study, "CHECKPOINTS", (0, 2, 4))
    monkeypatch.setattr(study, "VALIDATION", (40,))
    raw, _ = corpus
    corpus_path = tmp_path / "corpus.txt"
    _, dataset = pilot.corpus_plan(corpus_path, 4, 41)
    shape = pilot.layout(16)
    controls = SimpleNamespace(train_error=None, close_error=None, evaluate_mutation=False,
                               import_mutation=False, wrong_native=False, opened=[], calls=[])

    def initializer(layout, data, plan, arm):
        initial = layout.build_init(0)
        initial[layout.OFF_CFG + 1] = 2
        for name in pilot.WEIGHTS:
            start, count = getattr(layout, "OFF_" + name), getattr(layout, "WORDS_" + name)
            initial[start:start + count] = [3 if arm == "count_fitted" else 0] * count
        return initial

    def observed(weights, tokens, targets, layout, zshift):
        value = weights["W_HEAD"][0]
        routes = {name: [0] * getattr(layout, "WORDS_" + name)
                  for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}
        return {"predictions": [int(token) for token in tokens], "gradients": [0] * 16384,
                "native_routes": routes, "lut_normalized_nll": 4.0 + value / 100,
                "accuracy": 0.25, "saturation_events": 0, "rounded_zero_target_mass": 0,
                "route_diagnostics": {"positions": 128, "nonzero_logit_gradient_positions": 128,
                    "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
                    "head_winners_ge16_positions": 128, "single_shared_head_winner_positions": 0,
                    "nonzero_embedding_input_vjp_positions": 128, "max_abs_head_input_vjp": 3,
                    "max_abs_embedding_input_vjp": 3, "backward_int32_overflow_detected": False,
                    "head_winner_counts": [16] * 128}}

    def disabled(weights, tokens, targets, layout, zshift, use_context):
        result = observed(weights, tokens, targets, layout, zshift)
        result["lut_normalized_nll"] += 0.2
        return result

    def build(directory, width):
        directory.mkdir(parents=True, exist_ok=False)
        executable = directory / "fake-native"
        executable.write_bytes(b"unit-only, never executable")
        sources = study.optimizer.source_identity()
        record = {"schema": study.optimizer.SCHEMA, "cpu_only": True, "existing_afi_compatible": False,
                  "source_hashes": sources, "source_sha256": study.optimizer.source_digest(sources),
                  "layout": pilot.layout_values(shape), "executable_sha256": pilot.digest(executable.read_bytes()),
                  "sanitizer": "undefined, nonrecovering", "compiler_version": "offline fixture",
                  "command": ["offline-cxx", *study.optimizer._flags(width, sources), "-I" + str(pilot.HLS),
                              str(study.optimizer.CPP), "-o", str(executable.resolve())]}
        pilot.write_json(directory / "build.json", record)
        return executable

    class Native:
        save_checkpoint = study.optimizer.EFSGD.save_checkpoint

        def __init__(self, executable, layout, initial, k, *, timeout_s, log_path):
            self.shape, self.k, self.step = layout, k, 0
            self.sources = study.optimizer.source_identity()
            self.weights = {name: initial[getattr(layout, "OFF_" + name):
                                         getattr(layout, "OFF_" + name) + getattr(layout, "WORDS_" + name)]
                            for name in pilot.WEIGHTS}
            self.remainders = {name: [0] * len(values) for name, values in self.weights.items()}
            self.identity = {"schema": study.optimizer.SCHEMA,
                "executable_sha256": pilot.digest(executable.read_bytes()),
                "build_sha256": pilot.digest((executable.parent / "build.json").read_bytes()),
                "source_hashes": self.sources, "layout": pilot.layout_values(layout),
                "cpu_only": True, "existing_afi_compatible": False}
            self.closed = False
            log_path.write_text("unit fake native, no process\n")
            controls.opened.append(self)
            assert timeout_s == 60

        def export_state(self):
            count = sum(map(len, self.weights.values()))
            digest = study.optimizer.source_digest(self.sources)
            identity = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
            identity = [value if value < 2 ** 31 else value - 2 ** 32 for value in identity]
            header = [study.optimizer.MAGIC, 1, 16, shape.ML_BUFFER_WORDS, count,
                      self.k, 1, 2, self.step, 0, shape.WORDS_CE_LUT, 0, *identity]
            values = header + [round(65536 * 2.0 ** (-i / 16.0)) for i in range(256)]
            for section in (self.weights, self.remainders):
                for name in pilot.WEIGHTS:
                    values.extend(section[name])
            return study.optimizer._pack(values)

        def import_state(self, payload):
            decoded = study.optimizer.decode_state(payload, shape)
            self.weights, self.remainders = decoded["weights"], decoded["remainders"]
            self.step, self.k = decoded["step"], decoded["k"]
            if controls.import_mutation:
                self.remainders["EMBED"][0] += 1

        def evaluate(self, tokens, targets):
            result = observed(self.weights, tokens, targets, shape, 2)
            if controls.evaluate_mutation:
                self.remainders["EMBED"][0] += 1
            if controls.wrong_native:
                result["predictions"][0] ^= 1
            return result

        def train(self, tokens, targets):
            controls.calls.append((list(tokens), list(targets)))
            if controls.train_error:
                raise controls.train_error("injected native failure")
            self.step += 1
            self.weights["W_HEAD"][0] -= 1
            self.remainders["EMBED"][0] = self.step
            return {"step": self.step, **observed(self.weights, tokens, targets, shape, 2)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True
            if controls.close_error:
                raise controls.close_error("injected cleanup failure")

    monkeypatch.setattr(study, "initial_buffer", initializer)
    monkeypatch.setattr(study, "observe", observed)
    monkeypatch.setattr(study.pilot, "integer_forward", disabled)
    monkeypatch.setattr(study.optimizer, "EFSGD", Native)
    monkeypatch.setattr(study.optimizer, "build", build)
    parent = tmp_path / "parent"
    parent.mkdir()
    executable = build(parent / "build", 16)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "dataset": dataset,
                "source_hashes": study.source_identity(), "plan_sha256": pilot.digest(study.PLAN.read_bytes()),
                "executable": {"path": str(executable), "sha256": pilot.digest(executable.read_bytes())}}
    pilot.write_json(parent / "manifest.json", manifest)
    config = SimpleNamespace(parent=parent, corpus=corpus_path, output=parent / "anchors_k18", arm="anchors", k=18)
    return SimpleNamespace(config=config, controls=controls, manifest=manifest, shape=shape, raw=raw,
                           parent=parent, executable=executable)


@pytest.mark.parametrize("arm,k", [("anchors", 18), ("count_fitted", 20)])
def test_fake_worker_full_binary_state_roundtrip_and_independent_verification(study, fake_worker, arm, k):
    case = fake_worker
    case.config.arm, case.config.k = arm, k
    case.config.output = case.parent / ("%s_k%d" % (arm, k))
    result = study.run_worker(case.config)
    assert result == study.verify_worker(case.config.output, case.manifest, arm, k)
    assert result["complete"] is True and case.controls.opened[0].closed
    assert result["gradient_pair_exposures"] == 512
    assert result["host_fitting_pair_exposures"] == (512 if arm == "count_fitted" else 0)
    assert result["distinct_training_pairs"] == 512
    assert result["three_seed_quality_gate_evaluated"] is False
    assert result["larger_paid_learning_benchmark_gate"] is False
    assert [row["step"] for row in result["evaluations"]] == [0, 2, 4]
    assert len(case.controls.calls) == 4
    for index, actual in enumerate(case.controls.calls):
        expected = pilot.window(case.raw, case.manifest["dataset"], "train", index)
        assert actual == (list(expected[0]), list(expected[1]))
    for step in (0, 2, 4):
        state = study.optimizer.decode_state((case.config.output / ("state_%06d.i32" % step)).read_bytes(), case.shape)
        assert state["step"] == step and state["k"] == k
        assert state["remainders"]["EMBED"][0] == step
    comparator = min(result["baselines_by_step"]["4"]["bigram"]["nll"], result["one_pass_final_baseline"]["bigram"]["nll"])
    assert result["conservative_bigram_advantage_nats"] == comparator - result["evaluations"][-1]["lut_normalized_nll"]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_or_cancellation_retains_exclusive_failure_artifacts(study, fake_worker, failure):
    case = fake_worker
    case.controls.train_error = failure
    with pytest.raises(failure):
        study.run_worker(case.config)
    root = case.config.output
    assert study._read(root / "failure.json")["type"] == failure.__name__
    assert not (root / "summary.json").exists()
    assert case.controls.opened[0].closed
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)
    prior = (root / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        study.run_worker(case.config)
    assert (root / "failure.json").read_bytes() == prior


@pytest.mark.parametrize("field", ["evaluate_mutation", "import_mutation", "wrong_native"])
def test_any_native_state_or_prediction_mismatch_fails_closed(study, fake_worker, field):
    case = fake_worker
    setattr(case.controls, field, True)
    with pytest.raises(ValueError):
        study.run_worker(case.config)
    assert (case.config.output / "failure.json").is_file()
    assert not (case.config.output / "summary.json").exists()
    assert case.controls.opened[0].closed


def test_success_is_not_written_before_native_cleanup(study, fake_worker):
    case = fake_worker
    case.controls.close_error = RuntimeError
    with pytest.raises(RuntimeError, match="cleanup"):
        study.run_worker(case.config)
    assert not (case.config.output / "summary.json").exists()
    assert (case.config.output / "failure.json").is_file()


@pytest.mark.parametrize("mutation", ["source", "corpus", "executable"])
def test_source_and_payload_changes_before_worker_start_do_not_create_output(study, fake_worker, monkeypatch, mutation):
    case = fake_worker
    if mutation == "source":
        monkeypatch.setattr(study, "source_identity", lambda: {})
    elif mutation == "corpus":
        case.config.corpus.write_bytes(b"changed")
    else:
        case.executable.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        study.run_worker(case.config)
    assert not case.config.output.exists() and not case.controls.opened


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_parent_build_failure_or_cancellation_is_sealed_and_not_overwritten(study, tmp_path, corpus, monkeypatch, failure):
    root = tmp_path / "new-study"
    config = SimpleNamespace(output=root, corpus=Path(corpus[1]["path"]))

    def fail(*args, **kwargs):
        raise failure("injected compiler failure")

    monkeypatch.setattr(study.optimizer, "build", fail)
    with pytest.raises(failure):
        study.run(config)
    assert study._read(root / "failure.json")["type"] == failure.__name__
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)
    before = (root / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        study.run(config)
    assert (root / "failure.json").read_bytes() == before


def rewrite_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def reseal(study, root):
    rewrite_json(root / "artifact_manifest.json", study._artifacts(root))


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("step", False), ("k", 20), ("config", {}),
    ("identity", {}), ("regions", []), ("format", "weights only"),
    ("file", "../state_000000.i32"), ("bytes", True),
])
def test_resealed_checkpoint_metadata_cannot_lie_about_state_or_provenance(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    record = study._read(root / "state_000000.json")
    record[field] = value
    rewrite_json(root / "state_000000.json", record)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["checkpoint"] = record
    rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, "anchors", 18)


@pytest.mark.parametrize("field,value", [
    ("window_index", 41), ("window_index", 40.0), ("native_reference_equal", 1),
    ("native_routes_equal", False), ("native_arrays", {}),
])
def test_resealed_evaluation_rows_require_exact_window_and_payload_receipts(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    row = study._read(root / "eval_000000_040.json")
    row[field] = value
    rewrite_json(root / "eval_000000_040.json", row)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["per_window"][0] = row
    rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, "anchors", 18)


@pytest.mark.parametrize("part", ["state", "native", "training", "baseline", "summary", "failure"])
def test_resealed_scientific_payload_and_success_forgery_are_rejected(study, fake_worker, part):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    if part == "state":
        path = root / "state_000002.i32"
        path.write_bytes(path.read_bytes()[:-4])
    elif part == "native":
        path = root / "eval_000002_040_gradients.i32"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    elif part == "training":
        path = root / "training_predictions.i32"
        path.write_bytes(path.read_bytes()[:-4])
    elif part == "failure":
        rewrite_json(root / "failure.json", {"complete": False})
    else:
        summary = study._read(root / "summary.json")
        if part == "baseline":
            summary["baselines_by_step"]["2"]["bigram"]["nll"] = 0
        else:
            summary["evaluations"][0]["step"] = False
        rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, "anchors", 18)


@pytest.mark.parametrize("failure_index", [None, 0, 3])
def test_parent_requires_every_declared_cell_and_retains_each_failed_receipt(study, fake_worker, tmp_path, monkeypatch, failure_index):
    case, launches = fake_worker, []

    def execute(command, path):
        path.write_text("unit subprocess fixture\n")
        launch_index = len(launches)
        launches.append(command)
        if failure_index == launch_index:
            return {"returncode": 2, "timed_out": False, "timeout_s": 1200, "elapsed_s": 0.1}
        get = lambda flag: command[command.index(flag) + 1]
        study.run_worker(SimpleNamespace(parent=Path(get("--parent")), output=Path(get("--output")),
                                        corpus=Path(get("--corpus")), arm=get("--arm"), k=int(get("--k"))))
        return {"returncode": 0, "timed_out": False, "timeout_s": 1200, "elapsed_s": 0.1}

    monkeypatch.setattr(study, "_run_process", execute)
    output = tmp_path / "full-parent"
    result = study.run(SimpleNamespace(output=output, corpus=case.config.corpus))
    assert len(launches) == 4
    assert result["declared_cells"] == 4
    assert result["completed_cells"] == (4 if failure_index is None else 3)
    assert result["complete"] is (failure_index is None)
    assert result["larger_paid_learning_benchmark_gate"] is False
    assert result["three_seed_quality_gate_evaluated"] is False
    assert result["physical_fpga_execution"] is False
    assert len(result["failures"]) == (0 if failure_index is None else 1)
    assert study._read(output / "artifact_manifest.json") == study._artifacts(output)
    if failure_index is not None:
        with pytest.raises(ValueError):
            study.publish_record(tmp_path / "forged-success.json", output, result)
        assert not (tmp_path / "forged-success.json").exists()


@pytest.mark.parametrize("returncode", [0, 17])
def test_even_completed_or_crashed_workers_have_owned_descendants_cleaned(study, tmp_path, monkeypatch, returncode):
    kills, launches = [], []
    process = SimpleNamespace(pid=31389, returncode=returncode, wait=lambda timeout: returncode)

    def launch(command, **kwargs):
        launches.append(kwargs)
        return process

    monkeypatch.setattr(study.subprocess, "Popen", launch)
    monkeypatch.setattr(study.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    result = study._run_process(["unit-only"], tmp_path / "worker.log")
    assert result["returncode"] == returncode
    assert result["timed_out"] is False
    assert launches[0]["start_new_session"] is True
    assert kills == [(31389, signal.SIGKILL)]


def test_nested_outputs_and_symlink_aliases_cannot_touch_immutable_prior_runs(study, tmp_path, corpus):
    history = tmp_path / "history"
    history.mkdir()
    pilot.write_json(history / "manifest.json", {"schema": "tikitorch.fpga_lm_earlier.v1"})
    alias = tmp_path / "history-alias"
    alias.symlink_to(history, target_is_directory=True)
    for ancestor in (history, alias):
        output = ancestor / "new-study"
        with pytest.raises(ValueError, match="nested"):
            study.run(SimpleNamespace(output=output, corpus=Path(corpus[1]["path"])))
        assert not output.exists()


def test_worker_cannot_escape_the_declared_child_directory(study, fake_worker, tmp_path):
    fake_worker.config.output = tmp_path / "unowned"
    with pytest.raises(ValueError, match="exactly"):
        study.run_worker(fake_worker.config)
    assert not fake_worker.config.output.exists()


@pytest.mark.parametrize("section", ["weights", "remainders"])
def test_even_rehashed_full_initial_state_must_match_declared_weights_and_zero_remainders(study, fake_worker, section):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    path = root / "state_000000.i32"
    values = study.optimizer._unpack(path.read_bytes())
    offset = study.optimizer.STATE_HEADER + case.shape.WORDS_CE_LUT
    if section == "remainders":
        offset += sum(getattr(case.shape, "WORDS_" + name) for name in pilot.WEIGHTS)
    values[offset] += 1
    payload = study.optimizer._pack(values)
    path.write_bytes(payload)
    record = study._read(root / "state_000000.json")
    record["sha256"] = pilot.digest(payload)
    rewrite_json(root / "state_000000.json", record)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["checkpoint"] = record
    summary["evaluations"][0]["state_sha256"] = record["sha256"]
    rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError, match="initial weights or zero"):
        study.verify_worker(root, case.manifest, "anchors", 18)


def test_real_abrupt_python_worker_cannot_leave_its_owned_python_child_running(study, tmp_path):
    """Only standard Python sleeping children, never a scientific/native job."""
    log = tmp_path / "abrupt-worker.log"
    command = [sys.executable, "-c",
               "import os,subprocess,sys; "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
               "print(p.pid,flush=True); os._exit(17)"]
    child_pid = None
    try:
        result = study._run_process(command, log)
        child_pid = int(log.read_text().strip())
        assert result["returncode"] == 17 and result["timed_out"] is False
        deadline = time.monotonic() + 3
        while True:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            # Minimal Linux containers can retain an inactive zombie until
            # their init reaps it; no platform-specific process-list tool.
            status = Path("/proc/%d/stat" % child_pid)
            try:
                if status.is_file() and status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    break
            except FileNotFoundError:
                break
            if time.monotonic() >= deadline:
                pytest.fail("abrupt worker left an active child outside its bounded lifetime")
            time.sleep(0.05)
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("saturation", [1, True, False, 0.0])
def test_matching_recomputed_clipping_requires_literal_integer_zero(study, fake_worker, monkeypatch, saturation):
    """Matching receipts cannot legitimize clipping or bool-as-integer counts."""
    original = study.observe

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["saturation_events"] = saturation
        return result

    monkeypatch.setattr(study, "observe", observed)
    with pytest.raises(ValueError, match="clipp|saturat"):
        study.run_worker(fake_worker.config)
        study.verify_worker(fake_worker.config.output, fake_worker.manifest, "anchors", 18)
