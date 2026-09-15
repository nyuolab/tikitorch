"""Offline contracts for the declared intermediate-scale CPU training study.

No study, AWS call, native training or frozen-source modification is performed.
"""

import importlib
import json
import math
import random
import signal
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def scales():
    return importlib.import_module("manuscript.tools.train_fpga_lm_scales")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(8501)
    raw = bytes(rng.randrange(128) for _ in range(190_000))
    path = tmp_path / "corpus.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 1024, 56)
    return raw, dataset


def scalar_controls(raw, dataset, steps):
    """Independent within-window pairs with explicit alpha1 target counting."""
    unigram = [1] * 128
    bigram = [[1] * 128 for _ in range(128)]
    start = dataset["splits"]["train"]["start"]
    for index in range(steps):
        offset = start + index * 129
        for position in range(128):
            token, target = raw[offset + position:offset + position + 2]
            unigram[target] += 1
            bigram[token][target] += 1
    total = sum(unigram)
    totals = [sum(row) for row in bigram]
    result = {"unigram": [], "bigram": [], "unigram_counts": unigram, "bigram_counts": bigram}
    start = dataset["splits"]["validation"]["start"]
    for index in range(40, 56):
        offset = start + index * 129
        unigram_nll, bigram_nll = 0.0, 0.0
        for position in range(128):
            token, target = raw[offset + position:offset + position + 2]
            unigram_nll -= math.log(unigram[target] / total)
            bigram_nll -= math.log(bigram[token][target] / totals[token])
        result["unigram"].append(unigram_nll / 128)
        result["bigram"].append(bigram_nll / 128)
    return result


@pytest.mark.parametrize("steps,pairs", [(256, 32_768), (1024, 131_072)])
def test_controls_match_the_exact_training_budget_on_new_validation_cohort(scales, corpus, steps, pairs):
    raw, dataset = corpus
    result = scales.matched_baselines(raw, dataset, range(steps), range(40, 56), 128)
    expected = scalar_controls(raw, dataset, steps)
    assert result["training_pairs"] == pairs
    assert result["alpha"] == 1 and result["vocabulary"] == 128
    assert result["train_indices"] == list(range(steps))
    assert result["validation_indices"] == list(range(40, 56))
    assert result["unigram_target_counts"] == expected["unigram_counts"]
    assert result["bigram_pair_counts"] == expected["bigram_counts"]
    assert sum(result["unigram_target_counts"]) == pairs + 128
    assert sum(map(sum, result["bigram_pair_counts"])) == pairs + 128 * 128
    for name in ("unigram", "bigram"):
        assert result[name]["nll"] == pytest.approx(sum(expected[name]) / 16, abs=1e-12)
        assert [row["window_index"] for row in result[name]["per_window"]] == list(range(40, 56))
        assert [row["nll"] for row in result[name]["per_window"]] == pytest.approx(expected[name], abs=1e-12)


@pytest.mark.parametrize("steps", [256, 1024])
def test_prior_cohorts_unused_training_and_test_cannot_leak_into_comparison(scales, corpus, steps):
    raw, dataset = corpus
    modified = bytearray(raw)
    train = dataset["splits"]["train"]
    unused_start = train["start"] + steps * 129
    modified[unused_start:train["stop"]] = b"!" * (train["stop"] - unused_start)
    validation_start = dataset["splits"]["validation"]["start"]
    modified[validation_start:validation_start + 40 * 129] = b" " * (40 * 129)
    test_start = dataset["splits"]["test"]["start"]
    modified[test_start:] = b"?" * (len(modified) - test_start)
    arguments = (dataset, range(steps), range(40, 56), 128)
    assert scales.matched_baselines(bytes(modified), *arguments) == scales.matched_baselines(raw, *arguments)


def test_update256_baseline_cannot_use_data_seen_only_by_update1024(scales, corpus):
    raw, dataset = corpus
    modified = bytearray(raw)
    start = dataset["splits"]["train"]["start"]
    modified[start + 256 * 129:start + 1024 * 129] = b"!" * ((1024 - 256) * 129)
    arguments = (dataset, range(256), range(40, 56), 128)
    assert scales.matched_baselines(bytes(modified), *arguments) == scales.matched_baselines(raw, *arguments)
    arguments = (dataset, range(1024), range(40, 56), 128)
    assert scales.matched_baselines(bytes(modified), *arguments)["bigram_pair_counts"] != \
        scales.matched_baselines(raw, *arguments)["bigram_pair_counts"]


def test_protocol_separates_new_training_from_unchanged_endpoint_controls(scales):
    assert scales.TRAIN_ARMS == ("anchors_shift1", "anchors_shift2")
    assert scales.CONTROL_ARMS == ("anchors", "anchors_fine")
    assert scales.ARMS == scales.TRAIN_ARMS + scales.CONTROL_ARMS
    assert scales.ARM_SHIFTS == {"anchors_shift1": 1, "anchors_shift2": 2, "anchors": 4, "anchors_fine": 0}
    recipe = scales.recipe()
    assert recipe["arms"] == list(scales.ARMS)
    assert recipe["seeds"] == [0, 1, 2]
    assert recipe["steps"] == 1024
    assert recipe["training_indices"] == list(range(1024))
    assert recipe["validation_indices"] == list(range(40, 56))
    assert recipe["validation_steps"] == [0, 256, 1024]
    assert recipe["baseline_budgets"] == {"256": 32_768, "1024": 131_072}
    assert recipe["initial_baseline_comparison"] is False
    assert recipe["worker_timeout_s"] == 900
    assert [recipe[name] for name in ("width", "layers", "vocabulary", "sequence", "batch")] == [512, 4, 128, 128, 1]
    assert recipe["lr"] == 1 and recipe["threshold"] == 2048
    assert recipe["heldout_test_evaluated"] is False


def test_source_binding_retains_the_frozen_native_and_budget_dependencies(scales):
    from manuscript.tools import extend_fpga_lm_budget as budget

    identity = scales.source_identity()
    for name, digest in budget.source_identity().items():
        assert identity[name] == digest
    source = Path(scales.__file__).resolve()
    assert identity[source.relative_to(pilot.REPO).as_posix()] == pilot.digest(source.read_bytes())


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired, KeyboardInterrupt, RuntimeError])
def test_parent_worker_timeout_or_interruption_reaps_only_its_owned_group(scales, tmp_path, monkeypatch, failure):
    class Process:
        pid, returncode = 23431, -9

        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                if failure is subprocess.TimeoutExpired:
                    raise failure(["offline-unit-worker"], timeout)
                raise failure("offline unit interruption")
            return self.returncode

    process, kills, launches = Process(), [], []

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return process

    monkeypatch.setattr(scales.subprocess, "Popen", launch)
    monkeypatch.setattr(scales.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    command = ["python", "file with spaces.py", "--worker"]
    if failure is subprocess.TimeoutExpired:
        assert scales._run_process(command, tmp_path / "worker.log") == {"returncode": -9, "timed_out": True}
    else:
        with pytest.raises(failure):
            scales._run_process(command, tmp_path / "worker.log")
    assert kills == [(process.pid, signal.SIGKILL)]
    assert process.waits == [900, 10]
    assert launches[0][0] == command and launches[0][1]["start_new_session"] is True
    assert launches[0][1].get("shell", False) is False


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_new_loss_scales_change_only_cfg_not_any_initial_weight_byte(scales, seed):
    from manuscript.tools import screen_fpga_lm_heads as screen

    shape = pilot.layout(512)
    original = screen.initialize_buffer(shape, seed, "anchors")
    for arm, shift in {"anchors_shift1": 1, "anchors_shift2": 2, "anchors": 4, "anchors_fine": 0}.items():
        expected = original.copy()
        expected[shape.OFF_CFG + 1] = shift
        assert scales.initialize_buffer(shape, seed, arm) == expected


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired, KeyboardInterrupt, RuntimeError])
def test_control_checkpoint_child_inherits_outer_group_and_never_kills_parent_group(
        scales, tmp_path, monkeypatch, failure):
    class Process:
        pid, returncode = 23821, -9

        def __init__(self):
            self.waits, self.kills = [], 0

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                if failure is subprocess.TimeoutExpired:
                    raise failure(["offline-checkpoint-child"], timeout)
                raise failure("checkpoint child interruption")
            return self.returncode

        def kill(self):
            self.kills += 1

    process, launches = Process(), []

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return process

    def forbidden_group_kill(*args):
        raise AssertionError("checkpoint child must not kill its parent's inherited group")

    monkeypatch.setattr(scales.subprocess, "Popen", launch)
    monkeypatch.setattr(scales.os, "killpg", forbidden_group_kill)
    if failure is subprocess.TimeoutExpired:
        assert scales._run_checkpoint_process(["offline-checkpoint-child"], tmp_path / "child.log") == \
            {"returncode": -9, "timed_out": True}
    else:
        with pytest.raises(failure):
            scales._run_checkpoint_process(["offline-checkpoint-child"], tmp_path / "child.log")
    assert process.kills == 1
    assert process.waits == [240, 10]
    assert launches[0][1].get("start_new_session", False) is False
    assert launches[0][1].get("shell", False) is False


@pytest.fixture
def historical_checkpoint(scales, tmp_path, monkeypatch):
    """The narrow checkpoint reader must use actual complete immutable bytes."""
    monkeypatch.setattr(scales, "WIDTH", 16)
    shape = pilot.layout(16)
    root = tmp_path / "old-budget"
    root.mkdir()
    directory = root / "anchors_seed_0"
    directory.mkdir()
    initial = shape.build_init(0)
    weights = {name: initial[getattr(shape, "OFF_" + name):
                             getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
               for name in pilot.WEIGHTS}
    checkpoints = {}
    for step in (0, 256, 1024):
        value = deepcopy(weights)
        value["W_HEAD"][0] += step
        checkpoints[str(step)] = pilot.checkpoint(directory, step, value, shape)
    binding = {"run_directory": str(root), "audit_record": {"file": "unit-test.json", "sha256": "a" * 64},
               "manifest_sha256": "b" * 64, "artifact_manifest_sha256": "c" * 64,
               "workers": {"anchors_seed_0": {"manifest_sha256": "d" * 64,
                                               "artifact_manifest_sha256": "e" * 64,
                                               "checkpoints": checkpoints}}}
    return SimpleNamespace(root=root, directory=directory, binding=binding, checkpoints=checkpoints)


@pytest.mark.parametrize("step", [0, 256, 1024])
def test_control_reader_returns_actual_full_checkpoint_at_each_original_budget(scales, historical_checkpoint, step):
    fixture = historical_checkpoint
    result = scales.read_budget_checkpoint(fixture.binding, "anchors", 0, step)
    assert result == (fixture.directory / ("checkpoint_%06d.i32" % step)).read_bytes()
    assert pilot.digest(result) == fixture.checkpoints[str(step)]["sha256"]


@pytest.mark.parametrize("failure", ["bytes", "truncated", "redirected", "symlink"])
def test_forged_control_metadata_cannot_replace_actual_complete_checkpoint_bytes(scales, historical_checkpoint, failure):
    fixture = historical_checkpoint
    record = fixture.binding["workers"]["anchors_seed_0"]["checkpoints"]["256"]
    path = fixture.directory / "checkpoint_000256.i32"
    if failure == "bytes":
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif failure == "truncated":
        path.write_bytes(path.read_bytes()[:-4])
        record["bytes"] = path.stat().st_size
        record["sha256"] = pilot.digest(path.read_bytes())
    elif failure == "redirected":
        record["file"] = "../anchors_fine_seed_0/checkpoint_000256.i32"
    else:
        original = path.with_name("original.i32")
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(ValueError):
        scales.read_budget_checkpoint(fixture.binding, "anchors", 0, 256)


@pytest.fixture
def fake_worker(scales, tmp_path, monkeypatch):
    """Exercise storage/control flow with a tiny ABI and no native execution."""
    monkeypatch.setattr(scales, "WIDTH", 16)
    monkeypatch.setattr(scales, "TRAIN_STEPS", 4)
    monkeypatch.setattr(scales, "EVAL_STEPS", (0, 2, 4))
    monkeypatch.setattr(scales, "VALIDATION_INDICES", (0,))
    monkeypatch.setattr(scales, "source_identity", lambda: {"frozen-scales.py": "a" * 64})

    def initialize(shape, seed, arm):
        result = shape.build_init(seed)
        result[shape.OFF_CFG + 1] = scales.ARM_SHIFTS[arm]
        return result

    monkeypatch.setattr(scales, "initialize_buffer", initialize)
    corpus_path = tmp_path / "tiny-corpus.txt"
    corpus_path.write_bytes(bytes(range(128)) * 100)
    library = tmp_path / "fake.so"
    library.write_bytes(b"fake offline transport library, never loaded")
    budget_run = tmp_path / "historical-budget"
    budget_run.mkdir()
    v1_run = tmp_path / "historical-v1"
    v1_run.mkdir()
    (budget_run / "manifest.json").write_text(json.dumps({"v1_evidence": {"run_directory": str(v1_run)}}))
    audit = tmp_path / "historical-audit.json"
    audit.write_text('{"test_fixture":true}')
    shape = pilot.layout(16)
    payloads, workers = {}, {}
    for arm in ("anchors", "anchors_fine"):
        for seed in (0, 1, 2):
            initial = shape.build_init(seed)
            original = {name: initial[getattr(shape, "OFF_" + name):
                                      getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                        for name in pilot.WEIGHTS}
            checkpoints = {}
            for step in (0, 2, 4):
                weights = deepcopy(original)
                weights["W_HEAD"][0] -= step
                payload = pilot.weight_bytes(weights)
                payloads[(arm, seed, step)] = payload
                checkpoints[str(step)] = {"file": "checkpoint_%06d.i32" % step, "step": step,
                                           "bytes": len(payload), "sha256": pilot.digest(payload)}
            workers[arm + "_seed_%d" % seed] = {"manifest_sha256": "d" * 64,
                                               "artifact_manifest_sha256": "e" * 64,
                                               "checkpoints": checkpoints}
    binding = {"run_directory": str(budget_run),
               "audit_record": {"file": str(audit), "sha256": pilot.digest(audit.read_bytes())},
               "manifest_sha256": "b" * 64, "artifact_manifest_sha256": "c" * 64,
               "workers": workers}
    monkeypatch.setattr(scales, "budget_binding", lambda *args: deepcopy(binding))
    def historical_payload(binding, arm, seed, step):
        if arm in ("anchors_shift1", "anchors_shift2"):
            assert step == 0, "new trained weights must not be matched to old endpoints"
            arm = "anchors"
        return payloads[(arm, seed, step)]

    monkeypatch.setattr(scales, "read_budget_checkpoint", historical_payload)
    forward = {"predictions": [0] * shape.ML_L_SEQ, "gradients": [0] * shape.WORDS_GRAD_LOGIT,
               "lut_normalized_nll": 4.0, "accuracy": 0.0, "saturation_events": 0,
               "rounded_zero_target_mass": 0}
    route_arrays = {name: [0] * getattr(shape, "WORDS_" + name)
                    for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}
    routes = {"positions": 128, "nonzero_logit_gradient_positions": 128,
              "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
              "head_winners_ge16_positions": 128, "single_shared_head_winner_positions": 0,
              "nonzero_embedding_input_vjp_positions": 128,
              "max_abs_head_input_vjp": 1, "max_abs_embedding_input_vjp": 1,
              "backward_int32_overflow_detected": False, "head_winner_counts": [16] * 128}
    monkeypatch.setattr(pilot, "integer_forward", lambda *args, **kwargs: deepcopy(forward))

    class Bridge:
        def __init__(self):
            self.memory = [0] * shape.ML_BUFFER_WORDS
            self.calls = []
            self.closed = False
            self.close_error = None
            self.mismatch = False

        def write(self, offset, values):
            self.memory[offset:offset + len(values)] = values

        def read(self, offset, count):
            return self.memory[offset:offset + count]

        def step(self, lr, threshold):
            self.calls.append((lr, threshold, self.read(shape.OFF_TOKENS, shape.ML_L_SEQ)))
            self.write(shape.OFF_PRED, [int(self.mismatch)] * shape.ML_L_SEQ)
            self.write(shape.OFF_GRAD_LOGIT, forward["gradients"])
            for name, values in route_arrays.items():
                self.write(getattr(shape, "OFF_" + name), values)
            # Deliberately differ from earlier controls at trained checkpoints.
            # Only initialization, not new-scale updates, must match them.
            if lr:
                self.memory[shape.OFF_W_HEAD] += 1

        def close(self):
            self.closed = True
            if self.close_error:
                raise self.close_error

    bridge = Bridge()

    def factory(path, backend, layout):
        assert path == library and backend == "csim" and layout.ML_D == 16
        return bridge

    def observer(*args, **kwargs):
        return {**deepcopy(forward), "native_routes": deepcopy(route_arrays), "route_diagnostics": deepcopy(routes)}

    config = SimpleNamespace(corpus=corpus_path, output=tmp_path / "worker", budget_run=budget_run,
                             budget_audit=audit, library=library, arm="anchors_shift1", seed=0)
    raw, _, expected = scales._common_manifest(corpus_path, budget_run, audit)
    return SimpleNamespace(config=config, bridge=bridge, factory=factory, observer=observer, shape=shape,
                           raw=raw, expected=expected, historical_payloads=payloads,
                           library_hash=pilot.digest(library.read_bytes()), v1_run=v1_run)


def execute_worker(scales, fixture):
    return scales.run_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)


def verify_worker(scales, fixture):
    return scales.verify_worker(fixture.config.output, fixture.expected, fixture.config.arm, 0, fixture.library_hash)


def test_new_scale_trains_fresh_data_and_does_not_require_old_trained_weight_equality(scales, fake_worker):
    fixture = fake_worker
    result = execute_worker(scales, fixture)
    assert result["complete"] is True and fixture.bridge.closed
    assert result["training_performed"] is True and result["training_updates_performed"] == 4
    assert result["steps"] == 4 and result["training_pairs"] == 512
    assert [result[key]["step"] for key in ("initial", "intermediate", "final")] == [0, 2, 4]
    assert set(result["baselines_by_step"]) == {"2", "4"}
    assert result["baselines_by_step"]["2"]["training_pairs"] == 256
    assert result["baselines_by_step"]["4"]["training_pairs"] == 512
    assert result["initial"]["baseline_training_pairs"] is None
    assert result["initial"]["per_window"][0]["baseline_minus_model_nll"] == {}
    assert [call[:2] for call in fixture.bridge.calls] == [
        (0, 2048), (1, 2048), (1, 2048), (0, 2048), (1, 2048), (1, 2048), (0, 2048)]
    trained_tokens = [tokens for lr, _, tokens in fixture.bridge.calls if lr]
    assert trained_tokens == [pilot.window(fixture.raw, fixture.expected["dataset"], "train", index, 128)[0]
                              for index in range(4)]
    output = fixture.config.output
    assert (output / "checkpoint_000000.i32").read_bytes() == fixture.historical_payloads[("anchors", 0, 0)]
    assert (output / "checkpoint_000002.i32").read_bytes() != fixture.historical_payloads[("anchors", 0, 2)]
    assert (output / "checkpoint_000004.i32").read_bytes() != fixture.historical_payloads[("anchors", 0, 4)]
    assert json.loads((output / "manifest.json").read_text())["zshift"] == 1
    assert verify_worker(scales, fixture) == result


def prepare_control_manifest(scales, fixture, arm="anchors", step=0):
    output = fixture.config.output
    output.mkdir()
    fixture.config.arm = arm
    fixture.config.checkpoint_step = step
    fixture.config.parent_manifest = output / "manifest.json"
    initial = scales.initialize_buffer(fixture.shape, 0, arm)
    manifest = dict(fixture.expected, arm=arm, seed=0, zshift=scales.ARM_SHIFTS[arm],
                    training_performed=False, training_updates_performed=0,
                    library_path=str(fixture.config.library), library_sha256=fixture.library_hash,
                    initial_buffer_sha256=pilot.digest(scales.screen._i32(initial)))
    pilot.write_json(output / "manifest.json", manifest)
    return manifest


@pytest.mark.parametrize("arm", ["anchors", "anchors_fine"])
@pytest.mark.parametrize("step", [0, 2, 4])
def test_control_checkpoint_only_evaluates_exact_old_weights_at_original_shift(scales, fake_worker, arm, step):
    fixture = fake_worker
    prepare_control_manifest(scales, fixture, arm, step)
    receipt = scales.run_checkpoint_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)
    assert receipt["complete"] is True and fixture.bridge.closed
    assert receipt["step"] == step
    assert receipt["training_updates_performed"] == 0 and receipt["native_learning_rate"] == 0
    assert [call[:2] for call in fixture.bridge.calls] == [(0, 2048)]
    assert fixture.bridge.read(fixture.shape.OFF_CFG + 1, 1) == [scales.ARM_SHIFTS[arm]]
    output = fixture.config.output
    expected_bytes = fixture.historical_payloads[(arm, 0, step)]
    assert (output / ("checkpoint_%06d.i32" % step)).read_bytes() == expected_bytes
    assert receipt["historical_checkpoint_sha256"] == pilot.digest(expected_bytes)
    evaluation = json.loads((output / ("evaluation_%06d.json" % step)).read_text())
    assert evaluation["historical_checkpoint_equal"] is True
    assert evaluation["baseline_training_pairs"] == (step * 128 if step else None)
    if not step:
        assert evaluation["per_window"][0]["baseline_minus_model_nll"] == {}
        assert not (output / "baselines_000000.json").exists()
    else:
        controls = json.loads((output / ("baselines_%06d.json" % step)).read_text())
        assert controls["training_pairs"] == step * 128
    assert not (output / "training.jsonl").exists()
    assert not (output / "training_predictions.i32").exists()
    assert not (output / "training_gradients.i32").exists()


def test_control_checkpoint_cannot_mutate_weights_or_report_success(scales, fake_worker, monkeypatch):
    fixture = fake_worker
    prepare_control_manifest(scales, fixture, step=2)
    original_step = fixture.bridge.step

    def corrupt_frozen_evaluation(lr, threshold):
        original_step(lr, threshold)
        fixture.bridge.memory[fixture.shape.OFF_W_HEAD] += 1

    monkeypatch.setattr(fixture.bridge, "step", corrupt_frozen_evaluation)
    with pytest.raises(RuntimeError, match="changed full weights"):
        scales.run_checkpoint_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)
    job = fixture.config.output / "checkpoint_jobs/step_000002"
    assert json.loads((job / "failure.json").read_text())["complete"] is False
    assert not (job / "summary.json").exists()
    assert fixture.bridge.closed


def test_control_checkpoint_refuses_a_training_arm_before_native_access(scales, fake_worker):
    fixture = fake_worker
    prepare_control_manifest(scales, fixture, step=2)
    fixture.config.arm = "anchors_shift1"
    with pytest.raises(ValueError, match="control manifest"):
        scales.run_checkpoint_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)
    assert not fixture.bridge.calls
    assert not (fixture.config.output / "checkpoint_jobs").exists()


def test_new_training_requires_actual_historical_initial_weights_before_native_access(scales, fake_worker, monkeypatch):
    fixture = fake_worker
    payload = bytearray(fixture.historical_payloads[("anchors", 0, 0)])
    payload[0] ^= 1
    monkeypatch.setattr(scales, "read_budget_checkpoint", lambda *args: bytes(payload))
    with pytest.raises(ValueError, match="initial weights differ"):
        execute_worker(scales, fixture)
    assert not fixture.bridge.calls
    assert (fixture.config.output / "failure.json").is_file()
    assert not (fixture.config.output / "summary.json").exists()


@pytest.mark.parametrize("arm", ["anchors_shift1", "anchors_shift2"])
def test_historical_reader_refuses_to_substitute_old_trained_weights_for_new_scales(scales, historical_checkpoint, arm):
    with pytest.raises(ValueError, match="initial new-arm"):
        scales.read_budget_checkpoint(historical_checkpoint.binding, arm, 0, 256)


@pytest.mark.parametrize("failure", ["native", "close", "interrupt", "source", "history"])
def test_new_training_failure_retains_evidence_without_success(scales, fake_worker, monkeypatch, failure):
    fixture = fake_worker
    expected_error = RuntimeError
    if failure == "native":
        fixture.bridge.mismatch = True
    elif failure == "close":
        fixture.bridge.close_error = RuntimeError("injected close failure")
    elif failure == "interrupt":
        def interrupt(*args):
            raise KeyboardInterrupt("injected cancellation")

        monkeypatch.setattr(fixture.bridge, "step", interrupt)
        expected_error = KeyboardInterrupt
    else:
        original_close = fixture.bridge.close

        def close_and_mutate_identity():
            original_close()
            if failure == "source":
                monkeypatch.setattr(scales, "source_identity", lambda: {"changed.py": "c" * 64})
            else:
                changed = dict(fixture.expected["budget_evidence"], artifact_manifest_sha256="f" * 64)
                monkeypatch.setattr(scales, "budget_binding", lambda *args: changed)

        monkeypatch.setattr(fixture.bridge, "close", close_and_mutate_identity)
    with pytest.raises(expected_error):
        execute_worker(scales, fixture)
    output = fixture.config.output
    assert json.loads((output / "failure.json").read_text())["complete"] is False
    assert fixture.bridge.closed
    assert (output / "manifest.json").is_file()
    assert not (output / "summary.json").exists()
    if failure == "native":
        assert (output / "eval_000000_window_000_predictions.i32").is_file()


def test_control_child_requires_the_exact_owned_parent_manifest(scales, fake_worker, tmp_path):
    fixture = fake_worker
    prepare_control_manifest(scales, fixture)
    alternate = tmp_path / "different-parent.json"
    alternate.write_bytes(fixture.config.parent_manifest.read_bytes())
    fixture.config.parent_manifest = alternate
    with pytest.raises(ValueError, match="owned control manifest"):
        scales.run_checkpoint_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)
    assert not fixture.bridge.calls
    assert not (fixture.config.output / "checkpoint_jobs").exists()


@pytest.mark.parametrize("entrypoint", ["worker", "study"])
@pytest.mark.parametrize("historical", ["budget", "v1"])
def test_new_outputs_cannot_be_nested_in_either_immutable_history(scales, fake_worker, entrypoint, historical):
    fixture = fake_worker
    root = fixture.config.budget_run if historical == "budget" else fixture.v1_run
    before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    fixture.config.output = root / "forbidden-new-result"
    with pytest.raises(ValueError, match="immutable historical run"):
        if entrypoint == "worker":
            execute_worker(scales, fixture)
        else:
            scales.run_study(fixture.config)
    assert not fixture.config.output.exists()
    assert {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert not fixture.bridge.calls


def test_history_guard_resolves_symlinked_parent_before_creating_output(scales, fake_worker, tmp_path):
    alias = tmp_path / "history-alias"
    alias.symlink_to(fake_worker.v1_run, target_is_directory=True)
    fake_worker.config.output = alias / "new-result"
    with pytest.raises(ValueError, match="immutable historical run"):
        execute_worker(scales, fake_worker)
    assert not (fake_worker.v1_run / "new-result").exists()


def execute_control_cell(scales, fixture):
    fixture.config.arm = "anchors"
    children, commands = [], []

    def child_process(command, log_path):
        commands.append(command)
        with log_path.open("x") as stream:
            stream.write("injected checkpoint process\n")
        config = SimpleNamespace(**vars(fixture.config))
        config.checkpoint_step = int(command[command.index("--checkpoint-step") + 1])
        config.parent_manifest = Path(command[command.index("--parent-manifest") + 1])
        bridge = type(fixture.bridge)()
        children.append(bridge)
        scales.run_checkpoint_worker(config, bridge_factory=lambda *args: bridge, observer=fixture.observer)
        return {"returncode": 0, "timed_out": False}

    def forbidden_parent_transport(*args):
        raise AssertionError("control parent must not reuse a resident native transport across checkpoints")

    result = scales.run_worker(fixture.config, bridge_factory=forbidden_parent_transport,
                               observer=fixture.observer, checkpoint_runner=child_process)
    return result, children, commands


def test_control_parent_re_evaluates_three_exact_checkpoints_without_retraining(scales, fake_worker):
    fixture = fake_worker
    result, children, commands = execute_control_cell(scales, fixture)
    assert result["complete"] is True
    assert result["training_performed"] is False and result["training_updates_performed"] == 0
    assert result["steps"] == 4 and result["training_pairs"] == 512
    assert len(children) == len(commands) == 3
    assert [int(command[command.index("--checkpoint-step") + 1]) for command in commands] == [0, 2, 4]
    assert all("--checkpoint-worker" in command and "--parent-manifest" in command for command in commands)
    assert all(child.closed and [call[:2] for call in child.calls] == [(0, 2048)] for child in children)
    assert not fixture.bridge.calls
    assert verify_worker(scales, fixture) == result


@pytest.mark.parametrize("failed_index", [0, 5, 6, 11])
def test_any_incomplete_training_or_control_cell_blocks_all_gates_after_exactly_twelve_attempts(
        scales, fake_worker, monkeypatch, failed_index):
    fixture = fake_worker
    commands, gates = [], []
    monkeypatch.setattr(pilot, "build_csim", lambda output, width: fixture.config.library)

    def process(command, log_path):
        index = len(commands)
        commands.append(command)
        log_path.write_text("injected unit process\n")
        return {"returncode": 1 if index == failed_index else 0, "timed_out": False}

    def verified(directory, expected, arm, seed, library_hash):
        assert expected["source_hashes"] == fixture.expected["source_hashes"]
        return {"arm": arm, "seed": seed}

    monkeypatch.setattr(scales, "_run_process", process)
    monkeypatch.setattr(scales, "verify_worker", verified)
    monkeypatch.setattr(scales, "_compact_worker", lambda row: row)
    monkeypatch.setattr(scales, "gate_arm", lambda values: gates.append(values))
    result = scales.run_study(fixture.config)
    assert result["complete"] is False
    assert result["declared_workers"] == len(commands) == 12
    assert result["declared_training_workers"] == result["declared_control_workers"] == 6
    assert result["completed_workers"] == 11
    assert not gates
    assert all(row["larger_paid_learning_benchmark_gate"] is False for row in result["gates"].values())
    expected_cells = [(arm, str(seed)) for arm in ("anchors_shift1", "anchors_shift2", "anchors", "anchors_fine")
                      for seed in (0, 1, 2)]
    assert [(command[command.index("--arm") + 1], command[command.index("--seed") + 1])
            for command in commands] == expected_cells
    assert all("--worker" in command and "--parent-manifest" in command and "--budget-audit" in command
               for command in commands)
    arm, seed = expected_cells[failed_index]
    failed_name = arm + "_seed_" + seed
    output = fixture.config.output
    assert (output / (failed_name + "_failure.json")).is_file()
    assert json.loads((output / failed_name / "failure.json").read_text())["complete"] is False
    assert (output / "failure.json").is_file()
    assert (output / "artifact_manifest.json").is_file()


def rehash_result(scales, output):
    record = scales._artifact_manifest(output)
    record["files"] = [row for row in record["files"] if row["file"] != "artifact_manifest.json"]
    (output / "artifact_manifest.json").write_text(json.dumps(record))


@pytest.mark.parametrize("failure", ["aggregate", "budget", "source", "native", "repeat", "role"])
def test_rehashed_training_results_cannot_forge_scientific_or_execution_identity(scales, fake_worker, failure):
    fixture = fake_worker
    execute_worker(scales, fixture)
    output = fixture.config.output
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    if failure == "aggregate":
        summary["final"]["lut_normalized_nll"] = 0.01
        (output / "evaluation_000004.json").write_text(json.dumps(summary["final"]))
    elif failure == "budget":
        summary["baselines_by_step"]["2"] = deepcopy(summary["baselines_by_step"]["4"])
        (output / "baselines_000002.json").write_text(json.dumps(summary["baselines_by_step"]["2"]))
    elif failure == "source":
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_hashes"] = {"different-source.py": "c" * 64}
        path.write_text(json.dumps(manifest))
    elif failure == "native":
        path = output / "training_gradients.i32"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif failure == "repeat":
        path = output / "training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["window_index"] = 0
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        for record in (manifest, summary):
            record["training_performed"] = False
            record["training_updates_performed"] = 0
        path.write_text(json.dumps(manifest))
    summary_path.write_text(json.dumps(summary))
    rehash_result(scales, output)
    with pytest.raises(ValueError):
        verify_worker(scales, fixture)


@pytest.mark.parametrize("failure", ["training_stream", "child_failure", "child_source", "child_update", "checkpoint"])
def test_rehashed_control_records_must_remain_readonly_and_match_actual_history(scales, fake_worker, failure):
    fixture = fake_worker
    execute_control_cell(scales, fixture)
    output = fixture.config.output
    job = output / "checkpoint_jobs/step_000002"
    if failure == "training_stream":
        (output / "training.jsonl").write_text("")
    elif failure == "child_failure":
        (job / "failure.json").write_text('{"complete":false}')
    elif failure == "child_source":
        path = job / "manifest.json"
        record = json.loads(path.read_text())
        record["source_hashes"] = {"changed.py": "f" * 64}
        path.write_text(json.dumps(record))
    elif failure == "child_update":
        path = job / "summary.json"
        record = json.loads(path.read_text())
        record["training_updates_performed"] = 1
        path.write_text(json.dumps(record))
    else:
        path = output / "checkpoint_000002.i32"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text())
        checkpoint = summary["intermediate"]["checkpoint"]
        checkpoint["sha256"] = pilot.digest(bytes(payload))
        (output / "checkpoint_000002.json").write_text(json.dumps(checkpoint))
        (output / "evaluation_000002.json").write_text(json.dumps(summary["intermediate"]))
        summary_path.write_text(json.dumps(summary))
    rehash_result(scales, output)
    with pytest.raises(ValueError):
        verify_worker(scales, fixture)


@pytest.mark.parametrize("entrypoint", ["worker", "study"])
def test_existing_result_directory_is_never_overwritten(scales, fake_worker, entrypoint):
    output = fake_worker.config.output
    output.mkdir()
    sentinel = output / "historical.json"
    sentinel.write_text('{"keep":true}')
    with pytest.raises(FileExistsError):
        if entrypoint == "worker":
            execute_worker(scales, fake_worker)
        else:
            scales.run_study(fake_worker.config)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == '{"keep":true}'
    assert not fake_worker.bridge.calls


@pytest.mark.parametrize("file_kind,field,value", [
    ("summary", "complete", 1),
    ("summary", "step", False),
    ("summary", "training_updates_performed", False),
    ("summary", "native_learning_rate", False),
    ("summary", "native_validation_windows", True),
    ("manifest", "seed", False),
    ("manifest", "step", False),
    ("manifest", "native_training_updates", False),
    ("execution", "timeout_s", 240.0),
])
def test_control_child_metadata_uses_literal_integer_and_boolean_types(scales, fake_worker, file_kind, field, value):
    fixture = fake_worker
    execute_control_cell(scales, fixture)
    output = fixture.config.output
    path = (output / "checkpoint_000000_execution.json" if file_kind == "execution"
            else output / "checkpoint_jobs/step_000000" / (file_kind + ".json"))
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    rehash_result(scales, output)
    with pytest.raises(ValueError):
        verify_worker(scales, fixture)


@pytest.mark.parametrize("field", ["seed", "checkpoint_step"])
def test_control_child_rejects_boolean_zero_identifier_before_native_access(scales, fake_worker, field):
    fixture = fake_worker
    prepare_control_manifest(scales, fixture)
    setattr(fixture.config, field, False)
    with pytest.raises(ValueError):
        scales.run_checkpoint_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)
    assert not fixture.bridge.calls
