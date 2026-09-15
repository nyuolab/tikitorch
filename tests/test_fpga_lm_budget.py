"""Independent offline contracts for the fixed 1,024-update CPU extension.

No production experiment, native library, PCI device or AWS resource is used.
"""

import importlib
import json
import math
import random
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def extension():
    return importlib.import_module("manuscript.tools.extend_fpga_lm_budget")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(991)
    raw = bytes(rng.randrange(128) for _ in range(190_000))
    path = tmp_path / "corpus.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 1024, 40)
    return raw, dataset


def independent_controls(raw, dataset, steps):
    """Fit explicit within-window targets/pairs, never an adjacent flat slice."""
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
    row_totals = [sum(row) for row in bigram]
    result = {"unigram": [], "bigram": [], "unigram_counts": unigram, "bigram_counts": bigram}
    validation_start = dataset["splits"]["validation"]["start"]
    for index in range(24, 40):
        offset = validation_start + index * 129
        unigram_nll, bigram_nll = 0.0, 0.0
        for position in range(128):
            token, target = raw[offset + position:offset + position + 2]
            unigram_nll -= math.log(unigram[target] / total)
            bigram_nll -= math.log(bigram[token][target] / row_totals[token])
        result["unigram"].append(unigram_nll / 128)
        result["bigram"].append(bigram_nll / 128)
    return result


def test_declared_protocol_has_two_arms_six_seeds_and_no_new_geometry(extension):
    recipe = extension.recipe()
    assert recipe["arms"] == ["anchors", "anchors_fine"]
    assert recipe["seeds"] == [0, 1, 2]
    assert recipe["steps"] == 1024
    assert recipe["validation_steps"] == [0, 256, 1024]
    assert recipe["validation_indices"] == list(range(24, 40))
    assert recipe["training_indices"] == list(range(1024))
    assert recipe["worker_timeout_s"] == 900
    assert [recipe[name] for name in ("width", "layers", "vocabulary", "sequence", "batch")] == [512, 4, 128, 128, 1]
    assert recipe["lr"] == 1 and recipe["threshold"] == 2048
    assert recipe["heldout_test_evaluated"] is False
    assert recipe["baseline_budgets"] == {"256": 32_768, "1024": 131_072}
    assert recipe["initial_baseline_comparison"] is False
    assert set(recipe["head_initialization"]) == {"anchors", "anchors_fine"}


@pytest.mark.parametrize("steps,pairs", [(256, 32_768), (1024, 131_072)])
def test_both_checkpoint_controls_use_exact_matched_budget(extension, corpus, steps, pairs):
    raw, dataset = corpus
    controls = extension.matched_baselines(raw, dataset, range(steps), range(24, 40), 128)
    expected = independent_controls(raw, dataset, steps)
    assert controls["training_pairs"] == pairs
    assert controls["alpha"] == 1 and controls["vocabulary"] == 128
    assert controls["train_indices"] == list(range(steps))
    assert controls["validation_indices"] == list(range(24, 40))
    assert controls["unigram_target_counts"] == expected["unigram_counts"]
    assert controls["bigram_pair_counts"] == expected["bigram_counts"]
    assert sum(controls["unigram_target_counts"]) == pairs + 128
    assert sum(map(sum, controls["bigram_pair_counts"])) == pairs + 128 * 128
    for name in ("unigram", "bigram"):
        assert controls[name]["nll"] == pytest.approx(sum(expected[name]) / 16, abs=1e-12)
        assert [row["window_index"] for row in controls[name]["per_window"]] == list(range(24, 40))
        assert [row["nll"] for row in controls[name]["per_window"]] == pytest.approx(expected[name], abs=1e-12)


@pytest.mark.parametrize("steps", [256, 1024])
def test_unused_training_old_validation_and_final_test_cannot_affect_controls(extension, corpus, steps):
    raw, dataset = corpus
    modified = bytearray(raw)
    train = dataset["splits"]["train"]
    unused_start = train["start"] + steps * 129
    modified[unused_start:train["stop"]] = b" " * (train["stop"] - unused_start)
    validation_start = dataset["splits"]["validation"]["start"]
    modified[validation_start:validation_start + 24 * 129] = b"!" * (24 * 129)
    test_start = dataset["splits"]["test"]["start"]
    modified[test_start:] = b"?" * (len(modified) - test_start)
    assert extension.matched_baselines(bytes(modified), dataset, range(steps), range(24, 40), 128) == \
        extension.matched_baselines(raw, dataset, range(steps), range(24, 40), 128)


def test_intermediate_controls_cannot_see_later_training_windows(extension, corpus):
    raw, dataset = corpus
    modified = bytearray(raw)
    offset = dataset["splits"]["train"]["start"]
    modified[offset + 256 * 129:offset + 1024 * 129] = b"!" * ((1024 - 256) * 129)
    arguments = (dataset, range(256), range(24, 40), 128)
    assert extension.matched_baselines(bytes(modified), *arguments) == extension.matched_baselines(raw, *arguments)
    arguments = (dataset, range(1024), range(24, 40), 128)
    assert extension.matched_baselines(bytes(modified), *arguments)["bigram_pair_counts"] != \
        extension.matched_baselines(raw, *arguments)["bigram_pair_counts"]


def test_extension_source_identity_includes_unchanged_native_and_v1_sources(extension):
    identity = extension.source_identity()
    from manuscript.tools import screen_fpga_lm_heads as v1

    for name, digest in v1.source_identity().items():
        assert identity[name] == digest
    harness_path = Path(extension.__file__).resolve()
    assert identity[harness_path.relative_to(pilot.REPO).as_posix()] == pilot.digest(harness_path.read_bytes())


def test_worker_process_uses_bounded_owned_session_and_exclusive_log(extension, tmp_path, monkeypatch):
    launches = []

    class Process:
        def wait(self, timeout):
            assert timeout == 900
            return 0

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        return Process()

    monkeypatch.setattr(extension.subprocess, "Popen", launch)
    log_path = tmp_path / "worker.log"
    command = ["python", "path with spaces.py", "--worker"]
    assert extension._run_process(command, log_path) == {"returncode": 0, "timed_out": False}
    assert launches[0][0] == command
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1].get("shell", False) is False
    with pytest.raises(FileExistsError):
        extension._run_process(command, log_path)
    assert len(launches) == 1


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired, KeyboardInterrupt, RuntimeError])
def test_failed_wait_kills_and_reaps_only_its_owned_worker(extension, tmp_path, monkeypatch, failure):
    class Process:
        pid, returncode = 21231, -9

        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                if failure is subprocess.TimeoutExpired:
                    raise failure(["unit-test-worker"], timeout)
                raise failure("unit test interrupt")
            return self.returncode

    process, kills = Process(), []
    monkeypatch.setattr(extension.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(extension.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    if failure is subprocess.TimeoutExpired:
        assert extension._run_process(["unit-test-worker"], tmp_path / "worker.log") == \
            {"returncode": -9, "timed_out": True}
    else:
        with pytest.raises(failure):
            extension._run_process(["unit-test-worker"], tmp_path / "worker.log")
    assert kills == [(process.pid, extension.signal.SIGKILL)]
    assert process.waits == [900, 10]


def rewrite_json(path, value):
    path.write_text(json.dumps(value))


def rehash_artifacts(extension, directory, schema):
    record = extension.screen._artifact_manifest(directory)
    record["schema"] = schema
    record["files"] = [row for row in record["files"] if row["file"] != "artifact_manifest.json"]
    rewrite_json(directory / "artifact_manifest.json", record)


@pytest.fixture
def historical_v1(extension, corpus, tmp_path, monkeypatch):
    """Exercise the outer v1 binding; frozen worker verification is a stub here.

    Its detailed payload/initialization/coverage contracts are independently
    covered in test_fpga_lm_head_screen.py, not bypassed by production code.
    """
    monkeypatch.setattr(extension, "WIDTH", 16)
    monkeypatch.setattr(extension.screen, "source_identity", lambda: {"frozen-v1.py": "a" * 64})
    _, dataset = corpus
    root = tmp_path / "historical-v1"
    root.mkdir()
    shape = pilot.layout(16)
    manifest = {"schema": extension.screen.SCHEMA, "backend": "csim", "recipe": extension.screen.recipe(),
                "source_hashes": extension.screen.source_identity(), "layout": pilot.layout_values(shape),
                "dataset": dataset}
    summary = {"schema": extension.screen.SCHEMA, "complete": True, "failures": [],
               "declared_workers": 12, "completed_workers": 12, "heldout_test_evaluated": False,
               "hardware_execution": False, "speed_advantage_claim": False,
               "statistical_significance_claim": False}
    rewrite_json(root / "manifest.json", manifest)
    rewrite_json(root / "summary.json", summary)
    build = root / "build"
    build.mkdir()
    library = build / "libmorphlm.so"
    library.write_bytes(b"unit-test library, never loaded")
    library_hash = pilot.digest(library.read_bytes())
    rewrite_json(root / "library.json", {"path": "/historical/build/libmorphlm.so", "sha256": library_hash})
    rewrite_json(build / "build.json", {"library_sha256": library_hash, "source_hashes": pilot.source_identity(),
                                        "layout": manifest["layout"]})
    checkpoint_records, calls = {}, []
    for arm in ("anchors", "anchors_fine"):
        for seed in (0, 1, 2):
            directory = root / (arm + "_seed_%d" % seed)
            directory.mkdir()
            initial = shape.build_init(seed)
            weights = {name: initial[getattr(shape, "OFF_" + name):
                                     getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
                       for name in pilot.WEIGHTS}
            checkpoint = pilot.checkpoint(directory, 256, weights, shape)
            checkpoint_records[(arm, seed)] = checkpoint
            rewrite_json(directory / "manifest.json", {"unit_test": "worker verifier stub boundary"})
            rewrite_json(directory / "artifact_manifest.json", {"unit_test": True})

    def verify_worker(directory, expected, arm, seed, expected_library):
        calls.append((arm, seed))
        assert expected == manifest
        assert directory == root / (arm + "_seed_%d" % seed)
        assert expected_library == library_hash
        return {"final": {"checkpoint": deepcopy(checkpoint_records[(arm, seed)])}}

    monkeypatch.setattr(extension.screen, "verify_worker", verify_worker)
    rehash_artifacts(extension, root, extension.screen.SCHEMA)
    return SimpleNamespace(root=root, dataset=dataset, calls=calls, checkpoints=checkpoint_records,
                           manifest=manifest, summary=summary, library=library)


def test_v1_binding_requires_each_of_the_six_verified_actual_payloads(extension, historical_v1):
    fixture = historical_v1
    binding = extension.v1_binding(fixture.root, fixture.dataset)
    assert fixture.calls == [(arm, seed) for arm in ("anchors", "anchors_fine") for seed in (0, 1, 2)]
    assert binding["run_directory"] == str(fixture.root)
    assert binding["manifest_sha256"] == pilot.digest((fixture.root / "manifest.json").read_bytes())
    assert binding["artifact_manifest_sha256"] == pilot.digest((fixture.root / "artifact_manifest.json").read_bytes())
    assert set(binding["workers"]) == {arm + "_seed_%d" % seed
                                      for arm in ("anchors", "anchors_fine") for seed in (0, 1, 2)}
    for arm, seed in fixture.calls:
        payload = extension.read_v1_checkpoint(binding, arm, seed)
        assert payload == (fixture.root / (arm + "_seed_%d" % seed) / "checkpoint_000256.i32").read_bytes()


@pytest.mark.parametrize("failure", ["incomplete", "source", "corpus", "library", "checkpoint"])
def test_rehashed_historical_files_do_not_bypass_v1_identity_checks(extension, historical_v1, failure):
    fixture = historical_v1
    if failure == "incomplete":
        fixture.summary["completed_workers"] = 11
        rewrite_json(fixture.root / "summary.json", fixture.summary)
    elif failure == "source":
        fixture.manifest["source_hashes"] = {"wrong-source.py": "b" * 64}
        rewrite_json(fixture.root / "manifest.json", fixture.manifest)
    elif failure == "corpus":
        fixture.manifest["dataset"] = dict(fixture.dataset, sha256="c" * 64)
        rewrite_json(fixture.root / "manifest.json", fixture.manifest)
    elif failure == "library":
        fixture.library.write_bytes(b"different library")
    else:
        path = fixture.root / "anchors_seed_0/checkpoint_000256.i32"
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
    rehash_artifacts(extension, fixture.root, extension.screen.SCHEMA)
    with pytest.raises(ValueError):
        extension.v1_binding(fixture.root, fixture.dataset)


@pytest.mark.parametrize("failure", ["uncovered", "failure", "symlink"])
def test_historical_artifacts_must_be_complete_regular_unchanged_files(extension, historical_v1, failure):
    root = historical_v1.root
    if failure == "symlink":
        (root / "foreign.txt").symlink_to(root / "manifest.json")
    else:
        (root / ("failure.json" if failure == "failure" else "new-uncovered.json")).write_text("{}")
    with pytest.raises(ValueError):
        extension.v1_binding(root, historical_v1.dataset)


@pytest.mark.parametrize("failure", ["changed_bytes", "incomplete_size", "redirected_file", "symlink"])
def test_bound_v1_checkpoint_is_reread_as_full_actual_bytes(extension, historical_v1, failure):
    fixture = historical_v1
    binding = extension.v1_binding(fixture.root, fixture.dataset)
    record = binding["workers"]["anchors_seed_0"]["checkpoint_256"]
    path = fixture.root / "anchors_seed_0/checkpoint_000256.i32"
    if failure == "changed_bytes":
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif failure == "incomplete_size":
        path.write_bytes(path.read_bytes()[:-4])
        record["bytes"] = path.stat().st_size
        record["sha256"] = pilot.digest(path.read_bytes())
    elif failure == "redirected_file":
        record["file"] = "../anchors_fine_seed_0/checkpoint_000256.i32"
    else:
        renamed = path.with_name("original.i32")
        path.rename(renamed)
        path.symlink_to(renamed)
    with pytest.raises(ValueError):
        extension.read_v1_checkpoint(binding, "anchors", 0)


@pytest.fixture
def fake_worker(extension, tmp_path, monkeypatch):
    """Tiny injected transport, retaining real full-region ABI/checkpoint sizes."""
    monkeypatch.setattr(extension, "WIDTH", 16)
    monkeypatch.setattr(extension, "TRAIN_STEPS", 4)
    monkeypatch.setattr(extension, "EVAL_STEPS", (0, 2, 4))
    monkeypatch.setattr(extension, "VALIDATION_INDICES", (0,))
    monkeypatch.setattr(extension, "initialize_buffer", lambda shape, seed, arm: shape.build_init(seed))
    monkeypatch.setattr(extension, "source_identity", lambda: {"frozen-extension.py": "a" * 64})
    corpus_path = tmp_path / "worker-corpus.txt"
    corpus_path.write_bytes(bytes(range(128)) * 100)
    library = tmp_path / "fake.so"
    library.write_bytes(b"unit test fake library, never loaded")
    historical = tmp_path / "historical"
    historical.mkdir()
    binding = {"run_directory": str(historical), "schema": extension.screen.SCHEMA,
               "manifest_sha256": "1" * 64, "artifact_manifest_sha256": "2" * 64, "workers": {}}
    monkeypatch.setattr(extension, "v1_binding", lambda *args: deepcopy(binding))
    shape = pilot.layout(16)
    initial = shape.build_init(0)
    weights = {name: initial[getattr(shape, "OFF_" + name):
                             getattr(shape, "OFF_" + name) + getattr(shape, "WORDS_" + name)]
               for name in pilot.WEIGHTS}
    payload = pilot.weight_bytes(weights)
    monkeypatch.setattr(extension, "read_v1_checkpoint", lambda *args: payload)
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

    config = SimpleNamespace(corpus=corpus_path, output=tmp_path / "worker", v1_run=historical,
                             library=library, arm="anchors", seed=0)
    raw, _, expected = extension._common_manifest(corpus_path, historical)
    return SimpleNamespace(config=config, bridge=bridge, factory=factory, observer=observer, shape=shape,
                           initial=initial, weights=weights, raw=raw, expected=expected,
                           library_hash=pilot.digest(library.read_bytes()), v1_payload=payload)


def test_initial_evaluation_is_untrained_control_with_no_fitted_baseline(extension, fake_worker):
    fixture = fake_worker
    fixture.config.output.mkdir()
    fixture.bridge.write(0, fixture.initial)
    manifest = dict(fixture.expected, arm="anchors", seed=0, zshift=4)
    before = pilot.weight_bytes(pilot.read_weights(fixture.bridge, fixture.shape))
    result = extension._evaluate(fixture.config.output, fixture.bridge, fixture.weights, fixture.shape,
                                 fixture.raw, manifest, 0, fixture.observer, None)
    assert result["baseline_training_pairs"] is None
    assert all(row["baseline_minus_model_nll"] == {} for row in result["per_window"])
    assert [(lr, threshold) for lr, threshold, _ in fixture.bridge.calls] == [(0, 2048)]
    assert pilot.weight_bytes(pilot.read_weights(fixture.bridge, fixture.shape)) == before


def test_intermediate_evaluation_requires_actual_v1_bytes_not_claimed_equal_flag(extension, fake_worker, monkeypatch):
    fixture = fake_worker
    fixture.config.output.mkdir()
    fixture.bridge.write(0, fixture.initial)
    manifest = dict(fixture.expected, arm="anchors", seed=0, zshift=4)
    wrong_payload = bytearray(fixture.v1_payload)
    wrong_payload[-1] ^= 1
    monkeypatch.setattr(extension, "read_v1_checkpoint", lambda *args: bytes(wrong_payload))
    controls = extension.matched_baselines(fixture.raw, manifest["dataset"], range(2), [0], 128)
    with pytest.raises(RuntimeError, match="full weights"):
        extension._evaluate(fixture.config.output, fixture.bridge, fixture.weights, fixture.shape,
                            fixture.raw, manifest, 2, fixture.observer, controls)
    assert (fixture.config.output / "checkpoint_000002.i32").is_file()
    assert not (fixture.config.output / "evaluation_000002.json").exists()


def execute_worker(extension, fixture):
    return extension.run_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)


def verify_worker(extension, fixture):
    return extension.verify_worker(fixture.config.output, fixture.expected, "anchors", 0, fixture.library_hash)


def test_worker_is_from_initialization_with_three_frozen_checkpoints_and_matched_budgets(extension, fake_worker):
    fixture = fake_worker
    result = execute_worker(extension, fixture)
    assert result["complete"] is True and fixture.bridge.closed
    assert result["steps"] == 4 and result["training_pairs"] == 512
    assert result["v1_checkpoint_256_equal"] is True
    assert [result[key]["step"] for key in ("initial", "intermediate", "final")] == [0, 2, 4]
    assert set(result["baselines_by_step"]) == {"2", "4"}
    assert result["baselines_by_step"]["2"]["training_pairs"] == 256
    assert result["baselines_by_step"]["4"]["training_pairs"] == 512
    assert result["initial"]["baseline_training_pairs"] is None
    assert result["initial"]["per_window"][0]["baseline_minus_model_nll"] == {}
    assert result["baselines"] == result["baselines_by_step"]["4"]
    assert all(result[name] is False for name in ("heldout_test_evaluated", "hardware_execution", "speed_advantage_claim"))
    assert [call[:2] for call in fixture.bridge.calls] == [
        (0, 2048), (1, 2048), (1, 2048), (0, 2048), (1, 2048), (1, 2048), (0, 2048)]
    training = [tokens for lr, threshold, tokens in fixture.bridge.calls if lr == 1]
    assert training == [pilot.window(fixture.raw, fixture.expected["dataset"], "train", index, 128)[0]
                        for index in range(4)]
    assert verify_worker(extension, fixture) == result


def test_worker_retains_midpoint_v1_mismatch_and_never_claims_complete(extension, fake_worker, monkeypatch):
    fixture = fake_worker
    changed = bytearray(fixture.v1_payload)
    changed[0] ^= 1
    monkeypatch.setattr(extension, "read_v1_checkpoint", lambda *args: bytes(changed))
    with pytest.raises(RuntimeError, match="full weights"):
        execute_worker(extension, fixture)
    output = fixture.config.output
    failure = json.loads((output / "failure.json").read_text())
    assert failure["complete"] is False and failure["steps_completed"] == 2
    assert fixture.bridge.closed
    assert (output / "checkpoint_000002.i32").is_file()
    assert not (output / "checkpoint_000004.i32").exists()
    assert not (output / "summary.json").exists()
    with pytest.raises(ValueError, match="failed evidence"):
        verify_worker(extension, fixture)


@pytest.mark.parametrize("entrypoint", ["worker", "extension"])
def test_existing_output_is_preserved_before_any_new_work(extension, fake_worker, entrypoint):
    output = fake_worker.config.output
    output.mkdir()
    record = output / "previous-result.json"
    record.write_text('{"historical":true}')
    with pytest.raises(FileExistsError):
        if entrypoint == "worker":
            execute_worker(extension, fake_worker)
        else:
            extension.run_extension(fake_worker.config)
    assert list(output.iterdir()) == [record]
    assert record.read_text() == '{"historical":true}'
    assert not fake_worker.bridge.calls


@pytest.mark.parametrize("failure", ["native", "close", "interrupt", "source", "historical"])
def test_worker_failures_are_retained_and_cannot_become_success(extension, fake_worker, monkeypatch, failure):
    fixture = fake_worker
    expected_error = RuntimeError
    if failure == "native":
        fixture.bridge.mismatch = True
    elif failure == "close":
        fixture.bridge.close_error = RuntimeError("injected close failure")
    elif failure == "interrupt":
        def interrupt(*args):
            raise KeyboardInterrupt("injected worker cancellation")

        monkeypatch.setattr(fixture.bridge, "step", interrupt)
        expected_error = KeyboardInterrupt
    else:
        original_close = fixture.bridge.close

        def close_and_mutate_identity():
            original_close()
            if failure == "source":
                monkeypatch.setattr(extension, "source_identity", lambda: {"changed-source.py": "b" * 64})
            else:
                changed = dict(fixture.expected["v1_evidence"], artifact_manifest_sha256="f" * 64)
                monkeypatch.setattr(extension, "v1_binding", lambda *args: changed)

        monkeypatch.setattr(fixture.bridge, "close", close_and_mutate_identity)
    with pytest.raises(expected_error):
        execute_worker(extension, fixture)
    output = fixture.config.output
    assert json.loads((output / "failure.json").read_text())["complete"] is False
    assert fixture.bridge.closed
    assert (output / "manifest.json").is_file()
    assert not (output / "summary.json").exists()
    if failure == "native":
        assert (output / "eval_000000_window_000_predictions.i32").is_file()
        assert (output / "eval_000000_window_000_gradients.i32").is_file()
    with pytest.raises(ValueError, match="failed evidence"):
        verify_worker(extension, fixture)


def test_parent_reference_mismatch_is_rejected_before_native_access(extension, fake_worker, tmp_path):
    fixture = fake_worker
    parent = tmp_path / "parent.json"
    incorrect = deepcopy(fixture.expected)
    incorrect["v1_evidence"]["manifest_sha256"] = "f" * 64
    rewrite_json(parent, incorrect)
    fixture.config.parent_manifest = parent
    opened = []

    def forbidden_factory(*args):
        opened.append(args)
        raise AssertionError("transport opened before parent validation")

    with pytest.raises(RuntimeError, match="differs from parent"):
        extension.run_worker(fixture.config, bridge_factory=forbidden_factory, observer=fixture.observer)
    assert not opened
    assert (fixture.config.output / "failure.json").is_file()


@pytest.mark.parametrize("failure", ["aggregate", "baseline_budget", "initial_baseline", "midpoint", "initial"])
def test_rehashed_records_cannot_forge_budget_or_checkpoint_science(extension, fake_worker, failure):
    fixture = fake_worker
    execute_worker(extension, fixture)
    output = fixture.config.output
    summary = json.loads((output / "summary.json").read_text())
    if failure == "aggregate":
        summary["intermediate"]["lut_normalized_nll"] = 0.01
        rewrite_json(output / "evaluation_000002.json", summary["intermediate"])
    elif failure == "baseline_budget":
        summary["baselines_by_step"]["2"] = deepcopy(summary["baselines_by_step"]["4"])
        rewrite_json(output / "baselines_000002.json", summary["baselines_by_step"]["2"])
    elif failure == "initial_baseline":
        summary["initial"]["per_window"][0]["baseline_minus_model_nll"] = {"unigram": 0.1, "bigram": 0.2}
        rewrite_json(output / "eval_000000_window_000.json", summary["initial"]["per_window"][0])
        rewrite_json(output / "evaluation_000000.json", summary["initial"])
    else:
        key, step = ("intermediate", 2) if failure == "midpoint" else ("initial", 0)
        path = output / ("checkpoint_%06d.i32" % step)
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
        summary[key]["checkpoint"]["sha256"] = pilot.digest(bytes(payload))
        rewrite_json(output / ("checkpoint_%06d.json" % step), summary[key]["checkpoint"])
        rewrite_json(output / ("evaluation_%06d.json" % step), summary[key])
    rewrite_json(output / "summary.json", summary)
    rehash_artifacts(extension, output, extension.SCHEMA)
    with pytest.raises(ValueError):
        verify_worker(extension, fixture)


@pytest.mark.parametrize("failure", ["repeat_window", "native_payload", "source", "corpus"])
def test_verifier_requires_fresh_source_bound_native_evidence(extension, fake_worker, failure):
    fixture = fake_worker
    execute_worker(extension, fixture)
    output = fixture.config.output
    if failure == "repeat_window":
        path = output / "training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["window_index"] = 0
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    elif failure == "native_payload":
        path = output / "training_gradients.i32"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif failure == "source":
        path = output / "manifest.json"
        record = json.loads(path.read_text())
        record["source_hashes"] = {"unrelated.py": "b" * 64}
        rewrite_json(path, record)
    else:
        path = fixture.config.corpus
        path.write_bytes(b" " * len(path.read_bytes()))
    rehash_artifacts(extension, output, extension.SCHEMA)
    with pytest.raises(ValueError):
        verify_worker(extension, fixture)


@pytest.mark.parametrize("failed_index", [0, 5])
def test_any_incomplete_cell_blocks_both_arms_after_exactly_six_declared_attempts(
        extension, fake_worker, monkeypatch, failed_index):
    fixture = fake_worker
    commands, gates = [], []
    monkeypatch.setattr(pilot, "build_csim", lambda output, width: fixture.config.library)

    def process(command, log_path):
        index = len(commands)
        commands.append(command)
        log_path.write_text("injected offline process\n")
        return {"returncode": 1 if index == failed_index else 0, "timed_out": False}

    def verified(directory, expected, arm, seed, library_hash):
        assert expected["source_hashes"] == fixture.expected["source_hashes"]
        return {"arm": arm, "seed": seed}

    monkeypatch.setattr(extension, "_run_process", process)
    monkeypatch.setattr(extension, "verify_worker", verified)
    monkeypatch.setattr(extension, "_compact_worker", lambda value: value)
    monkeypatch.setattr(extension, "gate_arm", lambda values: gates.append(values))
    result = extension.run_extension(fixture.config)
    assert result["complete"] is False
    assert result["declared_workers"] == len(commands) == 6
    assert result["completed_workers"] == 5
    assert not gates
    assert all(row["larger_paid_learning_benchmark_gate"] is False for row in result["gates"].values())
    expected_cells = [(arm, str(seed)) for arm in ("anchors", "anchors_fine") for seed in (0, 1, 2)]
    assert [(command[command.index("--arm") + 1], command[command.index("--seed") + 1])
            for command in commands] == expected_cells
    assert all("--worker" in command and "--parent-manifest" in command and "--v1-run" in command
               for command in commands)
    failed_arm, failed_seed = expected_cells[failed_index]
    failed_name = failed_arm + "_seed_" + failed_seed
    output = fixture.config.output
    assert (output / (failed_name + "_failure.json")).is_file()
    assert json.loads((output / failed_name / "failure.json").read_text())["complete"] is False
    assert (output / "failure.json").is_file()
    assert (output / "artifact_manifest.json").is_file()


def test_final_gates_are_exactly_the_frozen_three_seed_initial_context_rules(extension):
    # Reuse is intentional: the separately tested v1 thresholds cannot drift in
    # this extension while intermediate metrics are being added to the record.
    assert extension.gate_arm is extension.screen.gate_arm
    assert extension.recipe()["gates"] == extension.screen.recipe()["gates"]
