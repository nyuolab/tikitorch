"""Offline, independently derived contracts for the declared CPU head screen.

Tests never open PCI devices, contact AWS, or execute a production screen.
"""

import json
import math
import random
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from manuscript.tools import screen_fpga_lm_heads as screen
from scripts import fpga_lm_pretrain as pilot


@pytest.fixture(scope="module")
def shape():
    return pilot.layout(512)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_declared_head_arms_are_deterministic_and_nonhead_bytes_are_paired(shape, seed):
    reference = shape.build_init(seed)
    start, stop = shape.OFF_W_HEAD, shape.OFF_W_HEAD + shape.WORDS_W_HEAD
    buffers = {arm: screen.initialize_buffer(shape, seed, arm)
               for arm in ("reference", "wider", "anchors", "anchors_fine")}
    assert buffers["reference"] == reference
    for arm, buffer in buffers.items():
        assert len(buffer) == len(reference)
        assert buffer[:start] == reference[:start]
        assert buffer[stop:shape.OFF_CFG] == reference[stop:shape.OFF_CFG]
        assert buffer[shape.OFF_CFG + 2:] == reference[shape.OFF_CFG + 2:]
        assert buffer[shape.OFF_CFG:shape.OFF_CFG + 2] == [1, 0 if arm == "anchors_fine" else 4]
    # Independent scalar recurrence: do not call the implementation's RNG.
    state = (seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
    expected_wider = []
    for _ in range(shape.WORDS_W_HEAD):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        expected_wider.append(-512 + state % 1025)
    assert buffers["wider"][start:stop] == expected_wider
    expected_anchors = reference[start:stop].copy()
    for vocabulary in range(128):
        anchor = (5 * vocabulary + seed) % 512
        for feature in range(512):
            if feature != anchor:
                expected_anchors[vocabulary * 512 + feature] -= 8192
    assert buffers["anchors"][start:stop] == expected_anchors
    assert buffers["anchors_fine"][start:stop] == expected_anchors
    assert all(-8196 <= value <= 4 for value in expected_anchors)
    assert screen.initialize_buffer(shape, seed, "wider") == buffers["wider"]


def test_unknown_initialization_arm_cannot_silently_become_reference(shape):
    with pytest.raises(ValueError):
        screen.initialize_buffer(shape, 0, "undeclared-arm")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(419)
    raw = bytes(rng.randrange(128) for _ in range(50_000))
    path = tmp_path / "ascii.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 256, 24)
    return raw, dataset


def baseline_oracle(raw, dataset, train_indices, eval_indices, sequence=128):
    """Explicit supervised pairs, never joined across windows or extra data."""
    unigrams = [1] * 128
    bigrams = [[1] * 128 for _ in range(128)]
    pairs = 0
    for window_index in train_indices:
        start = dataset["splits"]["train"]["start"] + window_index * (sequence + 1)
        for position in range(sequence):
            token, target = raw[start + position:start + position + 2]
            unigrams[target] += 1
            bigrams[token][target] += 1
            pairs += 1
    unigram_total = sum(unigrams)
    unigram_prediction = max(range(128), key=unigrams.__getitem__)
    bigram_totals = [sum(row) for row in bigrams]
    bigram_predictions = [max(range(128), key=row.__getitem__) for row in bigrams]
    result = {"unigram": [], "bigram": [], "training_pairs": pairs}
    for window_index in eval_indices:
        start = dataset["splits"]["validation"]["start"] + window_index * (sequence + 1)
        unigram_nll, bigram_nll, unigram_correct, bigram_correct = 0.0, 0.0, 0, 0
        for position in range(sequence):
            token, target = raw[start + position:start + position + 2]
            unigram_nll -= math.log(unigrams[target] / unigram_total)
            bigram_nll -= math.log(bigrams[token][target] / bigram_totals[token])
            unigram_correct += unigram_prediction == target
            bigram_correct += bigram_predictions[token] == target
        result["unigram"].append({"window": window_index, "nll": unigram_nll / sequence,
                                  "accuracy": unigram_correct / sequence})
        result["bigram"].append({"window": window_index, "nll": bigram_nll / sequence,
                                 "accuracy": bigram_correct / sequence})
    return result


def test_baselines_fit_exact_supervised_targets_and_within_window_pairs(corpus):
    raw, dataset = corpus
    result = screen.matched_baselines(raw, dataset)
    expected = baseline_oracle(raw, dataset, range(256), range(8, 24))
    assert result["training_pairs"] == expected["training_pairs"] == 32_768
    assert sum(result["unigram_target_counts"]) == 32_768 + 128
    assert sum(map(sum, result["bigram_pair_counts"])) == 32_768 + 128 * 128
    assert result["train_indices"] == list(range(256))
    assert result["validation_indices"] == list(range(8, 24))
    for name in ("unigram", "bigram"):
        assert result[name]["nll"] == pytest.approx(sum(row["nll"] for row in expected[name]) / 16, abs=1e-12)
        assert result[name]["accuracy"] == pytest.approx(sum(row["accuracy"] for row in expected[name]) / 16)
        assert len(result[name]["per_window"]) == 16
        for observed, row in zip(result[name]["per_window"], expected[name]):
            assert observed["window_index"] == row["window"]
            assert observed["nll"] == pytest.approx(row["nll"], abs=1e-12)
            assert observed["accuracy"] == pytest.approx(row["accuracy"])


def test_unused_training_and_test_bytes_cannot_affect_matched_controls(corpus):
    raw, dataset = corpus
    changed = bytearray(raw)
    end_used_training = dataset["splits"]["train"]["start"] + 256 * 129
    end_training = dataset["splits"]["train"]["stop"]
    changed[end_used_training:end_training] = b" " * (end_training - end_used_training)
    test_start = dataset["splits"]["test"]["start"]
    changed[test_start:] = b"!" * (len(changed) - test_start)
    assert screen.matched_baselines(bytes(changed), dataset) == screen.matched_baselines(raw, dataset)


def test_baseline_tiny_oracle_exposes_cross_window_transition_leakage():
    # Two windows contain pairs 0->1,1->2 and 3->4,4->5. The pair 2->3
    # crosses the window boundary and must not be fitted by either model.
    raw = bytes([0, 1, 2, 3, 4, 5, 2, 3, 2, 3, 4, 5])
    dataset = {"splits": {"train": {"start": 0, "stop": 6, "windows": 2},
                           "validation": {"start": 6, "stop": 12, "windows": 2},
                           "test": {"start": 12, "stop": 12, "windows": 0}}}
    expected = baseline_oracle(raw, dataset, range(2), range(2), sequence=2)
    result = screen.matched_baselines(raw, dataset, train_indices=range(2), eval_indices=range(2), sequence=2)
    assert result["training_pairs"] == 4
    for name in ("unigram", "bigram"):
        assert result[name]["nll"] == pytest.approx(sum(row["nll"] for row in expected[name]) / 2, abs=1e-12)


def successful_seed_records():
    """Exact, complete three-seed fixture with margins above every screen gate."""
    records = []
    for seed in (0, 1, 2):
        initial = {"lut_normalized_nll": 3.2, "context_disabled_nll": 3.2,
                   "saturation_events": 0, "native_reference_equal": True, "native_routes_equal": True,
                   "route_diagnostics": {"positions": 2048, "nonzero_logit_gradient_positions": 2048,
                       "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 512,
                       "head_winners_ge16_positions": 1536, "backward_int32_overflow_detected": False}}
        final = deepcopy(initial)
        final.update(lut_normalized_nll=3.0, context_disabled_nll=3.04)
        records.append({"seed": seed, "initial": initial, "final": final,
                        "baselines": {"training_pairs": 32768, "unigram": {"nll": 3.1},
                                      "bigram": {"nll": 3.03}}})
    return records


def test_all_three_seeds_must_pass_without_authorizing_paid_execution():
    result = screen.gate_arm(successful_seed_records())
    assert result["mechanism"] is True
    assert result["context"] is True
    assert result["bigram"] is True
    assert result["larger_paid_learning_benchmark_gate"] is True
    assert result["paid_execution_authorized"] is False
    assert result["statistical_significance_claim"] is False
    assert [row["seed"] for row in result["per_seed"]] == [0, 1, 2]
    assert result["mean_bigram_advantage_nats"] == pytest.approx(0.03)


@pytest.mark.parametrize("seeds", [[0, 1], [0, 0, 2], [0, 1, 3], [0, 1, 2, 2], [False, 1, 2]])
def test_missing_duplicate_or_undeclared_seed_cannot_pass(seeds):
    template = successful_seed_records()[0]
    records = [dict(deepcopy(template), seed=seed) for seed in seeds]
    with pytest.raises(ValueError):
        screen.gate_arm(records)


@pytest.mark.parametrize("field,value", [
    ("exact_zero_head_input_vjp_on_nonzero_gradient_positions", 513),
    ("head_winners_ge16_positions", 1535),
    ("backward_int32_overflow_detected", True),
])
def test_one_seed_mechanism_failure_blocks_the_whole_arm(field, value):
    records = successful_seed_records()
    records[1]["final"]["route_diagnostics"][field] = value
    result = screen.gate_arm(records)
    assert result["mechanism"] is False
    assert result["larger_paid_learning_benchmark_gate"] is False
    assert [row["mechanism"] for row in result["per_seed"]] == [True, False, True]


@pytest.mark.parametrize("field,value", [
    ("saturation_events", 1), ("native_reference_equal", False), ("native_routes_equal", False),
])
def test_no_observed_saturation_and_both_native_checks_are_required(field, value):
    records = successful_seed_records()
    records[2]["final"][field] = value
    assert screen.gate_arm(records)["larger_paid_learning_benchmark_gate"] is False


def test_zero_vjp_fraction_denominator_is_nonzero_logit_gradient_positions():
    records = successful_seed_records()
    records[0]["final"]["route_diagnostics"].update(
        nonzero_logit_gradient_positions=1024,
        exact_zero_head_input_vjp_on_nonzero_gradient_positions=257)
    result = screen.gate_arm(records)
    assert result["per_seed"][0]["zero_head_vjp_fraction_on_nonzero_logit_gradients"] == 257 / 1024
    assert result["mechanism"] is False


def test_no_nonzero_logit_gradient_is_not_a_trainability_success():
    records = successful_seed_records()
    records[0]["final"]["route_diagnostics"].update(
        nonzero_logit_gradient_positions=0,
        exact_zero_head_input_vjp_on_nonzero_gradient_positions=0)
    result = screen.gate_arm(records)
    assert result["mechanism"] is False
    assert result["per_seed"][0]["zero_head_vjp_fraction_on_nonzero_logit_gradients"] is None


@pytest.mark.parametrize("field,value", [
    ("positions", True), ("nonzero_logit_gradient_positions", 2049),
    ("exact_zero_head_input_vjp_on_nonzero_gradient_positions", -1),
    ("head_winners_ge16_positions", 2049),
])
def test_invalid_route_counts_are_rejected(field, value):
    records = successful_seed_records()
    records[0]["final"]["route_diagnostics"][field] = value
    with pytest.raises(ValueError):
        screen.gate_arm(records)


def test_incomplete_validation_positions_cannot_satisfy_mechanism():
    records = successful_seed_records()
    records[0]["final"]["route_diagnostics"].update(
        positions=128, nonzero_logit_gradient_positions=128,
        exact_zero_head_input_vjp_on_nonzero_gradient_positions=32,
        head_winners_ge16_positions=96)
    assert screen.gate_arm(records)["mechanism"] is False


@pytest.mark.parametrize("failure", ["unigram", "final_context", "inherited_context"])
def test_context_gate_requires_useful_learning_beyond_initial_bias(failure):
    records = successful_seed_records()
    record = records[1]
    if failure == "unigram":
        record["baselines"]["unigram"]["nll"] = 3.049
    elif failure == "final_context":
        record["final"]["context_disabled_nll"] = 3.019
    else:
        # Strong context sensitivity that already existed at initialization
        # is not evidence that training learned to use temporal context.
        record["initial"]["context_disabled_nll"] = 3.24
    result = screen.gate_arm(records)
    assert result["context"] is False
    assert result["larger_paid_learning_benchmark_gate"] is False


def test_inclusive_numerical_gate_boundaries():
    records = successful_seed_records()
    for record in records:
        record["baselines"]["unigram"]["nll"] = 3.05
        record["baselines"]["bigram"]["nll"] = 3.02
        record["final"]["context_disabled_nll"] = 3.02
    assert screen.gate_arm(records)["larger_paid_learning_benchmark_gate"] is True


@pytest.mark.parametrize("advantages", [(0.019, 0.019, 0.019), (-0.021, 0.05, 0.05)])
def test_bigram_requires_both_mean_advantage_and_no_bad_single_seed(advantages):
    records = successful_seed_records()
    for record, advantage in zip(records, advantages):
        record["baselines"]["bigram"]["nll"] = 3.0 + advantage
    result = screen.gate_arm(records)
    assert result["mechanism"] and result["context"]
    assert result["bigram"] is False
    assert result["larger_paid_learning_benchmark_gate"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scientific_metrics_are_rejected(value):
    records = successful_seed_records()
    records[0]["final"]["lut_normalized_nll"] = value
    with pytest.raises(ValueError):
        screen.gate_arm(records)


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    """Small transport-only fixture; scientific math has independent tests."""
    monkeypatch.setattr(screen, "WIDTH", 16)
    monkeypatch.setattr(screen, "TRAIN_STEPS", 2)
    monkeypatch.setattr(screen, "VALIDATION_INDICES", (0,))
    monkeypatch.setattr(screen, "initialize_buffer", lambda layout, seed, arm: layout.build_init(seed))
    monkeypatch.setattr(screen, "source_identity", lambda: {"frozen-source.py": "a" * 64})
    path = tmp_path / "corpus.txt"
    path.write_bytes(bytes(range(128)) * 100)
    library = tmp_path / "fake.so"
    library.write_bytes(b"fake library: never loaded")
    config = SimpleNamespace(corpus=path, output=tmp_path / "worker", library=library, seed=0, arm="reference")
    shape = pilot.layout(16)
    forward = {"predictions": [0] * shape.ML_L_SEQ,
               "gradients": [0] * shape.WORDS_GRAD_LOGIT,
               "lut_normalized_nll": 4.0, "accuracy": 0.0,
               "saturation_events": 0, "rounded_zero_target_mass": 0}
    route_arrays = {name: [0] * getattr(shape, "WORDS_" + name)
                    for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}
    routes = {"positions": 128, "nonzero_logit_gradient_positions": 128,
              "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
              "head_winners_ge16_positions": 128, "single_shared_head_winner_positions": 0,
              "nonzero_embedding_input_vjp_positions": 128,
              "max_abs_head_input_vjp": 1, "max_abs_embedding_input_vjp": 1,
              "backward_int32_overflow_detected": False, "head_winner_counts": [16] * 128}
    monkeypatch.setattr(pilot, "integer_forward", lambda *args, **kwargs: deepcopy(forward))

    class FakeBridge:
        def __init__(self, observed_library, backend, layout):
            assert observed_library == library and backend == "csim" and layout.ML_D == 16
            self.memory = [0] * layout.ML_BUFFER_WORDS
            self.calls = []
            self.closed = False
            self.close_error = None
            self.mismatch = False
            self.mutate_source = False

        def write(self, offset, values):
            self.memory[offset:offset + len(values)] = values

        def read(self, offset, words):
            return self.memory[offset:offset + words]

        def step(self, lr, threshold):
            self.calls.append((lr, threshold, self.read(shape.OFF_TOKENS, shape.ML_L_SEQ)))
            self.write(shape.OFF_PRED, [int(self.mismatch)] * shape.ML_L_SEQ)
            self.write(shape.OFF_GRAD_LOGIT, forward["gradients"])
            for name, values in route_arrays.items():
                self.write(getattr(shape, "OFF_" + name), values)

        def close(self):
            self.closed = True
            if self.mutate_source:
                monkeypatch.setattr(screen, "source_identity", lambda: {"changed": "b" * 64})
            if self.close_error:
                raise self.close_error

    bridge = FakeBridge(library, "csim", shape)
    observer = lambda *args, **kwargs: {**deepcopy(forward), "native_routes": deepcopy(route_arrays),
                                      "route_diagnostics": deepcopy(routes)}
    _, _, expected = screen._common_manifest(path)
    return SimpleNamespace(config=config, bridge=bridge, factory=lambda *args: bridge,
                           observer=observer, expected=expected, shape=shape,
                           library_hash=pilot.digest(library.read_bytes()))


def execute_fake(fixture):
    return screen.run_worker(fixture.config, bridge_factory=fixture.factory, observer=fixture.observer)


def verify_fake(fixture):
    return screen.verify_worker(fixture.config.output, fixture.expected, "reference", 0, fixture.library_hash)


def refresh_manifest(directory):
    artifact_path = directory / "artifact_manifest.json"
    records = screen._artifact_manifest(directory)
    records["files"] = [row for row in records["files"] if row["file"] != artifact_path.name]
    artifact_path.write_text(json.dumps(records))


def test_worker_uses_exact_fresh_training_and_frozen_evaluations(fake_worker):
    summary = execute_fake(fake_worker)
    assert summary["complete"] is True
    assert summary["training_pairs"] == 256
    assert summary["heldout_test_evaluated"] is False
    assert summary["hardware_execution"] is False
    assert summary["speed_advantage_claim"] is False
    assert fake_worker.bridge.closed
    assert [call[:2] for call in fake_worker.bridge.calls] == [(0, 2048), (1, 2048), (1, 2048), (0, 2048)]
    raw = fake_worker.config.corpus.read_bytes()
    for index, call in enumerate(fake_worker.bridge.calls[1:3]):
        expected_tokens, _ = pilot.window(raw, fake_worker.expected["dataset"], "train", index, 128)
        assert call[2] == expected_tokens
    assert verify_fake(fake_worker) == summary


@pytest.mark.parametrize("entrypoint", ["worker", "screen"])
def test_existing_output_is_never_overwritten(fake_worker, entrypoint):
    fake_worker.config.output.mkdir()
    sentinel = fake_worker.config.output / "existing.txt"
    sentinel.write_text("previous scientific evidence")
    with pytest.raises(FileExistsError):
        if entrypoint == "worker":
            execute_fake(fake_worker)
        else:
            screen.run_screen(fake_worker.config)
    assert list(fake_worker.config.output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "previous scientific evidence"
    assert not fake_worker.bridge.calls


@pytest.mark.parametrize("failure", ["native", "close", "source"])
def test_failed_worker_retains_evidence_and_cannot_emit_success(fake_worker, failure):
    if failure == "native":
        fake_worker.bridge.mismatch = True
    elif failure == "close":
        fake_worker.bridge.close_error = RuntimeError("close failure")
    else:
        fake_worker.bridge.mutate_source = True
    with pytest.raises(RuntimeError):
        execute_fake(fake_worker)
    output = fake_worker.config.output
    assert json.loads((output / "failure.json").read_text())["complete"] is False
    assert not (output / "summary.json").exists()
    assert fake_worker.bridge.closed
    assert (output / "manifest.json").is_file()
    if failure == "native":
        assert (output / "eval_000000_window_000_predictions.i32").is_file()
        assert (output / "eval_000000_window_000_gradients.i32").is_file()
    with pytest.raises(ValueError, match="failed worker"):
        verify_fake(fake_worker)


def test_changed_corpus_cannot_rebind_completed_evidence(fake_worker):
    execute_fake(fake_worker)
    fake_worker.config.corpus.write_bytes(b" " * len(fake_worker.config.corpus.read_bytes()))
    with pytest.raises(ValueError, match="corpus changed"):
        verify_fake(fake_worker)


def test_uncovered_added_artifact_is_rejected(fake_worker):
    execute_fake(fake_worker)
    (fake_worker.config.output / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        verify_fake(fake_worker)


def test_artifact_symlinks_are_rejected_before_following_payloads(fake_worker, tmp_path):
    execute_fake(fake_worker)
    outside = tmp_path / "unrelated.txt"
    outside.write_text("not part of these results")
    (fake_worker.config.output / "outside.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        verify_fake(fake_worker)


def test_source_manifest_mismatch_cannot_be_covered_by_new_hashes(fake_worker):
    execute_fake(fake_worker)
    path = fake_worker.config.output / "manifest.json"
    record = json.loads(path.read_text())
    record["source_hashes"] = {"unrelated.py": "0" * 64}
    path.write_text(json.dumps(record))
    refresh_manifest(fake_worker.config.output)
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_fake(fake_worker)


@pytest.mark.parametrize("change", ["aggregate_nll", "aggregate_routes", "baseline"])
def test_consistent_file_hashes_do_not_bless_inconsistent_scientific_claims(fake_worker, change):
    execute_fake(fake_worker)
    output = fake_worker.config.output
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    if change == "baseline":
        summary["baselines"]["bigram"]["nll"] = 100.0
        (output / "baselines.json").write_text(json.dumps(summary["baselines"]))
    else:
        final = summary["final"]
        if change == "aggregate_nll":
            final["lut_normalized_nll"] = 0.01
        else:
            final["route_diagnostics"]["head_winners_ge16_positions"] = 0
        (output / "evaluation_000002.json").write_text(json.dumps(final))
    summary_path.write_text(json.dumps(summary))
    refresh_manifest(output)
    with pytest.raises(ValueError):
        verify_fake(fake_worker)


def test_subprocess_timeout_kills_and_reaps_only_owned_process_group(tmp_path, monkeypatch):
    class Process:
        pid, returncode = 17321, -9

        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            if len(self.waits) == 1:
                raise subprocess.TimeoutExpired(["fake-worker"], timeout)
            return self.returncode

    process, invocations, kills = Process(), [], []

    def launch(command, **kwargs):
        invocations.append((command, kwargs))
        return process

    monkeypatch.setattr(screen.subprocess, "Popen", launch)
    monkeypatch.setattr(screen.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    command = ["python", "path with spaces.py"]
    result = screen._run_process(command, tmp_path / "worker.log")
    assert result == {"returncode": -9, "timed_out": True}
    assert process.waits == [screen.WORKER_TIMEOUT_S, 10]
    assert kills == [(process.pid, screen.signal.SIGKILL)]
    assert invocations[0][0] == command
    assert invocations[0][1]["start_new_session"] is True
    assert invocations[0][1].get("shell", False) is False


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interrupted_parent_reaps_owned_worker_before_propagating(tmp_path, monkeypatch, failure):
    class Process:
        pid, returncode = 19341, -9

        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise failure("interrupted wait")
            return self.returncode

    process, kills = Process(), []
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(screen.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    with pytest.raises(failure):
        screen._run_process(["fake-worker"], tmp_path / "worker.log")
    assert kills == [(process.pid, screen.signal.SIGKILL)]
    assert process.waits == 2


def test_interrupted_worker_records_failure_and_closes_transport(fake_worker, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt("cancelled offline unit test")

    monkeypatch.setattr(fake_worker.bridge, "step", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute_fake(fake_worker)
    failure = json.loads((fake_worker.config.output / "failure.json").read_text())
    assert failure["complete"] is False and failure["type"] == "KeyboardInterrupt"
    assert fake_worker.bridge.closed
    assert not (fake_worker.config.output / "summary.json").exists()


def test_parent_source_mismatch_is_detected_before_opening_native_transport(fake_worker, tmp_path):
    parent = tmp_path / "parent.json"
    expected = deepcopy(fake_worker.expected)
    expected["source_hashes"]["extra.py"] = "0" * 64
    parent.write_text(json.dumps(expected))
    fake_worker.config.parent_manifest = parent
    factories = []

    def forbidden_factory(*args):
        factories.append(args)
        raise AssertionError("native access before validating source")

    with pytest.raises(RuntimeError, match="parent manifest"):
        screen.run_worker(fake_worker.config, bridge_factory=forbidden_factory, observer=fake_worker.observer)
    assert not factories
    assert (fake_worker.config.output / "failure.json").is_file()


def test_even_rehashed_initial_checkpoint_must_match_the_declared_initializer(fake_worker):
    execute_fake(fake_worker)
    output = fake_worker.config.output
    path = output / "checkpoint_000000.i32"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    checkpoint = summary["initial"]["checkpoint"]
    checkpoint["sha256"] = pilot.digest(bytes(payload))
    (output / "checkpoint_000000.json").write_text(json.dumps(checkpoint))
    (output / "evaluation_000000.json").write_text(json.dumps(summary["initial"]))
    summary_path.write_text(json.dumps(summary))
    refresh_manifest(output)
    with pytest.raises(ValueError, match="initial checkpoint"):
        verify_fake(fake_worker)


def test_repeated_training_window_cannot_count_as_fresh_data(fake_worker):
    execute_fake(fake_worker)
    path = fake_worker.config.output / "training.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["window_index"] = 0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    refresh_manifest(fake_worker.config.output)
    with pytest.raises(ValueError, match="fresh training sequence"):
        verify_fake(fake_worker)


def test_saved_native_arrays_are_covered_by_payload_hashes(fake_worker):
    execute_fake(fake_worker)
    path = fake_worker.config.output / "eval_000002_window_000_gradients.i32"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    refresh_manifest(fake_worker.config.output)
    with pytest.raises(ValueError, match="array coverage/hash"):
        verify_fake(fake_worker)


def test_parent_attempts_only_twelve_declared_cells_and_any_failure_blocks_all_gates(fake_worker, monkeypatch):
    commands, gate_calls = [], []
    monkeypatch.setattr(pilot, "build_csim", lambda output, width: fake_worker.config.library)

    def process(command, log_path):
        commands.append(command)
        log_path.write_text("offline fake worker\n")
        return {"returncode": 1 if len(commands) == 1 else 0, "timed_out": False}

    def verified(directory, expected, arm, seed, library_hash):
        assert expected["source_hashes"] == fake_worker.expected["source_hashes"]
        return dict(successful_seed_records()[seed], arm=arm)

    monkeypatch.setattr(screen, "_run_process", process)
    monkeypatch.setattr(screen, "verify_worker", verified)
    monkeypatch.setattr(screen, "_compact_worker", lambda row: row)
    monkeypatch.setattr(screen, "gate_arm", lambda rows: gate_calls.append(rows))
    result = screen.run_screen(fake_worker.config)
    assert result["complete"] is False
    assert result["declared_workers"] == len(commands) == 12
    assert result["completed_workers"] == 11
    assert not gate_calls
    assert all(row["larger_paid_learning_benchmark_gate"] is False for row in result["gates"].values())
    expected_cells = [(arm, str(seed)) for arm in ("reference", "wider", "anchors", "anchors_fine")
                      for seed in (0, 1, 2)]
    assert [(command[command.index("--arm") + 1], command[command.index("--seed") + 1])
            for command in commands] == expected_cells
    assert all("--worker" in command and "--parent-manifest" in command for command in commands)
    output = fake_worker.config.output
    assert json.loads((output / "reference_seed_0" / "failure.json").read_text())["complete"] is False
    assert (output / "reference_seed_0_failure.json").is_file()
    assert (output / "failure.json").is_file()
    assert (output / "artifact_manifest.json").is_file()
