"""Offline contracts for fixed-recipe, independently ordered confirmation.

Synthetic corpora and a fake full-state native ABI are used here. These tests
never execute a scientific study, launch native training, or contact AWS.
"""

import importlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.confirm_fpga_lm_efsgd")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(83091)
    raw = bytes(rng.randrange(128) for _ in range(200_000))
    path = tmp_path / "corpus.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 1024, 72)
    return raw, dataset


def exposure_oracle(raw, dataset, indices, validation):
    """Independent byte slicing/counting, including first-max tie handling."""
    uni = [1] * 128
    bi = [[1] * 128 for _ in range(128)]
    for index in indices:
        start = dataset["splits"]["train"]["start"] + index * 129
        for offset in range(start, start + 128):
            token, target = raw[offset], raw[offset + 1]
            uni[target] += 1
            bi[token][target] += 1
    results = {}
    for name, counts in (("unigram", [uni] * 128), ("bigram", bi)):
        totals = [sum(row) for row in counts]
        predictions = [max(range(128), key=row.__getitem__) for row in counts]
        losses, correct = [], 0
        for index in validation:
            start = dataset["splits"]["validation"]["start"] + index * 129
            for offset in range(start, start + 128):
                token, target = raw[offset], raw[offset + 1]
                losses.append(-math.log(counts[token][target] / totals[token]))
                correct += predictions[token] == target
        results[name] = {"nll": sum(losses) / len(losses), "accuracy": correct / len(losses)}
    return results


def test_protocol_keeps_optimizer_fixed_and_changes_order_not_initialization(study):
    assert (study.WIDTH, study.ZSHIFT, study.K, study.STEPS) == (512, 2, 18, 1024)
    assert study.SEEDS == (0, 1, 2)
    assert study.CHECKPOINTS == (0, 256, 1024)
    assert study.VALIDATION == tuple(range(56, 72))
    assert study.WORKER_TIMEOUT_S == 1200
    recipe = study.recipe()
    assert recipe["inherited_host_fitting_pair_exposures_per_seed"] == 131072
    assert recipe["new_host_fitting_pair_exposures_per_seed"] == 0
    assert recipe["initialization_refitted"] is False
    assert recipe["gradient_pair_exposures_per_seed"] == 131072
    assert recipe["model_history_pair_exposures_per_seed"] == 262144
    assert recipe["distinct_training_pairs_per_seed"] == 131072
    assert recipe["validation_indices"] == list(range(56, 72))
    assert recipe["fitting_indices"] == list(range(1024))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_training_order_is_exact_seeded_permutation_and_does_not_touch_global_rng(study, seed):
    expected = list(range(1024))
    random.Random(seed).shuffle(expected)
    before = random.getstate()
    actual = study.training_order(seed)
    assert actual == expected
    assert actual == study.training_order(seed)
    assert len(actual) == len(set(actual)) == 1024
    assert sorted(actual) == list(range(1024))
    assert random.getstate() == before
    actual[0] = -1
    assert study.training_order(seed) == expected


def test_three_orders_are_distinct_and_their_midpoint_subsets_differ(study):
    orders = [study.training_order(seed) for seed in study.SEEDS]
    assert len({tuple(order) for order in orders}) == 3
    assert len({frozenset(order[:256]) for order in orders}) == 3
    assert all(order != list(range(1024)) for order in orders)


@pytest.mark.parametrize("seed", [True, False, 0.0, 1.0, -1, 3, "0", None])
def test_order_rejects_undeclared_seed_and_bool_integer_aliases(study, seed):
    with pytest.raises(ValueError):
        study.training_order(seed)


def test_source_identity_covers_native_and_all_transitive_frozen_references(study):
    sources = study.source_identity()
    for dependency in (study.optimizer,):
        for name, digest in dependency.source_identity().items():
            assert sources[name] == digest
    own = Path(study.__file__).resolve()
    assert sources[own.relative_to(pilot.REPO).as_posix()] == pilot.digest(own.read_bytes())
    for name in ("train_fpga_lm_efsgd.py", "check_fpga_lm_bigram_capacity.py",
                 "observe_fpga_lm_routes.py", "fpga_lm_efsgd_v1.cpp"):
        assert any(Path(path).name == name for path in sources)


@pytest.mark.parametrize("seed,step", [(seed, step) for seed in (0, 1, 2) for step in (0, 256, 1024)])
def test_baseline_counts_full_fit_then_exact_shuffled_replay_prefix(study, corpus, seed, step):
    raw, dataset = corpus
    order = list(range(1024))
    random.Random(seed).shuffle(order)
    indices = list(range(1024)) + order[:step]
    actual = study.exposure_baseline(raw, dataset, seed, step)
    expected = exposure_oracle(raw, dataset, indices, range(56, 72))
    assert actual["supervised_exposures"] == (1024 + step) * 128
    assert actual["distinct_supervised_pairs"] == 131072
    assert actual["alpha"] == 1
    assert actual["exposure_indices"] == indices
    assert actual["validation_indices"] == list(range(56, 72))
    for name in ("unigram", "bigram"):
        assert actual[name]["nll"] == pytest.approx(expected[name]["nll"], abs=1e-12)
        assert actual[name]["accuracy"] == expected[name]["accuracy"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_midpoint_baseline_is_not_sequential_prefix_or_twice_shuffled_prefix(study, corpus, seed):
    raw, dataset = corpus
    actual = study.exposure_baseline(raw, dataset, seed, 256)
    order = list(range(1024))
    random.Random(seed).shuffle(order)
    for wrong_indices in (list(range(1024)) + list(range(256)), order[:256] * 2):
        wrong = exposure_oracle(raw, dataset, wrong_indices, range(56, 72))
        assert abs(actual["bigram"]["nll"] - wrong["bigram"]["nll"]) > 1e-6


def test_one_pass_comparator_and_final_replay_are_order_independent(study, corpus):
    raw, dataset = corpus
    one = study.one_pass_baseline(raw, dataset)
    expected = exposure_oracle(raw, dataset, list(range(1024)), range(56, 72))
    for name in ("unigram", "bigram"):
        assert one[name]["nll"] == pytest.approx(expected[name]["nll"], abs=1e-12)
        assert one[name]["accuracy"] == expected[name]["accuracy"]
    final = [study.exposure_baseline(raw, dataset, seed, 1024) for seed in study.SEEDS]
    for name in ("unigram", "bigram"):
        assert final[0][name] == final[1][name] == final[2][name]
    assert final[0]["supervised_exposures"] == 262144


@pytest.mark.parametrize("seed,step", [(True, 256), (3, 256), (0, True), (0, 0.0), (0, 255), (0, 1025)])
def test_baseline_rejects_undeclared_seed_and_step(study, corpus, seed, step):
    with pytest.raises(ValueError):
        study.exposure_baseline(*corpus, seed, step)


def test_fitting_ignores_old_validation_test_and_unused_train_suffix(study, corpus):
    raw, dataset = corpus
    changed = bytearray(raw)
    train = dataset["splits"]["train"]
    stop = train["start"] + 1024 * 129
    changed[stop:train["stop"]] = b"!" * (train["stop"] - stop)
    validation = dataset["splits"]["validation"]["start"]
    changed[validation:validation + 56 * 129] = b"?" * (56 * 129)
    test = dataset["splits"]["test"]["start"]
    changed[test:] = b"~" * (len(changed) - test)
    for seed in study.SEEDS:
        assert study.exposure_baseline(bytes(changed), dataset, seed, 256) == \
            study.exposure_baseline(raw, dataset, seed, 256)
    assert study.one_pass_baseline(bytes(changed), dataset) == study.one_pass_baseline(raw, dataset)


def rewrite_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def state_payload(study, shape, weights, remainders, *, step=0, k=18, zshift=2, loss_kind=1):
    """Build the documented binary ABI independently of export_state."""
    digest = study.optimizer.source_digest(study.optimizer.source_identity())
    identity = [int(digest[index:index + 8], 16) for index in range(0, 64, 8)]
    identity = [value if value < 2 ** 31 else value - 2 ** 32 for value in identity]
    header = [study.optimizer.MAGIC, 1, shape.ML_D, shape.ML_BUFFER_WORDS,
              sum(map(len, weights.values())), k, loss_kind, zshift, step, 0,
              shape.WORDS_CE_LUT, 0, *identity]
    values = header + [round(65536 * 2.0 ** (-index / 16.0)) for index in range(256)]
    for section in (weights, remainders):
        for name in pilot.WEIGHTS:
            values.extend(section[name])
    return study.optimizer._pack(values)


@pytest.fixture
def fake_worker(study, tmp_path, monkeypatch, corpus):
    """Real state/receipt formats, synthetic small corpus, no native process."""
    monkeypatch.setattr(study, "WIDTH", 16)
    monkeypatch.setattr(study, "STEPS", 4)
    monkeypatch.setattr(study, "CHECKPOINTS", (0, 2, 4))
    monkeypatch.setattr(study, "VALIDATION", (56,))
    raw, dataset = corpus
    corpus_path = Path(dataset["path"])
    shape = pilot.layout(16)
    controls = SimpleNamespace(train_error=None, close_error=None, evaluate_mutation=False,
                               import_mutation=False, wrong_native=False, wrong_step=False,
                               initial_mutation=False, opened=[], calls=[], deep_checks=[])
    weights = {name: [3] * getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS}
    remainders = {name: [0] * len(values) for name, values in weights.items()}
    initial_payload = state_payload(study, shape, weights, remainders)

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
        assert use_context is False
        result = observed(weights, tokens, targets, layout, zshift)
        result["lut_normalized_nll"] += (3 - weights["W_HEAD"][0]) / 20
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
            self.weights = {name: list(initial[getattr(layout, "OFF_" + name):
                                              getattr(layout, "OFF_" + name) + getattr(layout, "WORDS_" + name)])
                            for name in pilot.WEIGHTS}
            self.remainders = {name: [0] * len(values) for name, values in self.weights.items()}
            self.identity = {"schema": study.optimizer.SCHEMA,
                "executable_sha256": pilot.digest(executable.read_bytes()),
                "build_sha256": pilot.digest((executable.parent / "build.json").read_bytes()),
                "source_hashes": study.optimizer.source_identity(), "layout": pilot.layout_values(layout),
                "cpu_only": True, "existing_afi_compatible": False}
            self.closed = False
            if controls.initial_mutation:
                self.remainders["EMBED"][0] = 1
            log_path.write_text("unit fake native, no process\n")
            controls.opened.append(self)
            assert timeout_s == 60

        def export_state(self):
            return state_payload(study, self.shape, self.weights, self.remainders,
                                 step=self.step, k=self.k)

        def import_state(self, payload):
            state = study.optimizer.decode_state(payload, self.shape)
            self.weights, self.remainders = state["weights"], state["remainders"]
            self.step, self.k = state["step"], state["k"]
            if controls.import_mutation:
                self.remainders["EMBED"][0] += 1

        def evaluate(self, tokens, targets):
            result = observed(self.weights, tokens, targets, self.shape, 2)
            result["step"] = False if controls.wrong_step else self.step
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
            return {"step": self.step, **observed(self.weights, tokens, targets, self.shape, 2)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True
            if controls.close_error:
                raise controls.close_error("injected cleanup failure")

    monkeypatch.setattr(study, "observe", observed)
    monkeypatch.setattr(study.pilot, "integer_forward", disabled)
    monkeypatch.setattr(study.optimizer, "EFSGD", Native)
    monkeypatch.setattr(study.optimizer, "build", build)
    history = tmp_path / "prior"
    history.mkdir()
    old_executable = build(history / "build", 16)
    old_manifest = {"schema": study.prior.SCHEMA, "recipe": study.prior.recipe(), "dataset": dataset,
                    "source_hashes": study.prior.source_identity(),
                    "plan_sha256": pilot.digest(study.PLAN.read_bytes()),
                    "executable": {"path": str(old_executable), "sha256": pilot.digest(old_executable.read_bytes())}}
    pilot.write_json(history / "manifest.json", old_manifest)
    cells = []
    for arm in study.prior.ARMS:
        for k in study.prior.K_VALUES:
            directory = history / ("%s_k%d" % (arm, k))
            directory.mkdir()
            cell = {"arm": arm, "k": k, "complete": True, "evaluations": []}
            pilot.write_json(directory / "summary.json", cell)
            cells.append(cell)
    (history / "count_fitted_k18/state_000000.i32").write_bytes(initial_payload)
    old_summary = {"schema": study.prior.SCHEMA, "complete": True, "declared_cells": 4,
                   "completed_cells": 4, "failures": [], "heldout_test_evaluated": False,
                   "physical_fpga_execution": False, "speed_advantage_claim": False,
                   "three_seed_quality_gate_evaluated": False, "larger_paid_learning_benchmark_gate": False,
                   "recipe": study.prior.recipe(), "source_hashes": study.prior.source_identity(), "cells": cells}
    pilot.write_json(history / "summary.json", old_summary)
    artifact = study.prior._artifacts(history)
    pilot.write_json(history / "artifact_manifest.json", artifact)
    old_record = tmp_path / "prior-record.json"
    pilot.write_json(old_record, {**old_summary, "run_directory": str(history),
        "artifact_manifest_sha256": pilot.digest((history / "artifact_manifest.json").read_bytes()),
        "artifact_files_verified": len(artifact["files"]),
        "artifact_payload_bytes": sum(row["bytes"] for row in artifact["files"]),
        "full_optimizer_checkpoints_verified": 12, "checkpoint_validation_windows": 192,
        "distinct_validation_positions": 2048, "new_native_updates": 4096,
        "gradient_pair_exposures": 524288, "host_fitting_pair_exposures": 262144,
        "distinct_training_pairs": 131072, "native_training_trajectory_independently_replayed": False})

    def old_verify(directory, manifest, arm, k):
        controls.deep_checks.append((arm, k))
        return study._read(directory / "summary.json")

    monkeypatch.setattr(study.prior, "verify_worker", old_verify)
    binding = study.bind_prior(history, old_record, dataset, deep=False)
    parent = tmp_path / "confirmation"
    parent.mkdir()
    executable = build(parent / "build", 16)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "dataset": dataset,
                "source_hashes": study.source_identity(), "plan_sha256": pilot.digest(study.PLAN.read_bytes()),
                "executable": {"path": str(executable), "sha256": pilot.digest(executable.read_bytes())},
                "prior_evidence": binding}
    pilot.write_json(parent / "manifest.json", manifest)
    config = SimpleNamespace(parent=parent, corpus=corpus_path, output=parent / "seed_0", seed=0,
                             prior_run=history, prior_record=old_record)
    return SimpleNamespace(config=config, controls=controls, manifest=manifest, shape=shape, raw=raw,
                           parent=parent, executable=executable, history=history, old_record=old_record,
                           binding=binding, initial_payload=initial_payload)


def test_prior_binding_checks_all_four_old_cells_and_exact_full_initial_state(study, fake_worker):
    case = fake_worker
    bound = study.bind_prior(case.history, case.old_record, case.manifest["dataset"], deep=True)
    assert bound == case.binding
    assert case.controls.deep_checks == [(arm, k) for arm in study.prior.ARMS for k in study.prior.K_VALUES]
    assert study.read_initial_state(bound, case.shape) == case.initial_payload
    buffer = study.initial_buffer(case.shape, bound)
    state = study.optimizer.decode_state(case.initial_payload, case.shape)
    for name in pilot.WEIGHTS:
        offset, count = getattr(case.shape, "OFF_" + name), getattr(case.shape, "WORDS_" + name)
        assert buffer[offset:offset + count] == state["weights"][name]
    assert buffer[case.shape.OFF_CFG:case.shape.OFF_CFG + 2] == [1, 2]


@pytest.mark.parametrize("mutation", ["payload", "record", "summary", "source", "corpus", "extra_failure"])
def test_prior_binding_rejects_actual_payload_receipt_and_identity_changes(study, fake_worker, monkeypatch, mutation):
    case = fake_worker
    dataset = case.manifest["dataset"]
    if mutation == "payload":
        (case.history / "count_fitted_k18/state_000000.i32").write_bytes(b"changed")
    elif mutation == "record":
        receipt = study._read(case.old_record)
        receipt["native_training_trajectory_independently_replayed"] = True
        rewrite_json(case.old_record, receipt)
    elif mutation == "summary":
        path = case.history / "summary.json"
        summary = study._read(path)
        summary["complete"] = 1
        rewrite_json(path, summary)
    elif mutation == "source":
        monkeypatch.setattr(study.prior, "source_identity", lambda: {})
    elif mutation == "corpus":
        dataset = {**dataset, "sha256": "0" * 64}
    else:
        pilot.write_json(case.history / "failure.json", {"complete": False})
    with pytest.raises(ValueError):
        study.bind_prior(case.history, case.old_record, dataset, deep=False)


@pytest.mark.parametrize("change", ["step", "k", "zshift", "loss_kind", "remainder", "source"])
def test_even_rehashed_initial_state_must_be_zero_step_and_remainders_with_exact_config(study, fake_worker, change):
    case = fake_worker
    values = study.optimizer._unpack(case.initial_payload)
    positions = {"step": 8, "k": 5, "loss_kind": 6, "zshift": 7, "source": 12}
    if change == "remainder":
        index = study.optimizer.STATE_HEADER + case.shape.WORDS_CE_LUT + \
            sum(getattr(case.shape, "WORDS_" + name) for name in pilot.WEIGHTS)
    else:
        index = positions[change]
    values[index] += 1
    payload = study.optimizer._pack(values)
    path = Path(case.binding["initial_state"]["file"])
    path.write_bytes(payload)
    binding = {**case.binding, "initial_state": {"file": str(path), "bytes": len(payload), "sha256": pilot.digest(payload)}}
    with pytest.raises(ValueError):
        study.read_initial_state(binding, case.shape)


def test_initial_binding_rejects_file_alias_and_nonliteral_byte_count(study, fake_worker, tmp_path):
    case = fake_worker
    alias = tmp_path / "state-alias.i32"
    alias.symlink_to(case.binding["initial_state"]["file"])
    for changed in ({"file": str(alias)}, {"bytes": float(len(case.initial_payload))}):
        binding = {**case.binding, "initial_state": {**case.binding["initial_state"], **changed}}
        with pytest.raises(ValueError):
            study.read_initial_state(binding, case.shape)


def test_prior_binding_rehashes_payload_tree_after_deep_checks(study, fake_worker, monkeypatch):
    original = study.prior.verify_worker

    def mutate_after_verifying(directory, manifest, arm, k):
        result = original(directory, manifest, arm, k)
        if (arm, k) == ("count_fitted", 20):
            (fake_worker.history / "late-artifact.txt").write_text("changed during verification")
        return result

    monkeypatch.setattr(study.prior, "verify_worker", mutate_after_verifying)
    with pytest.raises(ValueError):
        study.bind_prior(fake_worker.history, fake_worker.old_record, fake_worker.manifest["dataset"], deep=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_worker_uses_exact_prior_state_then_actual_shuffled_windows(study, fake_worker, seed):
    case = fake_worker
    case.config.seed, case.config.output = seed, case.parent / ("seed_%d" % seed)
    result = study.run_worker(case.config)
    assert result == study.verify_worker(case.config.output, case.manifest, seed)
    assert result["complete"] is True and case.controls.opened[0].closed
    assert result["initial_full_state_equal_prior"] is True
    assert result["gradient_pair_exposures"] == result["inherited_host_fitting_pair_exposures"] == 512
    assert result["new_host_fitting_pair_exposures"] == 0
    assert result["initialization_refitted"] is False
    assert result["model_history_pair_exposures"] == 1024
    assert result["distinct_training_pairs"] == 512
    assert result["paid_execution_authorized"] is False
    assert result["physical_fpga_execution"] is False
    expected_order = list(range(4))
    random.Random(seed).shuffle(expected_order)
    assert result["training_order"] == expected_order
    assert [row["step"] for row in result["evaluations"]] == [0, 2, 4]
    assert (case.config.output / "state_000000.i32").read_bytes() == case.initial_payload
    assert len(case.controls.calls) == 4
    for actual, index in zip(case.controls.calls, expected_order):
        expected = pilot.window(case.raw, case.manifest["dataset"], "train", index)
        assert actual == (list(expected[0]), list(expected[1]))
    for step in (0, 2, 4):
        state = study.optimizer.decode_state((case.config.output / ("state_%06d.i32" % step)).read_bytes(), case.shape)
        assert state["step"] == step and state["k"] == 18
        assert state["remainders"]["EMBED"][0] == step
    stronger = min(result["baselines_by_step"]["4"]["bigram"]["nll"],
                   result["one_pass_baseline"]["bigram"]["nll"])
    assert result["conservative_bigram_advantage_nats"] == stronger - result["evaluations"][-1]["lut_normalized_nll"]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_retains_sealed_exclusive_failure_and_closes_native(study, fake_worker, failure):
    case = fake_worker
    case.controls.train_error = failure
    with pytest.raises(failure):
        study.run_worker(case.config)
    root = case.config.output
    assert study._read(root / "failure.json")["type"] == failure.__name__
    assert not (root / "summary.json").exists()
    assert case.controls.opened[0].closed
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)
    before = (root / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        study.run_worker(case.config)
    assert (root / "failure.json").read_bytes() == before


@pytest.mark.parametrize("field", ["evaluate_mutation", "import_mutation", "wrong_native", "wrong_step", "initial_mutation"])
def test_native_full_state_prediction_and_literal_step_mismatches_fail_closed(study, fake_worker, field):
    case = fake_worker
    setattr(case.controls, field, True)
    with pytest.raises(ValueError):
        study.run_worker(case.config)
    assert (case.config.output / "failure.json").exists()
    assert not (case.config.output / "summary.json").exists()
    assert case.controls.opened[0].closed


def test_success_is_written_only_after_native_cleanup(study, fake_worker):
    fake_worker.controls.close_error = RuntimeError
    with pytest.raises(RuntimeError, match="cleanup"):
        study.run_worker(fake_worker.config)
    assert (fake_worker.config.output / "failure.json").exists()
    assert not (fake_worker.config.output / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["source", "corpus", "executable", "prior"])
def test_input_changes_before_worker_start_cannot_create_output(study, fake_worker, monkeypatch, mutation):
    case = fake_worker
    if mutation == "source":
        monkeypatch.setattr(study, "source_identity", lambda: {})
    elif mutation == "corpus":
        case.config.corpus.write_bytes(b"changed")
    elif mutation == "executable":
        case.executable.write_bytes(b"changed")
    else:
        (case.history / "count_fitted_k18/state_000000.i32").write_bytes(b"changed")
    with pytest.raises(ValueError):
        study.run_worker(case.config)
    assert not case.config.output.exists() and not case.controls.opened


def reseal(study, root):
    rewrite_json(root / "artifact_manifest.json", study._artifacts(root))


@pytest.mark.parametrize("field,value", [
    ("schema", "wrong"), ("step", False), ("step", 0.0), ("k", 20), ("config", {}),
    ("identity", {}), ("regions", []), ("format", "weights only"),
    ("file", "../state_000000.i32"), ("bytes", True),
])
def test_resealed_checkpoint_metadata_cannot_replace_full_state_or_identity(study, fake_worker, field, value):
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
        study.verify_worker(root, case.manifest, 0)


@pytest.mark.parametrize("field,value", [
    ("window_index", 40), ("window_index", 56.0), ("native_reference_equal", 1),
    ("native_routes_equal", False), ("native_arrays", {}),
])
def test_resealed_validation_rows_cannot_substitute_old_cohort_or_loose_receipts(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    row = study._read(root / "eval_000000_056.json")
    row[field] = value
    rewrite_json(root / "eval_000000_056.json", row)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["per_window"][0] = row
    rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, 0)


@pytest.mark.parametrize("part", ["state", "native", "training", "baseline", "summary", "failure", "order"])
def test_resealed_scientific_payload_or_success_claim_forgery_fails(study, fake_worker, part):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    if part in ("state", "native", "training"):
        filename = {"state": "state_000002.i32", "native": "eval_000002_056_gradients.i32",
                    "training": "training_predictions.i32"}[part]
        path = root / filename
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    elif part == "failure":
        rewrite_json(root / "failure.json", {"complete": False})
    elif part == "order":
        path = root / "training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["window_index"] = (rows[0]["window_index"] + 1) % 4
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        summary = study._read(root / "summary.json")
        if part == "baseline":
            summary["baselines_by_step"]["2"]["bigram"]["nll"] = 0
        else:
            summary["paid_execution_authorized"] = True
        rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, 0)


@pytest.mark.parametrize("section", ["weights", "remainders"])
def test_even_rehashed_initial_optimizer_state_must_equal_prior_bytes(study, fake_worker, section):
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
    with pytest.raises(ValueError, match="initial state"):
        study.verify_worker(root, case.manifest, 0)


@pytest.mark.parametrize("saturation", [1, True, False, 0.0])
def test_clipping_or_bool_saturation_cannot_pass_even_with_matching_oracle(study, fake_worker, monkeypatch, saturation):
    original = study.observe

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["saturation_events"] = saturation
        return result

    monkeypatch.setattr(study, "observe", observed)
    with pytest.raises(ValueError):
        study.run_worker(fake_worker.config)


def test_worker_cannot_write_any_other_child_or_escape_parent(study, fake_worker, tmp_path):
    for output in (fake_worker.parent / "seed_1", tmp_path / "unowned"):
        fake_worker.config.output = output
        with pytest.raises(ValueError, match="exact"):
            study.run_worker(fake_worker.config)
        assert not output.exists()


def passing_gate_cells():
    rows = []
    for seed in (0, 1, 2):
        initial = {"lut_normalized_nll": 3.0, "context_disabled_nll": 3.0}
        final = {"lut_normalized_nll": 2.0, "context_disabled_nll": 2.02,
                 "saturation_events": 0, "native_reference_equal": True, "native_routes_equal": True,
                 "route_diagnostics": {"positions": 2048, "nonzero_logit_gradient_positions": 2048,
                     "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 512,
                     "head_winners_ge16_positions": 1536, "backward_int32_overflow_detected": False}}
        rows.append({"seed": seed, "complete": True, "evaluations": [initial, final],
                     "conservative_comparators": {"unigram": {"nll": 2.05}, "bigram": {"nll": 2.02}}})
    return rows


def test_gate_retains_exact_all_seed_thresholds_and_never_authorizes_paid_execution(study):
    result = study.gate_results(passing_gate_cells())
    assert result["mechanism"] is result["context"] is result["bigram"] is True
    assert result["larger_paid_learning_benchmark_gate"] is True
    assert result["paid_execution_authorized"] is False
    assert result["statistical_significance_claim"] is False
    assert "order replicates" in result["seed_meaning"]


@pytest.mark.parametrize("failure", ["cancellation", "diversity", "context", "growth", "unigram", "bigram_mean", "bigram_worst"])
def test_one_bad_seed_or_mean_does_not_get_promoted_by_other_seed_success(study, failure):
    rows = passing_gate_cells()
    final = rows[2]["evaluations"][-1]
    routes = final["route_diagnostics"]
    if failure == "cancellation":
        routes["exact_zero_head_input_vjp_on_nonzero_gradient_positions"] = 513
    elif failure == "diversity":
        routes["head_winners_ge16_positions"] = 1535
    elif failure == "context":
        final["context_disabled_nll"] = 2.019
    elif failure == "growth":
        rows[2]["evaluations"][0]["context_disabled_nll"] = 3.001
    elif failure == "unigram":
        rows[2]["conservative_comparators"]["unigram"]["nll"] = 2.049
    elif failure == "bigram_mean":
        rows[2]["conservative_comparators"]["bigram"]["nll"] = 2.019
    else:
        for row in rows[:2]:
            row["conservative_comparators"]["bigram"]["nll"] = 2.1
        rows[2]["conservative_comparators"]["bigram"]["nll"] = 1.979
    result = study.gate_results(rows)
    assert result["larger_paid_learning_benchmark_gate"] is False
    assert result["paid_execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bool_seed", "incomplete"])
def test_gate_requires_all_three_complete_literal_order_seeds(study, mutation):
    rows = passing_gate_cells()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1]["seed"] = 0
    elif mutation == "bool_seed":
        rows[0]["seed"] = False
    else:
        rows[-1]["complete"] = 1
    with pytest.raises(ValueError):
        study.gate_results(rows)


def fake_parent_run(study, case, monkeypatch, output, failure_index=None):
    """Exercise actual parent bookkeeping with in-process fake-native children."""
    launches = []

    def execute(command, path):
        path.write_text("unit subprocess fixture; no native child\n")
        index = len(launches)
        launches.append(command)
        if index == failure_index:
            return {"returncode": 2, "timed_out": False, "timeout_s": 1200, "elapsed_s": 0.1}
        get = lambda flag: command[command.index(flag) + 1]
        study.run_worker(SimpleNamespace(parent=Path(get("--parent")), output=Path(get("--output")),
            corpus=Path(get("--corpus")), prior_run=Path(get("--prior-run")),
            prior_record=Path(get("--prior-record")), seed=int(get("--seed"))))
        return {"returncode": 0, "timed_out": False, "timeout_s": 1200, "elapsed_s": 0.1}

    monkeypatch.setattr(study, "_run_process", execute)
    config = SimpleNamespace(output=output, corpus=case.config.corpus,
                             prior_run=case.history, prior_record=case.old_record)
    return study.run(config), launches


@pytest.mark.parametrize("failure_index", [None, 0, 2])
def test_parent_requires_all_order_cells_and_retains_every_failed_receipt(study, fake_worker, tmp_path, monkeypatch, failure_index):
    output = tmp_path / "parent-run"
    result, launches = fake_parent_run(study, fake_worker, monkeypatch, output, failure_index)
    assert len(launches) == 3
    assert result["declared_cells"] == 3
    assert result["completed_cells"] == (3 if failure_index is None else 2)
    assert result["complete"] is (failure_index is None)
    assert result["paid_execution_authorized"] is False
    assert result["physical_fpga_execution"] is False
    assert result["statistical_significance_claim"] is False
    assert len(result["failures"]) == (0 if failure_index is None else 1)
    assert study._read(output / "artifact_manifest.json") == study._artifacts(output)
    if failure_index is not None:
        assert result["gates"] is None
        with pytest.raises(ValueError):
            study.publish_record(tmp_path / "false-success.json", output, result)
        assert not (tmp_path / "false-success.json").exists()


def test_publication_recomputes_full_coverage_and_distinguishes_inherited_fitting(study, fake_worker, tmp_path, monkeypatch):
    output = tmp_path / "parent-run"
    result, _ = fake_parent_run(study, fake_worker, monkeypatch, output)
    target = tmp_path / "public-record.json"
    record = study.publish_record(target, output, result)
    assert record == study._read(target)
    assert record["full_optimizer_checkpoints_verified"] == 9
    assert record["checkpoint_validation_windows"] == 9
    assert record["distinct_validation_positions"] == 128
    assert record["new_native_updates"] == 12
    assert record["gradient_pair_exposures"] == 1536
    assert record["inherited_host_fitting_pair_exposures"] == 1536
    assert record["new_host_fitting_pair_exposures"] == 0
    assert record["initialization_refitted"] is False
    assert record["model_history_pair_exposures"] == 3072
    assert record["distinct_training_pairs"] == 512
    assert record["native_training_trajectory_independently_replayed"] is False
    assert record["paid_execution_authorized"] is False
    assert record["artifact_manifest_sha256"] == pilot.digest((output / "artifact_manifest.json").read_bytes())
    assert all("per_window" not in evaluation for row in record["cells"] for evaluation in row["evaluations"])
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        study.publish_record(target, output, result)
    assert target.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("returncode", 17), ("returncode", False), ("timed_out", True), ("timed_out", 0),
    ("timeout_s", 1200.0), ("elapsed_s", -1), ("elapsed_s", True), ("command", ["wrong-command"]),
])
def test_resealed_child_execution_receipts_cannot_publish_ambiguous_success(study, fake_worker, tmp_path, monkeypatch, field, value):
    output = tmp_path / "parent-run"
    result, _ = fake_parent_run(study, fake_worker, monkeypatch, output)
    path = output / "seed_1.execution.json"
    receipt = study._read(path)
    receipt[field] = value
    rewrite_json(path, receipt)
    reseal(study, output)
    target = tmp_path / "false-success.json"
    with pytest.raises(ValueError):
        study.publish_record(target, output, result)
    assert not target.exists()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_parent_build_failure_retains_exclusive_sealed_failure(study, fake_worker, tmp_path, monkeypatch, failure):
    output = tmp_path / "failed-parent"

    def fail(*args, **kwargs):
        raise failure("injected build failure")

    monkeypatch.setattr(study.optimizer, "build", fail)
    config = SimpleNamespace(output=output, corpus=fake_worker.config.corpus,
                             prior_run=fake_worker.history, prior_record=fake_worker.old_record)
    with pytest.raises(failure):
        study.run(config)
    assert study._read(output / "failure.json")["type"] == failure.__name__
    assert study._read(output / "artifact_manifest.json") == study._artifacts(output)
    before = (output / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        study.run(config)
    assert (output / "failure.json").read_bytes() == before


def test_existing_or_nested_outputs_and_record_aliases_cannot_modify_history(study, fake_worker, tmp_path, monkeypatch):
    monkeypatch.setattr(study.optimizer, "build", lambda *a, **k: pytest.fail("must not build"))
    history_alias = tmp_path / "history-alias"
    history_alias.symlink_to(fake_worker.history, target_is_directory=True)
    for directory in (fake_worker.history, history_alias):
        output = directory / "nested-confirmation"
        config = SimpleNamespace(output=output, corpus=fake_worker.config.corpus,
                                 prior_run=fake_worker.history, prior_record=fake_worker.old_record)
        with pytest.raises(ValueError, match="nested"):
            study.run(config)
        assert not output.exists()


def test_cli_existing_record_or_worker_publication_is_rejected_before_any_execution(study, fake_worker, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "run", lambda *a: pytest.fail("must not run"))
    monkeypatch.setattr(study, "run_worker", lambda *a: pytest.fail("must not run worker"))
    base = ["--corpus", str(fake_worker.config.corpus), "--prior-run", str(fake_worker.history),
            "--prior-record", str(fake_worker.old_record), "--output", str(tmp_path / "new-parent")]
    before = fake_worker.old_record.read_bytes()
    with pytest.raises(FileExistsError):
        study.main([*base, "--record", str(fake_worker.old_record)])
    with pytest.raises(SystemExit):
        study.main([*base, "--worker", "--parent", str(fake_worker.parent), "--seed", "0",
                    "--record", str(tmp_path / "new-publication.json")])
    assert fake_worker.old_record.read_bytes() == before


@pytest.mark.parametrize("resealed", [False, True])
def test_publication_final_guard_catches_payload_and_seal_changes_during_verification(study, fake_worker, tmp_path, monkeypatch, resealed):
    output = tmp_path / "parent-run"
    result, _ = fake_parent_run(study, fake_worker, monkeypatch, output)
    original_artifacts, original_frozen = study._artifacts, study._frozen
    seen_parent = {"value": False}

    def artifacts(path):
        value = original_artifacts(path)
        if Path(path).resolve() == output.resolve():
            seen_parent["value"] = True
        return value

    def frozen(manifest):
        original_frozen(manifest)
        if seen_parent["value"]:
            (output / "late-artifact.txt").write_text("changed after initial publication verification")
            if resealed:
                rewrite_json(output / "artifact_manifest.json", original_artifacts(output))

    monkeypatch.setattr(study, "_artifacts", artifacts)
    monkeypatch.setattr(study, "_frozen", frozen)
    target = tmp_path / "false-success.json"
    with pytest.raises(ValueError):
        study.publish_record(target, output, result)
    assert seen_parent["value"] and not target.exists()
