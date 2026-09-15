"""Offline contracts for the narrow, matched warm-start SignSGD control.

These tests use synthetic data and injected native transports. They do not
launch scientific training, execute an FPGA workload, or contact AWS.
"""

import importlib
import json
import math
import random
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fpga_lm_pretrain as pilot


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.compare_fpga_lm_warm_signsgd")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(73116)
    raw = bytes(rng.randrange(128) for _ in range(200_000))
    path = tmp_path / "corpus.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 1024, 56)
    return raw, dataset


def exposure_oracle(raw, dataset, indices, validation):
    """Count only the explicitly exposed, within-window input/target pairs."""
    uni = [1] * 128
    bi = [[1] * 128 for _ in range(128)]
    for index in indices:
        start = dataset["splits"]["train"]["start"] + index * 129
        for offset in range(start, start + 128):
            token, target = raw[offset], raw[offset + 1]
            uni[target] += 1
            bi[token][target] += 1
    result = {}
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
        result[name] = {"nll": sum(losses) / len(losses), "accuracy": correct / len(losses)}
    return result


def test_declared_control_changes_optimizer_not_start_order_or_cohort(study):
    assert (study.WIDTH, study.ZSHIFT, study.LR, study.THRESHOLD, study.STEPS) == \
        (512, 2, 1, 2048, 1024)
    assert study.CHECKPOINTS == (0, 256, 1024)
    assert study.VALIDATION == tuple(range(40, 56))
    assert study.WORKER_TIMEOUT_S == 1200


def test_source_identity_covers_own_and_all_reused_frozen_sources(study):
    sources = study.source_identity()
    for name, digest in pilot.source_identity().items():
        assert sources[name] == digest
    own = Path(study.__file__).resolve()
    assert sources[own.relative_to(pilot.REPO).as_posix()] == pilot.digest(own.read_bytes())
    for name in ("confirm_fpga_lm_efsgd.py", "train_fpga_lm_efsgd.py",
                 "check_fpga_lm_bigram_capacity.py", "observe_fpga_lm_routes.py",
                 "fpga_lm_efsgd_v1.cpp", "fpga_lm_efsgd.py"):
        assert any(Path(path).name == name for path in sources)


def candidate_inputs(shape):
    weights = {name: [7] * getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS}
    gradients = {name: [0] * len(values) for name, values in weights.items()}
    return weights, gradients


def test_sign_oracle_matches_strict_threshold_and_preserves_input_arrays(study):
    shape = pilot.layout(16)
    weights, gradients = candidate_inputs(shape)
    values = [-2147483647, -2049, -2048, -2047, -1, 0, 1, 2047, 2048, 2049, 2147483647]
    for name in pilot.WEIGHTS:
        gradients[name][:len(values)] = values
    expected = {name: [value - (1 if grad > 2048 else -1 if grad < -2048 else 0)
                       for value, grad in zip(weights[name], gradients[name])]
                for name in pilot.WEIGHTS}
    actual = study.signsgd_candidate(weights, gradients, shape)
    assert actual == expected
    assert all(set(value) == {7} for value in weights.values())
    assert all(value[:len(values)] == values for value in gradients.values())
    actual["EMBED"][0] = -3
    assert weights["EMBED"][0] == 7


@pytest.mark.parametrize("weight,gradient", [(32767, -2049), (-32768, 2049)])
def test_sign_oracle_rejects_clipping_before_publishing_any_candidate(study, weight, gradient):
    shape = pilot.layout(16)
    weights, gradients = candidate_inputs(shape)
    weights["W_HEAD"][-1] = weight
    gradients["W_HEAD"][-1] = gradient
    gradients["EMBED"][0] = 9999
    with pytest.raises(ValueError):
        study.signsgd_candidate(weights, gradients, shape)
    assert weights["EMBED"][0] == 7
    assert weights["W_HEAD"][-1] == weight


@pytest.mark.parametrize("region", pilot.WEIGHTS)
@pytest.mark.parametrize("which", ["weights", "gradients"])
def test_sign_oracle_requires_every_full_region(study, region, which):
    shape = pilot.layout(16)
    weights, gradients = candidate_inputs(shape)
    (weights if which == "weights" else gradients)[region].pop()
    with pytest.raises(ValueError):
        study.signsgd_candidate(weights, gradients, shape)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, math.nan, math.inf])
@pytest.mark.parametrize("which", ["weights", "gradients"])
def test_sign_oracle_rejects_nonliteral_finite_integer_values(study, value, which):
    shape = pilot.layout(16)
    weights, gradients = candidate_inputs(shape)
    (weights if which == "weights" else gradients)["EMBED"][0] = value
    with pytest.raises(ValueError):
        study.signsgd_candidate(weights, gradients, shape)


@pytest.mark.parametrize("which,value", [("weights", -32769), ("weights", 32768),
                                         ("gradients", -2147483649), ("gradients", 2147483648)])
def test_sign_oracle_checks_storage_bounds_before_arithmetic(study, which, value):
    shape = pilot.layout(16)
    weights, gradients = candidate_inputs(shape)
    (weights if which == "weights" else gradients)["EMBED"][0] = value
    with pytest.raises(ValueError):
        study.signsgd_candidate(weights, gradients, shape)


def test_sign_oracle_rejects_native_abs_int_min_undefined_behavior(study):
    shape = pilot.layout(16)
    weights, gradients = candidate_inputs(shape)
    gradients["MP"][-1] = -2147483648
    with pytest.raises(ValueError, match="INT_MIN"):
        study.signsgd_candidate(weights, gradients, shape)


@pytest.mark.parametrize("step", [0, 256, 1024])
def test_baselines_include_inherited_fit_and_exact_sequential_prefix(study, corpus, step):
    raw, dataset = corpus
    expected = exposure_oracle(raw, dataset, list(range(1024)) + list(range(step)), range(40, 56))
    actual = study.exposure_baseline(raw, dataset, step)
    assert actual["supervised_exposures"] == (1024 + step) * 128
    assert actual["distinct_supervised_pairs"] == 131072
    assert actual["alpha"] == 1
    for name in ("unigram", "bigram"):
        assert actual[name]["nll"] == pytest.approx(expected[name]["nll"], abs=1e-12)
        assert actual[name]["accuracy"] == expected[name]["accuracy"]


@pytest.mark.parametrize("step", [True, False, 0.0, 256.0, -1, 255, 1025, "256"])
def test_baselines_reject_undeclared_and_nonliteral_checkpoint(study, corpus, step):
    with pytest.raises(ValueError):
        study.exposure_baseline(*corpus, step)


def test_new_receipt_distinguishes_inherited_fit_from_new_execution(study):
    recipe = study.recipe()
    assert recipe["training_indices"] == list(range(1024))
    assert recipe["declared_cells"] == 1
    assert recipe["inherited_host_fitting_pair_exposures"] == 131072
    assert recipe["new_host_fitting_pair_exposures"] == 0
    assert recipe["initialization_refitted"] is False
    assert recipe["new_gradient_pair_exposures"] == 131072
    assert recipe["model_history_pair_exposures"] == 262144
    assert recipe["distinct_training_pairs"] == 131072


def rewrite_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def reseal(study, root):
    rewrite_json(root / "artifact_manifest.json", study._artifacts(root))


def state_payload(study, shape, weights):
    optimizer = study.prior.optimizer
    digest = optimizer.source_digest(optimizer.source_identity())
    identity = [int(digest[index:index + 8], 16) for index in range(0, 64, 8)]
    identity = [value if value < 2 ** 31 else value - 2 ** 32 for value in identity]
    header = [optimizer.MAGIC, 1, shape.ML_D, shape.ML_BUFFER_WORDS,
              sum(map(len, weights.values())), 18, 1, 2, 0, 0, shape.WORDS_CE_LUT, 0, *identity]
    values = header + [round(65536 * 2.0 ** (-index / 16.0)) for index in range(256)]
    for name in pilot.WEIGHTS:
        values.extend(weights[name])
    values.extend([0] * sum(map(len, weights.values())))
    return optimizer._pack(values)


@pytest.fixture
def fake_worker(study, tmp_path, monkeypatch, corpus):
    """Actual serialization and provenance checks; no native execution."""
    for module in (study, study.prior):
        monkeypatch.setattr(module, "WIDTH", 16)
        monkeypatch.setattr(module, "STEPS", 4)
        monkeypatch.setattr(module, "CHECKPOINTS", (0, 2, 4))
        monkeypatch.setattr(module, "VALIDATION", (40,))
    monkeypatch.setattr(study.confirmation, "WIDTH", 16)
    raw, dataset = corpus
    shape = pilot.layout(16)
    controls = SimpleNamespace(opened=[], calls=[], deep_checks=[], train_error=None,
        close_error=None, initial_mutation=False, wrong_native=False, wrong_routes=False,
        wrong_update=False, evaluation_mutation=False, saturation=0, overflow=False)
    weights = {name: [3] * getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS}
    initial_payload = state_payload(study, shape, weights)

    def observed(weights, tokens, targets, layout, zshift, include_parameter_gradients=False):
        assert zshift == 2
        value = weights["W_HEAD"][0]
        routes = {name: [0] * getattr(layout, "WORDS_" + name)
                  for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}
        result = {"predictions": [int(token) for token in tokens], "gradients": [0] * 16384,
            "native_routes": routes, "lut_normalized_nll": 4.0 + value / 100,
            "accuracy": 0.25, "saturation_events": controls.saturation, "rounded_zero_target_mass": 0,
            "route_diagnostics": {"positions": 128, "nonzero_logit_gradient_positions": 128,
                "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
                "head_winners_ge16_positions": 128, "single_shared_head_winner_positions": 0,
                "nonzero_embedding_input_vjp_positions": 128, "max_abs_head_input_vjp": 3,
                "max_abs_embedding_input_vjp": 3, "backward_int32_overflow_detected": controls.overflow,
                "head_winner_counts": [16] * 128}}
        if include_parameter_gradients:
            result["parameter_gradients"] = {name: [0] * len(values) for name, values in weights.items()}
            result["parameter_gradients"]["W_HEAD"][0] = 4096
        return result

    def disabled(weights, tokens, targets, layout, zshift, use_context):
        assert use_context is False
        result = observed(weights, tokens, targets, layout, zshift)
        result["lut_normalized_nll"] += (3 - weights["W_HEAD"][0]) / 20
        return result

    def build(directory, width):
        assert width == 16
        directory.mkdir(parents=True, exist_ok=False)
        library = directory / "libmorphlm.so"
        library.write_bytes(b"offline fake library; never loaded")
        pilot.write_json(directory / "build.json", {"command": ["unit-only"],
            "source_hashes": pilot.source_identity(), "compiler_version": "offline fixture",
            "library_sha256": pilot.digest(library.read_bytes()), "layout": pilot.layout_values(shape)})
        return library

    class Bridge:
        def __init__(self, library, backend, layout):
            assert backend == "csim" and layout.ML_D == 16
            assert library.read_bytes() == b"offline fake library; never loaded"
            self.buffer = [0] * layout.ML_BUFFER_WORDS
            self.shape, self.closed = layout, False
            controls.opened.append(self)

        def write(self, offset, values):
            self.buffer[offset:offset + len(values)] = values
            if offset == 0 and controls.initial_mutation:
                self.buffer[self.shape.OFF_EMBED] += 1

        def read(self, offset, count):
            return list(self.buffer[offset:offset + count])

        def step(self, lr, threshold):
            assert threshold == 2048
            tokens = self.read(self.shape.OFF_TOKENS, self.shape.WORDS_TOKENS)
            targets = self.read(self.shape.OFF_TARGETS, self.shape.WORDS_TARGETS)
            weights = pilot.read_weights(self, self.shape)
            result = observed(weights, tokens, targets, self.shape, 2)
            if lr:
                controls.calls.append((tokens, targets, lr, threshold))
                if controls.train_error:
                    raise controls.train_error("injected native training failure")
                self.buffer[self.shape.OFF_W_HEAD] -= 1
                if controls.wrong_update:
                    self.buffer[self.shape.OFF_EMBED] += 1
            elif controls.evaluation_mutation:
                self.buffer[self.shape.OFF_EMBED] += 1
            if controls.wrong_native:
                result["predictions"][0] ^= 1
            if controls.wrong_routes:
                result["native_routes"]["AD_HEAD"][0] = 1
            self.write(self.shape.OFF_PRED, result["predictions"])
            self.write(self.shape.OFF_GRAD_LOGIT, result["gradients"])
            for name, values in result["native_routes"].items():
                self.write(getattr(self.shape, "OFF_" + name), values)

        def close(self):
            self.closed = True
            if controls.close_error:
                raise controls.close_error("injected cleanup failure")

    monkeypatch.setattr(study, "observe", observed)
    monkeypatch.setattr(pilot, "integer_forward", disabled)
    monkeypatch.setattr(pilot, "build_csim", build)
    monkeypatch.setattr(pilot, "NativeBridge", Bridge)
    history = tmp_path / "prior"
    history.mkdir()
    old_executable = history / "native"
    old_executable.write_bytes(b"offline old native")
    old_manifest = {"schema": study.prior.SCHEMA, "recipe": study.prior.recipe(), "dataset": dataset,
                    "source_hashes": study.prior.source_identity(),
                    "executable": {"path": str(old_executable), "sha256": pilot.digest(old_executable.read_bytes())}}
    pilot.write_json(history / "manifest.json", old_manifest)
    cells = []
    for arm in study.prior.ARMS:
        for k in study.prior.K_VALUES:
            directory = history / ("%s_k%d" % (arm, k))
            directory.mkdir()
            evaluations = [{"step": step, "lut_normalized_nll": 4.03 - step / 50,
                "accuracy": 0.25, "context_disabled_nll": 4.03,
                "saturation_events": 0, "rounded_zero_target_mass": 0,
                "native_reference_equal": True, "native_routes_equal": True, "per_window": []}
                for step in (0, 2, 4)]
            cell = {"arm": arm, "k": k, "complete": True, "evaluations": evaluations}
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
    pilot.write_json(old_record, {**old_summary,
        "cells": [study.confirmation._compact_cell(cell) for cell in cells], "run_directory": str(history),
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
    parent = tmp_path / "warm-control"
    parent.mkdir()
    library = build(parent / "build", 16)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "run_directory": str(parent),
        "dataset": dataset, "layout": pilot.layout_values(shape), "prior_evidence": binding,
        "source_hashes": study.source_identity(), "plan_sha256": pilot.digest(study.PLAN.read_bytes()),
        "library": {"path": str(library), "sha256": pilot.digest(library.read_bytes())},
        "build_record_sha256": pilot.digest((parent / "build/build.json").read_bytes())}
    pilot.write_json(parent / "manifest.json", manifest)
    pilot.write_json(parent / "protocol.json", {"schema": study.SCHEMA, **study.recipe()})
    config = SimpleNamespace(parent=parent, corpus=Path(dataset["path"]), output=parent / "signsgd",
                             prior_run=history, prior_record=old_record)
    return SimpleNamespace(config=config, controls=controls, shape=shape, raw=raw, dataset=dataset,
        binding=binding, manifest=manifest, parent=parent, initial_payload=initial_payload,
        weights=weights, history=history, old_record=old_record, observer=observed, bridge=Bridge)


def test_worker_matches_initial_weights_order_guarded_updates_and_exact_exposures(study, fake_worker):
    case = fake_worker
    result = study.run_worker(case.config, case.bridge, case.observer)
    assert result["complete"] is True
    assert result["initial_weights_equal_prior"] is True
    assert result["inherited_host_fitting_pair_exposures"] == 512
    assert result["new_host_fitting_pair_exposures"] == 0
    assert result["initialization_refitted"] is False
    assert result["new_gradient_pair_exposures"] == 512
    assert result["model_history_pair_exposures"] == 1024
    assert result["training_updates"] == 4
    assert result["training_steps_guarded_and_compared_by_worker"] == 4
    assert result["posthoc_complete_training_trajectory_replayed"] is False
    assert (case.config.output / "checkpoint_000000.i32").read_bytes() == pilot.weight_bytes(case.weights)
    for index, actual in enumerate(case.controls.calls):
        tokens, targets = pilot.window(case.raw, case.dataset, "train", index)
        assert actual == (tokens, targets, 1, 2048)
    assert len(case.controls.calls) == 4 and case.controls.opened[0].closed
    assert [row["step"] for row in result["evaluations"]] == [0, 2, 4]
    for key in ("physical_fpga_execution", "three_seed_quality_gate_evaluated", "speed_advantage_claim",
                "heldout_test_evaluated", "quality_gate_candidate", "paid_execution_authorized",
                "larger_paid_learning_benchmark_gate", "statistical_significance_claim"):
        assert result[key] is False
    assert study._read(case.config.output / "artifact_manifest.json") == study._artifacts(case.config.output)
    assert study.verify_worker(case.config.output, case.manifest) == result


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_retains_exclusive_artifacts_and_closes_transport(study, fake_worker, failure):
    case = fake_worker
    case.controls.train_error = failure
    with pytest.raises(failure):
        study.run_worker(case.config, case.bridge, case.observer)
    assert case.controls.opened[0].closed
    assert not (case.config.output / "summary.json").exists()
    assert study._read(case.config.output / "failure.json")["complete"] is False
    before = study._artifacts(case.config.output)
    with pytest.raises(FileExistsError):
        study.run_worker(case.config, case.bridge, case.observer)
    assert study._artifacts(case.config.output) == before


@pytest.mark.parametrize("kind", ["initial_mutation", "wrong_native", "wrong_routes", "wrong_update",
                                  "evaluation_mutation"])
def test_actual_transport_and_all_weight_checks_fail_closed(study, fake_worker, kind):
    case = fake_worker
    setattr(case.controls, kind, True)
    with pytest.raises(ValueError):
        study.run_worker(case.config, case.bridge, case.observer)
    assert case.controls.opened[0].closed
    assert not (case.config.output / "summary.json").exists()
    assert (case.config.output / "failure.json").exists()


@pytest.mark.parametrize("value", [1, True, False, 0.0])
def test_observed_saturation_requires_literal_zero(study, fake_worker, value):
    case = fake_worker
    case.controls.saturation = value
    with pytest.raises(ValueError):
        study.run_worker(case.config, case.bridge, case.observer)
    assert not case.controls.calls


@pytest.mark.parametrize("value", [True, 0, None])
def test_backward_overflow_guard_requires_literal_false(study, fake_worker, value):
    case = fake_worker
    case.controls.overflow = value
    with pytest.raises(ValueError):
        study.run_worker(case.config, case.bridge, case.observer)
    assert not case.controls.calls


def test_cleanup_failure_cannot_leave_successful_summary(study, fake_worker):
    case = fake_worker
    case.controls.close_error = RuntimeError
    with pytest.raises(RuntimeError, match="cleanup"):
        study.run_worker(case.config, case.bridge, case.observer)
    assert not (case.config.output / "summary.json").exists()
    assert (case.config.output / "failure.json").exists()


@pytest.mark.parametrize("change", ["payload", "record", "source", "corpus", "failure"])
def test_complete_prior_binding_rejects_historical_mutations(study, fake_worker, monkeypatch, change):
    case = fake_worker
    if change == "payload":
        Path(case.binding["initial_state"]["file"]).write_bytes(b"changed")
    elif change == "record":
        record = study._read(case.old_record)
        record["complete"] = 1
        rewrite_json(case.old_record, record)
    elif change == "source":
        monkeypatch.setattr(study.prior, "source_identity", lambda: {})
    elif change == "corpus":
        case.config.corpus.write_bytes(b"changed")
    else:
        pilot.write_json(case.history / "failure.json", {"complete": False})
    with pytest.raises(ValueError):
        study.run_worker(case.config, case.bridge, case.observer)
    assert not case.controls.opened and not case.config.output.exists()


def test_lr0_native_evaluation_does_not_test_a_hypothetical_clipping_lr1_update(study, fake_worker, tmp_path):
    case = fake_worker
    weights = {name: list(values) for name, values in case.weights.items()}
    weights["W_HEAD"][0] = -32768
    tokens, targets = pilot.window(case.raw, case.dataset, "validation", 40)
    gradients = case.observer(weights, tokens, targets, case.shape, 2, True)["parameter_gradients"]
    with pytest.raises(ValueError, match="clip"):
        study.signsgd_candidate(weights, gradients, case.shape)
    study.validate_parameter_gradients(weights, gradients, case.shape)
    bridge = case.bridge(Path(case.manifest["library"]["path"]), "csim", case.shape)
    initial = study.initial_buffer(case.shape, case.binding)
    initial[case.shape.OFF_W_HEAD] = -32768
    bridge.write(0, initial)
    output = tmp_path / "boundary-evaluation"
    output.mkdir()
    row = study.evaluate_cell(bridge, output, 0, weights, case.raw, case.dataset, case.shape, case.observer)
    assert study._verify_evaluation(output, row, case.shape, case.raw, case.dataset) == weights
    assert not case.controls.calls
    assert pilot.read_weights(bridge, case.shape) == weights
    bridge.close()


@pytest.mark.parametrize("field,value", [("step", False), ("bytes", 0.0), ("location_portable", 1),
                                        ("regions", []), ("format", "guessed-format"), ("file", "elsewhere.i32")])
def test_resealed_checkpoint_metadata_must_be_exact_full_layout(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    record = study._read(root / "checkpoint_000000.json")
    record[field] = value
    rewrite_json(root / "checkpoint_000000.json", record)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["checkpoint"] = record
    rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest)


@pytest.mark.parametrize("part", ["checkpoint", "native", "training", "baseline", "summary", "failure",
                                 "initial", "weight_chain", "window_order", "route_digest"])
def test_resealed_corruption_cannot_be_published_as_complete_worker(study, fake_worker, part):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    if part in ("checkpoint", "native", "training"):
        names = {"checkpoint": "checkpoint_000002.i32", "native": "eval_000002_040_gradients.i32",
                 "training": "training_predictions.i32"}
        path = root / names[part]
        path.write_bytes(path.read_bytes()[:-4])
    elif part == "failure":
        pilot.write_json(root / "failure.json", {"complete": False})
    elif part in ("weight_chain", "window_order", "route_digest"):
        path = root / "training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if part == "weight_chain":
            rows[0]["after_weights_sha256"] = "0" * 64
        elif part == "window_order":
            rows[0]["window_index"] = 1
        else:
            rows[0]["route_hashes"]["AD_HEAD"]["words"] = 0
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        summary = study._read(root / "summary.json")
        if part == "baseline":
            summary["baselines_by_step"]["2"]["bigram"]["nll"] = 0
        elif part == "initial":
            path = root / "checkpoint_000000.i32"
            values = study.optimizer._unpack(path.read_bytes())
            values[0] += 1
            payload = study.optimizer._pack(values)
            path.write_bytes(payload)
            record = summary["evaluations"][0]["checkpoint"]
            record["sha256"] = pilot.digest(payload)
            rewrite_json(root / "checkpoint_000000.json", record)
        else:
            summary["new_host_fitting_pair_exposures"] = 512
        rewrite_json(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest)


@pytest.mark.parametrize("field,value", [("step", 1.0), ("window_index", False),
    ("saturation_events", False), ("token_pairs", 128.0), ("input_readback_equal", 1),
    ("full_weight_update_equal", False), ("weight_update_clipping", 0),
    ("backward_int32_overflow_detected", 0), ("native_reference_equal", 1), ("native_routes_equal", 1)])
def test_resealed_training_receipts_have_literal_counts_and_guard_flags(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    path = case.config.output / "training.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reseal(study, case.config.output)
    with pytest.raises(ValueError):
        study.verify_worker(case.config.output, case.manifest)


@pytest.fixture
def fake_parent(study, fake_worker, tmp_path, monkeypatch):
    case, launches = fake_worker, []
    control = SimpleNamespace(failure=None)

    def execute(command, path):
        path.write_text("offline injected subprocess\n")
        launches.append(command)
        if control.failure is not None:
            if isinstance(control.failure, BaseException):
                raise control.failure
            return control.failure
        get = lambda flag: command[command.index(flag) + 1]
        study.run_worker(SimpleNamespace(parent=Path(get("--parent")), output=Path(get("--output")),
            corpus=Path(get("--corpus")), prior_run=Path(get("--prior-run")),
            prior_record=Path(get("--prior-record"))))
        return {"returncode": 0, "timed_out": False, "timeout_s": 1200, "elapsed_s": 0.1}

    monkeypatch.setattr(study, "_run_process", execute)
    config = SimpleNamespace(output=tmp_path / "parent", corpus=case.config.corpus,
                             prior_run=case.history, prior_record=case.old_record)
    return SimpleNamespace(case=case, config=config, launches=launches, controls=control)


def test_parent_and_publication_require_one_complete_bound_cell(study, fake_parent, tmp_path):
    fixture = fake_parent
    result = study.run(fixture.config)
    assert result["complete"] is True
    assert result["declared_cells"] == result["completed_cells"] == len(fixture.launches) == 1
    assert fixture.case.controls.deep_checks == [(arm, k) for arm in study.prior.ARMS for k in study.prior.K_VALUES]
    root = fixture.config.output
    record = study.publish_record(tmp_path / "public.json", root, result)
    assert record["full_weight_checkpoints_verified"] == 3
    assert record["checkpoint_validation_windows"] == 3
    assert record["new_native_updates"] == 4
    assert record["inherited_host_fitting_pair_exposures"] == record["new_gradient_pair_exposures"] == 512
    assert record["new_host_fitting_pair_exposures"] == 0
    assert record["initialization_refitted"] is False
    assert record["model_history_pair_exposures"] == 1024
    assert record["training_steps_guarded_and_compared_by_worker"] == 4
    assert record["posthoc_complete_training_trajectory_replayed"] is False
    assert record["paid_execution_authorized"] is False
    assert record["physical_fpga_execution"] is False
    assert record["three_seed_quality_gate_evaluated"] is False
    assert all("per_window" not in row for row in record["cells"][0]["evaluations"])
    assert record["artifact_manifest_sha256"] == pilot.digest((root / "artifact_manifest.json").read_bytes())
    with pytest.raises(FileExistsError):
        study.publish_record(tmp_path / "public.json", root, result)


@pytest.mark.parametrize("returncode,timed_out", [(17, False), (-9, True), (False, False), (0, 0)])
def test_failed_or_ambiguous_execution_never_promotes_cell(study, fake_parent, tmp_path, returncode, timed_out):
    fixture = fake_parent
    fixture.controls.failure = {"returncode": returncode, "timed_out": timed_out,
                                "timeout_s": 1200, "elapsed_s": 0.1}
    result = study.run(fixture.config)
    assert result["complete"] is False and result["completed_cells"] == 0
    assert len(result["failures"]) == 1
    assert (fixture.config.output / "signsgd.failure.json").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        study.publish_record(tmp_path / "false-success.json", fixture.config.output, result)
    assert not (tmp_path / "false-success.json").exists()


def test_parent_interrupt_retains_failure_and_seal(study, fake_parent):
    fake_parent.controls.failure = KeyboardInterrupt("injected interrupt")
    with pytest.raises(KeyboardInterrupt):
        study.run(fake_parent.config)
    root = fake_parent.config.output
    assert study._read(root / "failure.json")["complete"] is False
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)
    assert not (root / "summary.json").exists()


@pytest.mark.parametrize("field,value", [("returncode", 1), ("returncode", False), ("timed_out", True),
    ("timeout_s", 1200.0), ("elapsed_s", True), ("elapsed_s", -1), ("elapsed_s", math.nan),
    ("elapsed_s", math.inf), ("command", ["wrong-command"])])
def test_resealed_execution_receipt_is_rechecked_during_publication(study, fake_parent, tmp_path, field, value):
    result = study.run(fake_parent.config)
    root = fake_parent.config.output
    receipt = study._read(root / "signsgd.execution.json")
    receipt[field] = value
    rewrite_json(root / "signsgd.execution.json", receipt)
    reseal(study, root)
    target = tmp_path / "corrupt-public.json"
    with pytest.raises(ValueError):
        study.publish_record(target, root, result)
    assert not target.exists()


@pytest.mark.parametrize("which", ["source", "library", "build", "protocol", "dataset"])
def test_changed_manifest_bindings_fail_before_native_access(study, fake_worker, which):
    case = fake_worker
    manifest = study._read(case.parent / "manifest.json")
    if which == "source":
        manifest["source_hashes"] = {}
    elif which == "library":
        Path(manifest["library"]["path"]).write_bytes(b"changed native bytes")
    elif which == "build":
        rewrite_json(case.parent / "build/build.json", {})
    elif which == "protocol":
        rewrite_json(case.parent / "protocol.json", {})
    else:
        manifest["dataset"]["splits"]["validation"]["start"] += 1
    rewrite_json(case.parent / "manifest.json", manifest)
    with pytest.raises((ValueError, KeyError)):
        study.run_worker(case.config)
    assert not case.controls.opened and not case.config.output.exists()


def test_output_guards_preserve_all_prior_artifacts(study, fake_worker, tmp_path):
    case = fake_worker
    alias = tmp_path / "prior-alias"
    alias.symlink_to(case.history, target_is_directory=True)
    for directory in (case.history, alias):
        config = SimpleNamespace(output=directory / "new-control", corpus=case.config.corpus,
                                  prior_run=case.history, prior_record=case.old_record)
        with pytest.raises(ValueError):
            study.run(config)
        assert not config.output.exists()
    case.config.output = tmp_path / "unowned-worker"
    with pytest.raises(ValueError):
        study.run_worker(case.config)
    assert not case.config.output.exists()


def test_source_change_during_build_keeps_failed_setup_and_never_launches_worker(study, fake_parent, monkeypatch):
    original_build = pilot.build_csim
    original_identity = study.source_identity()
    changed = dict(original_identity)
    changed[Path(study.__file__).resolve().relative_to(pilot.REPO).as_posix()] = "0" * 64

    def mutate_identity_during_build(directory, width):
        library = original_build(directory, width)
        monkeypatch.setattr(study, "source_identity", lambda: changed)
        return library

    monkeypatch.setattr(pilot, "build_csim", mutate_identity_during_build)
    with pytest.raises(ValueError):
        study.run(fake_parent.config)
    assert not fake_parent.launches
    root = fake_parent.config.output
    assert study._read(root / "failure.json")["complete"] is False
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)


@pytest.mark.parametrize("guard", ["saturation", "overflow", "abs_int_min"])
def test_every_training_window_is_guarded_before_native_update(study, fake_worker, guard):
    case = fake_worker
    first_tokens = pilot.window(case.raw, case.dataset, "train", 0)[0]

    def guarded(weights, tokens, targets, shape, zshift, include_parameter_gradients=False):
        result = case.observer(weights, tokens, targets, shape, zshift, include_parameter_gradients)
        if tokens == first_tokens:
            if guard == "saturation":
                result["saturation_events"] = 1
            elif guard == "overflow":
                result["route_diagnostics"]["backward_int32_overflow_detected"] = True
            else:
                result["parameter_gradients"]["W_HEAD"][0] = -2147483648
        return result

    with pytest.raises(ValueError):
        study.run_worker(case.config, case.bridge, guarded)
    assert not case.controls.calls
    assert (case.config.output / "checkpoint_000000.i32").exists()
    assert study._read(case.config.output / "failure.json")["training_updates_completed"] == 0


@pytest.mark.parametrize("value", [1, True, False, 0.0])
def test_context_disabled_probe_cannot_compare_clipped_or_nonliteral_results(study, fake_worker, monkeypatch, value):
    original = pilot.integer_forward

    def clipped(*args, **kwargs):
        result = original(*args, **kwargs)
        result["saturation_events"] = value
        return result

    monkeypatch.setattr(pilot, "integer_forward", clipped)
    with pytest.raises(ValueError):
        study.run_worker(fake_worker.config)
    assert not fake_worker.controls.calls


@pytest.mark.parametrize("kind", ["payload", "seal"])
def test_publication_rechecks_exact_tree_and_seal_after_final_source_guard(study, fake_parent, monkeypatch, tmp_path, kind):
    result = study.run(fake_parent.config)
    root = fake_parent.config.output
    original = study._frozen
    calls = []

    def change_after_last_frozen(manifest):
        original(manifest)
        calls.append(None)
        # publish: beginning, verifier beginning/end, final publication guard.
        if len(calls) == 4:
            if kind == "payload":
                (root / "late-uncovered.txt").write_text("late publication mutation")
            else:
                path = root / "artifact_manifest.json"
                path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(study, "_frozen", change_after_last_frozen)
    target = tmp_path / "mutated-public.json"
    with pytest.raises(ValueError):
        study.publish_record(target, root, result)
    assert len(calls) == 4 and not target.exists()


@pytest.mark.parametrize("mode", ["success", "nonzero", "timeout", "interrupt"])
def test_reused_process_bound_kills_only_owned_group_on_every_exit(study, monkeypatch, tmp_path, mode):
    assert study._run_process is study.prior._run_process
    calls, kills, waits = [], [], []
    process = SimpleNamespace(pid=71933, returncode=17 if mode == "nonzero" else 0)

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            if mode == "timeout":
                raise subprocess.TimeoutExpired("unit-only", timeout)
            if mode == "interrupt":
                raise KeyboardInterrupt("unit-only")
        return process.returncode

    process.wait = wait

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(study.prior.subprocess, "Popen", launch)
    monkeypatch.setattr(study.prior.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    if mode == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            study._run_process(["unit-only"], tmp_path / "worker.log")
    else:
        result = study._run_process(["unit-only"], tmp_path / "worker.log")
        assert result["timed_out"] is (mode == "timeout")
        assert result["timeout_s"] == 1200
    assert calls[0][1]["start_new_session"] is True
    assert waits[0] == 1200
    assert kills and all(item == (71933, signal.SIGKILL) for item in kills)
    assert len(kills) == (2 if mode in ("timeout", "interrupt") else 1)
