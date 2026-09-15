"""Offline provenance/accounting contracts for the one declared route repair.

All data below are synthetic. Lifecycle fixtures fake native execution and old
expensive verification, while preserving the actual binary state/receipt ABI.
They never launch training, an AWS operation, or a scientific run.
"""

import copy
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
    return importlib.import_module("manuscript.tools.train_fpga_lm_diverse_projection")


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(91417)
    raw = bytes(rng.randrange(128) for _ in range(200_000))
    path = tmp_path / "synthetic.txt"
    path.write_bytes(raw)
    _, dataset = pilot.corpus_plan(path, 1024, 88)
    return raw, dataset


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def exposure_oracle(raw, dataset, indices, validation):
    uni, bi = [1] * 128, [[1] * 128 for _ in range(128)]
    for index in indices:
        start = dataset["splits"]["train"]["start"] + index * 129
        for offset in range(start, start + 128):
            token, target = raw[offset], raw[offset + 1]
            uni[target] += 1
            bi[token][target] += 1
    result = {}
    for name, counts in (("unigram", [uni] * 128), ("bigram", bi)):
        totals = list(map(sum, counts))
        predictions = [max(range(128), key=row.__getitem__) for row in counts]
        loss, correct = [], 0
        for index in validation:
            start = dataset["splits"]["validation"]["start"] + index * 129
            for offset in range(start, start + 128):
                token, target = raw[offset], raw[offset + 1]
                loss.append(-math.log(counts[token][target] / totals[token]))
                correct += predictions[token] == target
        result[name] = {"nll": sum(loss) / len(loss), "accuracy": correct / len(loss)}
    return result


def test_protocol_declares_exact_one_repair_and_zero_training_historical_controls(study):
    assert study.ARMS == ("diverse", "historical")
    assert study.SEEDS == (0, 1, 2)
    assert (study.WIDTH, study.ZSHIFT, study.K, study.STEPS) == (512, 2, 18, 1024)
    assert study.CHECKPOINTS == (0, 256, 1024)
    assert study.VALIDATION == tuple(range(72, 88))
    assert study.WORKER_TIMEOUT_S == 1200
    recipe = study.recipe()
    assert (recipe["declared_cells"], recipe["new_training_cells"], recipe["historical_evaluation_cells"]) == (6, 3, 3)
    assert recipe["new_updates_by_arm"] == {"diverse": 1024, "historical": 0}
    assert recipe["inherited_gradient_pairs_by_arm"] == {"diverse": 0, "historical": 131072}
    assert recipe["new_gradient_pairs_by_arm"] == {"diverse": 131072, "historical": 0}
    assert recipe["inherited_fitting_pairs_per_cell"] == 131072
    assert recipe["model_history_pairs_per_cell"] == 262144
    assert recipe["distinct_training_pairs"] == 131072
    assert recipe["new_host_fitting_pair_exposures"] == 0
    assert recipe["initialization_refitted"] is False
    assert "after all six" in recipe["gate_policy"]
    assert "not weight initialization or data seeds" in recipe["seed_meaning"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_old_order_is_preserved_without_global_rng_or_alias(study, seed):
    expected = list(range(1024))
    random.Random(seed).shuffle(expected)
    before = random.getstate()
    actual = study.training_order(seed)
    assert actual == expected == study.confirmation.training_order(seed)
    assert sorted(actual) == list(range(1024))
    assert random.getstate() == before
    actual[0] = -1
    assert study.training_order(seed) == expected


@pytest.mark.parametrize("seed", [True, False, 0.0, 1.0, -1, 3, "0", None])
def test_order_rejects_nondeclared_and_nonliteral_seeds(study, seed):
    with pytest.raises(ValueError):
        study.training_order(seed)


def test_source_identity_binds_initializer_confirmation_optimizer_and_observer(study):
    actual = study.source_identity()
    for source in (study.initializer, study.confirmation, study.optimizer):
        assert all(actual[name] == digest for name, digest in source.source_identity().items())
    path = Path(study.__file__).resolve()
    assert actual[path.relative_to(pilot.REPO).as_posix()] == pilot.digest(path.read_bytes())


@pytest.mark.parametrize("seed,step", [(0, 0), (0, 256), (1, 256), (2, 256), (0, 1024)])
def test_baselines_match_full_inherited_fit_plus_exact_shuffled_prefix(study, corpus, seed, step):
    raw, dataset = corpus
    indices = list(range(1024)) + study.training_order(seed)[:step]
    actual = study.exposure_baseline(raw, dataset, seed, step)
    expected = exposure_oracle(raw, dataset, indices, range(72, 88))
    assert actual["supervised_exposures"] == (1024 + step) * 128
    assert actual["distinct_supervised_pairs"] == 131072
    assert actual["alpha"] == 1
    assert actual["exposure_indices"] == indices
    assert actual["validation_indices"] == list(range(72, 88))
    for name in ("unigram", "bigram"):
        assert actual[name]["nll"] == pytest.approx(expected[name]["nll"], abs=1e-12)
        assert actual[name]["accuracy"] == expected[name]["accuracy"]


def test_one_pass_baseline_and_final_replay_are_order_invariant(study, corpus):
    raw, dataset = corpus
    actual = study.one_pass_baseline(raw, dataset)
    expected = exposure_oracle(raw, dataset, range(1024), range(72, 88))
    for name in ("unigram", "bigram"):
        assert actual[name]["nll"] == pytest.approx(expected[name]["nll"], abs=1e-12)
        assert actual[name]["accuracy"] == expected[name]["accuracy"]
        final = [study.exposure_baseline(raw, dataset, seed, 1024)[name] for seed in (0, 1, 2)]
        assert final[0] == final[1] == final[2]


def test_fitting_does_not_read_validation_test_or_unused_training_suffix(study, corpus):
    raw, dataset = corpus
    changed = bytearray(raw)
    stop = dataset["splits"]["train"]["start"] + 1024 * 129
    train_stop = dataset["splits"]["train"]["stop"]
    changed[stop:train_stop] = b"!" * (train_stop - stop)
    validation = dataset["splits"]["validation"]["start"]
    changed[validation:validation + 72 * 129] = b"?" * (72 * 129)
    test = dataset["splits"]["test"]["start"]
    changed[test:] = b"~" * (len(changed) - test)
    assert study.exposure_baseline(bytes(changed), dataset, 1, 256) == study.exposure_baseline(raw, dataset, 1, 256)
    assert study.one_pass_baseline(bytes(changed), dataset) == study.one_pass_baseline(raw, dataset)


@pytest.mark.parametrize("seed,step", [(True, 256), (3, 256), (0, True), (0, 0.0), (0, 255), (0, 1025)])
def test_baseline_rejects_undeclared_seed_or_budget(study, corpus, seed, step):
    with pytest.raises(ValueError):
        study.exposure_baseline(*corpus, seed, step)


def state_payload(study, shape, weights, remainders, *, step=0, k=18, zshift=2, loss_kind=1):
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
def history(study, corpus, tmp_path, monkeypatch):
    """Old full-state format and immutable tree; expensive old verifier is fake."""
    for module in (study, study.confirmation):
        monkeypatch.setattr(module, "WIDTH", 16)
        monkeypatch.setattr(module, "STEPS", 4)
        monkeypatch.setattr(module, "CHECKPOINTS", (0, 2, 4))
    monkeypatch.setattr(study, "VALIDATION", (72,))
    shape, (_, dataset) = pilot.layout(16), corpus
    weights = {name: [3] * getattr(shape, "WORDS_" + name) for name in pilot.WEIGHTS}
    remainders = {name: [0] * len(values) for name, values in weights.items()}
    initial = state_payload(study, shape, weights, remainders)
    root = tmp_path / "historical-confirmation"
    root.mkdir()
    ancestry = {"fixture": "independently verified prior fitted initialization"}
    manifest = {"schema": study.confirmation.SCHEMA, "recipe": study.confirmation.recipe(),
                "dataset": dataset, "source_hashes": study.confirmation.source_identity(), "prior_evidence": ancestry}
    pilot.write_json(root / "manifest.json", manifest)
    cells = []
    for seed in (0, 1, 2):
        directory = root / ("seed_%d" % seed)
        directory.mkdir()
        for step in (0, 2, 4):
            current, rem = copy.deepcopy(weights), copy.deepcopy(remainders)
            current["W_HEAD"][0] -= step
            rem["EMBED"][0] = step
            (directory / ("state_%06d.i32" % step)).write_bytes(state_payload(study, shape, current, rem, step=step))
        cell = {"seed": seed, "complete": True, "training_order": study.training_order(seed), "evaluations": []}
        pilot.write_json(directory / "summary.json", cell)
        cells.append(cell)
    gates = {"mechanism": True, "bigram": True, "context": False, "passed": False}
    summary = {"schema": study.confirmation.SCHEMA, "complete": True, "declared_cells": 3,
        "completed_cells": 3, "failures": [], "heldout_test_evaluated": False, "physical_fpga_execution": False,
        "speed_advantage_claim": False, "statistical_significance_claim": False, "paid_execution_authorized": False,
        "recipe": study.confirmation.recipe(), "source_hashes": manifest["source_hashes"],
        "prior_evidence": ancestry, "cells": cells, "gates": gates}
    pilot.write_json(root / "summary.json", summary)
    artifacts = study.confirmation._artifacts(root)
    pilot.write_json(root / "artifact_manifest.json", artifacts)
    public = tmp_path / "historical-record.json"
    receipt = {**summary, "cells": [study.confirmation._compact_cell(cell) for cell in cells],
        "run_directory": str(root), "artifact_manifest_sha256": pilot.digest((root / "artifact_manifest.json").read_bytes()),
        "artifact_files_verified": len(artifacts["files"]), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts["files"]),
        "full_optimizer_checkpoints_verified": 9, "checkpoint_validation_windows": 144, "distinct_validation_positions": 2048,
        "new_native_updates": 3072, "gradient_pair_exposures": 393216, "inherited_host_fitting_pair_exposures": 393216,
        "new_host_fitting_pair_exposures": 0, "initialization_refitted": False, "model_history_pair_exposures": 786432,
        "inherited_exposure_accounting": "sum of per-seed histories; the same old fit is shared, not newly repeated",
        "distinct_training_pairs": 131072, "native_training_trajectory_independently_replayed": False}
    pilot.write_json(public, receipt)
    calls = []

    def verify_old(directory, declaration, seed):
        calls.append(seed)
        return study._read(directory / "summary.json")

    def frozen_old(declaration):
        if (declaration["source_hashes"] != study.confirmation.source_identity()
                or pilot.digest(Path(declaration["dataset"]["path"]).read_bytes()) != declaration["dataset"]["sha256"]):
            raise ValueError("fake historical source or corpus changed")

    monkeypatch.setattr(study.confirmation, "_frozen", frozen_old)
    monkeypatch.setattr(study.confirmation, "_verify_execution", lambda *args: None)
    monkeypatch.setattr(study.confirmation, "verify_worker", verify_old)
    monkeypatch.setattr(study.confirmation, "read_initial_state", lambda binding, layout: initial)
    monkeypatch.setattr(study.confirmation, "gate_results", lambda cells: gates)
    binding = study.confirmation_binding(root, public, dataset, deep=False)
    return SimpleNamespace(root=root, public=public, dataset=dataset, shape=shape, initial=initial,
                           weights=weights, remainders=remainders, binding=binding, calls=calls, manifest=manifest)


def test_completed_failed_quality_gate_is_valid_history_and_all_nine_states_bind(study, history):
    case = history
    bound = study.confirmation_binding(case.root, case.public, case.dataset, deep=True)
    assert bound == case.binding
    assert case.calls == [0, 1, 2]
    assert study._read(case.public)["gates"]["passed"] is False
    for seed in (0, 1, 2):
        for step in (0, 2, 4):
            actual = study.read_confirmation_state(bound, seed, step, case.shape)
            assert actual == (case.root / ("seed_%d/state_%06d.i32" % (seed, step))).read_bytes()


@pytest.mark.parametrize("mutation", ["payload", "receipt", "summary", "source", "corpus", "failure"])
def test_binding_rejects_changed_actual_history_and_overclaims(study, history, monkeypatch, mutation):
    case = history
    dataset = case.dataset
    if mutation == "payload":
        (case.root / "seed_2/state_000004.i32").write_bytes(b"changed")
    elif mutation == "receipt":
        value = study._read(case.public)
        value["new_native_updates"] = float(value["new_native_updates"])
        rewrite(case.public, value)
    elif mutation == "summary":
        value = study._read(case.root / "summary.json")
        value["complete"] = 1
        rewrite(case.root / "summary.json", value)
    elif mutation == "source":
        monkeypatch.setattr(study.confirmation, "source_identity", lambda: {})
    elif mutation == "corpus":
        dataset = {**dataset, "sha256": "0" * 64}
    else:
        pilot.write_json(case.root / "failure.json", {"complete": False})
    with pytest.raises(ValueError):
        study.confirmation_binding(case.root, case.public, dataset, deep=False)


def test_binding_rehashes_every_payload_after_deep_verification(study, history, monkeypatch):
    original = study.confirmation.verify_worker

    def mutate(directory, manifest, seed):
        result = original(directory, manifest, seed)
        if seed == 2:
            (history.root / "late.txt").write_text("mutation during verification")
        return result

    monkeypatch.setattr(study.confirmation, "verify_worker", mutate)
    with pytest.raises(ValueError):
        study.confirmation_binding(history.root, history.public, history.dataset, deep=True)


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_binding_requires_literal_depth(study, history, value):
    with pytest.raises(ValueError):
        study.confirmation_binding(history.root, history.public, history.dataset, deep=value)


@pytest.mark.parametrize("field,index", [("step", 8), ("k", 5), ("loss", 6), ("shift", 7), ("source", 12)])
def test_rehashed_historical_state_cannot_change_native_config(study, history, field, index):
    case = history
    binding = copy.deepcopy(case.binding)
    path = Path(binding["states"]["1"]["2"]["file"])
    values = study.optimizer._unpack(path.read_bytes())
    values[index] += 1
    payload = study.optimizer._pack(values)
    path.write_bytes(payload)
    binding["states"]["1"]["2"] = {"file": str(path), "bytes": len(payload), "sha256": pilot.digest(payload)}
    with pytest.raises(ValueError):
        study.read_confirmation_state(binding, 1, 2, case.shape)


def test_initial_state_must_equal_actual_inherited_full_state_even_when_rehashed(study, history):
    binding = copy.deepcopy(history.binding)
    path = Path(binding["states"]["0"]["0"]["file"])
    values = study.optimizer._unpack(path.read_bytes())
    values[study.optimizer.STATE_HEADER + history.shape.WORDS_CE_LUT] += 1
    payload = study.optimizer._pack(values)
    path.write_bytes(payload)
    binding["states"]["0"]["0"] = {"file": str(path), "bytes": len(payload), "sha256": pilot.digest(payload)}
    with pytest.raises(ValueError):
        study.read_confirmation_state(binding, 0, 0, history.shape)


def test_historical_state_receipt_requires_literal_size_and_exact_path(study, history):
    for changes in ({"bytes": float(len(history.initial))}, {"file": "../other.i32"}):
        binding = copy.deepcopy(history.binding)
        binding["states"]["0"]["0"].update(changes)
        with pytest.raises(ValueError):
            study.read_confirmation_state(binding, 0, 0, history.shape)


@pytest.fixture
def fake_worker(study, history, corpus, tmp_path, monkeypatch):
    """Small ABI-faithful simulation, never a native or scientific run."""
    case = history
    shape = case.shape
    controls = SimpleNamespace(train_error=None, close_error=None, initial_mutation=False,
        evaluate_mutation=False, import_mutation=False, wrong_native=False, wrong_step=False,
        observed_saturation=False, observed_overflow=False, probe_saturation=False,
        opened=[], calls=[], imports=[])

    def transform(weights, layout):
        assert layout.ML_D == 16
        changed = copy.deepcopy(weights)
        changed["MP"][0] = -8192
        return changed, {"schema": study.initializer.SCHEMA, "synthetic_initializer_fixture": True,
            "source_weights_sha256": pilot.digest(pilot.weight_bytes(weights)),
            "transformed_weights_sha256": pilot.digest(pilot.weight_bytes(changed))}

    def equivalence(original, changed, tokens, targets, layout):
        assert changed == transform(original, layout)[0]
        return {"initial_logits_equal": True, "initial_predictions_gradients_equal": True,
                "initial_head_tap_routes_equal": True, "initial_projection_routes_equal_mapping": True,
                "initial_upper_three_projection_l1_preserved": True, "synthetic_fixture": True}

    def observed(weights, tokens, targets, layout, zshift):
        value = weights["W_HEAD"][0]
        routes = {name: [0] * getattr(layout, "WORDS_" + name)
                  for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}
        return {"predictions": list(tokens), "gradients": [0] * 16384, "native_routes": routes,
            "lut_normalized_nll": 4.0 + value / 100, "accuracy": .25,
            "saturation_events": int(controls.observed_saturation), "rounded_zero_target_mass": 0,
            "route_diagnostics": {"positions": 128, "nonzero_logit_gradient_positions": 128,
                "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
                "head_winners_ge16_positions": 128, "single_shared_head_winner_positions": 0,
                "nonzero_embedding_input_vjp_positions": 128, "max_abs_head_input_vjp": 3,
                "max_abs_embedding_input_vjp": 3, "backward_int32_overflow_detected": controls.observed_overflow,
                "head_winner_counts": [16] * 128}}

    def disabled(weights, tokens, targets, layout, zshift, use_context):
        assert use_context is False
        result = observed(weights, tokens, targets, layout, zshift)
        result["saturation_events"] = int(controls.probe_saturation)
        result["lut_normalized_nll"] += (3 - weights["W_HEAD"][0]) / 20
        return result

    def build(directory, width):
        directory.mkdir(parents=True, exist_ok=False)
        executable = directory / "fake-native"
        executable.write_bytes(b"not executable; offline fixture")
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
                getattr(layout, "OFF_" + name) + getattr(layout, "WORDS_" + name)]) for name in pilot.WEIGHTS}
            self.remainders = {name: [0] * len(values) for name, values in self.weights.items()}
            self.identity = {"schema": study.optimizer.SCHEMA,
                "executable_sha256": pilot.digest(executable.read_bytes()),
                "build_sha256": pilot.digest((executable.parent / "build.json").read_bytes()),
                "source_hashes": study.optimizer.source_identity(), "layout": pilot.layout_values(layout),
                "cpu_only": True, "existing_afi_compatible": False}
            self.closed = False
            if controls.initial_mutation:
                self.remainders["EMBED"][0] = 1
            log_path.write_text("fake native; no process\n")
            controls.opened.append(self)
            assert timeout_s == 60

        def export_state(self):
            return state_payload(study, self.shape, self.weights, self.remainders, step=self.step, k=self.k)

        def import_state(self, payload):
            state = study.optimizer.decode_state(payload, self.shape)
            self.weights, self.remainders = state["weights"], state["remainders"]
            self.step, self.k = state["step"], state["k"]
            controls.imports.append(payload)
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
                raise controls.train_error("injected native training failure")
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

    monkeypatch.setattr(study.initializer, "transform", transform)
    monkeypatch.setattr(study.initializer, "check_initial_equivalence", equivalence)
    monkeypatch.setattr(study, "observe", observed)
    monkeypatch.setattr(study.pilot, "integer_forward", disabled)
    monkeypatch.setattr(study.optimizer, "EFSGD", Native)
    monkeypatch.setattr(study.optimizer, "build", build)
    parent = tmp_path / "new-study"
    parent.mkdir()
    executable = build(parent / "build", 16)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(), "run_directory": str(parent),
        "dataset": case.dataset, "source_hashes": study.source_identity(),
        "plan_sha256": pilot.digest(study.PLAN.read_bytes()), "confirmation_evidence": case.binding,
        "executable": {"path": str(executable), "sha256": pilot.digest(executable.read_bytes())},
        "build_record_sha256": pilot.digest((parent / "build/build.json").read_bytes())}
    pilot.write_json(parent / "protocol.json", {"schema": study.SCHEMA, **study.recipe()})
    pilot.write_json(parent / "manifest.json", manifest)
    config = SimpleNamespace(parent=parent, output=parent / "diverse_seed_0", arm="diverse", seed=0,
        corpus=Path(case.dataset["path"]), confirmation_run=case.root, confirmation_record=case.public)
    return SimpleNamespace(config=config, controls=controls, shape=shape, raw=corpus[0], history=case,
                           parent=parent, manifest=manifest, executable=executable, build=build)


@pytest.mark.parametrize("arm", ["diverse", "historical"])
def test_initial_complete_state_changes_mp_only_in_repair_arm(study, fake_worker, arm):
    case = fake_worker
    buffer, payload, facts = study.initial_buffer(case.history.binding, arm, 0, case.shape)
    state = study.optimizer.decode_state(payload, case.shape)
    original = study.optimizer.decode_state(case.history.initial, case.shape)
    assert state["remainders"] == original["remainders"]
    assert state["config"] == original["config"] and state["step"] == 0
    for name in pilot.WEIGHTS:
        offset = getattr(case.shape, "OFF_" + name)
        assert buffer[offset:offset + len(state["weights"][name])] == state["weights"][name]
        if arm == "historical" or name != "MP":
            assert state["weights"][name] == original["weights"][name]
    if arm == "historical":
        assert payload == case.history.initial
    else:
        assert payload != case.history.initial and state["weights"]["MP"][0] == -8192


@pytest.mark.parametrize("arm,seed", [(arm, seed) for arm in ("diverse", "historical") for seed in (0, 1, 2)])
def test_actual_worker_accounts_imported_history_without_retraining_controls(study, fake_worker, arm, seed):
    case = fake_worker
    case.config.arm, case.config.seed = arm, seed
    case.config.output = case.parent / ("%s_seed_%d" % (arm, seed))
    result = study.run_worker(case.config)
    assert result == study.verify_worker(case.config.output, case.manifest, arm, seed)
    assert result["complete"] is True and case.controls.opened[0].closed
    assert result["training_performed"] is (arm == "diverse")
    assert result["training_updates"] == (4 if arm == "diverse" else 0)
    assert result["new_gradient_pair_exposures"] == (512 if arm == "diverse" else 0)
    assert result["inherited_gradient_pair_exposures"] == (0 if arm == "diverse" else 512)
    assert result["inherited_host_fitting_pair_exposures"] == 512
    assert result["new_host_fitting_pair_exposures"] == 0
    assert result["initialization_refitted"] is False
    assert result["model_history_pair_exposures"] == 1024
    assert result["distinct_training_pairs"] == 512
    assert result["initial_state_equal_declared_arm"] is True
    assert result["initializer"]["applied"] is (arm == "diverse")
    assert "proposal" in result["initializer"]
    assert "initial_state_equal_declared_transform" not in result
    assert result["historical_full_states_equal"] is (arm == "historical")
    for name in ("heldout_test_evaluated", "paid_execution_authorized", "physical_fpga_execution",
                 "speed_advantage_claim", "native_training_trajectory_independently_replayed"):
        assert result[name] is False
    assert [row["step"] for row in result["evaluations"]] == [0, 2, 4]
    assert len(case.controls.calls) == (4 if arm == "diverse" else 0)
    if arm == "diverse":
        expected = [pilot.window(case.raw, case.history.dataset, "train", index) for index in study.training_order(seed)]
        assert case.controls.calls == [(list(tokens), list(targets)) for tokens, targets in expected]
    else:
        for filename in ("training.jsonl", "training_predictions.i32", "training_gradients.i32"):
            assert (case.config.output / filename).read_bytes() == b""
        for step in (0, 2, 4):
            filename = "state_%06d.i32" % step
            assert (case.config.output / filename).read_bytes() == (case.history.root / ("seed_%d" % seed) / filename).read_bytes()
    stronger = min(result["baselines_by_step"]["4"]["bigram"]["nll"], result["one_pass_baseline"]["bigram"]["nll"])
    assert result["conservative_bigram_advantage_nats"] == stronger - result["evaluations"][-1]["lut_normalized_nll"]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_retains_exclusive_sealed_evidence_and_closes_child(study, fake_worker, failure):
    case = fake_worker
    case.controls.train_error = failure
    with pytest.raises(failure):
        study.run_worker(case.config)
    root = case.config.output
    assert study._read(root / "failure.json")["type"] == failure.__name__
    assert not (root / "summary.json").exists() and case.controls.opened[0].closed
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)
    before = (root / "failure.json").read_bytes()
    with pytest.raises(FileExistsError):
        study.run_worker(case.config)
    assert (root / "failure.json").read_bytes() == before


@pytest.mark.parametrize("field", ["initial_mutation", "evaluate_mutation", "import_mutation", "wrong_native",
                                    "wrong_step", "observed_saturation", "observed_overflow", "probe_saturation"])
def test_state_arrays_counter_saturation_and_overflow_fail_closed(study, fake_worker, field):
    case = fake_worker
    setattr(case.controls, field, True)
    with pytest.raises(ValueError):
        study.run_worker(case.config)
    assert (case.config.output / "failure.json").exists()
    assert not (case.config.output / "summary.json").exists() and case.controls.opened[0].closed


def test_success_is_not_published_before_native_cleanup(study, fake_worker):
    fake_worker.controls.close_error = RuntimeError
    with pytest.raises(RuntimeError, match="cleanup"):
        study.run_worker(fake_worker.config)
    assert (fake_worker.config.output / "failure.json").exists()
    assert not (fake_worker.config.output / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["source", "corpus", "executable", "build", "protocol", "history"])
def test_changed_inputs_reject_before_creating_worker_output(study, fake_worker, monkeypatch, mutation):
    case = fake_worker
    if mutation == "source":
        monkeypatch.setattr(study, "source_identity", lambda: {})
    elif mutation == "corpus":
        case.config.corpus.write_bytes(b"changed")
    elif mutation == "executable":
        case.executable.write_bytes(b"changed")
    elif mutation == "build":
        rewrite(case.parent / "build/build.json", {})
    elif mutation == "protocol":
        rewrite(case.parent / "protocol.json", {})
    else:
        (case.history.root / "seed_2/state_000004.i32").write_bytes(b"changed")
    with pytest.raises(ValueError):
        study.run_worker(case.config)
    assert not case.config.output.exists() and not case.controls.opened


def reseal(study, root):
    rewrite(root / "artifact_manifest.json", study._artifacts(root))


@pytest.mark.parametrize("field,value", [
    ("step", False), ("step", 0.0), ("k", 20), ("config", {}), ("identity", {}),
    ("regions", []), ("format", "weights only"), ("file", "../other.i32"), ("bytes", True),
])
def test_resealed_checkpoint_receipt_requires_exact_state_config_and_types(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    record = study._read(root / "state_000000.json")
    record[field] = value
    rewrite(root / "state_000000.json", record)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["checkpoint"] = record
    rewrite(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, "diverse", 0)


@pytest.mark.parametrize("field,value", [
    ("window_index", 56), ("window_index", 72.0), ("native_reference_equal", 1),
    ("native_routes_equal", False), ("native_arrays", {}), ("initial_equivalence", None),
])
def test_resealed_validation_cannot_relabel_old_cohort_or_skip_native_checks(study, fake_worker, field, value):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    row = study._read(root / "eval_000000_072.json")
    row[field] = value
    rewrite(root / "eval_000000_072.json", row)
    summary = study._read(root / "summary.json")
    summary["evaluations"][0]["per_window"][0] = row
    rewrite(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, "diverse", 0)


@pytest.mark.parametrize("part", ["state", "native", "training", "baseline", "order", "claim", "initializer", "failure"])
def test_resealed_arrays_baselines_order_and_scope_cannot_be_forged(study, fake_worker, part):
    case = fake_worker
    study.run_worker(case.config)
    root = case.config.output
    if part in ("state", "native", "training"):
        filename = {"state": "state_000002.i32", "native": "eval_000002_072_gradients.i32",
                    "training": "training_predictions.i32"}[part]
        path = root / filename
        payload = bytearray(path.read_bytes())
        payload[0] ^= 1
        path.write_bytes(payload)
    elif part == "order":
        path = root / "training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["window_index"] = (rows[0]["window_index"] + 1) % 4
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif part == "failure":
        rewrite(root / "failure.json", {"complete": False})
    else:
        summary = study._read(root / "summary.json")
        if part == "baseline":
            summary["baselines_by_step"]["2"]["bigram"]["nll"] = 0
        elif part == "initializer":
            summary["initializer"]["applied"] = False
        else:
            summary["paid_execution_authorized"] = True
        rewrite(root / "summary.json", summary)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.verify_worker(root, case.manifest, "diverse", 0)


@pytest.mark.parametrize("filename", ["training.jsonl", "training_predictions.i32", "training_gradients.i32"])
def test_historical_control_must_have_no_new_training_payload_even_if_resealed(study, fake_worker, filename):
    case = fake_worker
    case.config.arm, case.config.output = "historical", case.parent / "historical_seed_0"
    study.run_worker(case.config)
    (case.config.output / filename).write_bytes(b"x")
    reseal(study, case.config.output)
    with pytest.raises(ValueError):
        study.verify_worker(case.config.output, case.manifest, "historical", 0)


def gate_cells():
    cells = []
    for arm in ("diverse", "historical"):
        for seed in (0, 1, 2):
            initial = {"lut_normalized_nll": 3.0, "context_disabled_nll": 3.0}
            final = {"lut_normalized_nll": 2.0, "context_disabled_nll": 2.03,
                "saturation_events": 0, "native_reference_equal": True, "native_routes_equal": True,
                "route_diagnostics": {"positions": 2048, "nonzero_logit_gradient_positions": 2048,
                    "exact_zero_head_input_vjp_on_nonzero_gradient_positions": 0,
                    "head_winners_ge16_positions": 2048, "backward_int32_overflow_detected": False}}
            cells.append({"arm": arm, "seed": seed, "complete": True, "evaluations": [initial, final],
                          "conservative_comparators": {"unigram": {"nll": 2.1}, "bigram": {"nll": 2.03}}})
    return cells


def test_gate_is_exact_unchanged_per_arm_screen_and_not_cloud_authorization(study):
    cells = gate_cells()
    gates = study.gate_results(cells)
    for arm in study.ARMS:
        expected = study.confirmation.gate_results([row for row in cells if row["arm"] == arm])
        assert gates[arm] == expected
        assert gates[arm]["larger_paid_learning_benchmark_gate"] is True
        assert gates[arm]["paid_execution_authorized"] is False
        assert gates[arm]["statistical_significance_claim"] is False
    cells[4]["evaluations"][-1]["context_disabled_nll"] = 2.019
    gates = study.gate_results(cells)
    assert gates["diverse"]["larger_paid_learning_benchmark_gate"] is True
    assert gates["historical"]["context"] is False
    assert gates["historical"]["larger_paid_learning_benchmark_gate"] is False


@pytest.mark.parametrize("mutation", ["five", "diverse_only", "duplicate", "incomplete", "bool_complete", "bool_seed", "nan"])
def test_neither_arm_is_assessed_before_all_six_declared_finite_complete_cells(study, mutation):
    cells = gate_cells()
    if mutation == "five":
        cells.pop()
    elif mutation == "diverse_only":
        cells = cells[:3]
    elif mutation == "duplicate":
        cells[-1] = cells[-2]
    elif mutation in ("incomplete", "bool_complete"):
        cells[-1]["complete"] = False if mutation == "incomplete" else 1
    elif mutation == "bool_seed":
        cells[0]["seed"] = False
    else:
        cells[-1]["evaluations"][-1]["lut_normalized_nll"] = float("nan")
    with pytest.raises(ValueError):
        study.gate_results(cells)


@pytest.fixture
def fake_parent(study, fake_worker, tmp_path, monkeypatch):
    case = fake_worker
    parent_calls = []
    controls = SimpleNamespace(bad_execution=None)

    def invoke(command, log_path):
        arguments = {command[index]: command[index + 1] for index in range(3, len(command), 2)}
        arm, seed = arguments["--arm"], int(arguments["--seed"])
        config = SimpleNamespace(corpus=Path(arguments["--corpus"]), parent=Path(arguments["--parent"]),
            output=Path(arguments["--output"]), confirmation_run=Path(arguments["--confirmation-run"]),
            confirmation_record=Path(arguments["--confirmation-record"]), arm=arm, seed=seed)
        parent_calls.append((arm, seed))
        log_path.write_text("in-process fixture; no native execution\n")
        study.run_worker(config)
        receipt = {"returncode": 0, "timed_out": False, "timeout_s": 1200, "elapsed_s": .01}
        if controls.bad_execution is not None and (arm, seed) == ("diverse", 0):
            receipt.update(controls.bad_execution)
        return receipt

    monkeypatch.setattr(study, "_run_process", invoke)
    config = SimpleNamespace(output=tmp_path / "completed-six-cell-fixture", corpus=case.config.corpus,
        confirmation_run=case.history.root, confirmation_record=case.history.public)
    return SimpleNamespace(case=case, config=config, calls=parent_calls, controls=controls,
                           record=tmp_path / "public-six-cell-record.json")


def test_parent_retains_all_six_cells_and_publication_counts_new_work_separately(study, fake_parent):
    case = fake_parent
    result = study.run(case.config)
    assert case.calls == [(arm, seed) for arm in study.ARMS for seed in study.SEEDS]
    assert result["complete"] is True and result["completed_cells"] == result["declared_cells"] == 6
    assert result["paired_initial_native_outputs_equal"] is True
    assert all(value["passed"] is False for value in result["gates"].values())
    assert len(case.case.controls.calls) == 12  # Three new four-step fixtures, not six.
    value = study.publish_record(case.record, case.config.output, result)
    assert value["full_optimizer_checkpoints_verified"] == 18
    assert value["checkpoint_validation_windows"] == 18
    assert value["distinct_validation_positions"] == 128
    assert value["new_native_updates"] == 12
    assert value["new_gradient_pair_exposures"] == 1536
    assert value["inherited_gradient_pair_exposures"] == 1536
    assert value["inherited_host_fitting_pair_exposures"] == 3072
    assert value["new_host_fitting_pair_exposures"] == 0
    assert value["initialization_refitted"] is False
    assert value["model_history_pair_exposures"] == 6144
    assert value["distinct_training_pairs"] == 512
    assert value["native_training_trajectory_independently_replayed"] is False
    assert all("per_window" not in row for cell in value["cells"] for row in cell["evaluations"])
    assert study._read(case.record) == value
    before = case.record.read_bytes()
    with pytest.raises(FileExistsError):
        study.publish_record(case.record, case.config.output, result)
    assert case.record.read_bytes() == before


@pytest.mark.parametrize("invalid", [{"returncode": False}, {"returncode": 1}, {"timed_out": True},
                                    {"timeout_s": 1200.0}, {"elapsed_s": -1}])
def test_failed_execution_does_not_drop_cell_or_evaluate_either_arm_gate(study, fake_parent, invalid):
    case = fake_parent
    case.controls.bad_execution = invalid
    result = study.run(case.config)
    assert len(case.calls) == 6
    assert result["complete"] is False and result["completed_cells"] == 5
    assert result["gates"] is None and result["paired_initial_native_outputs_equal"] is None
    assert result["failures"][0]["arm"] == "diverse" and result["failures"][0]["seed"] == 0
    assert (case.config.output / "diverse_seed_0.failure.json").exists()
    with pytest.raises(ValueError):
        study.publish_record(case.record, case.config.output, result)
    assert not case.record.exists()


@pytest.mark.parametrize("name", ["predictions", "gradients", "AD_HEAD", "TAP_IDX"])
def test_initial_cross_arm_native_comparison_requires_actual_full_arrays(study, history, tmp_path, name):
    root = tmp_path / "paired-only"
    shape = history.shape
    sizes = {"predictions": 128, "gradients": 16384, "AD_HEAD": shape.WORDS_AD_HEAD,
             "TAP_IDX": shape.WORDS_TAP_IDX}
    for arm in study.ARMS:
        for seed in study.SEEDS:
            directory = root / ("%s_seed_%d" % (arm, seed))
            directory.mkdir(parents=True)
            for key, count in sizes.items():
                (directory / ("eval_000000_072_%s.i32" % key)).write_bytes(b"\0" * (count * 4))
    study._verify_paired_initializations(root)
    filename = "eval_000000_072_%s.i32" % name
    changed = root / "diverse_seed_2" / filename
    payload = bytearray(changed.read_bytes())
    payload[0] = 1
    changed.write_bytes(payload)
    with pytest.raises(ValueError):
        study._verify_paired_initializations(root)
    changed.write_bytes(b"")
    (root / "historical_seed_2" / filename).write_bytes(b"")
    with pytest.raises(ValueError):
        study._verify_paired_initializations(root)


def test_sources_cannot_change_during_native_build_and_be_relabelled(study, fake_parent, monkeypatch):
    original_build, original_sources = study.optimizer.build, study.source_identity()
    changed = {}
    monkeypatch.setattr(study, "source_identity", lambda: changed or original_sources)

    def build(*args, **kwargs):
        executable = original_build(*args, **kwargs)
        changed["changed-source.py"] = "0" * 64
        return executable

    monkeypatch.setattr(study.optimizer, "build", build)
    with pytest.raises(ValueError, match="across native build"):
        study.run(fake_parent.config)
    root = fake_parent.config.output
    assert not fake_parent.calls and not (root / "manifest.json").exists()
    assert (root / "failure.json").exists()
    assert study._read(root / "artifact_manifest.json") == study._artifacts(root)


@pytest.mark.parametrize("mutation", ["execution", "scope", "gate", "extra_failure", "raw_bytes"])
def test_publication_reaudits_actual_receipts_scope_gates_and_payloads(study, fake_parent, mutation):
    case = fake_parent
    result = study.run(case.config)
    root = case.config.output
    if mutation == "execution":
        path = root / "historical_seed_2.execution.json"
        value = study._read(path)
        value["command"][-1] = "1"
        rewrite(path, value)
    elif mutation in ("scope", "gate"):
        if mutation == "scope":
            result["statistical_significance_claim"] = True
        else:
            result["gates"]["historical"]["passed"] = True
        rewrite(root / "summary.json", result)
    elif mutation == "extra_failure":
        rewrite(root / "failure.json", {"complete": False})
    else:
        path = root / "historical_seed_2/eval_000004_072_gradients.i32"
        value = bytearray(path.read_bytes())
        value[0] ^= 1
        path.write_bytes(value)
        reseal(study, path.parent)
    reseal(study, root)
    with pytest.raises(ValueError):
        study.publish_record(case.record, root, result)
    assert not case.record.exists()


def test_publication_checks_raw_bytes_again_after_first_parent_artifact_check(study, fake_parent, monkeypatch):
    case = fake_parent
    result = study.run(case.config)
    original = study._artifacts
    calls = []

    def mutate_after_read(root):
        value = original(root)
        if root == case.config.output:
            calls.append(True)
            if len(calls) == 1:
                (root / "late.txt").write_text("changed during final publication checks")
        return value

    monkeypatch.setattr(study, "_artifacts", mutate_after_read)
    with pytest.raises(ValueError):
        study.publish_record(case.record, case.config.output, result)
    assert len(calls) == 2 and not case.record.exists()


def test_raw_output_and_public_record_are_exclusive_and_outside_historical_tree(study, fake_parent):
    case = fake_parent
    case.config.output = case.case.history.root / "new-nested-study"
    with pytest.raises(ValueError):
        study.run(case.config)
    assert not case.config.output.exists() and not case.calls
    with pytest.raises(ValueError):
        study.publish_record(case.case.parent / "embedded-record.json", case.case.parent, {})
    assert not (case.case.parent / "embedded-record.json").exists()


@pytest.mark.parametrize("extra", [["--arm", "diverse"], ["--seed", "0"], ["--parent", "ignored"],
                                    ["--worker", "--parent", "ignored", "--arm", "diverse", "--seed", "0", "--record", "record.json"]])
def test_cli_forbids_research_recipe_overrides_and_worker_publication(study, tmp_path, monkeypatch, extra):
    calls = []
    monkeypatch.setattr(study, "run", lambda config: calls.append(config))
    monkeypatch.setattr(study, "run_worker", lambda config: calls.append(config))
    args = ["--corpus", str(tmp_path / "corpus"), "--confirmation-run", str(tmp_path / "old"),
            "--confirmation-record", str(tmp_path / "old.json"), "--output", str(tmp_path / "out"), *extra]
    with pytest.raises(SystemExit):
        study.main(args)
    assert not calls and not (tmp_path / "out").exists()
