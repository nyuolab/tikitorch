"""Synthetic protocol/gate/lifecycle checks; no corpus or model study run."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_normalized_accumulation")


@pytest.fixture(autouse=True)
def no_science(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real model/corpus/native execution is forbidden in these unit tests")

    parent = study.parent
    for owner, name in ((parent.extension, "forward"), (parent.extension, "gradient"),
                        (parent.prior.reference, "forward"), (parent.prior.observer, "observe"),
                        (parent.pilot, "window"), (parent.pilot, "corpus_plan"),
                        (parent.old.native, "EFSGD"), (parent.old.native, "decode_state")):
        monkeypatch.setattr(owner, name, forbidden)


def mapping(value=0):
    shapes = {"EMBED": (3, 2), "MH": (4, 2, 7), "MP": (4, 2, 2),
              "MB": (4, 2), "W_HEAD": (3, 2)}
    return {name: torch.full(shape, float(value), dtype=torch.float64)
            for name, shape in shapes.items()}


def metric(ce=2.0, lut=2.1, accuracy=.25):
    return {"true_ce": ce, "lut_nll": lut, "accuracy": accuracy}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_source_map_binds_new_runner_both_test_files_and_shared_checked_arithmetic(study):
    root = Path(__file__).resolve().parents[1]
    sources = study.source_identity()
    expected = {"manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py",
                "tests/test_fpga_lm_normalized_accumulation.py",
                "tests/test_fpga_lm_normalized_accumulation_replay.py",
                "manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py",
                "manuscript/tools/fpga_lm_accumulation_arithmetic.py",
                "tikitorch/optim/fixed_point.py", "tikitorch/optim/__init__.py"}
    assert expected <= set(sources)
    for name in expected:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sources[name]


def test_recipe_returns_independent_repeatable_values(study):
    original = copy.deepcopy(study.recipe())
    changed = study.recipe()
    changed["synthetic-only"] = True
    assert study.recipe() == original


def test_fixed_arm_scales_batch_sizes_and_probe_exposures(study):
    assert study.ARMS == ("seq_k18", "seq_k20", "batch4_k20")
    assert study.ORDERS == ("consecutive", "shuffle_0", "shuffle_1")
    assert study.ARM_CONFIG == {"seq_k18": {"batch_size": 1, "k": 18},
                                "seq_k20": {"batch_size": 1, "k": 20},
                                "batch4_k20": {"batch_size": 4, "k": 20}}
    assert tuple(study.WINDOWS) == tuple(range(32))
    assert tuple(study.PROBES) == (0, 37, 511)
    assert tuple(study.PROBE_EXPOSURES) == (8, 32)
    assert study.DEADLINE_S == 3600


@pytest.mark.parametrize("label,seed", [("consecutive", None), ("shuffle_0", 0), ("shuffle_1", 1)])
def test_orders_match_independent_python_random_oracle_without_global_rng_mutation(study, label, seed):
    expected = list(range(32))
    if seed is not None:
        random.Random(seed).shuffle(expected)
    before = random.getstate()
    actual = study.training_order(label)
    assert actual == expected and random.getstate() == before
    assert sorted(actual) == list(range(32)) and all(type(index) is int for index in actual)
    actual.reverse()
    assert study.training_order(label) == expected


@pytest.mark.parametrize("bad", [None, True, 0, "shuffle_2", "consecutive ", "validation"])
def test_only_three_predeclared_orders_are_accepted(study, bad):
    with pytest.raises(ValueError):
        study.training_order(bad)


def test_k20_state_preserves_carry_that_is_invalid_at_k18(study, tmp_path):
    path = tmp_path / "state.npz"
    weights, remainders = mapping(7), mapping((1 << 18) + 17)
    study.parent.save(path, study.parent.state_arrays(weights, remainders))
    original = path.read_bytes()
    w, r = study.state(path, 20)
    study.parent.same(w, weights, "full weights")
    study.parent.same(r, remainders, "full k20 carry")
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        study.state(path, 18)


@pytest.mark.parametrize("k", [18, 20])
@pytest.mark.parametrize("sign", [-1, 1])
def test_state_requires_canonical_own_k_remainders(study, tmp_path, k, sign):
    path = tmp_path / "state.npz"
    study.parent.save(path, study.parent.state_arrays(mapping(), mapping(sign * ((1 << k) - 1))))
    _, remainder = study.state(path, k)
    assert remainder["EMBED"][0, 0] == sign * ((1 << k) - 1)
    arrays = study.parent.state_arrays(mapping(), mapping(sign * (1 << k)))
    with path.open("wb") as stream:
        np.savez_compressed(stream, **{name: value.numpy() for name, value in arrays.items()})
    with pytest.raises(ValueError):
        study.state(path, k)


@pytest.mark.parametrize("bad", [True, False, 18.0, 20.0, "20", None, 0, 19, 21])
def test_state_k_is_literal_and_one_of_declared_denominators(study, tmp_path, bad):
    path = tmp_path / "state.npz"
    study.parent.save(path, study.parent.state_arrays(mapping(), mapping()))
    with pytest.raises(ValueError):
        study.state(path, bad)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "shape", "fraction", "weight_range"])
def test_state_requires_complete_natural_shaped_exact_integer_payload(study, tmp_path, mutation):
    arrays = study.parent.state_arrays(mapping(), mapping())
    if mutation == "missing":
        arrays.pop("r_MH")
    elif mutation == "extra":
        arrays["extra"] = torch.zeros(1, dtype=torch.float64)
    elif mutation == "dtype":
        arrays["w_EMBED"] = arrays["w_EMBED"].to(torch.int64)
    elif mutation == "shape":
        arrays["w_EMBED"] = arrays["w_EMBED"].flatten()
    else:
        arrays["w_EMBED"][0, 0] = .5 if mutation == "fraction" else 32768
    path = tmp_path / "state.npz"
    study.parent.save(path, arrays)
    with pytest.raises(ValueError):
        study.state(path, 20)


def cells(study, *, batch=2.0, seq20=2.1, seq18=2.05):
    values = {"seq_k18": seq18, "seq_k20": seq20, "batch4_k20": batch}
    return [{"arm": arm, "order": order, "complete": True, "mechanics_pass": True,
             "final_mean": metric(values[arm], values[arm] + .1)}
            for arm in study.ARMS for order in study.ORDERS]


def cell(rows, arm, order="consecutive"):
    return next(row for row in rows if row["arm"] == arm and row["order"] == order)


def test_declared_accounting_is_288_queries_216_commits_and_434_scores(study):
    recipe = study.recipe()
    assert recipe["declared_cells"] == 9
    assert recipe["full_gradient_queries"] == 3 * 3 * 32 == 288
    assert recipe["optimizer_commits"] == 3 * (32 + 32 + 8) == 216
    assert recipe["complete_saved_states"] == 216 + 9 == 225
    assert recipe["score_readouts"] == 32 + 6 + 9 * (32 + 2 * 6) == 434
    assert recipe["new_gradient_pair_exposures"] == 288 * 128 == 36864
    assert recipe["distinct_gradient_pairs"] == 32 * 128 == 4096
    assert recipe["summed_inherited_fitting_pair_exposures"] == 9 * 131072
    assert recipe["summed_endpoint_history_pair_exposures"] == 9 * (131072 + 4096)
    assert recipe["distinct_fitted_pairs"] == 131072
    assert recipe["new_fitting_pair_exposures"] == recipe["new_native_training_calls"] == 0
    assert "no extra division" in recipe["normalization"]
    for key in ("heldout_corpus_evaluated", "physical_fpga_execution", "paid_execution_authorized",
                "recipe_promoted", "statistical_significance_claim", "adaptive_extension"):
        assert recipe[key] is False


def test_gates_require_both_losses_all_orders_and_never_promote(study):
    result = study.gates(cells(study), metric(3.0, 3.1))
    assert result["numerical_training_gate_pass"] is True
    assert all(value is True for value in result["clauses"].values())
    assert result["mean_seqk20_minus_batch"] == pytest.approx({"true_ce": .1, "lut_nll": .1})
    assert result["recipe_promoted"] is False and result["paid_execution_authorized"] is False
    assert result["requires_separate_successful_independent_recompute"] is True
    assert result["broader_study_authorized"] is False


@pytest.mark.parametrize("loss", ["true_ce", "lut_nll"])
@pytest.mark.parametrize("order", ["consecutive", "shuffle_0", "shuffle_1"])
def test_batch_initial_improvement_must_be_strict_in_every_order_and_loss(study, order, loss):
    rows, initial = cells(study), metric(3.0, 3.1)
    cell(rows, "batch4_k20", order)["final_mean"][loss] = initial[loss]
    result = study.gates(rows, initial)
    assert result["clauses"]["batch_improves_initial_every_order_both_losses"] is False
    assert result["numerical_training_gate_pass"] is False


@pytest.mark.parametrize("loss", ["true_ce", "lut_nll"])
@pytest.mark.parametrize("gain,passes", [(.01, True), (.01 - .5e-12, True), (.01 - 2e-12, False), (.009, False)])
def test_both_mean_gain_thresholds_use_only_declared_numerical_tolerance(study, loss, gain, passes):
    rows = cells(study, batch=2.0, seq20=2.1, seq18=2.1)
    for order in study.ORDERS:
        cell(rows, "seq_k20", order)["final_mean"][loss] = cell(rows, "batch4_k20", order)["final_mean"][loss] + gain
    result = study.gates(rows, metric(3.0, 3.1))
    assert result["clauses"]["batch_mean_gain_vs_seqk20_both_losses"] is passes
    assert result["numerical_training_gate_pass"] is passes


@pytest.mark.parametrize("loss", ["true_ce", "lut_nll"])
@pytest.mark.parametrize("regression,passes", [(0, True), (.5e-12, True), (2e-12, False), (.001, False)])
def test_positive_average_does_not_hide_an_order_regression(study, loss, regression, passes):
    rows = cells(study, batch=2.0, seq20=2.1, seq18=2.2)
    cell(rows, "batch4_k20")["final_mean"][loss] = cell(rows, "seq_k20")["final_mean"][loss] + regression
    result = study.gates(rows, metric(3.0, 3.1))
    assert result["clauses"]["batch_mean_gain_vs_seqk20_both_losses"] is True
    assert result["clauses"]["batch_no_order_regression_vs_seqk20_both_losses"] is passes


@pytest.mark.parametrize("loss", ["true_ce", "lut_nll"])
def test_batch_mean_must_not_be_worse_than_seq18_on_either_loss(study, loss):
    rows = cells(study)
    for order in study.ORDERS:
        cell(rows, "seq_k18", order)["final_mean"][loss] = cell(rows, "batch4_k20", order)["final_mean"][loss] - .001
    result = study.gates(rows, metric(3.0, 3.1))
    assert result["clauses"]["batch_mean_not_worse_than_seqk18_both_losses"] is False
    assert result["numerical_training_gate_pass"] is False


def test_seq18_comparison_is_mean_not_unrequested_per_order_gate(study):
    rows = cells(study, batch=2.0, seq20=2.2, seq18=2.1)
    cell(rows, "seq_k18")["final_mean"] = metric(1.95, 2.05)
    assert study.gates(rows, metric(3.0, 3.1))["numerical_training_gate_pass"] is True


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_arm", "wrong_order"])
def test_gates_reject_incomplete_duplicate_or_unselected_cells(study, mutation):
    rows = cells(study)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_arm":
        rows[-1]["arm"] = "batch4_k18"
    else:
        rows[-1]["order"] = "shuffle_2"
    with pytest.raises(ValueError):
        study.gates(rows, metric(3.0, 3.1))


@pytest.mark.parametrize("field", ["complete", "mechanics_pass"])
@pytest.mark.parametrize("value", [False, 1, None])
def test_failed_or_nonliteral_mechanics_do_not_produce_a_quality_verdict(study, field, value):
    rows = cells(study)
    rows[-1][field] = value
    with pytest.raises(ValueError, match="complete with checked mechanics"):
        study.gates(rows, metric(3.0, 3.1))


@pytest.mark.parametrize("location", ["initial", "endpoint"])
@pytest.mark.parametrize("field,value", [("true_ce", True), ("lut_nll", float("nan")),
                                         ("true_ce", float("inf")), ("lut_nll", -.1), ("accuracy", 1.1)])
def test_gate_metrics_reject_boolean_nonfinite_or_nonphysical_values(study, location, field, value):
    rows, initial = cells(study), metric(3.0, 3.1)
    (initial if location == "initial" else rows[-1]["final_mean"])[field] = value
    with pytest.raises(ValueError):
        study.gates(rows, initial)


def test_k20_checkpoint_reload_preserves_carry_across_block_commits(study, tmp_path):
    weights, remainder = mapping(10), mapping()
    gradient = mapping(100000)
    for commit in range(1, 4):
        result = study.accumulation.apply_block(weights, remainder, [gradient] * 4, k=20)
        study.state_record(tmp_path, commit * 4, commit, result["weights"], result["remainders"], 20, write=True)
        weights, remainder = study.state(tmp_path / f"state_{commit * 4:03d}.npz", 20)
        assert float(remainder["EMBED"][0, 0]) == (400000 * commit) % (1 << 20)
        assert float(weights["EMBED"][0, 0]) == 10 - (400000 * commit) // (1 << 20)
    assert float(weights["EMBED"][0, 0]) == 9
    assert float(remainder["EMBED"][0, 0]) == 151424


@pytest.mark.parametrize("field", ["k", "windows_consumed", "optimizer_commits", "bytes"])
def test_state_receipt_rejects_float_aliases_of_integer_metadata(study, tmp_path, field):
    weights, remainder = mapping(), mapping(400000)
    study.state_record(tmp_path, 4, 1, weights, remainder, 20, write=True)
    path = tmp_path / "state_004.json"
    row = study.read(path)
    row[field] = float(row[field])
    write_json(path, row)
    with pytest.raises(ValueError):
        study.state_record(tmp_path, 4, 1, weights, remainder, 20)


@pytest.mark.parametrize("field,bad", [("file", "../state_004.npz"), ("sha256", "0" * 64),
                                       ("weights_sha256", "1" * 64), ("remainders_sha256", "2" * 64),
                                       ("windows_consumed", True), ("optimizer_commits", 2), ("k", 18)])
def test_state_receipt_binds_exact_payload_names_hashes_and_counters(study, tmp_path, field, bad):
    weights, remainder = mapping(), mapping(400000)
    study.state_record(tmp_path, 4, 1, weights, remainder, 20, write=True)
    path = tmp_path / "state_004.json"
    row = study.read(path)
    row[field] = bad
    write_json(path, row)
    with pytest.raises(ValueError):
        study.state_record(tmp_path, 4, 1, weights, remainder, 20)


def test_state_receipt_write_is_exclusive(study, tmp_path):
    study.state_record(tmp_path, 4, 1, mapping(), mapping(), 20, write=True)
    before = tree_bytes(tmp_path)
    with pytest.raises(FileExistsError):
        study.state_record(tmp_path, 4, 1, mapping(1), mapping(), 20, write=True)
    assert tree_bytes(tmp_path) == before


@pytest.fixture
def fake_run(study, tmp_path, monkeypatch):
    """Actual new binding/lifecycle, opaque synthetic predecessor and body.

    The separate replay test module owns all numerical loop/readout coverage.
    This fixture deliberately replaces execute and its payload membership only.
    """
    prior_root, step_root = tmp_path / "prior", tmp_path / "step"
    prior_root.mkdir()
    step_root.mkdir()
    prior_record, step_record = tmp_path / "prior.json", tmp_path / "step.json"
    corpus, initial = tmp_path / "synthetic.txt", tmp_path / "initial.i32"
    corpus.write_bytes(b"synthetic metadata only; not a scientific corpus")
    initial.write_bytes(b"opaque synthetic initial state; never decoded")
    dataset = {"path": str(corpus), "synthetic": True}
    persistent = {"dataset": dataset, "initial_state": str(initial), "synthetic": True}
    write_json(prior_root / "manifest.json", {"schema": study.parent.SCHEMA, "binding": persistent})
    write_json(step_root / "manifest.json", {"schema": study.step_study.SCHEMA,
        "binding": {"prior_run": str(prior_root), "prior_record": str(prior_record)}})
    for directory, record in ((prior_root, prior_record), (step_root, step_record)):
        write_json(directory / "summary.json", {"complete": True, "synthetic": True})
        write_json(directory / "artifact_manifest.json", study.prior.artifact_manifest(directory))
        write_json(record, {"complete": True, "synthetic": True})

    def predecessor_check(directory, record, recompute=False):
        assert recompute is False
        study.equal(study.read(directory / "artifact_manifest.json"),
                    study.prior.artifact_manifest(directory), "synthetic parent integrity")
        return study.read(record)

    monkeypatch.setattr(study.parent, "check", predecessor_check)
    monkeypatch.setattr(study.step_study, "check", predecessor_check)

    def corpus_plan(path, steps, evaluation):
        assert Path(path) == corpus and (steps, evaluation) == (1024, 88)
        return b"synthetic", copy.deepcopy(dataset)

    monkeypatch.setattr(study.pilot, "corpus_plan", corpus_plan)
    sources = {"synthetic-source": "frozen"}
    monkeypatch.setattr(study, "source_identity", lambda: copy.deepcopy(sources))
    plan = tmp_path / "synthetic_plan.md"
    plan.write_text("Synthetic lifecycle declaration; no model run\n")
    monkeypatch.setattr(study, "PLAN", plan)
    controls = SimpleNamespace(executions=[], fail=False)
    payload_name = "synthetic_payload.npz"
    payload = b"opaque synthetic body: no model/gradient/scoring computation"

    def execute(manifest, output, raw, *, write=False, recompute=False):
        assert raw == b"synthetic"
        controls.executions.append((write, recompute))
        if controls.fail:
            raise RuntimeError("controlled synthetic body failure")
        path = output / payload_name
        if write:
            with path.open("xb") as stream:
                stream.write(payload)
        study.equal(path.read_bytes(), payload, "opaque body unchanged")
        rows = cells(study)
        return {"schema": study.SCHEMA, "complete": True, "recipe": study.recipe(),
                "source_hashes": manifest["source_hashes"], "cells": rows,
                "gates": study.gates(rows, metric(3.0, 3.1)), "declared_cells": 9, "completed_cells": 9,
                "full_gradient_queries": 288, "optimizer_commits": 216, "score_readouts": 434,
                "new_gradient_pair_exposures": 36864, "recipe_promoted": False,
                "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
                "paid_execution_authorized": False, "scope": "synthetic lifecycle fixture, not evidence"}

    monkeypatch.setattr(study, "execute", execute)
    monkeypatch.setattr(study, "expected_artifacts", lambda: {
        "manifest.json", "protocol.json", "declaration.md", "summary.json", payload_name})
    config = SimpleNamespace(output=tmp_path / "run", record=tmp_path / "record.json", corpus=corpus,
        prior_run=prior_root, prior_record=prior_record, step_run=step_root, step_record=step_record)
    return SimpleNamespace(config=config, controls=controls, sources=sources, dataset=dataset,
        initial=initial, parent_bytes=tree_bytes(prior_root), step_bytes=tree_bytes(step_root))


def test_mocked_body_full_publication_and_readonly_replay(study, fake_run):
    config = fake_run.config
    result = study.run(config)
    assert result["complete"] is True and result["completed_cells"] == 9
    assert result["gates"]["numerical_training_gate_pass"] is True
    assert result["gates"]["broader_study_authorized"] is False
    assert result["artifact_files_verified"] == 5  # explicitly reduced lifecycle fixture only
    before, public = tree_bytes(config.output), config.record.read_bytes()
    assert study.check(config.output, config.record, recompute=True) == result
    assert tree_bytes(config.output) == before and config.record.read_bytes() == public
    assert tree_bytes(config.prior_run) == fake_run.parent_bytes
    assert tree_bytes(config.step_run) == fake_run.step_bytes
    assert fake_run.controls.executions == [(True, False), (False, False), (False, True)]


def test_recursive_binding_has_both_actual_predecessors_and_exact_initial_state(study, fake_run):
    config = fake_run.config
    binding, raw = study.bind_inputs(config)
    assert raw == b"synthetic"
    expected = {config.prior_record, config.step_record, config.corpus, fake_run.initial}
    expected |= {root / name for root in (config.prior_run, config.step_run)
                 for name in ("manifest.json", "summary.json", "artifact_manifest.json")}
    assert {Path(item["path"]) for item in binding["inputs"]} == expected
    assert len(binding["inputs"]) == len(expected) == 10
    assert binding["initial_state"] == str(fake_run.initial)
    assert binding["persistent_binding"] == study.read(config.prior_run / "manifest.json")["binding"]
    for item in binding["inputs"]:
        data = Path(item["path"]).read_bytes()
        assert item["bytes"] == len(data) and item["sha256"] == hashlib.sha256(data).hexdigest()


def test_step_rationale_cannot_bind_an_unrelated_parent(study, fake_run):
    config = fake_run.config
    path = config.step_run / "manifest.json"
    row = study.read(path)
    row["binding"]["prior_run"] = str(config.prior_run) + "_unrelated"
    write_json(path, row)
    write_json(config.step_run / "artifact_manifest.json", study.prior.artifact_manifest(config.step_run))
    with pytest.raises(ValueError, match="identical accumulation predecessor"):
        study.bind_inputs(config)


@pytest.mark.parametrize("mutation", ["input_bytes", "missing_input", "initial_redirect", "source", "recipe"])
def test_frozen_verifies_hashes_and_canonical_recursive_membership(study, fake_run, mutation):
    binding, _ = study.bind_inputs(fake_run.config)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(),
                "source_hashes": copy.deepcopy(fake_run.sources), "binding": copy.deepcopy(binding)}
    study.frozen(manifest)
    if mutation == "input_bytes":
        fake_run.initial.write_bytes(b"changed initial payload")
    elif mutation == "missing_input":
        manifest["binding"]["inputs"].pop()
    elif mutation == "initial_redirect":
        manifest["binding"]["initial_state"] += ".other"
    elif mutation == "source":
        manifest["source_hashes"]["synthetic-source"] = "changed"
    else:
        manifest["recipe"]["arm_config"]["batch4_k20"]["k"] = 18
    with pytest.raises(ValueError):
        study.frozen(manifest)


@pytest.mark.parametrize("failure", ["binding", "body", "final_check", "after_link", "foreign_target", "disarm", "staging"])
def test_failure_timeout_interrupt_and_cleanup_retract_only_owned_receipt(study, fake_run, monkeypatch, failure):
    config = fake_run.config
    previous = study.signal.getsignal(study.signal.SIGALRM)

    def fail(*args, **kwargs):
        raise TimeoutError("controlled " + failure)

    if failure == "binding":
        monkeypatch.setattr(study, "bind_inputs", fail)
    elif failure == "body":
        fake_run.controls.fail = True
    elif failure == "final_check":
        monkeypatch.setattr(study, "check", fail)
    elif failure in ("after_link", "foreign_target"):
        original = study.os.link

        def fail_link(source, target):
            if failure == "after_link":
                original(source, target)
                raise KeyboardInterrupt("controlled post-link interruption")
            Path(target).write_bytes(b"unrelated user data")
            return original(source, target)

        monkeypatch.setattr(study.os, "link", fail_link)
    elif failure == "disarm":
        original, failed = study.signal.setitimer, []

        def fail_disarm(which, seconds, *args):
            if seconds == 0 and not failed:
                failed.append(True)
                fail()
            return original(which, seconds, *args)

        monkeypatch.setattr(study.signal, "setitimer", fail_disarm)
    else:
        original, failed = Path.unlink, []

        def fail_unlink(path, *args, **kwargs):
            if path.name == "receipt.json" and not failed:
                failed.append(True)
                fail()
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises((TimeoutError, RuntimeError, OSError, KeyboardInterrupt)):
        study.run(config)
    assert study.read(config.output / "failure.json")["complete"] is False
    assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    if failure == "foreign_target":
        assert config.record.read_bytes() == b"unrelated user data"
    else:
        assert not config.record.exists()
    assert (config.output / "rejected_public.json").exists() is (failure in {"final_check", "after_link", "disarm", "staging"})
    assert not [path for path in config.record.parent.iterdir() if path.name.startswith(".tt-")]
    assert tree_bytes(config.prior_run) == fake_run.parent_bytes
    assert tree_bytes(config.step_run) == fake_run.step_bytes


@pytest.mark.parametrize("mutation", ["summary", "payload", "extra", "failure", "public"])
def test_final_check_rejects_tampered_payload_or_complete_claim(study, fake_run, mutation):
    config = fake_run.config
    study.run(config)
    if mutation == "payload":
        (config.output / "synthetic_payload.npz").write_bytes(b"tampered body")
    elif mutation in ("extra", "failure"):
        write_json(config.output / ("extra.json" if mutation == "extra" else "failure.json"), {"complete": False})
    else:
        path = config.output / "summary.json" if mutation == "summary" else config.record
        row = study.read(path)
        row["gates"]["broader_study_authorized"] = True
        write_json(path, row)
    with pytest.raises(ValueError):
        study.check(config.output, config.record)


@pytest.mark.parametrize("target", ["public", "seal"])
def test_final_check_revalidates_raw_receipt_bytes_after_replay(study, fake_run, monkeypatch, target):
    config = fake_run.config
    study.run(config)
    original, calls = study.frozen, []
    path = config.record if target == "public" else config.output / "artifact_manifest.json"

    def change_at_end(manifest):
        original(manifest)
        calls.append(True)
        if len(calls) == 2:
            path.write_bytes(path.read_bytes() + b"\n")

    monkeypatch.setattr(study, "frozen", change_at_end)
    with pytest.raises(ValueError, match="bytes unchanged"):
        study.check(config.output, config.record)


@pytest.mark.parametrize("existing", ["output", "record"])
def test_existing_user_outputs_are_never_overwritten(study, fake_run, existing):
    path = getattr(fake_run.config, existing)
    path.write_bytes(b"existing user bytes")
    with pytest.raises(FileExistsError):
        study.run(fake_run.config)
    assert path.read_bytes() == b"existing user bytes"
    assert not fake_run.controls.executions


@pytest.mark.parametrize("nested", ["same", "public_in_raw", "historical_raw", "historical_public"])
def test_new_outputs_cannot_overlap_or_modify_either_history(study, fake_run, nested):
    config = fake_run.config
    if nested == "same":
        config.record = config.output
    elif nested == "public_in_raw":
        config.record = config.output / "record.json"
    elif nested == "historical_raw":
        config.output = config.prior_run / "new_run"
    else:
        config.record = config.step_run / "new_record.json"
    with pytest.raises(ValueError):
        study.run(config)
    assert not fake_run.controls.executions


def test_actual_artifact_membership_has_all_nine_cells_and_fixed_counters(study):
    files = study.expected_artifacts()
    assert len(files) == 1401
    assert len([name for name in files if "/query_" in name]) == 288
    assert len([name for name in files if "/state_" in name and name.endswith(".npz")]) == 225
    assert len([name for name in files if "/state_" in name and name.endswith(".json")]) == 225
    assert len([name for name in files if "/transition_" in name]) == 216
    assert len([name for name in files if "score_" in name]) == 434
    for arm in study.ARMS:
        for order in study.ORDERS:
            prefix = f"{arm}/{order}/"
            assert {prefix + f"query_{index:03d}.npz" for index in range(32)} <= files
            assert prefix + "state_000.npz" in files and prefix + "state_032.npz" in files
