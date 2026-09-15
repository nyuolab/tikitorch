"""Synthetic head-weighting contracts and lifecycle; no corpus study run."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def helper():
    return importlib.import_module("manuscript.tools.fpga_lm_head_weighting")


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_head_weighting")


def mapping(value=0):
    shapes = {"EMBED": (3, 2), "MH": (2, 2, 3), "MP": (2, 2, 2),
              "MB": (2, 2), "W_HEAD": (3, 2)}
    return {name: torch.full(shape, float(value), dtype=torch.float64)
            for name, shape in shapes.items()}


def metric(ce=2.0, lut=2.1, accuracy=.25):
    return {"true_ce": ce, "lut_nll": lut, "accuracy": accuracy}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def test_helper_source_identity_binds_its_actual_source(helper):
    root = Path(__file__).resolve().parents[1]
    sources = helper.source_identity()
    path = "manuscript/tools/fpga_lm_head_weighting.py"
    assert sources[path] == hashlib.sha256((root / path).read_bytes()).hexdigest()


def test_runner_source_identity_includes_both_new_test_files(study):
    root = Path(__file__).resolve().parents[1]
    sources = study.source_identity()
    for path in ("manuscript/tools/fpga_lm_head_weighting.py",
                 "manuscript/tools/diagnose_fpga_lm_head_weighting.py",
                 "tests/test_fpga_lm_head_weighting.py",
                 "tests/test_fpga_lm_head_weighting_replay.py"):
        assert sources[path] == hashlib.sha256((root / path).read_bytes()).hexdigest()


def oracle_head(logits, targets, c, arm):
    """Separate scalar integer-ROM construction; no helper/native call."""
    q16 = 65536
    rom = [round(q16 * 2.0 ** (-index / 16.0)) for index in range(256)]
    rounded = []
    for row in logits.tolist():
        top = max(row)
        energies = [rom[min(255, math.floor((top - value) / 4))] for value in row]
        shift = 0
        while sum(energies) >> shift > q16:
            shift += 1
        rounded.append([energy >> shift for energy in energies])
    rounded = torch.tensor(rounded, dtype=torch.int64)
    totals = rounded.sum(dim=1)
    masses = totals.to(torch.float64) / q16
    probabilities = rounded.to(torch.float64) / totals[:, None]
    onehot = torch.nn.functional.one_hot(torch.tensor(targets), logits.shape[1]).to(torch.float64)
    native = rounded.to(torch.float64) - totals[:, None] * onehot
    cotangent = ((c / masses.mean()) * native if arm == "relative_mass"
                 else c * q16 * (probabilities - onehot))
    return {"rounded_energies": rounded, "rounded_totals": totals, "mass_factor": masses,
            "truncated_probabilities": probabilities, "relative_weights": masses / masses.mean(),
            "row_weights": masses / masses.mean() if arm == "relative_mass" else torch.ones_like(masses),
            "cotangent": cotangent}


@pytest.mark.parametrize("arm", ["relative_mass", "uniform_mass"])
@pytest.mark.parametrize("independent", [False, True])
def test_head_matches_independent_scalar_rom_formula_and_detaches(helper, arm, independent):
    logits = torch.tensor([[0., 0., 0.], [2.75, -25., -117.25], [0., -1000., -2000.]], dtype=torch.float64)
    before = logits.clone()
    targets, c = [0, 2, 1], .72
    result = helper.head(logits, targets, c, arm, independent=independent)
    expected = oracle_head(logits, targets, c, arm)
    for key, value in expected.items():
        assert result[key].dtype == value.dtype and result[key].shape == value.shape
        if value.dtype == torch.int64:
            assert torch.equal(result[key], value)
        else:
            torch.testing.assert_close(result[key], value, atol=1e-9, rtol=1e-11)
        assert result[key].grad_fn is None and not result[key].requires_grad
        assert result[key].data_ptr() != logits.data_ptr()
    assert torch.equal(logits, before)
    assert torch.unique(result["mass_factor"]).numel() > 1
    assert float(result["relative_weights"].mean()) == pytest.approx(1.0, abs=1e-15)
    torch.testing.assert_close(result["cotangent"].sum(dim=1), torch.zeros(3, dtype=torch.float64), atol=1e-9, rtol=0)


def test_common_mean_head_scale_does_not_assert_equal_gradient_norms(helper):
    logits = torch.tensor([[0., 0., 0.], [0., -16., -1000.], [0., -1000., -2000.]], dtype=torch.float64)
    targets = [0, 2, 1]
    relative = helper.head(logits, targets, .7, "relative_mass")
    uniform = helper.head(logits, targets, .7, "uniform_mass")
    assert float(relative["row_weights"].mean()) == pytest.approx(1.0)
    assert torch.equal(uniform["row_weights"], torch.ones(3, dtype=torch.float64))
    assert not torch.equal(relative["cotangent"], uniform["cotangent"])
    assert not math.isclose(float(relative["cotangent"].norm()), float(uniform["cotangent"].norm()), rel_tol=1e-6)


@pytest.mark.parametrize("bad", [True, 1, 0., -.1, 1.00001, float("nan"), float("inf"), "0.7",
                                  torch.tensor(.7, dtype=torch.float64)])
def test_frozen_c_is_a_literal_bounded_positive_float(helper, bad):
    with pytest.raises((ValueError, TypeError)):
        helper.head(torch.zeros((2, 3), dtype=torch.float64), [0, 1], bad, "uniform_mass")


@pytest.mark.parametrize("targets", [[True, 1], [0., 1.], [-1, 1], [0, 3], [0], [[0], [1]], []])
def test_head_target_ids_have_strict_complete_geometry(helper, targets):
    with pytest.raises((ValueError, TypeError)):
        helper.head(torch.zeros((2, 3), dtype=torch.float64), targets, .7, "relative_mass")


@pytest.mark.parametrize("mutation", ["float32", "int64", "rank", "empty", "nan", "inf", "clip"])
def test_head_logit_contract_rejects_bad_dtype_geometry_and_range(helper, mutation):
    logits = torch.zeros((2, 3), dtype=torch.float64)
    if mutation in ("float32", "int64"):
        logits = logits.to(getattr(torch, mutation))
    elif mutation == "rank":
        logits = logits.flatten()
    elif mutation == "empty":
        logits = logits[:0]
    else:
        logits[0, 0] = {"nan": float("nan"), "inf": float("inf"), "clip": 32768.}[mutation]
    with pytest.raises((ValueError, TypeError, OverflowError)):
        helper.head(logits, [0, 1], .7, "relative_mass")


@pytest.mark.parametrize("arm", [None, True, "native", "exact_ce", "relative_mass "])
def test_head_accepts_only_two_declared_weighting_arms(helper, arm):
    with pytest.raises((ValueError, TypeError)):
        helper.head(torch.zeros((2, 3), dtype=torch.float64), [0, 1], .7, arm)


def test_fractional_step_is_fresh_unrounded_and_has_no_hidden_carry(helper):
    weights, gradient = mapping(1), mapping(.25)
    before = {key: value.clone() for key, value in weights.items()}
    result = helper.step(weights, gradient, k=18)
    for key in weights:
        assert torch.equal(result[key], weights[key] - .25 / (1 << 18))
        assert torch.equal(weights[key], before[key])
        assert result[key].data_ptr() != weights[key].data_ptr()
        assert result[key].data_ptr() != gradient[key].data_ptr()
        assert not result[key].requires_grad
    assert set(result) == set(weights)


@pytest.mark.parametrize("bad", [True, False, 18.0, "18", 0, 17, 19, 20, None])
def test_step_rejects_any_unapproved_scale(helper, bad):
    with pytest.raises((ValueError, TypeError)):
        helper.step(mapping(), mapping(), k=bad)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "rank", "width", "requires_grad", "nan", "range"])
def test_weights_require_five_detached_fp64_natural_regions(helper, mutation):
    weights = mapping()
    if mutation == "missing":
        weights.pop("MH")
    elif mutation == "extra":
        weights["extra"] = torch.zeros(1, dtype=torch.float64)
    elif mutation == "dtype":
        weights["EMBED"] = weights["EMBED"].to(torch.float32)
    elif mutation == "rank":
        weights["EMBED"] = weights["EMBED"].flatten()
    elif mutation == "width":
        weights["MH"] = torch.zeros((2, 3, 3), dtype=torch.float64)
    elif mutation == "requires_grad":
        weights["EMBED"].requires_grad_(True)
    else:
        weights["EMBED"][0, 0] = float("nan") if mutation == "nan" else 32767.1
    with pytest.raises((ValueError, TypeError)):
        helper.validate_weights(weights)


def test_step_overflow_is_transactional_even_in_last_parameter_region(helper):
    weights, gradient = mapping(), mapping(1)
    weights["W_HEAD"][0, 0] = -32768
    before = {key: value.clone() for key, value in weights.items()}
    with pytest.raises((ValueError, OverflowError)):
        helper.step(weights, gradient, k=18)
    assert all(torch.equal(weights[key], before[key]) for key in weights)


def tiny_weights():
    weights = mapping()
    weights["EMBED"] = torch.tensor([[0., -8.], [-8., 0.], [1.5, -4.5]], dtype=torch.float64)
    weights["MP"].fill_(-32)
    for feature in range(2):
        weights["MP"][:, feature, feature] = 0
    weights["W_HEAD"].fill_(-32)
    for label in range(3):
        weights["W_HEAD"][label, label % 2] = 0
    return weights


@pytest.mark.parametrize("arm", ["relative_mass", "uniform_mass"])
def test_tiny_manual_and_autograd_real_centered_vjp_agree_and_detach(helper, arm):
    weights = tiny_weights()
    before = {key: value.clone() for key, value in weights.items()}
    tokens, targets = torch.tensor([0, 1, 2, 0], dtype=torch.int64), [1, 2, 0, 2]
    base = helper.forward(weights, tokens)
    head = helper.head(base["logits"], targets, .7, arm)
    manual = helper.gradient(weights, tokens, base, head, independent=False)
    automatic = helper.gradient(weights, tokens, base, head, independent=True)
    assert set(manual) == set(automatic) == set(weights)
    assert any(bool(value.any()) for value in manual.values())
    for key in weights:
        torch.testing.assert_close(manual[key], automatic[key], atol=1e-9, rtol=1e-11)
        assert manual[key].shape == weights[key].shape and manual[key].dtype == torch.float64
        assert not manual[key].requires_grad and manual[key].grad_fn is None
        assert not automatic[key].requires_grad and automatic[key].grad_fn is None
        assert manual[key].data_ptr() != weights[key].data_ptr()
        assert torch.equal(weights[key], before[key])


@pytest.mark.parametrize("context", [True, False])
def test_tiny_independent_fractional_forward_matches_complete_trace(helper, context):
    weights = tiny_weights()
    tokens = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    normal = helper.forward(weights, tokens, use_context=context)
    independent = helper.forward(weights, tokens, use_context=context, independent=True)
    left, right = helper.trace(normal), helper.trace(independent)
    assert set(left) == set(right) == {"logits", "centering_offsets", "AD_HEAD", "PROJ_IDX", "TAP_IDX"}
    for key in left:
        if left[key].dtype == torch.int64:
            assert torch.equal(left[key], right[key])
        else:
            torch.testing.assert_close(left[key], right[key], atol=1e-9, rtol=1e-11)
    if not context:
        assert torch.count_nonzero(left["TAP_IDX"]) == 0
    for layer in range(2):
        assert torch.all(left["TAP_IDX"][layer] * (1 << layer) <= torch.arange(4)[:, None])


def test_tiny_forward_ties_choose_first_and_future_token_cannot_change_past(helper):
    tokens = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    for independent in (False, True):
        tied = helper.trace(helper.forward(mapping(), tokens, independent=independent))
        for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX"):
            assert torch.count_nonzero(tied[key]) == 0
        weights = tiny_weights()
        changed = tokens.clone()
        changed[-1] = 2
        before = helper.forward(weights, tokens, independent=independent)["logits"]
        after = helper.forward(weights, changed, independent=independent)["logits"]
        assert torch.equal(before[:-1], after[:-1])


@pytest.mark.parametrize("independent", [False, True])
def test_forward_rejects_fractional_intermediate_clipping(helper, independent):
    weights = mapping()
    weights["EMBED"].fill_(20000.25)
    weights["MB"].fill_(20000.25)
    with pytest.raises((ValueError, OverflowError)):
        helper.forward(weights, torch.tensor([0, 1], dtype=torch.int64), independent=independent)


def test_near_keeps_integer_routes_exact_and_float_dtype_shape_identity(helper):
    tensor = torch.tensor([1.], dtype=torch.float64)
    helper.near(tensor, tensor + 1e-10, "tiny float")
    helper.near({"x": tensor}, {"x": tensor + 1e-10}, "tiny mapping")
    for left, right in ((torch.tensor([0], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)),
                        (tensor, tensor + 1e-5), (tensor, tensor.to(torch.float32)),
                        (tensor, tensor.reshape(1, 1)), ({"x": tensor}, {"y": tensor})):
        with pytest.raises((ValueError, TypeError, AssertionError)):
            helper.near(left, right, "controlled mismatch")


@pytest.mark.parametrize("independent", [False, True])
def test_head_rejects_zero_truncated_mass(helper, monkeypatch, independent):
    monkeypatch.setattr(helper.numeric, "ROM", tuple(0 for _ in range(256)))
    with pytest.raises((ValueError, OverflowError)):
        helper.head(torch.zeros((2, 3), dtype=torch.float64), [0, 1], .7, "uniform_mass", independent=independent)


def test_head_never_differentiates_through_logit_dependent_masses(helper):
    logits = torch.tensor([[0., -16., -1000.], [0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    for arm in ("relative_mass", "uniform_mass"):
        result = helper.head(logits, [2, 1], .7, arm)
        assert all(not value.requires_grad and value.grad_fn is None for value in result.values())
    assert logits.grad is None


def test_gradient_rejects_an_attached_cotangent(helper):
    weights, tokens = tiny_weights(), torch.tensor([0, 1, 2], dtype=torch.int64)
    base = helper.forward(weights, tokens)
    head = helper.head(base["logits"], [1, 2, 0], .7, "uniform_mass")
    head["cotangent"].requires_grad_(True)
    with pytest.raises(ValueError, match="detached"):
        helper.gradient(weights, tokens, base, head)


def test_runner_state_is_fractional_weights_only_and_has_exact_receipt(study, tmp_path):
    weights = mapping(.125)
    receipt = study.state_record(tmp_path, 1, weights, write=True)
    assert receipt["contains_optimizer_remainder"] is False
    assert receipt["native_state"] is False and receipt["dtype"] == "float64"
    assert receipt["divisor_k"] == 18 and receipt["optimizer_commits"] == receipt["windows_consumed"] == 1
    before = tree_bytes(tmp_path)
    assert study.state_record(tmp_path, 1, weights) == receipt
    assert tree_bytes(tmp_path) == before
    with pytest.raises(FileExistsError):
        study.state_record(tmp_path, 1, weights, write=True)


@pytest.fixture
def ancestry(study, tmp_path, monkeypatch):
    """Real new closure/calibration logic over tiny synthetic saved arrays."""
    root, ancestor = tmp_path / "prior", tmp_path / "ancestor"
    (root / "shared").mkdir(parents=True)
    ancestor.mkdir()
    corpus, initial = tmp_path / "synthetic.txt", tmp_path / "initial.i32"
    corpus.write_bytes(b"synthetic saved-logit calibration fixture, never a corpus run")
    initial.write_bytes(b"opaque state; no native decoding")
    record, audit = tmp_path / "prior.json", tmp_path / "prior-audit.json"
    dataset = {"path": str(corpus), "synthetic": True}
    write_json(ancestor / "manifest.json", {"schema": study.normalized.SCHEMA})
    (ancestor / "payload.bin").write_bytes(b"nested exact ancestor payload")
    write_json(ancestor / "artifact_manifest.json", study.prior.artifact_manifest(ancestor))
    binding = {"dataset": dataset, "initial_state": str(initial), "prior_run": str(ancestor),
               "inputs": [study.old.file_identity(corpus), study.old.file_identity(initial)]}
    write_json(root / "manifest.json", {"schema": study.normalized.SCHEMA, "binding": binding})
    write_json(root / "summary.json", {"complete": True, "synthetic": True})
    logits = torch.tensor([[0., 0., 0.], [0., -16., -1000.]], dtype=torch.float64)
    for index in range(32):
        study.parent.save(root / "shared" / f"score_initial_{index:03d}_1.npz", {"logits": logits})
    write_json(root / "artifact_manifest.json", study.prior.artifact_manifest(root))
    write_json(record, {"complete": True, "gates": {"numerical_training_gate_pass": False}})
    source_name = "manuscript/tools/diagnose_fpga_lm_normalized_accumulation.py"
    source_hash = hashlib.sha256((study.REPO / source_name).read_bytes()).hexdigest()
    monkeypatch.setattr(study.normalized, "source_identity", lambda: {source_name: source_hash})
    receipt = {"schema": "tikitorch.fpga_lm_normalized_accumulation.cli_audit_receipt.v1",
        "complete": True, "independent_recompute": True, "exit_code": 0,
        "parent_record_sha256": study.old.digest(record.read_bytes()),
        "run_manifest_sha256": study.old.digest((root / "manifest.json").read_bytes()),
        "artifact_manifest_sha256": study.old.digest((root / "artifact_manifest.json").read_bytes()),
        "runner_sha256": source_hash, "source_files_verified": 1,
        "independent_full_gradient_queries": 288, "independent_score_readouts": 434,
        "complete_saved_states_verified": 225, "optimizer_commits_replayed": 216,
        "complete_parent_ancestry_verified": True, "source_input_raw_seal_public_bytes_unchanged": True,
        "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
        "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
        "broader_study_authorized": False, "numerical_training_gate_pass": False,
        "deadline_seconds": 3600, "direct_input_files_verified": len(binding["inputs"]),
        "artifact_files_verified": len(study.prior.artifact_manifest(root)),
        "artifact_payload_bytes": sum(row["bytes"] for row in study.prior.artifact_manifest(root)),
        "ordered_progress_blocks_verified": 72, "recomputed_gradient_pair_observations": 36864,
        "command": ["python", source_name, "--check", "--recompute", "--output", str(root),
                    "--record", str(record)],
        "started_at_utc": "2026-09-11 12:00:00 UTC", "completed_at_utc": "2026-09-11 12:30:00 UTC"}
    write_json(audit, receipt)
    calls = SimpleNamespace(semantic=[], corpus=[], windows=[])

    def semantic(path, public, recompute=False):
        assert Path(path) == root and Path(public) == record and recompute is False
        calls.semantic.append(True)
        return study.read(record)

    def corpus_plan(path, steps, validation):
        assert Path(path) == corpus and (steps, validation) == (1024, 88)
        calls.corpus.append(True)
        return b"synthetic", copy.deepcopy(dataset)

    def window(raw, supplied, partition, index):
        assert raw == b"synthetic" and supplied == dataset and partition == "train"
        assert type(index) is int and 0 <= index < 32
        calls.windows.append(index)
        return [0, 1], [1, 2]

    def forbidden(*args, **kwargs):
        pytest.fail("unapproved model/native execution in closure/lifecycle fixture")

    monkeypatch.setattr(study.normalized, "check", semantic)
    monkeypatch.setattr(study.pilot, "corpus_plan", corpus_plan)
    monkeypatch.setattr(study.pilot, "window", window)
    monkeypatch.setattr(study.helper, "forward", forbidden)
    monkeypatch.setattr(study.helper, "gradient", forbidden)
    monkeypatch.setattr(study.old.native, "decode_state", forbidden)
    monkeypatch.setattr(study.old.native, "EFSGD", forbidden)
    config = SimpleNamespace(prior_run=root, prior_record=record, prior_audit=audit, corpus=corpus,
        output=tmp_path / "new_run", record=tmp_path / "new_record.json")
    return SimpleNamespace(config=config, root=root, ancestor=ancestor, record=record, audit=audit,
        dataset=dataset, calls=calls, initial=initial, logits=logits, before=tree_bytes(root))


def test_calibration_uses_all32_saved_initial_windows_and_no_target_tuning(study, ancestry):
    binding, raw, validation = study.bind_inputs(ancestry.config)
    assert raw == b"synthetic"
    expected = oracle_head(ancestry.logits, [1, 2], 1.0, "uniform_mass")["mass_factor"].mean().item()
    calibration = binding["calibration"]
    assert calibration["c"] == expected and type(calibration["c"]) is float
    assert calibration["tokens"] == 64  # synthetic two-token rows, not a real corpus
    assert calibration["window_indices"] == list(range(32))
    assert len(calibration["source_logits"]) == 32
    assert ancestry.calls.windows == list(range(32))
    assert len(ancestry.calls.semantic) == 1
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(),
                "source_hashes": study.source_identity(), "binding": binding}
    for _ in range(3):
        study.frozen(manifest, validation)
    assert len(ancestry.calls.semantic) == 1  # closure rehashing does not rerun semantics
    assert tree_bytes(ancestry.root) == ancestry.before


def test_validation_proof_does_not_alias_working_manifest(study, ancestry):
    binding, _, validation = study.bind_inputs(ancestry.config)
    original = validation.binding["calibration"]["c"]
    binding["calibration"]["c"] = original / 2.0
    assert validation.binding["calibration"]["c"] == original
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(),
                "source_hashes": study.source_identity(), "binding": binding}
    with pytest.raises(ValueError):
        study.frozen(manifest, validation)


def test_versioned_ancestor_seal_is_verified_without_rewriting_history(study, ancestry):
    seal = {"schema": study.normalized.SCHEMA,
            "files": study.prior.artifact_manifest(ancestry.ancestor)}
    write_json(ancestry.ancestor / "artifact_manifest.json", seal)
    before = tree_bytes(ancestry.ancestor)
    snapshot = study.ancestry_snapshot(ancestry.root, ancestry.record, ancestry.audit)
    assert snapshot["roots"][str(ancestry.ancestor)]["artifacts"] == seal["files"]
    assert tree_bytes(ancestry.ancestor) == before
    seal["schema"] = "tikitorch.fpga_lm.unrelated.v1"
    write_json(ancestry.ancestor / "artifact_manifest.json", seal)
    with pytest.raises(ValueError):
        study.ancestry_snapshot(ancestry.root, ancestry.record, ancestry.audit)


def test_legacy_absolute_file_receipt_is_part_of_rehashed_closure(study, ancestry, tmp_path):
    external = tmp_path / "legacy-public.json"
    write_json(external, {"complete": True, "historical": True})
    ancestor_manifest = study.read(ancestry.ancestor / "manifest.json")
    ancestor_manifest["old_evidence"] = {"file": str(external),
                                          "sha256": study.old.digest(external.read_bytes())}
    write_json(ancestry.ancestor / "manifest.json", ancestor_manifest)
    write_json(ancestry.ancestor / "artifact_manifest.json", study.prior.artifact_manifest(ancestry.ancestor))
    snapshot = study.ancestry_snapshot(ancestry.root, ancestry.record, ancestry.audit)
    assert study.old.file_identity(external) in snapshot["files"]
    external.write_bytes(external.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        study.ancestry_snapshot(ancestry.root, ancestry.record, ancestry.audit)


@pytest.mark.parametrize("tamper", ["nested_payload", "new_nested_file", "public_bytes", "external_input", "local_token"])
def test_invocation_local_validation_still_rehashes_full_ancestry(study, ancestry, tamper):
    binding, _, validation = study.bind_inputs(ancestry.config)
    manifest = {"schema": study.SCHEMA, "recipe": study.recipe(),
                "source_hashes": study.source_identity(), "binding": binding}
    if tamper == "nested_payload":
        (ancestry.ancestor / "payload.bin").write_bytes(b"changed ancestor")
        write_json(ancestry.ancestor / "artifact_manifest.json", study.prior.artifact_manifest(ancestry.ancestor))
    elif tamper == "new_nested_file":
        (ancestry.ancestor / "extra.bin").write_bytes(b"unsealed member")
    elif tamper == "public_bytes":
        ancestry.record.write_bytes(ancestry.record.read_bytes() + b"\n")
    elif tamper == "external_input":
        ancestry.initial.write_bytes(b"changed external initial")
    else:
        validation = {"binding": binding, "sources": manifest["source_hashes"]}
    with pytest.raises(ValueError):
        study.frozen(manifest, validation)


@pytest.mark.parametrize("field,value", [("exit_code", 1), ("independent_recompute", 1),
                                         ("independent_full_gradient_queries", 287),
                                         ("parent_record_sha256", "0" * 64),
                                         ("numerical_training_gate_pass", True),
                                         ("artifact_files_verified", 1),
                                         ("ordered_progress_blocks_verified", 71),
                                         ("completed_at_utc", "2026-09-11 13:00:01 UTC")])
def test_historical_failed_quality_gate_is_allowed_but_audit_claims_must_match(study, ancestry, field, value):
    study._audit_receipt(ancestry.root, ancestry.record, ancestry.audit)
    receipt = study.read(ancestry.audit)
    receipt[field] = value
    write_json(ancestry.audit, receipt)
    with pytest.raises(ValueError):
        study._audit_receipt(ancestry.root, ancestry.record, ancestry.audit)


def gate_cells(study):
    return [{"arm": arm, "order": order, "complete": True, "mechanics_pass": True,
             "final_mean": metric(2.1 if arm == "relative_mass" else 2.0,
                                   2.2 if arm == "relative_mass" else 2.1)}
            for arm in study.ARMS for order in study.ORDERS]


def test_protocol_has_only_six_fixed_order_cells_and_declared_counts(study):
    assert study.ARMS == ("relative_mass", "uniform_mass")
    assert study.ORDERS == ("consecutive", "shuffle_0", "shuffle_1")
    assert tuple(study.WINDOWS) == tuple(range(32))
    recipe = study.recipe()
    assert recipe["divisor_k"] == 18 and recipe["deadline_s"] == 3600
    assert recipe["declared_cells"] == 6
    assert recipe["full_gradient_queries"] == recipe["optimizer_commits"] == 192
    assert recipe["complete_saved_states"] == 198 and recipe["score_readouts"] == 302
    assert recipe["new_gradient_pair_exposures"] == 24576
    assert recipe["new_fitting_pair_exposures"] == recipe["new_native_training_calls"] == 0
    assert "no EF remainder" in recipe["state"]
    assert "not equal parameter-gradient norm" in recipe["scale_scope"]
    for key in ("heldout_corpus_evaluated", "physical_fpga_execution", "paid_execution_authorized",
                "recipe_promoted", "adaptive_extension", "statistical_significance_claim"):
        assert recipe[key] is False


def test_artifact_set_covers_all_states_queries_and_readouts(study):
    files = study.expected_artifacts()
    assert len([name for name in files if "/query_" in name]) == 192
    assert len([name for name in files if "/state_" in name and name.endswith(".npz")]) == 198
    assert len([name for name in files if "/state_" in name and name.endswith(".json")]) == 198
    assert len([name for name in files if "/transition_" in name]) == 192
    assert len([name for name in files if "score_" in name]) == 302
    assert len(files) == 1092


def test_passing_numerical_gate_never_authorizes_promotion_or_skips_replay(study):
    result = study.gates(gate_cells(study), metric(3.0, 3.1))
    assert result["numerical_training_gate_pass"] is True
    assert result["requires_separate_successful_independent_recompute"] is True
    assert result["broader_study_authorized"] is False
    assert result["recipe_promoted"] is False and result["paid_execution_authorized"] is False


@pytest.mark.parametrize("loss", ["true_ce", "lut_nll"])
@pytest.mark.parametrize("case", ["threshold", "order_regression", "initial_equal"])
def test_each_loss_requires_gain_no_order_regression_and_strict_initial_improvement(study, loss, case):
    rows = gate_cells(study)
    uniform = [row for row in rows if row["arm"] == "uniform_mass"]
    relative = [row for row in rows if row["arm"] == "relative_mass"]
    if case == "threshold":
        for left, right in zip(relative, uniform):
            left["final_mean"][loss] = right["final_mean"][loss] + .009
    elif case == "order_regression":
        uniform[0]["final_mean"][loss] = relative[0]["final_mean"][loss] + .001
    else:
        uniform[0]["final_mean"][loss] = metric(3.0, 3.1)[loss]
    assert study.gates(rows, metric(3.0, 3.1))["numerical_training_gate_pass"] is False


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed_mechanics", "bool_alias"])
def test_gates_reject_incomplete_or_unhealthy_membership(study, change):
    rows = gate_cells(study)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[-1]["mechanics_pass"] = False if change == "failed_mechanics" else 1
    with pytest.raises(ValueError):
        study.gates(rows, metric(3.0, 3.1))


@pytest.fixture
def fake_run(study, ancestry, monkeypatch, tmp_path):
    """New real lifecycle/cache; body replaced, scientific paths blocked."""
    plan = tmp_path / "synthetic_plan.md"
    plan.write_text("Synthetic lifecycle declaration; not a corpus protocol\n")
    monkeypatch.setattr(study, "PLAN", plan)
    calls = SimpleNamespace(body=[], fail=False)
    payload = b"synthetic body; no numerical trajectory"

    def execute(manifest, output, raw, *, write=False, recompute=False):
        assert raw == b"synthetic"
        calls.body.append((write, recompute))
        if calls.fail:
            raise RuntimeError("controlled body failure")
        path = Path(output) / "synthetic.bin"
        if write:
            with path.open("xb") as stream:
                stream.write(payload)
        study.equal(path.read_bytes(), payload, "synthetic saved body")
        rows = gate_cells(study)
        return {"schema": study.SCHEMA, "complete": True, "recipe": study.recipe(),
            "source_hashes": manifest["source_hashes"], "cells": rows,
            "gates": study.gates(rows, metric(3.0, 3.1)), "completed_cells": 6,
            "full_gradient_queries": 192, "optimizer_commits": 192, "score_readouts": 302,
            "recipe_promoted": False, "scope": "synthetic lifecycle fixture, not evidence"}

    monkeypatch.setattr(study, "execute", execute)
    monkeypatch.setattr(study, "expected_artifacts", lambda: {
        "manifest.json", "protocol.json", "declaration.md", "summary.json", "synthetic.bin"})
    return SimpleNamespace(config=ancestry.config, ancestry=ancestry, calls=calls)


def test_lifecycle_reuses_semantics_only_inside_invocation_and_public_check_refreshes(study, fake_run):
    config, ancestry = fake_run.config, fake_run.ancestry
    result = study.run(config)
    assert result["complete"] is True and result["completed_cells"] == 6
    assert len(ancestry.calls.semantic) == 1
    before, receipt = tree_bytes(config.output), config.record.read_bytes()
    assert study.check(config.output, config.record, recompute=True) == result
    assert len(ancestry.calls.semantic) == 2
    assert tree_bytes(config.output) == before and config.record.read_bytes() == receipt
    assert tree_bytes(ancestry.root) == ancestry.before
    assert fake_run.calls.body == [(True, False), (False, False), (False, True)]


@pytest.mark.parametrize("failure", ["binding", "body", "after_link", "foreign_target", "final_check", "cleanup"])
def test_failure_retains_raw_and_retracts_only_owned_completion(study, fake_run, monkeypatch, failure):
    config = fake_run.config
    previous = study.signal.getsignal(study.signal.SIGALRM)

    def fail(*args, **kwargs):
        raise TimeoutError("controlled " + failure)

    if failure == "binding":
        monkeypatch.setattr(study, "bind_inputs", fail)
    elif failure == "body":
        fake_run.calls.fail = True
    elif failure == "final_check":
        monkeypatch.setattr(study, "_check", fail)
    elif failure in ("after_link", "foreign_target"):
        original = study.os.link

        def fail_link(source, target):
            if failure == "after_link":
                original(source, target)
                raise KeyboardInterrupt("controlled post-link interruption")
            Path(target).write_bytes(b"unrelated user receipt")
            return original(source, target)

        monkeypatch.setattr(study.os, "link", fail_link)
    else:
        original, failed = study.signal.setitimer, []

        def fail_cleanup(which, seconds, *args):
            if seconds == 0 and not failed:
                failed.append(True)
                fail()
            return original(which, seconds, *args)

        monkeypatch.setattr(study.signal, "setitimer", fail_cleanup)
    with pytest.raises((TimeoutError, RuntimeError, OSError, KeyboardInterrupt)):
        study.run(config)
    assert study.read(config.output / "failure.json")["complete"] is False
    assert study.signal.getsignal(study.signal.SIGALRM) == previous
    assert study.signal.getitimer(study.signal.ITIMER_REAL)[0] == 0
    if failure == "foreign_target":
        assert config.record.read_bytes() == b"unrelated user receipt"
    else:
        assert not config.record.exists()
    assert not [path for path in config.record.parent.iterdir() if path.name.startswith(".tt-")]


@pytest.mark.parametrize("target", ["public", "seal", "body"])
def test_final_replay_rejects_any_changed_receipt_or_payload(study, fake_run, target):
    config = fake_run.config
    study.run(config)
    path = {"public": config.record, "seal": config.output / "artifact_manifest.json",
            "body": config.output / "synthetic.bin"}[target]
    if target == "public":
        row = study.read(path)
        row["gates"]["broader_study_authorized"] = True
        write_json(path, row)
    else:
        path.write_bytes(b"corrupted")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        study.check(config.output, config.record)


@pytest.mark.parametrize("existing", ["output", "record"])
def test_existing_output_bytes_are_preserved_without_execution(study, fake_run, existing):
    path = getattr(fake_run.config, existing)
    path.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        study.run(fake_run.config)
    assert path.read_bytes() == b"existing user data"
    assert not fake_run.calls.body
