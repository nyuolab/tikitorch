"""Synthetic two-candidate channel controls; no real corpus or cloud runs."""
from __future__ import annotations

import copy
import importlib
import json
import random
import signal
from types import SimpleNamespace

import pytest
import torch


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_channel_choice")


def interleaved_fixture(width=16, vocab=4, layers=4, taps=3):
    """The production bank construction, scaled only for synthetic geometry."""
    anchors = [(5 * value) % width for value in range(vocab)]
    unused = sorted(set(range(width)) - set(anchors))
    banks = dict(zip(("A", "B", "C", "D"),
                     (anchors, unused[:vocab], unused[vocab:2 * vocab], unused[2 * vocab:])))
    sigma = list(range(vocab))
    random.Random(0).shuffle(sigma)
    inverse = [sigma.index(value) for value in range(vocab)]
    primary, alternate = [], []
    for layer in range(layers):
        p, a = [-1] * width, [-1] * width
        for source, target in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "B")):
            for index, row in enumerate(banks[source]):
                selected = inverse[(sigma[index] + (1 << layer)) % vocab]
                successor = inverse[(sigma[selected] + 1) % vocab]
                p[row], a[row] = banks[target][selected], banks[target][successor]
        primary.append(p)
        alternate.append(a)
    return SimpleNamespace(width=width, vocab=vocab, layers=layers, taps=taps,
                           banks=banks, sigma=sigma, inverse=inverse,
                           primary=primary, alternate=alternate)


def initial_weights(study, spec):
    values = {"EMBED": torch.zeros(spec.vocab, spec.width, dtype=torch.float64),
              "MH": torch.zeros(spec.layers, spec.width, spec.taps, dtype=torch.float64),
              "MP": torch.full((spec.layers, spec.width, spec.width),
                               -8192. * study.helper.BETA, dtype=torch.float64),
              "MB": torch.zeros(spec.layers, spec.width, dtype=torch.float64),
              "W_HEAD": torch.full((spec.vocab, spec.width),
                                   -8192. * study.helper.BETA, dtype=torch.float64)}
    for token in range(spec.vocab):
        values["EMBED"][token, spec.banks["A"]] = torch.tensor(
            [-4 * ((token + target) % spec.vocab) * study.helper.BETA
             for target in range(spec.vocab)], dtype=torch.float64)
    values["W_HEAD"][torch.arange(spec.vocab), spec.banks["A"]] = 0
    for layer, selected in enumerate(spec.primary):
        values["MP"][layer, torch.arange(spec.width), selected] = 0
    return values


def test_source_closure_adds_only_channel_runner_and_owned_test(study):
    previous, current = study.core.source_identity(), study.source_identity()
    assert len(previous) == 75 and len(current) == 77
    assert {key: current[key] for key in previous} == previous
    assert set(current) - set(previous) == {
        "manuscript/tools/diagnose_fpga_lm_channel_choice.py",
        "tests/test_fpga_lm_channel_choice.py"}


def test_exact_fixed_protocol_budget_cohorts_and_inventory(study):
    recipe = study.recipe()
    assert recipe["arm"] == "mixed_channel_choice" and recipe["order_seed"] == 0
    order = list(range(1024))
    random.Random(0).shuffle(order)
    assert recipe["training_order"] == order
    assert recipe["updates"] == recipe["raw_gradient_queries"] == recipe["applied_gradient_archives"] == 256
    assert recipe["optimizer_commits"] == 256 and recipe["complete_saved_states"] == 257
    assert recipe["batch_size"] == 4 and recipe["sequence"] == 128 and recipe["learning_rate"] == .01
    assert recipe["alternate_weight"] == -.01 and recipe["frozen_regions"] == ["MP", "MB"]
    assert recipe["trainable_regions"] == ["EMBED", "MH", "W_HEAD"]
    assert recipe["sampled_training_windows"] == list(range(0, 1024, 16))
    assert recipe["development_windows"] == list(range(88, 152))
    assert recipe["score_window_readouts"] == 256 and recipe["new_initial_score_evaluations"] == 128
    assert recipe["new_gradient_pair_exposures"] == recipe["inherited_fitting_pair_exposures"] == 131072
    assert recipe["inherited_gradient_pair_exposures"] == 0 and recipe["endpoint_history_pair_exposures"] == 262144
    assert recipe["new_baseline_evaluations"] == 0 and "none" in recipe["quality_gate"]
    for key in ("new_model_count_fit", "new_gru_training", "adaptive_selection", "recipe_promoted",
                "heldout_test_evaluated", "reserved_development_evaluated", "physical_fpga_execution",
                "paid_execution_authorized"):
        assert recipe[key] is False
    assert study.DEADLINE_S == 10800 and study.CELL_DEADLINE_S == 3600
    assert study.STORAGE_CAP == 8 * 2**30 and study.FREE_RESERVE == 4 * 2**30
    names = study.expected_artifacts()
    assert len(names) == 1288 and {"transform.json", "initial_parity.json"} <= names
    assert sum("/query_" in name for name in names) == sum("/applied_" in name for name in names) == 256
    assert sum("/state_" in name and name.endswith(".npz") for name in names) == 257
    assert sum("/score_" in name for name in names) == 256


def test_actual_d512_mapping_uses_interleaved_bank_seeded_successors(study):
    spec = interleaved_fixture(512, 128, 4, 7)
    value = study.mapping(study.pilot.layout(512))
    assert value["primary_by_layer"] == spec.primary
    assert value["alternate_by_layer"] == spec.alternate
    assert value["sigma"] == spec.sigma
    assert value["inverse_sigma"] == spec.inverse
    for bank in ("A", "B", "C", "D"):
        assert value[bank] == spec.banks[bank]
    collisions = []
    for layer in range(4):
        for row, (primary, alternate) in enumerate(zip(spec.primary[layer], spec.alternate[layer])):
            bank = next(name for name in ("B", "C", "D") if primary in spec.banks[name])
            assert alternate in spec.banks[bank] and alternate != primary
            # A contiguous-block shortcut can overwrite the selected primary.
            actual_class = spec.banks[bank].index(primary)
            physical = 128 * (primary // 128) + spec.inverse[(spec.sigma[actual_class] + 1) % 128]
            if physical == primary:
                collisions.append((layer, row))
    assert collisions, "the fixture must expose a primary-overwriting physical-block shortcut"
    assert len({coordinate for bank in spec.banks.values() for coordinate in bank}) == 512


def tiny_mapping(study, spec=None):
    spec = interleaved_fixture() if spec is None else spec
    value = copy.deepcopy(study.mapping())
    value.update(spec.banks)
    value.update(sigma=spec.sigma, inverse_sigma=spec.inverse,
                 primary_by_layer=spec.primary, alternate_by_layer=spec.alternate,
                 geometry={"vocab": spec.vocab, "width": spec.width,
                           "layers": spec.layers, "taps": spec.taps})
    return value


@pytest.fixture
def tiny_transform(study, monkeypatch):
    spec = interleaved_fixture()
    mapping = tiny_mapping(study, spec)
    monkeypatch.setattr(study, "mapping", lambda shape=None: copy.deepcopy(mapping))
    return SimpleNamespace(spec=spec, mapping=mapping, weights=initial_weights(study, spec))


def test_real_transform_initial_full_trace_raw_g_parity_and_no_alias(study, tiny_transform):
    value = tiny_transform
    before = {name: tensor.clone() for name, tensor in value.weights.items()}
    changed, metadata = study.transform(value.weights)
    assert isinstance(metadata, dict)
    for name, tensor in changed.items():
        assert tensor.data_ptr() != value.weights[name].data_ptr()
        if name != "MP":
            study.guard.byte_equal({name: tensor}, {name: before[name]}, "only alternate MP entries change")
    expected = before["MP"].clone()
    for layer in range(value.spec.layers):
        expected[layer, torch.arange(value.spec.width), value.spec.alternate[layer]] = -.01
    study.guard.byte_equal({"MP": changed["MP"]}, {"MP": expected}, "exact two-entry initialization")
    tokens, targets = torch.tensor([0, 1, 2, 3, 0, 2]), torch.tensor([1, 2, 3, 0, 2, 1])
    old = study.helper.forward(before, tokens, independent=True)
    new = study.helper.forward(changed, tokens, independent=True)
    study.guard.byte_equal(study.helper.trace(new), study.helper.trace(old), "all initial logits and routes")
    original_g = study.helper.gradient(before, tokens, targets, old, len(tokens))
    actual_g = study.helper.gradient(changed, tokens, targets, new, len(tokens))
    study.guard.byte_equal(actual_g, original_g, "actual initial raw gradients")
    oracle_g = study.helper.gradient(changed, tokens, targets, new, len(tokens), independent=True)
    for name in actual_g:
        torch.testing.assert_close(actual_g[name], oracle_g[name], atol=1e-9, rtol=1e-11)
    assert bool(actual_g["MP"].any()) and bool(actual_g["MB"].any())
    study.guard.byte_equal(value.weights, before, "transform does not mutate source")


def test_alternate_can_really_win_after_controlled_embedding_perturbation(study, tiny_transform):
    value = tiny_transform
    changed, _ = study.transform(value.weights)
    tokens = torch.tensor([0, 1, 2, 3])
    initial = study.helper.trace(study.helper.forward(changed, tokens, independent=True))
    assert initial["PROJ_IDX"][0, 0, 0].item() == value.spec.primary[0][0]
    changed["EMBED"][0, value.spec.alternate[0][0]] += .1
    perturbed = study.helper.trace(study.helper.forward(changed, tokens, independent=True))
    assert perturbed["PROJ_IDX"][0, 0, 0].item() == value.spec.alternate[0][0]


@pytest.mark.parametrize("case", ["unused_embed", "positive_anchor", "anchor_quantum", "mh", "mb",
                                  "primary", "head", "sentinel", "float32", "attached", "nan"])
def test_transform_rejects_noncanonical_initial_structure(study, tiny_transform, case):
    value = tiny_transform
    weights = {name: tensor.clone() for name, tensor in value.weights.items()}
    if case == "unused_embed":
        weights["EMBED"][0, value.spec.banks["B"][0]] = .01
    elif case in ("positive_anchor", "anchor_quantum"):
        weights["EMBED"][0, value.spec.banks["A"][0]] = .1 if case == "positive_anchor" else -.001
    elif case in ("mh", "mb"):
        weights[case.upper()].reshape(-1)[0] = .01
    elif case == "primary":
        weights["MP"][0, 0, value.spec.primary[0][0]] = -.01
    elif case == "head":
        weights["W_HEAD"][0, value.spec.banks["A"][0]] = -.01
    elif case == "sentinel":
        weights["MP"][0, 0, value.spec.alternate[0][0]] = -.01
    elif case == "float32":
        weights["EMBED"] = weights["EMBED"].float()
    elif case == "attached":
        weights["EMBED"].requires_grad_(True)
    else:
        weights["MP"].reshape(-1)[0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        study.transform(weights)


@pytest.mark.parametrize("case", ["collision", "wrong_bank", "out_of_range", "bool"])
def test_invalid_candidate_map_cannot_overwrite_primary(study, tiny_transform, case):
    value = tiny_transform
    bad = copy.deepcopy(value.mapping)
    bad["alternate_by_layer"][0][0] = {
        "collision": value.spec.primary[0][0], "wrong_bank": value.spec.banks["A"][0],
        "out_of_range": value.spec.width, "bool": True}[case]
    with pytest.raises((ValueError, TypeError)):
        study._transform(value.weights, bad)


def test_dual_choice_is_or_across_windows_at_one_checkpoint_not_across_updates(study, tiny_transform):
    value = tiny_transform
    changed, _ = study.transform(value.weights)
    trace = study.helper.trace(study.helper.forward(changed, torch.tensor([0, 1, 2, 3]), independent=True))
    second = {key: tensor.clone() for key, tensor in trace.items()}
    second["PROJ_IDX"][0, :, 0] = value.spec.alternate[0][0]
    first_counts = study.classify_routes([trace], value.mapping)
    second_counts = study.classify_routes([second], value.mapping)
    assert first_counts["total"]["both_choice_rows"] == second_counts["total"]["both_choice_rows"] == 0
    # Separately reported checkpoints have zero dual rows; only explicitly
    # grouped windows at one fixed checkpoint prove both choices were selected.
    combined = study.classify_routes([trace, second], value.mapping)
    assert combined["total"]["both_choice_rows"] == 1
    assert combined["per_layer"][0]["both_choice_rows"] == 1
    assert combined["total"]["alternate"] == 4
    assert combined["total"]["primary"] == 2 * 4 * 4 * 16 - 4
    assert combined["total"]["other"] == 0 and combined["total"]["total_rows"] == 64


def test_legitimate_other_sentinel_and_head_departures_are_counted_not_rejected(study, tiny_transform):
    value = tiny_transform
    changed, _ = study.transform(value.weights)
    trace = study.helper.trace(study.helper.forward(changed, torch.tensor([0, 1]), independent=True))
    other = next(i for i in range(value.spec.width)
                 if i not in (value.spec.primary[0][0], value.spec.alternate[0][0]))
    trace["PROJ_IDX"][0, 0, 0] = other
    trace["AD_HEAD"][0, 0] = (value.spec.banks["A"][0] + 1) % value.spec.width
    result = study.classify_routes([trace], value.mapping)
    assert result["total"]["other"] == 1 and result["total"]["alternate"] == 0
    assert result["head"] == {"departures_from_initial_anchor": 1, "total": 8}
    assert result["total"]["primary"] + result["total"]["alternate"] + result["total"]["other"] == result["total"]["total"]


@pytest.mark.parametrize("case", ["empty", "zero_positions", "missing", "float_route", "negative", "past_width",
                                  "nan_logit", "dilated_noncausal", "duplicate_map"])
def test_classifier_rejects_malformed_incomplete_and_noncausal_traces(study, tiny_transform, case):
    value = tiny_transform
    changed, _ = study.transform(value.weights)
    trace = study.helper.trace(study.helper.forward(changed, torch.tensor([0, 1]), independent=True))
    mapping = copy.deepcopy(value.mapping)
    traces = [trace]
    if case == "empty":
        traces = []
    elif case == "zero_positions":
        for name in trace:
            trace[name] = trace[name][:0] if name in ("logits", "AD_HEAD") else trace[name][:, :0]
    elif case == "missing":
        del trace["TAP_IDX"]
    elif case == "float_route":
        trace["PROJ_IDX"] = trace["PROJ_IDX"].double()
    elif case in ("negative", "past_width"):
        trace["PROJ_IDX"][0, 0, 0] = -1 if case == "negative" else value.spec.width
    elif case == "nan_logit":
        trace["logits"][0, 0] = float("nan")
    elif case == "dilated_noncausal":
        trace["TAP_IDX"][1, 1, 0] = 1
    else:
        mapping["alternate_by_layer"][0][0] = mapping["primary_by_layer"][0][0]
    with pytest.raises((ValueError, TypeError)):
        study.classify_routes(traces, mapping)


@pytest.mark.parametrize("pin", [None, "", "a" * 63, "G" * 64, 1])
def test_missing_successful_core_audit_fails_before_any_input_access(study, monkeypatch, tmp_path, pin):
    monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", pin)
    monkeypatch.setattr(study.parent.rate.head, "_regular",
                        lambda *a: pytest.fail("unadmitted predecessor input accessed"))
    with pytest.raises(ValueError, match="not yet pinned"):
        study._audit_capsule(tmp_path / "missing", tmp_path / "root", tmp_path / "public.json")


def test_mask_and_full_adam_preserve_both_projection_candidates_and_core_bias(study):
    spec = interleaved_fixture()
    weights = initial_weights(study, spec)
    for layer in range(spec.layers):
        weights["MP"][layer, torch.arange(spec.width), spec.alternate[layer]] = -.01
    model = {"kind": "tropical", "parameters": weights, "model": None, "prior_logits": None}
    initial = {name: weights[name].clone() for name in ("MP", "MB")}
    optimizer = study.parent.make_optimizer("mixed", weights.values())
    raw = {name: (torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape) % 9 - 4) / 32
           for name, value in weights.items()}
    assert bool(raw["MP"].any()) and bool(raw["MB"].any())
    before = {name: value.clone() for name, value in raw.items()}
    for step in range(1, 257):
        applied = study.applied_gradient(raw)
        for name in weights:
            assert applied[name].data_ptr() != raw[name].data_ptr()
            if name in initial:
                assert not applied[name].any() and not torch.signbit(applied[name]).any()
            else:
                study.guard.byte_equal({name: applied[name]}, {name: raw[name]}, "unmasked exact raw G")
            weights[name].grad = applied[name]
        optimizer.step()
        receipt = study.frozen_state(model, optimizer, initial, step)
        assert receipt["moments_exact_zero"] is True and receipt["all_counters"] == step
    study.guard.byte_equal(raw, before, "nonmutating gradient mask")
    study.guard.byte_equal({name: weights[name] for name in initial}, initial, "both candidates and every sentinel frozen")


@pytest.mark.parametrize("case", ["carried_first", "carried_second", "primary", "alternate", "bias"])
def test_frozen_core_rejects_any_carry_or_candidate_drift(study, case):
    spec = interleaved_fixture()
    weights = initial_weights(study, spec)
    for layer in range(spec.layers):
        weights["MP"][layer, torch.arange(spec.width), spec.alternate[layer]] = -.01
    model = {"kind": "tropical", "parameters": weights, "model": None, "prior_logits": None}
    initial = {name: weights[name].clone() for name in ("MP", "MB")}
    optimizer = study.parent.make_optimizer("mixed", weights.values())
    for value in weights.values():
        value.grad = torch.zeros_like(value)
    optimizer.step()
    if case.startswith("carried"):
        moment = "exp_avg" if case == "carried_first" else "exp_avg_sq"
        optimizer.state[weights["MP"]][moment].reshape(-1)[0] = .1
    elif case == "bias":
        weights["MB"][0, 0] = .1
    else:
        choice = spec.primary if case == "primary" else spec.alternate
        weights["MP"][0, 0, choice[0][0]] += .1
    with pytest.raises(ValueError):
        study.frozen_state(model, optimizer, initial, 1)


@pytest.fixture
def audit_capsule(study, tmp_path, monkeypatch):
    root, capsule = tmp_path / "core", tmp_path / "audit"
    root.mkdir()
    capsule.mkdir()
    public, receipt = tmp_path / "producer.json", tmp_path / "receipt.json"
    write_json(root / "artifact_manifest.json", {"synthetic": True})
    write_json(public, {"complete": True})
    sources = {"synthetic_source": "a" * 64}
    monkeypatch.setattr(study.core, "source_identity", lambda: sources)
    declaration = capsule / "declaration.md"
    declaration.write_text("Synthetic complete core audit capsule.\n")
    value = {"schema": study.core.SCHEMA + ".audit", "complete": True,
             "independent_recompute_completed": True, "run_directory": str(root), "record": str(public),
             "source_hashes": sources, "producer_public_sha256": study.old.digest(public.read_bytes()),
             "producer_seal_sha256": study.old.digest((root / "artifact_manifest.json").read_bytes()),
             "deadline_seconds": 10800, "cell_deadline_seconds": 3600,
             "independent_tropical_window_vjps": 1024, "independent_tropical_window_forwards": 1152,
             "independent_tropical_batch_gradients": 256, "supplemental_guarded_window_forwards": 1152,
             "full_batch_gradients_verified": 256, "applied_gradient_masks_verified": 256,
             "frozen_regions_verified": ["MP", "MB"], "complete_saved_states_verified": 257,
             "optimizer_commits_replayed": 256, "score_window_readouts": 128,
             "source_input_raw_seal_public_bytes_unchanged": True, "new_training_updates": 0,
             "new_gradient_pair_exposures": 0, "new_model_count_fit": False, "new_baseline_evaluations": 0,
             "old_numerical_trajectories_replayed": False, "heldout_test_evaluated": False,
             "reserved_development_evaluated": False, "physical_fpga_execution": False,
             "paid_execution_authorized": False, "recipe_promoted": False,
             "verification_elapsed_utc_seconds": 100., "verification_elapsed_monotonic_seconds": 90.,
             "audit_directory": str(capsule), "runtime": study.parent.runtime(),
             "started_at_utc": "2026-09-14 00:00:00 UTC"}
    manifest = {key: value[key] for key in ("run_directory", "record", "source_hashes", "producer_public_sha256",
                "producer_seal_sha256", "runtime", "started_at_utc", "deadline_seconds", "cell_deadline_seconds")}
    manifest.update(schema=study.core.SCHEMA, complete=False, mode="independent_audit",
                    audit_directory=str(capsule), audit_record=str(receipt),
                    declaration_sha256=study.old.digest(declaration.read_bytes()))
    write_json(capsule / "audit_manifest.json", manifest)
    value["audit_manifest_sha256"] = study.old.digest((capsule / "audit_manifest.json").read_bytes())

    def reseal():
        write_json(capsule / "verification.json", value)
        write_json(receipt, value)
        monkeypatch.setattr(study, "PRODUCER_SHA256", study.old.digest(public.read_bytes()))
        monkeypatch.setattr(study, "PRODUCER_SEAL_SHA256", value["producer_seal_sha256"])
        monkeypatch.setattr(study, "AUDIT_MANIFEST_SHA256", value["audit_manifest_sha256"])
        monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", study.old.digest(receipt.read_bytes()))

    reseal()
    return SimpleNamespace(root=root, capsule=capsule, public=public, receipt=receipt, value=value, reseal=reseal)


def test_exact_successful_core_capsule_admitted_without_old_numerical_replay(study, audit_capsule, monkeypatch):
    for name in ("run", "check", "execute", "audit", "initial_model"):
        monkeypatch.setattr(study.core, name, lambda *a, **k: pytest.fail("predecessor numerical call forbidden"))
    value = audit_capsule
    actual, proof = study._audit_capsule(value.receipt, value.root, value.public)
    assert actual == value.value
    assert set(proof["files"]) == {"audit_manifest.json", "declaration.md", "verification.json"}


@pytest.mark.parametrize("case", ["private_whitespace", "extra_failure", "public_pin", "seal_pin", "manifest_pin",
                                  "incomplete", "false_mask_count", "float_counter", "wrong_frozen_region", "expired"])
def test_pinned_proof_rejects_altered_and_resealed_incomplete_capsules(study, audit_capsule, case):
    value = audit_capsule
    if case == "private_whitespace":
        path = value.capsule / "verification.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif case == "extra_failure":
        write_json(value.capsule / "failure.json", {"complete": False})
    elif case in ("public_pin", "seal_pin", "manifest_pin"):
        path = {"public_pin": value.public, "seal_pin": value.root / "artifact_manifest.json",
                "manifest_pin": value.capsule / "audit_manifest.json"}[case]
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        key, bad = {"incomplete": ("complete", 1), "false_mask_count": ("applied_gradient_masks_verified", 255),
                    "float_counter": ("complete_saved_states_verified", 257.),
                    "wrong_frozen_region": ("frozen_regions_verified", ["MP"]),
                    "expired": ("verification_elapsed_utc_seconds", 10800.)}[case]
        value.value[key] = bad
        value.reseal()
    with pytest.raises(ValueError):
        study._audit_capsule(value.receipt, value.root, value.public)


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    """Reuse frozen test preparation, replacing only its synthetic geometry.

    This creates real tiny initial score/query/state archives, not real corpus
    evidence. No historical source or test file is edited.
    """
    from test_fpga_lm_core_bias import synthetic as old_fixture
    import test_fpga_lm_core_bias as old_tests

    patch = pytest.MonkeyPatch()
    original_factory = old_tests.tiny_models
    spec = interleaved_fixture()

    def models(helper):
        result = original_factory(helper)
        result["mixed"]["parameters"] = initial_weights(SimpleNamespace(helper=helper), spec)
        return result

    patch.setattr(old_tests, "tiny_models", models)
    generator = old_fixture.__wrapped__(tmp_path_factory)
    try:
        value = next(generator)
        value.spec = spec
        yield value
    finally:
        generator.close()
        patch.undo()


@pytest.fixture(scope="module")
def synthetic(historical):
    """Genuine tiny new trajectory, backed by saved synthetic old controls."""
    value = historical
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_channel_choice")
    patch = pytest.MonkeyPatch()
    temporary = value.temporary / "channel"
    temporary.mkdir()
    source, declaration = temporary / "source.txt", temporary / "PLAN.md"
    source.write_text("Synthetic channel-choice invocation identity.\n")
    declaration.write_text("Synthetic channel-choice recipe, not scientific evidence.\n")
    mapping = tiny_mapping(study, value.spec)
    patch.setattr(study, "mapping", lambda shape=None: copy.deepcopy(mapping))
    for key, item in {"TRAIN_WINDOWS": tuple(range(8)), "SCORE_TRAIN_WINDOWS": (0, 4),
                      "DEV_WINDOWS": (88, 89), "UPDATES": 2, "SEQUENCE": 4, "PLAN": declaration}.items():
        patch.setattr(study, key, item)
    patch.setattr(study, "batch_data", value.study.batch_data)
    binding = {"prior_run": str(value.root), "prior_record": str(value.record),
               "prior_audit": str(value.corpus), "core_binding": copy.deepcopy(value.binding),
               "dataset": copy.deepcopy(value.binding["dataset"]), "audit_receipt_sha256": "a" * 64}
    binding.update({key: copy.deepcopy(value.binding[key]) for key in
                    ("continuation_binding", "endpoint_binding", "development_binding")})
    binding["initial_scores"] = {
        "train": [study.old.file_identity(value.endpoint_root / "initial_mixed" / f"score_{i:04d}.npz")
                  for i in study.SCORE_TRAIN_WINDOWS],
        "development": [study.old.file_identity(value.dev_root / "shared_mixed" / f"score_initial_{i:03d}_1.npz")
                        for i in study.DEV_WINDOWS]}
    binding["initial_query"] = study.old.file_identity(value.root / study.core.ARM / "query_000000.npz")
    protected = list(value.protected)
    protected.extend(study.old.file_identity(path) for path in value.root.rglob("*") if path.is_file())
    protected.extend(study.old.file_identity(path) for path in (source, value.corpus, value.record))
    source_identity = lambda: {"synthetic_source": study.old.digest(source.read_bytes())}
    patch.setattr(study, "source_identity", source_identity)
    validation = study._Validation(binding, source_identity(), {}, object())
    binds = []

    def bind(config):
        binds.append(True)
        return copy.deepcopy(binding), value.corpus.read_bytes(), validation

    def frozen(manifest, token):
        assert isinstance(token, study._Validation)
        study.equal(manifest["schema"], study.SCHEMA, "synthetic schema")
        study.equal(manifest["recipe"], study.recipe(), "synthetic recipe")
        study.equal(manifest["binding"], token.binding, "private invocation binding")
        study.equal(token.binding, binding, "original invocation binding")
        study.equal(manifest["source_hashes"], token.sources, "invocation sources")
        study.equal(token.sources, source_identity(), "current source bytes")
        for descriptor in protected:
            study.equal(study.old.file_identity(descriptor["path"]), descriptor,
                        "immutable initial/query/control/source input")

    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    for name in ("run", "check", "execute", "audit"):
        patch.setattr(study.core, name, lambda *a, **k: pytest.fail("old core trajectory execution forbidden"))
    config = SimpleNamespace(prior_run=value.root, prior_record=value.record, prior_audit=value.corpus,
                             corpus=value.corpus, output=temporary / "run", record=temporary / "public.json")
    before = dict(value.calls)
    value.reads.clear()
    try:
        result = study.run(config)
        yield SimpleNamespace(study=study, config=config, root=config.output, record=config.record,
                              result=result, temporary=temporary, binding=binding, source=source,
                              corpus=value.corpus, historical=value, calls=value.calls,
                              producer_calls={key: value.calls[key] - before[key] for key in before},
                              protected=protected, binds=binds, reads=value.reads)
    finally:
        patch.undo()


def test_genuine_tiny_producer_preserves_first_batch_then_masks_all_steps(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    assert result["complete"] is True and result["arm"] == study.ARM
    assert result["raw_gradient_queries"] == result["applied_gradient_archives"] == result["optimizer_commits"] == 2
    assert result["complete_saved_states"] == 3 and result["score_window_readouts"] == 8
    assert value.producer_calls == {"guarded": 16, "dense": 0, "manual_g": 8, "independent_g": 0}
    assert result["training_order"] == study.training_order()
    assert result["new_initial_score_evaluations"] == 4
    assert result["new_baseline_evaluations"] == 0 and result["new_gru_training"] is False
    directory = value.root / study.ARM
    previous = study.load(directory / "state_000000.npz")
    expected, _ = study.transform(study.core.initial_model(value.binding["core_binding"])["parameters"])
    assert set(previous) == {"parameter/" + key for key in expected}
    study.guard.byte_equal({key: previous["parameter/" + key] for key in expected}, expected, "actual transformed empty-Adam start")
    fixed = {key: previous["parameter/" + key].clone() for key in ("MP", "MB")}
    has_nonzero_raw = {"MP": False, "MB": False}
    for step in range(2):
        query = study.load(directory / f"query_{step:06d}.npz")
        raw = {key[2:]: tensor for key, tensor in query.items() if key.startswith("g_")}
        applied = study.load(directory / f"applied_{step:06d}.npz")
        study.guard.byte_equal(applied, study.applied_gradient(raw), "raw-to-applied masks")
        after = study.load(directory / f"state_{step + 1:06d}.npz")
        for key, gradient in applied.items():
            p = previous["parameter/" + key]
            m = .9 * previous.get("exp_avg/" + key, torch.zeros_like(p)) + .1 * gradient
            v = .999 * previous.get("exp_avg_sq/" + key, torch.zeros_like(p)) + .001 * gradient.square()
            next_weight = p - .01 * m / (1 - .9 ** (step + 1)) / (v.sqrt() / (1 - .999 ** (step + 1)) ** .5 + 1e-8)
            torch.testing.assert_close(after["parameter/" + key], next_weight, atol=1e-12, rtol=1e-11)
            assert float(after["step/" + key]) == step + 1
            if key in fixed:
                has_nonzero_raw[key] |= bool(raw[key].any())
                study.guard.byte_equal({key: after["parameter/" + key]}, {key: fixed[key]}, "entire core remains frozen")
                assert not after["exp_avg/" + key].any() and not after["exp_avg_sq/" + key].any()
        previous = after
    assert all(has_nonzero_raw.values())
    assert set(value.reads) <= {("train", i) for i in range(8)} | {("validation", i) for i in (88, 89)}


def test_readback_is_model_free_and_independent_audit_rederives_every_new_query(synthetic, tmp_path):
    value, study = synthetic, synthetic.study
    before, original = dict(value.calls), tree_bytes(value.root)
    assert study.check(value.root, value.record) == value.result
    assert value.calls == before
    config = SimpleNamespace(**vars(value.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    result = study.audit(config)
    assert result["complete"] is True and result["independent_recompute_completed"] is True
    assert {key: value.calls[key] - before[key] for key in before} == {
        "guarded": 16, "dense": 16, "manual_g": 0, "independent_g": 8}
    assert result["full_batch_gradients_verified"] == result["applied_gradient_masks_verified"] == 2
    assert result["complete_saved_states_verified"] == 3 and result["score_window_readouts"] == 8
    assert (config.audit_output / "verification.json").read_bytes() == config.audit_record.read_bytes()
    assert tree_bytes(value.root) == original


def test_saved_route_counts_and_dense_guards_match_each_fixed_checkpoint(synthetic):
    value, study = synthetic, synthetic.study
    result = value.result
    mapping = result["transform"]["mapping"]
    directory = value.root / study.ARM
    totals = {key: 0 for key in ("primary", "alternate", "other", "total")}
    both = 0
    for step, row in enumerate(result["query_rows"]):
        values = study.load(directory / f"query_{step:06d}.npz")
        indices = study.training_order()[4 * step:4 * (step + 1)]
        traces = [{key: values[f"window_{index}/{key}"] for key in study.helper.TRACE_KEYS} for index in indices]
        counted = study.classify_routes(traces, mapping)
        assert row["channel_choices"] == counted
        for key in totals:
            totals[key] += counted["total"][key]
        both += counted["total"]["both_choice_rows"]
        assert values["guard_counts"].tolist() == [[0, 1, 0, 16, 0, 4]] * 4
    assert {key: result["query_channel_choice_totals"][key] for key in totals} == totals
    assert result["query_channel_choice_totals"]["both_choice_row_checkpoints"] == both
    assert result["query_channel_choice_totals"]["total_row_checkpoints"] == 2 * 4 * 16
    for cohort, indices in (("train", study.SCORE_TRAIN_WINDOWS), ("development", study.DEV_WINDOWS)):
        traces = []
        for index in indices:
            path = study._score_path(directory, cohort, index, "final")
            values = study.load(path)
            assert values["guard_counts"].tolist() == [[0, 1, 0, 16, 0, 4]]
            traces.append({key: values[key] for key in study.helper.TRACE_KEYS})
        assert result["final_channel_choices"][cohort] == study.classify_routes(traces, mapping)
        initial = result["initial_parity"]["support"][cohort]
        assert initial["total"]["alternate"] == initial["total"]["other"] == 0
    original = study.load(value.binding["initial_query"]["path"])
    candidate = study.load(directory / "query_000000.npz")
    assert not torch.equal(original["guard_counts"], candidate["guard_counts"])
    study.guard.byte_equal({key: tensor for key, tensor in original.items() if key != "guard_counts"},
                           {key: tensor for key, tensor in candidate.items() if key != "guard_counts"},
                           "initial exact raw traces/G deliberately exclude historical fast-path counters")
    assert result["initial_parity"]["first_batch"]["optimizer_commits_before_check"] == 0
    assert result["complete"] is True and "activation is descriptive" in result["scope"]


@pytest.mark.parametrize("target", ["raw_g", "applied_g", "moment", "route", "guard"])
def test_independent_replay_rejects_saved_gradient_state_route_or_guard_corruption(synthetic, target):
    value, study = synthetic, synthetic.study
    directory = value.root / study.ARM
    path = directory / {"raw_g": "query_000001.npz", "applied_g": "applied_000001.npz",
                        "moment": "state_000001.npz", "route": "query_000001.npz", "guard": "query_000001.npz"}[target]
    original = path.read_bytes()
    values = study.load(path)
    if target == "raw_g":
        values["g_MP"].reshape(-1)[0] += .1
    elif target == "applied_g":
        values["EMBED"].zero_()
    elif target == "moment":
        values["exp_avg/MP"].reshape(-1)[0] = .1
    elif target == "guard":
        values["guard_counts"][0, 0] = 1
    else:
        key = next(key for key in values if key.endswith("/PROJ_IDX"))
        values[key].reshape(-1)[0] = (values[key].reshape(-1)[0] + 1) % 16
    path.unlink()
    study.parent.parent.save(path, values)
    try:
        with pytest.raises(ValueError):
            study.execute(study.read(value.root / "manifest.json"), value.root, value.corpus.read_bytes(), recompute=True)
    finally:
        path.write_bytes(original)


def test_resealed_forged_channel_counter_is_recomputed_not_trusted(synthetic):
    value, study = synthetic, synthetic.study
    summary_path, seal_path = value.root / "summary.json", value.root / "artifact_manifest.json"
    original_summary, original_seal = summary_path.read_bytes(), seal_path.read_bytes()
    changed = study.read(summary_path)
    changed["query_channel_choice_totals"]["alternate"] += 1
    write_json(summary_path, changed)
    write_json(seal_path, {"schema": study.SCHEMA, "files": study.prior.artifact_manifest(value.root)})
    try:
        with pytest.raises(ValueError, match="summary"):
            study.check(value.root, value.record)
    finally:
        summary_path.write_bytes(original_summary)
        seal_path.write_bytes(original_seal)


@pytest.mark.parametrize("target", ["source", "corpus", "original_query", "original_initial_score", "initial_checkpoint"])
def test_source_and_original_control_mutations_fail_before_model_calls(synthetic, target):
    value, study = synthetic, synthetic.study
    path = {"source": value.source, "corpus": value.corpus,
            "original_query": study.Path(value.binding["initial_query"]["path"]),
            "original_initial_score": study.Path(value.binding["initial_scores"]["train"][0]["path"]),
            "initial_checkpoint": value.historical.dev_root / "mixed_seed_0/state_000000.npz"}[target]
    original, calls = path.read_bytes(), dict(value.calls)
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            study.check(value.root, value.record)
    finally:
        path.write_bytes(original)
    assert value.calls == calls


@pytest.mark.parametrize("target", ["raw_gradient", "trace"])
def test_first_batch_reference_mismatch_aborts_before_first_adam_commit(synthetic, tmp_path, monkeypatch, target):
    value, study = synthetic, synthetic.study
    config = SimpleNamespace(**{**vars(value.config), "output": tmp_path / "run", "record": tmp_path / "public.json"})
    original_load, original_step = study.load, torch.optim.Adam.step
    hits, commits = [], []
    source = study.Path(value.binding["initial_query"]["path"])

    def load(path):
        values = original_load(path)
        if study.Path(path) == source:
            hits.append(True)
            key = "g_MP" if target == "raw_gradient" else next(key for key in values if key.endswith("/logits"))
            values[key] = values[key].clone()
            values[key].reshape(-1)[0] += .1
        return values

    def step(*args, **kwargs):
        commits.append(True)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(study, "load", load)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    with pytest.raises(ValueError):
        study.run(config)
    assert hits and not commits
    assert study.read(config.output / "failure.json")["complete"] is False and not config.record.exists()


def mode_config(value, temporary, mode):
    if mode == "producer":
        config = SimpleNamespace(**{**vars(value.config), "output": temporary / "run", "record": temporary / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(value.config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failure_or_interruption_retains_attempt_without_success(synthetic, tmp_path, monkeypatch, mode, error):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0 and torch.get_num_threads() == 1
        raise error("synthetic channel-choice interruption")

    monkeypatch.setattr(study, "execute" if mode == "producer" else "_check", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(error):
        invoke(config)
    assert study.read(output / "failure.json")["complete"] is False and not target.exists()
    assert signal.getsignal(signal.SIGALRM) == handler and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["owned_corrupted", "foreign"])
def test_failed_publication_retracts_only_owned_inode(synthetic, tmp_path, monkeypatch, mode, replacement):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.result)
    original, payloads = study.frozen, []

    def frozen(*args):
        original(*args)
        if target.exists():
            payload = target.read_bytes()
            if replacement == "foreign":
                target.unlink()
                target.write_bytes(payload)
            else:
                payload += b"\n"
                target.write_bytes(payload)
            payloads.append(payload)
            raise ValueError("synthetic publication postcheck failure")

    monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError, match="postcheck"):
        invoke(config)
    failure = study.read(output / "failure.json")
    assert failure["complete"] is False and target.exists() is (replacement == "foreign")
    if replacement == "owned_corrupted":
        rejected = output / "rejected_public.json"
        assert rejected.read_bytes() == payloads[0]
        assert failure["retained_rejected_public"] == study.old.file_identity(rejected)
    else:
        assert target.read_bytes() == payloads[0]
