"""Synthetic coordinate-routing contracts; no corpus experiments or cloud."""
from __future__ import annotations

import copy
import importlib
import json
import signal
from types import SimpleNamespace

import pytest
import torch

from test_fpga_lm_channel_choice import initial_weights, interleaved_fixture, tiny_mapping


ARMS = ("contrast", "contrast_slow", "contrast_mean")


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def tree_bytes(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def study():
    return importlib.import_module("manuscript.tools.diagnose_fpga_lm_routing_contrasts")


def test_only_two_new_sources_extend_frozen_channel_closure(study):
    previous, current = study.channel.source_identity(), study.source_identity()
    assert len(previous) == 77 and len(current) == 79
    assert {key: current[key] for key in previous} == previous
    assert set(current) - set(previous) == {
        "manuscript/tools/diagnose_fpga_lm_routing_contrasts.py",
        "tests/test_fpga_lm_routing_contrasts.py"}


def test_three_predeclared_arms_change_only_coordinate_rates_and_mean_ownership(study):
    assert study.ARMS == ARMS
    recipes = {arm: study.recipe(arm) for arm in ARMS}
    for arm, recipe in recipes.items():
        assert recipe["arm"] == arm
        assert recipe["arm_settings"] == {"arm": arm, "body_lr": .01,
            "contrast_lr": .001 if arm == "contrast_slow" else .01,
            "mean_trainable": arm == "contrast_mean", "mean_lr": .01 if arm == "contrast_mean" else None}
        assert recipe["optimizer_coordinates"] == ["EMBED", "MH", "W_HEAD", "d"] + (["m"] if arm == "contrast_mean" else [])
        assert recipe["raw_gradient_queries"] == recipe["applied_gradient_archives"] == recipe["optimizer_commits"] == 256
        assert recipe["complete_saved_states"] == 257 and recipe["score_window_readouts"] == 256
        assert recipe["batch_size"] == 4 and recipe["sequence"] == 128 and recipe["new_initial_score_evaluations"] == 128
        assert recipe["sampled_training_windows"] == list(range(0, 1024, 16))
        assert recipe["development_windows"] == list(range(88, 152))
        assert recipe["new_gradient_pair_exposures"] == recipe["inherited_fitting_pair_exposures"] == 131072
        assert recipe["inherited_gradient_pair_exposures"] == 0 and recipe["endpoint_history_pair_exposures"] == 262144
        assert recipe["new_baseline_evaluations"] == 0
        for key in ("adaptive_selection", "recipe_promoted", "new_gru_training", "heldout_test_evaluated",
                    "reserved_development_evaluated", "physical_fpga_execution", "paid_execution_authorized"):
            assert recipe[key] is False
        assert "none" in recipe["quality_gate"] and "three" in recipe["quality_gate"]
    for key in ("training_order", "mapping", "batch_size", "sequence", "updates", "sampled_training_windows", "development_windows"):
        assert recipes[ARMS[0]][key] == recipes[ARMS[1]][key] == recipes[ARMS[2]][key]


@pytest.mark.parametrize("arm", [None, True, 0, "contrast_01", "contrast_001", "contrast_mean_01", "other"])
def test_undeclared_arm_or_rate_alias_is_rejected(study, arm):
    with pytest.raises(ValueError):
        study.recipe(arm)


def tiny_cell(study, arm):
    """Natural interleaved synthetic initializer; no scientific source input."""
    spec = interleaved_fixture()
    mapping = tiny_mapping(study.channel, spec)
    weights = initial_weights(study, spec)
    for layer in range(spec.layers):
        weights["MP"][layer, torch.arange(spec.width), spec.alternate[layer]] = -.01
    coordinates = {key: weights[key] for key in ("EMBED", "MH", "W_HEAD")}
    coordinates["d"] = torch.full((spec.layers, spec.width), .005, dtype=torch.float64)
    fixed = {"MB": weights["MB"].clone()}
    if arm == "contrast_mean":
        coordinates["m"] = torch.full_like(coordinates["d"], -.005)
    else:
        fixed["m"] = torch.full_like(coordinates["d"], -.005)
    return {"kind": "tropical", "parameters": weights, "model": None, "prior_logits": None,
            "optimizer_parameters": coordinates, "fixed": fixed, "mapping": mapping, "arm": arm}


def raw_gradient(cell):
    return {key: (torch.arange(value.numel(), dtype=torch.float64).reshape(value.shape) % 13 - 6) / 32
            for key, value in cell["parameters"].items()}


@pytest.mark.parametrize("arm", ARMS)
def test_coordinate_materialization_preserves_exact_zero_minus_point01_start(study, arm):
    cell = tiny_cell(study, arm)
    before = {key: value.clone() for key, value in cell["parameters"].items()}
    study.materialize(cell)
    study.guard.byte_equal(cell["parameters"], before, "identical initial physical function in all arms")
    assert list(cell["optimizer_parameters"]) == ["EMBED", "MH", "W_HEAD", "d"] + (["m"] if arm == ARMS[2] else [])
    assert set(cell["fixed"]) == ({"MB"} if arm == ARMS[2] else {"MB", "m"})
    for name in ("EMBED", "MH", "W_HEAD"):
        assert cell["parameters"][name].data_ptr() == cell["optimizer_parameters"][name].data_ptr()
    for name in ("MP", "MB"):
        assert name not in cell["optimizer_parameters"]


@pytest.mark.parametrize("arm", ARMS)
def test_pullback_matches_actual_scatter_autograd_and_coordinate_finite_differences(study, arm):
    cell = tiny_cell(study, arm)
    study.materialize(cell)
    raw = raw_gradient(cell)
    before = {key: value.clone() for key, value in raw.items()}
    actual = study.pullback(raw, cell)
    independent = study.pullback(raw, cell, independent=True)
    assert set(actual) == set(cell["optimizer_parameters"])
    for key in actual:
        torch.testing.assert_close(actual[key], independent[key], atol=1e-12, rtol=1e-12)
    p = torch.tensor(cell["mapping"]["primary_by_layer"])
    a = torch.tensor(cell["mapping"]["alternate_by_layer"])
    gp = raw["MP"].gather(-1, p[..., None]).squeeze(-1)
    ga = raw["MP"].gather(-1, a[..., None]).squeeze(-1)
    torch.testing.assert_close(actual["d"], gp - ga, atol=0, rtol=0)
    assert bool(actual["d"].any())
    if "m" in actual:
        torch.testing.assert_close(actual["m"], gp + ga, atol=0, rtol=0)
    for name in ("EMBED", "MH", "W_HEAD"):
        study.guard.byte_equal({name: actual[name]}, {name: raw[name]}, "unrescaled body raw G")
    # A linear cotangent probe isolates the exact coordinate map, avoiding a
    # false symmetric finite-difference claim at max/min routing boundaries.
    for name in ("d", "m"):
        if name not in cell["optimizer_parameters"]:
            continue
        parameter = cell["optimizer_parameters"][name]
        centre = float(parameter[0, 0]); epsilon = 1e-6
        losses = []
        for sign in (-1, 1):
            parameter[0, 0] = centre + sign * epsilon
            study.materialize(cell)
            losses.append(float((cell["parameters"]["MP"][0, 0] * raw["MP"][0, 0]).sum()))
        parameter[0, 0] = centre
        assert (losses[1] - losses[0]) / (2 * epsilon) == pytest.approx(float(actual[name][0, 0]), abs=1e-7)
    study.materialize(cell)
    study.guard.byte_equal(raw, before, "pullback cannot mutate raw evidence")


def test_fixed_pair_mean_does_not_fix_selected_output_bias(study):
    cell = tiny_cell(study, ARMS[0])
    study.materialize(cell)
    spec = cell["mapping"]
    primary, alternate = spec["primary_by_layer"][0][0], spec["alternate_by_layer"][0][0]
    signal = torch.zeros(16, dtype=torch.float64)
    signal[primary], signal[alternate] = 2., -2.
    old = (cell["parameters"]["MP"][0, 0] + signal).max()
    fixed = cell["fixed"]["m"].clone()
    cell["optimizer_parameters"]["d"][0, 0] += .125
    study.materialize(cell)
    new = cell["parameters"]["MP"][0, 0] + signal
    assert int(new.argmax()) == primary
    assert float(new.max() - old) == pytest.approx(.125)
    study.guard.byte_equal({"m": cell["fixed"]["m"]}, {"m": fixed}, "fixed coordinate, not fixed output")


def test_mean_arm_first_step_is_correct_coordinate_adam_not_physical_step_matched(study):
    cells = {arm: tiny_cell(study, arm) for arm in ("contrast", "contrast_mean")}
    for cell in cells.values():
        study.materialize(cell)
        raw = {key: torch.zeros_like(value) for key, value in cell["parameters"].items()}
        for layer in range(4):
            raw["MP"][layer, torch.arange(16), cell["mapping"]["primary_by_layer"][layer]] = 1.
        gradient = study.pullback(raw, cell)
        assert torch.equal(gradient["d"], torch.ones_like(gradient["d"]))
        if "m" in gradient:
            assert torch.equal(gradient["m"], gradient["d"])
        optimizer = study.make_optimizer(cell)
        for name, parameter in cell["optimizer_parameters"].items():
            parameter.grad = gradient[name].clone()
        optimizer.step()
        study.materialize(cell)
    a, c = cells["contrast"], cells["contrast_mean"]
    primary = a["mapping"]["primary_by_layer"][0][0]
    alternate = a["mapping"]["alternate_by_layer"][0][0]
    assert float(c["parameters"]["MP"][0, 0, primary]) == pytest.approx(2 * float(a["parameters"]["MP"][0, 0, primary]))
    assert float(c["parameters"]["MP"][0, 0, alternate]) == pytest.approx(-.01)
    assert not torch.equal(a["parameters"]["MP"], c["parameters"]["MP"])
    for name in ("EMBED", "MH", "W_HEAD"):
        study.guard.byte_equal({name: a["parameters"][name]}, {name: c["parameters"][name]}, "body scale remains matched")


@pytest.mark.parametrize("case", ["recovered_fast", "other_sentinel"])
def test_valid_learned_certificate_changes_or_third_winners_are_not_censored(study, case):
    cell = tiny_cell(study, "contrast_mean")
    parameters = cell["optimizer_parameters"]
    off = study.channel.OFF_WEIGHT
    if case == "recovered_fast":
        parameters["m"].fill_(off / 2)
        parameters["d"].fill_(-off / 2)
    else:
        parameters["m"].fill_(off - 1)
        parameters["d"].fill_(.005)
    study.materialize(cell)
    tokens = torch.tensor([0, 1, 2, 3])
    base = study.guard.guarded.forward(cell["parameters"], tokens)
    trace = study.helper.trace(base)
    counts = torch.tensor([study.parent._guard_arrays(base)], dtype=torch.int64)
    actual = study.trace_arrays({**trace, "guard_counts": counts}, cell)
    independent = study.helper.trace(study.helper.forward(cell["parameters"], tokens, independent=True))
    study.guard.byte_equal(actual, independent, "actual full-D guarded/dense trace equality")
    if case == "recovered_fast":
        assert counts[0, 1] == 0 and counts[0, 2] > 0 and counts[0, 4] > 0
    else:
        assert study.channel.classify_routes([trace], cell["mapping"])["total"]["other"] > 0
    bad = counts.clone()
    bad[0, 2] += 1
    with pytest.raises(ValueError):
        study.trace_arrays({**trace, "guard_counts": bad}, cell)


@pytest.mark.parametrize("arm", ARMS)
def test_optimizer_owns_named_coordinates_and_arm_specific_rates_not_physical_mp(study, arm):
    cell = tiny_cell(study, arm)
    study.materialize(cell)
    optimizer = study.make_optimizer(cell)
    assert not optimizer.state
    names = {id(value): key for key, value in cell["optimizer_parameters"].items()}
    seen = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names[id(parameter)]
            seen.append(name)
            expected = .001 if name == "d" and arm == "contrast_slow" else .01
            assert group["lr"] == expected
        assert group["betas"] == (.9, .999) and group["eps"] == 1e-8 and group["weight_decay"] == 0
        assert group["foreach"] is False and group["fused"] is False and group["amsgrad"] is False
    assert seen == list(cell["optimizer_parameters"]) and len(seen) == len(set(seen))
    assert id(cell["parameters"]["MP"]) not in names and id(cell["parameters"]["MB"]) not in names


@pytest.mark.parametrize("arm", ARMS)
def test_256_coordinate_adam_steps_preserve_fixed_buffers_and_canonical_physical_map(study, arm):
    cell = tiny_cell(study, arm)
    study.materialize(cell)
    optimizer = study.make_optimizer(cell)
    buffers = {key: value.clone() for key, value in cell["fixed"].items()}
    initial = {key: value.clone() for key, value in cell["parameters"].items()}
    parameters = cell["optimizer_parameters"]
    first = {key: torch.zeros_like(value) for key, value in parameters.items()}
    second = {key: torch.zeros_like(value) for key, value in parameters.items()}
    for step in range(1, 257):
        raw = {key: value * (1 if step % 2 else -.5) for key, value in raw_gradient(cell).items()}
        gradients = study.pullback(raw, cell)
        old = {key: value.clone() for key, value in parameters.items()}
        for key, parameter in parameters.items():
            parameter.grad = gradients[key].clone()
            first[key] = .9 * first[key] + .1 * gradients[key]
            second[key] = .999 * second[key] + .001 * gradients[key].square()
        optimizer.step()
        study.materialize(cell)
        for key, parameter in parameters.items():
            rate = .001 if key == "d" and arm == "contrast_slow" else .01
            expected = old[key] - rate * first[key] / (1 - .9**step) / (second[key].sqrt() / (1 - .999**step)**.5 + 1e-8)
            torch.testing.assert_close(parameter, expected, atol=1e-12, rtol=1e-11)
            state = optimizer.state[parameter]
            assert float(state["step"]) == step
            torch.testing.assert_close(state["exp_avg"], first[key], atol=1e-12, rtol=1e-11)
            torch.testing.assert_close(state["exp_avg_sq"], second[key], atol=1e-12, rtol=1e-11)
        study.guard.byte_equal(cell["fixed"], buffers, "fixed m/MB cannot acquire optimizer state or drift")
        physical = initial["MP"].clone()
        centre = parameters["m"] if "m" in parameters else cell["fixed"]["m"]
        for layer in range(4):
            physical[layer, torch.arange(16), cell["mapping"]["primary_by_layer"][layer]] = centre[layer] + parameters["d"][layer]
            physical[layer, torch.arange(16), cell["mapping"]["alternate_by_layer"][layer]] = centre[layer] - parameters["d"][layer]
        study.guard.byte_equal({"MP": cell["parameters"]["MP"], "MB": cell["parameters"]["MB"]},
                               {"MP": physical, "MB": initial["MB"]}, "exact physical reconstruction, no false row-sum equality")


@pytest.mark.parametrize("case", ["missing", "extra", "float32", "attached", "nan", "shape"])
def test_pullback_rejects_incomplete_or_invalid_raw_tensors(study, case):
    cell = tiny_cell(study, ARMS[0])
    study.materialize(cell)
    raw = raw_gradient(cell)
    if case == "missing":
        del raw["MP"]
    elif case == "extra":
        raw["unknown"] = torch.zeros(1, dtype=torch.float64)
    elif case == "float32":
        raw["MP"] = raw["MP"].float()
    elif case == "attached":
        raw["MP"].requires_grad_(True)
    elif case == "nan":
        raw["MP"].reshape(-1)[0] = float("nan")
    else:
        raw["MP"] = raw["MP"][:, :1]
    with pytest.raises((ValueError, TypeError)):
        study.pullback(raw, cell)


@pytest.mark.parametrize("case", ["coordinate_alias", "fixed_alias"])
def test_separate_tensor_views_cannot_share_coordinate_or_fixed_storage(study, case):
    cell = tiny_cell(study, "contrast_mean" if case == "coordinate_alias" else "contrast")
    contrast = cell["optimizer_parameters"]["d"]
    if case == "coordinate_alias":
        cell["optimizer_parameters"]["m"] = contrast.detach()
        assert cell["optimizer_parameters"]["m"] is not contrast
    else:
        contrast.fill_(-.005)
        cell["fixed"]["m"] = contrast.detach()
    with pytest.raises(ValueError):
        study.materialize(cell)


@pytest.mark.parametrize("arm", ARMS)
def test_full_state_records_physical_fixed_coordinates_and_real_adam_groups(study, tmp_path, arm):
    cell = tiny_cell(study, arm)
    study.materialize(cell)
    optimizer = study.make_optimizer(cell)
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    zero = study.state_record(tmp_path, 0, cell, optimizer, write=True, storage=storage)
    a0 = study.load(tmp_path / "state_000000.npz")
    keys = ({"physical/" + name for name in cell["parameters"]}
            | {"coordinate/" + name for name in cell["optimizer_parameters"]}
            | {"fixed/" + name for name in cell["fixed"]})
    assert set(a0) == keys and zero["initial_optimizer_state_empty"] is True
    assert zero["coordinate_order"] == list(cell["optimizer_parameters"])
    raw = raw_gradient(cell)
    applied = study.pullback(raw, cell)
    for name, parameter in cell["optimizer_parameters"].items():
        parameter.grad = applied[name].clone()
    optimizer.step()
    # Effective MP is a derived tensor, not an optimized tensor: a stale copy
    # must fail before it can be published as a physically consistent state.
    with pytest.raises(ValueError, match="physical state"):
        study.state_record(tmp_path, 1, cell, optimizer, write=True, storage=storage)
    assert not (tmp_path / "state_000001.npz").exists()
    study.materialize(cell)
    one = study.state_record(tmp_path, 1, cell, optimizer, write=True, storage=storage)
    a1 = study.load(tmp_path / "state_000001.npz")
    assert set(a1) == keys | {prefix + "/" + name for prefix in ("exp_avg", "exp_avg_sq", "step")
                             for name in cell["optimizer_parameters"]}
    assert not any(name in a1 for name in ("exp_avg/MP", "exp_avg/MB", "step/MP", "step/MB"))
    assert study.state_record(tmp_path, 1, cell, optimizer) == one
    for name, parameter in cell["optimizer_parameters"].items():
        study.guard.byte_equal({name: a1["coordinate/" + name]}, {name: parameter}, "actual coordinate value")
        assert float(a1["step/" + name]) == 1 and a1["step/" + name].dtype == optimizer.state[parameter]["step"].dtype
    if arm != "contrast_mean":
        assert "fixed/m" in a1 and "coordinate/m" not in a1 and "exp_avg/m" not in a1
    reset = study.make_optimizer(cell)
    with pytest.raises(ValueError, match="complete.*coordinate Adam state"):
        study.state_record(tmp_path, 1, cell, reset)


@pytest.mark.parametrize("case", ["physical", "coordinate", "fixed", "moment", "counter", "group_rate", "group_ownership"])
def test_state_rejects_inconsistent_full_coordinates_moments_counters_and_groups(study, tmp_path, case):
    cell = tiny_cell(study, "contrast_mean")
    study.materialize(cell)
    optimizer = study.make_optimizer(cell)
    applied = study.pullback(raw_gradient(cell), cell)
    for name, parameter in cell["optimizer_parameters"].items():
        parameter.grad = applied[name].clone()
    optimizer.step()
    study.materialize(cell)
    storage = study.Storage(tmp_path, SimpleNamespace(check=lambda: None))
    study.state_record(tmp_path, 1, cell, optimizer, write=True, storage=storage)
    d = cell["optimizer_parameters"]["d"]
    if case == "physical":
        cell["parameters"]["MP"].reshape(-1)[0] += .1
    elif case == "coordinate":
        d.reshape(-1)[0] += .1
    elif case == "fixed":
        cell["fixed"]["MB"].reshape(-1)[0] += .1
    elif case == "moment":
        optimizer.state[d]["exp_avg"].reshape(-1)[0] += .1
    elif case == "counter":
        optimizer.state[d]["step"].zero_()
    elif case == "group_rate":
        optimizer.param_groups[1]["lr"] = .001
    else:
        optimizer.param_groups[1]["params"] = [cell["optimizer_parameters"]["m"]]
    with pytest.raises(ValueError):
        study.state_record(tmp_path, 1, cell, optimizer)


@pytest.mark.parametrize("pin", [None, "", "0" * 63, "G" * 64, 1])
def test_missing_completed_channel_audit_pin_blocks_before_input_access(study, monkeypatch, tmp_path, pin):
    monkeypatch.setattr(study, "AUDIT_RECEIPT_SHA256", pin)
    monkeypatch.setattr(study.parent.rate.head, "_regular", lambda *a: pytest.fail("unadmitted input accessed"))
    with pytest.raises(ValueError, match="not yet pinned"):
        study._audit_capsule(tmp_path / "audit.json", tmp_path / "run", tmp_path / "public.json")


@pytest.fixture
def audit_capsule(study, tmp_path, monkeypatch):
    from test_fpga_lm_channel_choice import audit_capsule as previous_fixture

    value = previous_fixture.__wrapped__(study, tmp_path, monkeypatch)
    sources = value.value["source_hashes"]
    monkeypatch.setattr(study.channel, "source_identity", lambda: sources)
    value.value.update(schema=study.channel.SCHEMA + ".audit", independent_tropical_window_forwards=1280,
                       supplemental_guarded_window_forwards=1280, score_window_readouts=256,
                       initial_score_parity_checks=128, first_batch_parity_checks=1,
                       all_whole_model_dense_fallback_counts_verified=True)
    manifest_path = value.capsule / "audit_manifest.json"
    manifest = study.read(manifest_path)
    manifest["schema"] = study.channel.SCHEMA
    write_json(manifest_path, manifest)
    value.value["audit_manifest_sha256"] = study.old.digest(manifest_path.read_bytes())
    value.reseal()
    return value


def test_complete_channel_capsule_admission_is_file_only(study, audit_capsule, monkeypatch):
    for name in ("run", "execute", "check", "audit", "initial_model"):
        monkeypatch.setattr(study.channel, name, lambda *a, **k: pytest.fail("old numerical method forbidden"))
    value = audit_capsule
    actual, proof = study._audit_capsule(value.receipt, value.root, value.public)
    assert actual == value.value and set(proof["files"]) == {"audit_manifest.json", "declaration.md", "verification.json"}


@pytest.mark.parametrize("case", ["private_bytes", "incomplete", "source", "count", "deadline", "extra_failure"])
def test_altered_resealed_or_incomplete_prerequisite_cannot_admit_new_arm(study, audit_capsule, case):
    value = audit_capsule
    if case == "private_bytes":
        path = value.capsule / "verification.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif case == "extra_failure":
        write_json(value.capsule / "failure.json", {"complete": False})
    else:
        key, bad = {"incomplete": ("complete", 1), "source": ("source_hashes", {}),
                    "count": ("initial_score_parity_checks", 127),
                    "deadline": ("verification_elapsed_monotonic_seconds", 10800.)}[case]
        value.value[key] = bad
        value.reseal()
    with pytest.raises(ValueError):
        study._audit_capsule(value.receipt, value.root, value.public)


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    """Saved synthetic channel evidence, never a corpus or old-source edit."""
    import test_fpga_lm_channel_choice as fixtures

    core_generator = fixtures.historical.__wrapped__(tmp_path_factory)
    channel_generator = None
    try:
        previous = next(core_generator)
        channel_generator = fixtures.synthetic.__wrapped__(previous)
        yield next(channel_generator)
    finally:
        if channel_generator is not None:
            channel_generator.close()
        core_generator.close()


@pytest.fixture(scope="module")
def synthetic(historical):
    """All three actual tiny runners, sharing only immutable historical inputs."""
    value = historical
    study = importlib.import_module("manuscript.tools.diagnose_fpga_lm_routing_contrasts")
    patch = pytest.MonkeyPatch()
    temporary = value.temporary / "coordinate_study"
    temporary.mkdir()
    source, declaration = temporary / "source.txt", temporary / "PLAN.md"
    source.write_text("Synthetic three-arm coordinate source identity.\n")
    declaration.write_text("Synthetic coordinate contracts, not scientific evidence.\n")
    for key, item in {"TRAIN_WINDOWS": tuple(range(8)), "SCORE_TRAIN_WINDOWS": (0, 4),
                      "DEV_WINDOWS": (88, 89), "UPDATES": 2, "SEQUENCE": 4, "PLAN": declaration}.items():
        patch.setattr(study, key, item)
    patch.setattr(study, "batch_data", value.study.batch_data)
    binding = {"prior_run": str(value.root), "prior_record": str(value.record), "prior_audit": str(value.corpus),
               "channel_binding": copy.deepcopy(value.binding), "audit_receipt_sha256": "a" * 64}
    for key in ("core_binding", "continuation_binding", "endpoint_binding", "development_binding", "dataset", "initial_scores"):
        binding[key] = copy.deepcopy(value.binding[key])
    binding["initial_query"] = study.old.file_identity(value.root / study.channel.ARM / "query_000000.npz")
    path = value.root / study.channel.ARM / "state_000000.npz"
    binding["initial_state"] = {"file": study.old.file_identity(path), "metadata": study.old.file_identity(path.with_suffix(".json"))}
    protected = list(value.protected)
    protected.extend(study.old.file_identity(path) for path in value.root.rglob("*") if path.is_file())
    protected.extend(study.old.file_identity(path) for path in (source, value.record))
    source_identity = lambda: {"synthetic_source": study.old.digest(source.read_bytes())}
    patch.setattr(study, "source_identity", source_identity)
    validation = study._Validation(binding, source_identity(), {}, object())
    binds = []

    def bind(config):
        assert config.arm in ARMS
        binds.append(config.arm)
        return copy.deepcopy(binding), value.corpus.read_bytes(), validation

    def frozen(manifest, token):
        assert isinstance(token, study._Validation)
        study.equal(manifest["schema"], study.SCHEMA, "synthetic schema")
        study.equal(manifest["recipe"], study.recipe(manifest["arm"]), "fixed declared arm")
        study.equal(manifest["binding"], token.binding, "private invocation binding")
        study.equal(token.binding, binding, "original shared historical binding")
        study.equal(manifest["source_hashes"], token.sources, "invocation sources")
        study.equal(token.sources, source_identity(), "current source bytes")
        for descriptor in protected:
            study.equal(study.old.file_identity(descriptor["path"]), descriptor, "immutable source/control/initial state")

    patch.setattr(study, "bind_inputs", bind)
    patch.setattr(study, "frozen", frozen)
    for name in ("run", "check", "execute", "audit"):
        patch.setattr(study.channel, name, lambda *a, **k: pytest.fail("old channel numerical execution forbidden"))
    before_all, cells = tree_bytes(value.root), {}
    try:
        for arm in ARMS:
            config = SimpleNamespace(arm=arm, prior_run=value.root, prior_record=value.record, prior_audit=value.corpus,
                                     corpus=value.corpus, output=temporary / arm, record=temporary / (arm + ".json"))
            before = dict(value.calls)
            result = study.run(config)
            cells[arm] = SimpleNamespace(config=config, result=result, root=config.output, record=config.record,
                producer_calls={key: value.calls[key] - before[key] for key in before})
        assert tree_bytes(value.root) == before_all
        yield SimpleNamespace(study=study, temporary=temporary, source=source, corpus=value.corpus,
                              binding=binding, cells=cells, historical=value, calls=value.calls, binds=binds)
    finally:
        patch.undo()


@pytest.mark.parametrize("arm", ARMS)
def test_each_real_tiny_arm_keeps_all_raw_coordinate_and_physical_states(synthetic, arm):
    value, study = synthetic, synthetic.study
    run = value.cells[arm]
    result = run.result
    assert result["complete"] is True and result["arm"] == arm
    assert result["raw_gradient_queries"] == result["applied_gradient_archives"] == result["optimizer_commits"] == 2
    assert result["complete_saved_states"] == 3 and result["score_window_readouts"] == 8
    assert result["initial_score_parity_checks"] == 4 and result["first_batch_parity_checks"] == 1
    assert result["new_original_model_forwards"] == result["new_baseline_evaluations"] == 0
    assert result["new_gru_training"] is False
    assert run.producer_calls == {"guarded": 16, "dense": 0, "manual_g": 8, "independent_g": 0}
    directory = run.root / arm
    previous = study.load(directory / "state_000000.npz")
    names = result["states"][0]["coordinate_order"]
    mapping = result["transform"]["mapping"]
    p = torch.tensor(mapping["primary_by_layer"]).unsqueeze(-1)
    a = torch.tensor(mapping["alternate_by_layer"]).unsqueeze(-1)
    initial = {key: tensor.clone() for key, tensor in previous.items()}
    for step in range(2):
        query = study.load(directory / f"query_{step:06d}.npz")
        raw = {key[2:]: tensor for key, tensor in query.items() if key.startswith("g_")}
        applied = study.load(directory / f"applied_{step:06d}.npz")
        gp, ga = raw["MP"].gather(-1, p).squeeze(-1), raw["MP"].gather(-1, a).squeeze(-1)
        expected_g = {key: raw[key] for key in ("EMBED", "MH", "W_HEAD")}
        expected_g["d"] = gp - ga
        if "m" in names:
            expected_g["m"] = gp + ga
        study.guard.byte_equal(applied, expected_g, "actual full raw G -> saved active coordinate G")
        after = study.load(directory / f"state_{step + 1:06d}.npz")
        for name in names:
            parameter = previous["coordinate/" + name]
            first = .9 * previous.get("exp_avg/" + name, torch.zeros_like(parameter)) + .1 * applied[name]
            second = .999 * previous.get("exp_avg_sq/" + name, torch.zeros_like(parameter)) + .001 * applied[name].square()
            rate = .001 if name == "d" and arm == "contrast_slow" else .01
            expected = parameter - rate * first / (1 - .9**(step + 1)) / (second.sqrt() / (1 - .999**(step + 1))**.5 + 1e-8)
            torch.testing.assert_close(after["coordinate/" + name], expected, atol=1e-12, rtol=1e-11)
            assert float(after["step/" + name]) == step + 1
            torch.testing.assert_close(after["exp_avg/" + name], first, atol=1e-12, rtol=1e-11)
        centre = after["coordinate/m"] if "m" in names else after["fixed/m"]
        physical = initial["physical/MP"].clone()
        for layer in range(4):
            physical[layer, torch.arange(16), mapping["primary_by_layer"][layer]] = centre[layer] + after["coordinate/d"][layer]
            physical[layer, torch.arange(16), mapping["alternate_by_layer"][layer]] = centre[layer] - after["coordinate/d"][layer]
        study.guard.byte_equal({"MP": after["physical/MP"]}, {"MP": physical}, "actual saved physical matrix from saved coordinates")
        for name in ("EMBED", "MH", "W_HEAD"):
            study.guard.byte_equal({name: after["physical/" + name]}, {name: after["coordinate/" + name]}, "body parameter correspondence")
        study.guard.byte_equal({k: tensor for k, tensor in after.items() if k.startswith("fixed/")},
                               {k: tensor for k, tensor in initial.items() if k.startswith("fixed/")}, "no fixed-buffer optimizer drift")
        previous = after


def test_all_arms_share_initial_physical_bytes_order_data_and_body_first_update(synthetic):
    value, study = synthetic, synthetic.study
    assert set(value.cells) == set(ARMS)
    old = study.load(value.binding["initial_state"]["file"]["path"])
    reference = {key.removeprefix("parameter/"): tensor for key, tensor in old.items()}
    first_body = None
    for arm, run in value.cells.items():
        assert run.result["training_order"] == study.training_order()
        assert study.read(run.root / "manifest.json")["binding"] == value.binding
        zero = study.load(run.root / arm / "state_000000.npz")
        physical = {key.removeprefix("physical/"): tensor for key, tensor in zero.items() if key.startswith("physical/")}
        study.guard.byte_equal(physical, reference, "same actual archived physical starting point for every arm")
        one = study.load(run.root / arm / "state_000001.npz")
        body = {key: tensor for key, tensor in one.items()
                if key.split("/")[-1] in ("EMBED", "MH", "W_HEAD") and not key.startswith("physical/")}
        if first_body is None:
            first_body = body
        else:
            study.guard.byte_equal(body, first_body, "body first update remains matched, unlike full physical MP")
        assert run.result["initial_parity"]["first_batch"]["optimizer_commits_before_check"] == 0
        assert run.result["recipe_promoted"] is False and run.result["material_progress"]["requires_successful_independent_audit"] is True


@pytest.mark.parametrize("arm", ARMS)
def test_every_tiny_arm_readback_and_independent_audit_are_complete(synthetic, tmp_path, arm):
    value, study = synthetic, synthetic.study
    run = value.cells[arm]
    before, original = dict(value.calls), tree_bytes(run.root)
    assert study.check(run.root, run.record, arm=arm) == run.result
    assert value.calls == before
    config = SimpleNamespace(**vars(run.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    result = study.audit(config)
    assert result["complete"] is True and result["independent_recompute_completed"] is True
    assert {key: value.calls[key] - before[key] for key in before} == {
        "guarded": 16, "dense": 16, "manual_g": 0, "independent_g": 8}
    assert result["applied_coordinate_gradients_verified"] == result["independent_coordinate_pullbacks_verified"] == 2
    assert result["complete_saved_states_verified"] == 3 and result["score_window_readouts"] == 8
    assert result["actual_guard_counters_verified"] is True
    assert (config.audit_output / "verification.json").read_bytes() == config.audit_record.read_bytes()
    assert tree_bytes(run.root) == original


def test_independent_near_gradient_never_substitutes_for_saved_coordinate_adam_g(synthetic, tmp_path, monkeypatch):
    value, study = synthetic, synthetic.study
    run = value.cells["contrast_mean"]
    original = study.pullback
    seen = []

    def pullback(raw, cell, *, independent=False):
        result = original(raw, cell, independent=independent)
        if independent:
            seen.append(True)
            result["d"] = result["d"] + 1e-13
        return result

    monkeypatch.setattr(study, "pullback", pullback)
    before = tree_bytes(run.root)
    config = SimpleNamespace(**vars(run.config), audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
    assert study.audit(config)["complete"] is True
    assert len(seen) == 2 and tree_bytes(run.root) == before


@pytest.mark.parametrize("target", ["raw_g", "applied_d", "applied_m", "moment", "physical", "route", "guard"])
def test_actual_replay_rejects_saved_coordinate_physical_gradient_and_trace_corruption(synthetic, target):
    value, study = synthetic, synthetic.study
    run = value.cells["contrast_mean"]
    filename = {"raw_g": "query_000001.npz", "applied_d": "applied_000001.npz",
                "applied_m": "applied_000001.npz", "moment": "state_000001.npz",
                "physical": "state_000001.npz", "route": "query_000001.npz", "guard": "query_000001.npz"}[target]
    path = run.root / run.config.arm / filename
    original, arrays = path.read_bytes(), study.load(path)
    key = {"raw_g": "g_MP", "applied_d": "d", "applied_m": "m", "moment": "exp_avg/d",
           "physical": "physical/MP", "guard": "guard_counts"}.get(target)
    if target == "route":
        key = next(key for key in arrays if key.endswith("/PROJ_IDX"))
        arrays[key].reshape(-1)[0] = (arrays[key].reshape(-1)[0] + 1) % 16
    elif target == "guard":
        arrays[key][0] = torch.tensor([1, 0, 64, 0, 4, 0])
    else:
        arrays[key].reshape(-1)[0] += .125
    path.unlink()
    study.parent.parent.save(path, arrays)
    try:
        with pytest.raises(ValueError):
            study.execute(study.read(run.root / "manifest.json"), run.root, value.corpus.read_bytes(), recompute=True)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("target", ["source", "initial_state", "initial_query", "initial_score"])
def test_bound_source_initial_and_original_parity_inputs_fail_before_model_calls(synthetic, target):
    value, study = synthetic, synthetic.study
    path = {"source": value.source,
            "initial_state": study.Path(value.binding["initial_state"]["file"]["path"]),
            "initial_query": study.Path(value.binding["initial_query"]["path"]),
            "initial_score": study.Path(value.binding["initial_scores"]["train"][0]["path"])}[target]
    original, calls = path.read_bytes(), dict(value.calls)
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            run = value.cells["contrast"]
            study.check(run.root, run.record)
    finally:
        path.write_bytes(original)
    assert value.calls == calls


def mode_config(value, temporary, mode):
    run = value.cells["contrast_mean"]
    if mode == "producer":
        config = SimpleNamespace(**{**vars(run.config), "output": temporary / "run", "record": temporary / "public.json"})
        return config, value.study.run, config.output, config.record
    config = SimpleNamespace(**vars(run.config), audit_output=temporary / "audit", audit_record=temporary / "audit.json")
    return config, value.study.audit, config.audit_output, config.audit_record


@pytest.mark.parametrize("target", ["raw_gradient", "trace"])
def test_first_raw_batch_mismatch_stops_before_any_coordinate_adam_commit(synthetic, tmp_path, monkeypatch, target):
    value, study = synthetic, synthetic.study
    config, invoke, output, record = mode_config(value, tmp_path, "producer")
    original_load, original_step = study.channel.load, torch.optim.Adam.step
    hits, commits = [], []
    source = study.Path(value.binding["initial_query"]["path"])

    def load(path):
        arrays = original_load(path)
        if study.Path(path) == source:
            hits.append(True)
            key = "g_MP" if target == "raw_gradient" else next(key for key in arrays if key.endswith("/logits"))
            arrays[key] = arrays[key].clone()
            arrays[key].reshape(-1)[0] += .125
        return arrays

    def step(*args, **kwargs):
        commits.append(True)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(study.channel, "load", load)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    with pytest.raises(ValueError):
        invoke(config)
    assert hits and not commits
    assert study.read(output / "failure.json")["complete"] is False and not record.exists()
    assert (output / config.arm / "state_000000.npz").is_file()
    assert not (output / config.arm / "state_000001.npz").exists()


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, TimeoutError])
def test_failed_or_interrupted_attempt_retains_failure_and_restores_timer(synthetic, tmp_path, monkeypatch, mode, error):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)

    def fail(*args, **kwargs):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0 and torch.get_num_threads() == 1
        raise error("synthetic coordinate run interruption")

    monkeypatch.setattr(study, "execute" if mode == "producer" else "_check", fail)
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(error):
        invoke(config)
    failure = study.read(output / "failure.json")
    assert failure["complete"] is False and failure["arm"] == config.arm and not target.exists()
    assert failure["audit_only"] is (mode == "audit")
    assert signal.getsignal(signal.SIGALRM) == handler and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


@pytest.mark.parametrize("mode", ["producer", "audit"])
@pytest.mark.parametrize("replacement", ["owned_corrupted", "foreign"])
def test_failed_publication_retains_corrupted_owned_bytes_but_preserves_foreign_inode(synthetic, tmp_path, monkeypatch, mode, replacement):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "audit":
        monkeypatch.setattr(study, "_check", lambda *args: value.cells[config.arm].result)
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


@pytest.mark.parametrize("mode", ["check", "audit"])
def test_wrong_arm_cannot_relabel_complete_producer(synthetic, tmp_path, mode):
    value, study = synthetic, synthetic.study
    run, before = value.cells["contrast"], dict(value.calls)
    with pytest.raises(ValueError, match="arm"):
        if mode == "check":
            study.check(run.root, run.record, "contrast_slow")
        else:
            config = SimpleNamespace(**{**vars(run.config), "arm": "contrast_slow"},
                audit_output=tmp_path / "audit", audit_record=tmp_path / "audit.json")
            study.audit(config)
    assert value.calls == before
    if mode == "audit":
        assert study.read(config.audit_output / "failure.json")["complete"] is False
        assert not config.audit_record.exists()


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_existing_output_is_rejected_without_model_calls(synthetic, tmp_path, mode):
    value = synthetic
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_bytes(b"user-owned existing directory")
    before = dict(value.calls)
    with pytest.raises((ValueError, FileExistsError)):
        invoke(config)
    assert tree_bytes(output) == {"user.txt": b"user-owned existing directory"}
    assert value.calls == before and not target.exists()


@pytest.mark.parametrize("mode", ["producer", "audit"])
def test_postcheck_exact_seal_or_private_receipt_mutation_cannot_publish_success(synthetic, tmp_path, monkeypatch, mode):
    value, study = synthetic, synthetic.study
    config, invoke, output, target = mode_config(value, tmp_path, mode)
    if mode == "producer":
        original = study._check

        def check(*args):
            result = original(*args)
            seal = output / "artifact_manifest.json"
            before = seal.read_bytes()
            changed = before.replace(b"\n  ", b"\n\t ", 1)
            assert before != changed and len(before) == len(changed)
            seal.write_bytes(changed)
            return result

        monkeypatch.setattr(study, "_check", check)
    else:
        monkeypatch.setattr(study, "_check", lambda *args: value.cells[config.arm].result)
        original = study.frozen

        def frozen(*args):
            original(*args)
            if target.exists():
                private = output / "verification.json"
                private.write_bytes(private.read_bytes() + b"\n")

        monkeypatch.setattr(study, "frozen", frozen)
    with pytest.raises(ValueError):
        invoke(config)
    assert not target.exists() and study.read(output / "failure.json")["complete"] is False
