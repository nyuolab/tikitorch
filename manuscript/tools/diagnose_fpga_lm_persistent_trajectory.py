"""CPU-only matched persistent-EF/unrounded-native-rule trajectory diagnostic.

This is not exact-CE SGD, a new optimizer recipe, held-out evaluation, or FPGA
execution. Existing scientific sources and artifacts remain immutable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_learning_contract as prior  # noqa: E402
from manuscript.tools import fpga_lm_trajectory_arithmetic as arithmetic  # noqa: E402
from manuscript.tools import fpga_lm_trajectory_reference as extension  # noqa: E402

native, reference, numeric, pilot = prior.native, prior.reference, prior.numeric, prior.pilot
SCHEMA = "tikitorch.fpga_lm_persistent_trajectory.v1"
ARMS, STEPS, K = ("historical", "diverse"), 32, 18
CHECKPOINTS, PROBES = (0, 1, 8, 32), (0, 37, 511)
ANCHORS = (("new", 8), ("new", 32), ("historical", 256), ("historical", 1024))
VARIANTS = prior.VARIANTS
DEADLINE_S = 1800
PLAN = REPO / "manuscript/research/PLAN.md"
WEIGHTS = reference.WEIGHTS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return pilot._finite_json(Path(path).read_text())


def equal(actual, expected, label):
    prior.equal(actual, expected, label)


def source_identity():
    result = {**prior.source_identity(), **extension.source_identity(), **arithmetic.source_identity()}
    for relative in ("manuscript/tools/diagnose_fpga_lm_persistent_trajectory.py",
                     "tests/test_fpga_lm_persistent_trajectory.py",
                     "tests/test_fpga_lm_persistent_lifecycle.py",
                     "tests/test_fpga_lm_trajectory_reference.py",
                     "tests/test_fpga_lm_trajectory_arithmetic.py"):
        result[relative] = digest((REPO / relative).read_bytes())
    return result


def recipe():
    return {"schema": SCHEMA, "cpu_only": True, "torch_threads": 1,
        "arms": list(ARMS), "updates_per_trajectory": STEPS,
        "trajectories": ["native_ef", "unrounded_native_rule"],
        "training_windows": list(range(STEPS)), "partition": "train",
        "checkpoints": list(CHECKPOINTS), "probe_windows": list(PROBES),
        "checkpoint_models": ["native_visible", "native_shadow", "unrounded_native_rule"],
        "checkpoint_context": [True, False], "k": K, "beta": reference.BETA,
        "float_control": "continuous FP64 weights; same floor forward, dynamic routes, raw LUT head and legacy floor backward; w-=G/2**18",
        "float_control_is_exact_ce_sgd": False,
        "shadow": "z=w-r/2**18; z_after=z_before-G_native/2**18",
        "paired_identity": "z_native-w_float=-sum(G_native-G_float)/2**18",
        "initial_state": "arm seed0 step0 from complete prior study; exact weights and zero remainders",
        "anchors": [{"source": source, "step": step} for source, step in ANCHORS],
        "historical_anchor_scope": "saved shuffled-order states, NOT later checkpoints of the new consecutive trajectory",
        "probe_variants": list(VARIANTS), "probe_scales": [1.0],
        "probe_remainders": "preserve anchor r in every candidate; never commit candidate",
        "probe_candidates": "fresh-only w-G/2**18 and accumulated ideal z-G/2**18 versus rounded w-trunc((r+G)/2**18); affine, rounded affine, fixed floor, dynamic floor",
        "range_scope": "all candidate parameters and dynamic/fixed-integer activations bounded; affine-only internal values are unconstrained nondeployable counterfactuals",
        "native_updates": 64, "fp64_updates": 64, "counterfactual_candidates": 120,
        "new_gradient_pair_exposures": 16384, "distinct_new_gradient_pairs": 4096,
        "new_count_fits": 0, "summed_inherited_fit_exposures": 524288,
        "summed_trajectory_history_exposures": 540672, "distinct_history_pairs": 131072,
        "counterfactual_gradient_pair_evaluations_not_updates": 15360,
        "overall_deadline_s": DEADLINE_S, "heldout_corpus_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False,
        "statistical_significance_claim": False, "adaptive_retry_or_extension": False}


def file_identity(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def bind_inputs(config):
    old = prior.check(config.prior_run, config.prior_record)
    old_manifest = read(Path(config.prior_run) / "manifest.json")
    binding, raw, decoded = prior.prior_binding(old_manifest["prior"]["prior_run"],
        old_manifest["prior"]["prior_record"], config.corpus)
    equal(old["recipe"], prior.recipe(), "complete learning-contract predecessor")
    paths = [Path(config.prior_record), *[Path(config.prior_run) / name
             for name in ("manifest.json", "summary.json", "artifact_manifest.json")]]
    late = {}
    shape = pilot.layout(512)
    for arm in ARMS:
        late[arm] = {}
        for step in (256, 1024):
            path = Path(binding["prior_run"]) / (arm + "_seed_0") / ("state_%06d.i32" % step)
            state = native.decode_state(path.read_bytes(), shape)
            equal([state["step"], state["k"], state["config"]["loss_kind"], state["config"]["zshift"]],
                  [step, K, 1, 2], "historical late anchor configuration")
            late[arm][str(step)] = str(path)
            paths.append(path)
    return {"initial": binding, "late_states": late,
            "learning_contract_run": str(Path(config.prior_run).resolve(strict=True)),
            "learning_contract_record": str(Path(config.prior_record).resolve(strict=True)),
            "inputs": binding["inputs"] + [file_identity(path) for path in paths]}, raw, decoded


def verify_inputs(manifest):
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "frozen recipe")
    equal(manifest["source_hashes"], source_identity(), "frozen source/test identities")
    binding = manifest["binding"]
    inventory = {item["path"]: item for item in binding["inputs"]}
    if len(inventory) != len(binding["inputs"]):
        raise ValueError("duplicate bound input path")
    for item in binding["inputs"]:
        equal(file_identity(item["path"]), item, "bound input")
    initial = binding["initial"]
    original = Path(initial["prior_run"])
    learning = Path(binding["learning_contract_run"])
    executable = Path(initial["executable"]["path"])
    references = [initial["prior_record"], binding["learning_contract_record"], str(executable),
        str(executable.parent / "build.json"),
        *[str(directory / name) for directory in (original, learning)
          for name in ("manifest.json", "summary.json", "artifact_manifest.json")]]
    equal(set(initial["states"]), set(ARMS), "initial arm mapping")
    equal(set(binding["late_states"]), set(ARMS), "late arm mapping")
    for arm in ARMS:
        equal(initial["states"][arm], str(original / (arm + "_seed_0/state_000000.i32")), "canonical initial state")
        equal(set(binding["late_states"][arm]), {"256", "1024"}, "late step mapping")
        references.append(initial["states"][arm])
        for step in (256, 1024):
            path = binding["late_states"][arm][str(step)]
            equal(path, str(original / (arm + "_seed_0") / ("state_%06d.i32" % step)), "canonical late state")
            references.append(path)
    for path in references:
        if path not in inventory:
            raise ValueError("referenced artifact is outside the bound input inventory")
    for item in initial["inputs"]:
        equal(inventory.get(item["path"]), item, "original input inventory identity")
    equal(initial, read(learning / "manifest.json")["prior"], "exact learning-contract predecessor binding")


def save_arrays(path, arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **{key: value.detach().cpu().numpy() for key, value in arrays.items()})
    return {"file": Path(path).name, "bytes": Path(path).stat().st_size,
            "sha256": digest(Path(path).read_bytes())}


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(keys) or set(saved.files) != set(keys):
            raise ValueError("complete tensor archive keys differ")
        result = {}
        for key in keys:
            array = saved[key]
            if array.dtype != np.dtype("float64") or not np.isfinite(array).all():
                raise ValueError("finite full FP64 tensor archive required")
            result[key] = torch.from_numpy(array.copy())
        return result


def tensor_hash(values):
    return prior.vector_hash(prior.vector(values))


def state_tensors(state):
    shape = pilot.layout(512)
    return prior.shaped(state["weights"], shape), prior.shaped(state["remainders"], shape)


def subtract(left, right):
    if set(left) != set(WEIGHTS) or set(right) != set(WEIGHTS) or any(left[key].shape != right[key].shape for key in WEIGHTS):
        raise ValueError("full parameter geometry differs; broadcasting is forbidden")
    return {name: left[name] - right[name] for name in WEIGHTS}


def stats(values):
    vector = prior.vector(values)
    return {"l1": float(vector.abs().sum()), "l2": float(vector.norm()),
            "max_abs": float(vector.abs().max()), "nonzero_words": int(torch.count_nonzero(vector))}


def require_zero(values, label):
    error = stats(values)["max_abs"]
    if error != 0:
        raise ValueError(label + " differs: " + str(error))
    return error


def score(weights, tokens, targets, use_context=True):
    base = extension.forward(weights, tokens, use_context=use_context)
    return prior.metrics(base["logits"], targets), base


def checkpoint(root, step, endpoint, weights, remainders, floating, raw, dataset):
    native_receipt = endpoint.save_checkpoint(root, "native_%06d" % step)
    float_receipt = save_arrays(root / ("float_%06d.npz" % step), floating)
    before = endpoint.export_state()
    rows = []
    models = {"native_visible": weights, "native_shadow": arithmetic.shadow(weights, remainders, k=K),
              "unrounded_native_rule": floating}
    for index in PROBES:
        tokens, targets = pilot.window(raw, dataset, "train", index)
        tokens = torch.tensor(tokens, dtype=torch.int64)
        bases = {}
        for model, values in models.items():
            on, base = score(values, tokens, targets)
            off, _ = score(values, tokens, targets, False)
            bases[model] = base
            rows.append({"model": model, "window_index": index, "context_on": on, "context_off": off,
                "context_benefit": {key: off[key] - on[key] for key in ("true_ce", "lut_nll")}})
        if step == 0:
            for name in models:
                equal(bases[name]["logits"].tolist(), bases["native_visible"]["logits"].tolist(), "identical starting logits")
                if any(prior.route_changes(bases[name], bases["native_visible"]).values()):
                    raise ValueError("initial full routes differ")
    if endpoint.export_state() != before:
        raise ValueError("checkpoint inference mutated full native state")
    record = {"step": step, "native": native_receipt, "float": float_receipt,
              "float_tensor_sha256": tensor_hash(floating), "inference_state_preserved": True,
              "probe_scores": rows}
    pilot.write_json(root / ("checkpoint_%06d.json" % step), record)
    return record


def native_parity(outputs, base, info, gradient, before_w, before_r, after):
    if outputs["predictions"] != info["head"]["predictions"].tolist():
        raise ValueError("native predictions differ")
    if outputs["gradients"] != info["head"]["native_gradient"].reshape(-1).tolist():
        raise ValueError("native raw head gradient differs")
    if outputs["native_routes"] != {key: value.reshape(-1).tolist() for key, value in base["routes"].items()}:
        raise ValueError("native routes differ")
    expected = arithmetic.ef_step(before_w, before_r, gradient, k=K)
    expected_w, expected_r = expected["weights"], expected["remainders"]
    after_w, after_r = state_tensors(after)
    require_zero(subtract(after_w, expected_w), "full native weights")
    require_zero(subtract(after_r, expected_r), "full native remainders")
    return after_w, after_r


def trajectories(root, binding, raw, decoded):
    arms = []
    for arm in ARMS:
        directory = root / arm
        directory.mkdir()
        weights, remainders = state_tensors(decoded[arm])
        floating = prior.clone(weights)
        cumulative = {key: torch.zeros_like(value) for key, value in weights.items()}
        buffer = prior.screen.initialize_buffer(pilot.layout(512), 0, "anchors")
        buffer[pilot.layout(512).OFF_CFG + 1] = 2
        endpoint = native.EFSGD(binding["initial"]["executable"]["path"], pilot.layout(512), buffer, K,
                               timeout_s=120, log_path=directory / "native.log")
        rows, checkpoints = [], []
        try:
            endpoint.import_state(Path(binding["initial"]["states"][arm]).read_bytes())
            checkpoints.append(checkpoint(directory, 0, endpoint, weights, remainders, floating, raw, binding["initial"]["dataset"]))
            for index in range(STEPS):
                tokens, targets = pilot.window(raw, binding["initial"]["dataset"], "train", index)
                tensor = torch.tensor(tokens, dtype=torch.int64)
                native_before, native_base = score(weights, tensor, targets)
                float_before, float_base = score(floating, tensor, targets)
                native_info = extension.gradient(weights, tensor, targets, native_base, variant="legacy")
                float_info = extension.gradient(floating, tensor, targets, float_base, variant="legacy")
                ng, fg = native_info["gradients"], float_info["gradients"]
                gradient_file = save_arrays(directory / ("gradients_%06d.npz" % (index + 1)),
                    {**{"native_" + key: ng[key] for key in WEIGHTS}, **{"float_" + key: fg[key] for key in WEIGHTS}})
                previous_shadow = arithmetic.shadow(weights, remainders, k=K)
                before_hashes = {"native": tensor_hash(weights), "remainder": tensor_hash(remainders), "float": tensor_hash(floating)}
                outputs = endpoint.train(tokens, targets, routes=True)
                equal(outputs["step"], index + 1, "persistent step counter")
                after_state = native.decode_state(endpoint.export_state(), pilot.layout(512))
                next_w, next_r = native_parity(outputs, native_base, native_info, ng, weights, remainders, after_state)
                next_float = arithmetic.unrounded_step(floating, fg, k=K)
                shadow = arithmetic.shadow(next_w, next_r, k=K)
                shadow_error = require_zero({key: shadow[key] - previous_shadow[key] + ng[key] / (1 << K) for key in WEIGHTS}, "EF shadow recurrence")
                cumulative = {key: cumulative[key] + ng[key] - fg[key] for key in WEIGHTS}
                paired_error = require_zero({key: shadow[key] - next_float[key] + cumulative[key] / (1 << K) for key in WEIGHTS}, "paired accumulated-gradient identity")
                native_after, native_after_base = score(next_w, tensor, targets)
                float_after, float_after_base = score(next_float, tensor, targets)
                row = {"step": index + 1, "window_index": index, "gradient_file": gradient_file,
                    "before_hashes": before_hashes,
                    "after_hashes": {"native": tensor_hash(next_w), "remainder": tensor_hash(next_r), "float": tensor_hash(next_float)},
                    "native_before": native_before, "native_after": native_after,
                    "float_before": float_before, "float_after": float_after,
                    "native_full_gradient_weight_remainder_parity": True,
                    "shadow_recurrence_error": shadow_error, "paired_gradient_identity_error": paired_error,
                    "native_visible_motion": stats(subtract(next_w, weights)),
                    "native_shadow_motion": stats(subtract(shadow, previous_shadow)),
                    "float_motion": stats(subtract(next_float, floating)),
                    "pending_native_motion": stats(subtract(next_w, shadow)),
                    "native_shadow_minus_float": stats(subtract(shadow, next_float)),
                    "native_visible_minus_float": stats(subtract(next_w, next_float)),
                    "native_gradient_vs_float": prior.gradient_stats(ng, fg),
                    "pre_update_route_disagreement": prior.route_changes(native_base, float_base),
                    "native_route_changes": prior.route_changes(native_base, native_after_base),
                    "float_route_changes": prior.route_changes(float_base, float_after_base),
                    "native_mass_range": [int(native_info["head"]["rounded_totals"].min()), int(native_info["head"]["rounded_totals"].max())],
                    "float_mass_range": [int(float_info["head"]["rounded_totals"].min()), int(float_info["head"]["rounded_totals"].max())]}
                pilot.write_json(directory / ("step_%06d.json" % (index + 1)), row)
                rows.append(row)
                weights, remainders, floating = next_w, next_r, next_float
                if index + 1 in CHECKPOINTS:
                    checkpoints.append(checkpoint(directory, index + 1, endpoint, weights, remainders, floating, raw, binding["initial"]["dataset"]))
            arms.append({"arm": arm, "updates_per_trajectory": STEPS, "steps": rows, "checkpoints": checkpoints})
        finally:
            endpoint.close()
    return arms


def later_probes(root, binding, raw):
    directory = root / "probes"
    directory.mkdir()
    rows = []
    for arm in ARMS:
        for source, step in ANCHORS:
            path = (root / arm / ("native_%06d.i32" % step) if source == "new" else
                    Path(binding["late_states"][arm][str(step)]))
            identity = file_identity(path)
            state = native.decode_state(path.read_bytes(), pilot.layout(512))
            equal([state["step"], state["k"], state["config"]["loss_kind"], state["config"]["zshift"]],
                  [step, K, 1, 2], "late numerical anchor")
            weights, remainders = state_tensors(state)
            shadow = arithmetic.shadow(weights, remainders, k=K)
            for index in PROBES:
                tokens, targets = pilot.window(raw, binding["initial"]["dataset"], "train", index)
                tokens = torch.tensor(tokens, dtype=torch.int64)
                before, base = score(weights, tokens, targets)
                tape = extension.reference_base(base)
                shadow_metrics, _ = score(shadow, tokens, targets)
                shadow_affine = prior.metrics(reference.forward(shadow, tokens, mode="fixed_routes", base=tape)["logits"], targets)
                expected = prior.observer.observe(state["weights"], tokens.tolist(), targets,
                    pilot.layout(512), zshift=2, include_parameter_gradients=True)
                if expected["saturation_events"] != 0 or expected["route_diagnostics"]["backward_int32_overflow_detected"] is not False:
                    raise ValueError("late observer arithmetic guard failed")
                old_logits = prior.initializer.calibration.integer_logits(state["weights"], tokens.tolist(), pilot.layout(512))
                if old_logits["saturation_events"] != 0 or old_logits["logits"] != base["logits"].tolist():
                    raise ValueError("late independent logits differ")
                candidates, arrays, exact = [], {}, None
                for variant in VARIANTS:
                    info = extension.gradient(weights, tokens, targets, base, variant=variant)
                    gradient = info["gradients"]
                    if variant == "exact_ce":
                        exact = gradient
                    if variant == "legacy":
                        if any(gradient[key].reshape(-1).tolist() != expected["parameter_gradients"][key] for key in WEIGHTS):
                            raise ValueError("late full raw observer gradient differs")
                        if info["head"]["native_gradient"].reshape(-1).tolist() != expected["gradients"]:
                            raise ValueError("late raw head gradient differs")
                        if {key: value.reshape(-1).tolist() for key, value in base["routes"].items()} != expected["native_routes"]:
                            raise ValueError("late observer routes differ")
                    arrays.update({variant + "_" + key: gradient[key] for key in WEIGHTS})
                    fresh = {key: weights[key] - gradient[key] / (1 << K) for key in WEIGHTS}
                    ideal = {key: shadow[key] - gradient[key] / (1 << K) for key in WEIGHTS}
                    rounded = {key: weights[key] - torch.trunc((remainders[key] + gradient[key]) / (1 << K)) for key in WEIGHTS}
                    # Affine internal values are intentionally unconstrained; all
                    # candidate parameters still obey the declared value bounds.
                    extension.parameter_digest(fresh)
                    extension.parameter_digest(ideal)
                    if variant == "legacy":
                        oracle = arithmetic.ef_step(weights, remainders, gradient, k=K)
                        require_zero(subtract(rounded, oracle["weights"]), "late EF quotient")
                        require_zero(subtract(ideal, arithmetic.shadow(oracle["weights"], oracle["remainders"], k=K)), "late shadow successor")
                    dynamic, dynamic_base = score(rounded, tokens, targets)
                    candidates.append({"variant": variant, "scale": 1.0,
                        "gradient_statistics": prior.gradient_stats(gradient, exact),
                        "fresh_only_affine": prior.metrics(reference.forward(fresh, tokens, mode="fixed_routes", base=tape)["logits"], targets),
                        "accumulated_ideal_affine": prior.metrics(reference.forward(ideal, tokens, mode="fixed_routes", base=tape)["logits"], targets),
                        "rounded_affine": prior.metrics(reference.forward(rounded, tokens, mode="fixed_routes", base=tape)["logits"], targets),
                        "fixed_integer": prior.metrics(reference.forward(rounded, tokens, mode="fixed_integer", base=tape)["logits"], targets),
                        "dynamic_integer": dynamic, "dynamic_route_changes": prior.route_changes(base, dynamic_base),
                        "changed_weight_words": stats(subtract(rounded, weights))["nonzero_words"],
                        "rounded_weight_sha256": tensor_hash(rounded),
                        "raw_gradient_sha256": tensor_hash(gradient),
                        "legacy_ef_algebra_checked": True if variant == "legacy" else None})
                stem = "%s_%s_%06d_%03d" % (arm, source, step, index)
                archive = save_arrays(directory / (stem + ".npz"), arrays)
                row = {"arm": arm, "source": source, "step": step, "window_index": index,
                    "anchor": identity, "initial_metrics": before, "shadow_dynamic_before": shadow_metrics,
                    "shadow_affine_before": shadow_affine, "pending_motion": stats(subtract(weights, shadow)),
                    "gradient_query_point": "visible anchor weights, not shadow weights",
                    "initial_remainder_sha256": tensor_hash(remainders),
                    "independent_raw_logits_routes_head_parameter_gradient_equal": True,
                    "candidate_committed": False, "gradient_file": archive, "candidates": candidates}
                if file_identity(path) != identity:
                    raise ValueError("probe mutated its source checkpoint")
                pilot.write_json(directory / (stem + ".json"), row)
                rows.append(row)
    return rows


def local_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ValueError("noncanonical artifact path")
    candidate = root / path
    if candidate.is_symlink() or root.resolve() not in candidate.resolve(strict=True).parents:
        raise ValueError("artifact path escape")
    return candidate


def check_archive(root, item):
    path = local_file(root, item["file"])
    equal({"file": item["file"], "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}, item, "full tensor archive receipt")
    return path


def validate_summary(root, manifest, summary):
    """Validate coverage and replay saved optimizer algebra; no model forward."""
    equal(summary["schema"], SCHEMA, "summary schema")
    equal(summary["complete"], True, "complete execution")
    equal(summary["recipe"], recipe(), "summary recipe")
    equal(summary["source_hashes"], source_identity(), "summary sources")
    equal(summary["scope"], "CPU learning-rule diagnostic; no held-out quality or FPGA claim", "scope")
    for flag in ("heldout_corpus_evaluated", "physical_fpga_execution", "paid_execution_authorized"):
        equal(summary[flag], False, flag)
    equal([row["arm"] for row in summary["trajectories"]], list(ARMS), "all trajectory pairs")
    shape = pilot.layout(512)
    for arm in summary["trajectories"]:
        directory = root / arm["arm"]
        equal(arm["updates_per_trajectory"], STEPS, "both trajectory budgets")
        equal([row["step"] for row in arm["steps"]], list(range(1, STEPS + 1)), "full step coverage")
        equal([row["step"] for row in arm["checkpoints"]], list(CHECKPOINTS), "full checkpoint coverage")
        state = native.decode_state(Path(manifest["binding"]["initial"]["states"][arm["arm"]]).read_bytes(), shape)
        weights, remainders = state_tensors(state)
        floating = prior.clone(weights)
        cumulative = {key: torch.zeros_like(value) for key, value in weights.items()}
        for step in range(STEPS + 1):
            if step in CHECKPOINTS:
                checkpoint_row = arm["checkpoints"][CHECKPOINTS.index(step)]
                equal(read(directory / ("checkpoint_%06d.json" % step)), checkpoint_row, "raw checkpoint versus summary")
                equal(checkpoint_row["inference_state_preserved"], True, "state-preserving inference")
                equal([[item["window_index"], item["model"]] for item in checkpoint_row["probe_scores"]],
                    [[index, model] for index in PROBES for model in recipe()["checkpoint_models"]], "all checkpoint inference models/windows")
                for item in checkpoint_row["probe_scores"]:
                    equal(item["context_benefit"], {key: item["context_off"][key] - item["context_on"][key] for key in ("true_ce", "lut_nll")}, "context difference")
                equal(checkpoint_row["native"]["file"], "native_%06d.i32" % step, "native canonical filename")
                payload = local_file(directory, checkpoint_row["native"]["file"]).read_bytes()
                saved = native.decode_state(payload, shape)
                equal([saved["step"], saved["k"], saved["config"]["loss_kind"], saved["config"]["zshift"]], [step, K, 1, 2], "saved native state configuration")
                equal(checkpoint_row["native"]["sha256"], digest(payload), "native full payload hash")
                equal(checkpoint_row["native"]["bytes"], len(payload), "native full payload bytes")
                executable = Path(manifest["binding"]["initial"]["executable"]["path"])
                identity = {"schema": native.SCHEMA, "executable_sha256": digest(executable.read_bytes()),
                    "build_sha256": digest((executable.parent / "build.json").read_bytes()),
                    "source_hashes": native.source_identity(), "layout": pilot.layout_values(shape),
                    "cpu_only": True, "existing_afi_compatible": False}
                equal(checkpoint_row["native"], {"schema": native.SCHEMA, "file": "native_%06d.i32" % step,
                    "bytes": len(payload), "sha256": digest(payload), "step": step, "k": K,
                    "config": saved["config"], "identity": identity,
                    "regions": [{"name": name, "weight_words": getattr(shape, "WORDS_" + name),
                                 "remainder_words": getattr(shape, "WORDS_" + name)} for name in WEIGHTS],
                    "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}, "canonical complete checkpoint receipt")
                equal(read(directory / ("native_%06d.json" % step)), checkpoint_row["native"], "native complete receipt")
                saved_w, saved_r = state_tensors(saved)
                require_zero(subtract(saved_w, weights), "replayed saved native weights")
                require_zero(subtract(saved_r, remainders), "replayed saved native remainders")
                equal(checkpoint_row["float"]["file"], "float_%06d.npz" % step, "FP64 canonical filename")
                fp = load_arrays(check_archive(directory, checkpoint_row["float"]), WEIGHTS)
                require_zero(subtract(fp, floating), "replayed full FP64 weights")
                equal(checkpoint_row["float_tensor_sha256"], tensor_hash(fp), "full FP64 tensor digest")
            if step == STEPS:
                break
            row = arm["steps"][step]
            equal(read(directory / ("step_%06d.json" % (step + 1))), row, "raw step versus summary")
            equal(row["window_index"], step, "exact consecutive training order")
            equal(row["native_full_gradient_weight_remainder_parity"], True, "full native step parity")
            equal(row["shadow_recurrence_error"], 0.0, "shadow identity")
            equal(row["paired_gradient_identity_error"], 0.0, "paired identity")
            equal(row["before_hashes"], {"native": tensor_hash(weights), "remainder": tensor_hash(remainders), "float": tensor_hash(floating)}, "before-state chain")
            equal(row["gradient_file"]["file"], "gradients_%06d.npz" % (step + 1), "gradient canonical filename")
            arrays = load_arrays(check_archive(directory, row["gradient_file"]),
                [prefix + key for prefix in ("native_", "float_") for key in WEIGHTS])
            ng, fg = ({key: arrays[prefix + key] for key in WEIGHTS} for prefix in ("native_", "float_"))
            previous_shadow = arithmetic.shadow(weights, remainders, k=K)
            result = arithmetic.ef_step(weights, remainders, ng, k=K)
            next_w, next_r = result["weights"], result["remainders"]
            next_float = arithmetic.unrounded_step(floating, fg, k=K)
            shadow = arithmetic.shadow(next_w, next_r, k=K)
            cumulative = {key: cumulative[key] + ng[key] - fg[key] for key in WEIGHTS}
            require_zero({key: shadow[key] - previous_shadow[key] + ng[key] / (1 << K) for key in WEIGHTS}, "replayed shadow recurrence")
            require_zero({key: shadow[key] - next_float[key] + cumulative[key] / (1 << K) for key in WEIGHTS}, "replayed accumulated-gradient identity")
            equal(row["after_hashes"], {"native": tensor_hash(next_w), "remainder": tensor_hash(next_r), "float": tensor_hash(next_float)}, "after-state chain")
            for label, values in (("native_visible_motion", subtract(next_w, weights)), ("native_shadow_motion", subtract(shadow, previous_shadow)),
                ("float_motion", subtract(next_float, floating)), ("pending_native_motion", subtract(next_w, shadow)),
                ("native_shadow_minus_float", subtract(shadow, next_float)), ("native_visible_minus_float", subtract(next_w, next_float))):
                equal(row[label], stats(values), "derived " + label)
            equal(row["native_gradient_vs_float"], prior.gradient_stats(ng, fg), "saved gradient comparison")
            weights, remainders, floating = next_w, next_r, next_float
    expected_cases = [[arm, source, step, index] for arm in ARMS for source, step in ANCHORS for index in PROBES]
    equal([[row["arm"], row["source"], row["step"], row["window_index"]] for row in summary["probes"]], expected_cases, "all later probe cases")
    for row in summary["probes"]:
        stem = "%s_%s_%06d_%03d" % (row["arm"], row["source"], row["step"], row["window_index"])
        equal(read(root / "probes" / (stem + ".json")), row, "raw probe versus summary")
        equal(row["independent_raw_logits_routes_head_parameter_gradient_equal"], True, "late full gradient parity")
        equal(row["candidate_committed"], False, "uncommitted candidate")
        equal(row["gradient_query_point"], "visible anchor weights, not shadow weights", "probe query point")
        path = (root / row["arm"] / ("native_%06d.i32" % row["step"]) if row["source"] == "new" else
                Path(manifest["binding"]["late_states"][row["arm"]][str(row["step"])]))
        equal(file_identity(path), row["anchor"], "exact late anchor binding")
        state = native.decode_state(path.read_bytes(), shape)
        weights, remainders = state_tensors(state)
        equal(row["initial_remainder_sha256"], tensor_hash(remainders), "same persistent probe remainders")
        equal(row["pending_motion"], stats(subtract(weights, arithmetic.shadow(weights, remainders, k=K))), "probe pending motion")
        equal([item["variant"] for item in row["candidates"]], list(VARIANTS), "all probe variants")
        equal(row["gradient_file"]["file"], stem + ".npz", "probe gradient filename")
        arrays = load_arrays(check_archive(root / "probes", row["gradient_file"]), [variant + "_" + key for variant in VARIANTS for key in WEIGHTS])
        exact = {key: arrays["exact_ce_" + key] for key in WEIGHTS}
        for item in row["candidates"]:
            if type(item["scale"]) is not float:
                raise ValueError("candidate scale must retain its declared float type")
            equal(item["scale"], 1.0, "fixed candidate scale")
            gradient = {key: arrays[item["variant"] + "_" + key] for key in WEIGHTS}
            equal(item["raw_gradient_sha256"], tensor_hash(gradient), "full probe gradient")
            equal(item["gradient_statistics"], prior.gradient_stats(gradient, exact), "probe gradient comparison")
            rounded = {key: weights[key] - torch.trunc((remainders[key] + gradient[key]) / (1 << K)) for key in WEIGHTS}
            equal(item["rounded_weight_sha256"], tensor_hash(rounded), "counterfactual rounded weights")
            equal(item["changed_weight_words"], stats(subtract(rounded, weights))["nonzero_words"], "candidate changed words")
            equal(item["legacy_ef_algebra_checked"], True if item["variant"] == "legacy" else None, "legacy EF control")
            if item["variant"] == "legacy":
                require_zero(subtract(rounded, arithmetic.ef_step(weights, remainders, gradient, k=K)["weights"]), "late native EF formula")


def check(root, record):
    torch.set_num_threads(1)
    root = Path(root).resolve(strict=True)
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    verify_inputs(manifest)
    equal(manifest["run_directory"], str(root), "run location")
    equal(read(root / "protocol.json"), recipe(), "actual frozen protocol")
    equal(digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "actual pre-run declaration")
    artifacts = prior.artifact_manifest(root)
    equal(read(root / "artifact_manifest.json"), artifacts, "complete actual artifact tree")
    if any("failure" in row["file"] for row in artifacts):
        raise ValueError("failed execution cannot be complete")
    validate_summary(root, manifest, summary)
    expected = {**summary, "run_directory": str(root),
        "artifact_manifest_sha256": digest((root / "artifact_manifest.json").read_bytes()),
        "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
        "check_scope": "saved full-gradient optimizer algebra replay plus integrity; no forward or training replay"}
    equal(read(record), expected, "public record versus sealed execution")
    return expected


def run(config):
    output, target = (prior.capacity._new_external_path(path) for path in (config.output, config.record))
    if target == output or output in target.parents:
        raise ValueError("public record must be outside immutable raw run")
    sources = source_identity()
    binding, raw, decoded = bind_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding,
        "run_directory": str(output), "plan_sha256": digest(PLAN.read_bytes()),
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"}}
    pilot.write_json(output / "manifest.json", manifest)
    pilot.write_json(output / "protocol.json", recipe())
    with (output / "declaration.md").open("xb") as handle:
        handle.write(PLAN.read_bytes())
    def timeout_handler(signum, frame):
        raise TimeoutError("declared persistent diagnostic deadline exceeded")
    previous = signal.getsignal(signal.SIGALRM)
    installed, published_digest = False, None
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        verify_inputs(manifest)
        paired = trajectories(output, binding, raw, decoded)
        probes = later_probes(output, binding, raw)
        summary = {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": sources,
            "trajectories": paired, "probes": probes, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "scope": "CPU learning-rule diagnostic; no held-out quality or FPGA claim"}
        verify_inputs(manifest)
        validate_summary(output, manifest, summary)
        pilot.write_json(output / "summary.json", summary)
        artifacts = prior.artifact_manifest(output)
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = {**summary, "run_directory": str(output),
            "artifact_manifest_sha256": digest((output / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "saved full-gradient optimizer algebra replay plus integrity; no forward or training replay"}
        pilot.write_json(target, public)
        published_digest = digest((json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        result = check(output, target)
        # Successful cleanup is part of completion, inside the same failure
        # handler that retracts this run's public receipt if anything fails.
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        installed = False
        return result
    except BaseException as error:
        # Retract only the exact receipt this invocation just created. Preserve
        # it with the failed run; never overwrite or remove a concurrent edit.
        if published_digest is not None and target.is_file() and digest(target.read_bytes()) == published_digest:
            rejected = output / "rejected_public.json"
            if rejected.exists():
                raise RuntimeError("cannot preserve rejected public receipt without overwriting") from error
            target.rename(rejected)
        if not (output / "failure.json").exists():
            pilot.write_json(output / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--prior-record", type=Path)
    parser.add_argument("--check", action="store_true")
    config = parser.parse_args(argv)
    if not config.check and any(getattr(config, key) is None for key in ("corpus", "prior_run", "prior_record")):
        parser.error("run requires corpus and complete prior learning-contract run/record")
    if config.check:
        torch.set_num_threads(1)
    result = check(config.output, config.record) if config.check else run(config)
    print(json.dumps({"complete": result["complete"], "output": str(config.output), "check_only": config.check}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
