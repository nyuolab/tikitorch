"""Bounded CPU learning-contract diagnostics, not a new LM quality benchmark.

Frozen scientific sources are imported unchanged. Small teacher optimization,
fixed-point numerical counterfactuals and native one-step parity have different
claims and are recorded separately. No final-test evaluation or cloud execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import check_fpga_lm_bigram_capacity as capacity  # noqa: E402
from manuscript.tools import fpga_lm_diverse_projection as initializer  # noqa: E402
from manuscript.tools import fpga_lm_efsgd as native  # noqa: E402
from manuscript.tools import fpga_lm_float_reference as reference  # noqa: E402
from manuscript.tools import fpga_lm_numeric_diagnostic as numeric  # noqa: E402
from manuscript.tools import observe_fpga_lm_routes as observer  # noqa: E402
from manuscript.tools import screen_fpga_lm_heads as screen  # noqa: E402
from manuscript.tools import train_fpga_lm_diverse_projection as historical  # noqa: E402
from scripts import fpga_lm_pretrain as pilot  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_learning_contract.v1"
ARMS = ("historical", "diverse")
WINDOWS = (0, 37, 511)
VARIANTS = ("exact_ce", "lut_normalized", "truncated_normalized", "variable_mass", "legacy")
SCALES = (0.0625, 0.25, 1.0)
TOY_CELLS = ("fixed_near", "dynamic_near", "dynamic_cold")
TOY_STEPS, DEADLINE_S = 64, 1800
PLAN = REPO / "manuscript/research/PLAN.md"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return pilot._finite_json(Path(path).read_text())


def equal(actual, expected, label):
    historical.confirmation._equal(actual, expected, label)


def source_identity():
    result = {**capacity.source_identity(), **historical.source_identity(), **native.source_identity(),
              **reference.source_identity(), **numeric.source_identity()}
    for relative in ("manuscript/tools/diagnose_fpga_lm_learning_contract.py",
                     "tests/test_fpga_lm_float_reference.py",
                     "tests/test_fpga_lm_numeric_diagnostic.py",
                     "tests/test_fpga_lm_learning_contract.py"):
        result[relative] = digest((REPO / relative).read_bytes())
    return result


def recipe():
    return {"schema": SCHEMA, "cpu_only": True, "torch_threads": 1,
        "teacher_geometry": {"D": 16, "V": 4, "N": 4, "K": 7, "T": 12},
        "teacher_seed": 20260910, "teacher_targets": "exact soft distributions, not sampled labels",
        "teacher_train_windows": 4, "teacher_evaluation_windows": 4,
        "teacher_cells": list(TOY_CELLS), "teacher_updates_per_cell": TOY_STEPS,
        "teacher_rate": "1/L0; L0=beta**2*lambda_max(J0 J0.T)/(2*48), fixed across steps",
        "teacher_scope": "local affine/full-route diagnostic plus separate structured-cold control; not fully random or corpus pretraining",
        "teacher_context_minimum_kl": 1e-6, "teacher_near_relative_kl_goal": 0.5,
        "toy_train_seed": 601, "toy_evaluation_seed": 602,
        "native_arms": list(ARMS), "native_training_windows": list(WINDOWS),
        "native_state": "each case resets to its exact historical step0 state, all remainders zero",
        "native_updates": 6, "native_updates_are_a_training_trajectory": False,
        "numerical_variants": list(VARIANTS), "counterfactual_scales": list(SCALES),
        "head_units": "65536/beta times the per-token normalized raw-logit derivative; no mean reduction",
        "centering": "exact for first4 variants; legacy floors raw cotangents in the last variant",
        "counterfactuals": "90 independent candidates; ideal affine, rounded affine, fixed integer, dynamic integer",
        "counterfactual_rounding": "truncate weight displacement; only legacy scale1 is actual native EFk18",
        "smoothing_counts": [[40, 9, 1, 0], [3, 2, 0, 1]], "smoothing_alpha": 1,
        "stationarity_probe_rates": [1.0, 64.0, 512.0],
        "duplicate_bias_control": "two identical additive quadratic coordinates at eta vs one at2eta",
        "beta": reference.BETA, "native_k": 18, "overall_deadline_s": DEADLINE_S,
        "heldout_corpus_evaluated": False, "physical_fpga_execution": False,
        "paid_execution_authorized": False, "statistical_significance_claim": False,
        "adaptive_retry_or_extension": False}


def clone(weights, grad=False):
    return {name: value.detach().clone().requires_grad_(grad) for name, value in weights.items()}


def vector(weights):
    return torch.cat([weights[name].reshape(-1) for name in reference.WEIGHTS])


def vector_hash(value):
    return digest(value.detach().cpu().numpy().astype("<f8").tobytes())


def shaped(state, shape):
    sizes = {"EMBED": (shape.ML_V, shape.ML_D),
        "MH": (shape.ML_N_LAYERS, shape.ML_D, shape.ML_K),
        "MP": (shape.ML_N_LAYERS, shape.ML_D, shape.ML_D),
        "MB": (shape.ML_N_LAYERS, shape.ML_D), "W_HEAD": (shape.ML_V, shape.ML_D)}
    return {name: torch.tensor(state[name], dtype=torch.float64).reshape(size) for name, size in sizes.items()}


def teacher_weights():
    generator = torch.Generator().manual_seed(20260910)
    weights = {"EMBED": 8 * torch.rand((4, 16), generator=generator, dtype=torch.float64) - 4,
        "MH": torch.zeros((4, 16, 7), dtype=torch.float64),
        "MP": torch.full((4, 16, 16), -8192.0, dtype=torch.float64),
        "MB": torch.zeros((4, 16), dtype=torch.float64),
        "W_HEAD": torch.full((4, 16), -8192.0, dtype=torch.float64)}
    weights["MP"][:, torch.arange(16), torch.arange(16)] = 0
    weights["W_HEAD"][torch.arange(4), torch.arange(4)] = 0
    weights["MH"][0, :, 1] = 32
    for layer in range(1, 4):
        weights["MH"][layer, :, 1:] = 256 if layer % 2 else -256
    return weights


def toy_windows(seed):
    rng = random.Random(seed)
    return [torch.tensor([rng.randrange(4) for _ in range(12)], dtype=torch.int64) for _ in range(4)]


def student_weights(teacher, cold=False):
    generator = torch.Generator().manual_seed(603 if cold else 604)
    weights = clone(teacher)
    for name, value in weights.items():
        amplitude = 2.0 if name == "EMBED" else 0.25
        weights[name] = value + amplitude * (2 * torch.rand(value.shape, generator=generator, dtype=torch.float64) - 1)
    if cold:
        weights["EMBED"] = 16 * torch.rand((4, 16), generator=generator, dtype=torch.float64) - 8
        weights["MH"] = 2 * torch.rand((4, 16, 7), generator=generator, dtype=torch.float64) - 1
    return weights


def soft_metrics(logits, targets):
    log_p = torch.log_softmax(logits * reference.BETA, dim=-1)
    ce = -(targets * log_p).sum(dim=-1).mean()
    entropy = -(targets * targets.log()).sum(dim=-1).mean()
    return ce, ce - entropy


def route_changes(left, right):
    return {name: int((left["routes"][name] != right["routes"][name]).sum())
            for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}


def minimum_margin(base):
    if base.get("margins") is None:
        raise ValueError("actual dynamic route margins required")
    selected = [value[base["candidate_counts"][name] > 1] for name, value in base["margins"].items()]
    selected = torch.cat([value.reshape(-1) for value in selected if value.numel()])
    if not selected.numel() or not torch.isfinite(selected).all():
        raise ValueError("finite nontrivial route margins required")
    return float(selected.min())


def jacobian(weights, windows, bases):
    rows = []
    for tokens, base in zip(windows, bases):
        for index in range(base["logits"].numel()):
            cotangent = torch.zeros_like(base["logits"])
            cotangent.reshape(-1)[index] = 1
            rows.append(vector(reference.manual_vjp(weights, tokens, cotangent, base)))
    return torch.stack(rows)


def toy_diagnostic(root):
    teacher, train, evaluation = teacher_weights(), toy_windows(601), toy_windows(602)
    if set(map(tuple, [x.tolist() for x in train])) & set(map(tuple, [x.tolist() for x in evaluation])):
        raise ValueError("teacher full training/evaluation windows overlap")
    with torch.no_grad():
        teacher_train = [reference.forward(teacher, x) for x in train]
        teacher_eval = [reference.forward(teacher, x) for x in evaluation]
        targets = [torch.softmax(row["logits"] * reference.BETA, dim=-1) for row in teacher_train]
        eval_targets = [torch.softmax(row["logits"] * reference.BETA, dim=-1) for row in teacher_eval]
        context_kl = sum(float(soft_metrics(reference.forward(teacher, x, use_context=False)["logits"], y)[1])
                         for x, y in zip(evaluation, eval_targets)) / len(evaluation)
    teacher_margin = min(minimum_margin(base) for base in teacher_train + teacher_eval)
    if teacher_margin <= 0:
        raise ValueError("teacher has ties; do not adaptively reroll")
    pairs = [(int(token), p) for tokens, probabilities in zip(train, targets)
             for token, p in zip(tokens, probabilities)]
    same_token_difference = max(float((a - b).abs().max()) for token, a in pairs for other, b in pairs if token == other)
    if context_kl <= 1e-6 or same_token_difference <= 1e-6:
        raise ValueError("declared teacher lacks certified context dependence; do not reroll")
    teacher_meta = {"weight_sha256": vector_hash(vector(teacher)), "train_windows": [x.tolist() for x in train],
        "evaluation_windows": [x.tolist() for x in evaluation], "context_disabled_kl": context_kl,
        "minimum_nontrivial_route_margin": teacher_margin,
        "same_current_token_probability_difference": same_token_difference,
        "train_target_sha256": vector_hash(torch.cat(targets)), "evaluation_target_sha256": vector_hash(torch.cat(eval_targets))}
    pilot.write_json(root / "teacher.json", teacher_meta)
    cells = []
    for name in TOY_CELLS:
        weights = student_weights(teacher, cold=name == "dynamic_cold")
        bases = [reference.forward(weights, x) for x in train]
        eval_bases = [reference.forward(weights, x) for x in evaluation]
        initial_margin = min(minimum_margin(base) for base in bases + eval_bases)
        if initial_margin <= 0:
            raise ValueError("declared student has a route tie; directional check is not qualified")
        if name != "dynamic_cold":
            for base, truth in zip(bases + eval_bases, teacher_train + teacher_eval):
                if any(route_changes(base, truth).values()):
                    raise ValueError("near student and teacher do not share a route cell")
        j = jacobian(weights, train, bases)
        norm2 = float(torch.linalg.eigvalsh(j @ j.T).max())
        bound = reference.BETA ** 2 * norm2 / (2 * sum(x.numel() for x in train))
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("invalid fixed affine Hessian bound")
        learning_rate = 1 / bound
        # The chosen fixed affine derivative is checked against autograd, not itself.
        params = clone(weights, grad=True)
        replay = reference.forward(params, train[0], mode="fixed_routes", base=bases[0])
        cotangent = torch.linspace(-0.2, 0.3, replay["logits"].numel(), dtype=torch.float64).reshape_as(replay["logits"])
        grads = torch.autograd.grad((replay["logits"] * cotangent).sum(), tuple(params.values()))
        manual = reference.manual_vjp(weights, train[0], cotangent, bases[0])
        error = max(float((a - manual[key]).abs().max()) for key, a in zip(params, grads))
        if error > 1e-10:
            raise ValueError("independent fixed-route VJP disagrees with autodiff")
        direction = {key: torch.sin(torch.arange(value.numel(), dtype=torch.float64)).reshape_as(value) / 100
                     for key, value in weights.items()}
        epsilon = 1e-4
        plus = {key: weights[key] + epsilon * direction[key] for key in weights}
        minus = {key: weights[key] - epsilon * direction[key] for key in weights}
        a = reference.forward(plus, train[0], mode="fixed_routes", base=bases[0])["logits"]
        b = reference.forward(minus, train[0], mode="fixed_routes", base=bases[0])["logits"]
        finite = float(((a - b) * cotangent).sum() / (2 * epsilon))
        predicted = float(sum((manual[key] * direction[key]).sum() for key in manual))
        if abs(finite - predicted) > 1e-7:
            raise ValueError("fixed affine directional derivative failed")
        affine_error = float(((a + b) / 2 - replay["logits"].detach()).abs().max())
        if affine_error > 1e-10:
            raise ValueError("fixed-route affine interpolation failed")
        for candidate in (plus, minus):
            dynamic = reference.forward(candidate, train[0])
            if any(route_changes(bases[0], dynamic).values()):
                raise ValueError("derivative perturbation left the certified route cell")
        shifted = clone(weights)
        shifted["EMBED"] += 0.125
        before_p = torch.softmax(bases[0]["logits"] * reference.BETA, dim=-1)
        after_p = torch.softmax(reference.forward(shifted, train[0])["logits"] * reference.BETA, dim=-1)
        common_shift_error = float((after_p - before_p).abs().max())
        if common_shift_error > 1e-12:
            raise ValueError("common-logit shift probability invariance failed")
        trajectory = []
        initial_hash = vector_hash(vector(weights))
        for step in range(TOY_STEPS + 1):
            params = clone(weights, grad=step < TOY_STEPS)
            mode = "fixed_routes" if name == "fixed_near" else "smooth"
            results = [reference.forward(params, x, mode=mode, base=base if mode == "fixed_routes" else None)
                       for x, base in zip(train, bases)]
            objective = sum(soft_metrics(row["logits"], y)[0] for row, y in zip(results, targets)) / len(train)
            kl = sum(float(soft_metrics(row["logits"], y)[1].detach()) for row, y in zip(results, targets)) / len(train)
            with torch.no_grad():
                heldout = [reference.forward(params, x, mode=mode, base=base if mode == "fixed_routes" else None)
                           for x, base in zip(evaluation, eval_bases)]
                heldout_kl = sum(float(soft_metrics(row["logits"], y)[1]) for row, y in zip(heldout, eval_targets)) / len(evaluation)
            trajectory.append({"step": step, "train_ce": float(objective.detach()), "train_kl": kl,
                "evaluation_kl": heldout_kl, "route_changes_from_initial": {
                    key: sum(route_changes(base, row)[key] for base, row in zip(bases, results)) for key in bases[0]["routes"]}})
            if step == 0 and (not math.isfinite(kl) or not math.isfinite(heldout_kl) or min(kl, heldout_kl) <= 0):
                raise ValueError("positive finite initial KL required for a relative-improvement diagnostic")
            if step < TOY_STEPS:
                grads = torch.autograd.grad(objective, tuple(params.values()))
                weights = {key: value.detach() - learning_rate * grad for (key, value), grad in zip(params.items(), grads)}
                if not torch.isfinite(vector(weights)).all():
                    raise ValueError("teacher learning diverged")
        monotone = all(b["train_ce"] <= a["train_ce"] + 1e-11 for a, b in zip(trajectory, trajectory[1:]))
        if name == "fixed_near" and not monotone:
            raise ValueError("fixed affine CE violates its conservative step bound")
        with torch.no_grad():
            dynamic_train = [reference.forward(weights, x) for x in train]
            dynamic_eval = [reference.forward(weights, x) for x in evaluation]
            final_dynamic_train_kl = sum(float(soft_metrics(row["logits"], y)[1]) for row, y in zip(dynamic_train, targets)) / len(train)
            final_dynamic_eval_kl = sum(float(soft_metrics(row["logits"], y)[1]) for row, y in zip(dynamic_eval, eval_targets)) / len(evaluation)
        final_dynamic_changes = {key: sum(route_changes(base, row)[key] for base, row in
            zip(bases + eval_bases, dynamic_train + dynamic_eval)) for key in bases[0]["routes"]}
        weight_file = name + "_final_weights.json"
        pilot.write_json(root / weight_file, {key: value.tolist() for key, value in weights.items()})
        result = {"cell": name, "updates": TOY_STEPS, "learning_rate": learning_rate,
            "initial_minimum_nontrivial_margin": initial_margin, "common_shift_probability_error": common_shift_error,
            "initial_jacobian_squared_norm": norm2, "affine_hessian_upper_bound": bound,
        "hessian_bound_scope": "global for frozen routes; initial-cell only for dynamic routes",
            "initial_weight_sha256": initial_hash, "final_weight_sha256": vector_hash(vector(weights)),
            "manual_autodiff_max_error": error, "directional_error": abs(finite - predicted), "affine_error": affine_error,
            "train_ce_monotone": monotone, "trajectory": trajectory,
            "final_weights_file": weight_file, "final_weights_file_sha256": digest((root / weight_file).read_bytes()),
            "final_dynamic_train_kl": final_dynamic_train_kl, "final_dynamic_evaluation_kl": final_dynamic_eval_kl,
            "final_dynamic_route_changes_from_initial": final_dynamic_changes,
            "final_dynamic_routes_equal_initial": not any(final_dynamic_changes.values()),
            "relative_train_kl": trajectory[-1]["train_kl"] / trajectory[0]["train_kl"],
            "relative_evaluation_kl": trajectory[-1]["evaluation_kl"] / trajectory[0]["evaluation_kl"],
            "local_kl_goal_met": trajectory[-1]["train_kl"] <= 0.5 * trajectory[0]["train_kl"]}
        pilot.write_json(root / (name + ".json"), result)
        cells.append(result)
    return {"teacher": teacher_meta, "cells": cells, "native_training": False,
            "new_fp64_updates": 192, "soft_target_training_position_exposures": 9216,
            "training_position_indices": 48, "synthetic_evaluation_position_indices": 48,
            "nearby_success_is_not_from_scratch_learning": True}


def prior_binding(root, record, corpus):
    root, record = Path(root).resolve(strict=True), Path(record).resolve(strict=True)
    manifest, receipt, summary = read(root / "manifest.json"), read(record), read(root / "summary.json")
    if manifest.get("schema") != historical.SCHEMA or summary.get("schema") != historical.SCHEMA:
        raise ValueError("wrong historical study schema")
    equal(manifest.get("source_hashes"), historical.source_identity(), "complete historical source identity")
    if receipt.get("complete") is not True or receipt.get("declared_cells") != 6 or receipt.get("completed_cells") != 6:
        raise ValueError("complete six-cell prior evidence required")
    if receipt.get("run_directory") != str(root) or receipt.get("source_hashes") != manifest.get("source_hashes"):
        raise ValueError("prior public/raw identity differs")
    for relative, expected in manifest["source_hashes"].items():
        if digest((REPO / relative).read_bytes()) != expected:
            raise ValueError("historical executable source changed")
    artifacts = read(root / "artifact_manifest.json")
    if artifacts.get("schema") != historical.SCHEMA or not isinstance(artifacts.get("files"), list):
        raise ValueError("invalid prior artifact inventory")
    if digest((root / "artifact_manifest.json").read_bytes()) != receipt["artifact_manifest_sha256"]:
        raise ValueError("prior artifact seal differs")
    actual_files = artifact_manifest(root)
    if artifacts["files"] != actual_files or any("failure" in row["file"] for row in actual_files):
        raise ValueError("prior complete artifact membership differs or includes failure")
    for item in artifacts["files"]:
        relative = Path(item["file"])
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or root not in path.resolve(strict=True).parents:
            raise ValueError("unsafe historical artifact path")
        data = path.read_bytes()
        if len(data) != item["bytes"] or digest(data) != item["sha256"]:
            raise ValueError("historical artifact changed")
    if summary.get("complete") is not True or summary.get("failures") != [] or summary.get("recipe") != historical.recipe():
        raise ValueError("prior execution is not complete and declared")
    if summary.get("gates") != historical.gate_results(summary["cells"]):
        raise ValueError("prior quality gates differ, whether pass or fail")
    expected_public = {**summary, "cells": [historical.confirmation._compact_cell(cell) for cell in summary["cells"]],
        "run_directory": str(root), "artifact_manifest_sha256": digest((root / "artifact_manifest.json").read_bytes()),
        "artifact_files_verified": len(actual_files), "artifact_payload_bytes": sum(row["bytes"] for row in actual_files),
        "full_optimizer_checkpoints_verified": 18, "checkpoint_validation_windows": 288,
        "distinct_validation_positions": 2048, "new_native_updates": 3072,
        "new_gradient_pair_exposures": 393216, "inherited_gradient_pair_exposures": 393216,
        "inherited_host_fitting_pair_exposures": 786432, "new_host_fitting_pair_exposures": 0,
        "initialization_refitted": False, "model_history_pair_exposures": 1572864,
        "distinct_training_pairs": 131072,
        "inherited_exposure_accounting": "sum of per-cell histories; shared old fitting/control updates were not reexecuted",
        "native_training_trajectory_independently_replayed": False}
    equal(receipt, expected_public, "exact prior public/raw receipt")
    raw, dataset = pilot.corpus_plan(Path(corpus), 1024, 88)
    if dataset != manifest["dataset"]:
        raise ValueError("prior corpus/split differs")
    states = {arm: root / (arm + "_seed_0/state_000000.i32") for arm in ARMS}
    shape = pilot.layout(512)
    decoded = {arm: native.decode_state(path.read_bytes(), shape) for arm, path in states.items()}
    if any(state["step"] != 0 or state["k"] != 18 or any(any(row) for row in state["remainders"].values())
           for state in decoded.values()):
        raise ValueError("independent counterfactuals require k18 initial zero-remainder states")
    for state in decoded.values():
        equal({key: state["config"].get(key) for key in ("loss_kind", "zshift")},
              {"loss_kind": 1, "zshift": 2}, "fixed initial loss configuration")
    changed, _ = initializer.transform(decoded["historical"]["weights"], shape)
    if changed != decoded["diverse"]["weights"]:
        raise ValueError("paired initialization is not the declared MP-only transform")
    paths = [record, root / "manifest.json", root / "summary.json", root / "artifact_manifest.json",
             Path(manifest["executable"]["path"]), Path(manifest["executable"]["path"]).parent / "build.json",
             Path(corpus), *states.values()]
    return {"inputs": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())} for path in paths],
        "prior_record": str(record), "prior_run": str(root), "states": {key: str(value) for key, value in states.items()},
        "executable": manifest["executable"], "dataset": dataset}, raw, decoded


def metrics(logits, targets):
    row = numeric.loss_head(logits, targets, "exact_ce", reduction="mean")
    return {"true_ce": float(row["true_ce"]), "lut_nll": float(row["lut_nll"]),
            "accuracy": float((row["predictions"] == torch.as_tensor(targets)).to(torch.float64).mean())}


def gradient_stats(gradients, exact):
    result = {}
    for name in gradients:
        a, b = gradients[name].reshape(-1), exact[name].reshape(-1)
        denominator = float(a.norm() * b.norm())
        result[name] = {"l1": float(a.abs().sum()), "l2": float(a.norm()), "max_abs": float(a.abs().max()),
            "exact_cosine": float(a @ b) / denominator if denominator else None,
            "max_difference": float((a - b).abs().max())}
    return result


def actual_diagnostic(root, binding, raw, decoded):
    shape, cells = pilot.layout(512), []
    for arm in ARMS:
        weights = shaped(decoded[arm]["weights"], shape)
        buffer = screen.initialize_buffer(shape, 0, "anchors")
        buffer[shape.OFF_CFG + 1] = 2
        endpoint = native.EFSGD(binding["executable"]["path"], shape, buffer, 18,
                               timeout_s=120, log_path=root / (arm + "_native.log"))
        try:
            for index in WINDOWS:
                start = time.monotonic()
                tokens, targets = pilot.window(raw, binding["dataset"], "train", index)
                token_tensor = torch.tensor(tokens, dtype=torch.int64)
                with torch.no_grad():
                    base = reference.forward(weights, token_tensor, mode="integer_anchor")
                    head = numeric.loss_head(base["logits"], targets)
                raw_head = head["native_gradient"].to(torch.float64)
                def legacy_center(g):
                    if not torch.equal(g, torch.trunc(g)):
                        raise ValueError("legacy VJP lost raw integer units")
                    return torch.from_numpy(numeric.legacy_center_vjp_raw(g.numpy().astype(np.int64))).to(torch.float64)
                native_gradient = reference.manual_vjp(weights, token_tensor, raw_head, base, center_vjp=legacy_center)
                expected = observer.observe(decoded[arm]["weights"], tokens, targets, shape,
                                            zshift=2, include_parameter_gradients=True)
                if expected["saturation_events"] != 0 or expected["route_diagnostics"]["backward_int32_overflow_detected"] is not False:
                    raise ValueError("historical observer arithmetic guard failed")
                for key, values in base["routes"].items():
                    if values.reshape(-1).tolist() != expected["native_routes"][key]:
                        raise ValueError("new reference selects different native routes")
                if raw_head.reshape(-1).tolist() != expected["gradients"]:
                    raise ValueError("new loss head differs from frozen observer")
                if head["predictions"].tolist() != expected["predictions"]:
                    raise ValueError("new-reference predictions differ from frozen observer")
                logits = initializer.calibration.integer_logits(decoded[arm]["weights"], tokens, shape)
                if logits["saturation_events"] != 0 or base["logits"].tolist() != logits["logits"]:
                    raise ValueError("independent raw integer logits differ")
                if any(native_gradient[key].reshape(-1).tolist() != expected["parameter_gradients"][key] for key in weights):
                    raise ValueError("new full VJP differs from frozen parameter-gradient observer")
                # The native one-step result checks all gradient words via weights+remainders.
                payload = Path(binding["states"][arm]).read_bytes()
                endpoint.import_state(payload)
                actual = endpoint.train(tokens, targets, routes=True)
                if actual["step"] != 1 or actual["gradients"] != expected["gradients"] or actual["predictions"] != expected["predictions"]:
                    raise ValueError("native one-step outputs disagree")
                if actual["native_routes"] != expected["native_routes"]:
                    raise ValueError("native one-step routes disagree")
                checkpoint = endpoint.save_checkpoint(root, "%s_%03d_after" % (arm, index))
                after = native.decode_state((root / checkpoint["file"]).read_bytes(), shape)
                for key in weights:
                    gradient = native_gradient[key].reshape(-1).numpy()
                    if not np.equal(gradient, np.trunc(gradient)).all():
                        raise ValueError("legacy gradient is not an exact integer")
                    oracle = numeric.ef_rounding(np.asarray(decoded[arm]["weights"][key]),
                        np.zeros(gradient.shape, dtype=np.int64), gradient.astype(np.int64), 18)
                    if oracle["weights"].tolist() != after["weights"][key] or oracle["remainders"].tolist() != after["remainders"][key]:
                        raise ValueError("native complete weight/remainder update differs from independent VJP")
                endpoint.import_state(payload)
                if endpoint.export_state() != payload:
                    raise ValueError("native reset between independent cases failed")
                initial_metrics = metrics(base["logits"], targets)
                directions, rows = {}, []
                for variant in VARIANTS:
                    if variant == "legacy":
                        directions[variant] = native_gradient
                    else:
                        info = numeric.loss_head(base["logits"], targets, variant)
                        cotangent = (raw_head if variant == "variable_mass" else
                                     info["supplied_gradient"] * (65536 / reference.BETA))
                        directions[variant] = reference.manual_vjp(weights, token_tensor, cotangent, base)
                    gradient = directions[variant]
                    candidate_rows = []
                    for scale in SCALES:
                        delta = {key: scale * value / (1 << 18) for key, value in gradient.items()}
                        ideal = {key: value - delta[key] for key, value in weights.items()}
                        rounded = {key: value - torch.trunc(delta[key]) for key, value in weights.items()}
                        with torch.no_grad():
                            affine = reference.forward(ideal, token_tensor, mode="fixed_routes", base=base)
                            rounded_affine = reference.forward(rounded, token_tensor, mode="fixed_routes", base=base)
                            fixed_int = reference.forward(rounded, token_tensor, mode="fixed_integer", base=base)
                            dynamic = reference.forward(rounded, token_tensor, mode="integer_anchor")
                        candidate_rows.append({"scale": scale, "ideal_affine": metrics(affine["logits"], targets),
                            "rounded_affine": metrics(rounded_affine["logits"], targets),
                            "fixed_integer": metrics(fixed_int["logits"], targets),
                            "dynamic_integer": metrics(dynamic["logits"], targets),
                            "dynamic_route_changes": route_changes(base, dynamic),
                            "changed_weight_words": int(sum((torch.trunc(value) != 0).sum() for value in delta.values())),
                            "integer_weight_sha256": vector_hash(vector(rounded)),
                            "legacy_scale1_matches_actual_native_weights": (
                                all(rounded[key].reshape(-1).tolist() == after["weights"][key] for key in weights)
                                if variant == "legacy" and scale == 1 else None)})
                        if variant == "legacy" and scale == 1 and candidate_rows[-1]["legacy_scale1_matches_actual_native_weights"] is not True:
                            raise ValueError("legacy counterfactual differs from actual native weights")
                    rows.append({"variant": variant, "gradients": gradient_stats(gradient, directions["exact_ce"]),
                                 "candidates": candidate_rows})
                result = {"arm": arm, "window_index": index, "partition": "train", "initial_metrics": initial_metrics,
                    "initial_logits_sha256": vector_hash(base["logits"]), "native_checkpoint": checkpoint,
                    "independent_raw_logits_equal": True, "frozen_full_parameter_gradient_equal": True,
                    "native_prediction_gradient_route_equal": True, "native_full_weight_remainder_equal": True,
                    "independent_native_updates": 1, "initial_step": 0, "final_step": 1,
                    "row_mass_min": int(head["rounded_totals"].min()), "row_mass_max": int(head["rounded_totals"].max()),
                    "numerical_ladder": rows, "elapsed_s": time.monotonic() - start}
                pilot.write_json(root / ("%s_%03d.json" % (arm, index)), result)
                cells.append(result)
        finally:
            endpoint.close()
    return {"cells": cells, "new_native_updates": 6, "new_native_gradient_pair_exposures": 768,
        "distinct_probe_pairs": 384, "new_model_fits": 0,
        "inherited_fit_pairs_per_reset": 131072, "summed_inherited_fit_pair_histories": 786432,
        "summed_native_model_history_pair_exposures": 787200, "distinct_model_history_pairs": 131072,
        "history_scope": "six independent resets share the same inherited fit; no new count fitting",
        "scope": "independent training-window update probes, not a six-update learning trajectory"}


def algebra_diagnostic(root):
    counts = np.asarray(recipe()["smoothing_counts"], dtype=np.int64)
    control = numeric.smoothing_stationarity(counts, alpha=1)
    rows = []
    for index, (empirical, target) in enumerate(zip(control["empirical_probabilities"], control["smoothed_probabilities"])):
        q = torch.tensor(target, dtype=torch.float64)
        logits = (q.log() / reference.BETA).requires_grad_()
        log_p = torch.log_softmax(logits * reference.BETA, dim=-1)
        matched = torch.autograd.grad(-(q * log_p).sum(), logits, retain_graph=True)[0]
        unsmoothed = torch.autograd.grad(-(torch.tensor(empirical) * log_p).sum(), logits)[0]
        expected = torch.tensor(control["unsmoothed_expected_gradient"][index])
        if float(matched.abs().max()) > 1e-15 or not torch.allclose(unsmoothed, expected, atol=1e-15, rtol=1e-12):
            raise ValueError("exact smoothed/unsmoothed stationarity control failed")
        variants = []
        for variant in numeric.LOSS_VARIANTS:
            repeated = logits.detach()[None, :].repeat(len(q), 1)
            head = numeric.loss_head(repeated, np.arange(len(q)), variant)
            gradient = (q[:, None] * head["supplied_gradient"]).sum(dim=0)
            steps = []
            for rate in recipe()["stationarity_probe_rates"]:
                after = logits.detach() - rate * gradient
                kl = float((q * (q.log() - torch.log_softmax(after * reference.BETA, dim=-1))).sum())
                steps.append({"learning_rate_raw_units": rate, "smoothed_target_kl_after": kl})
            variants.append({"variant": variant, "expected_gradient": gradient.tolist(),
                             "expected_gradient_l1": float(gradient.abs().sum()), "steps": steps})
        rows.append({"counts": counts[index].tolist(), "smoothed_target": q.tolist(),
            "exact_matched_gradient_max_abs": float(matched.abs().max()),
            "exact_unsmoothed_gradient": unsmoothed.tolist(), "variants": variants})
    duplicate = numeric.duplicate_bias_step(0.0, 0.0, 1.0, 0.125)
    if duplicate["sum_after"] != duplicate["double_rate_single_after"]:
        raise ValueError("duplicate-offset SGD identity failed")
    ef = []
    for gradient in (131073, -393219, 262147):
        result = numeric.ef_rounding(np.asarray([0]), np.asarray([0]), np.asarray([gradient]), 18)
        if int(result["integer_identity_residual"][0]) != 0:
            raise ValueError("EF arithmetic identity failed")
        ef.append({"gradient": gradient, "weight": int(result["weights"][0]),
                   "remainder": int(result["remainders"][0]), "identity_residual": 0})
    result = {"smoothing": rows, "duplicate_bias": duplicate, "ef_single_steps": ef,
        "scope": "analytic synthetic controls; no corpus fitting or training"}
    pilot.write_json(root / "algebra.json", result)
    return result


def artifact_manifest(root):
    root = Path(root)
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in new evidence")
        if path.is_file() and path != root / "artifact_manifest.json":
            data = path.read_bytes()
            result.append({"file": path.relative_to(root).as_posix(), "bytes": len(data), "sha256": digest(data)})
    return result


def verify_inputs(manifest):
    if manifest.get("schema") != SCHEMA or manifest["source_hashes"] != source_identity() or manifest["recipe"] != recipe():
        raise ValueError("diagnostic source or protocol changed")
    for row in manifest["prior"]["inputs"]:
        data = Path(row["path"]).read_bytes()
        if len(data) != row["bytes"] or digest(data) != row["sha256"]:
            raise ValueError("bound input changed")


def validate_summary(root, manifest, summary):
    root = Path(root)
    # Also rejects nonfinite numbers passed directly, not only parsed JSON.
    pilot._finite_json(json.dumps(summary, allow_nan=False))
    equal(summary.get("recipe"), recipe(), "exact summary recipe")
    if (summary.get("schema") != SCHEMA or summary.get("complete") is not True
            or summary.get("recipe") != recipe() or summary.get("source_hashes") != manifest["source_hashes"]
            or any(summary.get(key) is not False for key in
                   ("heldout_corpus_evaluated", "physical_fpga_execution", "paid_execution_authorized"))):
        raise ValueError("summary schema/source/protocol/scope differs")
    toy, actual = summary["toy"], summary["actual"]
    equal({key: toy.get(key) for key in ("new_fp64_updates", "soft_target_training_position_exposures", "training_position_indices", "synthetic_evaluation_position_indices")},
          {"new_fp64_updates": 192, "soft_target_training_position_exposures": 9216, "training_position_indices": 48, "synthetic_evaluation_position_indices": 48}, "teacher exposure accounting")
    if toy["teacher"] != read(root / "teacher.json") or summary["algebra"] != read(root / "algebra.json"):
        raise ValueError("actual synthetic control receipts differ")
    if [cell["cell"] for cell in toy["cells"]] != list(TOY_CELLS):
        raise ValueError("incomplete or reordered teacher cells")
    for cell in toy["cells"]:
        if cell != read(root / (cell["cell"] + ".json")) or type(cell["updates"]) is not int or cell["updates"] != TOY_STEPS:
            raise ValueError("actual teacher trajectory receipt differs")
        trace = cell["trajectory"]
        equal([row["step"] for row in trace], list(range(TOY_STEPS + 1)), "teacher update coverage")
        for key, tolerance in (("manual_autodiff_max_error", 1e-10), ("directional_error", 1e-7),
                               ("affine_error", 1e-10), ("common_shift_probability_error", 1e-12)):
            if type(cell[key]) not in (int, float) or not 0 <= cell[key] <= tolerance:
                raise ValueError("teacher affine derivative checks failed")
        if min(trace[0]["train_kl"], trace[0]["evaluation_kl"]) <= 0:
            raise ValueError("initial KL must be positive")
        equal(cell["relative_train_kl"], trace[-1]["train_kl"] / trace[0]["train_kl"], "relative training KL")
        equal(cell["relative_evaluation_kl"], trace[-1]["evaluation_kl"] / trace[0]["evaluation_kl"], "relative evaluation KL")
        equal(cell["local_kl_goal_met"], trace[-1]["train_kl"] <= 0.5 * trace[0]["train_kl"], "local KL goal")
        equal(cell["train_ce_monotone"], all(b["train_ce"] <= a["train_ce"] + 1e-11 for a, b in zip(trace, trace[1:])), "monotonicity")
        equal(cell["final_dynamic_routes_equal_initial"], not any(cell["final_dynamic_route_changes_from_initial"].values()), "final route claim")
        weight_path = root / cell["final_weights_file"]
        if weight_path.name != cell["cell"] + "_final_weights.json" or weight_path.parent != root:
            raise ValueError("unexpected final teacher weight path")
        if digest(weight_path.read_bytes()) != cell["final_weights_file_sha256"]:
            raise ValueError("final teacher weight bytes differ")
        saved = {key: torch.tensor(value, dtype=torch.float64) for key, value in read(weight_path).items()}
        if vector_hash(vector(saved)) != cell["final_weight_sha256"]:
            raise ValueError("final teacher weight identity differs")
        if cell["cell"] == "fixed_near" and cell["train_ce_monotone"] is not True:
            raise ValueError("fixed affine monotonicity failed")
    equal([[cell["arm"], cell["window_index"]] for cell in actual["cells"]], [[arm, index] for arm in ARMS for index in WINDOWS], "native case coverage")
    expected_exposures = {"new_native_updates": 6, "new_native_gradient_pair_exposures": 768,
        "distinct_probe_pairs": 384, "new_model_fits": 0, "inherited_fit_pairs_per_reset": 131072,
        "summed_inherited_fit_pair_histories": 786432, "summed_native_model_history_pair_exposures": 787200,
        "distinct_model_history_pairs": 131072}
    equal({key: actual.get(key) for key in expected_exposures}, expected_exposures, "native exposure accounting")
    for cell in actual["cells"]:
        if cell != read(root / ("%s_%03d.json" % (cell["arm"], cell["window_index"]))):
            raise ValueError("actual native case receipt differs")
        for key in ("independent_raw_logits_equal", "frozen_full_parameter_gradient_equal", "native_prediction_gradient_route_equal", "native_full_weight_remainder_equal"):
            if cell.get(key) is not True:
                raise ValueError("native parity gate did not pass")
        equal([cell["initial_step"], cell["final_step"], cell["independent_native_updates"]], [0, 1, 1], "native reset scope")
        checkpoint = cell["native_checkpoint"]
        path = root / checkpoint["file"]
        if path.name != "%s_%03d_after.i32" % (cell["arm"], cell["window_index"]) or path.parent != root:
            raise ValueError("unexpected checkpoint path")
        data = path.read_bytes()
        if len(data) != checkpoint["bytes"] or digest(data) != checkpoint["sha256"] or checkpoint != read(path.with_suffix(".json")):
            raise ValueError("actual native state receipt differs")
        shape = pilot.layout(512)
        state = native.decode_state(data, shape)
        equal([state["step"], state["k"], state["config"]["loss_kind"], state["config"]["zshift"]], [1, 18, 1, 2], "actual native final state")
        executable = Path(manifest["prior"]["executable"]["path"])
        identity = {"schema": native.SCHEMA, "executable_sha256": manifest["prior"]["executable"]["sha256"],
            "build_sha256": digest((executable.parent / "build.json").read_bytes()),
            "source_hashes": native.source_identity(), "layout": pilot.layout_values(shape),
            "cpu_only": True, "existing_afi_compatible": False}
        equal(checkpoint, {"schema": native.SCHEMA, "file": path.name, "bytes": len(data),
            "sha256": digest(data), "step": 1, "k": 18, "config": state["config"], "identity": identity,
            "regions": [{"name": name, "weight_words": getattr(shape, "WORDS_" + name),
                         "remainder_words": getattr(shape, "WORDS_" + name)} for name in native.WEIGHTS],
            "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}, "complete native checkpoint identity")
        if [row["variant"] for row in cell["numerical_ladder"]] != list(VARIANTS):
            raise ValueError("numerical variant coverage differs")
        for row in cell["numerical_ladder"]:
            if [candidate["scale"] for candidate in row["candidates"]] != list(SCALES):
                raise ValueError("counterfactual scale coverage differs")
        if cell["numerical_ladder"][-1]["candidates"][-1]["legacy_scale1_matches_actual_native_weights"] is not True:
            raise ValueError("native EF/counterfactual parity failed")


def check(root, record):
    root = Path(root).resolve(strict=True)
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    verify_inputs(manifest)
    if manifest["run_directory"] != str(root) or summary.get("complete") is not True:
        raise ValueError("incomplete or relocated run")
    if read(root / "protocol.json") != recipe() or digest((root / "declaration.md").read_bytes()) != manifest["plan_sha256"]:
        raise ValueError("actual pre-run declaration differs")
    validate_summary(root, manifest, summary)
    artifacts = artifact_manifest(root)
    if read(root / "artifact_manifest.json") != artifacts or any("failure" in row["file"] for row in artifacts):
        raise ValueError("raw artifact verification failed")
    expected = {**summary, "run_directory": str(root), "artifact_manifest_sha256": digest((root / "artifact_manifest.json").read_bytes()),
        "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts)}
    equal(read(record), expected, "public record versus sealed run")
    return expected


def run(config):
    output = capacity._new_external_path(config.output)
    target = capacity._new_external_path(config.record)
    if target == output or output in target.parents:
        raise ValueError("public record must be outside immutable raw run")
    sources = source_identity()
    binding, raw, decoded = prior_binding(config.prior_run, config.prior_record, config.corpus)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "prior": binding,
        "run_directory": str(output), "plan_sha256": digest(PLAN.read_bytes()),
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"}}
    pilot.write_json(output / "manifest.json", manifest)
    pilot.write_json(output / "protocol.json", recipe())
    with (output / "declaration.md").open("xb") as handle:
        handle.write(PLAN.read_bytes())
    def timeout_handler(signum, frame):
        raise TimeoutError("declared overall diagnostic deadline exceeded")
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
    try:
        torch.set_num_threads(1)
        verify_inputs(manifest)
        algebra = algebra_diagnostic(output)
        toy = toy_diagnostic(output)
        actual = actual_diagnostic(output, binding, raw, decoded)
        summary = {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": sources,
            "algebra": algebra, "toy": toy, "actual": actual, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "scientific_scope": "finite diagnostic, not an optimized recipe, generalization result, or hardware gate"}
        verify_inputs(manifest)
        validate_summary(output, manifest, summary)
        pilot.write_json(output / "summary.json", summary)
        artifacts = artifact_manifest(output)
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = {**summary, "run_directory": str(output),
            "artifact_manifest_sha256": digest((output / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts)}
        pilot.write_json(target, public)
        return check(output, target)
    except BaseException as error:
        if not (output / "failure.json").exists():
            pilot.write_json(output / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
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
        parser.error("run requires corpus and complete prior run/record")
    result = check(config.output, config.record) if config.check else run(config)
    print(json.dumps({"complete": result["complete"], "output": str(config.output), "check_only": config.check}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
