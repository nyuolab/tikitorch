"""Predeclared paired CPU update-scale screen; never a paid/scaling gate.

Six completed k18 histories are immutable controls. Six k20 histories replay
the identical fitting starts/data/orders. Validation is development, not test.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_learning_contract as ancestry  # noqa: E402
from manuscript.tools import train_fpga_lm_diverse_projection as old  # noqa: E402
from tikitorch.optim.fixed_point import fixed_point_ef_step  # noqa: E402

pilot, native, screen = old.pilot, old.optimizer, old.screen
read, equal = ancestry.read, ancestry.equal
SCHEMA = "tikitorch.fpga_lm_update_scale.v1"
ARMS, SEEDS, K_VALUES = ("historical", "diverse"), (0, 1, 2), (18, 20)
STEPS, WIDTH, CHECKPOINTS = 1024, 512, (0, 256, 1024)
VALIDATION, RESERVED = tuple(range(88, 152)), tuple(range(152, 864))
PARITY_STEPS = (1, 257)
DEADLINE_S = 7200
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    result = ancestry.source_identity()
    for name in ("manuscript/tools/test_fpga_lm_update_scale.py", "tests/test_fpga_lm_update_scale.py",
                 "tikitorch/optim/fixed_point.py", "tests/test_fixed_point_ef.py"):
        result[name] = pilot.digest((REPO / name).read_bytes())
    return result


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "seeds": list(SEEDS), "k_values": list(K_VALUES),
        "width": WIDTH, "steps": STEPS, "checkpoints": list(CHECKPOINTS),
        "validation_indices": list(VALIDATION), "reserved_validation_indices": list(RESERVED),
        "selection_checkpoint": STEPS, "primary_initialization": "diverse",
        "training_orders": {str(seed): old.training_order(seed) for seed in SEEDS},
        "seed_meaning": "order only; shared weights and corpus, not independent data samples",
        "intervention": "only k18 -> k20 at step0 with r0; fourfold smaller raw-gradient update",
        "prior_counterevidence": "historical count-fitted k20 was worse than k18 on sequential order/val40..55; not a novel untried rate",
        "control_policy": "import exact completed k18 states0/256/1024; no control retraining",
        "initial_evaluation": "two shared arm starts, k18; same function for k20 and all order seeds",
        "evaluation": "full independent integer observer plus native predictions/gradients/routes on every window; context-off host forward",
        "native_library_parity_steps": list(PARITY_STEPS),
        "thresholds": {"mean_paired_improvement_nats": 0.01, "minimum_paired_improvement_each_seed": 0.0,
            "context_nats_each_seed": 0.02, "context_growth_nats_each_seed": 0.02,
            "unigram_advantage_nats_each_seed": 0.05, "mean_bigram_advantage_nats": 0.02,
            "minimum_bigram_advantage_each_seed": 0.0, "maximum_zero_head_vjp_fraction": 0.25,
            "minimum_diverse_head_fraction": 0.75},
        "confirmation_policy": "only if primary screen passes, separately frozen inference-only same paired endpoints on152..863; no retraining/fallback selection",
        "new_native_updates": 6144, "new_gradient_pair_exposures": 786432,
        "inherited_control_gradient_pair_exposures": 786432,
        "summed_inherited_fit_pair_exposures": 1572864, "new_fit_pair_exposures": 0,
        "summed_model_history_pair_exposures": 3145728, "distinct_training_pairs": 131072,
        "accounting": "sum of12 endpoint histories; same old fit shared, six control histories not reexecuted; parity gradients duplicate already-charged update windows",
        "deadline_s": DEADLINE_S, "heldout_test_evaluated": False, "reserved_cohort_evaluated": False,
        "paid_execution_authorized": False, "physical_fpga_execution": False,
        "statistical_significance_claim": False, "adaptive_extension": False}


def identity(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": pilot.digest(data)}


def cell_name(arm, seed, k):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS or type(k) is not int or k not in K_VALUES:
        raise ValueError("undeclared paired cell")
    return f"{arm}_seed_{seed}_k{k}"


def state_config(payload, shape, step, k):
    state = native.decode_state(payload, shape)
    equal([state["step"], state["k"], state["config"]["loss_kind"], state["config"]["zshift"]],
          [step, k, 1, 2], "full state configuration")
    return state


def initial_buffer(payload, shape, k):
    if type(k) is not int or k not in K_VALUES:
        raise ValueError("undeclared update shift")
    state = state_config(payload, shape, 0, 18)
    if any(any(row) for row in state["remainders"].values()):
        raise ValueError("initial state requires zero remainders")
    buffer = shape.build_init(0)
    for name, values in state["weights"].items():
        offset = getattr(shape, "OFF_" + name)
        buffer[offset:offset + len(values)] = values
    buffer[shape.OFF_CFG:shape.OFF_CFG + 2] = [1, 2]
    buffer[shape.OFF_CE_LUT:shape.OFF_CE_LUT + shape.WORDS_CE_LUT] = state["config"]["lut"]
    return buffer


def bind_inputs(config):
    binding, raw, _ = ancestry.prior_binding(config.prior_run, config.prior_record, config.corpus)
    paths = {row["path"] for row in binding["inputs"]}
    shape = pilot.layout(WIDTH)
    states = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = cell_name(arm, seed, 18)
            states[name] = {}
            for step in CHECKPOINTS:
                path = Path(binding["prior_run"]) / f"{arm}_seed_{seed}" / f"state_{step:06d}.i32"
                payload = path.read_bytes()
                state_config(payload, shape, step, 18)
                if step == 0:
                    equal(pilot.digest(payload), pilot.digest(Path(binding["states"][arm]).read_bytes()), "all order seeds share exact fitted start")
                states[name][str(step)] = str(path)
                paths.add(str(path))
    raw2, dataset = pilot.corpus_plan(config.corpus, STEPS, max(VALIDATION) + 1)
    equal(raw, raw2, "same corpus bytes")
    # Only the requested validation budget changes; immutable split/source do not.
    for key in ("sha256", "bytes", "encoding", "splits", "window_stride", "test_policy"):
        equal(dataset[key], binding["dataset"][key], "unchanged corpus/split")
    manifest = read(Path(binding["prior_run"]) / "manifest.json")
    old._native_identity(manifest, shape)
    return {"ancestry": binding, "inputs": [identity(path) for path in sorted(paths)],
            "states": states, "dataset": dataset, "executable": binding["executable"]}, raw


def frozen(manifest):
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "fixed recipe")
    equal(manifest["source_hashes"], source_identity(), "frozen sources and tests")
    for row in manifest["binding"]["inputs"]:
        equal(identity(row["path"]), row, "unchanged actual input bytes")
    binding = manifest["binding"]
    rebuilt, _ = bind_inputs(SimpleNamespace(prior_run=binding["ancestry"]["prior_run"],
        prior_record=binding["ancestry"]["prior_record"], corpus=binding["dataset"]["path"]))
    equal(binding, rebuilt, "canonical complete source-state ancestry and input membership")


def save_arrays(path, arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **{name: np.asarray(value, dtype="<i4") for name, value in arrays.items()})


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        if any(archive[name].dtype != np.dtype("<i4") or archive[name].ndim != 1 for name in archive.files):
            raise ValueError("saved arrays must be flat signed int32")
        return {name: archive[name].tolist() for name in archive.files}


def evaluate(endpoint, root, raw, dataset, shape, step, k):
    payload = endpoint.export_state()
    state = state_config(payload, shape, step, k)
    rows = []
    for index in VALIDATION:
        tokens, targets = pilot.window(raw, dataset, "validation", index)
        found = endpoint.evaluate(tokens, targets)
        equal(found["step"], step, "evaluation update count")
        expected = old._checked_observer(state["weights"], tokens, targets, shape)
        arrays = {"predictions": found["predictions"], "gradients": found["gradients"], **found["native_routes"]}
        equal(arrays, {"predictions": expected["predictions"], "gradients": expected["gradients"], **expected["native_routes"]}, "native/reference full evaluation arrays")
        save_arrays(root / f"eval_{index:03d}.npz", arrays)
        row = {**screen._scalar_result(expected), "window_index": index, "native_reference_equal": True,
            "native_routes_equal": True, "probes": {"context_disabled": old._probe(state["weights"], tokens, targets, shape)}}
        pilot.write_json(root / f"eval_{index:03d}.json", row)
        rows.append(row)
    equal(endpoint.export_state(), payload, "evaluation full-state invariance")
    endpoint.import_state(payload)
    equal(endpoint.export_state(), payload, "checkpoint import full-state invariance")
    return {"step": step, "state_sha256": pilot.digest(payload), "full_state_evaluation_invariant": True,
        "full_state_import_equal": True, **screen._aggregate(rows)}


def verify_evaluation(root, payload, raw, dataset, shape, step, k, recompute):
    state = state_config(payload, shape, step, k)
    rows = []
    for index in VALIDATION:
        row = read(root / f"eval_{index:03d}.json")
        equal(row["window_index"], index, "exact development window coverage")
        if (row["native_reference_equal"] is not True or row["native_routes_equal"] is not True
                or type(row["saturation_events"]) is not int or row["saturation_events"] != 0
                or row["route_diagnostics"]["backward_int32_overflow_detected"] is not False):
            raise ValueError("every initial/control/candidate window requires complete numeric parity")
        arrays = load_arrays(root / f"eval_{index:03d}.npz")
        sizes = {"predictions": 128, "gradients": 16384,
            **{name: getattr(shape, "WORDS_" + name) for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}
        equal({name: len(value) for name, value in arrays.items()}, sizes, "full native evaluation array coverage")
        if recompute:
            tokens, targets = pilot.window(raw, dataset, "validation", index)
            expected = old._checked_observer(state["weights"], tokens, targets, shape)
            equal(arrays, {"predictions": expected["predictions"], "gradients": expected["gradients"], **expected["native_routes"]}, "independently recomputed native arrays")
            equal(row, {**screen._scalar_result(expected), "window_index": index, "native_reference_equal": True,
                "native_routes_equal": True, "probes": {"context_disabled": old._probe(state["weights"], tokens, targets, shape)}}, "independent checkpoint metrics")
        rows.append(row)
    return {"step": step, "state_sha256": pilot.digest(payload), "full_state_evaluation_invariant": True,
        "full_state_import_equal": True, **screen._aggregate(rows)}


def baselines(raw, dataset):
    output = {}
    for passes in (1, 2):
        counts = np.ones((128, 128), dtype=np.int64)
        uni = np.ones(128, dtype=np.int64)
        for index in range(STEPS):
            tokens, targets = pilot.window(raw, dataset, "train", index)
            np.add.at(counts, (tokens, targets), passes)
            np.add.at(uni, targets, passes)
        result = {"alpha": 1, "exposures": passes * STEPS * 128, "distinct_pairs": STEPS * 128}
        for name, values in (("unigram", np.broadcast_to(uni, (128, 128))), ("bigram", counts)):
            rows = []
            for index in VALIDATION:
                tokens, targets = pilot.window(raw, dataset, "validation", index)
                losses = np.log(values.sum(axis=1)[tokens]) - np.log(values[tokens, targets])
                rows.append({"window_index": index, "nll": float(losses.mean())})
            result[name] = {"nll": sum(row["nll"] for row in rows) / len(rows), "per_window": rows}
        output[str(passes)] = result
    output["stronger"] = {name: {"nll": min(output[str(p)][name]["nll"] for p in (1, 2))} for name in ("unigram", "bigram")}
    return output


def gate_results(cells, initial, controls):
    pilot._finite_json(json.dumps([cells, initial, controls], allow_nan=False))
    losses = [value[key] for value in [*initial.values(), *[row["evaluation"] for row in cells]]
              for key in ("lut_normalized_nll", "context_disabled_nll")]
    losses.extend(controls["stronger"][name]["nll"] for name in ("unigram", "bigram"))
    if any(type(value) not in (int, float) or value < 0 for value in losses):
        raise ValueError("losses require finite nonnegative numbers, never booleans")
    for row in cells:
        cell_name(row["arm"], row["seed"], row["k"])
    equal([(row["arm"], row["seed"], row["k"]) for row in cells],
          [(arm, seed, k) for arm in ARMS for seed in SEEDS for k in K_VALUES], "all12 ordered cells")
    thresholds = recipe()["thresholds"]
    result = {}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            pair = {row["k"]: row for row in cells if row["arm"] == arm and row["seed"] == seed}
            if any(row.get("complete") is not True for row in pair.values()):
                raise ValueError("incomplete cells cannot pass screening")
            final = pair[20]["evaluation"]
            routes = final["route_diagnostics"]
            positions, nonzero, zero, diverse = (routes[key] for key in ("positions", "nonzero_logit_gradient_positions",
                "exact_zero_head_input_vjp_on_nonzero_gradient_positions", "head_winners_ge16_positions"))
            if (any(type(v) is not int for v in (positions, nonzero, zero, diverse)) or positions != len(VALIDATION) * 128
                    or not 0 <= zero <= nonzero <= positions or not 0 <= diverse <= positions):
                raise ValueError("invalid complete-cohort mechanism counts")
            mechanism = (nonzero > 0 and zero / nonzero <= thresholds["maximum_zero_head_vjp_fraction"]
                and diverse / positions >= thresholds["minimum_diverse_head_fraction"]
                and type(final["saturation_events"]) is int and final["saturation_events"] == 0
                and final["native_reference_equal"] is True and final["native_routes_equal"] is True
                and routes["backward_int32_overflow_detected"] is False)
            context = final["context_disabled_nll"] - final["lut_normalized_nll"]
            growth = context - (initial[arm]["context_disabled_nll"] - initial[arm]["lut_normalized_nll"])
            unigram = controls["stronger"]["unigram"]["nll"] - final["lut_normalized_nll"]
            rows.append({"seed": seed, "k18_nll": pair[18]["evaluation"]["lut_normalized_nll"],
                "k20_nll": final["lut_normalized_nll"],
                "paired_improvement_nats": pair[18]["evaluation"]["lut_normalized_nll"] - final["lut_normalized_nll"],
                "context_nats": context, "context_growth_nats": growth, "unigram_advantage_nats": unigram,
                "bigram_advantage_nats": controls["stronger"]["bigram"]["nll"] - final["lut_normalized_nll"],
                "mechanism": mechanism, "context": context >= 0.02 - 1e-12 and growth >= 0.02 - 1e-12 and unigram >= 0.05 - 1e-12})
        paired = sum(row["paired_improvement_nats"] for row in rows) / len(rows)
        bigram = sum(row["bigram_advantage_nats"] for row in rows) / len(rows)
        passes = (paired >= 0.01 - 1e-12 and bigram >= 0.02 - 1e-12
            and all(row["paired_improvement_nats"] >= -1e-12 and row["bigram_advantage_nats"] >= -1e-12
                    and row["mechanism"] and row["context"] for row in rows))
        result[arm] = {"per_seed": rows, "mean_paired_improvement_nats": paired,
            "mean_bigram_advantage_nats": bigram, "all_screen_criteria": passes}
    return {"by_initialization": result, "primary_screen_pass": result["diverse"]["all_screen_criteria"],
        "historical_is_fallback_candidate": False, "confirmation_required_before_recipe_promotion": True,
        "recipe_promoted": False, "paid_execution_authorized": False}


def parity_step(endpoint, root, step, tokens, targets, shape):
    before = endpoint.export_state()
    state = state_config(before, shape, step - 1, 20)
    observed = old.observe(state["weights"], tokens, targets, shape, zshift=2, include_parameter_gradients=True)
    if observed["saturation_events"] or observed["route_diagnostics"]["backward_int32_overflow_detected"]:
        raise ValueError("parity window overflows or clips")
    gradients = observed["parameter_gradients"]
    expected = {}
    for name in pilot.WEIGHTS:
        expected[name] = fixed_point_ef_step(torch.tensor(state["weights"][name], dtype=torch.int16),
            torch.tensor(state["remainders"][name], dtype=torch.int32),
            torch.tensor(gradients[name], dtype=torch.int32), k=20)
    result = endpoint.train(tokens, targets, routes=True)
    outputs = {"predictions": result["predictions"], "gradients": result["gradients"], **result["native_routes"]}
    equal(outputs, {"predictions": observed["predictions"], "gradients": observed["gradients"], **observed["native_routes"]}, "native/reference parity-step full output arrays")
    after = endpoint.export_state()
    updated = state_config(after, shape, step, 20)
    for name in pilot.WEIGHTS:
        equal(updated["weights"][name], expected[name][0].tolist(), "library/native full weight update parity")
        equal(updated["remainders"][name], expected[name][1].tolist(), "library/native full remainder parity")
    with (root / f"parity_{step:04d}_before.i32").open("xb") as handle:
        handle.write(before)
    with (root / f"parity_{step:04d}_after.i32").open("xb") as handle:
        handle.write(after)
    save_arrays(root / f"parity_{step:04d}_gradients.npz", gradients)
    save_arrays(root / f"parity_{step:04d}_outputs.npz", outputs)
    return result


def verify_training(root, order):
    lines = (root / "training.jsonl").read_text().splitlines()
    if len(lines) != STEPS:
        raise ValueError("incomplete training history")
    records = [pilot._finite_json(line) for line in lines]
    for name, words in (("predictions", 128), ("gradients", 16384)):
        payload = (root / f"training_{name}.i32").read_bytes()
        if len(payload) != STEPS * words * 4:
            raise ValueError("incomplete training arrays")
        for i, row in enumerate(records):
            equal([row["step"], row["window_index"]], [i + 1, order[i]], "declared training order")
            equal(row[name + "_sha256"], pilot.digest(payload[i * words * 4:(i + 1) * words * 4]), "actual logged training arrays")


def execute(manifest, output, raw):
    binding, shape = manifest["binding"], pilot.layout(WIDTH)
    dataset, exe = binding["dataset"], Path(binding["executable"]["path"])
    initial, cells = {}, []
    for arm in ARMS:
        directory = output / ("initial_" + arm)
        directory.mkdir()
        payload = Path(binding["states"][cell_name(arm, 0, 18)]["0"]).read_bytes()
        buffer = initial_buffer(payload, shape, 18)
        with native.EFSGD(exe, shape, buffer, 18, timeout_s=60, log_path=directory / "native.log") as endpoint:
            equal(endpoint.export_state(), payload, "exact fitted native start")
            initial[arm] = evaluate(endpoint, directory, raw, dataset, shape, 0, 18)
        print(f"initial {arm} evaluation complete", flush=True)
        for seed in SEEDS:
            for k in K_VALUES:
                name = cell_name(arm, seed, k)
                directory = output / name
                directory.mkdir()
                with native.EFSGD(exe, shape, buffer, k, timeout_s=60, log_path=directory / "native.log") as endpoint:
                    start = native.decode_state(endpoint.export_state(), shape)
                    expected = native.decode_state(payload, shape)
                    expected["k"] = k
                    equal(start, expected, "initial full state changes only k")
                    if k == 18:
                        for step in CHECKPOINTS:
                            old_payload = Path(binding["states"][name][str(step)]).read_bytes()
                            endpoint.import_state(old_payload)
                            equal(endpoint.export_state(), old_payload, "unchanged imported control state")
                            endpoint.save_checkpoint(directory, f"state_{step:06d}")
                    else:
                        endpoint.save_checkpoint(directory, "state_000000")
                        with (directory / "training.jsonl").open("x") as log, (directory / "training_predictions.i32").open("xb") as preds, (directory / "training_gradients.i32").open("xb") as grads:
                            for step, index in enumerate(old.training_order(seed), 1):
                                tokens, targets = pilot.window(raw, dataset, "train", index)
                                found = (parity_step(endpoint, directory, step, tokens, targets, shape) if step in PARITY_STEPS else endpoint.train(tokens, targets))
                                equal(found["step"], step, "native actual update count")
                                pred, grad = screen._i32(found["predictions"]), screen._i32(found["gradients"])
                                preds.write(pred)
                                grads.write(grad)
                                log.write(json.dumps({"step": step, "window_index": index,
                                    "predictions_sha256": pilot.digest(pred), "gradients_sha256": pilot.digest(grad)}) + "\n")
                                log.flush()
                                if step in CHECKPOINTS:
                                    endpoint.save_checkpoint(directory, f"state_{step:06d}")
                                    print(f"{name} trained {step}/{STEPS}", flush=True)
                    evaluation = evaluate(endpoint, directory, raw, dataset, shape, STEPS, k)
                cell = {"arm": arm, "seed": seed, "k": k, "complete": True,
                    "new_updates": STEPS if k == 20 else 0, "inherited_updates": STEPS if k == 18 else 0,
                    "evaluation": evaluation}
                pilot.write_json(directory / "summary.json", cell)
                cells.append(cell)
                print(f"{name} final NLL {evaluation['lut_normalized_nll']:.9f}", flush=True)
    controls = baselines(raw, dataset)
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
        "initial": initial, "cells": cells, "baselines": controls, "gates": gate_results(cells, initial, controls),
        "reserved_cohort_evaluated": False, "heldout_test_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False}


def validate_summary(root, manifest, summary, recompute=False):
    frozen(manifest)
    binding, shape = manifest["binding"], pilot.layout(WIDTH)
    raw, dataset = pilot.corpus_plan(binding["dataset"]["path"], STEPS, max(VALIDATION) + 1)
    equal(dataset, binding["dataset"], "actual dataset")
    native_identity = old._native_identity(read(Path(binding["ancestry"]["prior_run"]) / "manifest.json"), shape)
    initial, cells = {}, []
    for arm in ARMS:
        payload = Path(binding["states"][cell_name(arm, 0, 18)]["0"]).read_bytes()
        initial[arm] = verify_evaluation(root / ("initial_" + arm), payload, raw, dataset, shape, 0, 18, recompute)
        for seed in SEEDS:
            for k in K_VALUES:
                name = cell_name(arm, seed, k)
                directory = root / name
                for step in CHECKPOINTS:
                    checkpoint = directory / f"state_{step:06d}.i32"
                    saved = checkpoint.read_bytes()
                    state = state_config(saved, shape, step, k)
                    metadata = read(checkpoint.with_suffix(".json"))
                    equal(metadata, {"schema": native.SCHEMA, "file": checkpoint.name, "bytes": len(saved),
                        "sha256": pilot.digest(saved), "step": step, "k": k, "config": state["config"], "identity": native_identity,
                        "regions": [{"name": region, "weight_words": getattr(shape, "WORDS_" + region),
                            "remainder_words": getattr(shape, "WORDS_" + region)} for region in pilot.WEIGHTS],
                        "format": "v1 little-endian int32 header+source digest+LUT+full weights+full remainders"}, "complete checkpoint bytes/metadata")
                    if k == 18:
                        equal(saved, Path(binding["states"][name][str(step)]).read_bytes(), "historical control full-state bytes")
                    if step == 0:
                        expected = native.decode_state(payload, shape)
                        expected["k"] = k
                        equal(state, expected, "only k changed at fitted start")
                if k == 20:
                    verify_training(directory, old.training_order(seed))
                    for step in PARITY_STEPS:
                        before = state_config((directory / f"parity_{step:04d}_before.i32").read_bytes(), shape, step - 1, k)
                        after = state_config((directory / f"parity_{step:04d}_after.i32").read_bytes(), shape, step, k)
                        equal((directory / f"parity_{step:04d}_before.i32").read_bytes(),
                            (directory / f"state_{step-1:06d}.i32").read_bytes(), "parity anchor is actual checkpoint")
                        grads = load_arrays(directory / f"parity_{step:04d}_gradients.npz")
                        equal(list(sorted(grads)), list(sorted(pilot.WEIGHTS)), "full parity gradient regions")
                        outputs = load_arrays(directory / f"parity_{step:04d}_outputs.npz")
                        equal({name: len(value) for name, value in outputs.items()}, {"predictions": 128, "gradients": 16384,
                            **{name: getattr(shape, "WORDS_" + name) for name in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}}, "complete parity output coverage")
                        for region, words in (("predictions", 128), ("gradients", 16384)):
                            with (directory / f"training_{region}.i32").open("rb") as handle:
                                handle.seek((step - 1) * words * 4)
                                equal(screen._i32(outputs[region]), handle.read(words * 4), "parity outputs belong to actual training log")
                        if recompute:
                            tokens, targets = pilot.window(raw, dataset, "train", old.training_order(seed)[step - 1])
                            observed = old.observe(before["weights"], tokens, targets, shape, zshift=2, include_parameter_gradients=True)
                            equal(grads, observed["parameter_gradients"], "independent full parity-anchor parameter gradients")
                            equal(outputs, {"predictions": observed["predictions"], "gradients": observed["gradients"], **observed["native_routes"]}, "independent parity-anchor output arrays")
                        for region in pilot.WEIGHTS:
                            w, r = fixed_point_ef_step(torch.tensor(before["weights"][region], dtype=torch.int16),
                                torch.tensor(before["remainders"][region], dtype=torch.int32), torch.tensor(grads[region], dtype=torch.int32), k=k)
                            equal([w.tolist(), r.tolist()], [after["weights"][region], after["remainders"][region]], "saved full-gradient library/native arithmetic replay")
                evaluation = verify_evaluation(directory, (directory / f"state_{STEPS:06d}.i32").read_bytes(), raw, dataset, shape, STEPS, k, recompute)
                cell = {"arm": arm, "seed": seed, "k": k, "complete": True, "new_updates": STEPS if k == 20 else 0,
                    "inherited_updates": STEPS if k == 18 else 0, "evaluation": evaluation}
                equal(read(directory / "summary.json"), cell, "complete paired cell")
                cells.append(cell)
    controls = baselines(raw, dataset)
    expected = {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
        "initial": initial, "cells": cells, "baselines": controls, "gates": gate_results(cells, initial, controls),
        "reserved_cohort_evaluated": False, "heldout_test_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False}
    equal(summary, expected, "full reconstructed summary and gates")
    return expected


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "artifact_manifest_sha256": pilot.digest((root / "artifact_manifest.json").read_bytes()),
        "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
        "check_scope": "integrity, complete state/array coverage, aggregation, baselines and12 saved library/native arithmetic steps; optional --recompute-evaluation independently repeats all896 checkpoint windows; not a full training replay"}


def check(root, record, recompute=False):
    root = Path(root).resolve(strict=True)
    manifest, summary = read(root / "manifest.json"), read(root / "summary.json")
    equal(manifest["run_directory"], str(root), "owned run path")
    equal(read(root / "protocol.json"), recipe(), "actual pre-run protocol")
    equal(pilot.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "actual pre-run declaration")
    artifacts = ancestry.artifact_manifest(root)
    equal(read(root / "artifact_manifest.json"), artifacts, "complete sealed raw artifact tree")
    if any("failure" in row["file"] for row in artifacts):
        raise ValueError("failed execution cannot be published")
    validate_summary(root, manifest, summary, recompute)
    expected = public_record(root, summary, artifacts)
    equal(read(record), expected, "exact public versus raw receipt")
    frozen(manifest)
    equal(ancestry.artifact_manifest(root), artifacts, "all artifacts unchanged through verification")
    equal(read(root / "artifact_manifest.json"), artifacts, "seal unchanged through verification")
    equal(read(record), expected, "public receipt unchanged through verification")
    return expected


def run(config):
    output, target = (old.prior.capacity._new_external_path(path) for path in (config.output, config.record))
    if target == output or output in target.parents or target in output.parents:
        raise ValueError("public record must be outside raw output")
    if not target.parent.is_dir():
        raise FileNotFoundError("public record parent directory must already exist")
    sources = source_identity()
    binding, raw = bind_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    previous, installed, published = None, False, None
    staging_directory, staged, owned_public = None, None, False
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed update-scale deadline exceeded; no adaptive retry")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        manifest = {"schema": SCHEMA, "run_directory": str(output), "recipe": recipe(), "binding": binding,
            "source_hashes": sources, "plan_sha256": pilot.digest(PLAN.read_bytes()),
            "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"}}
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", recipe())
        with (output / "declaration.md").open("xb") as handle:
            handle.write(PLAN.read_bytes())
        frozen(manifest)
        summary = execute(manifest, output, raw)
        validate_summary(output, manifest, summary)
        pilot.write_json(output / "summary.json", summary)
        artifacts = ancestry.artifact_manifest(output)
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = public_record(output, summary, artifacts)
        published = pilot.digest((json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-update-scale-", dir=target.parent))
        staged = staging_directory / "receipt.json"
        pilot.write_json(staged, public)
        # Atomic, exclusive publication on the same filesystem. Even an
        # interrupt immediately after link can identify our exact owned inode.
        os.link(staged, target)
        owned_public = True
        result = check(output, target)
        staged.unlink()
        staging_directory.rmdir()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        installed = False
        return result
    except BaseException as error:
        same_inode = staged is not None and staged.is_file() and target.is_file() and os.path.samefile(staged, target)
        if (owned_public or same_inode) and published is not None and target.is_file() and pilot.digest(target.read_bytes()) == published:
            rejected = output / "rejected_public.json"
            if rejected.exists():
                raise RuntimeError("cannot preserve rejected receipt without overwrite") from error
            target.rename(rejected)
        if not (output / "failure.json").exists():
            pilot.write_json(output / "failure.json", {"complete": False, "type": type(error).__name__, "error": str(error)})
        raise
    finally:
        try:
            if staged is not None and staged.exists():
                staged.unlink()
            if staging_directory is not None and staging_directory.exists():
                staging_directory.rmdir()
        finally:
            if installed:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--recompute-evaluation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    for name in ("prior-run", "prior-record", "corpus"):
        parser.add_argument("--" + name, type=Path)
    config = parser.parse_args(argv)
    if config.recompute_evaluation and not config.check:
        parser.error("--recompute-evaluation requires --check")
    if not config.check and any(getattr(config, name) is None for name in ("prior_run", "prior_record", "corpus")):
        parser.error("a run needs --prior-run, --prior-record and --corpus")
    result = check(config.output, config.record, config.recompute_evaluation) if config.check else run(config)
    print(json.dumps({key: result[key] for key in ("schema", "complete", "gates", "artifact_files_verified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
