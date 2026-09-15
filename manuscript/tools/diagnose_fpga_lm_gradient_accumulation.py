"""Frozen CPU timing diagnostic: sequential versus summed four-window EF.

No native/cloud execution, held-out evaluation, or quality-promotion gate.
The optional replay recomputes saved queries, not a new optimizer trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
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

from manuscript.tools import diagnose_fpga_lm_persistent_trajectory as old  # noqa: E402
from manuscript.tools import fpga_lm_accumulation_arithmetic as accumulation  # noqa: E402

prior, extension, arithmetic = old.prior, old.extension, old.arithmetic
pilot, read, equal = old.pilot, old.read, old.equal
WEIGHTS = old.WEIGHTS
SCHEMA = "tikitorch.fpga_lm_gradient_accumulation.v1"
ARMS = ("sequential", "batch4")
WINDOWS, PROBES, PROBE_EXPOSURES = tuple(range(32)), (0, 37, 511), (0, 8, 32)
K, BLOCK, DEADLINE_S = 18, 4, 1800
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    sources = {**old.source_identity(), **accumulation.source_identity()}
    for name in ("manuscript/tools/diagnose_fpga_lm_gradient_accumulation.py",
                 "manuscript/tools/fpga_lm_accumulation_arithmetic.py",
                 "tests/test_fpga_lm_gradient_accumulation.py",
                 "tests/test_fpga_lm_gradient_accumulation_replay.py",
                 "tests/test_fpga_lm_accumulation_arithmetic.py",
                 "tikitorch/optim/fixed_point.py", "tests/test_fixed_point_ef.py"):
        sources[name] = old.digest((REPO / name).read_bytes())
    return sources


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "initialization": "diverse seed0 fitted step0",
        "windows": list(WINDOWS), "partition": "train", "block_size": BLOCK, "k": K,
        "gradient_rule": "legacy raw LUT, zshift2, dynamic integer floor forward/backward",
        "normalization": "sum four gradients at unchanged block weights; divide by 2**18, not 4*2**18",
        "intervention": "gradient-query timing AND signed-quantization release timing",
        "optimizer_commits": {"sequential": 32, "batch4": 8},
        "gradient_queries": {"sequential": 32, "batch4": 32},
        "boundary_exposures": list(range(0, 33, 4)), "probe_windows": list(PROBES),
        "probe_exposures": list(PROBE_EXPOSURES), "probe_context": [True, False],
        "endpoint_scoring": "all32 training windows at initial and final weights; report every window",
        "fixed_gradient_counterfactual": "same batch-start w/r and four saved G, sequential vs summed; no new gradient query",
        "control": "host sequential rerun checked against every old native gradient and w/r hash",
        "torch_threads": 1, "deadline_s": DEADLINE_S, "new_native_updates": 0,
        "new_gradient_pair_exposures": 8192, "distinct_new_gradient_pairs": 4096,
        "summed_inherited_fit_pair_exposures": 262144, "new_fit_pair_exposures": 0,
        "summed_endpoint_history_pair_exposures": 270336, "distinct_fitted_pairs": 131072,
        "replay_is_training": False, "heldout_corpus_evaluated": False,
        "physical_fpga_execution": False, "paid_execution_authorized": False,
        "recipe_promoted": False, "statistical_significance_claim": False, "adaptive_extension": False}


def bind_inputs(config):
    root, record = Path(config.prior_run).resolve(strict=True), Path(config.prior_record).resolve(strict=True)
    old.check(root, record)
    previous = read(root / "manifest.json")
    initial = previous["binding"]["initial"]
    raw, dataset = pilot.corpus_plan(Path(config.corpus), 1024, 88)
    equal(dataset, initial["dataset"], "same existing corpus/split")
    paths = [record, root / "manifest.json", root / "summary.json", root / "artifact_manifest.json",
             Path(initial["states"]["diverse"]), Path(config.corpus)]
    paths += sorted(path for path in (root / "diverse").iterdir()
                    if path.name.startswith(("step_", "gradients_", "native_")) and path.is_file())
    return {"prior_run": str(root), "prior_record": str(record), "dataset": dataset,
            "initial_state": initial["states"]["diverse"],
            "inputs": [old.file_identity(path) for path in paths]}, raw


def frozen(manifest):
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "fixed recipe")
    equal(manifest["source_hashes"], source_identity(), "frozen source/tests")
    binding = manifest["binding"]
    for item in binding["inputs"]:
        equal(old.file_identity(item["path"]), item, "bound input unchanged")
    rebuilt, _ = bind_inputs(SimpleNamespace(prior_run=binding["prior_run"], prior_record=binding["prior_record"],
                                             corpus=binding["dataset"]["path"]))
    equal(rebuilt, binding, "canonical complete predecessor binding")


def save(path, values):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **{key: value.detach().numpy() for key, value in values.items()})


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate array name")
        result = {}
        for key in archive.files:
            values = archive[key]
            if values.dtype not in (np.dtype("float64"), np.dtype("int64")) or not np.isfinite(values).all():
                raise ValueError("finite FP64 or int64 arrays required")
            result[key] = torch.from_numpy(values.copy())
        return result


def same(left, right, label):
    if set(left) != set(right) or any(left[key].dtype != right[key].dtype or left[key].shape != right[key].shape
                                     or not torch.equal(left[key], right[key]) for key in left):
        raise ValueError(label + " full tensor arrays differ")


def state_arrays(weights, remainders):
    return {**{"w_" + key: weights[key] for key in WEIGHTS}, **{"r_" + key: remainders[key] for key in WEIGHTS}}


def state(path):
    arrays = load(path)
    equal(sorted(arrays), sorted(prefix + key for prefix in ("w_", "r_") for key in WEIGHTS), "complete state keys")
    weights, remainders = ({key: arrays[prefix + key] for key in WEIGHTS} for prefix in ("w_", "r_"))
    arithmetic.shadow(weights, remainders, k=K)
    return weights, remainders


def window(raw, dataset, index):
    if type(index) is not int or index not in set(WINDOWS) | set(PROBES):
        raise ValueError("undeclared training window")
    tokens, targets = pilot.window(raw, dataset, "train", index)
    return torch.tensor(tokens, dtype=torch.int64), targets


def mean_metrics(rows):
    if not rows:
        raise ValueError("nonempty equal-length window metrics required")
    keys = ("true_ce", "lut_nll", "accuracy")
    if any(not isinstance(row, dict) or set(row) != set(keys)
           or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0 for key in keys)
           or row["accuracy"] > 1 for row in rows):
        raise ValueError("complete finite metrics required")
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def decision(endpoints):
    equal(set(endpoints), set(ARMS), "both endpoint arms")
    if any(not isinstance(endpoints[arm], list) or len(endpoints[arm]) != len(WINDOWS) for arm in ARMS):
        raise ValueError("exactly all declared endpoint windows required")
    means = {arm: mean_metrics(endpoints[arm]) for arm in ARMS}
    improvements = {key: means["sequential"][key] - means["batch4"][key] for key in ("true_ce", "lut_nll")}
    return {"endpoint_means": means, "sequential_minus_batch4": improvements,
            "direction": "both_training_losses_improve" if all(value > 0 for value in improvements.values())
                         else "mixed_or_nonpositive_training_result",
            "recipe_promoted": False, "scope": "descriptive fitted-training mechanism diagnostic only"}


def collect_scores(directory, label, weights, indices, raw, dataset, context=True, *, write=False, recompute=False):
    rows = []
    for index in indices:
        tokens, targets = window(raw, dataset, index)
        path = directory / f"score_{label}_{index:03d}_{int(context)}.npz"
        if write:
            result = extension.forward(weights, tokens, use_context=context)
            save(path, {"logits": result["logits"]})
        arrays = load(path)
        equal(set(arrays), {"logits"}, "complete score archive")
        logits = arrays["logits"]
        if logits.dtype != torch.float64 or logits.shape != (len(targets), weights["EMBED"].shape[0]):
            raise ValueError("complete D512 head logits required")
        if bool(((logits < -32768) | (logits > 32767) | (logits != logits.round())).any()):
            raise ValueError("integer int16 score logits required")
        if recompute:
            result = prior.reference.forward(weights, tokens, mode="integer_anchor", use_context=context)
            same(arrays, {"logits": result["logits"]}, "independent readout replay")
        rows.append(prior.metrics(logits, targets))
    return rows


def validate_query(arrays, weights, length):
    vocab, width = weights["EMBED"].shape
    layers, _, taps = weights["MH"].shape
    keys = {"logits", "head", "predictions", "AD_HEAD", "PROJ_IDX", "TAP_IDX"} | {"g_" + key for key in WEIGHTS}
    equal(set(arrays), keys, "complete full-gradient query arrays")
    geometry = {"logits": (torch.float64, (length, vocab), -32768, 32767),
                "head": (torch.int64, (length, vocab), -(1 << 31), (1 << 31) - 1),
                "predictions": (torch.int64, (length,), 0, vocab - 1),
                "AD_HEAD": (torch.int64, (length, vocab), 0, width - 1),
                "PROJ_IDX": (torch.int64, (layers, length, width), 0, width - 1),
                "TAP_IDX": (torch.int64, (layers, length, width), 0, taps - 1)}
    for key, (dtype, shape, lower, upper) in geometry.items():
        value = arrays[key]
        if (value.dtype != dtype or value.shape != shape or not torch.isfinite(value).all()
                or bool(((value < lower) | (value > upper) | (value != value.round())).any())):
            raise ValueError("complete bounded query array required: " + key)
    for layer in range(layers):
        if bool((arrays["TAP_IDX"][layer] * (1 << min(layer, 6)) > torch.arange(length)[:, None]).any()):
            raise ValueError("query tap references a future/negative history position")
    arithmetic._gradient({key: arrays["g_" + key] for key in WEIGHTS}, weights)


def query(directory, index, weights, raw, dataset, *, write=False, recompute=False):
    tokens, targets = window(raw, dataset, index)
    path = directory / f"query_{index:03d}.npz"
    if write:
        base = extension.forward(weights, tokens)
        info = extension.gradient(weights, tokens, targets, base, variant="legacy")
        save(path, {**{"g_" + key: info["gradients"][key] for key in WEIGHTS}, "logits": base["logits"],
                    "head": info["head"]["native_gradient"], "predictions": info["head"]["predictions"],
                    **base["routes"]})
    arrays = load(path)
    validate_query(arrays, weights, len(targets))
    gradient = {key: arrays["g_" + key] for key in WEIGHTS}
    head = prior.numeric.loss_head(arrays["logits"], targets, "variable_mass", reduction="sum")
    same({"head": arrays["head"], "predictions": arrays["predictions"]},
         {"head": head["native_gradient"], "predictions": head["predictions"]}, "raw head and logits")
    if recompute:
        shape = pilot.layout(512)
        flat = {key: weights[key].to(torch.int64).reshape(-1).tolist() for key in WEIGHTS}
        observed = prior.observer.observe(flat, tokens.tolist(), targets, shape, zshift=2, include_parameter_gradients=True)
        equal([observed["saturation_events"], observed["route_diagnostics"]["backward_int32_overflow_detected"]],
              [0, False], "independent gradient numerical guards")
        equal(observed["parameter_gradients"], {key: gradient[key].reshape(-1).tolist() for key in WEIGHTS}, "all raw gradients")
        equal(observed["native_routes"], {key: arrays[key].reshape(-1).tolist() for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")}, "all routes")
        equal(observed["gradients"], arrays["head"].reshape(-1).tolist(), "all raw head gradients")
        equal(observed["predictions"], arrays["predictions"].tolist(), "all predictions")
        independent = prior.initializer.calibration.integer_logits(flat, tokens.tolist(), shape)
        equal(independent["saturation_events"], 0, "independent forward bounds")
        equal(independent["logits"], arrays["logits"].tolist(), "all raw query logits")
    return gradient


def fixed_partition(weights, remainders, gradients, batched):
    sequential_w, sequential_r = weights, remainders
    for gradient in gradients:
        result = accumulation.apply_block(sequential_w, sequential_r, [gradient], k=K)
        sequential_w, sequential_r = result["weights"], result["remainders"]
    same(arithmetic.shadow(sequential_w, sequential_r, K), arithmetic.shadow(batched["weights"], batched["remainders"], K),
         "fixed-gradient partition shadow")
    return {"visible_difference": old.stats(old.subtract(sequential_w, batched["weights"])),
            "remainder_difference": old.stats(old.subtract(sequential_r, batched["remainders"])),
            "same_shadow": True, "counterfactual_committed": False}


def sequential_control(binding, index, before_w, before_r, gradient, result):
    directory = Path(binding["prior_run"]) / "diverse"
    row = read(directory / f"step_{index+1:06d}.json")
    saved = old.load_arrays(directory / f"gradients_{index+1:06d}.npz",
                            [prefix + key for prefix in ("native_", "float_") for key in WEIGHTS])
    same(gradient, {key: saved["native_" + key] for key in WEIGHTS}, "complete historical native gradient")
    for section, w, r in (("before_hashes", before_w, before_r), ("after_hashes", result["weights"], result["remainders"])):
        equal([old.tensor_hash(w), old.tensor_hash(r)], [row[section]["native"], row[section]["remainder"]],
              "every historical native weight/remainder hash")
    if index + 1 in old.CHECKPOINTS:
        decoded = old.native.decode_state((directory / f"native_{index+1:06d}.i32").read_bytes(), pilot.layout(512))
        old_w, old_r = old.state_tensors(decoded)
        same(result["weights"], old_w, "saved native weights")
        same(result["remainders"], old_r, "saved native remainders")


def execute(manifest, output, raw, *, write=False, recompute=False):
    binding, dataset = manifest["binding"], manifest["binding"]["dataset"]
    initial = old.native.decode_state(Path(binding["initial_state"]).read_bytes(), pilot.layout(512))
    equal([initial["step"], initial["k"], initial["config"]["loss_kind"], initial["config"]["zshift"]], [0, K, 1, 2], "fitted start config")
    start_w, start_r = old.state_tensors(initial)
    old.require_zero(start_r, "zero initial remainders")
    arms, endpoints, final_states, totals = [], {}, {}, {}
    for arm in ARMS:
        directory = output / arm
        if write:
            directory.mkdir()
            save(directory / "state_000.npz", state_arrays(start_w, start_r))
        weights, remainders = state(directory / "state_000.npz")
        same(state_arrays(weights, remainders), state_arrays(start_w, start_r), "same fitted start")
        total = {key: torch.zeros_like(start_w[key]) for key in WEIGHTS}
        initial_scores = collect_scores(directory, "initial", weights, WINDOWS, raw, dataset, write=write, recompute=recompute)
        blocks, probes, commits = [], [], 0
        for consumed in range(0, 33, BLOCK):
            if consumed in PROBE_EXPOSURES:
                probes.append({"windows_consumed": consumed,
                    "context_on": collect_scores(directory, f"probe_{consumed:03d}", weights, PROBES, raw, dataset, write=write, recompute=recompute),
                    "context_off": collect_scores(directory, f"probe_{consumed:03d}", weights, PROBES, raw, dataset, False, write=write, recompute=recompute)})
            if consumed == 32:
                break
            indices = list(range(consumed, consumed + BLOCK))
            before = collect_scores(directory, f"before_{consumed:03d}", weights, indices, raw, dataset, write=write, recompute=recompute)
            gradients, prefix, partition = [], [], None
            for offset, index in enumerate(indices):
                gradient = query(directory, index, weights, raw, dataset, write=write, recompute=recompute)
                gradients.append(gradient)
                total = {key: total[key] + gradient[key] for key in WEIGHTS}
                if arm == "sequential" or offset == BLOCK - 1:
                    group = [gradient] if arm == "sequential" else gradients
                    result = accumulation.apply_block(weights, remainders, group, k=K)
                    if arm == "sequential":
                        sequential_control(binding, index, weights, remainders, gradient, result)
                    else:
                        partition = fixed_partition(weights, remainders, group, result)
                    prefix.append(result["prefix_max_abs"])
                    commits += 1
                    next_w, next_r = result["weights"], result["remainders"]
                    path = directory / f"state_{index+1:03d}.npz"
                    if write:
                        save(path, state_arrays(next_w, next_r))
                    saved_w, saved_r = state(path)
                    same(state_arrays(saved_w, saved_r), state_arrays(next_w, next_r), "every full commit replay")
                    weights, remainders = next_w, next_r
            after = collect_scores(directory, f"after_{consumed:03d}", weights, indices, raw, dataset, write=write, recompute=recompute)
            vectors = [prior.vector(gradient) for gradient in gradients]
            norm_sum = sum(float(vector.norm()) for vector in vectors)
            ratio = float(sum(vectors).norm()) / norm_sum if norm_sum else None
            blocks.append({"windows": indices, "commits_so_far": commits, "before": before, "after": after,
                "mean_before": mean_metrics(before), "mean_after": mean_metrics(after),
                "gradient_sum_norm_over_sum_norms": ratio, "gradients_share_query_weights": arm == "batch4",
                "accumulation_prefix_max_abs": prefix, "fixed_gradient_partition": partition})
            if write:
                print(json.dumps({"arm": arm, "windows_consumed": consumed + BLOCK, "commits": commits}), flush=True)
        final = collect_scores(directory, "final", weights, WINDOWS, raw, dataset, write=write, recompute=recompute)
        endpoints[arm] = final
        old.require_zero({key: (weights[key] - start_w[key]) * (1 << K) - remainders[key] + start_r[key] + total[key]
                          for key in WEIGHTS}, "full accumulated shadow identity")
        final_states[arm], totals[arm] = (weights, remainders), total
        arms.append({"arm": arm, "optimizer_commits": commits, "gradient_queries": 32, "windows_consumed": 32,
                     "initial_scores": initial_scores, "final_scores": final, "blocks": blocks, "probes": probes,
                     "public_frozen_arithmetic_parity": True, "exact_shadow_conservation": True,
                     "old_native_control_matches": True if arm == "sequential" else None})
    left, right = (arithmetic.shadow(*final_states[arm], k=K) for arm in ARMS)
    old.require_zero({key: (left[key] - right[key]) * (1 << K) + totals["sequential"][key] - totals["batch4"][key]
                      for key in WEIGHTS}, "paired cumulative-gradient shadow identity")
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "arms": arms, "decision": decision(endpoints), "paired_shadow_identity": True,
            "heldout_corpus_evaluated": False, "physical_fpga_execution": False, "paid_execution_authorized": False,
            "recipe_promoted": False, "scope": "host integer-reference mechanism diagnostic; no native batch implementation"}


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "artifact_manifest_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts), "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "complete saved-gradient state replay and metrics; --recompute independently repeats all64 full gradients and292 readouts"}


def expected_artifacts():
    names = {"manifest.json", "protocol.json", "declaration.md", "summary.json"}
    for arm in ARMS:
        prefix = arm + "/"
        commits = range(33) if arm == "sequential" else range(0, 33, BLOCK)
        names.update(prefix + f"state_{step:03d}.npz" for step in commits)
        names.update(prefix + f"query_{index:03d}.npz" for index in WINDOWS)
        for label in ("initial", "final"):
            names.update(prefix + f"score_{label}_{index:03d}_1.npz" for index in WINDOWS)
        for consumed in range(0, 32, BLOCK):
            for label in ("before", "after"):
                names.update(prefix + f"score_{label}_{consumed:03d}_{index:03d}_1.npz"
                             for index in range(consumed, consumed + BLOCK))
        for consumed in PROBE_EXPOSURES:
            names.update(prefix + f"score_probe_{consumed:03d}_{index:03d}_{context}.npz"
                         for index in PROBES for context in (0, 1))
    return names


def check(root, record, recompute=False):
    torch.set_num_threads(1)
    root = Path(root).resolve(strict=True)
    manifest = read(root / "manifest.json")
    frozen(manifest)
    equal(manifest["run_directory"], str(root), "owned run path")
    equal(read(root / "protocol.json"), recipe(), "frozen protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "saved declaration")
    artifacts = prior.artifact_manifest(root)
    seal_bytes, public_bytes = (root / "artifact_manifest.json").read_bytes(), Path(record).read_bytes()
    equal(read(root / "artifact_manifest.json"), artifacts, "complete artifact inventory")
    if any("failure" in row["file"] for row in artifacts):
        raise ValueError("failed run cannot be complete")
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact declared artifact membership")
    raw, dataset = pilot.corpus_plan(Path(manifest["binding"]["dataset"]["path"]), 1024, 88)
    equal(dataset, manifest["binding"]["dataset"], "same bound corpus")
    summary = execute(manifest, root, raw, recompute=recompute)
    equal(read(root / "summary.json"), summary, "fully reconstructed summary")
    expected = public_record(root, summary, artifacts)
    equal(read(record), expected, "public/raw record equality")
    frozen(manifest)
    equal(prior.artifact_manifest(root), artifacts, "run unchanged through check")
    equal(read(root / "artifact_manifest.json"), artifacts, "seal unchanged through check")
    equal(read(record), expected, "public unchanged through check")
    equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "seal bytes unchanged through check")
    equal(Path(record).read_bytes(), public_bytes, "public bytes unchanged through check")
    return expected


def run(config):
    output, target = (prior.capacity._new_external_path(path) for path in (config.output, config.record))
    if target == output or output in target.parents or target in output.parents:
        raise ValueError("public receipt must be outside raw output")
    if not target.parent.is_dir():
        raise FileNotFoundError("receipt parent must exist")
    sources = source_identity()
    output.mkdir(parents=True, exist_ok=False)
    previous, installed, published = None, False, None
    staging_directory, staged, owned = None, None, False
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed accumulation diagnostic deadline exceeded")
        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
        torch.set_num_threads(1)
        binding, raw = bind_inputs(config)
        declaration = PLAN.read_bytes()
        manifest = {"schema": SCHEMA, "run_directory": str(output), "recipe": recipe(), "binding": binding,
                    "source_hashes": sources, "plan_sha256": old.digest(declaration),
                    "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"}}
        pilot.write_json(output / "manifest.json", manifest)
        pilot.write_json(output / "protocol.json", recipe())
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        frozen(manifest)
        summary = execute(manifest, output, raw, write=True)
        frozen(manifest)
        pilot.write_json(output / "summary.json", summary)
        artifacts = prior.artifact_manifest(output)
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = public_record(output, summary, artifacts)
        published = old.digest((json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-accumulation-", dir=target.parent))
        staged = staging_directory / "receipt.json"
        pilot.write_json(staged, public)
        os.link(staged, target)
        owned = True
        result = check(output, target)
        staged.unlink()
        staging_directory.rmdir()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        installed = False
        return result
    except BaseException as error:
        same_inode = staged is not None and staged.is_file() and target.is_file() and os.path.samefile(staged, target)
        if (owned or same_inode) and published is not None and target.is_file() and old.digest(target.read_bytes()) == published:
            rejected = output / "rejected_public.json"
            if rejected.exists():
                raise RuntimeError("cannot overwrite rejected receipt") from error
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
    for name in ("output", "record", "corpus", "prior-run", "prior-record"):
        parser.add_argument("--" + name, type=Path, required=name in ("output", "record"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    config = parser.parse_args(argv)
    if config.recompute and not config.check:
        parser.error("--recompute requires --check")
    if not config.check and any(getattr(config, key) is None for key in ("corpus", "prior_run", "prior_record")):
        parser.error("run requires corpus and complete persistent predecessor run/record")
    if config.check:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed read-only check deadline exceeded")
        previous = signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
            result = check(config.output, config.record, config.recompute)
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            finally:
                signal.signal(signal.SIGALRM, previous)
    else:
        result = run(config)
    print(json.dumps({"complete": result["complete"], "decision": result["decision"],
                      "check_only": config.check, "independent_recompute": config.recompute}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
