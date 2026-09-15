"""Uncommitted first-failing-block fraction diagnostic on training windows4..7.

All four fractions are retained. This is neither a training continuation nor a
line search selecting an optimizer rate. Read-only replay repeats saved-point
observations; only integer dynamic forwards and full gradients use distinct
reference implementations. Affine and fractional-shadow checks reuse their
frozen defining helpers.
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

from manuscript.tools import diagnose_fpga_lm_gradient_accumulation as parent  # noqa: E402
from manuscript.tools import fpga_lm_step_fraction as fraction  # noqa: E402

old, prior, pilot = parent.old, parent.prior, parent.pilot
extension, arithmetic = parent.extension, parent.arithmetic
read, equal, WEIGHTS = parent.read, parent.equal, parent.WEIGHTS
SCHEMA = "tikitorch.fpga_lm_batch_step.v1"
WINDOWS = (4, 5, 6, 7)
SHIFTS = (None, 4, 2, 0)
LABELS = ("zero", "sixteenth", "quarter", "full")
MODELS = ("fresh_affine", "carry_affine", "rounded_affine", "fixed_integer", "dynamic_integer")
BASELINES = ("visible", "shadow")
K, DEADLINE_S = 18, 1800
PLAN = REPO / "manuscript/research/PLAN.md"


def source_identity():
    sources = {**parent.source_identity(), **fraction.source_identity()}
    for name in ("manuscript/tools/diagnose_fpga_lm_batch_step.py", "tests/test_fpga_lm_batch_step.py",
                 "tests/test_fpga_lm_step_fraction.py"):
        sources[name] = old.digest((REPO / name).read_bytes())
    return sources


def recipe():
    return {"schema": SCHEMA, "partition": "train", "windows": list(WINDOWS),
            "parent_arm": "batch4", "parent_before_exposures": 4, "parent_after_exposures": 8,
            "block_selection": "fixed already-observed first failing block; not searched at execution",
            "labels": list(LABELS), "shifts": list(SHIFTS), "fractions": [0.0, 0.0625, 0.25, 1.0],
            "base_k": K, "models": list(MODELS), "baselines": list(BASELINES),
            "gradient_rule": "four saved raw legacy LUT zshift2 gradients at the same original block-start weights",
            "carry_rule": "preserve z=w-r/2**18; scale fresh G only, never reset or rescale its physical carry",
            "fractional_state": "positive shift j stores r*2**j with denominator2**(18+j) before adding G",
            "zero_state": "exact unchanged original visible weights/remainders/k18",
            "full_state": "exact actual parent batch4 state008 weights/remainders/k18",
            "fresh_affine": "w-alpha*G/2**18 with original visible-state selections and centering offsets",
            "carry_affine": "z-alpha*G/2**18 with the SAME original visible-state affine tape",
            "rounded_affine": "candidate integer weights with original affine tape",
            "fixed_integer": "candidate integer weights, original routes, recomputed floor means",
            "dynamic_integer": "candidate integer weights with recomputed routes and floor means",
            "baseline_shadow": "original fractional shadow z with dynamic floor forward; not native integer",
            "comparison_policy": "compare each model with its OWN zero-fraction readout; dynamic shadow is not "
                                 "the baseline for the visible-tape carry-affine model",
            "range_policy": "all readout parameters are finite and within int16 bounds, including fractional ideals; "
                            "affine internal activations are unconstrained algebraic extrapolations, not deployable; "
                            "dynamic and fixed-integer forwards reject all would-clip intermediate outputs",
            "readouts": 88, "candidate_readouts": 80, "baseline_readouts": 8,
            "recompute_gradient_queries": 4, "recompute_gradient_pair_observations": 512,
            "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_fitting_pair_exposures": 0,
            "recompute_scope": "four independent observer gradients; distinct integer dynamic oracle; "
                               "affine/fixed-integer/shadow replay reuse frozen defining helpers",
            "torch_threads": 1, "deadline_s": DEADLINE_S, "candidates_committed": False,
            "selected_fraction": None, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "recipe_promoted": False, "statistical_significance_claim": False, "adaptive_extension": False}


def bind_inputs(config):
    root = Path(config.prior_run).resolve(strict=True)
    record = Path(config.prior_record).resolve(strict=True)
    seal_path = root / "artifact_manifest.json"
    before_tree = prior.artifact_manifest(root)
    seal_bytes, public_bytes = seal_path.read_bytes(), record.read_bytes()
    parent.check(root, record, recompute=False)
    manifest = read(root / "manifest.json")
    raw, dataset = pilot.corpus_plan(Path(config.corpus), 1024, 88)
    equal(dataset, manifest["binding"]["dataset"], "identical complete corpus/split")
    files = [root / name for name in ("manifest.json", "summary.json", "artifact_manifest.json")]
    files += [record, Path(dataset["path"]), root / "batch4/state_004.npz", root / "batch4/state_008.npz"]
    files += [root / f"batch4/query_{index:03d}.npz" for index in WINDOWS]
    binding = {"prior_run": str(root), "prior_record": str(record), "dataset": dataset,
               "state_steps": [4, 8], "query_windows": list(WINDOWS),
               "inputs": [old.file_identity(path) for path in files],
               "parent_artifact_files": len(before_tree),
               "parent_artifact_payload_bytes": sum(item["bytes"] for item in before_tree)}
    equal(prior.artifact_manifest(root), before_tree, "parent tree unchanged during binding")
    equal(seal_path.read_bytes(), seal_bytes, "parent seal bytes unchanged during binding")
    equal(record.read_bytes(), public_bytes, "parent public bytes unchanged during binding")
    return binding, raw


def frozen(manifest):
    equal(manifest["schema"], SCHEMA, "schema")
    equal(manifest["recipe"], recipe(), "declared exact diagnostic")
    equal(manifest["source_hashes"], source_identity(), "frozen complete sources/tests")
    binding = manifest["binding"]
    for item in binding["inputs"]:
        equal(old.file_identity(item["path"]), item, "bound predecessor/input unchanged")
    rebuilt, _ = bind_inputs(SimpleNamespace(prior_run=binding["prior_run"], prior_record=binding["prior_record"],
                                            corpus=binding["dataset"]["path"]))
    equal(rebuilt, binding, "canonical complete predecessor binding")


def expected_artifacts():
    files = {"manifest.json", "protocol.json", "declaration.md", "summary.json", "gradient_sum.npz",
             "parent_state_004.npz", "parent_state_008.npz"}
    files.update(f"query_{index:03d}.npz" for index in WINDOWS)
    files.update(f"baseline_{baseline}_{index:03d}.npz" for baseline in BASELINES for index in WINDOWS)
    for label in LABELS:
        files.update((f"{label}/state.npz", f"{label}/candidate.json"))
        files.update(f"{label}/{model}_{index:03d}.npz" for model in MODELS for index in WINDOWS)
    return files


def _copy_exact(source, destination, write):
    payload = Path(source).read_bytes()
    if write:
        with Path(destination).open("xb") as stream:
            stream.write(payload)
    equal(Path(destination).read_bytes(), payload, "exact bound parent payload copy")


def _window(raw, dataset, index):
    if type(index) is not int or index not in WINDOWS:
        raise ValueError("only fixed training windows4..7 are permitted")
    tokens, targets = pilot.window(raw, dataset, "train", index)
    return torch.tensor(tokens, dtype=torch.int64), targets


def _trace(result):
    return {"logits": result["logits"], "centering_offsets": result["centering_offsets"], **result["routes"]}


def _validate_trace(arrays, weights, tokens, *, integer):
    geometry = prior.reference._geometry(weights, tokens)
    v, t = (geometry[key] for key in ("V", "T"))
    equal(set(arrays), {"logits", "centering_offsets", "AD_HEAD", "PROJ_IDX", "TAP_IDX"}, "complete forward trace")
    if arrays["logits"].dtype != torch.float64 or arrays["logits"].shape != (t, v):
        raise ValueError("full finite FP64 logit geometry required")
    if integer and bool(((arrays["logits"] < -32768) | (arrays["logits"] > 32767)
                         | (arrays["logits"] != arrays["logits"].round())).any()):
        raise ValueError("integer dynamic/floor logits must fit int16")
    # Reuse the strict old route/cell certificate validator. No model execution.
    base = {"schema": prior.reference.SCHEMA, "mode": "fixed_routes", "geometry": geometry,
            "tokens": tokens, "use_context": True, "logits": arrays["logits"],
            "routes": {key: arrays[key] for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")},
            "centering_offsets": arrays["centering_offsets"]}
    prior.reference._base(base, geometry, tokens, True)


def _base_from_trace(arrays, weights, tokens):
    _validate_trace(arrays, weights, tokens, integer=True)
    return {"schema": prior.reference.SCHEMA, "mode": "integer_anchor",
            "geometry": prior.reference._geometry(weights, tokens), "tokens": tokens,
            "use_context": True, "logits": arrays["logits"],
            "routes": {key: arrays[key] for key in ("AD_HEAD", "PROJ_IDX", "TAP_IDX")},
            "centering_offsets": arrays["centering_offsets"]}


def readout(path, weights, tokens, targets, *, mode, base=None, write=False, recompute=False):
    # Arithmetic-only ideals/shadows do not themselves authorize a bounded
    # model forward. This also applies when only reading existing artifacts.
    extension.parameter_digest(weights)

    def forward(independent):
        if mode == "dynamic_integer":
            return (prior.reference.forward(weights, tokens, mode="integer_anchor", use_context=True)
                    if independent else extension.forward(weights, tokens))
        if mode == "shadow":
            return extension.forward(weights, tokens)
        return prior.reference.forward(weights, tokens, mode=mode, base=base, use_context=True)

    if write:
        parent.save(path, _trace(forward(False)))
    arrays = parent.load(path)
    _validate_trace(arrays, weights, tokens, integer=mode in ("dynamic_integer", "fixed_integer"))
    if base is not None:
        parent.same({key: arrays[key] for key in base["routes"]}, base["routes"], "fixed selected routes")
        if mode == "fixed_routes":
            parent.same({"centering_offsets": arrays["centering_offsets"]},
                        {"centering_offsets": base["centering_offsets"]}, "fixed affine offsets")
    if recompute:
        parent.same(arrays, _trace(forward(True)), "complete recomputed logits/routes/offsets")
    return {"metrics": prior.metrics(arrays["logits"], targets)}, arrays


def candidate_arrays(result):
    return {prefix + key: result[field][key]
            for prefix, field in (("w_", "weights"), ("r_", "remainders"),
                                  ("ideal_", "ideal"), ("fresh_", "fresh_ideal"),
                                  ("shadow_", "shadow_before")) for key in WEIGHTS}


def candidate_metadata(result):
    tensor_fields = {"weights", "remainders", "gradient", "ideal", "fresh_ideal", "shadow_before"}
    return {**{key: value for key, value in result.items() if key not in tensor_fields},
            "tensor_sha256": {key: old.tensor_hash(result[key]) for key in sorted(tensor_fields)},
            "visible_minus_fresh_ideal": old.stats(old.subtract(result["weights"], result["fresh_ideal"]))}


def _check_candidate(result, weights, remainders, summed, shift):
    effective_k = K if shift is None else K + shift
    alpha = 0.0 if shift is None else 2.0 ** (-shift)
    if type(result["alpha"]) is not float:
        raise ValueError("candidate alpha must retain its declared literal float type")
    equal([result["schema"], result["shift"], result["alpha"], result["base_k"], result["k"]],
          [fraction.SCHEMA, shift, alpha, K, effective_k], "declared candidate scalar identity")
    equal([result["committed"], result["native_execution"], result["public_reference_equal"], result["exact_shadow_recurrence"]],
          [False, False, True, True], "uncommitted checked candidate flags")
    parent.same(result["gradient"], summed, "complete unchanged gradient sum")
    arithmetic.shadow(result["weights"], result["remainders"], k=effective_k)
    zero = {key: torch.zeros_like(weights[key]) for key in WEIGHTS}
    scaled_r = remainders if shift is None else {key: value * (1 << shift) for key, value in remainders.items()}
    oracle = arithmetic.ef_step(weights, scaled_r, zero if shift is None else summed, k=effective_k)
    parent.same(result["weights"], oracle["weights"], "independent frozen candidate weights")
    parent.same(result["remainders"], oracle["remainders"], "independent frozen candidate remainders")
    shadow = arithmetic.shadow(weights, remainders, k=K)
    parent.same(result["shadow_before"], shadow, "unchanged inherited shadow")
    parent.same(result["ideal"], {key: shadow[key] - alpha * summed[key] / (1 << K) for key in WEIGHTS},
                "carry-preserving ideal")
    parent.same(result["fresh_ideal"], {key: weights[key] - alpha * summed[key] / (1 << K) for key in WEIGHTS},
                "fresh-only ideal")


def execute(manifest, output, raw, *, write=False, recompute=False):
    output = Path(output)
    binding, dataset = manifest["binding"], manifest["binding"]["dataset"]
    original = Path(binding["prior_run"]) / "batch4"
    for step in (4, 8):
        _copy_exact(original / f"state_{step:03d}.npz", output / f"parent_state_{step:03d}.npz", write)
    weights, remainders = parent.state(output / "parent_state_004.npz")
    end_w, end_r = parent.state(output / "parent_state_008.npz")
    gradients = []
    for index in WINDOWS:
        _copy_exact(original / f"query_{index:03d}.npz", output / f"query_{index:03d}.npz", write)
        gradients.append(parent.query(output, index, weights, raw, dataset, recompute=recompute))
    zero = {key: torch.zeros_like(weights[key]) for key in WEIGHTS}
    summed = parent.accumulation.checked_sum(gradients, zero)
    if write:
        parent.save(output / "gradient_sum.npz", summed)
    parent.same(parent.load(output / "gradient_sum.npz"), summed, "complete saved gradient sum")
    shadow = arithmetic.shadow(weights, remainders, k=K)
    baselines, bases, baseline_traces = [], {}, {}
    for index in WINDOWS:
        tokens, targets = _window(raw, dataset, index)
        row = {"window_index": index}
        for label, values, mode in (("visible", weights, "dynamic_integer"), ("shadow", shadow, "shadow")):
            score, trace = readout(output / f"baseline_{label}_{index:03d}.npz", values, tokens, targets,
                                   mode=mode, write=write, recompute=recompute)
            row[label] = score["metrics"]
            baseline_traces[(label, index)] = trace
            if label == "visible":
                bases[index] = _base_from_trace(trace, weights, tokens)
                actual = parent.load(output / f"query_{index:03d}.npz")
                parent.same({key: trace[key] for key in ("logits", "AD_HEAD", "PROJ_IDX", "TAP_IDX")},
                            {key: actual[key] for key in ("logits", "AD_HEAD", "PROJ_IDX", "TAP_IDX")},
                            "actual original query logits/routes are the affine anchor")
        baselines.append(row)

    candidates = []
    for label, shift in zip(LABELS, SHIFTS):
        directory = output / label
        if write:
            directory.mkdir()
        result = fraction.step_fraction(weights, remainders, gradients, shift)
        _check_candidate(result, weights, remainders, summed, shift)
        metadata = candidate_metadata(result)
        if write:
            parent.save(directory / "state.npz", candidate_arrays(result))
            pilot.write_json(directory / "candidate.json", metadata)
        parent.same(parent.load(directory / "state.npz"), candidate_arrays(result), "complete candidate state/ideals")
        equal(read(directory / "candidate.json"), metadata, "complete candidate metadata")
        if shift is None:
            parent.same(result["weights"], weights, "zero fraction exact unchanged weights")
            parent.same(result["remainders"], remainders, "zero fraction exact unchanged carry")
        if shift == 0:
            parent.same(result["weights"], end_w, "full fraction actual parent endpoint weights")
            parent.same(result["remainders"], end_r, "full fraction actual parent endpoint carry")
        rows = []
        for index in WINDOWS:
            tokens, targets = _window(raw, dataset, index)
            base = bases[index]
            models = {"fresh_affine": (result["fresh_ideal"], "fixed_routes"),
                      "carry_affine": (result["ideal"], "fixed_routes"),
                      "rounded_affine": (result["weights"], "fixed_routes"),
                      "fixed_integer": (result["weights"], "fixed_integer"),
                      "dynamic_integer": (result["weights"], "dynamic_integer")}
            row = {"window_index": index, "models": {}}
            for model in MODELS:
                values, mode = models[model]
                score, trace = readout(directory / f"{model}_{index:03d}.npz", values, tokens, targets,
                                       mode=mode, base=None if mode == "dynamic_integer" else base,
                                       write=write, recompute=recompute)
                row["models"][model] = score["metrics"]
                if shift is None and model in ("fresh_affine", "rounded_affine", "fixed_integer", "dynamic_integer"):
                    parent.same(trace, baseline_traces[("visible", index)], "zero visible-state readout unchanged")
            rows.append(row)
        means = {model: parent.mean_metrics([row["models"][model] for row in rows]) for model in MODELS}
        candidates.append({"label": label, "candidate": metadata, "windows": rows, "mean_metrics": means,
                           "fraction_committed": False})
        if write:
            print(json.dumps({"fraction": label, "readouts_complete": len(rows) * len(MODELS), "committed": False}), flush=True)
    return {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": manifest["source_hashes"],
            "baselines": baselines,
            "baseline_means": {label: parent.mean_metrics([row[label] for row in baselines]) for label in BASELINES},
            "candidates": candidates, "completed_fractions": len(candidates), "readouts": 88,
            "new_training_updates": 0, "new_gradient_pair_exposures": 0, "selected_fraction": None,
            "candidates_committed": False, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "zero_fraction_original_state_equal": True, "full_fraction_parent_endpoint_equal": True,
            "scope": "same-point exploratory training-block diagnostics; no training continuation, selection or promotion"}


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root),
            "artifact_manifest_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
            "artifact_files_verified": len(artifacts),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "check_scope": "complete saved-point/state algebra and metrics; --recompute repeats four existing full gradients "
                           "and all88 forwards with the recipe-declared oracle boundaries"}


def check(root, record, recompute=False):
    if type(recompute) is not bool:
        raise ValueError("recompute requires a literal boolean")
    torch.set_num_threads(1)
    root, record = Path(root).resolve(strict=True), Path(record).resolve(strict=True)
    if record == root or root in record.parents or record in root.parents:
        raise ValueError("public receipt must be outside raw output")
    seal_bytes, public_bytes = (root / "artifact_manifest.json").read_bytes(), record.read_bytes()
    manifest = read(root / "manifest.json")
    frozen(manifest)
    equal(manifest["run_directory"], str(root), "owned run location")
    equal(read(root / "protocol.json"), recipe(), "fixed protocol")
    equal(old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "actual pre-run declaration")
    artifacts = prior.artifact_manifest(root)
    equal({item["file"] for item in artifacts}, expected_artifacts(), "complete exact diagnostic membership")
    equal(read(root / "artifact_manifest.json"), artifacts, "complete actual artifact seal")
    raw, dataset = pilot.corpus_plan(Path(manifest["binding"]["dataset"]["path"]), 1024, 88)
    equal(dataset, manifest["binding"]["dataset"], "complete bound corpus")
    summary = execute(manifest, root, raw, recompute=recompute)
    equal(read(root / "summary.json"), summary, "reconstructed all-fraction summary")
    expected = public_record(root, summary, artifacts)
    equal(read(record), expected, "public/raw equality")
    frozen(manifest)
    equal(prior.artifact_manifest(root), artifacts, "raw tree unchanged through replay")
    equal((root / "artifact_manifest.json").read_bytes(), seal_bytes, "seal bytes unchanged through replay")
    equal(record.read_bytes(), public_bytes, "public bytes unchanged through replay")
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
            raise TimeoutError("fixed batch-step diagnostic deadline exceeded")
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
        equal({item["file"] for item in artifacts}, expected_artifacts(), "exact completed run inventory")
        pilot.write_json(output / "artifact_manifest.json", artifacts)
        public = public_record(output, summary, artifacts)
        published = old.digest((json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-batch-step-", dir=target.parent))
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
        parser.error("run requires corpus and complete accumulation predecessor run/record")
    if config.check:
        def timeout_handler(signum, frame):
            raise TimeoutError("fixed read-only batch-step check deadline exceeded")
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
    print(json.dumps({"complete": result["complete"], "completed_fractions": result["completed_fractions"],
                      "check_only": config.check, "recompute": config.recompute}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
