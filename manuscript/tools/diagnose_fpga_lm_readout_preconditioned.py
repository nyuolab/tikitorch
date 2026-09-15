"""One same-objective Gram-preconditioned readout solve on audited cached features.

No backbone execution, new cohorts, bias refit, or FPGA performance claim.
The independent audit checks the transform and physical endpoint gradients,
not unretained L-BFGS trial arrays or formal floating-point error bounds.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_readout_oracle as previous  # noqa: E402
from manuscript.tools import fpga_lm_readout_preconditioned as helper  # noqa: E402
from manuscript.tools import fpga_lm_readout_preconditioned_validation as validator  # noqa: E402

parent, old, prior, guard = previous.parent, previous.old, previous.prior, previous.guard
read, equal, load, near = previous.read, previous.equal, previous.load, previous.near
compare_results = previous.compare_results
Deadline, Storage = previous.Deadline, previous.Storage
TRAIN_WINDOWS, SCORE_TRAIN_WINDOWS, DEV_WINDOWS = previous.TRAIN_WINDOWS, previous.SCORE_TRAIN_WINDOWS, previous.DEV_WINDOWS
SEQUENCE, PLAN = previous.SEQUENCE, previous.PLAN
WIDTH, VOCAB = 512, 128
ARMS = ("affine_preconditioned",)
SCHEMA = "tikitorch.fpga_lm_readout_preconditioned.v1"
PRODUCER_SHA256 = "d893d1e5f6be61e37eb747f6c1be2d110145f56019047ac51c147092da7db3de"
SEAL_SHA256 = "7d93056f9b6558b96c76ad1a9e8498675368642b6d1deda51a635fdb508836f1"
MANIFEST_SHA256 = "a92d69402255fe99d57f2ab9dbf7b1b907f37e3b431be7dafee80607ac72b7bd"
AUDIT_SHA256 = "09c9390c2a051163ada61dd3760310c1a8462389362723aa744159242e86221a"
AUDIT_MANIFEST_SHA256 = "a1642db369a963b3ace60f40c40ec152f54eb5e71773d6741def42e008fb7466"
DECLARATION_SHA256 = "b083c16d7885fe33a27e51ca49e18c3f17351cf003fa227a183dbcaefd2c45ab"


def source_identity():
    values = dict(previous.source_identity())
    for name in ("manuscript/tools/fpga_lm_readout_preconditioned.py",
                 "manuscript/tools/diagnose_fpga_lm_readout_preconditioned.py",
                 "manuscript/tools/fpga_lm_readout_preconditioned_validation.py",
                 "tests/test_fpga_lm_readout_preconditioned.py"):
        values[name] = old.digest((REPO / name).read_bytes())
    return values


def recipe():
    return {"schema": SCHEMA, "arms": list(ARMS), "lambda": 0.001,
            "fit_windows": list(TRAIN_WINDOWS), "sampled_training_windows": list(SCORE_TRAIN_WINDOWS),
            "development_windows": list(DEV_WINDOWS), "sequence": SEQUENCE,
            "preconditioner": "X=[h,1]; M=.001I+X.T@X/(2N); M=L@L.T; S=L^(-T)",
            "coordinates": "A=[W,b]=U@S.T; physical theta concatenates W.ravel() then b; G_U=G_A@S",
            "objective": "unchanged mean CE(z0+h@W.T+b,y)+.001/2*(sum(W**2)+sum(b**2))",
            "feature_source": "sealed independently audited frozen-core context-ON readout cache; no feature reconstruction",
            "initialization": "exact positive-zero U and physical correction, not prior capped endpoint",
            "solver": "SciPy L-BFGS-B; maxcor10/maxls20/ftol0/gtol0; custom ORIGINAL-gradient stopping",
            "max_iterations": 200, "max_function_calls": 400, "solver_seconds": 3600,
            "numerical_gap_target": 1e-5, "numerical_gap_estimate": "physical gradient squared norm/(2*.001), not a formal rounding certificate",
            "chunk_size": 2048, "torch_threads": 1, "blas_threads": 1,
            "gram_scope": "one training-only augmented Gram construction; fixed factor, no sweep or damping search",
            "gram_cost": "reported separately from full objective/gradient evaluations and solve time",
            "producer_training_gram_passes": 4, "independent_training_gram_passes": 1,
            "gram_pass_accounting": "one construction, one pre-solve validation, two saved-output checks; independent audit adds one Torch Gram",
            "factor_checks": "64*(D+1)*eps normalized backward residuals; inverse absolute residual<=1e-6; independent Gram atol1e-9/rtol1e-11",
            "independent_audit": "Torch training Gram/factor residuals and coordinate maps/pullbacks; initial/final original objective+autograd gradient+128 scores each",
            "optimizer_trajectory_replayed": False, "formal_optimization_certificate": False,
            "new_backbone_forwards": 0, "backbone_gradient_queries": 0, "backbone_optimizer_steps": 0,
            "bias_refitted": False, "new_distinct_training_pairs": 0,
            "deadline_seconds": 10800, "combined_output_cap_bytes": 8 * (1 << 30), "free_reserve_bytes": 4 * (1 << 30),
            "heldout_test_evaluated": False, "reserved_development_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False,
            "scope": "same-objective bounded conditioning comparison, not an FPGA-compatible readout or matched wall-time benchmark"}


def runtime():
    import numpy.linalg.linalg as numpy_python
    import numpy.linalg._umath_linalg as numpy_binary
    import scipy.linalg._basic as basic
    import scipy.linalg._flapack as lapack

    return {**previous.runtime(), "factor_implementation": {
        "python": old.file_identity(basic.__file__), "lapack": old.file_identity(lapack.__file__),
        "numpy_python": old.file_identity(numpy_python.__file__), "numpy_binary": old.file_identity(numpy_binary.__file__)}}


@dataclass(frozen=True)
class Validation:
    binding: dict
    sources: dict
    predecessor_binding: dict
    predecessor_validation: object

    def __post_init__(self):
        for name in ("binding", "sources", "predecessor_binding"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))


def _pinned(path, digest, label):
    path = parent.rate.head._regular(path)
    descriptor = old.file_identity(path)
    equal(descriptor["sha256"], digest, label)
    return descriptor


def bind_inputs(config):
    """File-only binding to a completed proof, not a re-execution of that proof."""
    sources = source_identity()
    root = Path(config.prior_run).resolve(strict=True)
    record = _pinned(config.prior_record, PRODUCER_SHA256, "actual completed readout publication")
    manifest_pin = _pinned(root / "manifest.json", MANIFEST_SHA256, "actual readout manifest")
    seal = _pinned(root / "artifact_manifest.json", SEAL_SHA256, "actual completed cache seal")
    declaration = _pinned(root / "declaration.md", DECLARATION_SHA256, "historical declaration, not current PLAN")
    manifest, summary = read(manifest_pin["path"]), read(root / "summary.json")
    equal(manifest["schema"], previous.SCHEMA, "predecessor schema")
    equal(manifest["run_directory"], str(root), "actual predecessor root")
    equal(manifest["record"], record["path"], "actual predecessor publication path")
    equal(manifest["source_hashes"], previous.source_identity(), "all 82 historical sources unchanged")
    equal(manifest["recipe"], previous.recipe(), "actual predecessor objective/cohorts")
    equal(previous.runtime(), manifest["runtime"], "same physical-objective runtime as historical controls")
    equal(manifest["declaration_sha256"], declaration["sha256"], "saved predecessor declaration binding")
    artifacts = previous.inventory(root)
    equal(read(record["path"]), previous.public_record(root, summary, artifacts), "predecessor sealed payload/public equivalence")
    equal(summary["complete"], True, "complete predecessor producer")
    equal(summary["binding"], manifest["binding"], "predecessor input ancestry")
    equal(summary["source_hashes"], manifest["source_hashes"], "predecessor summary sources")
    audit = _pinned(config.prior_audit, AUDIT_SHA256, "actual successful independent cache audit")
    receipt = read(audit["path"])
    audit_root = Path(receipt["audit_directory"]).resolve(strict=True)
    audit_manifest = _pinned(audit_root / "audit_manifest.json", AUDIT_MANIFEST_SHA256, "actual audit admission manifest")
    audit_declaration = _pinned(audit_root / "declaration.md", DECLARATION_SHA256, "same saved historical audit declaration")
    verification = _pinned(audit_root / "verification.json", AUDIT_SHA256, "exact private/public audit bytes")
    equal({p.relative_to(audit_root).as_posix() for p in audit_root.rglob("*")},
          {"audit_manifest.json", "declaration.md", "verification.json"}, "exact completed three-file audit capsule")
    expected = {"schema": previous.SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                "producer_public_sha256": PRODUCER_SHA256, "producer_seal_sha256": SEAL_SHA256,
                "audit_manifest_sha256": AUDIT_MANIFEST_SHA256, "run_directory": str(root), "record": record["path"],
                "source_hashes": manifest["source_hashes"], "runtime": manifest["runtime"],
                "independent_dense_feature_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
                "supplemental_guarded_feature_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
                "independent_readout_full_gradients": 2 * len(previous.ARMS),
                "independent_initial_readout_score_windows": len(previous.ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                "independent_final_readout_score_windows": len(previous.ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                "initial_function_parity_checks": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS),
                "source_input_raw_seal_public_bytes_unchanged": True,
                "optimizer_trajectory_replayed": False, "formal_optimization_certificate": False,
                "new_training_updates": 0, "backbone_gradient_queries": 0,
                "heldout_test_evaluated": False, "reserved_development_evaluated": False,
                "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False}
    for name, value in expected.items():
        equal(receipt[name], value, "completed cache audit semantics: " + name)
    private = read(audit_manifest["path"])
    for name in ("run_directory", "record", "source_hashes", "runtime", "producer_public_sha256", "producer_seal_sha256"):
        equal(private[name], receipt[name], "audit capsule cross binding: " + name)
    equal(private["audit_record"], audit["path"], "actual predecessor audit public path")
    equal(private["audit_directory"], str(audit_root), "actual predecessor audit private path")
    equal(private["declaration_sha256"], DECLARATION_SHA256, "old declaration remains immutable")
    ancestral, raw, token = previous.bind_inputs(previous._config(manifest["binding"]))
    equal(ancestral, manifest["binding"], "fresh canonical core/checkpoint/corpus ancestry")
    corpus = old.file_identity(parent.rate.head._regular(config.corpus))
    equal(corpus, old.file_identity(ancestral["core"]["dataset"]["path"]), "same caller corpus bytes and path")
    binding = {"prior_run": str(root), "prior_record": record["path"], "prior_audit": audit["path"],
               "corpus": corpus["path"], "dataset": corpus,
               "files": {"manifest": manifest_pin, "record": record, "seal": seal, "declaration": declaration,
                         "audit": audit, "audit_manifest": audit_manifest, "audit_declaration": audit_declaration,
                         "verification": verification},
               "prior_payload_count": len(artifacts), "prior_payload_bytes": sum(r["bytes"] for r in artifacts)}
    equal(source_identity(), sources, "no source adoption during binding")
    return binding, raw, Validation(binding, sources, ancestral, token)


def frozen(binding, validation):
    if not isinstance(validation, Validation):
        raise ValueError("invocation-local preconditioning validation required")
    equal(binding, validation.binding, "unchanged preconditioning inputs")
    equal(source_identity(), validation.sources, "all frozen project sources")
    previous.frozen(validation.predecessor_binding, validation.predecessor_validation)
    for descriptor in (*binding["files"].values(), binding["dataset"]):
        equal(_pinned(descriptor["path"], descriptor["sha256"], "unchanged regular inherited file"),
              descriptor, "all actual inherited proof bytes unchanged")
    audit_root = Path(binding["files"]["audit_manifest"]["path"]).parent
    equal({p.relative_to(audit_root).as_posix() for p in audit_root.rglob("*")},
          {"audit_manifest.json", "declaration.md", "verification.json"}, "inherited audit capsule remains exact")
    previous.inventory(binding["prior_run"])


def _config(binding):
    return argparse.Namespace(**{name: binding[name] for name in ("prior_run", "prior_record", "prior_audit", "corpus")})


def collect_cache(binding, raw, clock):
    root, result = Path(binding["prior_run"]), {}
    manifest = read(root / "manifest.json")
    dataset = manifest["binding"]["core"]["dataset"]
    summary = read(root / "summary.json")
    rows = []
    for cohort, indices in (("train", TRAIN_WINDOWS), ("development", DEV_WINDOWS)):
        collected = {key: [] for key in ("h", "logits", "targets")}
        for index in indices:
            clock.check()
            path = previous.cache_path(root, cohort, index)
            arrays = load(path)
            equal(set(arrays), set(previous.helper.TRACE_KEYS) | {"h", "tokens", "targets", "guard_counts"}, "exact inherited cache fields")
            tokens, targets = parent.batch_data(raw, dataset, (index,), "train" if cohort == "train" else "validation")
            guard.byte_equal({"tokens": arrays["tokens"], "targets": arrays["targets"]},
                             {"tokens": tokens[0], "targets": targets[0]}, "inherited cache exact data alignment")
            h, z, y = (arrays[key].numpy() for key in ("h", "logits", "targets"))
            equal(previous.oracle.validate_data(h, z, y), (SEQUENCE, WIDTH, VOCAB), "original finite cached geometry")
            for key in collected:
                collected[key].append(arrays[key].numpy())
            rows.append({"cohort": cohort, "window": index, "sha256": old.digest(path.read_bytes())})
        result[cohort] = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    equal(rows, summary["cache"]["files"], "all exact audited cache windows reused")
    return result


def expected_artifacts():
    return {"manifest.json", "declaration.md", "summary.json", "transform.npz", "transform.json",
            "initial.npz", "final.npz", "solver.json"}


def inventory(root):
    root = Path(root)
    artifacts = prior.artifact_manifest(root)
    equal({row["file"] for row in artifacts}, expected_artifacts(), "exact preconditioning payload membership")
    equal(read(root / "artifact_manifest.json"), {"schema": SCHEMA, "files": artifacts}, "all sealed payload hashes")
    equal([p for p in root.rglob("*") if p.is_dir()], [], "flat private run, no extra directories")
    return artifacts


def public_record(root, summary, artifacts):
    return {**summary, "run_directory": str(root), "artifact_files_verified": len(artifacts),
            "artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
            "artifact_manifest_sha256": old.digest((Path(root) / "artifact_manifest.json").read_bytes()),
            "independent_audit_required": True}


def execute(root, binding, raw, *, storage=None, independent=False, clock=None):
    root = Path(root)
    cache = collect_cache(binding, raw, clock)
    data = cache["train"]
    if storage is not None:
        started_utc, started_mono = time.time(), time.monotonic()
        transform = helper.build_preconditioner(data["h"], checkpoint=clock.check)
        construction = {"summary": transform["summary"],
                        "construction_elapsed_utc_seconds": time.time() - started_utc,
                        "construction_elapsed_monotonic_seconds": time.monotonic() - started_mono}
        storage.arrays(root / "transform.npz", {key: torch.from_numpy(transform[key]) for key in ("M", "L", "S")})
        storage.json(root / "transform.json", construction)
        print(json.dumps({"stage": "training_gram_complete", **construction}), flush=True)
        fit = helper.solve(data["h"], data["logits"], data["targets"], preconditioner=transform, checkpoint=clock.check)
        for name, values in (("initial", fit["initial"]), ("final", fit)):
            arrays = {"U": values["U"], "theta": values["theta"],
                      "gradient": values["physical_gradient"], "transformed_gradient": values["coordinate_gradient"]}
            storage.arrays(root / (name + ".npz"), {key: torch.from_numpy(value) for key, value in arrays.items()})
        storage.json(root / "solver.json", {key: fit[key] for key in ("summary", "history", "iteration_history")})
    construction = read(root / "transform.json")
    equal(set(construction), {"summary", "construction_elapsed_utc_seconds", "construction_elapsed_monotonic_seconds"}, "complete construction receipt")
    for name in ("construction_elapsed_utc_seconds", "construction_elapsed_monotonic_seconds"):
        if type(construction[name]) is not float or not np.isfinite(construction[name]) or construction[name] < 0:
            raise ValueError("finite nonnegative actual construction clock required")
    factors = load(root / "transform.npz")
    equal(set(factors), {"M", "L", "S"}, "complete saved fixed factor")
    transform = {**{key: value.numpy() for key, value in factors.items()}, "summary": construction["summary"]}
    helper.verify_preconditioner(data["h"], transform, independent=independent, checkpoint=clock.check)
    solver = read(root / "solver.json")
    endpoints, archives, scores = {}, {}, {}
    for name in ("initial", "final"):
        clock.check()
        arrays = load(root / (name + ".npz"))
        equal(set(arrays), {"U", "theta", "gradient", "transformed_gradient"}, "complete coordinate and physical endpoint")
        result = helper.objective(arrays["U"].numpy(), data["h"], data["logits"], data["targets"],
                                  transform, independent=independent, checkpoint=clock.check)
        near({"theta": arrays["theta"], "gradient": arrays["gradient"], "transformed_gradient": arrays["transformed_gradient"]},
             {"theta": torch.from_numpy(result["theta"]), "gradient": torch.from_numpy(result["physical_gradient"]),
              "transformed_gradient": torch.from_numpy(result["coordinate_gradient"])}, "original-coordinate endpoint and pullback")
        endpoints[name] = {key: value for key, value in result.items()
                           if key not in {"theta", "physical_gradient", "coordinate_gradient", "transformed_gradient_norm_sq"}}
        archives[name] = arrays
        scores[name] = previous.score(arrays["theta"].numpy(), cache, "affine", independent=independent)
    validator.validate_solver_record(solver, endpoints, archives, data, transform)
    predecessor = read(Path(binding["prior_run"]) / "summary.json")
    controls = previous.channel.controls(predecessor["binding"]["core"])
    equal(scores["initial"], predecessor["arms"]["affine"]["initial_scores"], "same positive-zero original head scores on all 128 windows")
    references = {"unpreconditioned_affine": predecessor["arms"]["affine"]["after"],
                  "bias": predecessor["arms"]["bias"]["after"], "frozen_core": scores["initial"],
                  "gru": controls["prior_gru"], "original_mixed": controls["mixed"]}
    comparisons = {"new_minus_" + name: previous.paired(scores["final"], value) for name, value in references.items()}
    progress = {cohort: comparisons["new_minus_frozen_core"][cohort]["ce_difference"] <= -0.01
                for cohort in ("train", "development")}
    summary = solver["summary"]
    result = {"schema": SCHEMA, "complete": True, "recipe": recipe(), "source_hashes": source_identity(),
              "binding": binding, "construction": construction, "solver": solver, "endpoints": endpoints,
              "initial_scores": scores["initial"], "after": scores["final"], **comparisons,
              "controls": {name: {cohort: value[cohort]["mean"] for cohort in ("train", "development")}
                           for name, value in references.items()},
              "regularized_objective_minus_unpreconditioned_affine": endpoints["final"]["objective"] - predecessor["arms"]["affine"]["endpoints"]["final"]["objective"],
              "solver_complete_labeled_position_evaluations": summary["complete_function_evaluations"] * len(data["targets"]),
              "solver_attempted_labeled_position_evaluations_upper_bound": summary["function_calls"] * len(data["targets"]),
              "producer_endpoint_verification_full_gradients": 4, "independent_endpoint_audit_full_gradients": 2,
              "new_backbone_forwards": 0, "backbone_gradient_queries": 0, "backbone_optimizer_steps": 0,
              "inherited_audited_cache_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS), "new_distinct_training_pairs": 0,
              "material_progress": {"per_cohort": progress, "both_cohorts": all(progress.values()),
                                    "numerically_completed": summary["numerically_converged"],
                                    "requires_successful_independent_audit": True,
                                    "scope": "descriptive next-review flag, not confirmation or promotion"},
              "optimizer_trajectory_replayed": False, "formal_optimization_certificate": False,
              "heldout_test_evaluated": False, "reserved_development_evaluated": False,
              "physical_fpga_execution": False, "paid_execution_authorized": False, "recipe_promoted": False}
    print(json.dumps({"stage": "preconditioned_readout_complete", "independent": independent,
                      "scores": {key: value["mean"] for key, value in scores["final"].items()},
                      "numerical_gap_estimate": endpoints["final"]["numerical_gap_estimate"],
                      "stopping_reason": summary["stopping_reason"]}), flush=True)
    return result


def _invoke(config, independent=False):
    from threadpoolctl import threadpool_limits
    clock, publication, output = Deadline(), None, None
    created, closed, cleanup_done = False, False, False
    stage = "setup"
    try:
        clock.arm()
        torch.set_num_threads(1)
        runtime()  # Load SciPy/BLAS before limiting every discovered thread pool.
        with threadpool_limits(limits=1):
            output, target = parent._new_outputs(config.audit_output, config.audit_record) if independent else parent._new_outputs(config.output, config.record)
            input_root = Path(config.output if independent else config.prior_run).resolve(strict=True)
            for new_path in (output, target):
                if new_path == input_root or input_root in new_path.parents or new_path in input_root.parents:
                    raise ValueError("new outputs must be disjoint from the immutable input run")
                for ancestor in new_path.parents:
                    if (ancestor / "artifact_manifest.json").exists() or (ancestor / "audit_manifest.json").exists():
                        raise ValueError("new outputs cannot be nested inside any sealed run or audit capsule")
            sources, declaration = source_identity(), PLAN.read_bytes()
            output.mkdir(parents=True, exist_ok=False)
            created = True
            storage = Storage(output, clock)
            root = Path(config.output).resolve(strict=True) if independent else output
            print(json.dumps({"stage": "input_validation", "independent": independent}), flush=True)
            stage = "input_validation"
            if independent:
                record = parent.rate.head._regular(config.record)
                manifest = read(root / "manifest.json")
                equal(manifest["schema"], SCHEMA, "exact producer manifest schema")
                equal(manifest["run_directory"], str(root), "actual producer root")
                equal(manifest["record"], str(record), "actual producer public path")
                equal(old.digest((root / "declaration.md").read_bytes()), manifest["declaration_sha256"], "producer saved declaration pin")
                equal((root / "declaration.md").read_bytes(), declaration, "same frozen declaration across producer and audit")
                equal(manifest["source_hashes"], sources, "same producer source closure")
                equal(manifest["recipe"], recipe(), "exact readout recipe")
                equal(manifest["runtime"], runtime(), "same readout runtime")
                artifacts = inventory(root)
                original = record.read_bytes()
                summary = read(root / "summary.json")
                originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json", "declaration.md")}
                equal(read(record), public_record(root, summary, artifacts), "producer publication exactly binds sealed data")
                storage.base_bytes = sum(row["bytes"] for row in artifacts) + (root / "artifact_manifest.json").stat().st_size + len(original)
                storage.used = storage.base_bytes
                binding, raw, validation = bind_inputs(_config(manifest["binding"]))
                equal(binding, manifest["binding"], "fresh canonical readout binding")
                private = {"schema": SCHEMA, "mode": "independent_endpoint_audit", "complete": False,
                           "producer_public_sha256": old.digest(original), "producer_seal_sha256": old.digest((root / "artifact_manifest.json").read_bytes()),
                           "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                           "audit_record": str(target), "source_hashes": sources, "runtime": runtime(),
                           "declaration_sha256": old.digest(declaration), "started_at_utc": guard.parent._utc(clock.started_utc)}
                storage.json(output / "audit_manifest.json", private)
                private_bytes = (output / "audit_manifest.json").read_bytes()
            else:
                binding, raw, validation = bind_inputs(config)
                manifest = {"schema": SCHEMA, "recipe": recipe(), "source_hashes": sources, "binding": binding,
                            "runtime": runtime(), "run_directory": str(root), "record": str(target),
                            "declaration_sha256": old.digest(declaration), "started_at_utc": guard.parent._utc(clock.started_utc)}
                storage.json(root / "manifest.json", manifest)
                pass  # All eight new payloads are flat; reuse the sealed predecessor cache.
            equal(source_identity(), sources, "no source relabeling during invocation admission")
            equal(validation.sources, sources, "validation binds exact pre-admission sources")
            storage.reserve(output / "declaration.md", len(declaration))
            with (output / "declaration.md").open("xb") as stream:
                stream.write(declaration)
            storage.used += len(declaration)
            frozen(binding, validation)
            stage = "independent_endpoint_audit" if independent else "frozen_feature_fit"
            result = execute(root, binding, raw, storage=None if independent else storage, independent=independent, clock=clock)
            if independent:
                # Endpoint gradients are independently checked at declared tolerance;
                # all score/gradient-derived scalar fields are compared numerically.
                compare_results(result, summary)
                utc, monotonic = clock.check()
                published = {"schema": SCHEMA + ".audit", "complete": True, "independent_recompute_completed": True,
                             **{key: private[key] for key in ("producer_public_sha256", "producer_seal_sha256", "run_directory", "record", "audit_directory", "source_hashes", "runtime", "started_at_utc")},
                             "audit_manifest_sha256": old.digest((output / "audit_manifest.json").read_bytes()),
                             "verification_completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                             "verification_elapsed_utc_seconds": utc, "verification_elapsed_monotonic_seconds": monotonic,
                             "independent_dense_feature_windows": 0,
                             "supplemental_guarded_feature_windows": 0,
                             "inherited_audited_cache_windows": len(TRAIN_WINDOWS) + len(DEV_WINDOWS),
                             "independent_training_gram_checks": 1,
                             "independent_coordinate_endpoint_checks": 2,
                             "independent_readout_full_gradients": 2 * len(ARMS),
                             "independent_initial_readout_score_windows": len(ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                             "independent_final_readout_score_windows": len(ARMS) * (len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS)),
                             "initial_function_parity_checks": len(SCORE_TRAIN_WINDOWS) + len(DEV_WINDOWS),
                             "optimizer_trajectory_replayed": False, "formal_optimization_certificate": False,
                             "new_training_updates": 0, "backbone_gradient_queries": 0,
                             "source_input_raw_seal_public_bytes_unchanged": True,
                             "heldout_test_evaluated": False, "reserved_development_evaluated": False,
                             "physical_fpga_execution": False, "paid_execution_authorized": False,
                             "recipe_promoted": False}
                storage.json(output / "verification.json", published)
                verification_bytes = (output / "verification.json").read_bytes()
            else:
                storage.json(root / "summary.json", result)
                artifacts = prior.artifact_manifest(root)
                equal({row["file"] for row in artifacts}, expected_artifacts(), "exact payloads before sealing")
                storage.json(root / "artifact_manifest.json", {"schema": SCHEMA, "files": artifacts})
                originals = {name: (root / name).read_bytes() for name in ("manifest.json", "artifact_manifest.json", "summary.json", "declaration.md")}
                stage = "ordinary_endpoint_readback"
                readback = execute(root, binding, raw, clock=clock)
                equal(readback, result, "exact saved endpoint/cache/metric readback")
                published = public_record(root, result, inventory(root))
            storage.reconcile()
            stage = "publication"
            publication = parent._Publication(target, output, storage)
            publication.publish(published)
            frozen(binding, validation)
            equal(runtime(), manifest["runtime"], "runtime and optimizer implementation unchanged")
            equal(inventory(root), artifacts, "all sealed producer bytes unchanged")
            for name, payload in originals.items():
                equal((root / name).read_bytes(), payload, "exact original producer metadata bytes")
            if independent:
                equal(record.read_bytes(), original, "producer publication unchanged during audit")
                equal((output / "audit_manifest.json").read_bytes(), private_bytes, "exact original audit manifest bytes")
                equal((output / "verification.json").read_bytes(), verification_bytes, "exact original verification bytes")
                equal((output / "verification.json").read_bytes(), target.read_bytes(), "exact private/public audit receipt")
                equal({path.name for path in output.iterdir()}, {"audit_manifest.json", "declaration.md", "verification.json"}, "exact audit capsule")
            equal(PLAN.read_bytes(), declaration, "protocol remains frozen throughout invocation")
            publication.verify()
            publication.cleanup()
            cleanup_done = True
            clock.close()
            closed = True
            utc, monotonic = clock.check()
            print(json.dumps({"stage": "complete", "independent": independent,
                              "completed_at_utc": guard.parent._utc(clock.started_utc + utc),
                              "elapsed_utc_seconds": utc, "elapsed_monotonic_seconds": monotonic,
                              "record_sha256": publication.digest}), flush=True)
            return published
    except BaseException as error:
        if publication is not None:
            publication.retract()
        if created:
            rejected = output / "rejected_public.json"
            payload = {"schema": SCHEMA, "complete": False, "stage": stage, "independent": independent,
                       "error": str(error)[:100000], "exception_type": type(error).__name__, "time_utc": time.time(),
                       "retained_rejected_public": old.file_identity(rejected) if rejected.is_file() and not rejected.is_symlink() else None,
                       "recipe_promoted": False, "paid_execution_authorized": False}
            try:
                with (output / "failure.json").open("x") as stream:
                    json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            except OSError:
                pass
        raise
    finally:
        try:
            if publication is not None and not cleanup_done:
                publication.cleanup()
        finally:
            if not closed:
                clock.close()


def run(config):
    return _invoke(config, False)


def audit(config):
    return _invoke(config, True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    for name in ("output", "record"):
        parser.add_argument("--" + name, required=True)
    for name in ("prior-run", "prior-record", "prior-audit", "corpus", "audit-output", "audit-record"):
        parser.add_argument("--" + name)
    config = parser.parse_args(argv)
    required = ("audit_output", "audit_record") if config.audit else ("prior_run", "prior_record", "prior_audit", "corpus")
    if any(getattr(config, name) is None for name in required):
        parser.error("all input/output paths for the selected mode are required")
    audit(config) if config.audit else run(config)


if __name__ == "__main__":
    main()
