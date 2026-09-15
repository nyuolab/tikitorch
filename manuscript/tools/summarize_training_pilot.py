"""Render a pinned, tiny single-seed CPU training pilot, using only the stdlib.

This is a complete-step timing observation, not a task-quality, tensor-peak-memory,
or general speedup result. The ten measured arm/policy cells are all shown. Raw
evidence is never rewritten; changing the selected run requires explicit review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics


REPO = Path(__file__).resolve().parents[2]
RUN_ID = "172834f0-c1cf-42c1-ab87-ec0cdf6f0da9"
DEFAULT_RECORD = REPO / f"manuscript/results/route_certificates/runs/device_route_training/{RUN_ID}/manifest.json"
GENERATED_TABLE = REPO / "manuscript/source/generated/training_pilot.tex"
SOURCE_SHA256 = "aad9d1c28ccd3d6c4fc9bf1326cf701ac38671d4af0a4afb6dbfb429cc279c1f"
FAILURE_RUN_ID = "907579f2-5b1c-42a5-93b0-0d1867e8577e"
FAILURE_RECORD = REPO / f"manuscript/results/route_certificates/runs/compiled_selective_followup/{FAILURE_RUN_ID}/manifest.json"
FAILURE_SHA256 = "cbadf238193a0c3e46f714e18d359b0c78c4d49f7e146506c55c877f0d768783"
HELPERS = (
    "scripts/route_certificates/training_benchmark.py",
    "scripts/route_certificates/actual_model_benchmark.py",
    "scripts/route_certificates/provenance.py",
)
CONFIG = dict(device="cpu", dtype="float32", width=16, layers=1, vocab=11,
              seed=0, lr=0.001, optimizer="adamw", adamw_eps=1e-4, warmup=2,
              reps=5, compile_backend="inductor", batch=2, sequence=12)
POLICIES = ("layer", "grouped", "selective")
CELLS = (("eager", None), ("compiled", None),
         *(("checkpoint", policy) for policy in POLICIES),
         *(("compiled_checkpoint", policy) for policy in POLICIES),
         ("unpacked", None), ("packed", None))
PARAMETERS = {
    "embed.weight", "output_head.weight", "output_head.bias", "ln_final.beta",
    "layers.0.norm.beta", "layers.0.morph_conv.weight", "layers.0.morph_conv.bias",
    "layers.0.out_proj.weight", "layers.0.out_proj.bias", "layers.0.post_norm.beta",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f"invalid {name}")
    return value


def read_record(path=DEFAULT_RECORD, expected_sha256=SOURCE_SHA256):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256,
            "training pilot evidence digest mismatch")
    record = json.loads(raw)
    require(isinstance(record, dict), "malformed training pilot manifest")
    return record


def check_sources(record, root=REPO):
    """Map original absolute inputs to this checkout; reject omissions or escapes."""
    root = Path(root).resolve()
    inputs = record["inputs"]
    require(isinstance(inputs, list) and inputs, "missing source inputs")
    paths = []
    for item in inputs:
        path = item["path"]
        require(isinstance(path, str), "invalid source input path")
        parsed = PurePosixPath(path)
        require(parsed.is_absolute() and ".." not in parsed.parts
                and "\\" not in path and str(parsed) == path, "unsafe source input path")
        paths.append(parsed)
    suffix = PurePosixPath(HELPERS[0]).parts
    runners = [path for path in paths if path.parts[-len(suffix):] == suffix]
    require(len(runners) == 1, "ambiguous source runner/root")
    original_root = runners[0].parents[len(suffix) - 1]
    expected = set(HELPERS) | {path.relative_to(root).as_posix()
                               for path in (root / "tikitorch").rglob("*.py")}
    seen = set()
    for item, path in zip(inputs, paths):
        require(path.is_relative_to(original_root), "source input outside original root")
        relative = path.relative_to(original_root).as_posix()
        require(relative not in seen, "duplicate source input")
        require(relative in expected, f"unexpected source input: {relative}")
        seen.add(relative)
        current = root / relative
        require(current.resolve().is_relative_to(root), "source input resolves outside checkout")
        data = current.read_bytes()
        require(type(item["bytes"]) is int and item["bytes"] == len(data)
                and hashlib.sha256(data).hexdigest() == item["sha256"],
                f"stale current source: {relative}")
    require(seen == expected, "incomplete source input set")


def check_failure(record):
    """Preserve an inconclusive deeper native-policy crash, never as a timing row."""
    require(record["schema"] == "tikitorch.route-certificates.run.v1"
            and record["experiment"] == "compiled_selective_followup"
            and record["run_id"] == FAILURE_RUN_ID, "unexpected failure identity")
    require(record["incident_status"] == "inconclusive" and record["device"] == "cpu",
            "failure must remain inconclusive")
    config = {**CONFIG, "layers": 2, "optimizer": "sgd", "adamw_eps": 1e-8,
              "warmup": 0, "reps": 1}
    require(record["config"] == dict(config=config,
            scope="isolated native optional-policy crash diagnostic, not a timing comparison"),
            "unexpected failure configuration/scope")
    metrics = record["metrics"]
    require(set(metrics) == {"row", "worker_outcome"}, "unexpected failure metrics")
    row, worker = metrics["row"], metrics["worker_outcome"]
    require(set(row) == {"status", "arm", "checkpoint_policy", "config", "error", "worker_outcome"}
            and row["status"] == "error" and row["arm"] == "compiled_checkpoint"
            and row["checkpoint_policy"] == "selective" and row["config"] == config
            and row["error"] == "worker exited before producing a row",
            "failure cannot supply successful or measured row")
    require(type(worker["returncode"]) is int and worker["returncode"] == -11
            and worker["stop_reason"] is None and worker["termination"] == {}
            and worker["elapsed_s"] == 5.6 and row["worker_outcome"] == worker,
            "unexpected negative worker outcome")


def selected_rows(record):
    """Derive timings from complete raw cells, never from a selected-winner summary."""
    require(record["schema"] == "tikitorch.route-certificates.run.v1"
            and record["experiment"] == "device_route_training"
            and record["run_id"] == RUN_ID, "unexpected training pilot identity")
    require(record["incident_status"] == "completed" and record["device"] == "cpu",
            "incomplete or non-CPU pilot")
    require(record["config"] == dict(cells=[CONFIG], learning_steps=50,
                                     protocol="device-route-training-v2"),
            "unexpected training pilot configuration")
    metrics = record["metrics"]
    checks = metrics["checks"]
    require(checks["sources_unchanged"] is True, "sources changed during pilot")
    require("error" not in checks, "failed auxiliary checks")
    trajectory = checks["trajectory"]
    require(trajectory["status"] == "pass" and trajectory["steps"] == 5,
            "failed trajectory check")
    require(finite(trajectory["max_grad_error"], "trajectory gradient") < 1e-4
            and finite(trajectory["max_param_error"], "trajectory update") < 1e-5,
            "failed trajectory metric")
    learning = checks["learning"]
    require(learning["status"] == "pass"
            and learning["task"] == "synthetic token successor; regression only",
            "unexpected learning-control status/scope")
    optimizer = learning["optimizer"]
    require(all(optimizer[key] == value for key, value in dict(
        name="AdamW", lr=0.01, eps=1e-4, betas=[0.9, 0.999],
        weight_decay=0.01, execution="eager").items()), "learning optimizer mismatch")
    require(set(learning["models"]) == {"route_morphssm", "dense_token_baseline"},
            "incomplete learning controls")
    for result in learning["models"].values():
        before, after = result["before"], result["after"]
        require(result["updates"] == 50
                and finite(after["loss"], "learning loss")
                < 0.75 * finite(before["loss"], "initial learning loss", positive=True)
                and 0.8 < finite(after["accuracy"], "learning accuracy") <= 1,
                "failed synthetic learning gate")
    summaries = metrics["summaries"]
    require(len(summaries) == 1 and summaries[0]["config"] == CONFIG
            and summaries[0]["status"] == "complete"
            and summaries[0]["matched_initialization_data_updates"] is True
            and summaries[0]["missing_arms"] == [], "incomplete pilot summary")
    rows = metrics["rows"]
    require(isinstance(rows, list) and len(rows) == len(CELLS), "incomplete pilot cells")
    by_cell = {}
    for row in rows:
        cell = (row["arm"], row["checkpoint_policy"])
        require(cell in CELLS and cell not in by_cell, "unexpected or duplicate pilot cell")
        by_cell[cell] = row
        require(row["config"] == CONFIG and row["update_count"] == 8,
                "unmatched cell configuration/update count")
        require(row["status"] == "ok" and row["measurement_status"] == "fit",
                "failed pilot cell")
        worker = row["worker_outcome"]
        require(type(worker["returncode"]) is int and worker["returncode"] == 0
                and worker["stop_reason"] is None and worker["termination"] == {},
                "unsuccessful worker outcome")
        compiled = row["arm"] in ("compiled", "compiled_checkpoint", "unpacked", "packed")
        require(row["compile_fullgraph"] is compiled
                and row["compile_backend"] == ("inductor" if compiled else None)
                and row["optimizer_execution"] == "eager", "compiled/optimizer scope mismatch")
        context = row["model_context"]
        require(context["convolution_layers"] == [dict(kernel_size=3, dilation=1)]
                and context["causal_receptive_field"] == 3, "incorrect causal context")
        require(all(row[key] is None for key in ("cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes", "cpu_tensor_peak_bytes")), "unexpected tensor-peak claim")
        for key in ("init_sha256", "batch_sha256", "final_params_sha256"):
            require(isinstance(row[key], str) and re.fullmatch(r"[0-9a-f]{64}", row[key]),
                    f"invalid {key}")
        samples = row["samples_ms"]
        require(isinstance(samples, list) and len(samples) == 5, "incorrect timing sample count")
        for sample in samples:
            finite(sample, "timing sample", positive=True)
        median = statistics.median(samples)
        mad = statistics.median(abs(sample - median) for sample in samples)
        require(finite(row["median_ms"], "median", positive=True) == median
                and finite(row["mad_ms"], "MAD") == mad, "inconsistent timing summary")
        finite(row["startup_and_first_update_ms"], "startup time", positive=True)
        finite(row["final_loss"], "final loss")
        parity = row["parity"]
        require(parity["status"] == "pass" and parity["updates_checked"] == 8
                and parity["compared_measured_state"] is True
                and parity["atol"] == 1e-4 and parity["rtol"] == 1e-4,
                "failed measured-state parity")
        require(finite(parity["loss_max_abs"], "loss parity error") <= 1e-4,
                "failed loss parity metric")
        parameters = parity["parameters"]
        require(len(parameters) == len(PARAMETERS)
                and {p["parameter"] for p in parameters} == PARAMETERS,
                "incomplete parameter parity")
        for parameter in parameters:
            # This pinned run satisfies the absolute bound alone. These maxima
            # cannot reconstruct elementwise relative tolerances for other runs.
            require(all(finite(parameter[key], "parameter parity error") <= 1e-4
                        for key in ("gradient_max_abs", "update_max_abs")),
                    "failed parameter parity metric")
    require(set(by_cell) == set(CELLS), "incomplete pilot cells")
    for key in ("init_sha256", "batch_sha256", "update_count"):
        require(len({row[key] for row in rows}) == 1, f"unmatched pilot {key}")
    eager_loss = by_cell[("eager", None)]["final_loss"]
    require(all(abs(row["final_loss"] - eager_loss) == row["parity"]["loss_max_abs"]
                for row in rows), "inconsistent loss parity summary")
    return [by_cell[cell] for cell in CELLS]


def render(rows):
    names = {"eager": "Eager autograd", "compiled": "Compiled autograd",
             "checkpoint": "Checkpoint", "compiled_checkpoint": "Compiled checkpoint",
             "unpacked": "Unpacked routes", "packed": "Packed routes"}
    lines = ["% Generated by manuscript/tools/summarize_training_pilot.py.",
             "% Tiny single-seed CPU pilot; not quality, tensor-peak memory, or general speedup evidence.",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Arm & Policy & Median (ms) & MAD (ms) & Startup (ms) \\", r"\midrule"]
    for row in rows:
        lines.append(f"{names[row['arm']]} & {row['checkpoint_policy'] or '--'} & "
                     f"{row['median_ms']:.3f} & {row['mad_ms']:.3f} & "
                     f"{row['startup_and_first_update_ms']:.1f} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def generate(root=REPO, record_path=DEFAULT_RECORD, failure_path=FAILURE_RECORD):
    record = read_record(record_path)
    failure = read_record(failure_path, FAILURE_SHA256)
    try:
        check_sources(record, root)
        check_sources(failure, root)
        check_failure(failure)
        return render(selected_rows(record))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed training pilot record: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args(argv)
    try:
        table = generate()
        if args.check:
            require(GENERATED_TABLE.read_bytes() == table.encode("utf-8"),
                    "stale generated training pilot table")
        else:
            GENERATED_TABLE.parent.mkdir(parents=True, exist_ok=True)
            GENERATED_TABLE.write_text(table, encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Training pilot evidence check failed: {exc}\n")
    print("Training pilot evidence and table verified." if args.check
          else "Generated training pilot table from pinned raw evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
