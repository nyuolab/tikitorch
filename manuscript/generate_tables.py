"""Render manuscript tables from pinned historical evidence, without rerunning it.

Run ``python manuscript/generate_tables.py --check`` to validate both the raw
artifact identities and checked-in generated outputs. Changing an evidence
source requires an explicit, reviewed update to SOURCE_SHA256, not a silent
regeneration of the paper. Only the standard library is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUN_ID = "529c45f4-7274-4999-b35c-c59d1db917e6"
SAVED = f"manuscript/results/route_certificates/actual_morphssm/exp1_saved_state/{RUN_ID}"
D512 = "manuscript/results/fpga_silicon/d512_verify"
MNIST = "manuscript/results/predictive_coding/onfabric_train"
SOURCE_SHA256 = {
    f"{SAVED}/saved_state.jsonl": "351c15fc09455d953db3f53dc9929af1dd961db9f351c9992ba48ba08d25623b",
    f"{SAVED}/manifest.json": "7f5bf7b657b160dc51c51a563267d3ee77412c9b29ad0cd40f802c55f84bac29",
    f"{D512}/silicon_fused.jsonl": "43bd34508d457b7d5662482aecb13012bf39136dc666fdb42a4f894d026bbeb3",
    f"{D512}/golden_fused.jsonl": "43bd34508d457b7d5662482aecb13012bf39136dc666fdb42a4f894d026bbeb3",
    f"{D512}/train_30.err": "59c863911b018499a3e49cb165e5c50e3fbd8db4b83dac351eb553a6ae407124",
    f"{D512}/train_200.err": "ddda8919bf946b514154bb5efc7e6ec8ad05ed6d37fbca31485b2f112dcc514d",
    f"{D512}/train_2000.err": "b8a5d29e207ad3591ca5639bc4a5094b13a5892b2dd273fa77b3403fcc074e36",
    f"{MNIST}/train.txt": "097bdc7d37dbe141feb90dc03b5d673fb3287eb7bf85e193372196368d86143b",
    f"{MNIST}/crosscheck.txt": "4203d9cff2234bfa9b312ec06e9a94e42165d8a4b5c190560f7fbd93254b7a0a",
}
SCALES = ((64, 2, 96), (128, 4, 96), (256, 4, 192))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_sources(root=REPO):
    sources = {}
    for name, expected in SOURCE_SHA256.items():
        data = (root / name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        require(actual == expected, f"evidence digest mismatch: {name}: {actual}")
        sources[name] = data.decode("utf-8")
    return sources


def retained_records(rows, manifest):
    """Validate exact selected cells; never trust legacy precomputed ratios."""
    require(manifest["run_id"] == RUN_ID, "wrong saved-state run")
    require(manifest["schema"] == "ontogenic_v2.run.v1", "unexpected legacy schema")
    require(manifest["incident_status"] == "completed", "incomplete saved-state run")
    require(manifest["device"] == "cpu", "unexpected saved-state device")
    require(manifest["config"] == {"scales": [list(s) for s in SCALES], "vocab": 38, "batch": 1},
            "unexpected saved-state configuration")
    selected = {}
    for row in rows:
        key = tuple(row[k] for k in ("d_model", "n_layers", "T"))
        require(key in SCALES, f"unexpected saved-state cell: {key}")
        require(key not in selected, f"ambiguous duplicate saved-state cell: {key}")
        selected[key] = row
    require(set(selected) == set(SCALES), "missing saved-state cell")
    require(manifest["metrics"]["n_points"] == len(SCALES), "manifest point count mismatch")
    require(manifest["metrics"]["lifetime_model"] == selected[(128, 4, 96)],
            "manifest lifetime cell differs from raw row")
    records = []
    for key in SCALES:
        row = selected[key]
        width, layers, sequence = key
        tokens = row["n_tok"]
        require(tokens == sequence and row["vocab"] == 38, "boundary shape mismatch")
        logits = tokens * 38 * 4
        targets = tokens * 8  # declared int64 targets; not a separately measured field
        require(row["boundary_logits"] == logits, "FP32 logit boundary mismatch")
        core = (layers + 1) * tokens * width * 4
        require(row["checkpoint_retained"] == core + logits, "checkpoint-set formula mismatch")
        packed = row["packed"]
        payload, serialized = packed["payload_bytes"], packed["serialized_bytes"]
        require(payload == (packed["logical_bits"] + 7) // 8, "payload coding mismatch")
        require(serialized >= payload > 0, "invalid portable certificate byte count")
        records.append(dict(
            width=width, layers=layers, sequence=sequence, batch=1, vocabulary=38,
            logical_bits=packed["logical_bits"], payload_bytes=payload,
            portable_bytes=serialized, checkpoint_core_bytes=core,
            boundary_logit_bytes=logits, boundary_target_bytes=targets,
            supplied_cotangent_ratio=core / serialized,
            loss_inclusive_ratio=(core + logits + targets) / (serialized + logits + targets),
        ))
    return records


def hardware_records(sources):
    silicon = json.loads(sources[f"{D512}/silicon_fused.jsonl"])
    golden = json.loads(sources[f"{D512}/golden_fused.jsonl"])
    require(silicon == golden, "D512 archived observable-record comparison failed")
    require(set(silicon) == {"fused_steps", "acc", "wcrc", "preds16"}, "D512 observable schema changed")
    require(silicon["fused_steps"] == 30 and len(silicon["preds16"]) == 16, "D512 match scope changed")
    d512 = []
    pattern = re.compile(r"silicon FUSED: (\d+) steps in ONE kick, wall ([\d.]+) s, (\d+) tok/s \((\d+) tokens,")
    for steps in (30, 200, 2000):
        matches = pattern.findall(sources[f"{D512}/train_{steps}.err"])
        require(len(matches) == 1, f"ambiguous D512 timing record for {steps}")
        found_steps, wall, displayed_rate, tokens = matches[0]
        require(int(found_steps) == steps and int(tokens) == 128 * steps, "D512 workload mismatch")
        d512.append(dict(updates=steps, processed_positions=int(tokens),
                         wall_seconds=float(wall), reported_integer_tokens_per_second=int(displayed_rate),
                         observable_reference_match=(steps == 30)))
    pattern = re.compile(r"\[train\] epoch (\d+): test acc = ([\d.]+)  \((\d+) kicks, ([\d.]+)s\)")
    matches = pattern.findall(sources[f"{MNIST}/train.txt"])
    require([int(r[0]) for r in matches] == list(range(1, 6)), "ambiguous MNIST epoch records")
    require(all(int(r[2]) == 1875 for r in matches), "MNIST kick count mismatch")
    mnist = [dict(epoch=int(e), driver_test_accuracy=float(a), kicks=int(k),
                  inclusive_epoch_wall_seconds=float(w)) for e, a, k, w in matches]
    matches = re.findall(r"RECIPE-EVAL of ON-FABRIC weights: test acc = ([\d.]+)",
                         sources[f"{MNIST}/crosscheck.txt"])
    require(len(matches) == 1, "ambiguous independent MNIST evaluation")
    return dict(d512=d512, d512_compared_record=silicon, mnist=mnist,
                mnist_independent_test_accuracy=float(matches[0]))


def render_retained(records):
    lines = ["% Generated by manuscript/generate_tables.py; historical portable-tape accounting.",
             r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$d/L/T$ & Payload B & Portable B & Checkpoint B & Cotan. & Loss \\", r"\midrule"]
    for row in records:
        config = f"{row['width']}/{row['layers']}/{row['sequence']}"
        lines.append(f"{config} & {row['payload_bytes']:,} & {row['portable_bytes']:,} & "
                     f"{row['checkpoint_core_bytes']:,} & {row['supplied_cotangent_ratio']:.2f}"
                     r"$\times$ & " + f"{row['loss_inclusive_ratio']:.2f}" + r"$\times$ \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def render_hardware(records):
    lines = ["% Generated from archived D512 host logs; repeated resident 128-position window.",
             r"\begin{tabular}{rrrr}", r"\toprule",
             r"Updates & Positions & Wall time (s) & Reported token/s \\", r"\midrule"]
    for row in records["d512"]:
        lines.append(f"{row['updates']:,} & {row['processed_positions']:,} & "
                     f"{row['wall_seconds']:.4f} & {row['reported_integer_tokens_per_second']:,} " + r"\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def generate_outputs(root=REPO):
    sources = read_sources(root)
    manifest = json.loads(sources[f"{SAVED}/manifest.json"])
    rows = [json.loads(line) for line in sources[f"{SAVED}/saved_state.jsonl"].splitlines() if line.strip()]
    retained = retained_records(rows, manifest)
    hardware = hardware_records(sources)
    provenance = dict(
        schema="tikitorch.manuscript_tables.v1", sources_sha256=SOURCE_SHA256,
        retained_state=dict(
            run_id=RUN_ID, status="historical imported representation evidence; not current device training",
            manifest=f"{SAVED}/manifest.json", command=manifest["command"],
            source_identity={k: manifest[k] for k in (
                "ontogenic_commit", "ontogenic_dirty_patch_sha256", "tikitorch_commit",
                "tikitorch_dirty_patch_sha256", "python", "torch", "platform", "device")},
            accounting=dict(
                checkpoint="declared layer/embedding retained set, not a measured checkpoint allocator peak",
                supplied_cotangent="(checkpoint_retained - boundary_logits) / serialized_bytes",
                loss_inclusive="(checkpoint_core + logits + int64 targets) / (serialized_bytes + logits + int64 targets)",
                exclusions="parameters, optimizer, supplied cotangent, transient workspaces and allocator overhead",
                ignored_legacy_fields="ratio_* fields are not used; they mix boundaries or omit directory bytes"),
            rows=retained),
        hardware=dict(status="historical feasibility; no current-build certification or accelerator comparison",
                      d512_scope="same resident window; 30-update observable record only, not full state",
                      mnist_scope="different evaluator pipelines; not paired-input parity; inclusive wall time",
                      **hardware),
    )
    return {
        "retained_state.tex": render_retained(retained),
        "d512_execution.tex": render_hardware(hardware),
        "table_provenance.json": json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Reject stale output or changed evidence without writing")
    args = parser.parse_args(argv)
    output = REPO / "manuscript/generated"
    try:
        generated = generate_outputs()
        if args.check:
            for name, text in generated.items():
                path = output / name
                require(path.exists() and path.read_text(encoding="utf-8") == text,
                        f"stale manuscript table: {path.relative_to(REPO)}")
        else:
            output.mkdir(parents=True, exist_ok=True)
            for name, text in generated.items():
                (output / name).write_text(text, encoding="utf-8")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Manuscript evidence check failed: {exc}\n")
    print("Manuscript tables and evidence digests verified." if args.check else "Generated manuscript tables from pinned evidence.")


if __name__ == "__main__":
    main()
