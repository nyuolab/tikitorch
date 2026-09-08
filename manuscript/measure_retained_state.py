"""Fresh CPU route-state accounting with source-bound, append-only evidence.

``--run`` measures three predefined cells and refuses to overwrite either output.
``--check`` validates source identities, row accounting, and the generated table;
it does not rerun the experiment. This is a supplied-cotangent representation
protocol, not a latency, allocator-peak, or optimal-checkpoint comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().relative_to(REPO).as_posix()
SCALES = ((64, 2, 96), (128, 4, 96), (256, 4, 192))
ATOL = RTOL = 1e-5
SCHEMA = "tikitorch.current_route_retained_state.v1"
DEFAULT_RECORD = REPO / "manuscript/evidence/current_retained_state.json"
DEFAULT_TABLE = REPO / "manuscript/generated/current_retained_state.tex"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def source_hashes():
    """Conservatively bind every library Python source, not just observed imports."""
    paths = sorted((REPO / "tikitorch").rglob("*.py")) + [REPO / SCRIPT]
    return {p.relative_to(REPO).as_posix(): digest(p.read_bytes()) for p in paths}


def git_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, stderr=subprocess.DEVNULL)
    return dict(head=git("rev-parse", "HEAD").decode().strip(),
                dirty=bool(git("status", "--porcelain", "--untracked-files=normal")),
                tracked_diff_sha256=digest(git("diff", "HEAD", "--binary")))


def tensor_digest(tensor):
    t = tensor.detach().cpu().contiguous()
    return digest(canonical(dict(shape=list(t.shape), dtype=str(t.dtype))) + t.numpy().tobytes())


def model_digest(model):
    return digest(canonical({name: tensor_digest(value) for name, value in model.state_dict().items()}))


def execute_cell(width, layers, length):
    import torch
    from tikitorch.autodiff import morphssm_forward_tape, morphssm_vjp, route_logits
    from tikitorch.models import MorphSSM

    torch.manual_seed(0)
    model = MorphSSM(d_model=width, n_layers=layers, vocab_size=38,
                     kernel_size=7, dropout=0.0, task_type="lm", use_soft=False,
                     tau=0.0, norm_type="tropical", group_size=1,
                     multi_scale=False, residual_scaling=False).float().eval()
    model.set_tau(0.0)
    ids = torch.randint(0, 38, (1, length), generator=torch.Generator().manual_seed(1))
    q = torch.randn((1, length, 38), generator=torch.Generator().manual_seed(2))
    q = q / q.norm()
    initial_model_digest = model_digest(model)
    params = dict(model.named_parameters())

    # Recover declared routes on the actual eager inputs. This deliberately
    # uses the same canonical route primitive, not an independent oracle.
    eager_routes, handles = {}, []
    def capture(name):
        def hook(module, args, output):
            _, route = module.forward_with_routing(args[0])
            eager_routes[name] = route.detach().clone()
        return hook
    for i, layer in enumerate(model.layers):
        handles.append(layer.morph_conv.register_forward_hook(capture(f"layer{i}.conv_winner")))
        handles.append(layer.out_proj.register_forward_hook(capture(f"layer{i}.proj_winner")))
    handles.append(model.output_head.register_forward_hook(capture("head_winner")))
    try:
        eager_logits = model(ids)
    finally:
        for handle in handles:
            handle.remove()
    eager_grads = torch.autograd.grad(eager_logits, tuple(params.values()), q)
    logits, packed = morphssm_forward_tape(model, ids, packed=True)
    unpacked_logits, unpacked = morphssm_forward_tape(model, ids, packed=False)
    public_logits = route_logits(model, ids, packed=True)
    public_grads = torch.autograd.grad(public_logits, tuple(params.values()), q)
    supplied_grads = morphssm_vjp(packed, model, q)
    require(set(supplied_grads) == set(params), "missing requested parameter VJP")
    for name, value in [("eager", eager_logits), ("packed", logits),
                        ("unpacked", unpacked_logits), ("public", public_logits)]:
        require(bool(torch.isfinite(value).all()), f"nonfinite {name} logits")
        require(torch.equal(eager_logits, value), f"nonidentical {name} forward")
    route_results = []
    for name, reference in eager_routes.items():
        # Projection routing flattens leading (batch, time) dimensions in the
        # adapter; flattening is an address convention, not a value change.
        actual = packed.read(name)
        require(reference.numel() == actual.numel(), f"route shape mismatch: {name}")
        equal = torch.equal(reference.reshape_as(actual), actual)
        require(equal, f"eager route mismatch: {name}")
        route_results.append(dict(field=name, integer_equal=True, entries=actual.numel()))
    fields = []
    for field in packed.fields:
        actual, original = packed.read(field.name), unpacked.read(field.name)
        require(torch.equal(actual, original), f"packed/unpacked field mismatch: {field.name}")
        fields.append(dict(name=field.name, shape=list(field.shape), entries=field.count,
                           bit_width=field.bit_width, cardinality=field.cardinality,
                           logical_bits=field.bit_length,
                           packed_device_bytes=packed._storage[field.name].numel(),
                           unpacked_device_bytes=unpacked._storage[field.name].numel() * 8,
                           decoded_sha256=tensor_digest(actual)))
    errors = []
    for (name, _), expected, public in zip(params.items(), eager_grads, public_grads):
        got = supplied_grads[name]
        for label, value in (("eager", expected), ("supplied", got), ("public", public)):
            require(bool(torch.isfinite(value).all()), f"nonfinite {label} gradient: {name}")
        supplied_ok = torch.allclose(got, expected, atol=ATOL, rtol=RTOL)
        public_ok = torch.allclose(public, expected, atol=ATOL, rtol=RTOL)
        err = float((got - expected).abs().max())
        public_err = float((public - expected).abs().max())
        require(supplied_ok and public_ok,
                f"gradient mismatch: {name}; supplied max {err}; public max {public_err}")
        errors.append(dict(parameter=name, supplied_max_abs=err,
                           public_max_abs=public_err, eager_max_abs=float(expected.abs().max()),
                           supplied_allclose=True, public_allclose=True))
    blob = packed.to_bytes()
    roundtrip = type(packed).from_bytes(blob, expected_fingerprint=packed.fingerprint)
    require(all(torch.equal(packed.read(f.name), roundtrip.read(f.name)) for f in packed.fields),
            "portable roundtrip mismatch")
    require(model_digest(model) == initial_model_digest, "measurement mutated model state")
    core = (layers + 1) * length * width * 4
    return dict(config=dict(batch=1, width=width, layers=layers, length=length,
                            vocabulary=38, kernel=7, dropout=0.0, tau=0.0,
                            head_k=model.output_head.k, dtype="torch.float32", mode="eval",
                            dilations=[layer.morph_conv.dilation for layer in model.layers]),
                seeds=dict(parameters=0, tokens=1, output_cotangent=2),
                model_sha256=initial_model_digest, input_sha256=tensor_digest(ids),
                cotangent_sha256=tensor_digest(q), cotangent_l2=float(q.norm()),
                logits_sha256=tensor_digest(logits), fingerprint=packed.fingerprint,
                fields=fields, logical_bits=packed.bit_length,
                packed_device_bytes=packed.nbytes, portable_bytes=len(blob),
                portable_sha256=digest(blob), unpacked_device_bytes=unpacked.nbytes,
                checkpoint_core_bytes=core, comparator_over_packed=core / packed.nbytes,
                exact_forward=True, exact_packed_unpacked_fields=True,
                exact_portable_roundtrip=True, eager_route_checks=route_results,
                gradient_atol=ATOL, gradient_rtol=RTOL, gradient_errors=errors,
                max_gradient_abs_error=max(max(e["supplied_max_abs"], e["public_max_abs"]) for e in errors),
                status="passed")


def validate_record(record, *, check_source=True):
    require(record["schema"] == SCHEMA and record["status"] == "completed", "invalid record status/schema")
    require(record["sources_before_sha256"] == record["sources_after_sha256"], "source changed during run")
    if check_source:
        require(record["sources_before_sha256"] == source_hashes(), "recorded executable sources differ from checkout")
    require(record["results_sha256"] == digest(canonical(record["rows"])), "result payload digest mismatch")
    require(len(record["rows"]) == len(SCALES), "wrong number of cells")
    for row, scale in zip(record["rows"], SCALES):
        cfg = row["config"]
        require(tuple(cfg[k] for k in ("width", "layers", "length")) == scale, "unexpected cell/order")
        require(cfg["batch"] == 1 and cfg["vocabulary"] == 38 and cfg["kernel"] == 7
                and cfg["tau"] == cfg["dropout"] == 0.0 and cfg["dtype"] == "torch.float32", "protocol changed")
        require(row["status"] == "passed" and all(row[k] for k in (
            "exact_forward", "exact_packed_unpacked_fields", "exact_portable_roundtrip")), "failed parity")
        require(row["gradient_atol"] == ATOL and row["gradient_rtol"] == RTOL, "tolerances changed")
        require(all(r["integer_equal"] for r in row["eager_route_checks"])
                and len(row["eager_route_checks"]) == 2 * cfg["layers"] + 1, "route check missing/failed")
        fields = row["fields"]
        require(len({f["name"] for f in fields}) == len(fields), "duplicate field")
        for f in fields:
            require(f["entries"] == math.prod(f["shape"]), "field shape/count mismatch")
            require(f["bit_width"] == (f["cardinality"] - 1).bit_length(), "field width mismatch")
            require(f["logical_bits"] == f["entries"] * f["bit_width"], "logical field bits mismatch")
            require(f["packed_device_bytes"] == (f["logical_bits"] + 7) // 8, "device field bytes mismatch")
            require(f["unpacked_device_bytes"] == f["entries"] * 8, "unpacked field bytes mismatch")
        require(row["logical_bits"] == sum(f["logical_bits"] for f in fields), "logical total mismatch")
        for key in ("packed_device_bytes", "unpacked_device_bytes"):
            require(row[key] == sum(f[key] for f in fields), f"{key} total mismatch")
        require(row["portable_bytes"] > (row["logical_bits"] + 7) // 8, "portable framing absent")
        core = (cfg["layers"] + 1) * cfg["length"] * cfg["width"] * 4
        require(row["checkpoint_core_bytes"] == core, "analytical comparator mismatch")
        require(row["comparator_over_packed"] == core / row["packed_device_bytes"], "ratio mismatch")
        errors = row["gradient_errors"]
        require(bool(errors) and all(e["supplied_allclose"] and e["public_allclose"] for e in errors), "VJP mismatch")
        for e in errors:
            require(all(math.isfinite(e[k]) and e[k] >= 0 for k in (
                "supplied_max_abs", "public_max_abs", "eager_max_abs")), "nonfinite gradient statistic")
        require(row["max_gradient_abs_error"] == max(max(e["supplied_max_abs"], e["public_max_abs"]) for e in errors),
                "gradient maximum mismatch")
    # Reject nonfinite numbers anywhere, not just in visible table columns.
    canonical(record)


def render_table(record):
    lines = ["% Generated by manuscript/measure_retained_state.py; supplied-cotangent CPU protocol.",
             "% Analytical checkpoint boundaries; not measured peak or optimal checkpointing.",
             f"% Results SHA256: {record['results_sha256']}",
             r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"$D/L/T$ & Packed B & Portable B & Boundary B & Ratio & Max. grad. error \\",
             r"\midrule"]
    for row in record["rows"]:
        c = row["config"]
        lines.append(f"{c['width']}/{c['layers']}/{c['length']} & {row['packed_device_bytes']:,} & "
                     f"{row['portable_bytes']:,} & {row['checkpoint_core_bytes']:,} & "
                     f"{row['comparator_over_packed']:.2f}" + r"$\times$ & "
                     + f"{row['max_gradient_abs_error']:.2e}" + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def run():
    before = source_hashes()
    identity = git_identity()
    # Invocation from any cwd must import this source tree, never an installed
    # package that happens to share its name.
    sys.path.insert(0, str(REPO))
    import numpy
    import torch
    import tikitorch
    require(Path(tikitorch.__file__).resolve().is_relative_to(REPO / "tikitorch"), "wrong library import")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    rows = []
    for scale in SCALES:
        rows.append(execute_cell(*scale))
        print(f"Passed {scale}: {rows[-1]['packed_device_bytes']} packed bytes", flush=True)
    after = source_hashes()
    require(before == after, "executable source changed during measurement; no completed record emitted")
    return dict(schema=SCHEMA, status="completed", started_utc=started,
                elapsed_seconds=time.monotonic() - clock,
                command="python manuscript/measure_retained_state.py --run",
                source_identity=identity, sources_before_sha256=before, sources_after_sha256=after,
                source_identity_scope="all tikitorch Python sources plus this measurement script; immutable release pending",
                environment=dict(python=platform.python_version(), torch=torch.__version__,
                                 numpy=numpy.__version__, system=platform.system(),
                                 architecture=platform.machine(), device="cpu", threads=1,
                                 interop_threads=1, deterministic_algorithms=True),
                protocol=dict(name="current hard MorphSSM supplied-cotangent retained-state v1",
                              cotangent="seed-2 Gaussian tensor divided by its L2 norm",
                              eager_routes="canonical route primitive on actual eager inputs; not an independent algorithm",
                              comparator="analytical (L+1)*B*T*D*4 bytes, one embedding/layer boundary set",
                              exclusions="loss/logits, output cotangent, parameters, optimizer state, metadata and all transient workspace",
                              interpretation="representation accounting only; no timing, peak, traffic, learning, or optimal-checkpoint claim",
                              confidence="one fixed initialization/input/cotangent per cell; no statistical generalization"),
                rows=rows, results_sha256=digest(canonical(rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args(argv)
    try:
        if args.check:
            record = json.loads(args.output.read_text(encoding="utf-8"))
            validate_record(record)
            require(args.table.read_text(encoding="utf-8") == render_table(record), "generated table is stale")
        else:
            require(not args.output.exists() and not args.table.exists(), "refusing to overwrite existing evidence/table")
            if hasattr(signal, "SIGALRM"):
                def deadline(signum, frame):
                    raise TimeoutError("120-second CPU measurement deadline exceeded")
                signal.signal(signal.SIGALRM, deadline)
                signal.alarm(120)
            record = run()
            validate_record(record)
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.table.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
            with args.table.open("x", encoding="utf-8") as output:
                output.write(render_table(record))
    except (ValueError, KeyError, OSError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"Current retained-state protocol failed: {exc}\n")
    print("Current retained-state source, accounting, parity record, and table verified.")


if __name__ == "__main__":
    main()
