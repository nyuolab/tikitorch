"""Small, untimed AdamW numerical diagnostic; not a performance benchmark.

Compares identical initial weights and batches, records the first differing
primitive, loss-boundary and route controls, and tracks gradients, moments and
parameter updates. Explicit epsilon candidates are distinct optimizer protocols,
not relaxed correctness tolerances or a change to PyTorch's AdamW algorithm.
Native compilation never falls back silently to another backend.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORD = ROOT / "manuscript/evidence/adamw_diagnostic_benchmarks_2026-09-09.json"


def _load_runtime():
    # PDF/source freshness checks must also work in a stdlib-only environment.
    global torch, F, B
    import torch
    import torch.nn.functional as F
    from scripts.route_certificates import training_benchmark as B


def _reset_compile_cache():
    """Use the public namespace when available, with the PyTorch 2.0 path."""
    reset = getattr(getattr(torch, "compiler", None), "reset", None)
    if reset is None:
        from torch._dynamo import reset
    reset()


def default_config():
    return dict(seed=0, width=16, layers=1, vocab=11, batch=2, sequence=12,
                device="cpu", dtype="float32", lr=0.001,
                compile_backend="inductor")


def tensor_error(expected, actual):
    difference = (expected.detach() - actual.detach()).abs()
    index = int(difference.reshape(-1).argmax())
    return dict(max_abs=float(difference.max()),
                allclose=bool(torch.allclose(expected, actual, atol=1e-4, rtol=1e-4)),
                differing_elements=int(torch.count_nonzero(difference)),
                flat_index=index, expected=float(expected.detach().reshape(-1)[index]),
                actual=float(actual.detach().reshape(-1)[index]))


def loss_boundary(logits, targets):
    # Same FP32 loss boundary as training_benchmark, including float64 models.
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


def _route_observation(model, ids):
    logits, tape = B.morphssm_forward_tape(model, ids, packed=False)
    return logits, tuple(tape.read(field.name) for field in tape.fields)


def _gradients(model, ids, targets, arm, compiled_forward, compiled_route):
    model.zero_grad(set_to_none=True)
    if arm in ("packed", "unpacked"):
        loss, gradients = compiled_route(model, ids, targets, arm == "packed")
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        return loss.detach()
    logits = compiled_forward(model, ids) if arm == "compiled" else model(ids)
    loss = loss_boundary(logits, targets)
    loss.backward()
    return loss.detach()


def primitive_controls(config):
    """Isolate common-input operations before any optimizer state can diverge."""
    _load_runtime()
    model = B.make_model(config)
    ids, targets = B.make_batch(config, config["seed"] + 100)
    inputs = model.embed(ids).detach()
    backend = config["compile_backend"]
    mean = lambda x: x.mean(dim=-1, keepdim=True)
    center = lambda x: x - x.mean(dim=-1, keepdim=True)
    operations = {}
    for name, function in (("embedding_feature_mean", mean),
                           ("embedding_mean_subtraction", center),
                           ("first_normalization", model.layers[0].norm)):
        operations[name] = tensor_error(function(inputs), torch.compile(
            function, fullgraph=True, backend=backend)(inputs))
    eager_logits, routes = _route_observation(model, ids)
    compiled_logits, compiled_routes = torch.compile(
        _route_observation, fullgraph=True, backend=backend)(model, ids)
    _, tape = B.morphssm_forward_tape(model, ids, packed=False)
    route_errors = {
        field.name: int(torch.count_nonzero(a != b))
        for field, a, b in zip(tape.fields, routes, compiled_routes)
    }
    gradient = torch.func.grad_and_value(loss_boundary)
    error, _ = gradient(eager_logits, targets)
    compiled_error, _ = torch.compile(gradient, fullgraph=True, backend=backend)(
        eager_logits, targets)
    changed_logits_error, _ = gradient(compiled_logits, targets)
    common_vjp = B.morphssm_vjp(tape, model, error)
    compiled_vjp = torch.compile(B.morphssm_vjp, fullgraph=True, backend=backend)(
        tape, model, error)

    # A feature selected by all classes (or no classes) in every row changes
    # logits only by a row-wise common offset. CE's exact derivative along this
    # final additive-bias coordinate is zero; no numerical threshold is used.
    winners = tape.read("head_winner")
    selected = F.one_hot(winners, num_classes=config["width"]).sum(-2).bool()
    counts = selected.sum(dim=1)
    null_features = ((counts == 0) | (counts == config["vocab"])).all(0)
    shadow = deepcopy(model).double()
    shadow_logits = shadow(ids)
    shadow_loss = F.cross_entropy(shadow_logits.reshape(-1, config["vocab"]), targets.flatten())
    shadow_loss.backward()  # Deliberately FP64 boundary, unlike the benchmark.
    shadow_winners = B.morphssm_forward_tape(shadow, ids, packed=False)[1].read("head_winner")
    null_coordinates = []
    for coordinate in torch.nonzero(null_features).flatten().tolist():
        null_coordinates.append(dict(
            parameter="ln_final.beta", coordinate=coordinate,
            classes_selecting_feature_per_row=counts[:, coordinate].tolist(),
            eager_fp32_gradient=float(common_vjp["ln_final.beta"][coordinate]),
            compiled_same_cotangent_gradient=float(compiled_vjp["ln_final.beta"][coordinate]),
            fp64_gradient=float(shadow.ln_final.beta.grad[coordinate]),
        ))
    # Both production arms use eager stock AdamW. Identical supplied gradients
    # must give identical optimizer states/updates regardless of their origin.
    clones = [deepcopy(model), deepcopy(model)]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=config["lr"], weight_decay=0.01)
                  for m in clones]
    for clone, optimizer in zip(clones, optimizers):
        for name, parameter in clone.named_parameters():
            parameter.grad = common_vjp[name].clone()
        optimizer.step()
    return dict(
        operations=operations, logits=tensor_error(eager_logits, compiled_logits),
        initial_route_differences=route_errors,
        same_logits_loss_cotangent=tensor_error(error, compiled_error),
        changed_logits_loss_cotangent=tensor_error(error, changed_logits_error),
        same_tape_same_cotangent_vjp={name: tensor_error(value, compiled_vjp[name])
                                     for name, value in common_vjp.items()},
        structural_null_coordinates=null_coordinates,
        fp64_shadow_head_route_differences=int(torch.count_nonzero(winners != shadow_winners)),
        same_gradient_adamw_max_parameter_error=max(float((a - b).abs().max().detach())
                                                   for a, b in zip(clones[0].parameters(), clones[1].parameters())),
    )


def compare_trajectory(config, *, eps=1e-8, steps=5, fresh_batches=True,
                       arms=("compiled", "unpacked", "packed")):
    """Keep the benchmark's unchanged FP32 1e-4 absolute/relative tolerances."""
    _load_runtime()
    if steps < 1 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("steps and epsilon must be positive and finite")
    # Parameter identity is deliberately guarded by the route contract. Each
    # independent diagnostic cell gets a fresh graph cache, just as benchmark
    # arms get fresh worker processes; this is not a timed performance claim.
    _reset_compile_cache()
    models = {arm: B.make_model(config) for arm in ("eager", *arms)}
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                       weight_decay=0.01, eps=eps)
                  for arm, model in models.items()}
    compile_options = dict(fullgraph=True, backend=config["compile_backend"])
    compiled_forward = torch.compile(type(models["eager"]).forward, **compile_options)
    compiled_route = torch.compile(B.route_gradient_function, **compile_options)
    compiled_observation = torch.compile(_route_observation, **compile_options)
    atol = rtol = 1e-4
    rows = []
    first_failure = None
    for step in range(steps):
        ids, targets = B.make_batch(config, config["seed"] + 100 + (step if fresh_batches else 0))
        losses = {arm: _gradients(model, ids, targets, arm, compiled_forward, compiled_route)
                  for arm, model in models.items()}
        before = {arm: {name: p.detach().clone() for name, p in model.named_parameters()}
                  for arm, model in models.items()}
        reference = dict(models["eager"].named_parameters())
        _, expected_routes = _route_observation(models["eager"], ids)
        same_state, route_changes, same_state_routes = {}, {}, {}
        for arm in arms:
            model = models[arm]
            parameters = dict(model.named_parameters())
            independent_gradients = {name: p.grad.detach().clone() for name, p in parameters.items()}
            _, independent_routes = _route_observation(model, ids)
            route_changes[arm] = sum(int(torch.count_nonzero(a != b))
                                     for a, b in zip(expected_routes, independent_routes))
            # Teacher forcing isolates the implementation's gradients from any
            # accumulated parameter/moment drift. Restore everything before
            # the real independent optimizer update; do not erase its failure.
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(before["eager"][name])
            _gradients(model, ids, targets, arm, compiled_forward, compiled_route)
            same_state[arm] = {name: tensor_error(reference[name].grad, p.grad)
                               for name, p in parameters.items()}
            _, native_routes = compiled_observation(model, ids)
            same_state_routes[arm] = sum(int(torch.count_nonzero(a != b))
                                        for a, b in zip(expected_routes, native_routes))
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(before[arm][name])
                    parameter.grad = independent_gradients[name]
        for optimizer in optimizers.values():
            optimizer.step()
        for arm in arms:
            parameters = []
            passed = bool(torch.allclose(losses["eager"], losses[arm], atol=atol, rtol=rtol))
            for name, actual in models[arm].named_parameters():
                expected = reference[name]
                grad = tensor_error(expected.grad, actual.grad)
                parameter = tensor_error(expected, actual)
                update = tensor_error(expected - before["eager"][name], actual - before[arm][name])
                index = parameter["flat_index"]
                expected_state = optimizers["eager"].state[expected]
                actual_state = optimizers[arm].state[actual]
                gradient_allclose = bool(torch.allclose(expected.grad, actual.grad, atol=atol, rtol=rtol))
                parameter_allclose = bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
                record = dict(name=name, gradient=grad, parameter=parameter, update=update,
                              same_state_gradient=same_state[arm][name],
                              gradient_allclose=gradient_allclose, parameter_allclose=parameter_allclose,
                              gradient_at_max_parameter_error=dict(
                                  expected=float(expected.grad.flatten()[index]),
                                  actual=float(actual.grad.flatten()[index])),
                              exp_avg=tensor_error(expected_state["exp_avg"], actual_state["exp_avg"]),
                              exp_avg_sq=tensor_error(expected_state["exp_avg_sq"], actual_state["exp_avg_sq"]))
                parameters.append(record)
                passed &= gradient_allclose and parameter_allclose
            rows.append(dict(update=step + 1, arm=arm, status="pass" if passed else "fail",
                             loss_allclose=bool(torch.allclose(losses["eager"], losses[arm], atol=atol, rtol=rtol)),
                             independent_state_route_changes=route_changes[arm],
                             same_state_compiled_route_changes=same_state_routes[arm],
                             batch_sha256=B.digest((ids, targets)),
                             loss=tensor_error(losses["eager"], losses[arm]), parameters=parameters))
            if not passed and first_failure is None:
                first_failure = dict(update=step + 1, arm=arm)
    return dict(config=config, optimizer=dict(name="torch.optim.AdamW", eps=eps,
                                             lr=config["lr"], weight_decay=0.01,
                                             betas=[0.9, 0.999], compiled=False),
                steps=steps, arms=list(arms), fresh_batches=fresh_batches, atol=atol, rtol=rtol,
                status="pass" if first_failure is None else "fail", first_failure=first_failure,
                initial_parameters_sha256=B.digest(B.make_model(config).parameters()), rows=rows)


def source_paths():
    runner = Path(__file__).resolve()
    return [runner, *(runner.with_name(name) for name in (
        "training_benchmark.py", "actual_model_benchmark.py", "provenance.py")),
        *sorted((ROOT / "tikitorch").rglob("*.py"))]


def source_identity():
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              check=True, text=True, capture_output=True).stdout.strip()
    return dict(git_revision=revision, files=hashes,
                sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())


def record_digest(record):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _check_finite(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item)
            if key in ("max_abs", "differing_elements", "flat_index",
                       "independent_state_route_changes", "same_state_compiled_route_changes"):
                if not isinstance(item, (int, float)) or isinstance(item, bool) or item < 0:
                    raise ValueError("optimizer diagnostic has invalid nonnegative measurement")
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("optimizer diagnostic has nonfinite measurements")


def _check_measurement(measurement):
    if set(measurement) != {"max_abs", "allclose", "differing_elements", "flat_index", "expected", "actual"}:
        raise ValueError("optimizer diagnostic parameter measurement is incomplete")
    if type(measurement["allclose"]) is not bool:
        raise ValueError("optimizer diagnostic measurement lacks a boolean comparison")
    for key in ("differing_elements", "flat_index"):
        if type(measurement[key]) is not int:
            raise ValueError("optimizer diagnostic index/count must be integers")
    for key in ("max_abs", "expected", "actual"):
        if type(measurement[key]) not in (int, float):
            raise ValueError("optimizer diagnostic metric must be numeric")


def check_record(path=DEFAULT_RECORD):
    """Validate source binding and recorded outcomes without importing Torch.

    Numerical failures are valid observations, never rewritten as successful
    performance comparisons. This is an integrity check, not a rerun.
    """
    record = json.loads(Path(path).read_text())
    _check_finite(record)
    if record.get("schema") != "tikitorch.optimizer_diagnostic.v1":
        raise ValueError("unsupported optimizer diagnostic schema")
    if record.get("record_sha256") != record_digest(record):
        raise ValueError("optimizer diagnostic content digest mismatch")
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths()}
    if record["source"]["files"] != hashes:
        raise ValueError("optimizer diagnostic source binding is stale")
    if record["source"]["sha256"] != hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest():
        raise ValueError("optimizer diagnostic source digest mismatch")
    trajectories = record.get("trajectories", [])
    if not trajectories:
        raise ValueError("optimizer diagnostic contains no trajectories")
    protocol = record["protocol"]
    for dimension in ("width", "layers", "vocab", "batch", "sequence"):
        if type(protocol["config"][dimension]) is not int or protocol["config"][dimension] < 1:
            raise ValueError("optimizer diagnostic has invalid model dimensions")
    if protocol["config"]["dtype"] != "float32" or protocol["config"]["device"] not in ("cpu", "cuda") or protocol["config"]["compile_backend"] not in ("inductor", "aot_eager"):
        raise ValueError("optimizer diagnostic has invalid numerical configuration")
    arms, seeds, epsilons, steps = (protocol[key] for key in ("arms", "seeds", "epsilons", "steps"))
    if type(steps) is not int or steps < 1 or not arms or len(set(arms)) != len(arms) or not set(arms) <= {"compiled", "packed", "unpacked"}:
        raise ValueError("optimizer diagnostic has invalid trajectory dimensions")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("optimizer diagnostic has invalid or duplicate seeds")
    if not epsilons or len(set(epsilons)) != len(epsilons) or any(type(eps) not in (int, float) or eps <= 0 for eps in epsilons):
        raise ValueError("optimizer diagnostic has invalid or duplicate epsilon protocols")
    if [(trajectory["optimizer"]["eps"], trajectory["config"]["seed"]) for trajectory in trajectories] != [
            (eps, seed) for eps in epsilons for seed in seeds]:
        raise ValueError("optimizer diagnostic protocol matrix is incomplete or duplicated")
    controls = record["primitive_controls"]
    if protocol["controls"] != (controls is not None):
        raise ValueError("optimizer diagnostic primitive controls are missing")
    if controls is not None:
        for name in ("embedding_feature_mean", "embedding_mean_subtraction", "first_normalization"):
            _check_measurement(controls["operations"][name])
        for name in ("logits", "same_logits_loss_cotangent", "changed_logits_loss_cotangent"):
            _check_measurement(controls[name])
        if not controls["initial_route_differences"] or not controls["same_tape_same_cotangent_vjp"]:
            raise ValueError("optimizer diagnostic route/VJP controls are empty")
        for count in controls["initial_route_differences"].values():
            if type(count) is not int or count < 0:
                raise ValueError("optimizer diagnostic route count is invalid")
        for measurement in controls["same_tape_same_cotangent_vjp"].values():
            _check_measurement(measurement)
        if controls["same_gradient_adamw_max_parameter_error"] != 0:
            raise ValueError("optimizer diagnostic identical-gradient control failed")
    if Path(path).resolve() == DEFAULT_RECORD.resolve():
        if seeds != [0, 1, 2] or epsilons != [1e-8, 1e-6, 1e-5, 1e-4] or steps != 20 or arms != ["compiled", "unpacked", "packed"] or not protocol["fresh_batches"] or controls is None:
            raise ValueError("selected manuscript optimizer diagnostic protocol is incomplete")
        if protocol["config"] != default_config():
            raise ValueError("selected manuscript optimizer diagnostic configuration changed")
    for trajectory in trajectories:
        rows = trajectory["rows"]
        if trajectory["steps"] != steps or trajectory["arms"] != arms or trajectory["fresh_batches"] != protocol["fresh_batches"]:
            raise ValueError("optimizer diagnostic trajectory does not match declared protocol")
        if trajectory["config"] != {**protocol["config"], "seed": trajectory["config"]["seed"]}:
            raise ValueError("optimizer diagnostic cell configuration changed")
        if len(rows) != trajectory["steps"] * len(trajectory["arms"]):
            raise ValueError("optimizer diagnostic trajectory is incomplete")
        if [(row["update"], row["arm"]) for row in rows] != [
                (step + 1, arm) for step in range(trajectory["steps"]) for arm in trajectory["arms"]]:
            raise ValueError("optimizer diagnostic trajectory order/arms mismatch")
        if trajectory["atol"] != 1e-4 or trajectory["rtol"] != 1e-4:
            raise ValueError("optimizer diagnostic tolerance changed")
        failures = [dict(update=row["update"], arm=row["arm"]) for row in rows if row["status"] == "fail"]
        if any(row["status"] not in ("pass", "fail") for row in rows):
            raise ValueError("unknown optimizer diagnostic numerical outcome")
        parameter_names = None
        for row in rows:
            for key in ("independent_state_route_changes", "same_state_compiled_route_changes"):
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError("optimizer diagnostic has invalid route-change counts")
            parameters = row["parameters"]
            names = {parameter["name"] for parameter in parameters}
            if not parameters or len(names) != len(parameters) or (parameter_names is not None and names != parameter_names):
                raise ValueError("optimizer diagnostic parameter comparison is incomplete")
            parameter_names = names
            _check_measurement(row["loss"])
            if row["loss_allclose"] != row["loss"]["allclose"]:
                raise ValueError("optimizer diagnostic loss status contradicts measurement")
            checks = [row["loss"]["allclose"]]
            for parameter in parameters:
                for quantity in ("gradient", "parameter", "update", "same_state_gradient", "exp_avg", "exp_avg_sq"):
                    _check_measurement(parameter[quantity])
                if parameter["gradient_allclose"] != parameter["gradient"]["allclose"] or parameter["parameter_allclose"] != parameter["parameter"]["allclose"]:
                    raise ValueError("optimizer diagnostic parameter status contradicts measurement")
                checks.extend((parameter["gradient"]["allclose"], parameter["parameter"]["allclose"]))
            if any(type(check) is not bool for check in checks) or row["status"] != ("pass" if all(checks) else "fail"):
                raise ValueError("optimizer diagnostic row status does not match its comparisons")
        if trajectory["status"] != ("fail" if failures else "pass") or trajectory["first_failure"] != (failures[0] if failures else None):
            raise ValueError("optimizer diagnostic status does not match recorded failures")
        optimizer = trajectory["optimizer"]
        if optimizer["name"] != "torch.optim.AdamW" or optimizer["compiled"]:
            raise ValueError("optimizer diagnostic no longer describes eager stock AdamW")
        if not math.isfinite(optimizer["eps"]) or optimizer["eps"] <= 0:
            raise ValueError("invalid optimizer diagnostic epsilon")
        if optimizer["lr"] != trajectory["config"]["lr"] or optimizer["lr"] <= 0 or optimizer["betas"] != [0.9, 0.999] or optimizer["weight_decay"] != 0.01:
            raise ValueError("optimizer diagnostic optimizer protocol changed")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("inductor", "aot_eager"), default="inductor")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for dimension in ("width", "layers", "vocab", "batch", "sequence"):
        parser.add_argument(f"--{dimension}", type=int, default=default_config()[dimension])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eps", nargs="+", type=float, default=[1e-8, 1e-6, 1e-5, 1e-4])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeat-batch", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--output", type=Path, help="New JSON evidence path; existing files are never overwritten")
    parser.add_argument("--check", action="store_true", help="Read-only stdlib source/evidence integrity check; no compilation")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD, help="Existing evidence to validate with --check")
    args = parser.parse_args(argv)
    if args.check:
        try:
            check_record(args.record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.exit(1, f"optimizer diagnostic check failed: {error}\n")
        print("Optimizer diagnostic source/evidence check passed (not a numerical rerun).")
        return 0
    if args.output is not None and args.output.exists():
        parser.error("output already exists; evidence is append-only")
    if args.steps < 1 or any(not math.isfinite(eps) or eps <= 0 for eps in args.eps):
        parser.error("steps and epsilon must be positive and finite")
    if min(args.width, args.layers, args.vocab, args.batch, args.sequence) < 1:
        parser.error("model and batch dimensions must be positive")
    if len(set(args.eps)) != len(args.eps) or len(set(args.seeds)) != len(args.seeds):
        parser.error("epsilon and seed cells must be unique")
    _load_runtime()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no CPU fallback")
    torch.set_num_threads(1)
    before = source_identity()
    config = {**default_config(), "compile_backend": args.backend, "device": args.device,
              **{dimension: getattr(args, dimension)
                 for dimension in ("width", "layers", "vocab", "batch", "sequence")}}
    controls = None if args.skip_controls else primitive_controls(config)
    trajectories = []
    for eps in args.eps:
        for seed in args.seeds:
            result = compare_trajectory({**config, "seed": seed}, eps=eps, steps=args.steps,
                                        fresh_batches=not args.repeat_batch)
            trajectories.append(result)
            print(f"epsilon={eps:g} seed={seed}: {result['status']} ({args.steps} updates)", flush=True)
    after = source_identity()
    if before != after:
        raise RuntimeError("executable source changed during diagnostic; discard this run")
    record = dict(schema="tikitorch.optimizer_diagnostic.v1", source=before,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv] if argv is None else list(argv),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   platform=platform.platform(), machine=platform.machine(),
                                   threads=torch.get_num_threads(),
                                   cuda=torch.version.cuda, torch_build=torch.__config__.show(),
                                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                  scope="untimed tiny numerical diagnostic, not task-quality or performance evidence",
                  protocol=dict(config=config, seeds=args.seeds, epsilons=args.eps, steps=args.steps,
                                arms=["compiled", "unpacked", "packed"], fresh_batches=not args.repeat_batch,
                                controls=not args.skip_controls),
                  primitive_controls=controls, trajectories=trajectories)
    record["record_sha256"] = record_digest(record)
    encoded = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
    if args.output is None:
        print(encoded)
    else:
        with args.output.open("x") as stream:
            stream.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
