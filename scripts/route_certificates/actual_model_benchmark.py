"""Exp-1 item 7 — crash-durable five-arm benchmark on the actual MorphSSM.

The public runner is an orchestrator.  Every (model configuration, seed) cell
runs in its own process with a timeout, is appended and fsync'ed immediately,
and cannot prevent earlier cells from being retained.  A heartbeat identifies
the active cell and a hard wall prevents a compiler stall from owning the
machine indefinitely.

Arms use identical seeded models, inputs and targets.  Their order is shuffled
deterministically per cell and recorded.  ``compiled`` applies only to the
lifetime-model configuration.  The packed arm reports the complete retained
reverse boundary: serialized route tape + logits + targets.  A separate
100-update trajectory cell asserts packed replay and cascade equality at both
5 and 100 AdamW steps.

Usage::

    python -m scripts.route_certificates.actual_model_benchmark [--dry-run]

Results are append-only under
``results/route_certificates/runs/exp1_benchmark/<uuid>/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tikitorch.research.pc_convrnn import MorphPCEngine, build_model  # noqa: E402

from tikitorch.autodiff.morphssm_tape import (  # noqa: E402
    deserialize, pack_engine_cache, replay, serialize)
from scripts.route_certificates.provenance import (  # noqa: E402
    finish, run_record)

V, B = 38, 1
CONFIGS = ((64, 2, 96), (128, 4, 96), (256, 4, 192))
LIFETIME = (128, 4, 96)
SEEDS = (0, 1, 2)
ARMS = ("eager", "compiled", "checkpoint", "cascade", "packed")
WARMUP, REPS = 5, 20
ORDER_SEED = 20_260_822
DEFAULT_CELL_TIMEOUT_S = 30 * 60
DEFAULT_HARD_WALL_S = 3 * 60 * 60
DEFAULT_HEARTBEAT_S = 10.0
TRAJECTORY_CONFIG = (*LIFETIME, 0)
TRAJECTORY_STEPS = 100
TRAJECTORY_CHECKPOINTS = (5, 100)


def make(d: int, layers: int, seed: int):
    torch.manual_seed(seed)
    ns = argparse.Namespace(d_model=d, n_layers=layers, kernel=7,
                            dropout=0.0, tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", V, ns)
    model.set_tau(0.0)
    model.eval()
    return model


def make_batch(T: int, seed: int):
    gen = torch.Generator().manual_seed(seed + 100)
    ids = torch.randint(0, V, (B, T), generator=gen)
    targets = torch.randint(0, V, (B, T), generator=gen)
    return ids, targets


def tensor_bytes(value: Any) -> int:
    return int(value.numel() * value.element_size()) if torch.is_tensor(value) else 0


def med(fn: Callable[[], Any], n: int = REPS, warm: int = WARMUP) -> float:
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(samples), 3)


def deterministic_arm_order(d: int, layers: int, T: int, seed: int,
                            order_seed: int = ORDER_SEED) -> list[str]:
    arms = list(ARMS if (d, layers, T) == LIFETIME
                else tuple(a for a in ARMS if a != "compiled"))
    mixed = (order_seed * 1_000_003 + d * 10_007 + layers * 1_009
             + T * 101 + seed)
    random.Random(mixed).shuffle(arms)
    return arms


def _grad_keys(model) -> list[str]:
    return sorted(name for name, param in model.named_parameters()
                  if param.grad is not None)


def _eager_arm(d, layers, T, seed, warm, reps):
    model, (ids, targets) = make(d, layers, seed), make_batch(T, seed)

    def update_grads():
        for param in model.parameters():
            param.requires_grad_(True)
            param.grad = None
        loss = torch.nn.functional.cross_entropy(
            model(ids).reshape(-1, V).float(), targets.reshape(-1))
        loss.backward()
        for param in model.parameters():
            param.requires_grad_(False)

    elapsed = med(update_grads, reps, warm)
    return dict(eager_ms=elapsed, grad_keys=_grad_keys(model))


def _compiled_arm(d, layers, T, seed, warm, reps):
    model, (ids, targets) = make(d, layers, seed), make_batch(T, seed)
    compiled_model = torch.compile(model)

    def update_grads():
        for param in model.parameters():
            param.requires_grad_(True)
            param.grad = None
        loss = torch.nn.functional.cross_entropy(
            compiled_model(ids).reshape(-1, V).float(), targets.reshape(-1))
        loss.backward()
        for param in model.parameters():
            param.requires_grad_(False)

    start = time.perf_counter()
    update_grads()
    overhead = round(time.perf_counter() - start, 3)
    elapsed = med(update_grads, reps, warm)
    return dict(compiled_ms=elapsed, compile_overhead_s=overhead,
                grad_keys=_grad_keys(model))


def _checkpoint_arm(d, layers, T, seed, warm, reps):
    from torch.utils.checkpoint import checkpoint

    model, (ids, targets) = make(d, layers, seed), make_batch(T, seed)

    def update_grads():
        for param in model.parameters():
            param.requires_grad_(True)
            param.grad = None
        hidden = (model.dropout(model.embed(ids)) if hasattr(model, "dropout")
                  else model.embed(ids))
        for layer in model.layers:
            hidden = checkpoint(layer, hidden, use_reentrant=False)
        logits = model.output_head(model.ln_final(hidden))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V).float(), targets.reshape(-1))
        loss.backward()
        for param in model.parameters():
            param.requires_grad_(False)

    elapsed = med(update_grads, reps, warm)
    return dict(checkpoint_ms=elapsed, grad_keys=_grad_keys(model))


def _cascade_arm(d, layers, T, seed, warm, reps):
    model, (ids, targets) = make(d, layers, seed), make_batch(T, seed)
    engine = MorphPCEngine(model)

    def update_grads():
        for param in model.parameters():
            param.grad = None
        engine.cascade_grads(ids, targets)

    elapsed = med(update_grads, reps, warm)
    return dict(cascade_ms=elapsed, grad_keys=_grad_keys(model))


def _packed_arm(d, layers, T, seed, warm, reps):
    model, (ids, targets) = make(d, layers, seed), make_batch(T, seed)
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    tape = pack_engine_cache(caches, top, model, ids)
    blob = serialize(tape, model)
    restored = deserialize(blob, model)

    def replay_only():
        return replay(restored, model, logits, targets)

    replay_ms = med(replay_only, reps, warm)

    def full_cycle():
        lg, cc, tp = engine._forward_cache(ids)
        packed = pack_engine_cache(cc, tp, model, ids)
        encoded = serialize(packed, model)
        decoded = deserialize(encoded, model)
        replay(decoded, model, lg, targets)

    full_ms = med(full_cycle, max(1, reps // 2), max(0, warm // 2))
    grads = replay_only()

    logits_bytes = tensor_bytes(logits)
    targets_bytes = tensor_bytes(targets)
    boundary_bytes = logits_bytes + targets_bytes
    state = dict(
        tape_logical_bits=int(tape.bit_length),
        tape_payload_bytes=int(tape.payload_nbytes),
        tape_serialized_bytes=len(blob),
        boundary_logits_shape=list(logits.shape),
        boundary_logits_dtype=str(logits.dtype),
        boundary_logits_bytes=logits_bytes,
        boundary_targets_shape=list(targets.shape),
        boundary_targets_dtype=str(targets.dtype),
        boundary_targets_bytes=targets_bytes,
        boundary_total_bytes=boundary_bytes,
        packed_reverse_retained_bytes=len(blob) + boundary_bytes,
        packed_reverse_retained_per_token=round(
            (len(blob) + boundary_bytes) / int(ids.numel()), 3),
        accounting_scope=("serialized route tape + logits + targets; resident "
                          "model parameters are common execution state and "
                          "excluded from every arm"),
    )
    return dict(replay_only_ms=replay_ms, packed_full_cycle_ms=full_ms,
                packed_state=state, grad_keys=sorted(grads))


ARM_FN = {
    "eager": _eager_arm,
    "compiled": _compiled_arm,
    "checkpoint": _checkpoint_arm,
    "cascade": _cascade_arm,
    "packed": _packed_arm,
}


def _forward_and_optimizer(d, layers, T, seed, warm, reps):
    model, (ids, targets) = make(d, layers, seed), make_batch(T, seed)
    forward_ms = med(lambda: model(ids), reps, warm)
    engine = MorphPCEngine(model)
    model.zero_grad(set_to_none=True)
    engine.cascade_grads(ids, targets)
    fixed = {name: param.grad.detach().clone()
             for name, param in model.named_parameters()
             if param.grad is not None}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01)

    def opt_step():
        for name, param in model.named_parameters():
            param.grad = fixed[name].clone() if name in fixed else None
        optimizer.step()

    adamw_ms = med(opt_step, max(1, reps // 2), max(0, warm // 2))
    return forward_ms, adamw_ms


def bench_config(d: int, layers: int, T: int, seed: int,
                 *, warm: int = WARMUP, reps: int = REPS,
                 order_seed: int = ORDER_SEED) -> dict[str, Any]:
    """Run one independently killable benchmark cell."""
    started = time.time()
    row: dict[str, Any] = dict(
        cell_id=f"d{d}_l{layers}_t{T}_s{seed}", d_model=d,
        n_layers=layers, T=T, seed=seed, status="RUNNING",
        arm_order=deterministic_arm_order(d, layers, T, seed, order_seed),
        warmup=warm, reps=reps,
    )
    grad_sets: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for arm in row["arm_order"]:
        try:
            result = ARM_FN[arm](d, layers, T, seed, warm, reps)
            grad_sets[arm] = result.pop("grad_keys")
            row.update(result)
        except Exception as exc:  # a measured arm error, never an imputation
            errors[arm] = f"{type(exc).__name__}: {exc}"[:500]
            row[f"{arm}_ms"] = None
    if "compiled" not in row["arm_order"]:
        row["compiled_ms"] = None
        row["compiled_status"] = "NOT_APPLICABLE_NON_LIFETIME_CONFIG"

    forward_ms, adamw_ms = _forward_and_optimizer(
        d, layers, T, seed, warm, reps)
    row["forward_ms"] = forward_ms
    row["adamw_step_ms"] = adamw_ms
    required = [a for a in ("eager", "checkpoint", "cascade", "packed")
                if a not in errors]
    grad_key_equal = bool(required) and all(
        grad_sets[a] == grad_sets[required[0]] for a in required[1:])
    if not grad_key_equal:
        errors["grad_key_equality"] = str(
            {arm: keys for arm, keys in grad_sets.items()})[:500]
    row["produced_grad_keys"] = grad_sets
    row["grad_key_equality"] = grad_key_equal
    row["arm_errors"] = errors
    if row.get("eager_ms") and row.get("cascade_ms"):
        row["cascade_vs_eager"] = round(
            row["eager_ms"] / row["cascade_ms"], 3)
        row["reverse_est_eager_ms"] = round(
            row["eager_ms"] - row["forward_ms"], 3)
        if row.get("replay_only_ms"):
            row["reverse_replay_vs_eager_reverse"] = round(
                row["reverse_est_eager_ms"] / row["replay_only_ms"], 3)
    row["status"] = "COMPLETE" if not errors else "COMPLETE_WITH_ARM_ERRORS"
    row["wall_s"] = round(time.time() - started, 3)
    return row


def _update_digest(model, optimizer) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode())
        h.update(param.detach().contiguous().cpu().numpy().tobytes())
        state = optimizer.state.get(param, {})
        for key in sorted(state):
            h.update(str(key).encode())
            value = state[key]
            if torch.is_tensor(value):
                h.update(value.detach().contiguous().cpu().numpy().tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def _first_tensor_mismatch(left, right):
    for name, value in left.named_parameters():
        other = dict(right.named_parameters())[name]
        if not torch.equal(value, other):
            return f"parameter:{name}"
    return None


def _optimizer_equal(left, right, lm, rm) -> tuple[bool, str | None]:
    lp, rp = dict(lm.named_parameters()), dict(rm.named_parameters())
    for name in lp:
        ls, rs = left.state.get(lp[name], {}), right.state.get(rp[name], {})
        if set(ls) != set(rs):
            return False, f"optimizer_keys:{name}"
        for key in ls:
            a, b = ls[key], rs[key]
            if torch.is_tensor(a):
                if not torch.equal(a, b):
                    return False, f"optimizer:{name}:{key}"
            elif a != b:
                return False, f"optimizer:{name}:{key}"
    return True, None


def trajectory_parity(d: int, layers: int, T: int, seed: int,
                      steps: int = TRAJECTORY_STEPS,
                      checkpoints=TRAJECTORY_CHECKPOINTS) -> dict[str, Any]:
    """Exact packed-replay/cascade AdamW trajectories on changing batches."""
    if max(checkpoints) > steps:
        raise ValueError("trajectory checkpoint exceeds requested steps")
    cascade_model = make(d, layers, seed)
    packed_model = make(d, layers, seed)
    if _first_tensor_mismatch(cascade_model, packed_model) is not None:
        raise AssertionError("seeded trajectory models differ at initialization")
    cascade_engine = MorphPCEngine(cascade_model)
    packed_engine = MorphPCEngine(packed_model)
    cascade_opt = torch.optim.AdamW(cascade_model.parameters(), lr=1e-3,
                                    weight_decay=0.01)
    packed_opt = torch.optim.AdamW(packed_model.parameters(), lr=1e-3,
                                   weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 90_001)
    records = []
    first_mismatch = None
    start = time.time()
    for step in range(1, steps + 1):
        ids = torch.randint(0, V, (B, T), generator=gen)
        targets = torch.randint(0, V, (B, T), generator=gen)

        cascade_opt.zero_grad(set_to_none=True)
        cascade_engine.cascade_grads(ids, targets)
        cascade_opt.step()

        packed_opt.zero_grad(set_to_none=True)
        logits, caches, top = packed_engine._forward_cache(ids)
        tape = pack_engine_cache(caches, top, packed_model, ids)
        decoded = deserialize(serialize(tape, packed_model), packed_model)
        grads = replay(decoded, packed_model, logits, targets)
        for name, param in packed_model.named_parameters():
            param.grad = grads[name].clone() if name in grads else None
        packed_opt.step()

        if step in checkpoints:
            mismatch = _first_tensor_mismatch(cascade_model, packed_model)
            opt_equal, opt_mismatch = _optimizer_equal(
                cascade_opt, packed_opt, cascade_model, packed_model)
            if mismatch is None and not opt_equal:
                mismatch = opt_mismatch
            digest_c = _update_digest(cascade_model, cascade_opt)
            digest_p = _update_digest(packed_model, packed_opt)
            equal = mismatch is None and digest_c == digest_p
            records.append(dict(step=step, bit_exact=equal,
                                first_mismatch=mismatch,
                                cascade_digest=digest_c,
                                packed_digest=digest_p))
            if not equal and first_mismatch is None:
                first_mismatch = dict(step=step, field=mismatch)
                break
    return dict(
        cell_id=f"trajectory_d{d}_l{layers}_t{T}_s{seed}",
        status="PASS" if first_mismatch is None and
        [r["step"] for r in records] == list(checkpoints) else "FAIL",
        d_model=d, n_layers=layers, T=T, seed=seed, steps=steps,
        checkpoints=list(checkpoints), records=records,
        first_mismatch=first_mismatch,
        wall_s=round(time.time() - start, 3),
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(value, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _descendant_pids(parent_pid: int) -> list[int]:
    """Snapshot all descendants, including children in escaped sessions."""
    try:
        import psutil

        return [int(child.pid) for child in psutil.Process(
            int(parent_pid)).children(recursive=True)]
    except (ImportError, OSError):
        pass
    except Exception:
        # ``NoSuchProcess`` and ``AccessDenied`` differ across psutil
        # versions. The dependency is optional, so fall through to the
        # platform process table without coupling the runner to its classes.
        pass
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid="], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return []
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        try:
            pid, ppid = (int(value) for value in line.split())
        except (TypeError, ValueError):
            continue
        children.setdefault(ppid, []).append(pid)
    found, stack = [], [int(parent_pid)]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, ()):
            if child not in found:
                found.append(child)
                stack.append(child)
    return found


def _send_pid(pid: int, sig: signal.Signals) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _send_group(pgid: int, sig: signal.Signals) -> bool:
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _terminate(proc: subprocess.Popen) -> dict[str, Any]:
    """Terminate the worker session and descendants that escaped it.

    The process-group id is the worker pid because workers start a new
    session. We still snapshot the process tree before signalling: compiler
    helpers are allowed to create their own sessions, and a dead group leader
    must not make us skip descendants that remain alive.
    """
    pgid = int(proc.pid)
    descendants = _descendant_pids(pgid)
    term_group = _send_group(pgid, signal.SIGTERM)
    term_pids = [pid for pid in reversed(descendants)
                 if _send_pid(pid, signal.SIGTERM)]
    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    # Always attempt the group kill, even if the leader exited after TERM:
    # the numeric process group can still contain live compiler children.
    kill_group = _send_group(pgid, signal.SIGKILL)
    refreshed = _descendant_pids(pgid) if proc.poll() is None else []
    all_descendants = list(dict.fromkeys(descendants + refreshed))
    kill_pids = [pid for pid in reversed(all_descendants)
                 if _send_pid(pid, signal.SIGKILL)]
    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover
            raise RuntimeError(
                f"worker {proc.pid} survived SIGKILL") from exc
    return dict(process_group=pgid, descendant_pids=all_descendants,
                sigterm_group_sent=term_group,
                sigterm_descendants=term_pids,
                sigkill_group_sent=kill_group,
                sigkill_descendants=kill_pids)


def run_subprocess_cell(command: list[str], log_path: Path,
                        heartbeat_path: Path, *, cell_id: str,
                        timeout_s: float, hard_deadline: float,
                        heartbeat_s: float = DEFAULT_HEARTBEAT_S) -> dict:
    """Run one worker with both a cell timeout and the run's hard wall."""
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        last_beat = 0.0
        reason = None
        termination = {}
        while proc.poll() is None:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= timeout_s:
                reason = "CELL_TIMEOUT"
                termination = _terminate(proc)
                break
            if now >= hard_deadline:
                reason = "HARD_WALL"
                termination = _terminate(proc)
                break
            if now - last_beat >= heartbeat_s:
                atomic_write_json(heartbeat_path, dict(
                    state="running", cell_id=cell_id, pid=proc.pid,
                    elapsed_s=round(elapsed, 3), timestamp=time.time()))
                last_beat = now
            time.sleep(min(0.25, max(heartbeat_s / 4.0, 0.01)))
        return dict(returncode=proc.returncode, process_group=proc.pid,
                    elapsed_s=round(time.monotonic() - started, 3),
                    stop_reason=reason, termination=termination)


def _worker_command(kind: str, output: Path, *, warm: int, reps: int,
                    order_seed: int, cell=None) -> list[str]:
    cmd = [sys.executable, "-m",
           "scripts.route_certificates.actual_model_benchmark",
           f"--worker-{kind}", "--worker-output", str(output),
           "--warmup", str(warm), "--reps", str(reps),
           "--order-seed", str(order_seed)]
    if cell is not None:
        d, layers, T, seed = cell
        cmd += ["--d-model", str(d), "--n-layers", str(layers),
                "--tokens", str(T), "--seed", str(seed)]
    return cmd


def _median(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(statistics.median(values), 3) if values else None


def _worker_main(args) -> None:
    if args.worker_cell:
        result = bench_config(args.d_model, args.n_layers, args.tokens,
                              args.seed, warm=args.warmup, reps=args.reps,
                              order_seed=args.order_seed)
    else:
        result = trajectory_parity(
            *TRAJECTORY_CONFIG, steps=TRAJECTORY_STEPS,
            checkpoints=TRAJECTORY_CHECKPOINTS)
    atomic_write_json(args.worker_output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cell-timeout-s", type=float,
                        default=DEFAULT_CELL_TIMEOUT_S)
    parser.add_argument("--hard-wall-s", type=float,
                        default=DEFAULT_HARD_WALL_S)
    parser.add_argument("--heartbeat-s", type=float,
                        default=DEFAULT_HEARTBEAT_S)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--order-seed", type=int, default=ORDER_SEED)
    # Hidden worker protocol.  Workers never create provenance directories.
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--worker-cell", action="store_true",
                         help=argparse.SUPPRESS)
    workers.add_argument("--worker-trajectory", action="store_true",
                         help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--d-model", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_cell or args.worker_trajectory:
        if args.worker_output is None:
            parser.error("worker mode requires --worker-output")
        if args.worker_cell and None in (
                args.d_model, args.n_layers, args.tokens, args.seed):
            parser.error("cell worker requires d/layers/tokens/seed")
        _worker_main(args)
        return

    cells = [(*cfg, seed) for cfg in CONFIGS for seed in SEEDS]
    if args.dry_run:
        print(json.dumps(dict(
            configs=list(CONFIGS), seeds=list(SEEDS), arms=list(ARMS),
            deterministic_arm_orders={
                f"d{d}_l{layers}_t{T}_s{seed}":
                deterministic_arm_order(d, layers, T, seed, args.order_seed)
                for d, layers, T, seed in cells},
            cell_timeout_s=args.cell_timeout_s,
            hard_wall_s=args.hard_wall_s,
            trajectory=dict(config=TRAJECTORY_CONFIG,
                            checkpoints=TRAJECTORY_CHECKPOINTS),
            output="results/route_certificates/runs/exp1_benchmark/<uuid>/"), indent=1))
        return

    if args.cell_timeout_s <= 0 or args.hard_wall_s <= 0:
        parser.error("timeouts must be positive")
    command = "python -m scripts.route_certificates.actual_model_benchmark"
    source_inputs = [Path(__file__).resolve(),
                     Path(sys.modules[
                         "tikitorch.research.pc_convrnn"].__file__).resolve(),
                     Path(sys.modules[
                         "tikitorch.autodiff.morphssm_tape"]
                          .__file__).resolve()]
    rec = run_record(
        "exp1_benchmark", command,
        config=dict(configs=list(CONFIGS), seeds=list(SEEDS), arms=list(ARMS),
                    warmup=args.warmup, reps=args.reps,
                    order_seed=args.order_seed,
                    cell_timeout_s=args.cell_timeout_s,
                    hard_wall_s=args.hard_wall_s,
                    heartbeat_s=args.heartbeat_s,
                    trajectory_config=TRAJECTORY_CONFIG,
                    trajectory_checkpoints=TRAJECTORY_CHECKPOINTS),
        inputs=source_inputs,
        modules=[sys.modules["tikitorch.research.pc_convrnn"],
                 sys.modules[
                     "tikitorch.autodiff.morphssm_tape"]])
    out_dir = Path(rec["_out_dir"])
    cell_dir, log_dir = out_dir / "cell_results", out_dir / "cell_logs"
    rows_path = out_dir / "benchmark.jsonl"
    trajectory_path = out_dir / "trajectory.jsonl"
    heartbeat = out_dir / "heartbeat.json"
    atomic_write_json(out_dir / "run_state.json", dict(
        state="running", run_id=rec["run_id"], started=time.time(),
        planned_cells=[f"d{d}_l{layers}_t{T}_s{seed}"
                       for d, layers, T, seed in cells]))
    # The previous incident had no durable provenance because the manifest
    # was deferred until every cell completed.  Persist the complete skeleton
    # before the first worker starts; ``manifest.json`` remains the finalized
    # record written by provenance.finish().
    atomic_write_json(out_dir / "manifest.start.json", {
        key: value for key, value in rec.items() if not key.startswith("_")})

    hard_deadline = time.monotonic() + args.hard_wall_s
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        d, layers, T, seed = cell
        cell_id = f"d{d}_l{layers}_t{T}_s{seed}"
        planned_order = deterministic_arm_order(
            d, layers, T, seed, args.order_seed)
        if time.monotonic() >= hard_deadline:
            row = dict(cell_id=cell_id, d_model=d, n_layers=layers, T=T,
                       seed=seed, arm_order=planned_order,
                       status="NOT_RUN_HARD_WALL")
            append_jsonl_durable(rows_path, row)
            rows.append(row)
            continue
        result_path = cell_dir / f"{cell_id}.json"
        cmd = _worker_command("cell", result_path, warm=args.warmup,
                              reps=args.reps, order_seed=args.order_seed,
                              cell=cell)
        proc = run_subprocess_cell(
            cmd, log_dir / f"{cell_id}.log", heartbeat,
            cell_id=cell_id, timeout_s=args.cell_timeout_s,
            hard_deadline=hard_deadline, heartbeat_s=args.heartbeat_s)
        if proc["stop_reason"]:
            row = dict(cell_id=cell_id, d_model=d, n_layers=layers, T=T,
                       seed=seed, arm_order=planned_order,
                       status=proc["stop_reason"], **proc)
        elif proc["returncode"] != 0 or not result_path.exists():
            row = dict(cell_id=cell_id, d_model=d, n_layers=layers, T=T,
                       seed=seed, arm_order=planned_order,
                       status="WORKER_ERROR", **proc)
        else:
            row = json.loads(result_path.read_text())
            row["worker_elapsed_s"] = proc["elapsed_s"]
        append_jsonl_durable(rows_path, row)
        rows.append(row)
        print(json.dumps(row), flush=True)

    trajectory = None
    if time.monotonic() < hard_deadline:
        result_path = cell_dir / "trajectory_5_100.json"
        cmd = _worker_command("trajectory", result_path, warm=args.warmup,
                              reps=args.reps, order_seed=args.order_seed)
        proc = run_subprocess_cell(
            cmd, log_dir / "trajectory_5_100.log", heartbeat,
            cell_id="trajectory_5_100", timeout_s=args.cell_timeout_s,
            hard_deadline=hard_deadline, heartbeat_s=args.heartbeat_s)
        if proc["stop_reason"]:
            trajectory = dict(cell_id="trajectory_5_100",
                              status=proc["stop_reason"], **proc)
        elif proc["returncode"] != 0 or not result_path.exists():
            trajectory = dict(cell_id="trajectory_5_100",
                              status="WORKER_ERROR", **proc)
        else:
            trajectory = json.loads(result_path.read_text())
            trajectory["worker_elapsed_s"] = proc["elapsed_s"]
    else:
        trajectory = dict(cell_id="trajectory_5_100",
                          status="NOT_RUN_HARD_WALL")
    append_jsonl_durable(trajectory_path, trajectory)

    complete = [row for row in rows if row.get("status") == "COMPLETE"]
    lifetime = [row for row in complete
                if (row["d_model"], row["n_layers"], row["T"]) == LIFETIME]
    gate = dict(
        n_cells_planned=len(cells), n_cells_complete=len(complete),
        n_cells_incident=len(cells) - len(complete),
        lifetime_eager_ms=_median(lifetime, "eager_ms"),
        lifetime_cascade_ms=_median(lifetime, "cascade_ms"),
        lifetime_compiled_ms=_median(lifetime, "compiled_ms"),
        lifetime_checkpoint_ms=_median(lifetime, "checkpoint_ms"),
        lifetime_replay_only_ms=_median(lifetime, "replay_only_ms"),
        lifetime_packed_full_cycle_ms=_median(
            lifetime, "packed_full_cycle_ms"),
        cascade_vs_eager_speedup=_median(lifetime, "cascade_vs_eager"),
        trajectory_5_100=trajectory.get("status"),
        boundary_accounting_complete=(
            len(lifetime) == len(SEEDS)
            and {row.get("seed") for row in lifetime} == set(SEEDS)
            and all(
            row.get("packed_state", {}).get("boundary_logits_bytes", 0) > 0
            and row.get("packed_state", {}).get("boundary_targets_bytes", 0) > 0
            for row in lifetime)),
    )
    atomic_write_json(out_dir / "gate_summary.json", gate)
    overall = ("completed" if len(complete) == len(cells)
               and trajectory.get("status") == "PASS"
               else "completed_with_incidents")
    atomic_write_json(heartbeat, dict(state=overall, timestamp=time.time(),
                                      completed_cells=len(complete),
                                      planned_cells=len(cells)))
    atomic_write_json(out_dir / "run_state.json", dict(
        state=overall, run_id=rec["run_id"], finished=time.time(),
        gate_summary=gate))
    finish(rec, metrics=gate, status=overall)
    print("GATE SUMMARY:", json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
