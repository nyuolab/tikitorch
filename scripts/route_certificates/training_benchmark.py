"""Matched route-training resource/learning benchmark (new, nonhistorical protocol).

One fresh worker per arm and batch/context cell. Compilation is fullgraph and
errors are recorded, never silently replaced by eager execution. Eager and
compiled checkpoint baselines each include per-layer, grouped and selective
policies when available. Selection is the fastest validated policy within the
declared GPU memory budget. Selective-policy coverage is observed separately
from timing and peak memory; compile-trace counts are not physical savings.
CPU process RSS includes interpreter/compiler state and is not tensor peak memory.
The synthetic successor task is a learning regression, not evidence of competitive
language modeling. Real-task publication measurements remain a separate gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache, partial
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from tikitorch.autodiff import morphssm_forward_tape, morphssm_vjp, route_logits
from tikitorch.models import MorphSSM

ARMS = ("eager", "compiled", "checkpoint", "compiled_checkpoint", "unpacked", "packed")
CHECKPOINT_ARMS = ("checkpoint", "compiled_checkpoint")
COMPILED_ARMS = ("compiled", "compiled_checkpoint", "unpacked", "packed")
BASELINE_ARMS = ("eager", "compiled", *CHECKPOINT_ARMS)
POLICIES = ("layer", "grouped", "selective")
ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=2)
def corpus_tokens(path):
    """Frozen contiguous 85/15 split; vocabulary learned only from training text."""
    text = Path(path).read_text(encoding="utf-8")
    split = int(len(text) * 0.85)
    alphabet = {char: i + 1 for i, char in enumerate(sorted(set(text[:split])))}
    encode = lambda value: torch.tensor([alphabet.get(c, 0) for c in value], dtype=torch.long)
    return encode(text[:split]), encode(text[split:]), len(alphabet) + 1


def make_model(config):
    torch.manual_seed(config["seed"])
    model = MorphSSM(d_model=config["width"], n_layers=config["layers"],
                     kernel_size=3, vocab_size=config["vocab"], task_type="lm",
                     dropout=0.0)
    model.set_tau(0)
    return model.to(device=config["device"], dtype=getattr(torch, config["dtype"])).eval()


def model_context_metadata(model):
    """Read the causal convolution span, which is not the timed sequence length."""
    layers = [dict(kernel_size=layer.morph_conv.kernel_size,
                   dilation=layer.morph_conv.dilation) for layer in model.layers]
    return dict(convolution_layers=layers,
                causal_receptive_field=1 + sum((layer["kernel_size"] - 1) * layer["dilation"]
                                               for layer in layers),
                scope="maximum causal input span; measured sequence length is not learned context capability")


def make_batch(config, seed):
    generator = torch.Generator().manual_seed(seed)
    if config.get("corpus"):
        train, heldout, _ = corpus_tokens(config["corpus"])
        data = heldout if config.get("split") == "eval" else train
        length = config["sequence"]
        if len(data) <= length:
            raise ValueError("each corpus split must exceed the sequence length")
        starts = torch.randint(len(data) - length, (config["batch"],), generator=generator)
        indices = starts[:, None] + torch.arange(length)
        return data[indices].to(config["device"]), data[indices + 1].to(config["device"])
    ids = torch.randint(config["vocab"], (config["batch"], config["sequence"]), generator=generator)
    return ids.to(config["device"]), ((ids + 1) % config["vocab"]).to(config["device"])


def digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def selective_checkpoint_ops():
    """Requested save targets; observing these requests is a separate check."""
    return (torch.ops.aten.sort.stable, torch.ops.aten.max.dim, torch.ops.aten.min.dim,
            torch.ops.aten.amax.default, torch.ops.aten.amin.default)


def checkpoint_forward(model, ids, policy, *, policy_callback=None):
    from torch.utils.checkpoint import checkpoint
    if policy not in POLICIES:
        raise ValueError(f"unknown checkpoint policy: {policy}")
    context = {}
    if policy == "selective":
        from torch.utils.checkpoint import create_selective_checkpoint_contexts
        # Save expensive routed reductions/sorts; recompute cheap pointwise work.
        save = list(selective_checkpoint_ops()) if policy_callback is None else policy_callback
        context["context_fn"] = partial(create_selective_checkpoint_contexts, save)
    hidden = model.dropout(model.embed(ids))
    group = 2 if policy == "grouped" else 1
    for start in range(0, len(model.layers), group):
        layers = tuple(model.layers[start:start + group])
        def block(x, modules=layers):
            for layer in modules:
                x = layer(x)
            return x
        hidden = checkpoint(block, hidden, use_reentrant=False, **context)
    return model.output_head(model.ln_final(hidden))


def selective_policy_coverage(config, arm):
    """Observe SAC decisions in a separate, untimed forward/backward.

    Compiled callbacks execute while tracing the graph, not once per device
    execution. Their counts establish that the requested ops reached the policy;
    they do not establish physical retention, kernel count, or saved compute.
    Eager recompute callbacks may be cache lookups, not recomputed operations.
    """
    from torch.utils.checkpoint import CheckpointPolicy
    if arm not in CHECKPOINT_ARMS:
        raise ValueError("selective coverage requires a checkpoint arm")
    save = selective_checkpoint_ops()
    counts = {str(op): dict(forward=0, recompute=0) for op in save}
    all_calls = dict(forward=0, recompute=0)
    visited = {}

    def policy(ctx, op, *args, **kwargs):
        phase = "recompute" if ctx.is_recompute else "forward"
        all_calls[phase] += 1
        observed = visited.setdefault(str(op), dict(forward=0, recompute=0,
                                      decision="MUST_SAVE" if op in save else "PREFER_RECOMPUTE"))
        observed[phase] += 1
        if op in save:
            counts[str(op)][phase] += 1
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    model = make_model(config)
    ids, targets = make_batch(config, config["seed"] + 100)
    forward = lambda x: checkpoint_forward(model, x, "selective", policy_callback=policy)
    compiled = arm == "compiled_checkpoint"
    if compiled:
        forward = torch.compile(forward, fullgraph=True, backend=config["compile_backend"])
    logits = forward(ids)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)).backward()
    synchronize(config["device"])
    seen = [op for op, phases in counts.items() if phases["forward"]]
    return dict(status="observed" if seen else "unobserved",
                scope="compile_trace_policy_calls" if compiled else "eager_runtime_policy_calls",
                separate_untimed_forward_backward=True,
                compile_fullgraph=compiled,
                compile_backend=config["compile_backend"] if compiled else None,
                requested_save_ops=list(counts), observed_save_ops=seen,
                unobserved_save_ops=[op for op in counts if op not in seen],
                target_policy_calls=counts, all_policy_calls=all_calls, visited_op_policy_calls=visited,
                physical_retention_or_recompute_savings_measured=False)


def route_gradient_function(model, ids, targets, packed):
    logits, tape = morphssm_forward_tape(model, ids, packed=packed)
    # Different algebraic CE derivative expressions can differ by a few ULPs;
    # AdamW can amplify near-zero differences. Use PyTorch's actual loss boundary.
    def boundary(z):
        return F.cross_entropy(z.reshape(-1, z.shape[-1]).float(), targets.reshape(-1))
    error, loss = torch.func.grad_and_value(boundary)(logits)
    return loss, morphssm_vjp(tape, model, error)


def adamw_epsilon(config):
    """Explicit shared optimizer protocol, never a parity-tolerance override."""
    eps = config.get("adamw_eps", 1e-8)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("AdamW epsilon must be finite and positive")
    return eps


def build_step(model, ids, targets, config, arm, policy="layer"):
    if arm not in ARMS:
        raise ValueError(f"unknown benchmark arm: {arm}")
    if config.get("optimizer", "adamw") == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01,
                                      eps=adamw_epsilon(config))
    forward = model
    routed = None
    if arm == "compiled":
        forward = torch.compile(model, fullgraph=True, backend=config["compile_backend"])
    elif arm in ("packed", "unpacked"):
        def gradient_fn(x, y):
            return route_gradient_function(model, x, y, arm == "packed")
        routed = torch.compile(gradient_fn, fullgraph=True, backend=config["compile_backend"])
    elif arm in CHECKPOINT_ARMS:
        forward = lambda x: checkpoint_forward(model, x, policy)
        if arm == "compiled_checkpoint":
            forward = torch.compile(forward, fullgraph=True, backend=config["compile_backend"])

    def step():
        optimizer.zero_grad(set_to_none=True)
        if routed is None:
            logits = forward(ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
            loss.backward()
        else:
            loss, grads = routed(ids, targets)
            for name, parameter in model.named_parameters():
                parameter.grad = grads.get(name)
        optimizer.step()
        return loss.detach()
    return step


def parameter_state_cpu(model):
    return {name: (p.detach().cpu().clone(), p.grad.cpu().clone())
            for name, p in model.named_parameters()}


def verify_arm(config, arm, policy, *, observed=None):
    """Check the full measured update count, using measured final state if supplied."""
    updates = 1 + config["warmup"] + config["reps"]
    reference = make_model(config)
    ids, targets = make_batch(config, config["seed"] + 100)
    reference_step = build_step(reference, ids, targets, config, "eager")
    for _ in range(updates):
        expected_loss = reference_step()
    expected = parameter_state_cpu(reference)
    expected_loss = expected_loss.cpu()
    del reference_step, reference
    gc.collect()
    if observed is None:
        measured = make_model(config)
        measured_step = build_step(measured, ids, targets, config, arm, policy)
        for _ in range(updates):
            actual_loss = measured_step()
        actual_loss, actual = actual_loss.cpu(), parameter_state_cpu(measured)
    else:
        actual_loss, actual = observed
    tolerance = 0.05 if config["dtype"] == "bfloat16" else 1e-4
    errors = []
    actual_loss = actual_loss.cpu()
    passed = torch.allclose(expected_loss, actual_loss, atol=tolerance, rtol=tolerance)
    for name, (a, ag) in expected.items():
        b, bg = actual[name]
        grad_error = float((ag - bg).abs().max())
        param_error = float((a - b).abs().max().detach())
        errors.append(dict(parameter=name, gradient_max_abs=grad_error, update_max_abs=param_error))
        passed = passed and torch.allclose(ag, bg, atol=tolerance, rtol=tolerance)
        passed = passed and torch.allclose(a, b, atol=tolerance, rtol=tolerance)
    return dict(status="pass" if passed else "fail", updates_checked=updates,
                compared_measured_state=observed is not None, atol=tolerance, rtol=tolerance,
                loss_max_abs=float((expected_loss - actual_loss).abs()), parameters=errors)


def measure_cell(config, arm, policy="layer"):
    torch.set_num_threads(1)
    if torch.device(config["device"]).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    model = make_model(config)
    ids, targets = make_batch(config, config["seed"] + 100)
    result = dict(arm=arm, checkpoint_policy=policy if arm in CHECKPOINT_ARMS else None,
                  init_sha256=digest(model.parameters()), batch_sha256=digest((ids, targets)),
                  config=config, compile_fullgraph=arm in COMPILED_ARMS,
                  compile_backend=config["compile_backend"] if arm in COMPILED_ARMS else None,
                  optimizer_execution="eager", model_context=model_context_metadata(model))
    step = build_step(model, ids, targets, config, arm, policy)
    synchronize(config["device"])
    begin = time.perf_counter()
    loss = step()
    synchronize(config["device"])
    result["startup_and_first_update_ms"] = (time.perf_counter() - begin) * 1000
    for _ in range(config["warmup"]):
        loss = step()
    synchronize(config["device"])
    cuda = torch.device(config["device"]).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(config["device"])
    samples = []
    for _ in range(config["reps"]):
        begin = time.perf_counter()
        loss = step()
        synchronize(config["device"])
        samples.append((time.perf_counter() - begin) * 1000)
    # Capture process peak BEFORE auxiliary accounting or validation forwards.
    if sys.platform != "win32":
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    else:
        result["process_peak_rss_bytes"] = None
    result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(config["device"]) if cuda else None
    result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(config["device"]) if cuda else None
    result["cpu_tensor_peak_bytes"] = None  # RSS is not an allocator measurement.
    result["samples_ms"] = samples
    result["median_ms"] = statistics.median(samples)
    result["mad_ms"] = statistics.median(abs(x - result["median_ms"]) for x in samples)
    result["update_count"] = 1 + config["warmup"] + config["reps"]
    result["final_loss"] = float(loss)
    result["final_params_sha256"] = digest(model.parameters())
    if not math.isfinite(result["final_loss"]):
        raise RuntimeError("nonfinite training loss")
    result["measurement_status"] = "fit"
    observed = (loss.cpu(), parameter_state_cpu(model))
    # Auxiliary retained-state pass, outside all timed/peak measurements.
    try:
        with torch.no_grad():
            logits, tape = morphssm_forward_tape(model, ids, packed=arm != "unpacked")
        result["auxiliary_route_payload_bytes"] = tape.nbytes
        result["auxiliary_route_storage_packed"] = tape.packed
        result["auxiliary_accounting_scope"] = "separate route forward; not autograd/checkpoint retained state"
        result["loss_boundary_bytes"] = logits.numel() * logits.element_size() + targets.numel() * targets.element_size()
        del tape, logits
    except Exception as exc:
        result["auxiliary_accounting_error"] = f"{type(exc).__name__}: {exc}"
    del step, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    try:
        result["parity"] = verify_arm(config, arm, policy, observed=observed)
    except Exception as exc:
        # A reference OOM must not be mislabeled a failure of the packed cell
        # to fit. Keep measured fit, but prohibit any unvalidated comparison.
        result["parity"] = dict(status="unavailable", error=f"{type(exc).__name__}: {exc}")
    result["status"] = "ok" if result["parity"]["status"] == "pass" else "parity_failed"
    if arm in CHECKPOINT_ARMS and policy == "selective":
        try:
            result["selective_policy_coverage"] = selective_policy_coverage(config, arm)
        except Exception as exc:
            result["selective_policy_coverage"] = dict(
                status="unavailable", error=f"{type(exc).__name__}: {exc}")
        if result["status"] == "ok" and result["selective_policy_coverage"]["status"] != "observed":
            result["status"] = "coverage_unavailable"
    return result


def trajectory_check(config, steps=5):
    reference = make_model(config)
    routed = deepcopy(reference)
    optimizers = [torch.optim.SGD(m.parameters(), lr=1e-4) for m in (reference, routed)]
    max_grad_error = max_param_error = 0.0
    for step in range(steps):
        ids, targets = make_batch(config, config["seed"] + 1000 + step)
        for model, opt in zip((reference, routed), optimizers):
            opt.zero_grad(set_to_none=True)
            logits = model(ids) if model is reference else route_logits(model, ids)
            F.cross_entropy(logits.reshape(-1, config["vocab"]).float(), targets.reshape(-1)).backward()
        for a, b in zip(reference.parameters(), routed.parameters()):
            max_grad_error = max(max_grad_error, float((a.grad - b.grad).abs().max()))
        for opt in optimizers:
            opt.step()
        for a, b in zip(reference.parameters(), routed.parameters()):
            max_param_error = max(max_param_error, float((a - b).abs().max().detach()))
    return dict(steps=steps, max_grad_error=max_grad_error, max_param_error=max_param_error,
                status="pass" if max_grad_error < 1e-4 and max_param_error < 1e-5 else "fail")


def learning_check(config, steps=50):
    """Held-out learning; synthetic regression has an explicit acceptance gate."""
    torch.set_num_threads(1)
    tropical = make_model(config)
    torch.manual_seed(config["seed"])
    baseline = "dense_token_baseline"
    if config.get("corpus"):
        class DenseSequenceBaseline(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(config["vocab"], config["width"])
                self.rnn = torch.nn.GRU(config["width"], config["width"], batch_first=True)
                self.head = torch.nn.Linear(config["width"], config["vocab"])
            def forward(self, ids):
                return self.head(self.rnn(self.embed(ids))[0])
        dense = DenseSequenceBaseline()
        baseline = "gru_character_baseline"
    else:
        dense = torch.nn.Sequential(torch.nn.Embedding(config["vocab"], config["width"]),
                                    torch.nn.Linear(config["width"], config["vocab"]))
    dense = dense.to(device=config["device"], dtype=getattr(torch, config["dtype"]))
    eval_ids, eval_targets = make_batch({**config, "batch": 32, "split": "eval"}, config["seed"] + 90000)
    results = {}
    for name, model in (("route_morphssm", tropical), (baseline, dense)):
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, eps=adamw_epsilon(config))
        def evaluate():
            with torch.no_grad():
                logits = model(eval_ids)
                loss = F.cross_entropy(logits.reshape(-1, config["vocab"]).float(), eval_targets.reshape(-1))
                accuracy = (logits.argmax(-1) == eval_targets).float().mean()
                if not torch.isfinite(loss) or not torch.isfinite(accuracy):
                    raise RuntimeError("nonfinite held-out learning metrics")
                return dict(loss=float(loss), accuracy=float(accuracy),
                            perplexity=math.exp(min(float(loss), 700)))
        before = evaluate()
        for step in range(steps):
            ids, targets = make_batch(config, config["seed"] + 10000 + step)
            opt.zero_grad(set_to_none=True)
            logits = route_logits(model, ids) if name == "route_morphssm" else model(ids)
            F.cross_entropy(logits.reshape(-1, config["vocab"]).float(), targets.reshape(-1)).backward()
            opt.step()
        results[name] = dict(before=before, after=evaluate(), updates=steps,
                             parameters=sum(p.numel() for p in model.parameters()))
    synthetic = not config.get("corpus")
    passes = all(m["after"]["loss"] < m["before"]["loss"] * 0.75
                 and m["after"]["accuracy"] > 0.8 for m in results.values())
    return dict(task="synthetic token successor; regression only" if synthetic else "held-out character prediction",
                models=results, status=("pass" if passes else "fail") if synthetic else "observed",
                acceptance="each model: loss < 0.75 * initial loss and accuracy > 0.8" if synthetic else "no universal corpus quality threshold; metrics only",
                optimizer=dict(name="AdamW", lr=0.01, eps=adamw_epsilon(config),
                               betas=[0.9, 0.999], weight_decay=0.01, execution="eager",
                               scope="learning control; separate from the timed update recipe"),
                train_seed_start=config["seed"] + 10000, eval_seed=config["seed"] + 90000)


def summarize(rows, memory_budget_bytes=None):
    """Never infer a winning baseline or capacity beyond successfully run cells."""
    successful = [r for r in rows if r.get("status") == "ok"
                  and (r.get("arm") not in CHECKPOINT_ARMS
                       or r.get("checkpoint_policy") != "selective"
                       or r.get("selective_policy_coverage", {}).get("status") == "observed")]
    matching = all(len({r.get(key) for r in successful}) == 1
                   and all(r.get(key) is not None for r in successful)
                   for key in ("init_sha256", "batch_sha256", "update_count"))
    if memory_budget_bytes is not None:
        successful = [r for r in successful if r.get("cuda_peak_allocated_bytes") is not None
                       and r["cuda_peak_allocated_bytes"] <= memory_budget_bytes]
    by_arm = {arm: [r for r in successful if r["arm"] == arm] for arm in ARMS}
    best = {arm: min(cells, key=lambda r: r["median_ms"]) for arm, cells in by_arm.items() if cells}
    complete = set(best) == set(ARMS) and matching
    baseline = min((best[a] for a in BASELINE_ARMS), key=lambda r: r["median_ms"]) if complete else None
    return dict(status="complete" if complete else "inconclusive",
                matched_initialization_data_updates=matching,
                selected_checkpoint_policy=best.get("checkpoint", {}).get("checkpoint_policy"),
                selected_compiled_checkpoint_policy=best.get("compiled_checkpoint", {}).get("checkpoint_policy"),
                best_baseline_arm=baseline["arm"] if complete else None,
                best_baseline_checkpoint_policy=baseline.get("checkpoint_policy") if complete else None,
                packed_speedup_vs_best_baseline=baseline["median_ms"] / best["packed"]["median_ms"] if complete else None,
                missing_arms=sorted(set(ARMS) - set(best)))


def validated_worker_row(row, outcome):
    """A partial result cannot override a failed or timed-out worker process."""
    row = {**row, "worker_outcome": outcome}
    if outcome.get("returncode") != 0 or outcome.get("stop_reason") is not None:
        if row.get("status") == "ok":
            row.update(status="worker_failed", measurement_status="unverified",
                       parity=dict(status="unavailable", error="worker did not finish successfully"))
    return row


def protocol_source_files(corpus=None):
    """Hash executable protocol helpers as well as the library and input data."""
    runner = Path(__file__).resolve()
    paths = [runner, runner.with_name("actual_model_benchmark.py"),
             runner.with_name("provenance.py"), *sorted((ROOT / "tikitorch").rglob("*.py"))]
    if corpus is not None:
        paths.append(Path(corpus).resolve())
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64", "bfloat16"))
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=11)
    parser.add_argument("--batches", type=int, nargs="+", default=[2])
    parser.add_argument("--sequences", type=int, nargs="+", default=[12])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--adamw-eps", type=float, default=1e-8,
                        help="Shared AdamW epsilon for every arm and learning control; default unchanged (1e-8)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--compile-backend", default="inductor", choices=("inductor", "aot_eager"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--memory-budget-mib", type=float)
    parser.add_argument("--learning-steps", type=int, default=50)
    parser.add_argument("--corpus", type=Path, help="Optional UTF-8 text for disjoint 85/15 character-LM train/eval")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        job = json.loads(args.worker)
        try:
            row = measure_cell(job["config"], job["arm"], job["policy"])
        except Exception as exc:
            row = dict(status="error", arm=job["arm"], checkpoint_policy=job["policy"],
                       config=job["config"], error=f"{type(exc).__name__}: {exc}")
        Path(job["output"]).write_text(json.dumps(row, indent=2))
        return 0 if row["status"] == "ok" else 1
    if min(args.width, args.layers, args.vocab, args.reps, args.learning_steps, *args.batches, *args.sequences) <= 0 or args.warmup < 0:
        parser.error("dimensions/reps/learning steps must be positive; warmup nonnegative")
    if args.memory_budget_mib is not None and args.device != "cuda":
        parser.error("allocator memory budgets require CUDA; CPU RSS is not tensor memory")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("timeout and learning rate must be finite and positive")
    if not math.isfinite(args.adamw_eps) or args.adamw_eps <= 0:
        parser.error("AdamW epsilon must be finite and positive")
    if args.memory_budget_mib is not None and (not math.isfinite(args.memory_budget_mib) or args.memory_budget_mib <= 0):
        parser.error("memory budget must be finite and positive")
    config = {k: v for k, v in vars(args).items() if k in (
        "device", "dtype", "width", "layers", "vocab", "seed", "lr", "optimizer", "adamw_eps", "warmup", "reps", "compile_backend")}
    if args.corpus:
        config["corpus"] = str(args.corpus.resolve())
        _, _, config["vocab"] = corpus_tokens(config["corpus"])
    cells = [{**config, "batch": b, "sequence": t} for b in args.batches for t in args.sequences]
    if args.dry_run:
        print(json.dumps(dict(cells=cells, arms=ARMS, checkpoint_arms=CHECKPOINT_ARMS,
                              checkpoint_policies=POLICIES), indent=2))
        return 0
    from scripts.route_certificates.actual_model_benchmark import run_subprocess_cell
    from scripts.route_certificates.provenance import file_digest, finish, run_record
    source_files = protocol_source_files(args.corpus)
    record = run_record("device_route_training", " ".join(sys.argv),
                        dict(cells=cells, learning_steps=args.learning_steps,
                             protocol="device-route-training-v2"),
                        inputs=source_files, device=args.device, output_root=args.output)
    out = Path(record["_out_dir"])
    (out / "manifest.start.json").write_text(json.dumps(record, indent=2))
    rows, summaries = [], []
    for cell_index, cell in enumerate(cells):
        cell_rows = []
        for arm in ARMS:
            for policy in POLICIES if arm in CHECKPOINT_ARMS else ("layer",):
                name = f"{cell_index}-{arm}-{policy}"
                destination = out / f"{name}.json"
                job = dict(config=cell, arm=arm, policy=policy, output=str(destination))
                outcome = run_subprocess_cell(
                    [sys.executable, "-m", "scripts.route_certificates.training_benchmark", "--worker", json.dumps(job)],
                    out / f"{name}.log", out / "heartbeat.json", cell_id=name,
                    timeout_s=args.timeout, hard_deadline=time.monotonic() + args.timeout + 10,
                    heartbeat_s=min(10, args.timeout / 2))
                row = json.loads(destination.read_text()) if destination.exists() else dict(
                    arm=arm, checkpoint_policy=policy, config=cell, status="error", error=outcome)
                row = validated_worker_row(row, outcome)
                cell_rows.append(row)
                print(f"{name}: {row['status']}", flush=True)
        rows.extend(cell_rows)
        summaries.append(dict(config=cell, **summarize(cell_rows,
            None if args.memory_budget_mib is None else args.memory_budget_mib * 1024 ** 2)))
    checks = {}
    try:
        checks["trajectory"] = trajectory_check(cells[0])
        checks["learning"] = learning_check(cells[0], args.learning_steps)
    except Exception as exc:
        checks["error"] = f"{type(exc).__name__}: {exc}"
    capacity = {arm: [dict(batch=r["config"]["batch"], sequence=r["config"]["sequence"],
                          validation=r.get("parity", {}).get("status"))
                     for r in rows if r["arm"] == arm and r.get("measurement_status") == "fit"
                     and (args.memory_budget_mib is None or r["cuda_peak_allocated_bytes"] <= args.memory_budget_mib * 1024 ** 2)] for arm in ARMS}
    checks["sources_unchanged"] = all(file_digest(item["path"])["sha256"] == item["sha256"]
                                      for item in record["inputs"])
    complete = (all(s["status"] == "complete" for s in summaries)
                and checks.get("trajectory", {}).get("status") == "pass"
                and checks.get("learning", {}).get("status") == ("observed" if args.corpus else "pass")
                and "error" not in checks and checks["sources_unchanged"])
    manifest = finish(record, dict(rows=rows, summaries=summaries, checks=checks,
                                   measured_fit_cells=capacity,
                                   capacity_scope="measured fits with separate validation status; unvalidated fits are not wins; no extrapolated maximum",
                                   platform=platform.platform()),
                      status="completed" if complete else "inconclusive")
    print(manifest)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
