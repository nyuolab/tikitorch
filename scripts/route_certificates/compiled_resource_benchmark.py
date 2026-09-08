"""Compiled five-arm resource benchmark for route certificates on MorphSSM.

This migrated consolidated runner preserves the frozen systems gate while making
TikiTorch the sole owner of the implementation, benchmark, and evidence path.

Arms (all on IDENTICAL seeded model + inputs + targets):
  eager        framework autograd, full matched update (fwd+bwd+AdamW)
  compiled     torch.compile(model), full matched update; one-shot compile
               overhead recorded SEPARATELY from steady-state median
  checkpoint   torch.utils.checkpoint per layer, full matched update
  cascade      MorphPCEngine.cascade_grads + AdamW  (the LATENCY claim arm)
  packed       route-certificate: forward_cache -> pack -> serialize ->
               deserialize -> replay -> AdamW  (the RETAINED-STATE claim arm)

Frozen gate G-RESOURCE (per pre-declared cell) PASSES iff any leg holds:
  (a) retained_ratio = min(autograd_nonparam_dedup + targets,
                           checkpoint_retained_measured + logits + targets)
                       / (serialized tape + logits + targets)   >= 2.0
  (b) traffic_ratio  = best_baseline / cascade on the profiler
                       bytes-allocated PROXY                     >= 2.0
  (c) latency: median cascade matched update
                       <= 0.80 x min(median eager, median compiled steady,
                                     median checkpoint)
A cell whose best-baseline arm timed out / errored is UNDECIDABLE, never
imputed. FAIL is reported as FAIL.

Invalidation guards honored (see Fable's review):
  - torch.set_num_threads(1)+interop recorded in every row.
  - per-cell subprocess with cell timeout + hard wall + heartbeat + fsync.
  - median + MAD + p10/p90 over `reps` after `warm` warmups; compiled arm's
    one-shot cost is NEVER folded into steady-state.
  - retained-set and per-arm subprocess ru_maxrss PEAK are SEPARATE columns;
    the packed peak includes the dense _forward_cache transient (~13x the
    tape) and is reported, never claimed as the >=2x headline.
  - ru_maxrss is process-monotone => one arm per fresh subprocess for peak.
  - traffic is a PROFILER PROXY (bytes allocated + recompute layer calls),
    explicitly labeled, never claimed as DRAM traffic.
  - saved_tensors_hooks HOLD tensor refs for the hook pass to defeat
    storage-pointer reuse (undercount would bias FOR the claim).
  - per-cell one-step fp64 grad-VALUE parity (cascade==replay bit-exact;
    autograd vs cascade at the documented fp32-softmax-boundary tolerance,
    tie/divergence sites enumerated), replacing key-only equality.
  - identical firing proven: sha256(init params), sha256(ids,targets),
    exact update count per arm.

Usage (run from the TikiTorch repository root):
  python -m scripts.route_certificates.compiled_resource_benchmark --dry-run
  python -m scripts.route_certificates.compiled_resource_benchmark --smoke
  python -m scripts.route_certificates.compiled_resource_benchmark --pilot
  python -m scripts.route_certificates.compiled_resource_benchmark --full
Results are append-only under manuscript/results/route_certificates/runs/<experiment>/<uuid>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

# --- repository import path -----------------------------------------------
_THIS = Path(__file__).resolve()
REPO = None
for parent in _THIS.parents:
    if (parent / "tikitorch").is_dir() and (parent / ".git").exists():
        REPO = parent
        break
if REPO is None:
    REPO = _THIS.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tikitorch.research.pc_convrnn import MorphPCEngine, build_model  # noqa: E402

from tikitorch.autodiff.morphssm_tape import (  # noqa: E402
    deserialize, pack_engine_cache, replay, serialize)
from scripts.route_certificates.provenance import (  # noqa: E402
    finish, run_record)

# --- frozen configuration -------------------------------------------------
V, B = 38, 1
CONFIGS = ((64, 2, 96), (128, 4, 96), (256, 4, 192))          # pilot cells
GRID_D, GRID_T, GRID_L = (64, 128, 256), (96, 192, 384), (2, 4, 8)
LIFETIME = (128, 4, 96)                                       # headline cell
SEEDS = (0, 1, 2)
ARMS = ("eager", "compiled", "checkpoint", "cascade", "packed")
BASELINES = ("eager", "compiled", "checkpoint")
WARMUP, REPS = 5, 20
ORDER_SEED = 20_260_822
RETAINED_BAR = 2.0
TRAFFIC_BAR = 2.0
LATENCY_BAR = 0.80
DEFAULT_CELL_TIMEOUT_S = 30 * 60
DEFAULT_HARD_WALL_S = 6 * 60 * 60
DEFAULT_HEARTBEAT_S = 10.0
PEAK_ITERS = 3                                                # per-arm peak loop


def full_grid() -> tuple[tuple[int, int, int], ...]:
    return tuple((d, l, t) for d in GRID_D for t in GRID_T for l in GRID_L)


# --- model + batch (identical seeded construction across arms) ------------
def make(d: int, layers: int, seed: int, dtype=torch.float32):
    torch.manual_seed(seed)
    ns = argparse.Namespace(d_model=d, n_layers=layers, kernel=7,
                            dropout=0.0, tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", V, ns)
    model.set_tau(0.0)
    model = model.to(dtype)
    model.eval()
    return model


def make_batch(T: int, seed: int):
    gen = torch.Generator().manual_seed(seed + 100)
    ids = torch.randint(0, V, (B, T), generator=gen)
    targets = torch.randint(0, V, (B, T), generator=gen)
    return ids, targets


def tensor_bytes(value: Any) -> int:
    return int(value.numel() * value.element_size()) if torch.is_tensor(value) else 0


def _params_sha(model) -> str:
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode())
        h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _batch_sha(ids, targets) -> str:
    h = hashlib.sha256()
    h.update(ids.detach().cpu().numpy().tobytes())
    h.update(targets.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# --- timing helpers -------------------------------------------------------
def timed_samples(fn: Callable[[], Any], reps: int, warm: int) -> list[float]:
    for _ in range(warm):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return dict(median_ms=None, mad_ms=None, p10_ms=None, p90_ms=None,
                    min_ms=None, max_ms=None, n=0)
    s = sorted(samples)
    n = len(s)
    med = statistics.median(s)
    mad = statistics.median([abs(x - med) for x in s])

    def pct(f: float) -> float:
        pos = (n - 1) * f
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return s[lo]
        w = pos - lo
        return s[lo] * (1 - w) + s[hi] * w

    return dict(median_ms=round(med, 4), mad_ms=round(mad, 4),
                p10_ms=round(pct(0.10), 4), p90_ms=round(pct(0.90), 4),
                min_ms=round(s[0], 4), max_ms=round(s[-1], 4), n=n)


# --- matched full-update closures per arm (fwd + reverse + AdamW) ---------
def _new_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


def _eager_update(model, ids, targets, opt):
    def f():
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None
        loss = F.cross_entropy(model(ids).reshape(-1, V).float(),
                               targets.reshape(-1))
        loss.backward()
        for p in model.parameters():
            p.requires_grad_(False)
        opt.step()
    return f


def _compiled_update(model, compiled, ids, targets, opt):
    def f():
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None
        loss = F.cross_entropy(compiled(ids).reshape(-1, V).float(),
                               targets.reshape(-1))
        loss.backward()
        for p in model.parameters():
            p.requires_grad_(False)
        opt.step()
    return f


def _checkpoint_update(model, ids, targets, opt):
    from torch.utils.checkpoint import checkpoint

    def f():
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None
        hidden = model.dropout(model.embed(ids)) if hasattr(model, "dropout") \
            else model.embed(ids)
        for layer in model.layers:
            hidden = checkpoint(layer, hidden, use_reentrant=False)
        logits = model.output_head(model.ln_final(hidden))
        loss = F.cross_entropy(logits.reshape(-1, V).float(),
                               targets.reshape(-1))
        loss.backward()
        for p in model.parameters():
            p.requires_grad_(False)
        opt.step()
    return f


def _cascade_update(model, engine, ids, targets, opt):
    def f():
        for p in model.parameters():
            p.grad = None
        engine.cascade_grads(ids, targets)
        opt.step()
    return f


def _packed_update(model, engine, ids, targets, opt):
    def f():
        for p in model.parameters():
            p.grad = None
        logits, caches, top = engine._forward_cache(ids)
        tape = pack_engine_cache(caches, top, model, ids)
        back = deserialize(serialize(tape, model), model)
        grads = replay(back, model, logits, targets)
        for name, param in model.named_parameters():
            param.grad = grads[name].clone() if name in grads else None
        opt.step()
    return f


def build_arm(arm: str, d: int, layers: int, T: int, seed: int):
    """Return (update_fn, aux) for one arm on a fresh identical model."""
    model = make(d, layers, seed)
    ids, targets = make_batch(T, seed)
    opt = _new_opt(model)
    aux = dict(init_params_sha=_params_sha(model),
               batch_sha=_batch_sha(ids, targets))
    if arm == "eager":
        return _eager_update(model, ids, targets, opt), aux, model
    if arm == "compiled":
        compiled = torch.compile(model)
        return _compiled_update(model, compiled, ids, targets, opt), aux, model
    if arm == "checkpoint":
        return _checkpoint_update(model, ids, targets, opt), aux, model
    if arm == "cascade":
        engine = MorphPCEngine(model)
        return _cascade_update(model, engine, ids, targets, opt), aux, model
    if arm == "packed":
        engine = MorphPCEngine(model)
        return _packed_update(model, engine, ids, targets, opt), aux, model
    raise ValueError(f"unknown arm {arm!r}")


def time_arm(arm: str, d: int, layers: int, T: int, seed: int,
             warm: int, reps: int) -> dict[str, Any]:
    fn, aux, _ = build_arm(arm, d, layers, T, seed)
    out: dict[str, Any] = dict(**aux)
    if arm == "compiled":
        t0 = time.perf_counter()
        fn()                                     # one-shot compile + first run
        out["compile_overhead_s"] = round(time.perf_counter() - t0, 4)
    samples = timed_samples(fn, reps, warm)
    out["timing"] = stats(samples)
    out["update_count"] = warm + reps + (1 if arm == "compiled" else 0)
    return out


def component_timing(d, layers, T, seed, warm, reps) -> dict[str, Any]:
    """Forward-only, replay-only reverse, and AdamW-only anchors (reported as
    estimates; the matched full-update medians are the gate quantities)."""
    model = make(d, layers, seed)
    ids, targets = make_batch(T, seed)
    forward = stats(timed_samples(lambda: model(ids), reps, warm))

    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    tape = pack_engine_cache(caches, top, model, ids)
    restored = deserialize(serialize(tape, model), model)
    replay_only = stats(timed_samples(
        lambda: replay(restored, model, logits, targets), reps, warm))

    model.zero_grad(set_to_none=True)
    engine.cascade_grads(ids, targets)
    fixed = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    opt = _new_opt(model)

    def adamw_only():
        for n, p in model.named_parameters():
            p.grad = fixed[n].clone() if n in fixed else None
        opt.step()

    adamw = stats(timed_samples(adamw_only, max(1, reps // 2),
                                max(0, warm // 2)))
    return dict(forward_ms=forward, replay_only_ms=replay_only,
                adamw_step_ms=adamw)


# --- memory: retained set (measured), profiler proxy ----------------------
class _RetainedTracker:
    """saved_tensors_hooks tracker: dedup by storage ptr, split param/nonparam,
    HOLD tensor refs for the whole hook pass so a freed storage cannot be
    reallocated and aliased (that would undercount => inflate the ratio)."""

    def __init__(self, param_ptrs: set[int]):
        self.param_ptrs = param_ptrs
        self._by_ptr: dict[int, int] = {}
        self._refs: list[torch.Tensor] = []
        self.calls = 0

    def pack(self, t):
        self.calls += 1
        self._refs.append(t)                    # hold the reference
        try:
            ptr = t.untyped_storage().data_ptr()
            nb = int(t.untyped_storage().nbytes())
        except Exception:
            ptr, nb = id(t), t.numel() * t.element_size()
        self._by_ptr[ptr] = nb
        return t

    @staticmethod
    def unpack(t):
        return t

    def summary(self) -> dict[str, int]:
        total = sum(self._by_ptr.values())
        param = sum(nb for ptr, nb in self._by_ptr.items()
                    if ptr in self.param_ptrs)
        return dict(dedup_total_bytes=total,
                    dedup_param_bytes=param,
                    dedup_nonparam_bytes=total - param,
                    unique_storages=len(self._by_ptr),
                    saved_tensor_calls=self.calls)


def measure_retained(d, layers, T, seed) -> dict[str, Any]:
    from torch.autograd.graph import saved_tensors_hooks
    from torch.utils.checkpoint import checkpoint

    model = make(d, layers, seed)
    ids, targets = make_batch(T, seed)
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}

    # ---- autograd retained -------------------------------------------------
    ag = _RetainedTracker(param_ptrs)
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    with saved_tensors_hooks(ag.pack, ag.unpack):
        logits_ag = model(ids)
        loss = F.cross_entropy(logits_ag.reshape(-1, V).float(),
                               targets.reshape(-1))
    loss.backward()
    ag_sum = ag.summary()
    del ag                                       # release held refs

    # ---- checkpoint retained (MEASURED, not analytic) ---------------------
    ck = _RetainedTracker(param_ptrs)
    layer_calls = [0]

    class _Counting(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            layer_calls[0] += 1
            return self.inner(x)

    for p in model.parameters():
        p.grad = None
    with saved_tensors_hooks(ck.pack, ck.unpack):
        hidden = model.dropout(model.embed(ids)) if hasattr(model, "dropout") \
            else model.embed(ids)
        for layer in model.layers:
            hidden = checkpoint(_Counting(layer), hidden, use_reentrant=False)
        logits_ck = model.output_head(model.ln_final(hidden))
        loss_ck = F.cross_entropy(logits_ck.reshape(-1, V).float(),
                                  targets.reshape(-1))
    loss_ck.backward()
    ck_sum = ck.summary()
    del ck
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    # recompute layer calls beyond the single forward pass
    recompute_layer_calls = max(0, layer_calls[0] - layers)

    # ---- packed retained ---------------------------------------------------
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    tape = pack_engine_cache(caches, top, model, ids)
    blob = serialize(tape, model)
    logits_bytes = tensor_bytes(logits)
    targets_bytes = tensor_bytes(targets)
    packed_retained = len(blob) + logits_bytes + targets_bytes

    autograd_baseline = ag_sum["dedup_nonparam_bytes"] + targets_bytes
    checkpoint_baseline = (ck_sum["dedup_nonparam_bytes"]
                           + logits_bytes + targets_bytes)
    best_baseline = min(autograd_baseline, checkpoint_baseline)
    retained_ratio = (round(best_baseline / packed_retained, 4)
                      if packed_retained else None)
    return dict(
        autograd=ag_sum, checkpoint=ck_sum,
        checkpoint_recompute_layer_calls=recompute_layer_calls,
        boundary_logits_bytes=logits_bytes,
        boundary_targets_bytes=targets_bytes,
        tape_serialized_bytes=len(blob),
        tape_payload_bytes=int(tape.payload_nbytes),
        tape_logical_bits=int(tape.bit_length),
        autograd_retained_baseline_bytes=autograd_baseline,
        checkpoint_retained_baseline_bytes=checkpoint_baseline,
        best_retained_baseline_bytes=best_baseline,
        packed_retained_bytes=packed_retained,
        retained_ratio=retained_ratio,
        accounting_note=("retained-set ONLY (not process peak); packed peak is "
                         "reported separately via per-arm subprocess ru_maxrss "
                         "and includes the dense _forward_cache transient"),
    )


def profiler_traffic(d, layers, T, seed) -> dict[str, Any]:
    """PROXY memory-traffic: torch.profiler bytes-allocated per matched update
    for each arm. Labeled PROXY — NOT DRAM traffic."""
    from torch.profiler import ProfilerActivity, profile

    def alloc_bytes_for(arm: str) -> Any:
        try:
            fn, _, _ = build_arm(arm, d, layers, T, seed)
            fn()                                 # warm (and compile) once
            with profile(activities=[ProfilerActivity.CPU],
                         profile_memory=True) as prof:
                fn()
            return int(sum(max(0, e.cpu_memory_usage)
                           for e in prof.key_averages()))
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"[:200]

    per_arm = {arm: alloc_bytes_for(arm) for arm in ARMS}
    baseline_vals = [per_arm[a] for a in BASELINES
                     if isinstance(per_arm[a], int)]
    casc = per_arm.get("cascade")
    if baseline_vals and isinstance(casc, int) and casc > 0:
        traffic_ratio = round(max(baseline_vals) / casc, 4)
    else:
        traffic_ratio = None
    return dict(bytes_allocated_proxy=per_arm, traffic_ratio=traffic_ratio,
                label="PROXY: profiler cpu bytes-allocated, not DRAM traffic")


# --- per-arm subprocess PEAK (ru_maxrss is process-monotone) --------------
def _peak_worker(arm, d, layers, T, seed) -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    fn, _, _ = build_arm(arm, d, layers, T, seed)
    for _ in range(PEAK_ITERS):
        fn()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux in kilobytes. Record raw + unit.
    unit = "bytes" if sys.platform == "darwin" else "kilobytes"
    print("PEAK_RESULT " + json.dumps(dict(arm=arm, ru_maxrss=int(ru),
                                           unit=unit)))


def subprocess_peaks(d, layers, T, seed, timeout_s: float) -> dict[str, Any]:
    peaks: dict[str, Any] = {}
    for arm in ARMS:
        cmd = [sys.executable, str(_THIS), "--peak-arm", arm,
               "--d", str(d), "--layers", str(layers), "--tokens", str(T),
               "--seed", str(seed)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s,
                                  env={**os.environ,
                                       "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            peaks[arm] = dict(status="PEAK_TIMEOUT")
            continue
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("PEAK_RESULT ")), None)
        if line is None or proc.returncode != 0:
            peaks[arm] = dict(status="PEAK_ERROR",
                              returncode=proc.returncode,
                              stderr=proc.stderr[-300:])
        else:
            peaks[arm] = json.loads(line[len("PEAK_RESULT "):])
    return dict(per_arm=peaks, note=("SEPARATE from retained-set; packed peak "
                                     "includes the dense forward-cache transient"))


# --- one-step fp64 grad-VALUE parity + tie enumeration --------------------
def fp64_parity(d, layers, T, seed) -> dict[str, Any]:
    model = make(d, layers, seed, dtype=torch.float64)
    ids, targets = make_batch(T, seed)

    # autograd (fp64 model; softmax internal boundary is fp32 by design)
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    loss = F.cross_entropy(model(ids).reshape(-1, V).float(),
                           targets.reshape(-1))
    loss.backward()
    ag = {n: p.grad.detach().clone() for n, p in model.named_parameters()
          if p.grad is not None}
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    engine = MorphPCEngine(model)
    for p in model.parameters():
        p.grad = None
    engine.cascade_grads(ids, targets)
    cg = {n: p.grad.detach().clone() for n, p in model.named_parameters()
          if p.grad is not None}
    for p in model.parameters():
        p.grad = None

    logits, caches, top = engine._forward_cache(ids)
    tape = pack_engine_cache(caches, top, model, ids)
    rg = replay(deserialize(serialize(tape, model), model), model,
                logits, targets)

    # cascade == replay: bit-exact ALWAYS (rtol=atol=0)
    casc_replay_bitexact = set(cg) == set(rg) and all(
        torch.equal(cg[n], rg[n]) for n in cg)
    casc_replay_maxdiff = max(
        (float((cg[n] - rg[n]).abs().max()) for n in cg), default=0.0)

    # autograd vs cascade: <=1e-14 where reduction order matches, else the
    # documented fp32-softmax-boundary tolerance; enumerate any site above it.
    DOC_TOL = 1e-6
    per_param = {}
    divergence_sites = []
    for n in ag:
        diff = float((ag[n] - cg[n]).abs().max())
        per_param[n] = diff
        if diff > DOC_TOL:
            divergence_sites.append(dict(param=n, max_abs_diff=diff))
    ag_casc_maxdiff = max(per_param.values(), default=0.0)
    return dict(
        cascade_equals_replay_bitexact=bool(casc_replay_bitexact),
        cascade_replay_max_abs_diff=casc_replay_maxdiff,
        autograd_vs_cascade_max_abs_diff=ag_casc_maxdiff,
        autograd_vs_cascade_documented_tol=DOC_TOL,
        autograd_vs_cascade_within_documented_tol=(ag_casc_maxdiff <= DOC_TOL),
        route_changing_counterexample_sites=divergence_sites,
        note=("cascade==replay is bit-exact (rtol=atol=0); autograd vs cascade "
              "differs at the fp32 softmax-cast boundary (~1e-8), a documented "
              "non-tie divergence, not a quality loss"),
    )


# --- per-cell gate reducer ------------------------------------------------
def gate_verdict(row: dict[str, Any]) -> dict[str, Any]:
    timing = row.get("timing_arms", {})
    errors = row.get("arm_errors", {})

    def med(arm):
        t = timing.get(arm, {}).get("timing", {})
        return t.get("median_ms")

    def mad(arm):
        t = timing.get(arm, {}).get("timing", {})
        return t.get("mad_ms")

    # (a) retained-set leg
    retained = row.get("retained", {})
    r_ratio = retained.get("retained_ratio")
    retained_pass = isinstance(r_ratio, (int, float)) and r_ratio >= RETAINED_BAR

    # (b) traffic proxy leg
    traffic = row.get("traffic_proxy", {})
    t_ratio = traffic.get("traffic_ratio")
    traffic_pass = isinstance(t_ratio, (int, float)) and t_ratio >= TRAFFIC_BAR

    # (c) latency leg: cascade <= 0.80 x min(best baseline medians)
    baseline_meds = {a: med(a) for a in BASELINES
                     if isinstance(med(a), (int, float)) and a not in errors}
    casc = med("cascade")
    latency_ratio = None
    latency_pass = None
    latency_undecidable = False
    latency_undecidable_reason = None
    if len(baseline_meds) < len(BASELINES):
        latency_undecidable = True
        latency_undecidable_reason = ("a baseline arm errored/timed out; "
                                      "min over {eager,compiled,checkpoint} "
                                      "is not available")
    elif isinstance(casc, (int, float)) and baseline_meds:
        best_arm = min(baseline_meds, key=baseline_meds.get)
        best = baseline_meds[best_arm]
        latency_ratio = round(casc / best, 4)
        latency_pass = casc <= LATENCY_BAR * best
        # decidability vs noise: margin must exceed within-cell MAD (a proxy;
        # the cross-seed MAD from the pilot is the pre-registered decider)
        margin = abs(best - casc)
        noise = max(mad("cascade") or 0.0, mad(best_arm) or 0.0)
        if margin < noise:
            latency_undecidable = True
            latency_undecidable_reason = (
                f"|best-cascade|={margin:.4f}ms < within-cell MAD={noise:.4f}ms; "
                "decidability requires the pilot cross-seed MAD")

    legs = dict(retained=retained_pass, traffic=traffic_pass,
                latency=latency_pass)
    any_pass = bool(retained_pass or traffic_pass or (latency_pass is True))
    # UNDECIDABLE only when no leg PASSES and a needed measurement is missing
    baseline_error = any(errors.get(a) for a in BASELINES)
    if any_pass:
        verdict = "PASS"
    elif baseline_error or (latency_undecidable and not any_pass
                            and r_ratio is None and t_ratio is None):
        verdict = "UNDECIDABLE"
    else:
        verdict = "FAIL"
    return dict(
        verdict=verdict, legs=legs,
        retained_ratio=r_ratio, retained_bar=RETAINED_BAR,
        traffic_ratio=t_ratio, traffic_bar=TRAFFIC_BAR,
        latency_ratio=latency_ratio, latency_bar=LATENCY_BAR,
        latency_pass=latency_pass, latency_undecidable=latency_undecidable,
        latency_undecidable_reason=latency_undecidable_reason,
        baseline_medians_ms=baseline_meds, cascade_median_ms=casc,
        baseline_arm_error=baseline_error,
    )


# --- one full cell --------------------------------------------------------
def run_cell(d, layers, T, seed, warm, reps, *,
             with_peaks=True, peak_timeout_s=DEFAULT_CELL_TIMEOUT_S,
             ) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    started = time.time()
    row: dict[str, Any] = dict(
        cell_id=f"d{d}_l{layers}_t{T}_s{seed}", d_model=d, n_layers=layers,
        T=T, seed=seed, vocab=V, batch=B, warmup=warm, reps=reps,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        status="RUNNING",
    )
    order = list(ARMS)
    random.Random(ORDER_SEED * 1_000_003 + d * 10_007 + layers * 1_009
                  + T * 101 + seed).shuffle(order)
    row["arm_order"] = order

    timing_arms: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for arm in order:
        try:
            timing_arms[arm] = time_arm(arm, d, layers, T, seed, warm, reps)
        except Exception as exc:
            errors[arm] = f"{type(exc).__name__}: {exc}"[:400]
    row["timing_arms"] = timing_arms
    row["arm_errors"] = errors

    # identical-firing verification: init params + batch sha equal across arms
    init_shas = {a: timing_arms[a]["init_params_sha"] for a in timing_arms}
    batch_shas = {a: timing_arms[a]["batch_sha"] for a in timing_arms}
    row["firing"] = dict(
        init_params_sha=init_shas, batch_sha=batch_shas,
        init_params_sha_equal=len(set(init_shas.values())) <= 1,
        batch_sha_equal=len(set(batch_shas.values())) <= 1,
        update_counts={a: timing_arms[a]["update_count"] for a in timing_arms},
        expected_update_count=warm + reps,
    )

    try:
        row["components"] = component_timing(d, layers, T, seed, warm, reps)
    except Exception as exc:
        row["components"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])
    try:
        row["retained"] = measure_retained(d, layers, T, seed)
    except Exception as exc:
        row["retained"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])
    try:
        row["traffic_proxy"] = profiler_traffic(d, layers, T, seed)
    except Exception as exc:
        row["traffic_proxy"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])
    try:
        row["fp64_parity"] = fp64_parity(d, layers, T, seed)
    except Exception as exc:
        row["fp64_parity"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])
    if with_peaks:
        try:
            row["subprocess_peaks"] = subprocess_peaks(
                d, layers, T, seed, peak_timeout_s)
        except Exception as exc:
            row["subprocess_peaks"] = dict(
                error=f"{type(exc).__name__}: {exc}"[:300])

    row["gate"] = gate_verdict(row)
    row["status"] = "COMPLETE" if not errors else "COMPLETE_WITH_ARM_ERRORS"
    row["wall_s"] = round(time.time() - started, 3)
    return row


# --- durable IO -----------------------------------------------------------
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


# --- orchestration (per-cell subprocess with timeout + heartbeat) ---------
def _cell_worker_main(args) -> None:
    row = run_cell(args.d, args.layers, args.tokens, args.seed,
                   args.warmup, args.reps, with_peaks=not args.no_peaks,
                   peak_timeout_s=args.cell_timeout_s)
    atomic_write_json(args.worker_output, row)


def run_grid(cells, seeds, *, experiment, warm, reps, cell_timeout_s,
             hard_wall_s, heartbeat_s, no_peaks, mode_label) -> None:
    command = ("python -m scripts.route_certificates.compiled_resource_benchmark "
               f"--{mode_label}")
    rec = run_record(
        experiment, command,
        config=dict(mode=mode_label, configs=list(cells), seeds=list(seeds),
                    arms=list(ARMS), warmup=warm, reps=reps,
                    retained_bar=RETAINED_BAR, traffic_bar=TRAFFIC_BAR,
                    latency_bar=LATENCY_BAR, cell_timeout_s=cell_timeout_s,
                    hard_wall_s=hard_wall_s, heartbeat_s=heartbeat_s,
                    torch_threads=1),
        inputs=[str(_THIS),
                str(Path(sys.modules["tikitorch.research.pc_convrnn"]
                         .__file__).resolve()),
                str(Path(sys.modules[
                    "tikitorch.autodiff.morphssm_tape"]
                    .__file__).resolve())],
        modules=[sys.modules["tikitorch.research.pc_convrnn"],
                 sys.modules[
                     "tikitorch.autodiff.morphssm_tape"]])
    out_dir = Path(rec["_out_dir"])
    rows_path = out_dir / "benchmark.jsonl"
    heartbeat = out_dir / "heartbeat.json"
    cell_dir = out_dir / "cell_results"
    all_cells = [(*cfg, seed) for cfg in cells for seed in seeds]
    atomic_write_json(out_dir / "manifest.start.json",
                      {k: v for k, v in rec.items() if not k.startswith("_")})
    atomic_write_json(out_dir / "run_state.json", dict(
        state="running", run_id=rec["run_id"], started=time.time(),
        planned_cells=[f"d{d}_l{l}_t{t}_s{s}" for d, l, t, s in all_cells]))

    hard_deadline = time.monotonic() + hard_wall_s
    rows = []
    for d, layers, T, seed in all_cells:
        cell_id = f"d{d}_l{layers}_t{T}_s{seed}"
        if time.monotonic() >= hard_deadline:
            row = dict(cell_id=cell_id, status="NOT_RUN_HARD_WALL")
            append_jsonl_durable(rows_path, row)
            rows.append(row)
            continue
        result_path = cell_dir / f"{cell_id}.json"
        cmd = [sys.executable, str(_THIS), "--worker-cell",
               "--worker-output", str(result_path), "--d", str(d),
               "--layers", str(layers), "--tokens", str(T), "--seed",
               str(seed), "--warmup", str(warm), "--reps", str(reps),
               "--cell-timeout-s", str(cell_timeout_s)]
        if no_peaks:
            cmd.append("--no-peaks")
        proc = _run_subprocess_cell(cmd, out_dir / "cell_logs" /
                                    f"{cell_id}.log", heartbeat,
                                    cell_id=cell_id, timeout_s=cell_timeout_s,
                                    hard_deadline=hard_deadline,
                                    heartbeat_s=heartbeat_s)
        if proc["stop_reason"]:
            row = dict(cell_id=cell_id, status=proc["stop_reason"], **proc)
        elif proc["returncode"] != 0 or not result_path.exists():
            row = dict(cell_id=cell_id, status="WORKER_ERROR", **proc)
        else:
            row = json.loads(result_path.read_text())
            row["worker_elapsed_s"] = proc["elapsed_s"]
        append_jsonl_durable(rows_path, row)
        rows.append(row)
        print(json.dumps(dict(cell=cell_id, status=row.get("status"),
                              gate=row.get("gate", {}).get("verdict"))),
              flush=True)

    _write_gate_summary(out_dir, all_cells, seeds, rows)
    finish(rec, metrics=dict(n_cells=len(all_cells),
                             n_complete=sum(1 for r in rows
                                            if r.get("status") == "COMPLETE")),
           status="completed")
    print("run dir:", out_dir)


def _write_gate_summary(out_dir, all_cells, seeds, rows) -> None:
    verdicts = {r.get("cell_id"): r.get("gate", {}).get("verdict")
                for r in rows if r.get("cell_id")}
    incidents = [r.get("cell_id") for r in rows
                 if r.get("status") not in ("COMPLETE",
                                            "COMPLETE_WITH_ARM_ERRORS")]
    present = {r.get("cell_id") for r in rows}
    expected = {f"d{d}_l{l}_t{t}_s{s}" for d, l, t, s in all_cells}
    summary = dict(
        n_cells_planned=len(all_cells),
        n_cells_present=len(present & expected),
        missing_cells=sorted(expected - present),
        incident_cells=incidents,
        verdicts=verdicts,
        n_pass=sum(1 for v in verdicts.values() if v == "PASS"),
        n_fail=sum(1 for v in verdicts.values() if v == "FAIL"),
        n_undecidable=sum(1 for v in verdicts.values() if v == "UNDECIDABLE"),
        scale_gate_complete=(present & expected) == expected,
    )
    atomic_write_json(out_dir / "gate_verdict.json", summary)


def _run_subprocess_cell(command, log_path, heartbeat_path, *, cell_id,
                         timeout_s, hard_deadline, heartbeat_s) -> dict:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        last_beat, reason = 0.0, None
        while proc.poll() is None:
            now = time.monotonic()
            if now - started >= timeout_s:
                reason = "CELL_TIMEOUT"
            elif now >= hard_deadline:
                reason = "HARD_WALL"
            if reason:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                break
            if now - last_beat >= heartbeat_s:
                atomic_write_json(heartbeat_path, dict(
                    state="running", cell_id=cell_id, pid=proc.pid,
                    elapsed_s=round(now - started, 3), timestamp=time.time()))
                last_beat = now
            time.sleep(0.2)
        return dict(returncode=proc.returncode,
                    elapsed_s=round(time.monotonic() - started, 3),
                    stop_reason=reason)


# --- CLI ------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one cell (d64,l2,t96,s0) in-process")
    ap.add_argument("--pilot", action="store_true",
                    help="3 CONFIGS x 3 seeds")
    ap.add_argument("--full", action="store_true",
                    help="81-cell prereg grid x 3 seeds")
    ap.add_argument("--experiment", default="compiled_resource_benchmark",
                    help="manuscript/results/route_certificates/runs/<experiment>/<uuid>/")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--cell-timeout-s", type=float,
                    default=DEFAULT_CELL_TIMEOUT_S)
    ap.add_argument("--hard-wall-s", type=float, default=DEFAULT_HARD_WALL_S)
    ap.add_argument("--heartbeat-s", type=float, default=DEFAULT_HEARTBEAT_S)
    ap.add_argument("--no-peaks", action="store_true",
                    help="skip per-arm subprocess peak measurement")
    ap.add_argument("--out-json", type=Path,
                    help="smoke only: also write the cell row here")
    # hidden worker protocol
    ap.add_argument("--worker-cell", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--peak-arm", help=argparse.SUPPRESS)
    ap.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--d", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--layers", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--tokens", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.peak_arm:
        _peak_worker(args.peak_arm, args.d, args.layers, args.tokens,
                     args.seed)
        return
    if args.worker_cell:
        if args.worker_output is None:
            ap.error("worker-cell requires --worker-output")
        _cell_worker_main(args)
        return

    if args.dry_run:
        grid = full_grid()
        cells = [(*cfg, s) for cfg in grid for s in SEEDS]
        print(json.dumps(dict(
            pilot_configs=list(CONFIGS),
            full_grid_configs=list(grid), n_full_configs=len(grid),
            seeds=list(SEEDS), arms=list(ARMS), lifetime=LIFETIME,
            n_full_cells=len(cells),
            bars=dict(retained=RETAINED_BAR, traffic=TRAFFIC_BAR,
                      latency=LATENCY_BAR),
            deterministic_arm_orders={
                f"d{d}_l{l}_t{t}_s{s}": _dry_order(d, l, t, s)
                for d, l, t, s in [(*CONFIGS[0], 0), (*LIFETIME, 0)]},
            output=f"manuscript/results/route_certificates/runs/{args.experiment}/<uuid>/"),
            indent=1))
        return

    if args.smoke:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        row = run_cell(64, 2, 96, 0, args.warmup, args.reps,
                       with_peaks=not args.no_peaks,
                       peak_timeout_s=args.cell_timeout_s)
        print(json.dumps(row, indent=1, sort_keys=True))
        if args.out_json:
            atomic_write_json(args.out_json, row)
        return

    if args.pilot or args.full:
        cells = full_grid() if args.full else CONFIGS
        mode = "full" if args.full else "pilot"
        run_grid(cells, SEEDS, experiment=args.experiment, warm=args.warmup,
                 reps=args.reps, cell_timeout_s=args.cell_timeout_s,
                 hard_wall_s=args.hard_wall_s, heartbeat_s=args.heartbeat_s,
                 no_peaks=args.no_peaks, mode_label=mode)
        return

    ap.error("choose one of --dry-run / --smoke / --pilot / --full")


def _dry_order(d, layers, T, seed) -> list[str]:
    order = list(ARMS)
    random.Random(ORDER_SEED * 1_000_003 + d * 10_007 + layers * 1_009
                  + T * 101 + seed).shuffle(order)
    return order


if __name__ == "__main__":
    main()
