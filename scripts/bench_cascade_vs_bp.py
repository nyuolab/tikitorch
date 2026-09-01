#!/usr/bin/env python3
"""Track A1 — matched cascade-vs-online-backprop memory & throughput
benchmark (Nature revision item).

Same model, same stream, same hardware, three learners:
  cascade   : MorphPCEngine.serve_and_learn (fused serve+learn; the
              paper's mechanism — no stored activation graph)
  online-bp : standard autograd — forward WITH graph, loss.backward(),
              opt.step() every window (activations retained by autograd)
  serve     : forward-only reference (the floor)

Measured per condition: tokens/s (steady-state, warmup excluded), peak RSS
delta over baseline (ru_maxrss), and torch CPU peak via a sampling thread;
plus the analytic activation-storage bytes for the BP graph at this shape.
3 seeds x 2 sizes (d128/4L, d512/4L). Output: committed jsonl.

Usage: python scripts/bench_cascade_vs_bp.py --out results/pc_convrnn/a1_bench.jsonl
"""
import argparse
import json
import resource
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tikitorch.research.pc_convrnn import MorphPCEngine, build_model  # noqa: E402

WINDOW = 96
UPDATES = 400
WARMUP = 50


class RssSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = False

    def run(self):
        import psutil
        p = psutil.Process()
        while not self.stop:
            self.peak = max(self.peak, p.memory_info().rss)
            time.sleep(0.005)


def _tensor_bytes(obj, seen):
    """Recursive unique-storage byte count over nested dict/list/tuple/Tensor."""
    if torch.is_tensor(obj):
        ptr = obj.untyped_storage().data_ptr()
        if ptr in seen:
            return 0
        seen.add(ptr)
        return obj.untyped_storage().nbytes()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v, seen) for v in obj)
    return 0


def act_storage_bytes(model, mode, x, y, vocab):
    """Per-update retained activation storage, measured not estimated.
    online-bp: unique saved-tensor bytes packed by autograd between forward
    and backward (the retained graph). cascade: bytes of the frozen routing
    caches serve_and_learn holds between serve and the reverse sweep."""
    if mode == "serve":
        return 0
    resident = {p.untyped_storage().data_ptr()          # params+buffers live
                for p in list(model.parameters()) + list(model.buffers())}
    if mode == "cascade":                               # anyway: not "activation"
        engine = MorphPCEngine(model)
        _, caches, top = engine._forward_cache(x)
        seen = set(resident)
        return _tensor_bytes(caches, seen) + _tensor_bytes(top, seen)
    seen, tally = set(resident), [0]

    def pack(t):
        if torch.is_tensor(t):
            ptr = t.untyped_storage().data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                tally[0] += t.untyped_storage().nbytes()
        return t

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        lg = model(x)
        loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)
    return tally[0]


def bench(d_model, seed, mode, vocab=64):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    ns = argparse.Namespace(d_model=d_model, n_layers=4, kernel=7,
                            dropout=0.0, tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", vocab, ns)
    model.set_tau(0.0)
    stream = torch.randint(0, vocab, (UPDATES + WARMUP + 2, WINDOW + 1))
    act_b = act_storage_bytes(model, mode, stream[0, :-1].unsqueeze(0),
                              stream[0, 1:].unsqueeze(0), vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    samp = RssSampler()
    samp.start()
    t0 = None
    if mode == "cascade":
        model.eval()
        engine = MorphPCEngine(model)
        for i in range(UPDATES + WARMUP):
            x, y = stream[i, :-1].unsqueeze(0), stream[i, 1:].unsqueeze(0)
            engine.serve_and_learn(x, y, opt=opt, fire=lambda *a, **k: True)
            if i == WARMUP - 1:
                t0 = time.time()
    elif mode == "online-bp":
        model.train()
        for i in range(UPDATES + WARMUP):
            x, y = stream[i, :-1].unsqueeze(0), stream[i, 1:].unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            lg = model(x)
            loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            if i == WARMUP - 1:
                t0 = time.time()
    else:  # serve
        model.eval()
        with torch.no_grad():
            for i in range(UPDATES + WARMUP):
                x = stream[i, :-1].unsqueeze(0)
                model(x)
                if i == WARMUP - 1:
                    t0 = time.time()
    wall = time.time() - t0
    samp.stop = True
    samp.join(timeout=1)
    n_par = sum(p.numel() for p in model.parameters())
    return dict(mode=mode, d=d_model, seed=seed, params=n_par,
                act_mb=round(act_b / 2 ** 20, 3),
                tok_s=round(UPDATES * WINDOW / wall, 1),
                wall_s=round(wall, 2),
                peak_rss_mb=round(samp.peak / 2 ** 20, 1),
                maxrss_mb=round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default=None, help="internal: d:seed:mode")
    a = ap.parse_args()
    if a.cell:                       # child: one cell, fresh process (clean RSS)
        d, seed, mode = a.cell.split(":")
        print(json.dumps(bench(int(d), int(seed), mode)))
        return
    import subprocess
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in (128, 512):
        for seed in (0, 1, 2):
            for mode in ("serve", "cascade", "online-bp"):
                r = subprocess.run(
                    [sys.executable, __file__, "--out", "-",
                     "--cell", f"{d}:{seed}:{mode}"],
                    capture_output=True, text=True)
                row = json.loads(r.stdout.strip().splitlines()[-1])
                rows.append(row)
                print(row, flush=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
