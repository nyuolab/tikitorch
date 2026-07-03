"""G3 — fleet consolidation: merged-beats-solo (pre-registered in ../G3_SPEC.md).

Two two-speed learners on disjoint domain shards (A=shakespeare, B=code) exchange ONLY
weights via DiLoCo-style outer averaging; reservoirs, surprise EMAs, and Adam moments stay
local. Question: does the merged model hold both competences?

Arms: solo_A | solo_B | fleet_m{150,500} | fleet_final | pooled (centralized oracle).

Run (repo root): PYTHONPATH=. python ontogenic_learner/experiments/g3_fleet.py [--smoke]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import two_speed_learner as tsl
from two_speed_learner import BASE_DOMAIN, onehot, val_bpc, windows

import torch
import torch.nn.functional as F
from tikitorch.nn.predictive_coding import DyadicPCNet

OUT_DIR = os.path.join(tsl.REPO, "ontogenic_learner/results/g3_fleet")
SHARD_A, SHARD_B = "shakespeare", "code"


class TwoSpeed:
    """The 3x-validated two-speed policy (G1 config) wrapped as a fleet learner."""

    def __init__(self, base_net, V, args, seed, device):
        self.V, self.args, self.dev = V, args, device
        self.net = copy.deepcopy(base_net)
        self.opt_fast = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.opt_slow = torch.optim.Adam(self.net.parameters(), lr=args.lr_slow)
        self.res = tsl.Reservoir(4096, seed)
        self.mu = self.var = None
        self.n_fast = self.updates = self.fwd_eq = 0

    def surprise(self, loss):
        if self.mu is None:
            self.mu, self.var = loss, 1e-2
            return 0.0
        z = (loss - self.mu) / math.sqrt(self.var + 1e-8)
        self.mu = 0.99 * self.mu + 0.01 * loss
        self.var = 0.99 * self.var + 0.01 * (loss - self.mu) ** 2
        return z

    def kick(self, xb, yb, slow=False):
        self.net.train_step_cascade(xb, yb, self.opt_slow if slow else self.opt_fast,
                                    mode="dyadic")
        self.updates += 1
        self.fwd_eq += 3

    def step(self, xi, yi, dom, step, fast_budget):
        xb, yb = onehot(xi, self.V).to(self.dev), yi.to(self.dev)
        with torch.no_grad():
            loss = F.cross_entropy(self.net.predict_logits(xb), yb).item()
        self.fwd_eq += 1
        z = self.surprise(loss)
        if z > self.args.gate_z and self.n_fast < fast_budget:
            self.kick(xb, yb)
            self.res.insert(xi, yi, prio=max(z, 0.1), dom=dom)
            self.n_fast += 1
        if step > 0 and step % self.args.sleep_every == 0:
            for _ in range(self.args.sleep_updates):
                rx, ry = self.res.sample(self.args.batch, weighted=True)
                if rx is not None:
                    self.kick(onehot(rx, self.V).to(self.dev), ry.to(self.dev), slow=True)

    def seed_memories(self, data, ctx, B):
        for w in range(8):
            xi, yi = windows(data, ctx, w * B * 40, B, 1)
            with torch.no_grad():
                l = F.cross_entropy(self.net.predict_logits(onehot(xi, self.V).to(self.dev)),
                                    yi.to(self.dev)).item()
            self.surprise(l)
            self.res.insert(xi, yi, prio=1.0, dom=BASE_DOMAIN)


def merge(learners, mode="m", ref=None):
    """Outer consolidation. Local state (reservoirs, EMAs, moments) never moves.
    m: plain average · asym: 0.3/0.7 favoring the distant specialist ·
    fish: per-param Adam-v importance weighting · ties: sign-resolved deltas vs the
    post-previous-merge reference. Returns the new reference (for ties)."""
    with torch.no_grad():
        params = [list(l.net.parameters()) for l in learners]
        vs = None
        if mode == "fish":
            vs = []
            for l in learners:
                v = []
                for p in l.net.parameters():
                    acc = torch.zeros_like(p)
                    for opt in (l.opt_fast, l.opt_slow):
                        st = opt.state.get(p, {})
                        if "exp_avg_sq" in st:
                            acc = acc + st["exp_avg_sq"]
                    v.append(acc)
                vs.append(v)
        new_ref = []
        for i, group in enumerate(zip(*params)):
            if mode == "asym":
                avg = 0.3 * group[0].data + 0.7 * group[1].data
            elif mode == "fish":
                wa, wb = vs[0][i], vs[1][i]
                tot = wa + wb
                mean = (group[0].data + group[1].data) / 2
                avg = torch.where(tot > 1e-12,
                                  (wa * group[0].data + wb * group[1].data) / (tot + 1e-12),
                                  mean)
            elif mode == "ties" and ref is not None:
                dA, dB = group[0].data - ref[i], group[1].data - ref[i]
                agree = torch.sign(dA) == torch.sign(dB)
                take = torch.where(dA.abs() >= dB.abs(), dA, dB)
                avg = ref[i] + torch.where(agree, (dA + dB) / 2, take)
            else:
                avg = torch.stack([p.data for p in group]).mean(0)
            for p in group:
                p.data.copy_(avg)
            new_ref.append(avg.clone())
    return new_ref


def run_arm(name, base_net, enc, V, val, args, seed, device, log_rows):
    ctx, B, T = args.context, args.batch, args.stream_batches
    t0 = time.time()
    mb_moved = 0.0
    n_params = sum(p.numel() for p in base_net.parameters())

    def report(net, tag, extra=None):
        row = {d: round(val_bpc(net, val[d][0].to(device), val[d][1].to(device), V), 4)
               for d in val}
        log_rows.append({"arm": name, "seed": seed, "event": "transfer", "after": tag,
                         **{f"bpc_{d}": v for d, v in row.items()}, **(extra or {})})
        print(f"  [{name} s{seed}] {tag}: " +
              " ".join(f"{d}={v:.3f}" for d, v in row.items()), flush=True)
        return row

    if name.startswith("solo"):
        dom = SHARD_A if name == "solo_A" else SHARD_B
        ln = TwoSpeed(base_net, V, args, seed, device)
        ln.seed_memories(enc[BASE_DOMAIN], ctx, B)
        for step in range(T):
            xi, yi = windows(enc[dom], ctx, step * B, B, 1)
            if xi.size(0) < B // 2:
                break
            ln.step(xi, yi, dom, step, args.fast_budget)
        final = ln
    elif name == "pooled":                       # centralized oracle: interleaved union
        ln = TwoSpeed(base_net, V, args, seed, device)
        ln.seed_memories(enc[BASE_DOMAIN], ctx, B)
        for step in range(2 * T):
            dom = SHARD_A if step % 2 == 0 else SHARD_B
            xi, yi = windows(enc[dom], ctx, (step // 2) * B, B, 1)
            if xi.size(0) < B // 2:
                break
            ln.step(xi, yi, dom, step, 2 * args.fast_budget)
        final = ln
    else:                     # fleet_{m|mc|ties|fish|asym}{N} | fleet_final
        import re
        consolidate = "_mc" in name
        if name == "fleet_final":
            mode, every = "m", T + 1
        else:
            mm = re.match(r"fleet_([a-z]+?)c?(\d+)$", name)
            mode, every = ("m" if mm.group(1) in ("m", "mc") else mm.group(1),
                           int(mm.group(2)))
        la = TwoSpeed(base_net, V, args, seed, device)
        lb = TwoSpeed(base_net, V, args, seed + 77, device)
        ref = [p.data.clone() for p in la.net.parameters()]   # shared post-base init
        for ln in (la, lb):
            ln.seed_memories(enc[BASE_DOMAIN], ctx, B)
        for step in range(T):
            for ln, dom in ((la, SHARD_A), (lb, SHARD_B)):
                xi, yi = windows(enc[dom], ctx, step * B, B, 1)
                if xi.size(0) < B // 2:
                    continue
                ln.step(xi, yi, dom, step, args.fast_budget)
            if step > 0 and step % every == 0:
                ref = merge([la, lb], mode, ref)
                mb_moved += 2 * n_params * 4 / 1e6
                if consolidate:                  # amendment A1: merge-then-sleep
                    for ln in (la, lb):
                        for _ in range(args.sleep_updates):
                            rx, ry = ln.res.sample(B, weighted=True)
                            if rx is not None:
                                ln.kick(onehot(rx, V).to(device), ry.to(device), slow=True)
                if step % (every * 2) == 0 or every >= 500:
                    report(la.net, f"merge@{step}")
        merge([la, lb], mode, ref)               # final consolidation
        mb_moved += 2 * n_params * 4 / 1e6
        final = la
    row = report(final.net, "final", {"mb_weights_moved": round(mb_moved, 1)})
    ups = final.updates if name != "fleet_final" else final.updates
    log_rows.append({"arm": name, "seed": seed, "event": "done",
                     "updates": final.updates, "fwd_eq": final.fwd_eq,
                     "mb_weights_moved": round(mb_moved, 1),
                     "wall_s": round(time.time() - t0, 1)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-slow", type=float, default=3e-4, dest="lr_slow")
    ap.add_argument("--gate-z", type=float, default=0.25, dest="gate_z")
    ap.add_argument("--base-steps", type=int, default=3000, dest="base_steps")
    ap.add_argument("--stream-batches", type=int, default=1500, dest="stream_batches")
    ap.add_argument("--fast-budget", type=int, default=450, dest="fast_budget")
    ap.add_argument("--sleep-every", type=int, default=150, dest="sleep_every")
    ap.add_argument("--sleep-updates", type=int, default=15, dest="sleep_updates")
    ap.add_argument("--val-windows", type=int, default=4096, dest="val_windows")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.base_steps, args.stream_batches, args.fast_budget = 200, 150, 45
        args.sleep_every, args.sleep_updates = 50, 5
        args.val_windows, args.seeds = 1024, [0]
        if args.arms is None:
            args.arms = ["solo_A", "solo_B", "fleet_m50", "pooled"]
    if args.arms is None:
        args.arms = ["solo_A", "solo_B", "fleet_m150", "fleet_m500",
                     "fleet_final", "pooled"]

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    enc, V = tsl.build_corpora()
    val = {d: windows(enc[d], args.context, len(enc[d]) - len(enc[d]) // 10,
                      args.val_windows, 3)
           for d in [BASE_DOMAIN, SHARD_A, SHARD_B]}
    dims = [args.context * V] + [args.hidden] * (args.depth - 1) + [V]
    os.makedirs(OUT_DIR, exist_ok=True)
    log_rows, t0 = [], time.time()

    finals = {}
    for seed in args.seeds:
        print(f"===== G3 seed {seed}: phylogeny ({args.base_steps} steps) =====", flush=True)
        torch.manual_seed(seed)
        base = DyadicPCNet(dims, residual=True, gamma=0.0625).to(args.device)
        opt = torch.optim.Adam(base.parameters(), lr=args.lr)
        data, hi = enc[BASE_DOMAIN], int(0.9 * len(enc[BASE_DOMAIN])) - 4000
        gen = torch.Generator().manual_seed(seed)
        for _ in range(args.base_steps):
            i = torch.randint(hi, (1,), generator=gen).item()
            xi, yi = windows(data, args.context, i, args.batch, 1)
            base.train_step_cascade(onehot(xi, V).to(args.device), yi.to(args.device),
                                    opt, mode="dyadic")
        for name in args.arms:
            finals[(name, seed)] = run_arm(name, base, enc, V, val, args, seed,
                                           args.device, log_rows)

    # ---------------------------------------------------------------- verdict
    tag = ("smoke" if args.smoke else "full") + args.tag
    with open(os.path.join(OUT_DIR, f"g3_{tag}.jsonl"), "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")

    def mean(name, d):
        vals = [finals[(name, s)][d] for s in args.seeds if (name, s) in finals]
        return sum(vals) / len(vals)

    print("\n================= G3 VERDICT (mean over seeds) =================", flush=True)
    print(f"{'arm':>12} {'A(shak)':>8} {'B(code)':>8} {'wiki':>7}")
    for name in args.arms:
        print(f"{name:>12} {mean(name, SHARD_A):8.3f} {mean(name, SHARD_B):8.3f} "
              f"{mean(name, 'wiki'):7.3f}", flush=True)
    have = {k[0] for k in finals}
    if not args.smoke and {"solo_A", "solo_B"} <= have:
        sA, sB = mean("solo_A", SHARD_A), mean("solo_B", SHARD_B)
        go = []
        for fl in [a for a in args.arms if a.startswith("fleet")]:
            home = (mean(fl, SHARD_A) <= sA + 0.05 and mean(fl, SHARD_B) <= sB + 0.05)
            away = (mean("solo_B", SHARD_A) - mean(fl, SHARD_A) >= 0.30 and
                    mean("solo_A", SHARD_B) - mean(fl, SHARD_B) >= 0.30)
            print(f"  {fl}: home retained (≤solo+0.05)? {home} | away gained (≥0.30)? {away}"
                  f" => {'GO' if home and away else 'no'}", flush=True)
            go.append(home and away)
        print(f"\n  G3 verdict: {'GO' if any(go) else 'NO-GO/PARTIAL — see spec §4'}",
              flush=True)
    print(f"({time.time()-t0:.0f}s total, device={args.device})", flush=True)


if __name__ == "__main__":
    main()
