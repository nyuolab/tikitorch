"""G2a — the rule-level pilot (pre-registered in ../G2_SPEC.md).

Probes, selected with --probe:
  h1 : sequential stream; full-wake ITERATIVE (settle-then-update) vs cascade/BP controls,
       step-matched AND FLOP-matched (bp_flop gets the settle arm's fwd-eq as extra updates).
  h3 : cyclic long-stream (domains x4 revisits) — plasticity-decay curves per arm.
  h2 : two-domain stream, then few-shot burst on the held-back code domain (k-sweep).

Arm grammar: bp | bp_flop | rehearsal_bal | ts (G1 two-speed reference)
             | iter_s{N}[_g{0625|125}]   (full-wake iterative, settle=N, gamma override)

Scale: ~4.1M params (hidden 768, depth 8) — the G1 family scaled per G2_SPEC §4.
Device: MPS by default (measured 10x CPU); numerics are torch-float (regime pilot, no
bit-exactness claim). Every iterative update carries the G1.1 snapshot divergence guard.

Run (repo root):  PYTHONPATH=. python ontogenic_learner/experiments/g2_rule_probe.py \
                      --probe h1 [--smoke]
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
import two_speed_learner as tsl                    # corpora, windows, eval, Reservoir
from two_speed_learner import BASE_DOMAIN, STREAM, onehot, val_bpc, windows

import torch
import torch.nn.functional as F
from tikitorch.nn.predictive_coding import DyadicPCNet

OUT_DIR = os.path.join(tsl.REPO, "ontogenic_learner/results/g2_rule_probe")


# ---------------------------------------------------------------- arms
class RuleArm:
    """One learner. kind: bp | bp_flop | rehearsal_bal | ts | iter_sN[_gX]."""

    def __init__(self, name, base_net, V, args, seed, device):
        self.name, self.V, self.args, self.dev = name, V, args, device
        self.net = copy.deepcopy(base_net)
        lr = args.lr
        self.settle = 0
        self.relax = None                       # (n_iter, participation) for relax arms
        for gk, gv in {"_g125": 0.125, "_g0625": 0.0625, "_g02": 0.02,
                       "_g01": 0.01}.items():
            if gk in name:
                self.net.gamma = gv
        if "_lr" in name:                       # ..._lr4 -> lr = 1e-4 (any arm kind)
            lr = 10.0 ** -int(name.rsplit("_lr", 1)[1])
        if name.startswith("iter"):
            self.settle = int(name.split("_s")[1].split("_")[0])
        elif name.startswith("relax"):          # relax_p50_i24[_lr4]: async error-relax
            self.relax = (int(name.split("_i")[1].split("_")[0]),
                          int(name.split("_p")[1].split("_")[0]) / 100.0)
            self.rgen = torch.Generator().manual_seed(seed + 4242)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.opt_slow = torch.optim.Adam(self.net.parameters(), lr=args.lr_slow)
        self.res = tsl.Reservoir(4096, seed + hash(name) % 1000)
        self.updates = self.fwd_eq = self.diverged = 0
        self.mu = self.var = None

    def surprise(self, loss):
        if self.mu is None:
            self.mu, self.var = loss, 1e-2
            return 0.0
        z = (loss - self.mu) / math.sqrt(self.var + 1e-8)
        self.mu = 0.99 * self.mu + 0.01 * loss
        self.var = 0.99 * self.var + 0.01 * (loss - self.mu) ** 2
        return z

    def kick(self, xb, yb, slow=False):
        """One update by this arm's rule, with divergence guard on iterative kicks."""
        if self.relax is not None:              # async error-relaxation (fabric regime)
            n_iter, part = self.relax
            snap = [p.detach().clone() for p in self.net.parameters()]
            self.net.train_step_relax(xb, yb, self.opt_slow if slow else self.opt,
                                      n_iter=n_iter, mode="dyadic",
                                      participation=part, stale=True, rand_order=True,
                                      gen=self.rgen)
            self.fwd_eq += n_iter + 3
            if not all(torch.isfinite(p).all() for p in self.net.parameters()):
                with torch.no_grad():
                    for p, s in zip(self.net.parameters(), snap):
                        p.copy_(s)
                self.diverged += 1
                return
            self.updates += 1
            return
        if self.settle:
            snap = [p.detach().clone() for p in self.net.parameters()]
            self.net.train_step(xb, yb, self.opt_slow if slow else self.opt,
                                n_steps=self.settle)
            self.fwd_eq += 2 * self.settle + 3
            if not all(torch.isfinite(p).all() for p in self.net.parameters()):
                with torch.no_grad():
                    for p, s in zip(self.net.parameters(), snap):
                        p.copy_(s)
                self.diverged += 1
                return
        else:
            self.net.train_step_cascade(xb, yb, self.opt_slow if slow else self.opt,
                                        mode="dyadic")
            self.fwd_eq += 3
        self.updates += 1


def update_slots(kind, n_batches, budget, settle_ref):
    """Evenly-spaced update steps. bp_flop gets the FLOP budget of an iter_s{settle_ref}
    arm converted into extra cascade updates: (2s+3)/3 x budget."""
    n = budget
    if kind == "bp_flop":
        n = min(n_batches, round(budget * (2 * settle_ref + 3) / 3))
    return set(torch.linspace(0, n_batches - 1, n).long().tolist()), n


def stream_phase(arm, data, dom, args, log_rows, seed, phase_tag):
    """Run one domain segment through an arm. Returns nothing; state lives in arm."""
    B, ctx = args.batch, args.context
    slots, _ = update_slots(arm.name if not arm.name.startswith("iter") else "iter",
                            args.stream_batches, args.budget, args.settle_ref)
    n_fast = 0
    for step in range(args.stream_batches):
        xi, yi = windows(data, ctx, step * B, B, 1)
        if xi.size(0) < B // 2:
            break
        xb, yb = onehot(xi, arm.V).to(arm.dev), yi.to(arm.dev)
        with torch.no_grad():
            loss = F.cross_entropy(arm.net.predict_logits(xb), yb).item()
        arm.fwd_eq += 1
        z = arm.surprise(loss)
        if step % args.eval_every == 0:
            log_rows.append({"arm": arm.name, "seed": seed, "event": "stream",
                             "phase": phase_tag, "step": step,
                             "loss_bpc": round(loss / math.log(2), 4), "z": round(z, 3)})
        if arm.name == "frozen":
            continue
        if (arm.name.startswith("bp") or arm.name.startswith("iter")
                or arm.name.startswith("relax")):
            if step in slots:
                arm.kick(xb, yb)
        elif arm.name == "rehearsal_bal":
            arm.res.insert(xi, yi, dom=dom)
            if step in slots:
                rx, ry = arm.res.sample(B // 4, weighted=False, balanced=True)
                if rx is not None:
                    xb = torch.cat([onehot(xi[:B - rx.size(0)], arm.V).to(arm.dev),
                                    onehot(rx, arm.V).to(arm.dev)])
                    yb = torch.cat([yi[:B - rx.size(0)].to(arm.dev), ry.to(arm.dev)])
                arm.kick(xb, yb)
        elif arm.name == "ts":                      # G1 two-speed reference (schedule)
            if z > 0.25 and n_fast < int(args.budget * 0.75):
                arm.kick(xb, yb)
                arm.res.insert(xi, yi, prio=max(z, 0.1), dom=dom)
                n_fast += 1
            if step > 0 and step % max(args.stream_batches // 4, 1) == 0:
                for _ in range(max(args.budget // 24, 1)):
                    rx, ry = arm.res.sample(B, weighted=True)
                    if rx is not None:
                        arm.kick(onehot(rx, arm.V).to(arm.dev), ry.to(arm.dev), slow=True)


def transfer(arm, val, log_rows, seed, tag):
    row = {d: round(val_bpc(arm.net, val[d][0].to(arm.dev), val[d][1].to(arm.dev),
                            arm.V), 4) for d in val}
    log_rows.append({"arm": arm.name, "seed": seed, "event": "transfer", "after": tag,
                     **{f"bpc_{d}": v for d, v in row.items()},
                     "updates": arm.updates, "fwd_eq": arm.fwd_eq,
                     "diverged": arm.diverged})
    print(f"  [{arm.name} s{seed}] after {tag}: " +
          " ".join(f"{d}={v:.3f}" for d, v in row.items()) +
          f"  (upd={arm.updates}, fwd_eq={arm.fwd_eq}, div={arm.diverged})", flush=True)
    return row


# ---------------------------------------------------------------- probes
def pretrain_base(enc, V, args, seed, device):
    dims = [args.context * V] + [args.hidden] * (args.depth - 1) + [V]
    torch.manual_seed(seed)
    net = DyadicPCNet(dims, residual=True, gamma=0.0625).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    data, ctx, B = enc[BASE_DOMAIN], args.context, args.batch
    gen = torch.Generator().manual_seed(seed)
    hi = int(0.9 * len(data)) - ctx - B - 1
    t0 = time.time()
    for step in range(args.base_steps):
        i = torch.randint(hi, (1,), generator=gen).item()
        xi, yi = windows(data, ctx, i, B, 1)
        net.train_step_cascade(onehot(xi, V).to(device), yi.to(device), opt, mode="dyadic")
    print(f"  base pretrained ({args.base_steps} steps, {time.time()-t0:.0f}s)", flush=True)
    return net


def run_probe(args, enc, V, device, log_rows):
    ctx = args.context
    val = {d: windows(enc[d], ctx, len(enc[d]) - len(enc[d]) // 10, args.val_windows, 3)
           for d in [BASE_DOMAIN] + STREAM}
    results = {}
    for seed in args.seeds:
        print(f"===== {args.probe} seed {seed} =====", flush=True)
        base = pretrain_base(enc, V, args, seed, device)
        for name in args.arms:
            arm = RuleArm(name, base, V, args, seed, device)
            # warm surprise EMA + seed replay memories from base domain
            data = enc[BASE_DOMAIN]
            for w in range(12):
                xi, yi = windows(data, ctx, w * args.batch * 40, args.batch, 1)
                with torch.no_grad():
                    l = F.cross_entropy(
                        arm.net.predict_logits(onehot(xi, V).to(device)),
                        yi.to(device)).item()
                arm.surprise(l)
                if name in ("rehearsal_bal", "ts") and w < 8:
                    arm.res.insert(xi, yi, prio=1.0, dom=BASE_DOMAIN)
            t0 = time.time()
            mat = {"base": transfer(arm, val, log_rows, seed, "base")}

            if args.probe == "h1":
                for dom in STREAM:
                    stream_phase(arm, enc[dom], dom, args, log_rows, seed, dom)
                    mat[dom] = transfer(arm, val, log_rows, seed, dom)
            elif args.probe == "h3":
                for r in range(args.revisits):
                    for dom in STREAM:
                        tag = f"{dom}@r{r}"
                        before = val_bpc(arm.net, val[dom][0].to(device),
                                         val[dom][1].to(device), V)
                        stream_phase(arm, enc[dom], dom, args, log_rows, seed, tag)
                        row = transfer(arm, val, log_rows, seed, tag)
                        log_rows.append({"arm": name, "seed": seed, "event": "revisit",
                                         "domain": dom, "revisit": r,
                                         "gain_bpc": round(before - row[dom], 4)})
            elif args.probe == "h2":
                for dom in ["shakespeare", "docs"]:
                    stream_phase(arm, enc[dom], dom, args, log_rows, seed, dom)
                    mat[dom] = transfer(arm, val, log_rows, seed, dom)
                # few-shot burst on held-back `code`: update EVERY batch, k-sweep evals
                data, B = enc["code"], args.batch
                for k in range(1, args.fewshot_k + 1):
                    xi, yi = windows(data, ctx, (k - 1) * B, B, 1)
                    arm.kick(onehot(xi, V).to(device), yi.to(device))
                    if k in (64, 256, 1024):
                        row = transfer(arm, val, log_rows, seed, f"code@k{k}")
            log_rows.append({"arm": name, "seed": seed, "event": "done",
                             "probe": args.probe, "updates": arm.updates,
                             "fwd_eq": arm.fwd_eq, "diverged": arm.diverged,
                             "wall_s": round(time.time() - t0, 1)})
            results[(name, seed)] = mat
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["h1", "h2", "h3"], required=True)
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-slow", type=float, default=1e-4, dest="lr_slow")
    ap.add_argument("--base-steps", type=int, default=3000, dest="base_steps")
    ap.add_argument("--stream-batches", type=int, default=1500, dest="stream_batches")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--settle-ref", type=int, default=8, dest="settle_ref")
    ap.add_argument("--revisits", type=int, default=4)
    ap.add_argument("--fewshot-k", type=int, default=1024, dest="fewshot_k")
    ap.add_argument("--val-windows", type=int, default=4096, dest="val_windows")
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.arms is None:
        args.arms = {"h1": ["bp", "bp_flop", "rehearsal_bal",
                            "iter_s8", "iter_s16", "iter_s8_g125"],
                     "h3": ["bp", "iter_s8", "ts"],
                     "h2": ["bp", "iter_s8"]}[args.probe]
    if args.smoke:
        args.base_steps, args.stream_batches, args.budget = 200, 150, 60
        args.revisits, args.fewshot_k = 2, 64
        args.val_windows, args.seeds = 1024, [0]

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    enc, V = tsl.build_corpora()
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.probe + ("_smoke" if args.smoke else "") + args.tag
    log_rows = []
    t0 = time.time()
    run_probe(args, enc, V, args.device, log_rows)
    with open(os.path.join(OUT_DIR, f"g2a_{tag}.jsonl"), "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote g2a_{tag}.jsonl ({time.time()-t0:.0f}s total, device={args.device})",
          flush=True)


if __name__ == "__main__":
    main()
