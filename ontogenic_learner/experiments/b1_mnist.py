"""B1-MNIST — the ontogenic learner on canonical continual-learning benchmarks
(pre-registered in ../BENCHMARKS.md). Permuted-MNIST (5 tasks) + Split-MNIST
class-incremental (5 tasks), absolute accuracy vs naive/rehearsal/joint at matched budget.

Run (repo root): PYTHONPATH=. python ontogenic_learner/experiments/b1_mnist.py [--smoke]
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

import torch
import torch.nn.functional as F
from tikitorch.nn.predictive_coding import DyadicPCNet

OUT = os.path.join(tsl.REPO, "ontogenic_learner/results/benchmarks")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def get_mnist():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Lambda(lambda t: t.view(-1))])
    tr = datasets.MNIST(os.path.join(tsl.REPO, "data"), train=True, download=True,
                        transform=tf)
    te = datasets.MNIST(os.path.join(tsl.REPO, "data"), train=False, download=True,
                        transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    Yte = torch.tensor([te[i][1] for i in range(len(te))])
    return Xtr, Ytr, Xte, Yte


def make_tasks(kind, Xtr, Ytr, Xte, Yte, n_tasks, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "permuted":
        perms = [torch.arange(784)] + [torch.randperm(784, generator=g)
                                       for _ in range(n_tasks - 1)]
        return [(Xtr[:, p], Ytr, Xte[:, p], Yte) for p in perms]
    tasks = []                                   # split: class-incremental pairs
    for t in range(n_tasks):
        cls = [2 * t, 2 * t + 1]
        mtr = (Ytr == cls[0]) | (Ytr == cls[1])
        mte = (Yte == cls[0]) | (Yte == cls[1])
        tasks.append((Xtr[mtr], Ytr[mtr], Xte[mte], Yte[mte]))
    return tasks


@torch.no_grad()
def acc(net, X, Y, bs=4096):
    hits = 0
    for i in range(0, X.size(0), bs):
        hits += (net.predict_logits(X[i:i + bs].to(DEV)).argmax(-1)
                 == Y[i:i + bs].to(DEV)).sum().item()
    return hits / X.size(0)


class Learner:
    def __init__(self, name, net, args, seed):
        self.name = name
        self.net = copy.deepcopy(net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.opt_slow = torch.optim.Adam(self.net.parameters(), lr=args.lr * 0.3)
        self.res = tsl.Reservoir(args.reservoir, seed + hash(name) % 997)
        self.mu = self.var = None
        self.updates = 0
        self.args = args

    def z(self, loss):
        if self.mu is None:
            self.mu, self.var = loss, 1e-2
            return 0.0
        zz = (loss - self.mu) / math.sqrt(self.var + 1e-8)
        self.mu = 0.99 * self.mu + 0.01 * loss
        self.var = 0.99 * self.var + 0.01 * (loss - self.mu) ** 2
        return zz

    def kick(self, xb, yb, slow=False):
        self.net.train_step_cascade(xb, yb, self.opt_slow if slow else self.opt,
                                    mode="dyadic")
        self.updates += 1


def run_arm(name, base, tasks, args, seed, log):
    a = args
    L = Learner(name, base, a, seed)
    hist = []
    if name == "joint":                          # i.i.d. oracle over all tasks
        Xa = torch.cat([t[0] for t in tasks]); Ya = torch.cat([t[1] for t in tasks])
        g = torch.Generator().manual_seed(seed)
        for _ in range(a.budget * len(tasks)):
            idx = torch.randint(Xa.size(0), (a.bs,), generator=g)
            L.kick(Xa[idx].to(DEV), Ya[idx].to(DEV))
        hist = [[acc(L.net, t[2], t[3]) for t in tasks]]
        return L, hist
    for ti, (Xtr, Ytr, _, _) in enumerate(tasks):
        g = torch.Generator().manual_seed(seed * 31 + ti)
        n_stream = a.budget * a.stream_per_slot
        slots = set(torch.linspace(0, n_stream - 1, a.budget).long().tolist())
        n_fast = 0
        for step in range(n_stream):
            idx = torch.randint(Xtr.size(0), (a.bs,), generator=g)
            xb, yb = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
            if name == "naive":
                if step in slots:
                    L.kick(xb, yb)
            elif name == "rehearsal_bal":
                L.res.insert(Xtr[idx], Ytr[idx], dom=str(ti))
                if step in slots:
                    rx, ry = L.res.sample(a.bs // 4, weighted=False, balanced=True)
                    if rx is not None:
                        xb = torch.cat([xb[:a.bs - rx.size(0)], rx.to(DEV)])
                        yb = torch.cat([yb[:a.bs - rx.size(0)], ry.to(DEV)])
                    L.kick(xb, yb)
            else:                                # two_speed
                with torch.no_grad():
                    loss = F.cross_entropy(L.net.predict_logits(xb), yb).item()
                zz = L.z(loss)
                if (zz > a.gate_z or step < 20) and n_fast < int(a.budget * 0.75):
                    L.kick(xb, yb)
                    L.res.insert(Xtr[idx], Ytr[idx], prio=max(zz, 0.1), dom=str(ti))
                    n_fast += 1
                if step > 0 and step % a.sleep_every == 0:
                    for _ in range(a.sleep_updates):
                        rx, ry = L.res.sample(a.bs, weighted=True)
                        if rx is not None:
                            L.kick(rx.to(DEV), ry.to(DEV), slow=True)
        row = [acc(L.net, t[2], t[3]) for t in tasks]
        hist.append(row)
        log.append({"arm": name, "seed": seed, "after_task": ti,
                    "acc": [round(x, 4) for x in row], "updates": L.updates})
        print(f"  [{name} s{seed}] after T{ti}: " +
              " ".join(f"{x:.3f}" for x in row) + f" (upd={L.updates})", flush=True)
    return L, hist


def score(hist):
    final = hist[-1]
    avg = sum(final) / len(final)
    forget = [hist[j][j] - final[j] for j in range(len(hist) - 1)] if len(hist) > 1 else [0]
    return avg, sum(forget) / max(len(forget), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=400)      # updates per task
    ap.add_argument("--stream-per-slot", type=int, default=2, dest="stream_per_slot")
    ap.add_argument("--gate-z", type=float, default=0.25, dest="gate_z")
    ap.add_argument("--sleep-every", type=int, default=100, dest="sleep_every")
    ap.add_argument("--sleep-updates", type=int, default=12, dest="sleep_updates")
    ap.add_argument("--reservoir", type=int, default=2048)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.budget, args.seeds = 60, [0]

    Xtr, Ytr, Xte, Yte = get_mnist()
    os.makedirs(OUT, exist_ok=True)
    log, summary = [], {}
    t0 = time.time()
    for kind in ["split", "permuted"]:
        print(f"===== B1-MNIST {kind} =====", flush=True)
        for seed in args.seeds:
            tasks = make_tasks(kind, Xtr, Ytr, Xte, Yte, 5, seed)
            torch.manual_seed(seed)
            base = DyadicPCNet([784, 256, 256, 10], residual=True,
                               gamma=0.0625).to(DEV)
            for name in ["naive", "rehearsal_bal", "two_speed", "joint"]:
                L, hist = run_arm(name, base, tasks, args, seed, log)
                avg, fg = score(hist)
                summary.setdefault((kind, name), []).append((avg, fg, L.updates))
                log.append({"arm": name, "seed": seed, "kind": kind, "event": "done",
                            "final_avg": round(avg, 4), "forgetting": round(fg, 4),
                            "updates": L.updates})
    tag = "smoke" if args.smoke else "full"
    with open(os.path.join(OUT, f"b1_mnist_{tag}.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(f"\n================= B1-MNIST ({tag}, mean over seeds) =================",
          flush=True)
    print(f"{'proto':>9} {'arm':>14} {'final_avg':>9} {'forget':>8} {'updates':>8}")
    for (kind, name), vals in summary.items():
        n = len(vals)
        print(f"{kind:>9} {name:>14} {sum(v[0] for v in vals)/n:9.3f} "
              f"{sum(v[1] for v in vals)/n:+8.3f} {sum(v[2] for v in vals)//n:8d}",
              flush=True)
    print(f"lit anchors (split class-inc): naive~0.20 EWC~0.20 rehearsal~0.80-0.90 "
          f"joint~0.98  ({time.time()-t0:.0f}s, {DEV})", flush=True)


if __name__ == "__main__":
    main()
