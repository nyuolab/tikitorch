"""Measured staleness frontier: the ROUND-TAX the throughput model books as zero.

The parallelism thesis was MODELED (2.3-3.9x throughput) but only ONE thing was
measured (async ~ sync final accuracy at a FIXED round budget). The throughput model's
win, however, assumes async hides latency *for free* — i.e. without needing extra local
rounds to reconverge. This script measures that missing cost directly, cheaply, and
honestly (no full training to convergence needed).

WHAT IS MEASURED. The sync error-relaxation is bit-exact equal to `cascade_grads` (=
backprop credit; a unit test asserts torch.equal). So we take that as the backprop
reference gradient and measure, per condition, how many LOCAL message-passing ROUNDS the
ASYNC error-relaxation needs before its gradient reaches cosine >= 0.999 to the backprop
reference — under partial participation (p), stale (lagged) neighbour reads, and random
update order (no global clock). The ratio rounds_async / rounds_sync is the staleness
"round-tax": the cost the throughput model assumes is zero.

  - If round-tax stays bounded (<= ~2x) and FLAT in depth while backprop's avoided
    backward-sweep cost GROWS with depth, the net parallelism advantage is real and
    grows with depth (the defensible, measured version of the thesis).
  - If round-tax grows with depth or as p shrinks (async needs O(d/p) rounds), the
    avoided sweep is eaten and the "for free / grows with depth" claim is FALSIFIED.

HONEST SCOPE. This measures the gradient-convergence DEPENDENCY structure (rounds to the
exact backprop credit), not wall-clock on async hardware. It quantifies the tolerance the
throughput model assumes and grounds the "synchronization-free" claim in a measured cost
axis. It does NOT establish that staleness-tolerance is unique to PC (async-SGD / Hogwild
tolerate staleness too); PC's genuinely-unique axis is weight-transport-free, separate-
backward-datapath-free LOCAL credit (a hardware-substrate property), measured here only
in that the same forward weight carries the error and credit is neighbour-local.
Run: PYTHONPATH=. python scripts/pc_staleness_frontier.py
"""
import sys, json
sys.path.insert(0, ".")
import torch
from torchvision import datasets
from tikitorch.nn.predictive_coding import DyadicPCNet

torch.manual_seed(0)

ds = datasets.MNIST("/tmp/_tiki_data", train=True, download=True)
Xall = (ds.data.float() / 255.0).reshape(-1, 784)
Yall = ds.targets
B = 256
xb, yb = Xall[:B], Yall[:B]

WIDTH = 256
DEPTHS = [8, 16, 32, 64]
ROUNDS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
N_TRIALS = 6
WARM_STEPS = 15          # a few STABLE (Adam, small lr) steps so weights aren't at init
COS_THR = 0.999


def build(depth):
    dims = [784] + [WIDTH] * (depth - 1) + [10]
    return DyadicPCNet(dims, residual=True)


def grads_vec(net):
    return [net.layers[l].weight.grad.detach().clone().reshape(-1) for l in range(net.depth)]


def layer_cos(g, ref, eps=1e-12):
    cs = []
    for a, b in zip(g, ref):
        nb = b.norm()
        if nb < eps:
            continue
        na = a.norm()
        cs.append(0.0 if na < eps else float((a @ b) / (na * nb)))
    return sum(cs) / len(cs) if cs else float("nan")


results = {"meta": {"width": WIDTH, "batch": B, "warm_steps": WARM_STEPS,
                    "n_trials": N_TRIALS, "rounds": ROUNDS, "cos_thr": COS_THR},
           "depths": {}}

CONDS = {
    "sync":            dict(participation=1.0, stale=False, rand_order=False, stoch=False),
    "partial50":       dict(participation=0.5, stale=False, rand_order=False, stoch=True),
    "partial25":       dict(participation=0.25, stale=False, rand_order=False, stoch=True),
    "stale":           dict(participation=1.0, stale=True, rand_order=False, stoch=False),
    "worst50":         dict(participation=0.5, stale=True, rand_order=True, stoch=True),
    "worst25":         dict(participation=0.25, stale=True, rand_order=True, stoch=True),
}


def rounds_to(curve, thr=COS_THR):
    for r, v in zip(ROUNDS, curve):
        if v >= thr:
            return r
    return None


for depth in DEPTHS:
    net = build(depth)
    # warm with a STABLE optimizer so weights are realistic (unstable SGD -> NaN)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(WARM_STEPS):
        net.train_step_cascade(xb, yb, opt)
    if any(p.grad is not None and not torch.isfinite(p).all() for p in net.parameters()):
        net = build(depth)                                   # fell over -> measure at init
    tgt = net._make_target(yb, ref=xb)

    # backprop reference credit = cascade_grads (== sync error-relax, bit-exact)
    for p in net.parameters():
        p.grad = None
    net.cascade_grads(xb, tgt, loss="ce")
    ref = grads_vec(net)

    curves = {}
    for name, c in CONDS.items():
        c = dict(c); stoch = c.pop("stoch")
        trials = N_TRIALS if stoch else 1
        curve = []
        for r in ROUNDS:
            acc = 0.0
            for t in range(trials):
                gen = torch.Generator(); gen.manual_seed(1000 * t + 7)
                for p in net.parameters():
                    p.grad = None
                net.error_relax_grads(xb, tgt, n_iter=r, loss="ce", gen=gen, **c)
                acc += layer_cos(grads_vec(net), ref)
            curve.append(round(acc / trials, 4))
        curves[name] = curve

    R = {name: rounds_to(cv) for name, cv in curves.items()}
    sync_R = R["sync"] or depth
    tax = {name: (round(R[name] / sync_R, 2) if R[name] else None) for name in R}
    results["depths"][depth] = {
        "curves": curves,
        "rounds_to_999": R,
        "round_tax_vs_sync": tax,
        "backprop_critical_path_ordered_stages": 2 * depth,   # forward L + backward L
        "backprop_global_barrier": True,
        "backprop_weight_transport": True,
        "pc_forward_stages": depth,
        "pc_async_relax_rounds": R,                            # measured
        "pc_global_barrier": False,
        "pc_weight_transport": False,
    }
    print(f"\n=== depth {depth} (residual MLP, width {WIDTH}) ===")
    cols = list(CONDS)
    print(f"  {'rounds':>7} | " + " | ".join(f"{c:>9}" for c in cols))
    for r in ROUNDS:
        row = [curves[c][ROUNDS.index(r)] for c in cols]
        print(f"  {r:>7} | " + " | ".join(f"{v:>9.4f}" for v in row))
    print("  rounds->0.999 cos: " + ", ".join(f"{c}={R[c]}" for c in cols))
    print("  round-tax vs sync: " + ", ".join(f"{c}={tax[c]}x" for c in cols if c != "sync"))
    print(f"  CRITICAL PATH: backprop = {2*depth} ordered stages + barrier + weight-transport")
    print(f"                 PC-async = {depth} fwd + R local rounds (above), NO barrier, NO transport")

out = "results/predictive_coding/staleness_frontier.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"\nwrote {out}")
