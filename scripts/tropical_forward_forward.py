"""Tropical Forward-Forward (TFF) — reproducible MNIST experiment.

Trains a forward-only, backprop-free, **closure-clean** (multiplier-free) tropical
Forward-Forward stack on MNIST and measures it honestly against three references
on a matched 2x2 design (learning rule x substrate):

    * TFF              : FF-local  + tropical (max-plus)   <- the new thing
    * backprop-tropical: backprop  + tropical (same layers, global tropical CE)
    * dense FF         : FF-local  + dense (Hinton's float FF: x^2 goodness, L2 norm)
    * dense backprop   : backprop  + dense (a vanilla MLP, the absolute reference)

Reports, for depths 2/4/8: test accuracy, per-layer positive-vs-negative goodness
separation, and the depth-scaling curve. Everything is measured, nothing asserted.

Usage::

    python scripts/tropical_forward_forward.py                 # full run (~10 min CPU)
    python scripts/tropical_forward_forward.py --quick          # smaller/faster
    python scripts/tropical_forward_forward.py --smoke          # 1 step, synthetic, asserts closure
    python scripts/tropical_forward_forward.py --depths 2 4 8

Writes a JSONL row per configuration to ``manuscript/results/forward_forward/`` and prints a
summary table. ``torchvision`` is only needed for the real dataset; ``--smoke``
runs on synthetic data with no download.
"""
import argparse
import json
import os
import sys
import time

# Use the tikitorch that ships in THIS repo (the standalone canonical tree), not
# whatever an editable install happens to resolve to — so the experiment always
# measures the code under test. Harmless when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import tikitorch as tt
import tikitorch.nn as ttnn
import tikitorch.optim as optim
import tikitorch.functional as ttF
from tikitorch.forward_forward import (
    TropicalFFNet, TropicalFFTrainer, overlay_label, make_negative_labels,
    goodness_classify,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mnist(train_size, test_size, data_root="./data"):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr = datasets.MNIST(data_root, train=True, download=True, transform=tf)
    te = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(train_size)])
    Ytr = torch.tensor([tr[i][1] for i in range(train_size)])
    Xte = torch.stack([te[i][0] for i in range(test_size)])
    Yte = torch.tensor([te[i][1] for i in range(test_size)])
    return Xtr, Ytr, Xte, Yte


def synthetic(train_size, test_size, d=784, c=10):
    """Class-structured synthetic data (each class = a different planted block),
    so --smoke exercises a learnable signal, not pure noise."""
    def make(n):
        y = torch.randint(0, c, (n,))
        x = torch.rand(n, d) * 0.1
        for cls in range(c):
            m = y == cls
            x[m, cls * 8:(cls + 1) * 8] += 2.0
        return x, y
    Xtr, Ytr = make(train_size)
    Xte, Yte = make(test_size)
    return Xtr, Ytr, Xte, Yte


def iterate(X, Y, batch, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    for i in range(0, len(X) - batch + 1, batch):
        idx = perm[i:i + batch]
        yield X[idx], Y[idx]


# ---------------------------------------------------------------------------
# 1) Tropical Forward-Forward (the subject)
# ---------------------------------------------------------------------------
def run_tff(data, depth, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    dims = [Xtr.size(1)] + [args.width] * depth
    net = TropicalFFNet(dims, num_classes=args.classes, goodness=args.goodness, k=args.k)
    trainer = TropicalFFTrainer(net, lr_shift=args.lr_shift, momentum_bits=args.momentum_bits,
                                margin=args.margin)
    t0 = time.time()
    last = {"loss": [], "g_pos": [], "g_neg": []}
    for ep in range(args.epochs):
        for x, y in iterate(Xtr, Ytr, args.batch, args.seed + ep):
            xp = overlay_label(x, y, args.classes, on=args.on)
            xn = overlay_label(x, make_negative_labels(y, args.classes), args.classes, on=args.on)
            last = trainer.train_step(xp, xn)
    preds, _ = goodness_classify(net, Xte, args.classes, on=args.on, skip_first=args.skip_first)
    acc = (preds == Yte).float().mean().item()
    sep = [round(p - n, 4) for p, n in zip(last["g_pos"], last["g_neg"])]
    return {"acc": acc, "layer_separation": sep, "secs": round(time.time() - t0, 1),
            "params": net.param_count}


# ---------------------------------------------------------------------------
# 2) Backprop-trained tropical (same substrate, GLOBAL objective)
# ---------------------------------------------------------------------------
class TropicalStack(nn.Module):
    def __init__(self, din, width, depth, classes):
        super().__init__()
        self.hidden = nn.ModuleList()
        d = din
        for _ in range(depth):
            self.hidden.append(ttnn.TropicalDilationModule(d, width))
            d = width
        self.head = ttnn.TropicalDilationModule(width, classes)

    def set_tau(self, tau):
        for h in self.hidden:
            h.set_tau(tau)
        self.head.set_tau(tau)

    def forward(self, x):
        for h in self.hidden:
            x = ttF.tropical_relu(h(x))
        return self.head(x)


def run_backprop_tropical(data, depth, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    m = TropicalStack(Xtr.size(1), args.width, depth, args.classes)
    opt = optim.SignSGD(m.parameters(), lr=2.0 ** -args.lr_shift)
    t0 = time.time()
    for ep in range(args.epochs):
        m.set_tau(max(0.05, 1.0 * (1 - ep / max(1, args.epochs - 1))))   # soft->hard anneal
        for x, y in iterate(Xtr, Ytr, args.batch, args.seed + ep):
            opt.zero_grad()
            ttF.tropical_cross_entropy(m(x), y).backward()
            opt.step()
    m.set_tau(0.0)                                                       # closed deploy
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == Yte).float().mean().item()
    return {"acc": acc, "secs": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in m.parameters())}


# ---------------------------------------------------------------------------
# 3) Dense Forward-Forward (Hinton's float FF: x^2 goodness, L2 norm) — NOT closed
# ---------------------------------------------------------------------------
class DenseFF(nn.Module):
    """Faithful (non-tropical) Forward-Forward: per-layer local logistic loss on a
    sum-of-squares goodness, L2 length-normalization between layers, ReLU, Adam.
    This is the float reference — it uses real multiplies throughout."""

    def __init__(self, dims, classes, lr=3e-3, theta=2.0):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.opts = [torch.optim.Adam(l.parameters(), lr=lr) for l in self.layers]
        self.classes = classes
        self.theta = theta

    @staticmethod
    def _norm(h):
        return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def _goodness(h):
        return h.pow(2).sum(dim=-1)                       # Hinton's sum of squares (a multiply)

    def train_step(self, x_pos, x_neg):
        hp, hn = x_pos, x_neg
        sep = []
        for layer, opt in zip(self.layers, self.opts):
            ap = F.relu(layer(hp.detach()))
            an = F.relu(layer(hn.detach()))
            gp, gn = self._goodness(ap), self._goodness(an)
            # logistic FF loss: positives want goodness>theta, negatives below.
            loss = (F.softplus(self.theta - gp) + F.softplus(gn - self.theta)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            sep.append((gp.mean().item(), gn.mean().item()))
            hp = self._norm(F.relu(layer(hp.detach()))).detach()
            hn = self._norm(F.relu(layer(hn.detach()))).detach()
        return sep

    @torch.no_grad()
    def goodness_score(self, x, skip_first=True):
        h = x; total = None; n = len(self.layers)
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if not (skip_first and n > 1 and i == 0):
                g = self._goodness(h)
                total = g if total is None else total + g
            h = self._norm(h)
        return total


def run_dense_ff(data, depth, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    dims = [Xtr.size(1)] + [args.width] * depth
    net = DenseFF(dims, args.classes)
    t0 = time.time()
    for ep in range(args.epochs):
        for x, y in iterate(Xtr, Ytr, args.batch, args.seed + ep):
            xp = overlay_label(x, y, args.classes, on=args.on)
            xn = overlay_label(x, make_negative_labels(y, args.classes), args.classes, on=args.on)
            net.train_step(xp, xn)
    # goodness classifier (overlay each label, pick the max-goodness one)
    B = Xte.size(0)
    scores = torch.zeros(B, args.classes)
    with torch.no_grad():
        for c in range(args.classes):
            xc = overlay_label(Xte, torch.full((B,), c), args.classes, on=args.on)
            scores[:, c] = net.goodness_score(xc, skip_first=args.skip_first)
    acc = (scores.argmax(1) == Yte).float().mean().item()
    return {"acc": acc, "secs": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in net.parameters())}


# ---------------------------------------------------------------------------
# 4) Dense backprop MLP — the absolute reference
# ---------------------------------------------------------------------------
def run_dense_backprop(data, depth, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    dims = [Xtr.size(1)] + [args.width] * depth + [args.classes]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    m = nn.Sequential(*layers)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(args.epochs):
        for x, y in iterate(Xtr, Ytr, args.batch, args.seed + ep):
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == Yte).float().mean().item()
    return {"acc": acc, "secs": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in m.parameters())}


# ---------------------------------------------------------------------------
# Closure receipt: prove the TFF train step is multiplier-free (an assertion)
# ---------------------------------------------------------------------------
def closure_receipt(data, args):
    Xtr, Ytr, _, _ = data
    net = TropicalFFNet([Xtr.size(1), args.width, args.width], num_classes=args.classes,
                        goodness=args.goodness, k=args.k)
    trainer = TropicalFFTrainer(net, lr_shift=args.lr_shift, momentum_bits=args.momentum_bits,
                                margin=args.margin)
    x, y = Xtr[:13], Ytr[:13]                                      # non-power-of-2 batch
    xp = overlay_label(x, y, args.classes, on=args.on)
    xn = overlay_label(x, make_negative_labels(y, args.classes), args.classes, on=args.on)
    with tt.assert_no_multiply():
        trainer.train_step(xp, xn)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--width", type=int, default=500)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--goodness", choices=["topk", "max"], default="topk")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--on", type=float, default=1.0, help="label-overlay 'on' value")
    p.add_argument("--lr-shift", type=int, default=4)
    p.add_argument("--momentum-bits", type=int, default=0)
    p.add_argument("--skip-first", action="store_true", default=False,
                   help="drop layer-0 goodness from the classifier (Hinton); off by default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="smaller/faster sanity run")
    p.add_argument("--smoke", action="store_true", help="1 step on synthetic data; asserts closure")
    p.add_argument("--out", type=str, default="manuscript/results/forward_forward/mnist_results.jsonl")
    p.add_argument("--no-baselines", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.train_size, args.test_size, args.width, args.epochs = 3000, 1000, 200, 4
        args.depths = [2, 4]

    if args.smoke:
        args.train_size, args.test_size, args.epochs, args.depths = 256, 128, 1, [2]
        if args.out == p.get_default("out"):     # don't clobber the canonical artifact
            args.out = "/tmp/tff_results_smoke.jsonl"
        data = synthetic(args.train_size, args.test_size)
    else:
        data = load_mnist(args.train_size, args.test_size)

    print(f"== Tropical Forward-Forward experiment ==  device=cpu seed={args.seed}")
    print(f"train={args.train_size} test={args.test_size} width={args.width} "
          f"epochs={args.epochs} goodness={args.goodness} k={args.k} margin={args.margin} "
          f"on={args.on} lr=2^-{args.lr_shift} momentum_bits={args.momentum_bits} "
          f"skip_first={args.skip_first}")

    ok = closure_receipt(data, args)
    print(f"closure receipt: full TFF train step multiplier-free = {ok}  "
          f"(verified by tt.assert_no_multiply on a batch of 13)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    meta = {k: getattr(args, k) for k in ("width", "epochs", "batch", "train_size",
            "test_size", "goodness", "k", "margin", "on", "lr_shift", "momentum_bits",
            "skip_first", "seed")}

    print("\n%-20s %6s %8s %8s %s" % ("config", "depth", "acc", "secs", "params"))
    print("-" * 60)
    runners = [("tff", run_tff)]
    if not args.no_baselines:
        runners += [("backprop_tropical", run_backprop_tropical),
                    ("dense_ff", run_dense_ff),
                    ("dense_backprop", run_dense_backprop)]
    for depth in args.depths:
        for name, fn in runners:
            r = fn(data, depth, args)
            row = {"config": name, "depth": depth, **r, **meta}
            rows.append(row)
            extra = ""
            if "layer_separation" in r:
                extra = f"  sep(gp-gn)/layer={r['layer_separation']}"
            print("%-20s %6d %8.4f %8.1f %d%s" % (name, depth, r["acc"], r["secs"],
                                                  r["params"], extra))

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.out}")

    # Compact depth-scaling view for the report.
    print("\ndepth-scaling (test acc):")
    print("%-20s %s" % ("config", "  ".join(f"d{d}" for d in args.depths)))
    for name, _ in runners:
        accs = {row["depth"]: row["acc"] for row in rows if row["config"] == name}
        print("%-20s %s" % (name, "  ".join(f"{accs.get(d, float('nan')):.3f}" for d in args.depths)))


if __name__ == "__main__":
    main()
