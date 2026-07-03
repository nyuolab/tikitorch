"""Tropical Forward-Forward — MNIST benchmark vs the Forward-Forward paper.

Trains the two closure-clean Forward-Forward substrates with the SAME forward-only,
backprop-free local rule and compares them to Hinton's FF paper (2022) baseline
(~1.36% test error ≈ 98.64%, a *dense* 4×2000 net):

    * DyadicForwardForward   — dense ±2^k (shift-add, zero-DSP) layers. Dense/
      additive, so the FF label overlay composes like the float paper -> high acc.
    * TropicalForwardForward — max-plus dilation layers. Winner-take-all, the
      faithful tropical substrate; caps low (the measured ~0.3 wall).

Plus the references that frame the gap: dense backprop and dyadic backprop (a
multiplier-free dense net trained globally) — the substrate ceilings.

Usage::

    python scripts/tropical_ff_mnist_benchmark.py                 # thorough (full 60k)
    python scripts/tropical_ff_mnist_benchmark.py --quick         # fast subset
    python scripts/tropical_ff_mnist_benchmark.py --smoke         # 1 epoch, tiny

Writes JSONL to results/forward_forward/ and prints a comparison table.
"""
import argparse
import json
import os
import sys
import time

# Use the co-located tikitorch (the standalone tree), not an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import tikitorch as tt
import tikitorch.optim as optim
import tikitorch.nn as ttnn
from tikitorch.forward_forward import goodness_classify

PAPER_ACC = 0.9864      # Hinton 2022, FF, MNIST, 1.36% test error (dense 4x2000).


def load_mnist(train_size, test_size, root="./data"):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    ntr, nte = min(train_size, len(tr)), min(test_size, len(te))
    Xtr = torch.stack([tr[i][0] for i in range(ntr)]); Ytr = torch.tensor([tr[i][1] for i in range(ntr)])
    Xte = torch.stack([te[i][0] for i in range(nte)]); Yte = torch.tensor([te[i][1] for i in range(nte)])
    return Xtr, Ytr, Xte, Yte


def synthetic(ntr, nte, d=784, c=10):
    def make(n):
        y = torch.randint(0, c, (n,)); x = torch.rand(n, d) * 0.1
        for cls in range(c):
            x[y == cls, cls * 8:(cls + 1) * 8] += 2.0
        return x, y
    return (*make(ntr), *make(nte))


def _iter(X, Y, batch, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    for i in range(0, len(X) - batch + 1, batch):
        idx = perm[i:i + batch]
        yield X[idx], Y[idx]


@torch.no_grad()
def ff_accuracy(model, Xte, Yte, batch=1000):
    correct = 0
    for i in range(0, len(Xte), batch):
        preds, _ = goodness_classify(model.net, Xte[i:i + batch], model.num_classes,
                                     on=model.on, off=model.off, skip_first=model.skip_first)
        correct += int((preds == Yte[i:i + batch]).sum())
    return correct / len(Xte)


# --- forward-forward (either substrate) ---------------------------------------
def run_ff(data, substrate, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    width = args.dyadic_width if substrate == "dyadic" else args.maxplus_width
    depth = args.dyadic_depth if substrate == "dyadic" else args.maxplus_depth
    epochs = args.dyadic_epochs if substrate == "dyadic" else args.maxplus_epochs
    # max-plus is slow (the (B,out,in) expand) — cap its train set independently.
    n = len(Xtr) if substrate == "dyadic" else min(len(Xtr), args.maxplus_train_size)
    Xtr, Ytr = Xtr[:n], Ytr[:n]

    cls = tt.models.DyadicForwardForward if substrate == "dyadic" else tt.models.TropicalForwardForward
    model = cls(d_model=width, n_layers=depth, input_dim=Xtr.size(1), num_classes=10,
                k=args.k, on=args.on, skip_first=args.skip_first)
    opt_factory = (lambda p: optim.SignSGD(p, lr=2.0 ** -args.lr_shift))
    trainer = model.make_trainer(optimizer_factory=opt_factory, margin=args.margin)

    t0 = time.time()
    for ep in range(epochs):
        if substrate == "maxplus":   # soft->hard anneal helps the max-plus path
            model.set_tau(max(0.0, 1.0 * (1 - ep / max(1, epochs - 1))))
        for x, y in _iter(Xtr, Ytr, args.batch, args.seed + ep):
            model.forward_forward_step(x, y, trainer)
    if substrate == "maxplus":
        model.set_tau(0.0)
    acc = ff_accuracy(model, Xte, Yte)
    return {"acc": acc, "secs": round(time.time() - t0, 1), "params": model.param_count,
            "train_size": n, "width": width, "depth": depth, "epochs": epochs}


# --- references: global backprop substrate ceilings ---------------------------
def run_dense_backprop(data, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    dims = [Xtr.size(1), 1024, 1024, 1024, 10]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    m = nn.Sequential(*layers); opt = torch.optim.Adam(m.parameters(), 1e-3)
    t0 = time.time()
    for ep in range(args.ref_epochs):
        for x, y in _iter(Xtr, Ytr, 128, args.seed + ep):
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == Yte).float().mean().item()
    return {"acc": acc, "secs": round(time.time() - t0, 1)}


def run_dyadic_backprop(data, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(args.seed)
    dims = [Xtr.size(1), 1024, 1024, 1024, 10]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.ModuleList([ttnn.DyadicLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

        def forward(self, x):
            for i, layer in enumerate(self.l):
                x = layer(x)
                if i < len(self.l) - 1:
                    x = F.relu(x)
            return x
    m = Net(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    t0 = time.time()
    for ep in range(args.ref_epochs):
        for x, y in _iter(Xtr, Ytr, 128, args.seed + ep):
            opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
    with torch.no_grad():
        acc = (m(Xte).argmax(1) == Yte).float().mean().item()
    return {"acc": acc, "secs": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are a CPU-reasonable "thorough" run (~20-40 min). The 0.9177 headline
    # used the heavier --dyadic-width 2000 --dyadic-depth 4 --dyadic-epochs 25 (full
    # 60k, ~5 h on CPU; minutes on GPU). Scale up for the best number.
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--test-size", type=int, default=10000)
    p.add_argument("--dyadic-width", type=int, default=1000)
    p.add_argument("--dyadic-depth", type=int, default=3)
    p.add_argument("--dyadic-epochs", type=int, default=12)
    p.add_argument("--maxplus-width", type=int, default=500)
    p.add_argument("--maxplus-depth", type=int, default=3)
    p.add_argument("--maxplus-epochs", type=int, default=12)
    p.add_argument("--maxplus-train-size", type=int, default=20000,
                   help="max-plus is slow (the (B,out,in) expand); cap its train set")
    p.add_argument("--ref-epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--on", type=float, default=1.0)
    p.add_argument("--lr-shift", type=int, default=8)
    p.add_argument("--skip-first", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-refs", action="store_true")
    p.add_argument("--out", type=str, default="results/forward_forward/mnist_benchmark.jsonl")
    args = p.parse_args()

    if args.quick:
        (args.train_size, args.test_size, args.dyadic_width, args.dyadic_depth,
         args.dyadic_epochs, args.maxplus_train_size, args.maxplus_epochs,
         args.ref_epochs) = 10000, 2000, 800, 3, 10, 8000, 8, 6
    if args.smoke:
        (args.train_size, args.test_size, args.dyadic_width, args.dyadic_depth,
         args.dyadic_epochs, args.maxplus_width, args.maxplus_epochs,
         args.maxplus_train_size, args.ref_epochs) = 512, 256, 64, 2, 1, 64, 1, 512, 1
        if args.out == p.get_default("out"):     # don't clobber the canonical artifact
            args.out = "/tmp/tff_benchmark_smoke.jsonl"

    data = (synthetic(args.train_size, args.test_size) if args.smoke
            else load_mnist(args.train_size, args.test_size))

    print(f"== Tropical Forward-Forward — MNIST benchmark vs FF paper ({PAPER_ACC:.4f}) ==")
    print(f"train={len(data[0])} test={len(data[2])} batch={args.batch} k={args.k} "
          f"margin={args.margin} on={args.on} lr=2^-{args.lr_shift} seed={args.seed}")

    rows = []
    print("\n%-26s %8s %8s %10s %s" % ("config", "acc", "secs", "params", "detail"))
    print("-" * 78)

    runners = [
        ("dyadic_ff (forward-only)", lambda: run_ff(data, "dyadic", args)),
        ("maxplus_ff (forward-only)", lambda: run_ff(data, "maxplus", args)),
    ]
    if not args.no_refs:
        runners += [("dyadic_backprop (ref)", lambda: run_dyadic_backprop(data, args)),
                    ("dense_backprop (ref)", lambda: run_dense_backprop(data, args))]
    for name, fn in runners:
        r = fn()
        rows.append({"config": name, **r})
        detail = (f"w={r.get('width')} d={r.get('depth')} ep={r.get('epochs')} "
                  f"ntr={r.get('train_size')}" if "width" in r else "")
        print("%-26s %8.4f %8.1f %10s %s" % (name, r["acc"], r["secs"],
                                             str(r.get("params", "-")), detail))

    print(f"\n{'FF paper (Hinton 2022, dense 4x2000)':<26} {PAPER_ACC:8.4f}  (reference target)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps({**row, "paper_acc": PAPER_ACC, "seed": args.seed}) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
