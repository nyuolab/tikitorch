"""Phase 0 scale test: tropical/dyadic predictive coding on FULL MNIST & Fashion-MNIST.

Moves the dyadic-PC results off the 12x12 toy onto full-resolution datasets with
deep MLPs and multiple seeds. This is the muPC-style benchmark (deep MLP on
MNIST/Fashion-MNIST) and directly tests the two PC regimes against their honest
baselines:

  cascade    -- DyadicPCNet single-pass cached-routing backward (the depth
                solution; multiplier-free). Should track dyadic-backprop.
  nudge      -- DyadicPCNet iterative nudged settling (the fully-local / async /
                distributed regime). Expected ~85% of backprop at depth.
  bp_dyadic  -- a +-2^k (DyadicLinear) ReLU MLP trained by ordinary backprop+Adam.
                The CLOSURE baseline: same multiplier-free forward, so the gap to
                cascade isolates "PC vs backprop", not "dyadic vs float".
  bp_float   -- a plain float nn.Linear ReLU MLP trained by backprop+Adam. The
                absolute ceiling: the gap to bp_dyadic is the price of being
                multiplier-free.

Each (dataset, depth, mode, seed) run logs one JSON line with the test accuracy.

Usage (full matrix, what the EC2 runner invokes):
    python scripts/pc_scale_mnist.py --matrix \
        --out results/predictive_coding/scale_mnist.jsonl

Usage (explicit):
    python scripts/pc_scale_mnist.py --datasets mnist fashion \
        --depths 4 8 16 32 --modes cascade bp_dyadic bp_float nudge \
        --seeds 0 1 2 --epochs 5 --hidden 256 --batch 128
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.nn import DyadicLinear
from tikitorch.nn.predictive_coding import DyadicPCNet

# Per-dataset (mean, std) for standardization.
_NORM = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}


def load_dataset(name: str, data_root: str, device: torch.device):
    """Return ((Xtr, ytr), (Xte, yte)) as flattened, standardized tensors on device."""
    from torchvision import datasets, transforms
    mean, std = _NORM[name]
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((mean,), (std,))])
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
    out = []
    for train in (True, False):
        ds = cls(root=data_root, train=train, download=True, transform=tf)
        X = ds.data.float()
        X = ((X / 255.0) - mean) / std            # match the transform, vectorized
        X = X.view(X.shape[0], -1).to(device)
        y = ds.targets.to(device)
        out.append((X, y))
    return out[0], out[1]


def make_dims(in_dim: int, hidden: int, depth: int, n_classes: int = 10):
    """depth = number of weight layers; dims has depth+1 entries."""
    return [in_dim] + [hidden] * (depth - 1) + [n_classes]


class DyadicMLP(nn.Module):
    """+-2^k (DyadicLinear) ReLU MLP with a linear readout -- the backprop closure
    baseline (same multiplier-free forward as DyadicPCNet). With ``residual`` it
    uses the TROPICAL max-skip ``max(relu(Wx), x)`` on matching interior dims,
    matching DyadicPCNet's cascade residual so cascade-vs-bp_dyadic stays apples
    to apples at depth."""

    def __init__(self, dims, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            [DyadicLinear(dims[l], dims[l + 1]) for l in range(len(dims) - 1)])

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            out = layer(x)
            if l < len(self.layers) - 1:
                out = torch.relu(out)
                if self.residual and out.shape[-1] == x.shape[-1]:
                    out = torch.maximum(out, x)        # tropical residual skip
            x = out
        return x


class FloatMLP(nn.Module):
    """Plain float ReLU MLP -- the absolute (non-multiplier-free) ceiling. With
    ``residual`` it uses the standard additive skip ``relu(Wx) + x``."""

    def __init__(self, dims, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            [nn.Linear(dims[l], dims[l + 1]) for l in range(len(dims) - 1)])

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            out = layer(x)
            if l < len(self.layers) - 1:
                out = torch.relu(out)
                if self.residual and out.shape[-1] == x.shape[-1]:
                    out = out + x                      # additive residual skip
            x = out
        return x


@torch.no_grad()
def accuracy(logits_fn, X, y, batch=1024):
    correct = 0
    for i in range(0, X.shape[0], batch):
        logits = logits_fn(X[i:i + batch])
        correct += (logits.argmax(-1) == y[i:i + batch]).sum().item()
    return correct / X.shape[0]


def iterate(X, y, batch, gen):
    perm = torch.randperm(X.shape[0], generator=gen, device=X.device)
    for i in range(0, X.shape[0], batch):
        idx = perm[i:i + batch]
        yield X[idx], y[idx]


def run_one(mode, dims, train, test, *, seed, epochs, batch, lr, n_sweeps,
            residual, device):
    """Train one configuration; return a metrics dict."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    Xtr, ytr = train
    Xte, yte = test
    t0 = time.time()

    if mode in ("cascade", "nudge", "relax", "relax_async"):
        net = DyadicPCNet(dims, residual=residual).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        depth = len(dims) - 1
        rgen = torch.Generator().manual_seed(seed + 99)   # CPU: async draws are control-flow
        for _ in range(epochs):
            for xb, yb in iterate(Xtr, ytr, batch, gen):
                if mode == "cascade":
                    net.train_step_cascade(xb, yb, opt, mode="dyadic")
                elif mode == "nudge":               # state-settling (diverges at scale)
                    net.train_step_nudge(xb, yb, opt, n_sweeps=n_sweeps,
                                         mf_mode="dyadic")
                elif mode == "relax":               # error-relaxation, sync (== cascade)
                    net.train_step_relax(xb, yb, opt, mode="dyadic")
                else:                               # error-relaxation, heavy async
                    net.train_step_relax(xb, yb, opt, n_iter=3 * depth, mode="dyadic",
                                         participation=0.5, stale=True,
                                         rand_order=True, gen=rgen)
        logits_fn = net.predict_logits
    elif mode in ("bp_dyadic", "bp_float"):
        net = (DyadicMLP(dims, residual=residual) if mode == "bp_dyadic"
               else FloatMLP(dims, residual=residual)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            for xb, yb in iterate(Xtr, ytr, batch, gen):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(net(xb), yb)
                loss.backward()
                opt.step()
        logits_fn = net.forward
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return {
        "test_acc": round(accuracy(logits_fn, Xte, yte), 4),
        "train_acc": round(accuracy(logits_fn, Xtr, ytr), 4),
        "wall_s": round(time.time() - t0, 1),
    }


def build_matrix(args):
    """The standard scale matrix. nudge is settling-heavy, so cap its depth/epochs."""
    jobs = []
    for dataset in args.datasets:
        for mode in args.modes:
            for depth in args.depths:
                # nudge settling is O(depth) sweeps per step -- bound the cost.
                if mode == "nudge" and depth > 16:
                    continue
                for seed in args.seeds:
                    jobs.append(dict(dataset=dataset, mode=mode, depth=depth,
                                     seed=seed))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["mnist", "fashion"])
    p.add_argument("--depths", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--modes", nargs="+",
                   default=["cascade", "bp_dyadic", "bp_float", "nudge"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-sweeps", type=int, default=15, help="nudge settling sweeps")
    p.add_argument("--residual", action="store_true",
                   help="tropical residual skip (helps the iterative regime go deep)")
    p.add_argument("--matrix", action="store_true",
                   help="run the standard scale matrix over all combos")
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./_data"))
    p.add_argument("--out", default="results/predictive_coding/scale_mnist.jsonl")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU sanity run (subset, 1 epoch)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.datasets = ["mnist"]
        args.depths = [4]
        args.modes = ["cascade", "bp_dyadic", "nudge"]
        args.seeds = [0]
        args.epochs = 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jobs = build_matrix(args) if args.matrix or not args.modes else build_matrix(args)

    # cache datasets once
    cache = {}
    for d in args.datasets:
        cache[d] = load_dataset(d, args.data_root, device)
        if args.smoke:
            (Xtr, ytr), (Xte, yte) = cache[d]
            cache[d] = ((Xtr[:2000], ytr[:2000]), (Xte[:2000], yte[:2000]))

    print(f"device={device}  jobs={len(jobs)}  out={args.out}", flush=True)
    fout = open(args.out, "a")
    for j, job in enumerate(jobs):
        train, test = cache[job["dataset"]]
        in_dim = train[0].shape[1]
        dims = make_dims(in_dim, args.hidden, job["depth"])
        ep = min(args.epochs, 3) if job["mode"] == "nudge" else args.epochs
        residual = args.residual or (job["mode"] == "nudge")
        m = run_one(job["mode"], dims, train, test, seed=job["seed"], epochs=ep,
                    batch=args.batch, lr=args.lr, n_sweeps=args.n_sweeps,
                    residual=residual, device=device)
        rec = {**job, "hidden": args.hidden, "epochs": ep, "residual": residual,
               "device": str(device), **m}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        print(f"[{j+1}/{len(jobs)}] {job['dataset']:8s} d{job['depth']:<3d} "
              f"{job['mode']:10s} s{job['seed']}  test={m['test_acc']:.4f}  "
              f"({m['wall_s']}s)", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
