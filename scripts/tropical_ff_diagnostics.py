"""Tropical Forward-Forward — archived diagnostics.

Reproduces and PERSISTS the supporting diagnostic numbers cited in
``results/forward_forward/TROPICAL_FORWARD_FORWARD.md`` (so every claim traces to a
shipped row, not a transient console run). Writes one JSON row per probe to
``results/forward_forward/diagnostics.jsonl``.

Probes (all MNIST, seed 0, CPU):
  A. substrate_ceiling  — global backprop on a max-plus dilation MLP, Adam vs
     SignSGD, input dims {784,196,49}. Shows the ~0.30-0.36 wall is optimizer- and
     input-size-independent (§3).
  B. dense_ff_epochs    — dense (Hinton float) FF at {6,20,40} epochs. Shows it
     plateaus well below its full-scale ceiling under the matched small budget (§3/§4).
  C. scramble_probe     — unsupervised: train max-plus FF on real-vs-scramble
     negatives, linear-probe the features vs a raw-pixel control (§6).
  D. softtau_warmup     — max-plus TFF trained with a soft->hard tau anneal (§4).

Usage::

    python scripts/tropical_ff_diagnostics.py             # all probes (~12 min CPU)
    python scripts/tropical_ff_diagnostics.py --only A B   # a subset
    python scripts/tropical_ff_diagnostics.py --quick      # smaller/faster
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import tikitorch.nn as ttnn
import tikitorch.optim as optim
import tikitorch.functional as ttF
from tikitorch.forward_forward import (
    TropicalFFNet, TropicalFFTrainer, overlay_label, make_negative_labels,
    scramble_features, goodness_classify,
)

OUT = "results/forward_forward/diagnostics.jsonl"


def load(ntr, nte, root="./data"):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(ntr)]); Ytr = torch.tensor([tr[i][1] for i in range(ntr)])
    Xte = torch.stack([te[i][0] for i in range(nte)]); Yte = torch.tensor([te[i][1] for i in range(nte)])
    return Xtr, Ytr, Xte, Yte


def _iter(X, Y, B, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(len(X), generator=g)
    for i in range(0, len(X) - B + 1, B):
        yield X[p[i:i + B]], Y[p[i:i + B]]


def _downsample(X, f):
    return F.avg_pool2d(X.view(-1, 1, 28, 28), f).flatten(1)


# A. substrate ceiling: max-plus dilation MLP, Adam vs SignSGD, input dims --------
def probe_substrate_ceiling(data, args):
    Xtr0, Ytr, Xte0, Yte = data
    rows = []
    for din, fac in [(784, None), (196, 2), (49, 4)]:
        Xtr = _downsample(Xtr0, fac) if fac else Xtr0
        Xte = _downsample(Xte0, fac) if fac else Xte0
        for opt_name in ("adam", "sign"):
            torch.manual_seed(0)

            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.l = nn.ModuleList()
                    d = din
                    for _ in range(3):
                        s.l.append(ttnn.TropicalDilationModule(d, 300)); d = 300
                    s.head = ttnn.TropicalDilationModule(300, 10)

                def set_tau(s, t):
                    for layer in s.l:
                        layer.set_tau(t)
                    s.head.set_tau(t)

                def forward(s, x):
                    for layer in s.l:
                        x = ttF.tropical_relu(layer(x))
                    return s.head(x)
            m = Net()
            opt = (torch.optim.Adam(m.parameters(), 1e-3) if opt_name == "adam"
                   else optim.SignSGD(m.parameters(), lr=2.0 ** -8))
            for ep in range(args.ceiling_epochs):
                m.set_tau(max(0.05, 1.0 * (1 - ep / max(1, args.ceiling_epochs - 1))))
                for x, y in _iter(Xtr, Ytr, 100, ep):
                    opt.zero_grad(); ttF.tropical_cross_entropy(m(x), y).backward(); opt.step()
            m.set_tau(0.0)
            with torch.no_grad():
                acc = (m(Xte).argmax(1) == Yte).float().mean().item()
            rows.append({"probe": "substrate_ceiling", "input_dim": din, "optimizer": opt_name,
                         "acc": round(acc, 4), "arch": "maxplus_dilation_300x3",
                         "epochs": args.ceiling_epochs, "train_size": len(Xtr)})
            print(f"  [A] din={din} {opt_name}: acc={acc:.4f}", flush=True)
    return rows


# B. dense-FF epoch sweep ---------------------------------------------------------
def probe_dense_ff_epochs(data, args):
    Xtr, Ytr, Xte, Yte = data
    rows = []

    class DenseFF(nn.Module):
        def __init__(s, dims, lr=3e-3, theta=2.0):
            super().__init__()
            s.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
            s.opts = [torch.optim.Adam(l.parameters(), lr) for l in s.layers]
            s.theta = theta

        @staticmethod
        def _n(h):
            return h / (h.norm(dim=-1, keepdim=True) + 1e-8)

        @staticmethod
        def _g(h):
            return h.pow(2).sum(-1)

        def step(s, xp, xn):
            hp, hn = xp, xn
            for layer, opt in zip(s.layers, s.opts):
                ap = F.relu(layer(hp.detach())); an = F.relu(layer(hn.detach()))
                loss = (F.softplus(s.theta - s._g(ap)) + F.softplus(s._g(an) - s.theta)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                hp = s._n(F.relu(layer(hp.detach()))).detach()
                hn = s._n(F.relu(layer(hn.detach()))).detach()

        @torch.no_grad()
        def score(s, x):
            h = x; tot = None
            for i, layer in enumerate(s.layers):
                h = F.relu(layer(h))
                g = s._g(h); tot = g if tot is None else tot + g
                h = s._n(h)
            return tot
    for E in args.dense_ff_epochs_list:
        torch.manual_seed(0)
        net = DenseFF([784, 300, 300])
        for ep in range(E):
            for x, y in _iter(Xtr, Ytr, 100, ep):
                xp = overlay_label(x, y, 10); xn = overlay_label(x, make_negative_labels(y, 10), 10)
                net.step(xp, xn)
        B = Xte.size(0); scores = torch.zeros(B, 10)
        with torch.no_grad():
            for c in range(10):
                scores[:, c] = net.score(overlay_label(Xte, torch.full((B,), c), 10))
        acc = (scores.argmax(1) == Yte).float().mean().item()
        rows.append({"probe": "dense_ff_epochs", "epochs": E, "acc": round(acc, 4),
                     "arch": "dense_FF_300x2_sumsq_goodness", "train_size": len(Xtr)})
        print(f"  [B] dense_ff epochs={E}: acc={acc:.4f}", flush=True)
    return rows


# C. unsupervised scramble-feature probe vs raw pixels ----------------------------
def probe_scramble(data, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(0)
    net = TropicalFFNet([784, 500, 500], num_classes=10, goodness="topk", k=10)
    trainer = TropicalFFTrainer(net, lr_shift=4, momentum_bits=0, margin=1.0)
    for ep in range(args.scramble_epochs):
        for x, _ in _iter(Xtr, Ytr, 100, ep):
            trainer.train_step(x, scramble_features(x))

    @torch.no_grad()
    def feats(X):
        acts = net.layer_activations(X)
        return torch.cat([net.layers[i].normalize(a) for i, a in enumerate(acts)], dim=1)
    Ftr, Fte = feats(Xtr), feats(Xte)

    def probe(Z, Zt):
        torch.manual_seed(0)
        lin = nn.Linear(Z.size(1), 10); opt = torch.optim.Adam(lin.parameters(), 1e-2)
        for _ in range(60):
            opt.zero_grad(); F.cross_entropy(lin(Z), Ytr).backward(); opt.step()
        return (lin(Zt).argmax(1) == Yte).float().mean().item()
    acc_feat = probe(Ftr, Fte)
    acc_raw = probe(Xtr, Xte)
    print(f"  [C] scramble-feature probe={acc_feat:.4f}  raw-pixel probe={acc_raw:.4f}", flush=True)
    return [{"probe": "scramble_linear_probe", "features": "ff_scramble_neg", "acc": round(acc_feat, 4),
             "train_size": len(Xtr), "note": "unsupervised real-vs-scramble features, linear probe"},
            {"probe": "scramble_linear_probe", "features": "raw_pixels_control", "acc": round(acc_raw, 4),
             "train_size": len(Xtr)}]


# D. soft-tau warmup (max-plus TFF) ----------------------------------------------
def probe_softtau(data, args):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(0)
    net = TropicalFFNet([784, 500, 500], num_classes=10, goodness="topk", k=10)
    trainer = TropicalFFTrainer(net, lr_shift=4, momentum_bits=0, margin=1.0)
    E = args.softtau_epochs
    for ep in range(E):
        net.set_tau(max(0.0, 1.0 * (1 - ep / max(1, E - 1))))     # soft -> hard
        for x, y in _iter(Xtr, Ytr, 100, ep):
            xp = overlay_label(x, y, 10); xn = overlay_label(x, make_negative_labels(y, 10), 10)
            trainer.train_step(xp, xn)
    net.set_tau(0.0)
    preds, _ = goodness_classify(net, Xte, 10, skip_first=False)
    acc = (preds == Yte).float().mean().item()
    print(f"  [D] soft-tau warmup -> hard eval: acc={acc:.4f}", flush=True)
    return [{"probe": "softtau_warmup", "acc": round(acc, 4), "arch": "maxplus_TFF_500x2",
             "epochs": E, "train_size": len(Xtr), "note": "soft logsumexp anneal -> hard deploy"}]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["A", "B", "C", "D"], default=["A", "B", "C", "D"])
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--test-size", type=int, default=2000)
    p.add_argument("--ceiling-epochs", type=int, default=10)
    p.add_argument("--dense-ff-epochs-list", type=int, nargs="+", default=[6, 20, 40])
    p.add_argument("--scramble-epochs", type=int, default=6)
    p.add_argument("--softtau-epochs", type=int, default=6)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=str, default=OUT)
    args = p.parse_args()
    if args.quick:
        (args.train_size, args.test_size, args.ceiling_epochs, args.dense_ff_epochs_list,
         args.scramble_epochs, args.softtau_epochs) = 2000, 1000, 4, [4, 8], 4, 4

    data = load(args.train_size, args.test_size)
    print(f"== TFF diagnostics ==  train={len(data[0])} test={len(data[2])} seed=0")
    probes = {"A": probe_substrate_ceiling, "B": probe_dense_ff_epochs,
              "C": probe_scramble, "D": probe_softtau}
    rows = []
    t0 = time.time()
    for key in args.only:
        print(f"[{key}] {probes[key].__name__} ...", flush=True)
        rows += probes[key](data, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # merge with any existing rows for probes we didn't run this time
    existing = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = [json.loads(l) for l in f if l.strip()]
    ran = set(args.only)
    keep_probes = {"A": "substrate_ceiling", "B": "dense_ff_epochs",
                   "C": "scramble_linear_probe", "D": "softtau_warmup"}
    drop = {keep_probes[k] for k in ran}
    merged = [r for r in existing if r.get("probe") not in drop] + rows
    with open(args.out, "w") as f:
        for r in merged:
            f.write(json.dumps({**r, "seed": 0}) + "\n")
    print(f"\nwrote {len(rows)} new rows ({len(merged)} total) -> {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
