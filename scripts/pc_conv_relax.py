"""Frontier 2: error-relaxation on CONVOLUTIONS.

Shows the async/iterative credit-assignment that worked on the MLP also works on a
convolutional net: relax the frozen-routing adjoint E_l = A_l^T E_{l+1} where A_l^T is
conv_transpose with the cached relu mask. Sync converges in one pass to the backprop
gradient; async (partial participation / stale reads / random order) stays robust
because the frozen-routing adjoint is a triangular (nilpotent) system.

Minimal clean net: a stack of stride-1 dyadic 3x3 convs + ReLU (same spatial size, so
the only routing is the relu mask), global average pool, dyadic linear head. No
maxpool / BN (those add index-unpool / freeze machinery — a follow-up). CE top-error.

  bp        -- autograd cross-entropy on the dyadic net (the backprop reference)
  relax     -- frozen-routing error-relaxation, synchronous (== bp credit)
  relax_async -- same, with 50% participation + stale reads + random layer order

Usage:  python scripts/pc_conv_relax.py --depth 6 --epochs 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.nn import dyadic_quantize as dq

_NORM = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


class DyadicConv2d(nn.Conv2d):
    _zero_dsp_layer = True

    def forward(self, x):
        return self._conv_forward(x, dq(self.weight), self.bias)


class DyadicConvNet(nn.Module):
    """stride-1 dyadic conv + ReLU stack -> global avg-pool -> dyadic linear head."""

    def __init__(self, depth, width=64, n_classes=10):
        super().__init__()
        self.convs = nn.ModuleList()
        c_in = 3
        for _ in range(depth):
            self.convs.append(DyadicConv2d(c_in, width, 3, padding=1, bias=True))
            c_in = width
        from tikitorch.nn import DyadicLinear
        self.head = DyadicLinear(width, n_classes)
        self.width = width

    def features(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x                                  # (B, width, H, W)

    def forward(self, x):
        f = self.features(x)
        pooled = f.mean(dim=(2, 3))               # global average pool
        return self.head(pooled)


@torch.no_grad()
def forward_cache(net, x):
    """Forward, caching per-conv relu mask + pre-pool feature. Returns (acts, masks, pooled)."""
    acts = [x]
    masks = []
    h = x
    for conv in net.convs:
        pre = conv(h)
        m = (pre > 0).float()
        h = pre * m
        masks.append(m)
        acts.append(h)
    pooled = h.mean(dim=(2, 3))
    return acts, masks, pooled


@torch.no_grad()
def error_relax_conv(net, x, y, n_iter, participation=1.0, stale=False,
                     rand_order=False, gen=None):
    """Set .grad via frozen-routing async error-relaxation through the conv stack."""
    acts, masks, pooled = forward_cache(net, x)
    L = len(net.convs)
    B, W, H, Wd = acts[-1].shape
    logits = net.head(pooled)
    onehot = F.one_hot(y, logits.shape[1]).float()
    e_logits = torch.softmax(logits, -1) - onehot                    # CE top-error
    # head weight grad (dyadic operands) + error into the pooled features
    net.head.weight.grad = dq(e_logits).transpose(0, 1) @ dq(pooled)
    net.head.bias.grad = dq(e_logits).sum(0)
    Wq_head = dq(net.head.weight)
    e_pooled = e_logits @ Wq_head                                    # (B, W)
    e_feat_top = e_pooled[:, :, None, None].expand(B, W, H, Wd) / (H * Wd)  # avgpool adjoint
    # E[j] = error w.r.t. acts[j] (post-relu feature). conv j maps acts[j]->acts[j+1];
    # delta at its (pre-relu) output = E[j+1]*mask[j]; adjoint into its input is
    # conv_transpose(delta) = E[j]. E[L] clamped to the pooled top-error; E[1..L-1] relaxed.
    E = [torch.zeros_like(a) for a in acts]
    E[L] = e_feat_top
    free = list(reversed(range(1, L)))                               # [L-1, ..., 1]
    for _ in range(n_iter):
        snap = [t.clone() for t in E] if stale else E
        order = free
        if rand_order:
            order = [free[i] for i in torch.randperm(len(free), generator=gen).tolist()]
        for j in order:
            if participation < 1.0 and torch.rand(1, generator=gen).item() >= participation:
                continue
            d = snap[j + 1] * masks[j]                               # delta at conv j's output
            E[j] = F.conv_transpose2d(d, dq(net.convs[j].weight), padding=1)
        E[L] = e_feat_top
    for j in range(L):
        delta = E[j + 1] * masks[j]
        net.convs[j].weight.grad = torch.nn.grad.conv2d_weight(
            dq(acts[j]), net.convs[j].weight.shape, dq(delta), padding=1)
        net.convs[j].bias.grad = dq(delta).sum(dim=(0, 2, 3))


def load(name, root, n_tr, n_te):
    from torchvision import datasets, transforms
    mean, std = _NORM
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    te = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    def stack(ds, n):
        xs = torch.stack([ds[i][0] for i in range(n)])
        ys = torch.tensor([ds[i][1] for i in range(n)])
        return xs, ys
    return stack(tr, n_tr), stack(te, n_te)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./_data"))
    p.add_argument("--out", default="results/predictive_coding/conv_relax.jsonl")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (Xtr, ytr), (Xte, yte) = load("cifar10", args.data_root, args.n_train, args.n_test)
    Xtr, ytr, Xte, yte = (t.to(device) for t in (Xtr, ytr, Xte, yte))

    @torch.no_grad()
    def acc(net):
        out = []
        for i in range(0, Xte.shape[0], 256):
            out.append(net(Xte[i:i + 256]).argmax(-1))
        return (torch.cat(out) == yte).float().mean().item()

    def train(kind, seed=0):
        torch.manual_seed(seed)
        net = DyadicConvNet(args.depth, args.width).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        g = torch.Generator().manual_seed(seed)
        rg = torch.Generator().manual_seed(seed + 99)
        t0 = time.time()
        for ep in range(args.epochs):
            perm = torch.randperm(Xtr.shape[0], generator=g)
            for i in range(0, Xtr.shape[0], args.batch):
                idx = perm[i:i + args.batch]
                xb, yb = Xtr[idx], ytr[idx]
                opt.zero_grad(set_to_none=True)
                if kind == "bp":
                    F.cross_entropy(net(xb), yb).backward()
                elif kind == "relax":
                    error_relax_conv(net, xb, yb, n_iter=args.depth + 2)
                else:  # relax_async
                    error_relax_conv(net, xb, yb, n_iter=3 * args.depth,
                                     participation=0.5, stale=True, rand_order=True, gen=rg)
                opt.step()
        return round(acc(net), 4), round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "a")
    print(f"device={device} depth={args.depth}", flush=True)
    for kind in ("bp", "relax", "relax_async"):
        a, sec = train(kind)
        rec = {"depth": args.depth, "width": args.width, "mode": kind,
               "epochs": args.epochs, "test_acc": a, "wall_s": sec, "device": str(device)}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"  {kind:12s} test={a:.4f} ({sec}s)", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
