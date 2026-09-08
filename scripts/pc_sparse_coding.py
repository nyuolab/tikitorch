"""Generative PC demo: dyadic sparse coding on MNIST (tikitorch.nn SparseCodingPC).

Trains a multiplier-free sparse-coding generative PC with NO backprop (settle the
top-k code, local dyadic dictionary update), then answers inference queries by
settling — reconstruction and in-painting under arbitrary masks. The same trained
model handles any corruption pattern, which a feed-forward discriminative net cannot.

Usage:  PYTHONPATH=. python scripts/pc_sparse_coding.py
"""
from __future__ import annotations

import argparse
import os

import torch

from tikitorch.nn.predictive_coding import SparseCodingPC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms", type=int, default=196)
    p.add_argument("--active", type=int, default=15)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--n-train", type=int, default=15000)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "./_data"))
    p.add_argument("--out-png", default="manuscript/results/predictive_coding/sparse_pc_infill.png")
    args = p.parse_args()

    from torchvision import datasets
    torch.manual_seed(0)
    ds = datasets.MNIST(args.data_root, train=True, download=True)
    dt = datasets.MNIST(args.data_root, train=False, download=True)
    Xtr = (ds.data[:args.n_train].float() / 255.0).reshape(-1, 784)
    Xte = (dt.data[:64].float() / 255.0).reshape(-1, 784)

    model = SparseCodingPC(n_pixels=784, n_atoms=args.atoms, n_active=args.active)
    print(f"training {model} (no backprop) ...", flush=True)
    for ep in range(args.epochs):
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0] - 128, 128):
            model.train_step(Xtr[perm[i:i + 128]], lr_dict=0.05, n_steps=25)
        if ep % 10 == 0 or ep == args.epochs - 1:
            mse = ((model.reconstruct(Xte) - Xte) ** 2).mean().item()
            print(f"  ep{ep} recon MSE={mse:.4f}", flush=True)

    g = torch.Generator().manual_seed(1)
    meanimg = Xtr.mean(0)

    def report(name, mask):
        filled = model.complete(Xte, mask, n_steps=150)
        miss = 1.0 - mask
        pc = (((filled - Xte) ** 2) * miss).sum() / miss.sum()
        mf = (((meanimg - Xte) ** 2) * miss).sum() / miss.sum()
        print(f"  {name}: sparse-PC MSE={pc:.4f}  mean-fill={mf:.4f}  "
              f"({100 * (1 - pc / mf):.0f}% better)", flush=True)
        return Xte * mask, filled

    print("in-painting (MSE on the MISSING pixels):", flush=True)
    mb = torch.ones(64, 784); mb.reshape(64, 28, 28)[:, 14:, :] = 0
    report("bottom-half", mb)
    mr = (torch.rand(64, 784, generator=g) > 0.5).float()
    cor, filled = report("random-50%", mr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Dn = model._dict().detach()
    usage = torch.zeros(args.atoms)
    s, _ = model.infer(Xtr[:2000]); usage += (s.abs() > 0).float().sum(0)
    top = usage.topk(12).indices
    fig, ax = plt.subplots(4, 12, figsize=(12, 4.2))
    for j in range(12):
        for r, im in enumerate([Xte[j], cor[j], filled[j]]):
            ax[r, j].imshow(im.reshape(28, 28).clamp(0, 1), cmap="gray"); ax[r, j].axis("off")
        ax[3, j].imshow(Dn[:, top[j]].reshape(28, 28), cmap="gray"); ax[3, j].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=85)
    print(f"saved {args.out_png}  (rows: orig / masked-50% / PC infill / top atoms)", flush=True)


if __name__ == "__main__":
    main()
