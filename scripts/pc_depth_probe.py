"""Tropical predictive-coding depth probe.

Validates the TropicalPCNet local learning rule and measures whether it trains
*deep* networks where conventional PC degrades. Three modes:

  smoke   -- closure check (no multiply) + a tiny synthetic task sanity learn
  depth   -- accuracy vs depth, forward-init vs zero-init, with winner-flip diag
  mnist   -- real MNIST (downsampled), accuracy vs depth

Run:
  python scripts/pc_depth_probe.py smoke
  python scripts/pc_depth_probe.py depth
  python scripts/pc_depth_probe.py mnist
"""
import sys
import time

import torch

import tikitorch as tt
from tikitorch.nn import TropicalPCNet


def gaussian_blobs(n_per, n_classes, d_in, seed=0, spread=0.6):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, d_in, generator=g) * 2.0
    xs, ys = [], []
    for c in range(n_classes):
        x = means[c] + spread * torch.randn(n_per, d_in, generator=g)
        xs.append(x)
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    X = torch.cat(xs)
    Y = torch.cat(ys)
    perm = torch.randperm(X.shape[0], generator=g)
    return X[perm], Y[perm]


def iterate(X, Y, bs):
    for i in range(0, X.shape[0] - bs + 1, bs):
        yield X[i:i + bs], Y[i:i + bs]


# --------------------------------------------------------------------------
def smoke():
    print("== closure check: one PC train step must run NO multiply ==")
    torch.manual_seed(0)
    net = TropicalPCNet([20, 32, 32, 4], gamma_shift=2, lr_shift=6)
    x = torch.randn(16, 20)
    y = torch.randint(0, 4, (16,))
    target = net._make_target(y, 4, ref=x)
    try:
        with tt.assert_no_multiply():
            out = net.infer(x, target=target, n_steps=6, init="forward")
            net._apply_local_updates(out["states"])
        print("   PASS: infer + local update are multiply-free (closure holds)\n")
    except tt.ClosureViolation as e:
        print("   FAIL: closure violated:\n", e, "\n")
        return

    print("== sanity learn: 4-class blobs, depth-3 net ==")
    X, Y = gaussian_blobs(400, 4, 20, seed=1)
    Xtr, Ytr, Xte, Yte = X[:1200], Y[:1200], X[1200:], Y[1200:]
    net = TropicalPCNet([20, 32, 32, 4], gamma_shift=2, lr_shift=5)
    acc0 = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    print(f"   pre-train test acc: {acc0:.3f}  (chance = 0.25)")
    for epoch in range(8):
        for xb, yb in iterate(Xtr, Ytr, 32):
            net.train_step(xb, yb, n_steps=8, init="forward")
        acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
        print(f"   epoch {epoch}: test acc {acc:.3f}")


# --------------------------------------------------------------------------
def depth_experiment(d=24, depths=(2, 5, 8, 12, 16, 24, 32), epochs=8,
                     n_classes=4, seed=1):
    """Baseline (uniform init) vs identity init across depth, square interior.

    Key finding: the uniform baseline collapses to chance by ~depth 8 (the
    top-clamp error never reaches the early layers); identity init keeps accuracy
    flat to depth 32. (Blobs are linearly separable, so this measures whether the
    *optimisation* survives depth, not whether depth adds representation.)
    """
    print(f"== depth scaling: {n_classes}-class blobs, square interior d={d} ==")
    X, Y = gaussian_blobs(500, n_classes, d, seed=seed)
    ntr = int(0.8 * X.shape[0])
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]

    print(f"{'depth':>6} {'baseline':>9} {'identity':>9} {'sec':>6}")
    rows = []
    for depth in depths:
        dims = [d] * depth + [n_classes]
        accs = {}
        for ident in (False, True):
            torch.manual_seed(0)
            net = TropicalPCNet(dims, gamma_shift=2, lr_shift=5, init_identity=ident)
            t0 = time.time()
            for epoch in range(epochs):
                for xb, yb in iterate(Xtr, Ytr, 32):
                    net.train_step(xb, yb, n_steps=10, init="forward")
            acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
            accs[ident] = (acc, time.time() - t0)
        print(f"{depth:>6} {accs[False][0]:>9.3f} {accs[True][0]:>9.3f} {accs[True][1]:>6.1f}")
        rows.append((depth, accs[False][0], accs[True][0]))
    return rows


# --------------------------------------------------------------------------
def mnist_experiment(depths=(2, 4, 8, 12), epochs=3, n_train=6000,
                     n_test=2000, down=12):
    """MNIST baseline vs identity init, square interior.

    HONEST: identity init prevents the depth collapse (baseline falls to ~chance
    by depth 8-12; identity holds ~0.27), but absolute accuracy is poor (~0.28-
    0.38 vs ~0.92 for a linear classifier). The interior layers stay ≈identity
    (inert); the effective model is a weak single max-plus readout that selects
    one pixel per class. See results/predictive_coding/TROPICAL_PC_DEPTH.md.
    """
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((down, down), antialias=True),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    tr = datasets.MNIST("data", train=True, download=True, transform=tf)
    te = datasets.MNIST("data", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(n_train)])
    Ytr = torch.tensor([tr[i][1] for i in range(n_train)])
    Xte = torch.stack([te[i][0] for i in range(n_test)])
    Yte = torch.tensor([te[i][1] for i in range(n_test)])
    d = down * down
    print(f"== MNIST {down}x{down} (d={d}), {n_train} train / {n_test} test, square interior ==")
    print(f"{'depth':>6} {'baseline':>9} {'identity':>9} {'sec':>6}")
    for depth in depths:
        dims = [d] * depth + [10]
        accs = {}
        for ident in (False, True):
            torch.manual_seed(0)
            net = TropicalPCNet(dims, gamma_shift=2, lr_shift=7, init_identity=ident)
            t0 = time.time()
            for epoch in range(epochs):
                for xb, yb in iterate(Xtr, Ytr, 64):
                    net.train_step(xb, yb, n_steps=10, init="forward")
            accs[ident] = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
        print(f"{depth:>6} {accs[False]:>9.3f} {accs[True]:>9.3f} {time.time()-t0:>6.1f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        smoke()
    elif mode == "depth":
        depth_experiment()
    elif mode == "mnist":
        mnist_experiment()
    else:
        print(f"unknown mode {mode!r}; use smoke|depth|mnist")
