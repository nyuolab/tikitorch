"""Path 2, step 1 (ceiling): does the DYADIC oblique primitive survive depth?

DyadicLinear+ReLU is a multiplier-free ReLU MLP (weights snapped to +/-2^k, so
mixing is a shift-add tree, not a lossy max). If a deep dyadic stack matches a
dense MLP and does NOT degrade with depth (unlike pure max-plus, which capped at
0.49 and fell to 0.12), the primitive is the fix and it's worth wiring into PC.
Backprop + Adam; this is the representational ceiling, not a closure claim.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.nn import DyadicLinear


def get_mnist(down, n_train, n_test):
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
    return Xtr, Ytr, Xte, Yte


def make_net(d, depth, n_classes, kind):
    Lin = DyadicLinear if kind == "dyadic" else nn.Linear
    layers, dim = [], d
    for l in range(depth):
        out = n_classes if l == depth - 1 else d
        layers.append(Lin(dim, out))
        if l < depth - 1:
            layers.append(nn.ReLU())
        dim = out
    return nn.Sequential(*layers)


def train(model, Xtr, Ytr, Xte, Yte, epochs, lr=1e-2, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        return (model(Xte).argmax(-1) == Yte).float().mean().item()


if __name__ == "__main__":
    down, n_train, n_test, epochs = 12, 8000, 2000, 12
    d = down * down
    Xtr, Ytr, Xte, Yte = get_mnist(down, n_train, n_test)
    print(f"== MNIST {down}x{down} (d={d}) — DYADIC vs DENSE ceiling (backprop+Adam) ==")
    print(f"{'depth':>6} {'dyadic(2^k)':>12} {'dense':>7} {'sec':>6}")
    for depth in (2, 4, 8, 16):
        t0 = time.time()
        torch.manual_seed(0)
        a_dy = train(make_net(d, depth, 10, "dyadic"), Xtr, Ytr, Xte, Yte, epochs)
        torch.manual_seed(0)
        a_de = train(make_net(d, depth, 10, "dense"), Xtr, Ytr, Xte, Yte, epochs)
        print(f"{depth:>6} {a_dy:>12.3f} {a_de:>7.3f} {time.time()-t0:>6.1f}")
