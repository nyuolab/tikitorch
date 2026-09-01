"""Ceiling probe: how good are DEEP tropical features under the BEST rule?

Trains a deep alternating dilation/erosion net with full backprop + Adam and a
soft->hard tau anneal (the repo's intended training mode). This is NOT closure-
clean and NOT PC — it's a diagnostic to locate the ceiling. If even backprop+Adam
on soft tropical caps low on MNIST, the limit is the tropical primitive (cf the
§2a capacity verdict), not predictive coding. Dense linear is the ~0.9 anchor.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.nn import TropicalDilationModule, TropicalLinearModule


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


class TropNet(nn.Module):
    """Alternating soft dilation/erosion stack, square interior, linear-ish readout."""
    def __init__(self, d, depth, n_classes, tau=1.0):
        super().__init__()
        layers = []
        for l in range(depth):
            out = n_classes if l == depth - 1 else d
            if l % 2 == 0:
                layers.append(TropicalDilationModule(d, out, use_softmax=True, tau=tau))
            else:
                layers.append(TropicalLinearModule(d, out, use_softmin=True, tau=tau))
        self.layers = nn.ModuleList(layers)

    def set_tau(self, tau):
        for m in self.layers:
            m.set_tau(tau)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, d, depth, n_classes):
        super().__init__()
        layers, dim = [], d
        for l in range(depth):
            out = n_classes if l == depth - 1 else d
            layers.append(nn.Linear(dim, out))
            if l < depth - 1:
                layers.append(nn.ReLU())
            dim = out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, Xtr, Ytr, Xte, Yte, epochs, anneal=None, lr=1e-2, bs=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for ep in range(epochs):
        if anneal is not None:
            model.set_tau(anneal(ep, epochs))
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (model(Xte).argmax(-1) == Yte).float().mean().item()


def anneal_fn(ep, epochs):
    # tau: 1.0 -> ~0 linearly; last fifth fully hard
    frac = ep / max(epochs - 1, 1)
    return max(0.0, 1.0 - 1.25 * frac)


if __name__ == "__main__":
    down, n_train, n_test, epochs = 12, 8000, 2000, 12
    d = down * down
    Xtr, Ytr, Xte, Yte = get_mnist(down, n_train, n_test)
    print(f"== MNIST {down}x{down} (d={d}) — representational CEILING (backprop+Adam) ==")
    print(f"{'depth':>6} {'trop(soft->hard)':>17} {'trop(hard)':>11} {'dense':>7} {'sec':>6}")
    for depth in (2, 4, 8):
        t0 = time.time()
        torch.manual_seed(0)
        a_soft = train(TropNet(d, depth, 10, tau=1.0), Xtr, Ytr, Xte, Yte, epochs, anneal=anneal_fn)
        torch.manual_seed(0)
        a_hard = train(TropNet(d, depth, 10, tau=0.0), Xtr, Ytr, Xte, Yte, epochs, anneal=None)
        torch.manual_seed(0)
        a_dense = train(DenseNet(d, depth, 10), Xtr, Ytr, Xte, Yte, epochs, anneal=None)
        print(f"{depth:>6} {a_soft:>17.3f} {a_hard:>11.3f} {a_dense:>7.3f} {time.time()-t0:>6.1f}")
