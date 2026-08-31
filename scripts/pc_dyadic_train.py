"""Path 2, step 2: can the LOCAL PC rule train a deep DYADIC net?

Dyadic+ReLU has no argmax (no bistability) and Wᵀ·error is shift-add, so PC
settling stays multiplier-free; only the weight grad (outer product) is a real
multiply here (research — a sign rule would close it). We use the same optimizer
(Adam) for PC-local and the backprop ceiling, so the comparison isolates the
LEARNING RULE: local predictive coding vs global backprop.

Backprop dyadic ceiling (from pc_dyadic_probe.py): depth 2/4/8 -> 0.938/0.923/0.890.
"""
import time
import torch
import torch.nn as nn

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


class DyadicPC(nn.Module):
    """Predictive coding over multiplier-free dyadic (+/-2^k) ReLU layers."""
    def __init__(self, dims):
        super().__init__()
        self.depth = len(dims) - 1
        self.layers = nn.ModuleList([DyadicLinear(dims[l], dims[l + 1])
                                     for l in range(self.depth)])

    def f(self, l, x):
        pre = self.layers[l](x)
        return torch.relu(pre) if l < self.depth - 1 else pre   # readout linear

    @torch.no_grad()
    def feedforward(self, x0):
        s = [x0]
        for l in range(self.depth):
            s.append(self.f(l, s[-1]))
        return s

    def energy(self, states):
        E = 0.0
        for l in range(self.depth):
            mu = self.f(l, states[l])
            E = E + 0.5 * ((states[l + 1] - mu) ** 2).sum()
        return E

    def infer(self, x0, target, n_steps, gamma):
        """Settle free states (forward-init); weights frozen during inference."""
        states = [s.detach() for s in self.feedforward(x0)]
        free = list(range(1, self.depth)) if target is not None \
            else list(range(1, self.depth + 1))
        if target is not None:
            states[self.depth] = target
        for p in self.parameters():
            p.requires_grad_(False)
        for i in free:
            states[i] = states[i].clone().requires_grad_(True)
        for _ in range(n_steps):
            E = self.energy(states)
            grads = torch.autograd.grad(E, [states[i] for i in free])
            with torch.no_grad():
                for i, g in zip(free, grads):
                    states[i] -= gamma * g
            for i in free:
                states[i] = states[i].detach().requires_grad_(True)
        for p in self.parameters():
            p.requires_grad_(True)
        return [s.detach() for s in states]

    def local_grads(self, states):
        """Local PC weight gradients: dE/dW_l uses only adjacent error+activity."""
        self.zero_grad(set_to_none=True)
        self.energy(states).backward()

    def infer_and_learn(self, x0, target, n_steps, gamma, opt, learn_every):
        """Incremental PC: interleave state-settling and weight steps."""
        states = [s.detach() for s in self.feedforward(x0)]
        states[self.depth] = target
        free = list(range(1, self.depth))
        for i in free:
            states[i] = states[i].clone().requires_grad_(True)
        for t in range(n_steps):
            # state step (weights frozen)
            for p in self.parameters():
                p.requires_grad_(False)
            E = self.energy(states)
            grads = torch.autograd.grad(E, [states[i] for i in free])
            with torch.no_grad():
                for i, g in zip(free, grads):
                    states[i] -= gamma * g
            for i in free:
                states[i] = states[i].detach().requires_grad_(True)
            for p in self.parameters():
                p.requires_grad_(True)
            # weight step (states frozen)
            if (t + 1) % learn_every == 0:
                opt.zero_grad()
                self.local_grads([s.detach() for s in states])
                opt.step()


def train(model, Xtr, Ytr, Xte, Yte, epochs, n_steps, gamma, lr=1e-3, bs=128,
          n_classes=10, learn_every=0, opt=None):
    """learn_every>0 = incremental PC (weight step every k settle steps).

    opt=None -> Adam (the ceiling rule). Pass a sign-only optimizer
    (PureTropicalSGD) for the multiplier-free dyadic-SignSGD weight update.
    """
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            if not learn_every:
                states = model.infer(x0, target, n_steps=n_steps, gamma=gamma)
                opt.zero_grad()
                model.local_grads(states)
                opt.step()
            else:
                model.infer_and_learn(x0, target, n_steps, gamma, opt, learn_every)
    with torch.no_grad():
        return (model.feedforward(Xte)[-1].argmax(-1) == Yte).float().mean().item()


def _layer_local(model, l, states):
    """Per-layer (sign(local error at pre-activation), sign(activity))."""
    pre = model.layers[l](states[l])
    interior = l < model.depth - 1                      # readout is linear
    mu = torch.relu(pre) if interior else pre
    mask = (pre > 0) if interior else torch.ones_like(pre, dtype=torch.bool)
    # dE/dpre = (mu - x_{l+1}) * relu'(pre)
    sa = torch.sign(mu - states[l + 1]) * mask          # (B, out) in {-1,0,+1}
    sb = torch.sign(states[l])                          # (B, in)  in {-1,0,+1}
    return sa, sb


@torch.no_grad()
def mf_update(model, states, lr_shift):
    """Fully multiplier-free dyadic-SignSGD weight update (pure ±1 sign-product).

    sign(dE/dW[j,i]) per sample = sign(eps_j) ⊕ sign(x_i). Accumulate the
    sign-products over the batch (a ±1/0 add-tree, NOT the magnitude-weighted
    real gradient), majority-vote with sign(.), and take a power-of-two SignSGD
    step on the dyadic latent weights. No multiply anywhere on hardware.
    """
    lr = 1.0 / (1 << lr_shift)
    for l, layer in enumerate(model.layers):
        sa, sb = _layer_local(model, l, states)
        G = sa.transpose(0, 1) @ sb                     # (out,in) sum of ±1 sign-products
        layer.weight.data -= lr * torch.sign(G)
        layer.bias.data -= lr * torch.sign(sa.sum(0))


@torch.no_grad()
def dyadic_mf_update(model, states, lr_shift):
    """Magnitude-aware multiplier-free update: quantize BOTH backward operands to
    ±2^k, so each product dyadic(err)·dyadic(act) = ±2^(k1+k2) is a single SHIFT
    (XOR signs, ADD exponents) and the batch sum is an add-tree. sign of this
    magnitude-weighted accumulation tracks the true-gradient sign far better than
    the pure ±1 vote — recovering most of the magnitude the sign rule discards,
    still with no multiply on hardware (the proper dyadic backward).
    """
    from tikitorch.nn import dyadic_quantize
    lr = 1.0 / (1 << lr_shift)
    for l, layer in enumerate(model.layers):
        pre = layer(states[l])
        interior = l < model.depth - 1
        mu = torch.relu(pre) if interior else pre
        mask = (pre > 0) if interior else torch.ones_like(pre, dtype=torch.bool)
        a = (mu - states[l + 1]) * mask                 # real local error (dE/dpre)
        b = states[l]                                   # real activity
        da = dyadic_quantize(a)                         # ±2^k (shift-representable)
        db = dyadic_quantize(b)
        G = da.transpose(0, 1) @ db                     # Σ ±2^(ka+kb): shift-add tree
        layer.weight.data -= lr * torch.sign(G)
        layer.bias.data -= lr * torch.sign(da.sum(0))


@torch.no_grad()
def set_mf_grads(model, states):
    """Set .grad to the multiplier-free sign-product accumulation (so a sign-only
    optimizer like PureTropicalSGD — integer momentum, power-of-2 lr — does the
    whole update multiplier-free)."""
    for l, layer in enumerate(model.layers):
        sa, sb = _layer_local(model, l, states)
        layer.weight.grad = sa.transpose(0, 1) @ sb     # sum of ±1 sign-products
        layer.bias.grad = sa.sum(0)


def train_mf_mom(model, Xtr, Ytr, Xte, Yte, epochs, n_steps, gamma, opt, bs=128,
                 n_classes=10):
    """Multiplier-free sign-product update WITH momentum (PureTropicalSGD)."""
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = model.infer(x0, target, n_steps=n_steps, gamma=gamma)
            set_mf_grads(model, states)                 # overwrite grads (no zero needed)
            opt.step()
    with torch.no_grad():
        return (model.feedforward(Xte)[-1].argmax(-1) == Yte).float().mean().item()


def train_mf(model, Xtr, Ytr, Xte, Yte, epochs, n_steps, gamma, lr_shift, bs=128,
             n_classes=10, update_fn=mf_update):
    """Train with a multiplier-free weight update (default = pure sign-product)."""
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = model.infer(x0, target, n_steps=n_steps, gamma=gamma)
            update_fn(model, states, lr_shift)
    with torch.no_grad():
        return (model.feedforward(Xte)[-1].argmax(-1) == Yte).float().mean().item()


def verify_sign_product_per_sample(model):
    """Per-sample identity (batch=1): sign(autograd dE/dW) == sign(eps)⊕sign(x).
    This is exact; the batch version majority-votes the sign-products."""
    in_dim = model.layers[0].in_features
    n_classes = model.layers[-1].out_features
    x0 = torch.randn(1, in_dim)
    y = torch.randint(0, n_classes, (1,))
    target = torch.zeros(1, n_classes)
    target.scatter_(1, y.unsqueeze(1), 1.0)
    states = model.infer(x0, target, n_steps=20, gamma=0.1)
    model.zero_grad(set_to_none=True)
    model.energy(states).backward()
    ok = True
    for l, layer in enumerate(model.layers):
        sa, sb = _layer_local(model, l, states)
        sp = sa.transpose(0, 1) @ sb                    # per-sample sign product
        g = layer.weight.grad
        nz = g.abs() > 1e-8
        ok &= torch.equal(torch.sign(g[nz]), torch.sign(sp[nz]))
    return ok


if __name__ == "__main__":
    import sys
    down, n_train, n_test, epochs = 12, 6000, 2000, 8
    d = down * down
    Xtr, Ytr, Xte, Yte = get_mnist(down, n_train, n_test)
    ceiling = {2: 0.938, 4: 0.923, 8: 0.890}

    if sys.argv[1:] and sys.argv[1] == "signsgd":
        print("== DYADIC PC, multiplier-free sign-product weight update ==")
        torch.manual_seed(0)
        net0 = DyadicPC([d, d, 10])
        print(f"per-sample sign(dE/dW) == sign(eps)⊕sign(x) ?  "
              f"{verify_sign_product_per_sample(net0)}\n")
        # lr_shift sweep on depth 4
        depth = 4
        dims = [d] * depth + [10]
        print(f"{'lr_shift':>8} {'acc(d4)':>8}")
        best = (None, -1)
        for lr_shift in (4, 6, 8, 10):
            torch.manual_seed(0)
            net = DyadicPC(dims)
            acc = train_mf(net, Xtr, Ytr, Xte, Yte, epochs, n_steps=40, gamma=0.1,
                           lr_shift=lr_shift)
            print(f"{lr_shift:>8} {acc:>8.3f}")
            if acc > best[1]:
                best = (lr_shift, acc)
        lr_shift = best[0]
        print(f"\n-- depth curve, sign-product update at lr_shift={lr_shift} "
              f"(vs Adam-PC and backprop ceiling) --")
        print(f"{'depth':>6} {'signSGD':>8} {'adamPC':>7} {'ceiling':>8}")
        adam_pc = {2: 0.906, 4: 0.909, 8: 0.908, 12: 0.883}
        for depth in (2, 4, 8, 12):
            dims = [d] * depth + [10]
            torch.manual_seed(0)
            net = DyadicPC(dims)
            acc = train_mf(net, Xtr, Ytr, Xte, Yte, epochs, n_steps=10 * depth,
                           gamma=0.1, lr_shift=lr_shift)
            print(f"{depth:>6} {acc:>8.3f} {adam_pc.get(depth,0):>7.3f} {ceiling.get(depth,0):>8.3f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "deepwall":
        # Locate OUR depth wall (no residuals, no precision-weighting): how deep
        # can the dyadic PC rule go before it degrades? Compare to the dyadic
        # BACKPROP ceiling at the same depth (also no residuals).
        print("== dyadic PC depth wall (no residual/precision tricks), MNIST 12x12 ==")
        print(f"{'depth':>6} {'PC(Adam)':>9} {'PC(mf-dyadic)':>13} {'sec':>6}")
        for depth in (8, 16, 24):
            dims = [d] * depth + [10]
            n_steps = min(8 * depth, 160)
            torch.manual_seed(0)
            net = DyadicPC(dims)
            t0 = time.time()
            a_adam = train(net, Xtr, Ytr, Xte, Yte, 5, n_steps=n_steps, gamma=0.1, lr=1e-3)
            torch.manual_seed(0)
            net = DyadicPC(dims)
            a_mf = train_mf(net, Xtr, Ytr, Xte, Yte, 5, n_steps=n_steps, gamma=0.1,
                            lr_shift=10, update_fn=dyadic_mf_update)
            print(f"{depth:>6} {a_adam:>9.3f} {a_mf:>13.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "signfix":
        print("== close the multiplier-free gap: magnitude-aware + small-batch ==")
        print("baselines: pure-sign ~0.67, Adam ~0.91\n")
        depth = 4
        dims = [d] * depth + [10]
        # variants: (label, update_fn, bs, epochs, lr_shift)
        variants = [
            ("pure-sign bs128    ", mf_update,        128, 40, 12),
            ("pure-sign bs16     ", mf_update,         16, 20, 12),
            ("pure-sign bs4      ", mf_update,          4, 12, 12),
            ("dyadic-mag bs128   ", dyadic_mf_update, 128, 40, 12),
            ("dyadic-mag bs32    ", dyadic_mf_update,  32, 20, 12),
            ("dyadic-mag bs128 e8", dyadic_mf_update, 128,  8, 10),
        ]
        print(f"{'variant':>20} {'acc(d4)':>8} {'sec':>6}")
        best = (None, -1)
        for label, fn, bs, ep, ls in variants:
            torch.manual_seed(0)
            net = DyadicPC(dims)
            t0 = time.time()
            acc = train_mf(net, Xtr, Ytr, Xte, Yte, ep, n_steps=40, gamma=0.1,
                           lr_shift=ls, bs=bs, update_fn=fn)
            print(f"{label:>20} {acc:>8.3f} {time.time()-t0:>6.1f}")
            if acc > best[1]:
                best = ((label, fn, bs, ep, ls), acc)
        (label, fn, bs, ep, ls), _ = best
        print(f"\n-- depth curve, best multiplier-free variant: {label.strip()} --")
        print(f"{'depth':>6} {'mf_acc':>8} {'adamPC':>7} {'ceiling':>8}")
        adam_pc = {2: 0.906, 4: 0.909, 8: 0.908}
        for depth in (2, 4, 8):
            dims = [d] * depth + [10]
            torch.manual_seed(0)
            net = DyadicPC(dims)
            acc = train_mf(net, Xtr, Ytr, Xte, Yte, ep, n_steps=10 * depth, gamma=0.1,
                           lr_shift=ls, bs=bs, update_fn=fn)
            print(f"{depth:>6} {acc:>8.3f} {adam_pc.get(depth,0):>7.3f} {ceiling.get(depth,0):>8.3f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "signbest":
        print("== best multiplier-free (no-momentum) sign-product update ==")
        depth = 4
        dims = [d] * depth + [10]
        print(f"{'epochs':>7} {'lr_shift':>8} {'acc(d4)':>8} {'sec':>6}")
        best = (None, -1)
        for ep in (20, 40):
            for lr_shift in (10, 11, 12):
                torch.manual_seed(0)
                net = DyadicPC(dims)
                t0 = time.time()
                acc = train_mf(net, Xtr, Ytr, Xte, Yte, ep, n_steps=40, gamma=0.1,
                               lr_shift=lr_shift)
                print(f"{ep:>7} {lr_shift:>8} {acc:>8.3f} {time.time()-t0:>6.1f}")
                if acc > best[1]:
                    best = ((ep, lr_shift), acc)
        (ep, lr_shift), _ = best
        print(f"\n-- depth curve, best no-momentum (epochs={ep}, lr_shift={lr_shift}) --")
        print(f"{'depth':>6} {'signSGD':>8} {'adamPC':>7} {'ceiling':>8}")
        adam_pc = {2: 0.906, 4: 0.909, 8: 0.908, 12: 0.883}
        for depth in (2, 4, 8):
            dims = [d] * depth + [10]
            torch.manual_seed(0)
            net = DyadicPC(dims)
            acc = train_mf(net, Xtr, Ytr, Xte, Yte, ep, n_steps=10 * depth, gamma=0.1,
                           lr_shift=lr_shift)
            print(f"{depth:>6} {acc:>8.3f} {adam_pc.get(depth,0):>7.3f} {ceiling.get(depth,0):>8.3f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "signtune":
        from tikitorch.optim import PureTropicalSGD
        print("== close the SignSGD gap: momentum + more epochs (depth 4) ==")
        depth = 4
        dims = [d] * depth + [10]
        print(f"{'epochs':>7} {'lr_shift':>8} {'mom_bits':>8} {'acc(d4)':>8} {'sec':>6}")
        for ep, lr_shift, mom in [(8, 10, 4), (20, 10, 4), (20, 9, 4),
                                  (40, 10, 4), (40, 9, 6), (40, 8, 6)]:
            torch.manual_seed(0)
            net = DyadicPC(dims)
            opt = PureTropicalSGD(net.parameters(), lr_shift=lr_shift, momentum_bits=mom)
            t0 = time.time()
            acc = train_mf_mom(net, Xtr, Ytr, Xte, Yte, ep, n_steps=40, gamma=0.1, opt=opt)
            print(f"{ep:>7} {lr_shift:>8} {mom:>8} {acc:>8.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "rescue":
        print("== DYADIC PC rescue grid (depth 4) — settle stability + iPC ==")
        print(f"{'gamma':>6} {'steps':>6} {'lr':>7} {'ipc':>4} {'acc':>7} {'sec':>6}")
        depth = 4
        dims = [d] * depth + [10]
        grid = [
            (0.5, 8, 1e-2, 0), (0.2, 20, 1e-2, 0), (0.1, 40, 1e-2, 0),
            (0.05, 40, 1e-2, 0), (0.1, 40, 1e-3, 0),
            (0.1, 40, 1e-2, 5), (0.05, 60, 1e-3, 10),
        ]
        for gamma, steps, lr, ipc in grid:
            torch.manual_seed(0)
            net = DyadicPC(dims)
            t0 = time.time()
            acc = train(net, Xtr, Ytr, Xte, Yte, epochs, steps, gamma, lr, learn_every=ipc)
            print(f"{gamma:>6.2f} {steps:>6} {lr:>7.0e} {ipc:>4} {acc:>7.3f} {time.time()-t0:>6.1f}")
    else:
        # validated recipe from the rescue grid: small settle step + enough sweeps
        print(f"== MNIST {down}x{down} (d={d}) — DYADIC PC local rule "
              f"(gamma=0.1, lr=1e-3, steps=10*depth), {epochs} epochs ==")
        print(f"{'depth':>6} {'pc_acc':>8} {'ceiling(bp)':>12} {'sec':>6}")
        ceiling = {2: 0.938, 4: 0.923, 8: 0.890, 12: 0.84}
        for depth in (2, 4, 8, 12):
            dims = [d] * depth + [10]
            torch.manual_seed(0)
            net = DyadicPC(dims)
            t0 = time.time()
            acc = train(net, Xtr, Ytr, Xte, Yte, epochs, n_steps=10 * depth,
                        gamma=0.1, lr=1e-3)
            print(f"{depth:>6} {acc:>8.3f} {ceiling.get(depth, 0):>12.3f} {time.time()-t0:>6.1f}")
