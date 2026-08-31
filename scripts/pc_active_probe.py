"""Does activating the interior (smaller identity gap) lift deep tropical PC
above the frozen-identity ceiling? Sweep identity_gap at depth 8 on MNIST.

active% = mean over interior layers of the fraction of outputs whose argmax is
OFF the diagonal on a test batch — i.e. the layer is doing real routing, not
sitting frozen as the identity permutation. If a moderate gap raises active% AND
accuracy over gap=30, the inert-interior diagnosis is confirmed and addressed.
"""
import time
import torch
from tikitorch.nn import TropicalPCNet


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


@torch.no_grad()
def active_fraction(net, x):
    """Mean fraction of off-diagonal argmax winners over square interior layers."""
    states, _ = net.feedforward(x)
    fracs = []
    for l, layer in enumerate(net.layers):
        if layer.in_features != layer.out_features:
            continue
        _, idx = layer.predict(states[l])               # (B, out)
        diag = torch.arange(layer.out_features, device=idx.device)
        fracs.append((idx != diag).float().mean().item())
    return sum(fracs) / max(len(fracs), 1)


def train_eval(Xtr, Ytr, Xte, Yte, d, depth, *, gap, noise, epochs, n_steps,
               lr_shift=7, n_classes=10):
    dims = [d] * depth + [n_classes]
    torch.manual_seed(0)
    net = TropicalPCNet(dims, gamma_shift=2, lr_shift=lr_shift, init_identity=True,
                        identity_gap=gap, identity_noise=noise)
    for ep in range(epochs):
        for i in range(0, Xtr.shape[0] - 64, 64):
            net.train_step(Xtr[i:i+64], Ytr[i:i+64], n_steps=n_steps, init="forward")
    acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
    act = active_fraction(net, Xte[:256])
    return acc, act


def main():
    down, n_train, n_test, epochs, n_steps = 12, 6000, 2000, 4, 12
    d = down * down
    Xtr, Ytr, Xte, Yte = get_mnist(down, n_train, n_test)
    print(f"== MNIST {down}x{down} (d={d}) depth=8, identity-init gap sweep "
          f"(noise=0.1, {epochs} ep, {n_steps} steps) ==")
    print(f"{'gap':>6} {'test_acc':>9} {'active%':>8} {'sec':>6}")
    for gap in (30.0, 8.0, 4.0, 2.0, 1.0, 0.5):
        t0 = time.time()
        acc, act = train_eval(Xtr, Ytr, Xte, Yte, d, 8, gap=gap, noise=0.1,
                              epochs=epochs, n_steps=n_steps)
        print(f"{gap:>6.1f} {acc:>9.3f} {100*act:>7.1f} {time.time()-t0:>6.1f}")

    # does depth help at the best-so-far gap? compare shallow vs deep
    print("\n== depth at gap=2.0, noise=0.1 ==")
    print(f"{'depth':>6} {'test_acc':>9} {'active%':>8}")
    for depth in (2, 4, 8, 16):
        acc, act = train_eval(Xtr, Ytr, Xte, Yte, d, depth, gap=2.0, noise=0.1,
                              epochs=epochs, n_steps=n_steps)
        print(f"{depth:>6} {acc:>9.3f} {100*act:>7.1f}")


if __name__ == "__main__":
    main()
