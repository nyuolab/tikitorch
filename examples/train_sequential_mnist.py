"""Sequential MNIST with a tropical MorphSSM, trained the PyTorch way.

This is the canonical idiomatic-usage demo for TikiTorch. It reads each 28x28
MNIST image as a length-784 sequence of scalar pixels (``input_dim=1``) and
classifies it with a ``MorphSSM`` (alternating max-plus/min-plus morphological
convolutions). The optimizer is ``PureTropicalSGD``, whose default update is
shift- and counter-oriented. This example deliberately trains with a soft
relaxation and the model's default classification loss, so it is not a
multiplier-free full-step recipe. Use the closure audit with a hardened model,
declared loss, reverse rule, and optimizer before making that claim.

The training loop below is deliberately a plain, vanilla PyTorch loop — the
point is that swapping in tropical layers/optimizers requires *no* change to how
you write training code.

Usage::

    python examples/train_sequential_mnist.py --smoke          # tiny synthetic run
    python examples/train_sequential_mnist.py                  # real MNIST, 1 epoch
    python examples/train_sequential_mnist.py --epochs 5 --batch-size 128

torchvision is only needed for the real dataset; pass ``--smoke`` to run on
synthetic data with no downloads (handy for CI).
"""
import argparse

import torch

import tikitorch as tt
import tikitorch.optim as optim


def get_dataloaders(batch_size: int, data_root: str):
    """Real sequential-MNIST loaders via torchvision (lazy import)."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit(
            'Real MNIST requires torchvision. Install `pip install -e ".[vision]"` '
            "or pass --smoke for synthetic data."
        ) from exc
    from torch.utils.data import DataLoader

    # Flatten 1x28x28 -> (784, 1): a length-784 sequence of scalar pixels.
    to_seq = transforms.Compose([
        transforms.ToTensor(),                       # (1, 28, 28) in [0, 1]
        transforms.Lambda(lambda t: t.view(784, 1)),  # (seq_len=784, feat=1)
    ])
    train = datasets.MNIST(data_root, train=True, download=True, transform=to_seq)
    test = datasets.MNIST(data_root, train=False, download=True, transform=to_seq)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def smoke_batches(batch_size: int, n_batches: int = 2):
    """Synthetic (B, 784, 1) batches so the script runs with no downloads."""
    for _ in range(n_batches):
        x = torch.rand(batch_size, 784, 1)
        y = torch.randint(0, 10, (batch_size,))
        yield x, y


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr-shift", type=int, default=8,
                   help="PureTropicalSGD learning rate = 2^-lr_shift (bit shift).")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--smoke", action="store_true",
                   help="Run 1 epoch on tiny synthetic data (no torchvision or downloads).")
    p.add_argument("--max-steps", type=int, default=0,
                   help="If >0, stop each epoch after this many steps (quick demo).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # --- idiomatic: build a tropical model exactly like a torch.nn model ---
    model = tt.models.MorphSSM(
        d_model=args.d_model,
        n_layers=args.n_layers,
        task_type="classification",
        input_dim=1,        # one scalar pixel per timestep
        num_classes=10,
        tau=1.0,            # soft (trainable) relaxation; anneal -> 0 to go hard
        use_soft=True,
    ).to(device)
    print(f"MorphSSM params: {model.param_count:,}")

    # --- idiomatic: a shift-oriented optimizer, plugged in like any optim ---
    opt = optim.PureTropicalSGD(model.parameters(), lr_shift=args.lr_shift,
                                momentum_bits=4)

    if args.smoke:
        train_iter_factory = lambda: smoke_batches(args.batch_size, n_batches=2)
        test_loader = None
    else:
        train_loader, test_loader = get_dataloaders(args.batch_size, args.data_root)
        train_iter_factory = lambda: train_loader

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_iter_factory()):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = model.compute_loss(x, y)   # tropical CE; same call as torch
            loss.backward()                   # sparse argmin/argmax gradients
            opt.step()                        # bit-shift weight update
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}  loss {loss.item():.4f}")
            if args.max_steps and step + 1 >= args.max_steps:
                break
        n = step + 1
        print(f"epoch {epoch} done  mean_loss {running / n:.4f}")
        if test_loader is not None:
            acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch}  test_acc {acc:.4f}")

    print("done.")


if __name__ == "__main__":
    main()
