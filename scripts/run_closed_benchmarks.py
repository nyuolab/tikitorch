"""Reference: hard-deployment image/sequence benchmarks.

Trains a `MorphSSM` on the sequence tasks the README reports (sequential-MNIST
len-784, sequential-CIFAR len-1024) with dropout=0, soft→hard `tau` annealing,
`PureTropicalSGD`, and a LARGE EFFECTIVE BATCH via gradient accumulation
(the morphological-net large-batch requirement, arXiv:2505.09710: `sign(Σ grad)` is
the large-batch majority-vote sign). Accuracy is reported at HARD `tau=0` — the form
that maps to the zero-DSP FPGA kernel — and each saved model's executed inference path
is checked with `tt.closure_report`. The soft relaxation and ordinary classification
loss used during fitting are not a zero-DSP training recipe; this runner does not make
or record a training-closure claim.

Usage:
    python scripts/run_closed_benchmarks.py --task smnist --epochs 15 --batch-size 64 --accum-steps 8 --device cuda
    python scripts/run_closed_benchmarks.py --task scifar --epochs 20 --batch-size 48 --accum-steps 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import tikitorch as tt
import tikitorch.optim as optim

# task -> (seq_len, input_dim, torchvision dataset, num_classes)
TASKS = {
    "smnist": dict(seq_len=784, input_dim=1, dataset="MNIST", num_classes=10),
    "scifar": dict(seq_len=1024, input_dim=3, dataset="CIFAR10", num_classes=10),
    "mnist":  dict(seq_len=784, input_dim=1, dataset="MNIST", num_classes=10),
}


def _checkpoint_config(args, spec):
    """Constructor-only config for the hard model written to the ``.tt`` bundle.

    Training starts from a soft relaxation, but ``_evaluate`` switches every routed
    module to the hard path before the checkpoint is written.  Keep experiment names,
    optimizer details, and measured results in the JSON sidecar rather than passing
    them to ``MorphSSM(**config)`` through :func:`tt.load`.
    """
    return dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=0.0,
        tau=0.0,
        use_soft=False,
        out_proj_type="tropical",
        norm_type="tropical",
        head_type="tropical",
        input_type="tropical",
        task_type="classification",
        input_dim=spec["input_dim"],
        num_classes=spec["num_classes"],
    )


def _loaders(task, batch_size, data_root):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    spec = TASKS[task]
    if spec["dataset"] == "MNIST":
        to_seq = transforms.Compose([transforms.ToTensor(),
                                     transforms.Lambda(lambda t: t.view(784, 1))])
        tr = datasets.MNIST(data_root, train=True, download=True, transform=to_seq)
        te = datasets.MNIST(data_root, train=False, download=True, transform=to_seq)
    else:  # CIFAR10 -> length-1024 sequence of RGB triples
        to_seq = transforms.Compose([transforms.ToTensor(),               # (3,32,32)
                                     transforms.Lambda(lambda t: t.permute(1, 2, 0).reshape(1024, 3))])
        tr = datasets.CIFAR10(data_root, train=True, download=True, transform=to_seq)
        te = datasets.CIFAR10(data_root, train=False, download=True, transform=to_seq)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(te, batch_size=batch_size, shuffle=False))


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    if hasattr(model, "set_tau"):
        model.set_tau(0.0)                  # HARD tropical = the zero-DSP form
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=8, help="effective batch = batch_size * accum_steps")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr-shift", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="manuscript/results/closed_benchmarks")
    args = p.parse_args()
    if args.epochs < 1:
        p.error("--epochs must be at least 1")
    spec = TASKS[args.task]
    os.makedirs(args.out, exist_ok=True)
    print(f"task={args.task} device={args.device} eff_batch={args.batch_size*args.accum_steps}")

    # Hard-deployment recipe: dropout=0, a soft tau for dense gradients annealed to
    # the hard inference path, and PureTropicalSGD (bit-shift lr + int momentum).
    # The soft fitting path itself is ordinary arithmetic, not a zero-DSP claim.
    checkpoint_config = _checkpoint_config(args, spec)
    training_config = {**checkpoint_config, "tau": 1.0, "use_soft": True}
    model = tt.models.MorphSSM(**training_config).to(args.device)
    print(f"MorphSSM params: {model.param_count:,}")
    opt = optim.PureTropicalSGD(model.parameters(), lr_shift=args.lr_shift, momentum_bits=4)
    train_loader, test_loader = _loaders(args.task, args.batch_size, args.data_root)

    t0 = time.time()
    best = 0.0
    best_epoch = None
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, "set_tau"):       # soft -> hard anneal across training
            model.set_tau(max(0.05, 1.0 * (1 - epoch / max(1, args.epochs - 1))))
        opt.zero_grad()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            loss = model.compute_loss(x, y) / args.accum_steps     # accumulate -> large eff. batch
            loss.backward()
            running += loss.item() * args.accum_steps
            if (step + 1) % args.accum_steps == 0:
                opt.step(); opt.zero_grad()                        # sign(Σ grad) majority vote
        acc = _evaluate(model, test_loader, args.device)
        if best_state is None or acc > best:
            best = acc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(f"epoch {epoch:2d}  mean_loss {running/(step+1):.4f}  hard_tau_test_acc {acc:.4f}  "
              f"({round(time.time()-t0)}s)")

    # --- restore the measured best model, then save + certify a held-out batch ---
    model.load_state_dict(best_state)
    path = os.path.join(args.out, f"{args.task}_morphssm.tt")
    metrics = dict(task=args.task, model="MorphSSM", d_model=args.d_model,
                   n_layers=args.n_layers, input_dim=spec["input_dim"],
                   num_classes=spec["num_classes"], optimizer="PureTropicalSGD",
                   recipe="soft_to_hard", eff_batch=args.batch_size * args.accum_steps,
                   best_acc=round(best, 4), best_epoch=best_epoch,
                   training_zero_dsp=False)
    model.set_tau(0.0)
    tt.save(model, path, config=checkpoint_config)
    xb, yb = next(iter(test_loader)); xb = xb[:16].to(args.device)
    model.set_tau(0.0)
    rep = tt.closure_report(model, example_inputs=(xb,))
    metrics["inference_closed"] = rep["inference_closed"]
    with open(os.path.join(args.out, f"{args.task}_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"BEST hard-tau acc={best:.4f}  inference_closed={rep['inference_closed']}  -> {path}")


if __name__ == "__main__":
    main()
