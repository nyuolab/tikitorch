"""Recipe-boundary ablation for a small hard-routed training experiment.

This reproduces, on a tiny self-contained task, the central idea behind the
"algebraic closure of the learning loop" result: each step you take *toward* a
an increasingly constrained recipe can cost accuracy. This script measures the
local effect of three choices; it does not certify a compiled learning loop.

Three knobs, flipped one at a time from the open (GPU-friendly) corner toward
the closed (silicon-friendly) corner:

  1. optimizer:   AdamW  ->  PureTropicalSGD
        AdamW uses multiplications (moment estimates, adaptive lr). PureTropicalSGD
        uses power-of-two step choices and integer-counter momentum.
  2. projection:  dense nn.Linear head/proj  ->  tropical projection
        Replaces multiply-accumulate matmuls with max-plus / min-plus selection.
  3. activation:  soft (tau>0, logsumexp)  ->  hard (tau=0, argmin/argmax)
        The soft relaxation has dense gradients (and uses exp); the hard form is
        the route-discrete form intended for separate closure/export auditing.

We train a small ``MorphSSM`` on a synthetic sequence-classification task (no
downloads) and report the final loss/accuracy at each configuration so the
penalty is visible as a monotone-ish degradation as closure increases.

Usage::

    python examples/closure_ablation.py
    python examples/closure_ablation.py --steps 400 --d-model 96
"""
import argparse

import torch

import tikitorch as tt
import tikitorch.optim as optim


def make_task(n: int, seq_len: int, n_classes: int, seed: int = 0):
    """Synthetic sequence classification.

    Each class has a distinct additive shift pattern over the sequence, so a
    morphological (max-plus / min-plus) model can separate them. Returns
    (x: (n, seq_len, 1), y: (n,)).
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (n,), generator=g)
    # Per-class signature: a smooth bump at a class-specific location.
    t = torch.linspace(0, 1, seq_len)
    centers = torch.linspace(0.15, 0.85, n_classes)
    sig = torch.stack([torch.exp(-((t - c) ** 2) / 0.01) for c in centers])  # (C, L)
    x = sig[y] + 0.2 * torch.randn(n, seq_len, generator=g)                   # (n, L)
    return x.unsqueeze(-1), y


def build_model(d_model, n_layers, tau, tropical_proj):
    return tt.models.MorphSSM(
        d_model=d_model,
        n_layers=n_layers,
        task_type="classification",
        input_dim=1,
        num_classes=4,
        tau=tau,
        use_soft=(tau > 0),
        tropical_proj=tropical_proj,
    )


def build_optimizer(model, kind):
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-3)
    elif kind == "puretropical":
        return optim.PureTropicalSGD(model.parameters(), lr_shift=7, momentum_bits=4)
    raise ValueError(kind)


def anneal_tau(model, tau):
    """Set the tropical temperature everywhere it lives in the model.

    Different submodules expose tau differently: tropical projection layers have
    a ``set_tau`` method, while the morphological conv carries a plain ``tau``
    attribute and a ``use_soft`` flag. Walk the whole module tree and update both
    so the soft->hard (Maslov-dequantization) curriculum is genuinely applied.
    """
    for m in model.modules():
        if hasattr(m, "set_tau") and callable(m.set_tau) and m is not model:
            m.set_tau(tau)
        elif hasattr(m, "tau"):
            m.tau = tau
        # Flip soft conv kernels to hard mode at the end of the anneal.
        if hasattr(m, "use_soft"):
            m.use_soft = tau > 0


def train_eval(model, opt, x, y, steps, batch_size, anneal_to_hard):
    n = x.size(0)
    n_train = int(0.8 * n)
    xtr, ytr = x[:n_train], y[:n_train]
    xte, yte = x[n_train:], y[n_train:]

    for step in range(steps):
        model.train()
        idx = torch.randint(0, n_train, (batch_size,))
        opt.zero_grad()
        loss = model.compute_loss(xtr[idx], ytr[idx])
        loss.backward()
        opt.step()
        # Optional Maslov-dequantization curriculum: anneal tau -> 0 (soft->hard).
        if anneal_to_hard:
            frac = step / max(steps - 1, 1)
            tau = max(1.0 * (1.0 - frac), 0.0)
            anneal_tau(model, tau)

    model.eval()
    with torch.no_grad():
        logits = model(xte)
        test_loss = torch.nn.functional.cross_entropy(logits, yte).item()
        acc = (logits.argmax(-1) == yte).float().mean().item()
    return test_loss, acc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=64)
    args = p.parse_args()

    x, y = make_task(args.n_samples, args.seq_len, n_classes=4)

    # Ablation ladder: ordinary corner -> increasingly constrained candidates.
    configs = [
        # name,                      optimizer,        tropical_proj, tau, anneal
        ("open  (AdamW, dense, soft)",      "adamw",        False, 1.0, False),
        ("+ tropical SGD",                   "puretropical", False, 1.0, False),
        ("+ tropical projection",            "puretropical", True,  1.0, False),
        ("+ hard ops (soft->hard anneal)",   "puretropical", True,  1.0, True),
        ("reference rerun (AdamW, dense, soft)", "adamw",    False, 1.0, False),
    ]

    print(f"{'configuration':36s}  {'test_loss':>10s}  {'test_acc':>9s}")
    print("-" * 60)
    results = []
    for name, opt_kind, trop_proj, tau, anneal in configs:
        torch.manual_seed(0)
        model = build_model(args.d_model, args.n_layers, tau, trop_proj)
        opt = build_optimizer(model, opt_kind)
        loss, acc = train_eval(model, opt, x, y, args.steps, args.batch_size, anneal)
        results.append((name, loss, acc))
        print(f"{name:36s}  {loss:10.4f}  {acc:9.4f}")

    open_acc = results[0][2]
    closed_acc = results[3][2]
    print("-" * 60)
    print(f"boundary-choice gap (reference - constrained accuracy): "
          f"{open_acc - closed_acc:+.4f}")
    print("Run the full-step auditor and physical build gates separately before "
          "making a multiplier-free or FPGA claim.")


if __name__ == "__main__":
    main()
