"""Top-k tropical softening: an honest ablation of the hard value pull.

The hard (tau=0) tropical value pull routes each output (i, d) through a SINGLE
key (its argmax), giving a one-index-per-output gradient. ``value_topk_k`` softens
that: each output averages its k best keys, so the gradient flows to k keys --
denser, and still strictly add/compare-only (no logsumexp, no multiply, unlike a
tau>0 soft relaxation).

This is presented as an ABLATION, not a guaranteed win. Densifying the gradient
is necessary but not sufficient: averaging the top-k also changes the forward
function (toward a local mean), which can hurt as easily as help. The script
reports BOTH the gradient densification (the mechanism, measured directly on the
value-pull op) and the eval accuracy per k (the outcome), and prints the sign of
the effect honestly -- a flat or negative accuracy curve is a real finding.

Scope (per the closure-decomposition review): this softens VALUE-PULL only. It
does not touch the score argmax, the FFN argmax, or the batch-shift, which are
also part of the hard-tropical cost bundle, so it cannot recover that whole
bundle and its isolated contribution must be read off this sweep, not assumed.

Usage::

    python examples/topk_softening.py
    python examples/topk_softening.py --k 1,2,3,4 --steps 1000
"""
import argparse
from pathlib import Path

import torch

import tikitorch as tt
import tikitorch.functional as F
import tikitorch.optim as optim
from tikitorch.nn import TropicalAttention

V = 128


def load_chars(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return torch.tensor([min(ord(c), V - 1) for c in text], dtype=torch.long)


def sample_batch(ids, batch, seq_len, gen):
    n = ids.numel()
    starts = torch.randint(0, n - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([ids[s:s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def set_value_topk(model, k):
    """Set the value-pull top-k on every tropical attention block."""
    n = 0
    for m in model.modules():
        if isinstance(m, TropicalAttention):
            m.value_topk_k = k
            n += 1
    return n


def measure_density(k, seed=0):
    """Gradient density of the value-pull op alone (the softening mechanism)."""
    g = torch.Generator().manual_seed(seed)
    Vv = torch.randn(2, 4, 32, 32, generator=g, requires_grad=True)  # (B,H,Lv,D)
    S = torch.randn(2, 4, 32, 32, generator=g, requires_grad=True)   # (B,H,Lq,Lv)
    out = F.tropical_value_pull_topk(Vv, S, k)
    out.sum().backward()
    return (S.grad != 0).float().mean().item()


@torch.no_grad()
def evaluate(model, ids, args, gen):
    model.eval()
    accs = []
    for _ in range(args.eval_batches):
        x, y = sample_batch(ids, args.batch_size, args.seq_len, gen)
        logits = model(x).reshape(-1, V)
        accs.append((logits.argmax(-1) == y.reshape(-1)).float().mean().item())
    return sum(accs) / len(accs)


def train_one(k, ids, args):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = tt.models.TropicalTransformer(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, tau=0.0, max_seq_len=args.seq_len,   # hard: top-k engages
        task_type="lm", vocab_size=V,
    )
    model.loss_type = args.loss
    set_value_topk(model, k)
    opt = optim.SignSGD(model.parameters(), lr=args.lr, momentum=0.9)

    correct = total = 0
    for _ in range(args.steps):
        model.train()
        x, y = sample_batch(ids, args.batch_size, args.seq_len, gen)
        opt.zero_grad()
        loss = model.compute_loss(x, y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = model(x).reshape(-1, V).argmax(-1)
            correct += (pred == y.reshape(-1)).sum().item()
            total += y.numel()
    ev = evaluate(model, ids, args, torch.Generator().manual_seed(args.seed + 1))
    return correct / total, ev


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=str,
                   default=str(Path(__file__).resolve().parents[2] / "data" / "shakespeare_input.txt"))
    p.add_argument("--k", type=str, default="1,2,3,4")
    p.add_argument("--loss", type=str, default="tropical_ce")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=8)
    args = p.parse_args()

    ks = [int(s) for s in args.k.split(",")]
    ids = load_chars(args.text)
    print(f"Shakespeare char-LM  |  D={args.d_model} V={V} tau=0 (HARD)  "
          f"loss={args.loss}  value-pull top-k sweep")
    print(f"steps={args.steps}  batch={args.batch_size}  seq_len={args.seq_len}\n")
    print(f"{'k':>3s}  {'grad_density':>12s}  {'train_acc':>9s}  {'eval_acc':>8s}")
    print("-" * 42)
    evals = {}
    for k in ks:
        dens = measure_density(k)
        tr, ev = train_one(k, ids, args)
        evals[k] = ev
        print(f"{k:3d}  {dens:12.3f}  {tr:9.4f}  {ev:8.4f}")
    print("-" * 42)
    base = evals[ks[0]]
    best_k = max(evals, key=evals.get)
    print("gradient density rises with k (the softening mechanism, multiplier-free).")
    print(f"best eval_acc at k={best_k}: {evals[best_k]:.4f}  "
          f"(vs k={ks[0]}: {base:.4f}, delta {evals[best_k]-base:+.4f})")
    if evals[best_k] <= base + 1e-4:
        print("HONEST FINDING: top-k value-pull softening did NOT help here "
              "(denser gradient != better accuracy). Reported as a null/negative ablation.")


if __name__ == "__main__":
    main()
