"""Loss-conditioning ablation for a tropical-transformer host experiment.

The compared boundaries are a 2-hot perceptron gradient, host-conditioned
softmax, an experimental lookup/shift approximation, and ordinary floating
cross-entropy.

This script measures how loss conditioning changes a small host run. Everything
is held fixed -- the same 1-layer tropical transformer
at the silicon dims (D=128, V=128, D_FF=512), the same sign-based optimizer, the
same seed, the same Shakespeare windows -- and only the loss conditioning varies:

    perceptron        2-hot sign-valued gradient boundary
    tropical_ce       softmax - onehot, host-conditioned exact softmax
    tropical_ce_lut   lookup/shift softmax approximation
    cross_entropy     F.cross_entropy float reference ceiling (uses multiplies)

These are pedagogical host rows, not silicon measurements or confirmatory
accuracy estimates.

Usage::

    python examples/closure_ce.py
    python examples/closure_ce.py --steps 1200 --seq-len 96 --tau 0.5
"""
import argparse
from pathlib import Path

import torch

import tikitorch as tt
import tikitorch.optim as optim

V = 128  # ASCII vocab, matching the silicon kernel


def load_chars(path: str):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    ids = torch.tensor([min(ord(c), V - 1) for c in text], dtype=torch.long)
    return ids


def sample_batch(ids, batch, seq_len, gen):
    n = ids.numel()
    starts = torch.randint(0, n - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([ids[s:s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def build_model(d_model, n_layers, n_heads, d_ff, tau, seq_len):
    return tt.models.TropicalTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        tau=tau, max_seq_len=seq_len, task_type="lm", vocab_size=V,
    )


@torch.no_grad()
def evaluate(model, ids, batch, seq_len, gen):
    model.eval()
    x, y = sample_batch(ids, batch, seq_len, gen)
    logits = model(x).reshape(-1, V)
    acc = (logits.argmax(-1) == y.reshape(-1)).float().mean().item()
    return acc


def train_one(loss_type, ids, args, eval_gen_state):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = build_model(args.d_model, args.n_layers, args.n_heads,
                        args.d_ff, args.tau, args.seq_len)
    model.loss_type = loss_type                      # <-- the only thing that varies
    opt = optim.SignSGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    run_correct = run_total = 0
    for step in range(args.steps):
        model.train()
        x, y = sample_batch(ids, args.batch_size, args.seq_len, gen)
        opt.zero_grad()
        loss = model.compute_loss(x, y)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = model(x).reshape(-1, V).argmax(-1)
            run_correct += (pred == y.reshape(-1)).sum().item()
            run_total += y.numel()

    # held-out accuracy on a fixed eval stream (same windows for every loss_type)
    eval_gen = torch.Generator().manual_seed(eval_gen_state)
    accs = [evaluate(model, ids, args.batch_size, args.seq_len, eval_gen)
            for _ in range(args.eval_batches)]
    return run_correct / run_total, sum(accs) / len(accs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=str,
                   default=str(Path(__file__).resolve().parents[2] / "data" / "shakespeare_input.txt"))
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.5,
                   help="tropical temperature; 0.0 = hard routing, >0 = soft relaxation")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=8)
    args = p.parse_args()

    ids = load_chars(args.text)
    chance = 1.0 / V
    # Majority-char baseline for context (predict the single most frequent char).
    maj = torch.bincount(ids, minlength=V).max().item() / ids.numel()

    print(f"Shakespeare char-LM  |  D={args.d_model} V={V} D_FF={args.d_ff} "
          f"L={args.seq_len} heads={args.n_heads} layers={args.n_layers} tau={args.tau}")
    print(f"optimizer=SignSGD(lr={args.lr}, momentum={args.momentum})  "
          f"steps={args.steps}  batch={args.batch_size}")
    print(f"chance={chance:.4f}   majority-char baseline={maj:.4f}\n")

    rows = [
        ("perceptron      (2-hot sign boundary)",                "perceptron"),
        ("tropical_ce     (host softmax)",                        "tropical_ce"),
        ("tropical_ce_lut (lookup/shift approximation)",         "tropical_ce_lut"),
        ("cross_entropy   (float reference ceiling)",            "cross_entropy"),
    ]
    print(f"{'loss conditioning':54s}  {'train_acc':>9s}  {'eval_acc':>8s}")
    print("-" * 78)
    results = {}
    for label, lt in rows:
        tr, ev = train_one(lt, ids, args, eval_gen_state=args.seed + 1)
        results[lt] = ev
        print(f"{label:54s}  {tr:9.4f}  {ev:8.4f}")

    print("-" * 78)
    rec = results["tropical_ce"] - results["perceptron"]
    rec_lut = results["tropical_ce_lut"] - results["perceptron"]
    ceil_gap = results["cross_entropy"] - results["tropical_ce"]
    print(f"recovery  perceptron -> tropical_ce      : {rec:+.4f}  "
          f"({'host-conditioned comparison'})")
    print(f"recovery  perceptron -> tropical_ce_lut  : {rec_lut:+.4f}  "
          f"({'lookup/shift host comparison'})")
    print(f"remaining gap to float CE ceiling        : {ceil_gap:+.4f}  "
          f"(not a silicon number)")


if __name__ == "__main__":
    main()
