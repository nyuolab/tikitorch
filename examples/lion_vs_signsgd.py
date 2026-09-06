"""Lion (integer momentum) vs SignSGD: recovering optimizer accuracy, multiplier-free.

SignSGD (``W -= lr * sign(grad)``) is the simplest multiplier-free optimizer, but
replacing AdamW with it is the second-largest cost in the closure waterfall
(~-12.8pp on Shakespeare char-LM). ``TropicalLion`` takes the sign of a real
*momentum* (an EMA of the gradient) instead of the sign of the raw gradient --
strictly stronger than SignSGD, and still multiplier-free (power-of-2 betas make
the EMA a shift+add; the update is a sign).

The honest catch, which this demo makes visible: on chip the momentum EMA is an
INTEGER buffer with a truncating arithmetic right shift, so it collapses to zero
on small gradients (degenerating back to SignSGD) unless the gradient is
pre-scaled into a wide integer domain. ``TropicalLion`` models that truncation
exactly, so the CPU number is a faithful silicon predictor -- not the optimistic
float-EMA number. We print ``momentum_density`` so you can see the momentum is
actually alive.

Three optimizers, everything else fixed (same tropical transformer at the silicon
dims, same seed, same character windows, same loss conditioning):

    signsgd     W -= lr * sign(grad)                 multiplier-free baseline
    lion_int    sign of integer-fixed-point momentum multiplier-free candidate
    adamw       float reference ceiling              uses multiplies (NOT silicon)

Usage::

    python examples/lion_vs_signsgd.py --smoke
    python examples/lion_vs_signsgd.py --loss perceptron --steps 1200
    python examples/lion_vs_signsgd.py --text path/to/corpus.txt
"""
import argparse

import torch

import tikitorch as tt
import tikitorch.optim as optim
from _char_data import load_ascii_ids

V = 128


def sample_batch(ids, batch, seq_len, gen):
    n = ids.numel()
    starts = torch.randint(0, n - seq_len - 1, (batch,), generator=gen)
    x = torch.stack([ids[s:s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def build_model(args):
    return tt.models.TropicalTransformer(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, tau=args.tau, max_seq_len=args.seq_len,
        task_type="lm", vocab_size=V,
    )


def build_opt(kind, params, args):
    if kind == "signsgd":
        return optim.SignSGD(params, lr=args.lr, momentum=0.0)
    if kind == "lion_int":
        return optim.TropicalLion(params, lr_shift=args.lr_shift,
                                  beta1_bits=1, beta2_bits=3, grad_bits=args.grad_bits)
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=args.lr)
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, ids, args, gen):
    model.eval()
    accs = []
    for _ in range(args.eval_batches):
        x, y = sample_batch(ids, args.batch_size, args.seq_len, gen)
        logits = model(x).reshape(-1, V)
        accs.append((logits.argmax(-1) == y.reshape(-1)).float().mean().item())
    return sum(accs) / len(accs)


def train_one(kind, ids, args):
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    model = build_model(args)
    model.loss_type = args.loss
    opt = build_opt(kind, model.parameters(), args)

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

    eval_gen = torch.Generator().manual_seed(args.seed + 1)
    ev = evaluate(model, ids, args, eval_gen)
    mdens = opt.momentum_density() if hasattr(opt, "momentum_density") else float("nan")
    return correct / total, ev, mdens


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=str, default=None,
                   help="UTF-8 corpus path (default: built-in deterministic text)")
    p.add_argument("--smoke", action="store_true",
                   help="Use a tiny configuration that checks the complete script quickly")
    p.add_argument("--loss", type=str, default="tropical_ce",
                   choices=["tropical_ce", "tropical_ce_lut", "perceptron", "cross_entropy"],
                   help="loss conditioning held fixed across optimizers")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2e-3, help="lr for signsgd/adamw")
    p.add_argument("--lr-shift", type=int, default=9, help="Lion lr = 2^-lr_shift")
    p.add_argument("--grad-bits", type=int, default=16, help="Lion gradient pre-scale")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=8)
    args = p.parse_args()

    if args.smoke:
        args.steps = 1
        args.batch_size = 2
        args.seq_len = 8
        args.d_model = 16
        args.d_ff = 32
        args.n_layers = 1
        args.n_heads = 2
        args.eval_batches = 1

    ids, source = load_ascii_ids(args.text, vocab_size=V, min_length=args.seq_len + 1)
    if args.smoke:
        print("SMOKE MODE: checks execution only; the one-step scores are not results.\n")
    print(f"Character LM ({source})  |  D={args.d_model} V={V} heads={args.n_heads} "
          f"layers={args.n_layers} tau={args.tau}  loss={args.loss}")
    print(f"steps={args.steps}  batch={args.batch_size}  seq_len={args.seq_len}\n")

    print(f"{'optimizer':40s}  {'train_acc':>9s}  {'eval_acc':>8s}  {'mom_density':>11s}")
    print("-" * 76)
    results = {}
    for kind, label in [
        ("signsgd",  "signsgd   (sign(grad), mult-free baseline)"),
        ("lion_int", "lion_int  (sign(int momentum), mult-free)"),
        ("adamw",    "adamw     (float ceiling, NOT silicon)"),
    ]:
        tr, ev, md = train_one(kind, ids, args)
        results[kind] = ev
        md_s = "    n/a" if md != md else f"{md:11.3f}"
        print(f"{label:40s}  {tr:9.4f}  {ev:8.4f}  {md_s}")

    print("-" * 76)
    print(f"recovery  signsgd -> lion_int : {results['lion_int'] - results['signsgd']:+.4f}  "
          f"(multiplier-free)")
    print(f"remaining gap to AdamW ceiling: {results['adamw'] - results['lion_int']:+.4f}  "
          f"(AdamW's adaptive 2nd moment is the multiply/divide part, not recovered)")
    print("\nmom_density > 0 confirms the integer momentum is alive (not SignSGD-degenerate).")


if __name__ == "__main__":
    main()
