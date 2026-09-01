"""induction_v2_curve.py — find the regime where STANDARD attention forms the
induction head on the variable-offset block-copy task, then check whether the
candidate tropical variant (l1_hard) tracks it.

Reuses the v2 task + model. Trains dot_soft (the ceiling/validity check) and
l1_hard (the candidate) with PERIODIC held-out eval, so we see the induction
phase-transition curve rather than a single under-trained snapshot.

Usage: PYTHONPATH=. python scripts/induction_v2_curve.py
"""
import argparse
import time

import torch
import torch.nn.functional as F

from tropical_induction_v2 import TinyLM, block_copy_batch, selectivity  # noqa: E402


def train_curve(cfg, args, device, evals):
    import math as _m
    seed = args.seed
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    model = TinyLM(args.vocab, args.d, args.layers, args.heads, cfg, args.L, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    total = max(evals)

    def lr_at(step):                       # linear warmup -> cosine decay to 10%
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + _m.cos(_m.pi * prog)))

    egen_seed = seed + 9999
    t0 = time.time()
    curve = []
    step = 0
    next_evals = sorted(evals)
    while step < total:
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits[mask], y[mask])   # only predictable (induction) positions
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if next_evals and step >= next_evals[0]:
            next_evals.pop(0)
            model.eval()
            egen = torch.Generator().manual_seed(egen_seed)
            accs = []
            with torch.no_grad():
                for _ in range(args.eval_batches):
                    _, xb, yb, mb, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
                    pred = model(xb).argmax(-1)
                    accs.append(((pred == yb) & mb).sum().item() / mb.sum().item())
            acc = sum(accs) / len(accs)
            sel = selectivity(model, args, seed, device) if cfg["mixer"] == "attn" else (0, 0, 0)
            curve.append((step, acc, sel[0], sel[1]))
            print(f"  [{cfg.get('label','?'):10s}] step {step:5d}  acc {acc:.3f}  "
                  f"sel_exact {sel[0]:.3f} sel_near {sel[1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=8000)
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"CURVE: L={args.L} vocab={args.vocab} m={args.m} d={args.d} layers={args.layers} "
          f"heads={args.heads} lr={args.lr} dev={device} chance={1/args.vocab:.3f}", flush=True)
    evals = [500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    evals = [e for e in evals if e <= args.max_steps]
    for label, cfg in [("dot_soft", {"mixer": "attn", "score": "dot", "agg": "soft", "tau": 0.25, "label": "dot_soft"}),
                       ("l1_hard", {"mixer": "attn", "score": "l1", "agg": "hard", "label": "l1_hard"})]:
        print(f"\n=== {label} ===", flush=True)
        train_curve(cfg, args, device, evals)


if __name__ == "__main__":
    main()
