"""induction_distill.py — DENSE-TEACHER DISTILLATION test for on-fabric attention
training (lever d, the brainstorm's top reframe).

The wall: no hand-rolled multiplier-free approximation of the softmax VALUE-prediction
gradient trains the attention router (uniform scatter, dyadic-operand, K-bit mantissa,
dyadic-rank soft weight all stall ~0.06). Diagnosis: the hard-attention gradient is
DEGENERATE and every mult-free surrogate of it is the wrong direction.

The reframe: don't backprop a routing gradient at all. Train a full-precision SOFT
teacher (offline / host; absorbs all multiplies), freeze it, and supervise the
mult-free STUDENT's per-layer attention SCORES with a PERCEPTRON loss over key INDICES
toward the teacher's argmax. Routing becomes a CLASSIFICATION (non-degenerate by
construction); the perceptron backward is sign-valued (mult-free) -> Q/K via the l1
score's sign backward. V/O/head still learn from the LM loss through the gather's
mult-free scatter (the gather's argmax blocks score grad, so Q/K get ONLY the distill
signal). Optimizer: AdamW (validate target learnable) and SignSGD (the mult-free test).

Arms (deploy eval = hard gather, mult-free):
  teacher          soft (gather_ste, anneal) -> the routing oracle (~1.0)
  distill_adamw    student: LM + perceptron-distill, AdamW    (is the target learnable?)
  distill_signsgd  student: LM + perceptron-distill, SignSGD  (THE on-fabric test)
  novalue_signsgd  student: LM only, SignSGD, no distill      (control: optimizer alone fails)

PASS: distill_signsgd >> novalue_signsgd (~0.06) and -> the teacher ceiling => discrete
routing-target supervision makes a mult-free/sign backward train the router on-fabric.

Usage: PYTHONPATH=. python scripts/induction_distill.py
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tropical_induction_v2 import TinyLM, block_copy_batch, Attn, selectivity


def _attn_layers(model):
    return [blk[1] for blk in model.blocks if isinstance(blk[1], Attn)]


def lm_loss(model, x, y, mask, vocab):
    logits = model(x)
    return F.cross_entropy(logits[mask], y[mask])


def perceptron_distill_loss(student, teacher, x, margin=1.0):
    """Sum over attn layers of a margin/perceptron loss pushing each query's student
    score at the TEACHER's argmax key above all other (causal-valid) keys. Mult-free
    backward (sign-valued: +/-1 scatter into scores). Returns scalar."""
    with torch.no_grad():
        teacher(x, record=True)
        tgt = [a.last_s.argmax(dim=-1) for a in _attn_layers(teacher)]   # per layer (B,H,Lq)
    student(x, record=True)
    loss = 0.0
    for a, t in zip(_attn_layers(student), tgt):
        s = a.last_s_live                                   # (B,H,Lq,Lv) grad-carrying
        star = s.gather(-1, t.unsqueeze(-1))               # (B,H,Lq,1) score at teacher's key
        valid = torch.isfinite(s)                          # causal-valid keys
        # hinge: max(0, margin + s_j - s_star) for j != target, summed over valid keys
        viol = torch.clamp(margin + s - star, min=0.0)
        viol = viol.scatter(-1, t.unsqueeze(-1), 0.0)      # exclude the target key itself
        viol = torch.where(valid, viol, torch.zeros_like(viol))
        loss = loss + viol.sum(-1).mean()
    return loss


@torch.no_grad()
def eval_gather(model, args, seed, device):
    saved = [a.agg for a in _attn_layers(model)]
    for a in _attn_layers(model):
        a.agg = "gather"
    model.eval()
    egen = torch.Generator().manual_seed(seed + 9999)
    accs = []
    for _ in range(args.eval_batches):
        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
        pred = model(x).argmax(-1)
        accs.append(((pred == y) & mask).sum().item() / mask.sum().item())
    for a, g in zip(_attn_layers(model), saved):
        a.agg = g
    return sum(accs) / len(accs)


def build(agg, vocab, args, device, k_bwd=4):
    cfg = {"mixer": "attn", "score": "l1", "agg": agg, "tau": 0.5, "k_bwd": k_bwd}
    return TinyLM(vocab, args.d, args.layers, args.heads, cfg, args.L, True).to(device)


def lr_sched(step, args):
    warm = min(1.0, (step + 1) / args.warmup)
    cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
    return args.lr * warm * cos


def train_teacher(seed, args, device):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    t = build("gather_ste", args.vocab, args, device)
    opt = torch.optim.AdamW(t.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_sched(step, args)
        tau = max(2.0 * (0.25 / 2.0) ** (step / max(1, args.steps - 1)), 0.25)  # 2.0->0.25 anneal
        for a in _attn_layers(t):
            a.tau = tau
        t.train()
        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, gen, device)
        loss = lm_loss(t, x, y, mask, args.vocab)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(t.parameters(), 1.0); opt.step()
    for a in _attn_layers(t):
        a.tau = 0.25
    return t


def train_student(teacher, seed, args, device, opt_kind, distill, lam=1.0,
                  lr_decay=False, exp_clamp=None):
    torch.manual_seed(seed + 1); gen = torch.Generator().manual_seed(seed + 7)
    st = build("gather", args.vocab, args, device)             # hard gather (mult-free deploy)
    if opt_kind == "adamw":
        opt = torch.optim.AdamW(st.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    else:                                                      # mult-free: signsgd / dyadic_mag
        opt = torch.optim.SGD(st.parameters(), lr=args.sign_lr)
    for step in range(args.steps):
        if opt_kind == "adamw":
            for g in opt.param_groups:
                g["lr"] = lr_sched(step, args)
        elif opt_kind in ("signsgd", "dyadic_mag") and lr_decay:
            # cosine lr-decay: a fixed mult-free step oscillates around the optimum; decay
            # lets it settle (bakeoff bit-shift-lr-decay finding).
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
            for g in opt.param_groups:
                g["lr"] = args.sign_lr * (0.05 + 0.95 * cos)
        st.train()
        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, gen, device)
        loss = lm_loss(st, x, y, mask, args.vocab)            # -> V/O/head via gather scatter
        if distill:
            loss = loss + lam * perceptron_distill_loss(st, teacher, x)   # -> Q/K (scores)
        opt.zero_grad(); loss.backward()
        if opt_kind == "signsgd":                            # pure ±1 sign (PC: lags)
            for p in st.parameters():
                if p.grad is not None:
                    p.grad = p.grad.sign()
        elif opt_kind == "dyadic_mag":                       # MAGNITUDE-AWARE mult-free update:
            # step ∝ dyadic_quantize(grad) = sign·2^round(log2|grad|); with a pow2 lr the
            # whole update is a bit-shift (no multiply) but keeps gradient MAGNITUDE (PC: the
            # working mult-free optimizer; pure sign lagged ~0.24). exp_clamp=(lo,hi) bounds
            # the per-element step exponent (caps huge steps from large grads, floors tiny).
            from tikitorch.nn import dyadic_quantize
            for p in st.parameters():
                if p.grad is not None:
                    g = dyadic_quantize(p.grad)
                    if exp_clamp is not None:
                        lo, hi = exp_clamp
                        g = g.sign() * g.abs().clamp(2.0 ** lo, 2.0 ** hi)
                    p.grad = g
        if opt_kind != "dyadic_mag":
            torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
        opt.step()
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=32); p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--m", type=int, default=5); p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sign-lr", type=float, default=2e-3); p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--steps", type=int, default=2500); p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"DISTILL test: L={args.L} vocab={args.vocab} m={args.m} d={args.d} steps={args.steps} "
          f"seeds={args.seeds} dev={device} chance={1/args.vocab:.3f}", flush=True)
    res = {}
    t0 = time.time()
    for seed in args.seeds:
        teacher = train_teacher(seed, args, device)
        tacc = eval_gather(teacher, args, seed, device)
        res.setdefault("teacher", []).append(tacc)
        print(f"  teacher          seed{seed} gather {tacc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        # (name, opt, distill, lr_decay, exp_clamp) -- TUNING the dyadic-magnitude on-fabric
        # optimizer toward the teacher ceiling (base reached 0.553).
        CLAMP = (-12, -1)
        for name, opt_kind, distill, dec, clamp in [
                ("distill_adamw", "adamw", True, False, None),                 # anchor (~1.0 ceiling)
                ("dyadicmag_base", "dyadic_mag", True, False, None),           # base (~0.553)
                ("dyadicmag_decay", "dyadic_mag", True, True, None),           # + lr-decay
                ("dyadicmag_clamp", "dyadic_mag", True, False, CLAMP),         # + exponent clamp
                ("dyadicmag_decay_clamp", "dyadic_mag", True, True, CLAMP)]:   # + both
            st = train_student(teacher, seed, args, device, opt_kind, distill,
                               lr_decay=dec, exp_clamp=clamp)
            acc = eval_gather(st, args, seed, device)
            res.setdefault(name, []).append(acc)
            print(f"  {name:18s} seed{seed} gather {acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== DISTILL RESULT (deploy-gather acc, mean +/- std) ===", flush=True)
    for name in ["teacher", "distill_adamw", "dyadicmag_base", "dyadicmag_decay",
                 "dyadicmag_clamp", "dyadicmag_decay_clamp"]:
        v = res.get(name, [])
        if v:
            mn = sum(v) / len(v); sd = (sum((x - mn) ** 2 for x in v) / len(v)) ** 0.5
            print(f"  {name:16s} {mn:.3f} +/- {sd:.3f}   {['%.3f'%x for x in v]}", flush=True)
    print("\nPASS: distill_signsgd >> novalue_signsgd (~0.06) and -> teacher => discrete "
          "routing-target supervision trains the router with a mult-free/sign backward.", flush=True)


if __name__ == "__main__":
    main()
