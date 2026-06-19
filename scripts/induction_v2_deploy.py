"""induction_v2_deploy.py — the decisive tropical-attention test on the VALID
(variable-offset, loss-masked) induction task.

Established facts (scripts/induction_v2_curve.py):
  - dot_soft (standard softmax attention) solves the variable-offset content-
    routing task: acc 1.000, sel_near 0.54 (a real induction head). Task is valid.
  - l1_hard trained FROM SCRATCH stalls ~0.15: the hard max-plus value pull has a
    sparse gradient (only the argmax key gets signal) and cannot efficiently learn
    routing from scratch.

The fix is the tropical north-star: TRAIN with a soft (dense-gradient) relaxation,
ANNEAL tau -> small so the model learns to operate near-hard, then DEPLOY with the
closure-clean hard max-plus pull (tau=0, max_j(S+V), zero multiply). This script
tests whether train-soft/deploy-hard recovers content routing for tropical scores.

Arms (variable-offset task, masked loss, 3 seeds, held-out eval):
  dot     train soft (anneal) -> eval {soft, hard}      real dot score (reference)
  l1      train soft (anneal) -> eval {soft, hard}      -sum_d|Q-K| (mult-free)
  maxplus train soft (anneal) -> eval {soft, hard}      max_d(Q+K)  (mult-free)
  l1_scratch_hard  train HARD -> eval {hard}            the from-scratch baseline (stalls)
  conv    control

The decisive read: for a tropical score, is eval-HARD acc ~ eval-SOFT acc ~ the
dot ceiling? If yes, a FULLY MULTIPLIER-FREE attention does content routing at
deploy, trained via a relaxation. sel_near on the hard-eval is the mechanistic proof.

Usage: PYTHONPATH=. python scripts/induction_v2_deploy.py
"""
import argparse
import math
import time

import torch
import torch.nn.functional as F

from tropical_induction_v2 import TinyLM, block_copy_batch, Attn, selectivity


def _eval_acc_sel(model, args, seed, device, agg, tau):
    """Eval held-out acc + selectivity with a forced (agg, tau) on all Attn layers."""
    saved = []
    for blk in model.blocks:
        mix = blk[1]
        if isinstance(mix, Attn):
            saved.append((mix, mix.agg, mix.tau))
            mix.agg, mix.tau = agg, tau
    model.eval()
    egen = torch.Generator().manual_seed(seed + 9999)
    accs = []
    with torch.no_grad():
        for _ in range(args.eval_batches):
            _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
            pred = model(x).argmax(-1)
            accs.append(((pred == y) & mask).sum().item() / mask.sum().item())
    sel = selectivity(model, args, seed, device)
    for mix, a, t in saved:
        mix.agg, mix.tau = a, t
    return sum(accs) / len(accs), sel[0], sel[1]


def train(cfg, seed, args, device, anneal=None):
    """anneal=(tau0,tau1): train soft with tau annealed tau0->tau1 (geometric)."""
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    model = TinyLM(args.vocab, args.d, args.layers, args.heads, cfg, args.L, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    total = args.steps

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def tau_at(s):
        if anneal is None:
            return None
        t0, t1 = anneal
        frac = s / max(1, total - 1)
        return t0 * (t1 / t0) ** frac          # geometric anneal

    for s in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        if anneal is not None:
            tv = tau_at(s)
            for blk in model.blocks:
                if isinstance(blk[1], Attn):
                    blk[1].tau = tv
        model.train()
        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, gen, device)
        logits = model(x)
        loss = F.cross_entropy(logits[mask], y[mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


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
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEPLOY test (variable-offset, masked loss): L={args.L} vocab={args.vocab} m={args.m} "
          f"d={args.d} h={args.heads} steps={args.steps} seeds={args.seeds} dev={device} "
          f"chance={1/args.vocab:.3f}", flush=True)
    t0 = time.time()

    # (label, score, train_agg, anneal, eval_list[(name,agg,tau)])
    SMALL = 1e-6
    # eval list: soft (final tau), gather (argmax-gather = TRUE hard-attention limit,
    # mult-free), hard (max-plus matmul = different operator, mult-free)
    # STE investigation: does a HARD forward (no soft-anneal) train via straight-through?
    #   l1            = soft-anneal -> deploy gather   (the validated baseline, ~1.0)
    #   l1_ste        = HARD gather forward + soft backward (STE), trained from scratch
    #   l1_scr_gather = HARD gather forward, NO surrogate grad (from scratch)   -> stalls
    # (d) FULLY-HARD attention backward: isolate the attention backward (linears/norm/opt
    # stay Adam+nn.Linear). Forward = hard gather (deploy) for all; vary the TRAIN backward.
    # KEY test: does a MULTIPLIER-FREE uniform top-k SCATTER surrogate (iter_topk, k_bwd) train
    # the router, or does it need the magnitude-weighted softmax surrogate (a multiply)?
    # (d) PC-FAITHFUL attempt: magnitude-aware MULT-FREE attention backward via dyadic-
    # quantized operands (q(grad)·q(V) alignment). Does restoring the magnitude signal
    # (which uniform top-k scatter lacked, and which stalled) train the router mult-free?
    # (d) GRADIENT-MANTISSA sweep: does a K-bit-mantissa (finer-than-1-bit) attention
    # gradient train the router? mbits=0 = the 1-bit dyadic that stalled; mbits=3,5 are
    # "nearly mult-free" (sums of shifts). k_bwd=4 (top-k) fixed.
    # (d) DEGENERACY FIX: dyadic-RANK soft weight (2^-rank over top-k) gives competing keys
    # a non-zero gradient (the soft p that hard-attention's one-hot kills). mult-free.
    arms = [
        ("ref_ste_softmax", {"mixer": "attn", "score": "l1"}, "gather_ste", (2.0, 0.25),
         [("gather", "gather", 0.0)]),                              # anchor: full-precision softmax ~0.99
        ("dyg_uniform", {"mixer": "attn", "score": "l1", "k_bwd": 4, "grad_mbits": 3}, "gather_dyadicgrad", None,
         [("gather", "gather", 0.0)]),                              # uniform top-k (the stall ~0.06)
        ("ranksoft_k4_m3", {"mixer": "attn", "score": "l1", "k_bwd": 4, "grad_mbits": 3, "soft_rank": True},
         "gather_dyadicgrad", None, [("gather", "gather", 0.0)]),   # DEGENERACY FIX, k=4
        ("ranksoft_k8_m3", {"mixer": "attn", "score": "l1", "k_bwd": 8, "grad_mbits": 3, "soft_rank": True},
         "gather_dyadicgrad", None, [("gather", "gather", 0.0)]),   # denser
        ("ranksoft_k8_m5", {"mixer": "attn", "score": "l1", "k_bwd": 8, "grad_mbits": 5, "soft_rank": True},
         "gather_dyadicgrad", None, [("gather", "gather", 0.0)]),   # denser + finer mantissa
    ]

    results = {}   # (arm, evalname) -> [accs]
    sels = {}
    for label, cfg, train_agg, anneal, evals in arms:
        for seed in args.seeds:
            tcfg = dict(cfg)
            if cfg["mixer"] == "attn":
                tcfg["agg"] = train_agg
                # gather_ste needs a fixed surrogate tau>0 for the soft backward.
                tcfg["tau"] = (0.5 if train_agg == "gather_ste"
                               else (anneal[0] if anneal else 0.0))
            model = train(tcfg, seed, args, device, anneal=anneal)
            if cfg["mixer"] == "conv":
                model.eval()
                egen = torch.Generator().manual_seed(seed + 9999)
                accs = []
                with torch.no_grad():
                    for _ in range(args.eval_batches):
                        _, x, y, mask, _ = block_copy_batch(args.batch, args.L, args.vocab, args.m, egen, device)
                        pred = model(x).argmax(-1)
                        accs.append(((pred == y) & mask).sum().item() / mask.sum().item())
                results.setdefault((label, "-"), []).append(sum(accs) / len(accs))
                print(f"  {label:11s} seed{seed} eval -    acc {accs and sum(accs)/len(accs):.3f}  ({time.time()-t0:.0f}s)", flush=True)
                continue
            for ename, eagg, etau in evals:
                acc, se, sn = _eval_acc_sel(model, args, seed, device, eagg, etau)
                results.setdefault((label, ename), []).append(acc)
                sels.setdefault((label, ename), []).append(sn)
                print(f"  {label:11s} seed{seed} eval {ename:4s} acc {acc:.3f}  sel_near {sn:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== DEPLOY RESULT (acc mean+/-std over seeds; chance="
          f"{1/args.vocab:.3f}) ===", flush=True)
    def stat(vals):
        mn = sum(vals) / len(vals)
        sd = (sum((v - mn) ** 2 for v in vals) / len(vals)) ** 0.5
        return mn, sd
    order = sorted(results.items(), key=lambda kv: -stat(kv[1])[0])
    for (label, ename), vals in order:
        mn, sd = stat(vals)
        sn = ""
        if (label, ename) in sels:
            sm, _ = stat(sels[(label, ename)])
            sn = f"   sel_near {sm:.3f}"
        print(f"  {label:11s} eval {ename:4s}  {mn:.3f} +/- {sd:.3f}{sn}", flush=True)

    print("\nKEY: for a tropical score (l1/maxplus), is eval-HARD ~ eval-SOFT ~ dot ceiling?", flush=True)
    print("  -> YES = fully multiplier-free attention does content routing at deploy "
          "(trained via soft relaxation). l1_scratch shows from-scratch hard stalls.", flush=True)


if __name__ == "__main__":
    main()
