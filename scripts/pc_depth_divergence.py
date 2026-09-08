"""Exp 1 — cascade-vs-relaxation gradient divergence vs depth (PREREG.md in
manuscript/results/pc_depth_traction/). No training: pure gradient comparison on fixed batches.

Run: PYTHONPATH=. python scripts/pc_depth_divergence.py
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F

from tikitorch.nn.predictive_coding import DyadicPCNet
from pc_charlm import load, onehot

OUT = "manuscript/results/pc_depth_traction/exp1_divergence.jsonl"
DEPTHS = [2, 4, 8, 16]
HIDDEN, CTX, BATCH, NBATCH = 256, 12, 256, 3


def grab_grads(net):
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in net.parameters()]


def compare(ref, other):
    rows = []
    for i, (a, b) in enumerate(zip(ref, other)):
        af, bf = a.flatten(), b.flatten()
        cos = F.cosine_similarity(af, bf, dim=0).item() if af.norm() > 0 else float("nan")
        rel = ((af - bf).norm() / (af.norm() + 1e-12)).item()
        rows.append({"param": i, "cos": round(cos, 6), "relL2": round(rel, 6)})
    return rows


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
    (Xtr, ytr), _, V, _, _ = load(CTX, subset=200_000)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    logs = []
    for L in DEPTHS:
        dims = [CTX * V] + [HIDDEN] * (L - 1) + [V]
        torch.manual_seed(0)
        net = DyadicPCNet(dims, residual=True)
        # brief warm so grads aren't at a degenerate init point
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(0)
        for _ in range(30):
            bi = torch.randperm(Xtr.size(0), generator=gen)[:BATCH]
            net.train_step_cascade(onehot(Xtr[bi], V), ytr[bi], opt)
        print(f"===== depth {L} =====", flush=True)
        for b in range(NBATCH):
            bi = torch.randperm(Xtr.size(0), generator=gen)[:BATCH]
            xb, yb = onehot(Xtr[bi], V), ytr[bi]
            target = net._make_target(yb, ref=xb)

            def zero():
                for p in net.parameters():
                    p.grad = None

            zero(); net.cascade_grads(xb, target)
            ref = grab_grads(net)

            configs = [("sync_L", dict(n_iter=L)),
                       ("sync_3L", dict(n_iter=3 * L)),
                       ("sync_10L", dict(n_iter=10 * L)),
                       ("async_3L", dict(n_iter=3 * L, participation=0.5, stale=True,
                                         rand_order=True,
                                         gen=torch.Generator().manual_seed(42 + b)))]
            for name, kw in configs:
                zero()
                net.error_relax_grads(xb, target, **kw)
                rows = compare(ref, grab_grads(net))
                cs = [r["cos"] for r in rows if r["cos"] == r["cos"]]
                worst = min(cs) if cs else float("nan")
                mean = sum(cs) / len(cs) if cs else float("nan")
                # attribute: which third of the depth holds the worst layer?
                worst_i = min(range(len(rows)), key=lambda i: rows[i]["cos"]
                              if rows[i]["cos"] == rows[i]["cos"] else 9)
                logs.append({"depth": L, "batch": b, "config": name,
                             "mean_cos": round(mean, 5), "worst_cos": round(worst, 5),
                             "worst_param": worst_i, "n_params": len(rows),
                             "per_param": rows})
                print(f"  [{name:>9}] mean_cos={mean:.5f} worst={worst:.5f} "
                      f"@param{worst_i}/{len(rows)}", flush=True)
    with open(OUT, "w") as f:
        for r in logs:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}", flush=True)
    # summary: bounded or compounding?
    print("\ndepth  config     mean_cos(worst over batches)")
    for L in DEPTHS:
        for cfgname in ["sync_10L", "async_3L"]:
            vals = [r["worst_cos"] for r in logs
                    if r["depth"] == L and r["config"] == cfgname]
            print(f"{L:>5}  {cfgname:>9}  {min(vals):.5f}")


if __name__ == "__main__":
    main()
