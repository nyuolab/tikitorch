"""dyadic_probe — THE decisive pure-tropical test: can MULTIPLY-FREE oblique weights close the gap?

Prior probes localized the §2a underfit to the max-plus cores' inability to make OBLIQUE slopes, and the
verdict audit (results/capacity_probe/ ... pure-tropical-verdict-audit) verified in-code that the shipped
TropicalProjection has an exactly one-hot input gradient → its piece-normals are the axis vectors {e_i};
weights move breakpoints, never orientations. The audit's load-bearing correction: obliqueness in tropical
comes from MONOMIAL EXPONENTS (repeated addition), and a LEARNED BIT-SHIFT layer — weights quantized to
signed powers of two w = ±2^round(s), so w·x is a sum of shifted inputs (shift+add, ZERO multiply, ZERO
DSP) — can form an oblique normal multiply-free. That is NEITHER a fixed rotation NOR a matmul.

This experiment isolates that claim with the minimal change: take the param-matched DENSE control and
quantize every weight to the nearest signed power of two (a multiply-free shift-add layer), keeping the
ReLU (= max(·,0), a tropical op). Same architecture, same param count. Does the multiply-free DYADIC net
close the held-out gap that the full-precision dense net closes?

GATES (per the audit): (a) the dyadic net must INTERPOLATE train (~0) else the STE-through-round is a
trainability artifact, not an expressivity verdict; (b) report gap-vs-dense AND the constant-mean baseline;
(c) confirm the dyadic net's input gradient is genuinely oblique (>1 nonzero, non-unit magnitudes).

Outcome: if dyadic ≈ dense on held-out (train interpolates) → multiply-free CAN do oblique → the flagship
stays pure (dyadic-weight readout). If dyadic ≫ dense → the power-of-two grid is too coarse for the real
normal continuum → pure multiply-free is honestly out for this domain → prototype elsewhere.

    python scripts/dyadic_probe.py --mode smoke | --mode full
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scripts.capacity_probe as cp
except ModuleNotFoundError:
    import capacity_probe as cp

PARAM_TOL = 0.02


def dyadic_quant(w):
    """Nearest SIGNED power of two, straight-through. w -> sign(w)*2^round(log2|w|) (a shift on fabric)."""
    mag = w.abs().clamp_min(1e-12)
    q = torch.sign(w) * torch.exp2(torch.log2(mag).round())
    return w + (q - w).detach()


class DyadicLinear(nn.Linear):
    """nn.Linear with weights quantized to signed powers of two — multiply-free (shift+add), same params."""

    def forward(self, x):
        return F.linear(x, dyadic_quant(self.weight), self.bias)


def dyadicize(net):
    """Replace every nn.Linear in `net` with a DyadicLinear sharing the same weight/bias Parameters."""
    for m in net.modules():
        for cn, cm in list(m.named_children()):
            if type(cm) is nn.Linear:
                d = DyadicLinear(cm.in_features, cm.out_features, bias=cm.bias is not None)
                d.weight = cm.weight
                d.bias = cm.bias
                setattr(m, cn, d)
    return net


def grid_for(mode):
    if mode == "smoke":
        return dict(targets=["random_relu_teacher"], Ls=[64], seeds=[0], max_steps=400)
    return dict(targets=["random_relu_teacher", "smooth_curvature_bowl"], Ls=[64, 128],
                seeds=list(range(5)), max_steps=4000)


def oblique_grad_check(net, X):
    """Verify the net's input gradient is genuinely oblique (>1 nonzero, non-unit magnitudes)."""
    x = X[:1].clone().requires_grad_(True)
    net.zero_grad()
    _, v = net(x)
    v.sum().backward()
    g = x.grad[0]
    nz = int((g.abs() > 1e-6).sum())
    return nz, float(g.abs().max()), float(g.abs().mean())


def run(mode):
    os.makedirs(cp.RESULTS, exist_ok=True)
    g = grid_for(mode)
    out = os.path.join(cp.RESULTS, f"dyadic_runs_{mode}.jsonl")
    n = 0
    with open(out, "w") as fh:
        for target in g["targets"]:
            for seed in g["seeds"]:
                data = cp.build_target(target, seed)
                obs, probe = data["obs_dim"], data["probe"]
                # constant-mean baseline: predict train-mean on test
                cmean = float(((data["yte"] - data["ytr"].mean()) ** 2).mean())
                for L in g["Ls"]:
                    # param-match dense to the tropical baseline (so numbers compare to prior probes)
                    tbase = cp.InitNet("tropical", obs, L, 1.0, None)
                    cp.configure_trainable(tbase, "R1", probe)
                    H, dp, pct = cp.match_dense_hidden(obs, L, None, "R1", probe, cp.n_trainable(tbase))
                    assert pct <= PARAM_TOL
                    for kind in ["dense", "dyadic"]:
                        net = cp.InitNet("dense", obs, L, 0.0, None, dense_hidden=H)
                        if kind == "dyadic":
                            net = dyadicize(net)
                        cp.configure_trainable(net, "R1", probe)
                        m = cp.train_cell(net, data, "S2_direct_hard", 3e-3, g["max_steps"], seed)
                        nz, gmax, gmean = oblique_grad_check(net, data["Xte"])
                        row = dict(target=target, kind=kind, latent_dim=L, seed=seed, probe=probe,
                                   dense_hidden=H, params=dp, const_mean_mse=cmean,
                                   grad_nonzero=nz, grad_max=gmax, grad_mean=gmean, **m)
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        n += 1
                        print(f"[{n}] {target[:20]:20s} {kind:6s} L={L:3d} s{seed} -> "
                              f"train={m['train_hard']:.4f} TEST={m['test_hard']:.4f} "
                              f"(cmean={cmean:.3f}) grad_nz={nz} conv={m['converged_flag']}")
    summarize(out, mode)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode):
    rows = [json.loads(l) for l in open(out)]
    pick = {(r["target"], r["latent_dim"], r["seed"], r["kind"]): r for r in rows}
    lines = ["# Dyadic probe — can a MULTIPLY-FREE (power-of-2 weight) net make oblique normals?", "",
             "_dyadic = dense architecture with weights quantized to ±2^k (shift+add, zero-DSP). Gap = "
             "dyadic − dense held-out test (>0 ⇒ dyadic worse). const-mean = trivial baseline. Both must "
             "interpolate train (~0) for the comparison to be an expressivity verdict._", "",
             "| target | L | n | dense TEST | dyadic TEST | dyadic−dense | const-mean | dyadic train | grad_nz |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (tgt, L) in sorted({(r["target"], r["latent_dim"]) for r in rows}):
        seeds = sorted({r["seed"] for r in rows if r["target"] == tgt and r["latent_dim"] == L})
        de = [pick.get((tgt, L, s, "dense")) for s in seeds]
        dy = [pick.get((tgt, L, s, "dyadic")) for s in seeds]
        pairs = [(a, b) for a, b in zip(de, dy) if a and b]
        if not pairs:
            continue
        dd = np.array([a["test_hard"] for a, _ in pairs]); yy = np.array([b["test_hard"] for _, b in pairs])
        gap = yy - dd
        tr = np.median([b["train_hard"] for _, b in pairs]); cm = np.median([a["const_mean_mse"] for a, _ in pairs])
        gnz = int(np.median([b["grad_nonzero"] for _, b in pairs]))
        lines.append(f"| {tgt[:20]} | {L} | {len(pairs)} | {np.median(dd):.4f} | {np.median(yy):.4f} | "
                     f"{gap.mean():+.4f}±{gap.std():.4f} | {cm:.3f} | {tr:.4f} | {gnz} |")
    path = os.path.join(cp.RESULTS, f"DYADIC_FINDINGS_{mode}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    run(ap.parse_args().mode)
