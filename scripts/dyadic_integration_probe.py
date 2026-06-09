"""dyadic_integration_probe — verify the INTEGRATED dyadic core closes the §2a gap.

Uses the REAL library classes (tikitorch.planning.muzero.networks.TropicalRepresentation /
TropicalPrediction with proj_kind, backed by tikitorch.nn.DyadicTropicalProjection) — not a bespoke probe
net — to confirm the wiring reproduces the standalone dyadic_probe result: maxplus core underfits
oblique/smooth value; dyadic core matches the param-matched dense bar.

    python scripts/dyadic_integration_probe.py --mode smoke | --mode full
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

try:
    import scripts.capacity_probe as cp
except ModuleNotFoundError:
    import capacity_probe as cp
from tikitorch.planning.muzero.networks import TropicalRepresentation, TropicalPrediction

PARAM_TOL = 0.02


class RealTropNet(nn.Module):
    """Initial-path net built from the REAL library classes via proj_kind ('maxplus' | 'dyadic')."""

    def __init__(self, proj_kind, obs, L, value_support=None):
        super().__init__()
        self.kind = "tropical" if proj_kind == "maxplus" else "dyadic"   # 'tropical' => dead-unit probe in train_cell
        self.repr = TropicalRepresentation(obs, L, tau=1.0, proj_kind=proj_kind)
        self.pred = TropicalPrediction(L, cp.N_ACTIONS, tau=1.0, value_support=value_support, proj_kind=proj_kind)

    def set_tau(self, t):
        self.repr.set_tau(t); self.pred.set_tau(t)

    def forward(self, obs):
        return self.pred(self.repr(obs))


def grid_for(mode):
    if mode == "smoke":
        return dict(targets=["random_relu_teacher"], kinds=["maxplus", "dyadic"], seeds=[0],
                    L=64, max_steps=400)
    return dict(targets=["random_relu_teacher", "smooth_curvature_bowl", "policy_greedy_multigrid"],
                kinds=["maxplus", "dyadic"], seeds=list(range(5)), L=64, max_steps=4000)


def run(mode):
    os.makedirs(cp.RESULTS, exist_ok=True)
    g = grid_for(mode)
    out = os.path.join(cp.RESULTS, f"dyadic_integ_runs_{mode}.jsonl")
    L = g["L"]; n = 0
    with open(out, "w") as fh:
        for target in g["targets"]:
            for seed in g["seeds"]:
                data = cp.build_target(target, seed)
                obs, probe = data["obs_dim"], data["probe"]
                key = "test_policy_top1" if probe == "policy" else "test_hard"
                for kind in g["kinds"]:
                    net = RealTropNet(kind, obs, L, value_support=None)
                    cp.configure_trainable(net, "R1", probe)
                    trop_n = cp.n_trainable(net)
                    H, dp, pct = cp.match_dense_hidden(obs, L, None, "R1", probe, trop_n)
                    assert pct <= PARAM_TOL, f"param-match {pct:.3%} {kind} {target}"
                    m = cp.train_cell(net, data, "S2_direct_hard", 3e-3, g["max_steps"], seed)
                    dn = cp.InitNet("dense", obs, L, 0.0, None, dense_hidden=H)
                    cp.configure_trainable(dn, "R1", probe)
                    md = cp.train_cell(dn, data, "S2_direct_hard", 3e-3, g["max_steps"], seed)
                    gap = (md[key] - m[key]) if probe == "policy" else (m[key] - md[key])
                    row = dict(target=target, kind=kind, seed=seed, probe=probe, params=trop_n,
                               metric=m[key], dense=md[key], gap_vs_dense=gap, train=m["train_hard"])
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    n += 1
                    print(f"[{n}] {target[:20]:20s} {kind:8s} s{seed} -> {key}={m[key]:.4f} "
                          f"dense={md[key]:.4f} gap={gap:+.4f} train={m['train_hard']:.4f}")
    summarize(out, mode)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode):
    rows = [json.loads(l) for l in open(out)]
    by = defaultdict(list)
    for r in rows:
        by[(r["target"], r["kind"])].append(r)
    lines = ["# Dyadic integration probe — maxplus vs dyadic core (REAL classes), gap vs param-matched dense",
             "", "_>0 ⇒ worse than dense. maxplus = current TropicalProjection cores; dyadic = "
             "DyadicTropicalProjection cores (multiplier-free oblique). Expect maxplus≫0, dyadic≈0._", "",
             "| target | core | n | metric (med) | dense (med) | gap vs dense (mean±std) | train (med) |",
             "|---|---|---|---|---|---|---|"]
    for (tgt, kind) in sorted(by):
        rs = by[(tgt, kind)]
        mv = np.array([r["metric"] for r in rs]); dv = np.array([r["dense"] for r in rs])
        gap = np.array([r["gap_vs_dense"] for r in rs])
        lines.append(f"| {tgt[:20]} | {kind} | {len(rs)} | {np.median(mv):.4f} | {np.median(dv):.4f} | "
                     f"{gap.mean():+.4f}±{gap.std():.4f} | {np.median([r['train'] for r in rs]):.4f} |")
    path = os.path.join(cp.RESULTS, f"DYADIC_INTEG_FINDINGS_{mode}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    run(ap.parse_args().mode)
