"""tropical_fix_probe — can PURE-TROPICAL (zero-DSP) levers close the §2a expressivity gap?

Constraint: multiply-free / zero-DSP only. The capacity + component probes localized the underfit to the
max-plus cores' inability to make OBLIQUE slopes (it only makes axis-aligned/lattice ones), in repr
(policy) and pred (value); dynamics is fine; and it is EXPRESSIVITY not trainability (train already ~0).

The only pure-tropical lever that can add oblique reach is a FIXED ±1 BASIS ROTATION before each tropical
projection: y = max_i(W[k,i] + (Rx)_i) makes each region's gradient a ROW of R — i.e. an oblique ±1
normal — using ADDS/SIGN-FLIPS only (a Hadamard butterfly; zero multiply, zero DSP, zero params).
Supporting pure-tropical knobs: TropicalScale (x·2^round(s) = bit-shift) to emphasize directions, and a
batchshift norm (subtract-mean = pure subtraction, removing the per-sample DIVIDE in the current
_minmax_unit). NOTE: torch's F.linear(x, R) numerically multiplies by ±1; that is only the SIMULATION —
on fabric a ±1 / signed-permutation map is adds+wiring (the closure claim the audit must confirm).

Question: do these close the held-out TEST gap to the param-matched DENSE bar?
  * relu   (arbitrary oblique) — the hard test; a fixed ±1 frame can only coarsely cover arbitrary normals
  * bowl   (all-ones normal)   — Hadamard-favorable; the easy win if any
  * policy (repr bottleneck)   — does rotating the encoder basis help the policy

If nothing closes relu, pure tropical is a poor substrate for arbitrary-oblique value -> prototype elsewhere.

    python scripts/tropical_fix_probe.py --mode smoke | --mode full
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scripts.capacity_probe as cp
except ModuleNotFoundError:
    import capacity_probe as cp
import tikitorch.nn as ttnn
from tikitorch.planning.muzero.networks import ScalarHead, _minmax_unit

RESULTS = cp.RESULTS
PARAM_TOL = 0.02

# config -> (basis, use_scale, norm_mode)   [all tropical except 'dense']
CONFIGS = {
    "identity":          (None,          False, "current"),     # = current tropical (baseline)
    "hadamard":          ("hadamard",    False, "current"),
    "signed_perm":       ("signed_perm", False, "current"),
    "hadamard_scale":    ("hadamard",    True,  "current"),
    "hadamard_batchshift": ("hadamard",  False, "batchshift"),  # also removes the per-sample divide
    "hadamard_norm":       ("hadamard_norm", False, "current"),  # ORTHONORMAL H/sqrt(D): fair-magnitude test
}


def basis_matrix(kind, d, seed):
    if kind is None:
        return None
    n = 1 << ((d - 1).bit_length())
    if kind == "hadamard":
        return torch.from_numpy(scipy.linalg.hadamard(n)).float()[:d, :d].contiguous()
    if kind == "hadamard_norm":      # orthonormal: H/sqrt(d) -> preserves magnitude (zero-DSP scalar)
        return (torch.from_numpy(scipy.linalg.hadamard(n)).float()[:d, :d] / (d ** 0.5)).contiguous()
    if kind == "signed_perm":
        g = torch.Generator().manual_seed(seed * 131 + 7)
        perm = torch.randperm(d, generator=g)
        signs = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1
        R = torch.zeros(d, d); R[torch.arange(d), perm] = signs
        return R
    raise ValueError(kind)


def ste_pow2(x, s):
    """x * 2**round(s) with straight-through gradient — a bit-shift (multiply-free on fabric)."""
    k = s.round().detach() + s - s.detach()
    return x * torch.exp2(k)


class _RotProj(nn.Module):
    """TropicalProjection preceded by a fixed ±1 basis rotation (+ optional power-of-2 scale)."""

    def __init__(self, d_in, d_out, R, use_scale):
        super().__init__()
        self.proj = ttnn.TropicalProjection(d_in, d_out, tau=1.0)
        self.register_buffer("R", R if R is not None else torch.empty(0))
        self.has_R = R is not None
        self.scale = nn.Parameter(torch.zeros(d_in)) if use_scale else None

    @property
    def W1(self):                       # so capacity_probe's dead-unit probe still works
        return self.proj.W1

    def set_tau(self, t):
        self.proj.set_tau(t)

    def forward(self, x):
        if self.has_R:
            x = F.linear(x, self.R)     # ±1 rotation -> adds/sign-flips on fabric
        if self.scale is not None:
            x = ste_pow2(x, self.scale)
        return self.proj(x)


class RotRepr(nn.Module):
    def __init__(self, obs, L, R, use_scale, norm_mode):
        super().__init__()
        self.enc = _RotProj(obs, L, R, use_scale)
        self.norm = ttnn.TropicalLayerNorm(L) if norm_mode == "current" else None
        self.norm_mode = norm_mode

    def set_tau(self, t):
        self.enc.set_tau(t)

    def forward(self, x):
        h = self.enc(x)
        if self.norm_mode == "current":
            h = _minmax_unit(self.norm(h))
        elif self.norm_mode == "batchshift":
            h = h - h.mean(-1, keepdim=True)
        return h


class RotPred(nn.Module):
    def __init__(self, L, A, R, use_scale):
        super().__init__()
        self.mix = _RotProj(L, L, R, use_scale)
        self.policy_head = nn.Linear(L, A)
        self.value_head = ScalarHead(L)

    def set_tau(self, t):
        self.mix.set_tau(t)

    def forward(self, h):
        z = self.mix(h)
        return self.policy_head(z), self.value_head(z)


class RotInitNet(nn.Module):
    kind = "tropical"

    def __init__(self, cfg, obs, L, A, seed):
        super().__init__()
        basis, use_scale, norm_mode = CONFIGS[cfg]
        Rr = basis_matrix(basis, obs, seed)
        Rp = basis_matrix(basis, L, seed + 1)
        self.repr = RotRepr(obs, L, Rr, use_scale, norm_mode)
        self.pred = RotPred(L, A, Rp, use_scale)

    def set_tau(self, t):
        self.repr.set_tau(t); self.pred.set_tau(t)

    def forward(self, obs):
        return self.pred(self.repr(obs))


def grid_for(mode):
    if mode == "smoke":
        return dict(targets=["random_relu_teacher"], cfgs=["identity", "hadamard"], Ls=[64],
                    seeds=[0], max_steps=400)
    if mode == "norm":   # fair-magnitude re-test: identity vs orthonormal Hadamard on the value targets
        return dict(targets=["random_relu_teacher", "smooth_curvature_bowl"],
                    cfgs=["identity", "hadamard_norm"], Ls=[64], seeds=list(range(5)), max_steps=4000)
    return dict(targets=["random_relu_teacher", "smooth_curvature_bowl", "policy_greedy_multigrid"],
                cfgs=list(CONFIGS), Ls=[64], seeds=list(range(5)), max_steps=4000)


def run(mode):
    os.makedirs(RESULTS, exist_ok=True)
    g = grid_for(mode)
    out = os.path.join(RESULTS, f"fix_runs_{mode}.jsonl")
    n = 0
    with open(out, "w") as fh:
        for target in g["targets"]:
            for seed in g["seeds"]:
                data = cp.build_target(target, seed)
                obs, A, probe = data["obs_dim"], cp.N_ACTIONS, data["probe"]
                for L in g["Ls"]:
                    # param-match dense to the identity-tropical trainable count
                    base = RotInitNet("identity", obs, L, A, seed)
                    cp.configure_trainable(base, "R1", probe)
                    trop_n = cp.n_trainable(base)
                    H, dp, pct = cp.match_dense_hidden(obs, L, None, "R1", probe, trop_n)
                    assert pct <= PARAM_TOL, f"param-match {pct:.3%} at {target} L={L}"
                    # dense bar
                    dn = cp.InitNet("dense", obs, L, 0.0, None, dense_hidden=H)
                    cp.configure_trainable(dn, "R1", probe)
                    dref = cp.train_cell(dn, data, "S3_anneal", 3e-3, g["max_steps"], seed)
                    for cfg in g["cfgs"]:
                        net = RotInitNet(cfg, obs, L, A, seed)
                        cp.configure_trainable(net, "R1", probe)
                        m = cp.train_cell(net, data, "S3_anneal", 3e-3, g["max_steps"], seed)
                        key = "test_policy_top1" if probe == "policy" else "test_hard"
                        row = dict(target=target, probe=probe, cfg=cfg, latent_dim=L, seed=seed,
                                   trop_params=trop_n, dense_params=dp, pct_diff=pct,
                                   dense_ref=dref[key], **m)
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        n += 1
                        gap = (dref[key] - m[key]) if probe == "policy" else (m[key] - dref[key])
                        print(f"[{n}] {target[:20]:20s} {cfg:18s} L={L} s{seed} -> "
                              f"{key}={m[key]:.4f} dense={dref[key]:.4f} gap_vs_dense={gap:+.4f} "
                              f"dead={m.get('dead_unit_fraction'):.2f}")
    summarize(out, mode)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode):
    rows = [json.loads(l) for l in open(out)]
    lines = ["# Pure-tropical fix probe — can a zero-DSP basis rotation close the §2a gap?", "",
             "_Gap-vs-dense = how far each PURE-TROPICAL config is from the param-matched dense bar "
             "(value: test_hard − dense; policy: dense_top1 − trop_top1). >0 ⇒ still worse than dense; "
             "→0 ⇒ closed the gap. `identity` = current tropical (the starting gap)._", "",
             "| target | L | config | n | config metric (med) | dense bar | gap-vs-dense (mean±std) |",
             "|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r["target"], r["latent_dim"], r["cfg"])].append(r)
    for (tgt, L, cfg), rs in sorted(by.items()):
        pol = rs[0]["probe"] == "policy"
        key = "test_policy_top1" if pol else "test_hard"
        mv = np.array([r[key] for r in rs]); dv = np.array([r["dense_ref"] for r in rs])
        gap = (dv - mv) if pol else (mv - dv)
        lines.append(f"| {tgt[:20]} | {L} | {cfg} | {len(rs)} | {np.median(mv):.4f} | "
                     f"{np.median(dv):.4f} | {gap.mean():+.4f}±{gap.std():.4f} |")
    path = os.path.join(RESULTS, f"FIX_FINDINGS_{mode}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "norm"], default="full")
    run(ap.parse_args().mode)
