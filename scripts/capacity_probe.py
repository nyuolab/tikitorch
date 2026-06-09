"""capacity_probe — is max-plus the *right prior* for value/policy, or does the
tropical core underfit? (the §2a question, measured as a generalization gap)

This is the methodologically-vetted expressivity probe for the Tropical MuZero
flagship. It fits the MuZero INITIAL-INFERENCE path  obs -> (policy_logits, value_raw)
as supervised function-fitting (no RL/MCTS), comparing TropicalMuZeroNet's max-plus
cores against a *param-matched* dense ReLU control.

WHY GENERALIZATION, NOT INTERPOLATION.  An early interpolation-capacity design (fit a
fixed point set, measure train error) was empirically rejected: in the over-parameterized
regime BOTH function classes drive train error to ~0 on every target (right- or
wrong-prior), so interpolation cannot discriminate the inductive bias. "Right prior"
*means* better generalization from limited data, so the headline is the HELD-OUT TEST
error: fit train to ~0 (both can), then measure hard-eval (tau=0) loss on unseen data.
The §2a signal is the SIGN of the paired tropical-minus-dense TEST gap across a prior
spectrum (verified at small scale: linf max-plus target -0.2, ReLU/bowl targets +0.6/+0.7).

Design choices (each fixes an empirically-verified confound):
  * PARAM MATCH, measured.  TropicalProjection sets d_intermediate=2*max(in,out); we
    ABANDON hidden_mult and binary-search an explicit dense hidden width to match the
    tropical *trainable* count (per head-regime) within 2%, asserting+aborting on miss.
  * SOFT-TRAIN / HARD-DEPLOY.  The shipped/silicon net is hard (tau=0). Headline = HARD
    test loss after annealing tau 1->0 during training (soft-train-eval-hard collapses).
  * STRONG OPTIMIZER.  AdamW only for the capacity claim (trainability ceiling); closed
    TropicalSGD/SignSGD are a separate labelled "shippability" column (full mode).
  * REALIZABILITY GATES.  An analytic single-dilation construction (verified 0.0 hard
    residual) and a frozen-max-plus teacher gate setup health.
  * LEAK-FREE SELECTION.  LR is selected per cell by TRAIN loss; TEST loss is only reported.
  * DEAD-UNIT INSTRUMENTATION.  Hard-path argmax routes gradient to one winner; plateaus
    with high grad-sparsity are flagged trainability-limited, not capacity.

Usage:
    python scripts/capacity_probe.py --mode smoke      # ~2 min, validates every gate
    python scripts/capacity_probe.py --mode headline   # ~1.5 h, the decisive number
    python scripts/capacity_probe.py --mode full        # overnight, the whole grid
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tikitorch.nn as ttnn
import tikitorch.planning as P
from tikitorch.planning.muzero.env import GridWorldNav
from tikitorch.planning.muzero.networks import TropicalRepresentation, TropicalPrediction
from tikitorch.planning.muzero.networks_dense import (
    _DenseRepresentation, _DensePrediction, _mlp)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results", "capacity_probe"))
GS = 5                     # grid side for the multigrid pool (obs_dim = 3*GS*GS = 75)
SYN_D = 128                # synthetic-target input dim (>= max latent_dim: no rank degeneracy)
NTR_SYN, NTE_SYN = 192, 512
N_ACTIONS = 4
PARAM_TOL = 0.02

# headline backbone (synthetic, densely sampled) + ecological + gate
RIGHT = "right prior"
WRONG = "wrong prior"


# ---------------------------------------------------------------------------
# Init-path net (the fit surface) — tropical or param-matched dense core
# ---------------------------------------------------------------------------
class InitNet(nn.Module):
    """obs -> (policy_logits, value_raw) via repr + pred. Mirrors TropicalMuZeroNet.initial."""

    def __init__(self, kind, obs_dim, L, tau, value_support, dense_hidden=None):
        super().__init__()
        self.kind = kind
        if kind == "tropical":
            self.repr = TropicalRepresentation(obs_dim, L, tau=tau)
            self.pred = TropicalPrediction(L, N_ACTIONS, tau=tau, value_support=value_support)
        else:
            self.repr = _DenseRepresentation(obs_dim, L, hidden_mult=1)
            self.repr.net = _mlp(obs_dim, L, hidden=dense_hidden)        # explicit hidden override
            self.pred = _DensePrediction(L, N_ACTIONS, value_support=value_support, hidden_mult=1)
            self.pred.mix = _mlp(L, L, hidden=dense_hidden)             # explicit hidden override

    def set_tau(self, tau):
        if self.kind == "tropical":
            self.repr.set_tau(tau)
            self.pred.set_tau(tau)

    def forward(self, obs):
        return self.pred(self.repr(obs))


def trainable_params(net):
    return [p for p in net.parameters() if p.requires_grad]


def n_trainable(net):
    return sum(p.numel() for p in trainable_params(net))


def configure_trainable(net, regime, probe):
    """requires_grad for the (regime, probe) cell. probe in {value,policy} (other head
    frozen). regime: R1=cores+head, R2=cores only (head frozen), R3=head only (cores frozen)."""
    for p in net.parameters():
        p.requires_grad_(True)
    other = net.pred.policy_head if probe == "value" else net.pred.value_head
    for p in other.parameters():
        p.requires_grad_(False)
    probed = net.pred.value_head if probe == "value" else net.pred.policy_head
    if regime == "R2":
        for p in probed.parameters():
            p.requires_grad_(False)
    elif regime == "R3":
        for p in net.repr.parameters():
            p.requires_grad_(False)
        for p in net.pred.mix.parameters():
            p.requires_grad_(False)
    return trainable_params(net)


def match_dense_hidden(obs_dim, L, value_support, regime, probe, target_trop):
    """Bisect the integer dense hidden width matching target_trop within PARAM_TOL."""
    def dense_count(h):
        net = InitNet("dense", obs_dim, L, 0.0, value_support, dense_hidden=h)
        configure_trainable(net, regime, probe)
        return n_trainable(net)
    lo, hi = 1, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if dense_count(mid) < target_trop:
            lo = mid + 1
        else:
            hi = mid
    best = min([max(1, lo - 1), lo, lo + 1], key=lambda h: abs(dense_count(h) - target_trop))
    dp = dense_count(best)
    return best, dp, abs(dp - target_trop) / max(target_trop, 1)


# ---------------------------------------------------------------------------
# Targets — each returns train + held-out test splits
# ---------------------------------------------------------------------------
def _rand_grid(rng, gs=GS, wall_p=0.22):
    cells = [["#" if rng.random() < wall_p else "." for _ in range(gs)] for _ in range(gs)]
    cells[rng.randrange(gs)][rng.randrange(gs)] = "G"
    return ["".join(row) for row in cells]


def _grid_pool(rng, n_grids, gs=GS):
    obs, vs, pis = [], [], []
    used = 0
    while used < n_grids:
        try:
            env = GridWorldNav(_rand_grid(rng, gs))
        except AssertionError:
            continue
        used += 1
        R, succ, _ = env.as_mdp()
        V, _ = P.TropicalValueIteration(env.n_states, env.n_actions, gamma=1.0,
                                        n_sweeps=env.n_states * 2)(torch.zeros(env.n_states), R, succ)
        for s in range(env.n_states):
            v = float(V[s])
            if v < -(gs * gs):                      # unreachable -> VI floor; drop
                continue
            q = torch.tensor([float(R[s, a] + V[succ[s, a]]) for a in range(env.n_actions)])
            mask = (q >= float(q.max()) - 1e-6).float()
            obs.append(env.obs(s).flatten()); vs.append(v); pis.append(mask / mask.sum())
    return torch.stack(obs), torch.tensor(vs), torch.stack(pis)


def _syn_X(seed, n):
    return torch.rand(n, SYN_D, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def _split_syn(name, seed, fn):
    Xtr, Xte = _syn_X(seed * 17 + 1, NTR_SYN), _syn_X(seed * 17 + 999, NTE_SYN)
    with torch.no_grad():
        ytr, yte = fn(Xtr), fn(Xte)
    mu, sd = ytr.mean(), ytr.std() + 1e-8                 # standardize by TRAIN stats
    return dict(Xtr=Xtr, ytr=(ytr - mu) / sd, Xte=Xte, yte=(yte - mu) / sd,
                probe="value", value_support=None, obs_dim=SYN_D)


def build_target(name, seed):
    g = torch.Generator().manual_seed(seed * 104729 + 3)
    if name == "linf_orderstat_dilation":
        a = torch.rand(SYN_D, generator=g) * 2 - 1
        d = _split_syn(name, seed, lambda X: (X + a).max(dim=-1).values)
        d.update(meta=f"{RIGHT}: single max-plus dilation max_i(a_i+x_i)", prior=RIGHT, _a=a)
        return d
    if name == "structured_maxplus_teacher":
        teacher = ttnn.TropicalProjection(SYN_D, 1, tau=0.0)
        for p in teacher.parameters():
            p.requires_grad_(False)
        d = _split_syn(name, seed, lambda X: teacher(X).squeeze(-1))
        d.update(meta="REALIZABILITY GATE: frozen hard max-plus teacher (tropical must generalize)",
                 prior=RIGHT)
        return d
    if name == "random_relu_teacher":
        teacher = nn.Sequential(nn.Linear(SYN_D, 64), nn.ReLU(), nn.Linear(64, 1))
        for p in teacher.parameters():
            p.requires_grad_(False)
        d = _split_syn(name, seed, lambda X: teacher(X).squeeze(-1))
        d.update(meta=f"{WRONG} headline negative: oblique random-ReLU field", prior=WRONG)
        return d
    if name == "smooth_curvature_bowl":
        c = torch.rand(SYN_D, generator=g) * 2 - 1
        d = _split_syn(name, seed, lambda X: -((X - c) ** 2).sum(-1))
        d.update(meta=f"{WRONG} (smooth curvature): PWL approximates a bowl", prior=WRONG)
        return d
    if name == "minplus_costtogo_multigrid":
        rng = random.Random(seed * 7919 + 1)
        Xtr, Vtr, _ = _grid_pool(rng, n_grids=8)          # train grids
        Xte, Vte, _ = _grid_pool(rng, n_grids=8)          # DISJOINT held-out grids
        sup = (-(GS * GS) - 1, 1)
        return dict(Xtr=Xtr, ytr=Vtr, Xte=Xte, yte=Vte, probe="value", value_support=sup,
                    obs_dim=Xtr.shape[1], prior=RIGHT,
                    meta=f"{RIGHT} ecological: (min,+) cost-to-go, HELD-OUT grids")
    if name == "policy_greedy_multigrid":
        rng = random.Random(seed * 7919 + 1)
        Xtr, _, Pitr = _grid_pool(rng, n_grids=8)
        Xte, _, Pite = _grid_pool(rng, n_grids=8)
        return dict(Xtr=Xtr, ytr=Pitr, Xte=Xte, yte=Pite, probe="policy", value_support=None,
                    obs_dim=Xtr.shape[1], prior=RIGHT,
                    meta=f"{RIGHT}: greedy cost-to-go policy, HELD-OUT grids")
    raise ValueError(name)


def analytic_realizability_residual(a):
    """Hand-build a TropicalProjection computing max_i(a_i+x_i) exactly; HARD residual ~0."""
    D = a.numel()
    proj = ttnn.TropicalProjection(D, 1, d_intermediate=1, tau=0.0)
    with torch.no_grad():
        proj.W1.copy_(a.view(1, D)); proj.b1.zero_(); proj.W2.zero_(); proj.b2.zero_()
        X = torch.rand(256, D) * 2 - 1
        return float((proj(X).squeeze(-1) - (X + a).max(dim=-1).values).abs().max())


# ---------------------------------------------------------------------------
# Train one cell -> hard-eval on held-out TEST
# ---------------------------------------------------------------------------
def _loss(net, pol, vraw, target, probe):
    if probe == "value":
        return net.pred.value_head.loss(vraw, target)
    return -(target * F.log_softmax(pol, dim=-1)).sum(-1).mean()


def tau_schedule(arm, step, max_steps):
    if arm == "S1_soft":
        return 1.0
    if arm == "S2_direct_hard":
        return 0.0
    frac = min(step / (0.8 * max_steps), 1.0)              # S3 anneal then hard
    return 0.0 if frac >= 1.0 else max(1e-3, (1e-3) ** frac)


def make_optimizer(name, params, lr):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    import tikitorch.optim as ttoptim
    return ttoptim.SignSGD(params, lr=lr) if name == "signsgd" else ttoptim.TropicalSGD(params, lr=lr)


def train_cell(net, data, arm, lr, max_steps, seed, optimizer="adamw"):
    torch.manual_seed(seed * 2_654_435_761 % (2 ** 31))
    Xtr, ytr, Xte, yte, probe = data["Xtr"], data["ytr"], data["Xte"], data["yte"], data["probe"]
    params = trainable_params(net)
    opt = make_optimizer(optimizer, params, lr)
    losses, hard_start, converged, steps_to_plateau = [], None, False, max_steps
    for step in range(max_steps):
        net.set_tau(tau_schedule(arm, step, max_steps))
        opt.zero_grad()
        pol, vraw = net(Xtr)
        loss = _loss(net, pol, vraw, ytr, probe)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        is_hard = tau_schedule(arm, step, max_steps) == 0.0
        if is_hard and hard_start is None:
            hard_start = step
        if is_hard and hard_start is not None and step - hard_start >= 500:
            ref, cur = losses[-500], losses[-1]
            if (ref - cur) / (abs(ref) + 1e-9) < 1e-5:
                converged, steps_to_plateau = True, step
                break
    soft_best = min(losses) if arm == "S1_soft" else float("nan")

    dead_frac, grad_sparsity = float("nan"), float("nan")
    if net.kind == "tropical" and net.repr.enc.W1.grad is not None:
        W1g = net.repr.enc.W1.grad
        dead_frac = float((W1g.abs().sum(dim=1) == 0).float().mean())
        cg = [p.grad for p in net.repr.parameters() if p.grad is not None]
        tot = sum(x.numel() for x in cg) or 1
        grad_sparsity = sum(int((x == 0).sum()) for x in cg) / tot

    # HARD-EVAL (tau=0): train (sanity ~0) + held-out TEST (headline) + tau ladder on test
    net.eval()
    ladder = {}
    with torch.no_grad():
        net.set_tau(0.0)
        ptr, vtr = net(Xtr)
        train_hard = float(_loss(net, ptr, vtr, ytr, probe))
        pte, vte = net(Xte)
        test_hard = float(_loss(net, pte, vte, yte, probe))
        test_rmse = test_exact = test_top1 = float("nan")
        if probe == "value":
            sc = net.pred.value_head.scalar(vte)
            test_rmse = float(((sc - yte) ** 2).mean().sqrt())
            test_exact = float((sc.round() == yte.round()).float().mean())
        else:
            test_top1 = float((pte.argmax(-1) == yte.argmax(-1)).float().mean())
        for t in [1.0, 0.5, 0.25, 0.1, 0.05, 0.0]:
            net.set_tau(t)
            p2, v2 = net(Xte)
            ladder[t] = float(_loss(net, p2, v2, yte, probe))
    net.train()
    return dict(train_hard=train_hard, test_hard=test_hard, test_rmse=test_rmse,
                test_exact_frac=test_exact, test_policy_top1=test_top1, soft_best_loss=soft_best,
                dead_unit_fraction=dead_frac, grad_sparsity=grad_sparsity,
                steps_to_plateau=steps_to_plateau, converged_flag=converged,
                n_train=int(Xtr.shape[0]), n_test=int(Xte.shape[0]), tau_ladder=ladder)


# ---------------------------------------------------------------------------
# Grid + driver
# ---------------------------------------------------------------------------
def grid_for(mode):
    if mode == "smoke":
        return dict(targets=["linf_orderstat_dilation", "random_relu_teacher"], Ls=[16],
                    regimes=["R1"], arms=["S3_anneal"], lrs=[3e-3], opts=["adamw"],
                    seeds=[0], max_steps=400)
    if mode == "headline":
        return dict(targets=["linf_orderstat_dilation", "structured_maxplus_teacher",
                             "random_relu_teacher", "smooth_curvature_bowl",
                             "minplus_costtogo_multigrid"],
                    Ls=[16, 64, 128], regimes=["R1"], arms=["S3_anneal"],
                    lrs=[3e-3], opts=["adamw"], seeds=list(range(5)), max_steps=4000)
    if mode == "full":
        return dict(targets=["linf_orderstat_dilation", "structured_maxplus_teacher",
                             "random_relu_teacher", "smooth_curvature_bowl",
                             "minplus_costtogo_multigrid", "policy_greedy_multigrid"],
                    Ls=[8, 16, 32, 64, 128], regimes=["R1", "R2"],
                    arms=["S1_soft", "S2_direct_hard", "S3_anneal"], lrs=[3e-4, 1e-3, 3e-3, 1e-2],
                    opts=["adamw", "tropicalsgd", "signsgd"], seeds=list(range(8)), max_steps=8000)
    raise ValueError(mode)


def run(mode):
    os.makedirs(os.path.join(RESULTS, "targets"), exist_ok=True)
    g = grid_for(mode)
    runs_path = os.path.join(RESULTS, f"runs_{mode}.jsonl")
    audit, n_rows = {}, 0

    resid = analytic_realizability_residual(build_target("linf_orderstat_dilation", 0)["_a"])
    print(f"[gate] analytic single-dilation hard residual = {resid:.3e} (must be ~0)")
    assert resid < 1e-3, "analytic realizability gate FAILED — harness broken"

    with open(runs_path, "w") as fh:
        for target in g["targets"]:
            for seed in g["seeds"]:
                data = build_target(target, seed)
                obs_dim, probe, sup, prior = data["obs_dim"], data["probe"], data["value_support"], data["prior"]
                torch.save({k: data[k] for k in ("Xtr", "ytr", "Xte", "yte")},
                           os.path.join(RESULTS, "targets", f"{target}_seed{seed}.pt"))
                for L in g["Ls"]:
                    for regime in g["regimes"]:
                        tnet = InitNet("tropical", obs_dim, L, 0.0, sup)
                        configure_trainable(tnet, regime, probe)
                        trop_n = n_trainable(tnet)
                        key = (target, L, regime, probe)
                        if key not in audit:
                            h, dp, pct = match_dense_hidden(obs_dim, L, sup, regime, probe, trop_n)
                            audit[key] = dict(dense_hidden=h, trop_params=trop_n, dense_params=dp, pct_diff=pct)
                            assert pct <= PARAM_TOL, f"param-match {pct:.3%} > {PARAM_TOL:.0%} at {key}"
                        am = audit[key]
                        for arm in g["arms"]:
                            for opt_name in g["opts"]:
                                for lr in g["lrs"]:
                                    for kind in ["tropical", "dense"]:
                                        net = (InitNet("tropical", obs_dim, L, 1.0, sup) if kind == "tropical"
                                               else InitNet("dense", obs_dim, L, 0.0, sup, dense_hidden=am["dense_hidden"]))
                                        configure_trainable(net, regime, probe)
                                        m = train_cell(net, data, arm, lr, g["max_steps"], seed, opt_name)
                                        row = dict(target_type=target, prior=prior, net=kind, latent_dim=L,
                                                   regime=regime, tau_arm=arm, optimizer=opt_name, lr=lr,
                                                   seed=seed, obs_dim=obs_dim, probe=probe,
                                                   trop_params=am["trop_params"], dense_params=am["dense_params"],
                                                   pct_diff=am["pct_diff"], **m)
                                        fh.write(json.dumps(row) + "\n"); fh.flush()
                                        n_rows += 1
                                        print(f"[{n_rows}] {target[:24]:24s} {kind:8s} L={L:3d} {opt_name:10s} "
                                              f"{arm:13s} lr={lr:.0e} s{seed} -> train={m['train_hard']:.4f} "
                                              f"TEST={m['test_hard']:.4f} conv={m['converged_flag']}")
    with open(os.path.join(RESULTS, f"param_match_audit_{mode}.json"), "w") as f:
        json.dump({str(k): v for k, v in audit.items()}, f, indent=2)
    summarize(runs_path, mode)
    print(f"\nDONE: {n_rows} rows -> {runs_path}")


def summarize(runs_path, mode):
    """Headline = paired tropical-minus-dense HELD-OUT TEST hard gap, per (target,L,regime).
    LR/arm/opt selected per (seed,net) by lowest TRAIN loss (no test leakage)."""
    rows = [json.loads(l) for l in open(runs_path)]
    rows = [r for r in rows if r["optimizer"] == "adamw"]           # capacity claim = AdamW only
    pick = {}                                                       # (tgt,L,reg,seed,net) -> chosen row
    for r in rows:
        k = (r["target_type"], r["latent_dim"], r["regime"], r["seed"], r["net"])
        if k not in pick or r["train_hard"] < pick[k]["train_hard"]:
            pick[k] = r
    lines = ["# Capacity probe — §2a expressivity (tropical vs param-matched dense)", "",
             f"_mode: {mode}. Headline = paired HELD-OUT TEST hard-eval (tau=0) gap tropical−dense "
             "(>0 ⇒ tropical underfits/wrong-prior; <0 ⇒ tropical generalizes better/right-prior). "
             "Both interpolate train (~0); LR selected by train loss (no test leakage)._", ""]
    for tgt in sorted({r["target_type"] for r in rows}, key=lambda t: (rows_prior(rows, t), t)):
        pr = rows_prior(rows, t=tgt)
        lines += [f"## {tgt}  ·  _{pr}_",
                  "| L | regime | n | trop train | trop TEST (med) | dense TEST (med) | "
                  "paired TEST gap (mean±std) | Wilcoxon p | trop wins |",
                  "|---|---|---|---|---|---|---|---|---|"]
        Ls = sorted({r["latent_dim"] for r in rows if r["target_type"] == tgt})
        regs = sorted({r["regime"] for r in rows if r["target_type"] == tgt})
        for L in Ls:
            for reg in regs:
                seeds = sorted({r["seed"] for r in rows
                                if r["target_type"] == tgt and r["latent_dim"] == L and r["regime"] == reg})
                tr = [pick.get((tgt, L, reg, s, "tropical")) for s in seeds]
                dn = [pick.get((tgt, L, reg, s, "dense")) for s in seeds]
                pairs = [(a, b) for a, b in zip(tr, dn) if a and b]
                if not pairs:
                    continue
                tt = np.array([a["test_hard"] for a, _ in pairs])
                dd = np.array([b["test_hard"] for _, b in pairs])
                trn = np.array([a["train_hard"] for a, _ in pairs])
                gap = tt - dd
                p = float("nan")
                if len(gap) >= 5 and np.any(gap != 0):
                    try:
                        from scipy.stats import wilcoxon
                        p = float(wilcoxon(tt, dd).pvalue)
                    except Exception:
                        pass
                lines.append(f"| {L} | {reg} | {len(gap)} | {np.median(trn):.4f} | {np.median(tt):.4f} | "
                             f"{np.median(dd):.4f} | {gap.mean():+.4f}±{gap.std():.4f} | {p:.3g} | "
                             f"{int((gap < 0).sum())}/{len(gap)} |")
        lines.append("")
    out = os.path.join(RESULTS, f"FINDINGS_{mode}.md")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[summary] -> {out}")


def rows_prior(rows, t):
    for r in rows:
        if r["target_type"] == t:
            return r["prior"]
    return "?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "headline", "full"], default="headline")
    run(ap.parse_args().mode)
