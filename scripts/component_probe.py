"""component_probe — WHICH part of the Tropical MuZero net is the expressivity bottleneck?

The capacity probe (scripts/capacity_probe.py, manuscript/results/capacity_probe/FINDINGS.md) localized the §2a
underfit to the LEARNED NET (not the search) and to non-lattice value surfaces. This probe localizes it
FURTHER — to a single net COMPONENT — so we know the minimal dense/DSP augmentation that closes the gap
while keeping the MCTS<->net loop on-die.

Method: factor the net into repr / dynamics / pred. To isolate component X, hold the other two as a
PARAM-MATCHED DENSE reference and compare X=tropical vs X=dense (X also param-matched). The two nets
differ ONLY in component X, so the held-out TEST gap (tropical_X − dense_X) is the expressivity cost of
making JUST that component tropical. Decoded value RMSE (ScalarHead/MSE) is the decision-relevant metric
(the capacity-probe panel showed categorical CE is a calibration artifact).

Targets:
  * relu / bowl   (WRONG prior, the barrier)            -> repr & pred isolation, initial-path value
  * costtogo_value (RIGHT prior, ecological)            -> repr & pred isolation, initial-path value
  * costtogo_traj  (the K-step UNROLL — the hot path)   -> DYNAMICS isolation (the unmeasured component)
  * costtogo_policy                                     -> repr & pred isolation, POLICY (user's concern)

Both classes interpolate train; the gap is generalization. Usage:
    python scripts/component_probe.py --mode smoke
    python scripts/component_probe.py --mode full
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scripts.capacity_probe as cp
except ModuleNotFoundError:
    import capacity_probe as cp
from tikitorch.planning.muzero.env import GridWorldNav
from tikitorch.planning.muzero.networks import (
    TropicalRepresentation, TropicalDynamics, TropicalPrediction)
from tikitorch.planning.muzero.networks_dense import (
    _DenseRepresentation, _DenseDynamics, _DensePrediction, _mlp)
import tikitorch.planning as P

RESULTS = cp.RESULTS
GS = cp.GS
PARAM_TOL = 0.02
K_UNROLL = 3


# ---------------------------------------------------------------------------
# Component builders (dense variants take an explicit hidden width)
# ---------------------------------------------------------------------------
def t_repr(obs, L):              return TropicalRepresentation(obs, L, tau=1.0)
def t_pred(L, A):                return TropicalPrediction(L, A, tau=1.0, value_support=None)
def t_dyn(L, A):                 return TropicalDynamics(L, A, tau=1.0)

def d_repr(obs, L, H):
    m = _DenseRepresentation(obs, L, hidden_mult=1); m.net = _mlp(obs, L, hidden=H); return m
def d_pred(L, A, H):
    m = _DensePrediction(L, A, value_support=None, hidden_mult=1); m.mix = _mlp(L, L, hidden=H); return m
def d_dyn(L, A, H):
    m = _DenseDynamics(L, A, reward_support=None, hidden_mult=1)
    m.blocks = nn.ModuleList([_mlp(L, L, hidden=H) for _ in m.blocks]); return m

def comp_params(builder, *a):
    return sum(p.numel() for p in builder(*a).parameters())

def match_component(comp, obs, L, A, target_trop):
    """Bisect the dense hidden for `comp` to match `target_trop` total params within PARAM_TOL."""
    def count(H):
        if comp == "repr": return comp_params(d_repr, obs, L, H)
        if comp == "pred": return comp_params(d_pred, L, A, H)
        return comp_params(d_dyn, L, A, H)
    lo, hi = 1, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target_trop: lo = mid + 1
        else: hi = mid
    best = min([max(1, lo - 1), lo, lo + 1], key=lambda H: abs(count(H) - target_trop))
    dp = count(best)
    return best, dp, abs(dp - target_trop) / max(target_trop, 1)


class ComponentNet(nn.Module):
    """repr / dynamics / pred, each independently 'T' (tropical) or 'D' (dense, matched hidden)."""

    def __init__(self, kinds, obs, L, A, hiddens):
        super().__init__()
        self.repr = t_repr(obs, L) if kinds["repr"] == "T" else d_repr(obs, L, hiddens["repr"])
        self.dyn = t_dyn(L, A) if kinds["dyn"] == "T" else d_dyn(L, A, hiddens["dyn"])
        self.pred = t_pred(L, A) if kinds["pred"] == "T" else d_pred(L, A, hiddens["pred"])
        self.kinds = kinds

    def set_tau(self, tau):
        for m in (self.repr, self.dyn, self.pred):
            if hasattr(m, "set_tau"):
                try: m.set_tau(tau)
                except Exception: pass

    def initial(self, obs):
        return self.pred(self.repr(obs))                       # (policy_logits, value_raw)

    def unroll(self, obs0, actions):                           # actions: (N, K) long
        h = self.repr(obs0)
        outs = [self.pred(h)]
        for k in range(actions.shape[1]):
            h, _ = self.dyn(h, actions[:, k])
            outs.append(self.pred(h))
        return outs                                            # list len K+1 of (pol, vraw)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def _grids_value(seed, n_grids):
    rng = random.Random(seed)
    obs, vs = [], []
    used = 0
    while used < n_grids:
        try: env = GridWorldNav(cp._rand_grid(rng))
        except AssertionError: continue
        used += 1
        R, succ, _ = env.as_mdp()
        V, _ = P.TropicalValueIteration(env.n_states, env.n_actions, gamma=1.0,
                                        n_sweeps=env.n_states * 2)(torch.zeros(env.n_states), R, succ)
        for s in range(env.n_states):
            if float(V[s]) < -(GS * GS): continue
            obs.append(env.obs(s).flatten()); vs.append(float(V[s]))
    return torch.stack(obs), torch.tensor(vs)


def _grids_policy(seed, n_grids):
    rng = random.Random(seed)
    obs, pis = [], []
    used = 0
    while used < n_grids:
        try: env = GridWorldNav(cp._rand_grid(rng))
        except AssertionError: continue
        used += 1
        R, succ, _ = env.as_mdp()
        V, _ = P.TropicalValueIteration(env.n_states, env.n_actions, gamma=1.0,
                                        n_sweeps=env.n_states * 2)(torch.zeros(env.n_states), R, succ)
        for s in range(env.n_states):
            if float(V[s]) < -(GS * GS): continue
            q = torch.tensor([float(R[s, a] + V[succ[s, a]]) for a in range(env.n_actions)])
            mask = (q >= float(q.max()) - 1e-6).float()
            obs.append(env.obs(s).flatten()); pis.append(mask / mask.sum())
    return torch.stack(obs), torch.stack(pis)


def _grids_traj(seed, n_grids, K=K_UNROLL):
    rng = random.Random(seed)
    obs0, acts, ys = [], [], []
    used = 0
    while used < n_grids:
        try: env = GridWorldNav(cp._rand_grid(rng))
        except AssertionError: continue
        used += 1
        R, succ, _ = env.as_mdp()
        V, _ = P.TropicalValueIteration(env.n_states, env.n_actions, gamma=1.0,
                                        n_sweeps=env.n_states * 2)(torch.zeros(env.n_states), R, succ)
        for s in range(env.n_states):
            if float(V[s]) < -(GS * GS): continue
            a_seq = [rng.randrange(env.n_actions) for _ in range(K)]   # FIXED non-greedy seq
            y = [float(V[s])]; cur = s
            for a in a_seq:
                cur, _, _ = env.step_state(cur, a)
                y.append(float(V[cur]) if float(V[cur]) >= -(GS * GS) else y[-1])
            obs0.append(env.obs(s).flatten()); acts.append(a_seq); ys.append(y)
    return torch.stack(obs0), torch.tensor(acts, dtype=torch.long), torch.tensor(ys)


def get_data(target, seed):
    if target in ("relu", "bowl"):
        name = "random_relu_teacher" if target == "relu" else "smooth_curvature_bowl"
        d = cp.build_target(name, seed)
        return dict(mode="value_initial", obs_dim=d["obs_dim"], A=cp.N_ACTIONS,
                    Xtr=d["Xtr"], ytr=d["ytr"], Xte=d["Xte"], yte=d["yte"])
    if target == "costtogo_value":
        Xtr, ytr = _grids_value(seed * 7919 + 1, 8)
        Xte, yte = _grids_value(seed * 7919 + 5001, 8)
        return dict(mode="value_initial", obs_dim=Xtr.shape[1], A=cp.N_ACTIONS,
                    Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    if target == "costtogo_policy":
        Xtr, ytr = _grids_policy(seed * 7919 + 1, 8)
        Xte, yte = _grids_policy(seed * 7919 + 5001, 8)
        return dict(mode="policy_initial", obs_dim=Xtr.shape[1], A=cp.N_ACTIONS,
                    Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    if target == "costtogo_traj":
        Xtr, atr, ytr = _grids_traj(seed * 7919 + 1, 8)
        Xte, ate, yte = _grids_traj(seed * 7919 + 5001, 8)
        return dict(mode="value_unroll", obs_dim=Xtr.shape[1], A=cp.N_ACTIONS,
                    Xtr=Xtr, atr=atr, ytr=ytr, Xte=Xte, ate=ate, yte=yte)
    raise ValueError(target)


# ---------------------------------------------------------------------------
# Train + hard-eval (held-out)
# ---------------------------------------------------------------------------
def _forward_loss(net, data, split):
    X = data["Xtr"] if split == "tr" else data["Xte"]
    y = data["ytr"] if split == "tr" else data["yte"]
    if data["mode"] == "policy_initial":
        pol, _ = net.initial(X)
        return -(y * F.log_softmax(pol, -1)).sum(-1).mean(), pol, None
    if data["mode"] == "value_unroll":
        acts = data["atr"] if split == "tr" else data["ate"]
        outs = net.unroll(X, acts)
        v = torch.stack([net.pred.value_head.scalar(o[1]) for o in outs], dim=1)  # (N,K+1) decoded
        return F.mse_loss(v, y), None, v
    pol, vraw = net.initial(X)
    v = net.pred.value_head.scalar(vraw)
    return F.mse_loss(v, y), pol, v


def train_component(net, data, lr, max_steps, seed):
    torch.manual_seed(seed * 2_654_435_761 % (2 ** 31))
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    hard_start, conv = None, False
    losses = []
    for step in range(max_steps):
        net.set_tau(cp.tau_schedule("S3_anneal", step, max_steps))
        opt.zero_grad()
        loss, _, _ = _forward_loss(net, data, "tr")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if cp.tau_schedule("S3_anneal", step, max_steps) == 0.0:
            if hard_start is None: hard_start = step
            if step - hard_start >= 400:
                ref, cur = losses[-400], losses[-1]
                if (ref - cur) / (abs(ref) + 1e-9) < 1e-5:
                    conv = True; break
    net.eval(); net.set_tau(0.0)
    with torch.no_grad():
        tr_loss, _, _ = _forward_loss(net, data, "tr")
        te_loss, pol, v = _forward_loss(net, data, "te")
        rmse = top1 = float("nan")
        if data["mode"] == "policy_initial":
            top1 = float((pol.argmax(-1) == data["yte"].argmax(-1)).float().mean())
        else:
            rmse = float(te_loss.sqrt())
    net.train()
    return dict(train_loss=float(tr_loss), test_loss=float(te_loss), test_rmse=rmse,
                test_top1=top1, converged=conv, steps=len(losses))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    ("repr", "relu"), ("pred", "relu"),
    ("repr", "bowl"), ("pred", "bowl"),
    ("repr", "costtogo_value"), ("pred", "costtogo_value"),
    ("dyn", "costtogo_traj"),
    ("repr", "costtogo_policy"), ("pred", "costtogo_policy"),
]


def grid_for(mode):
    if mode == "smoke":
        return dict(exps=[("repr", "relu"), ("dyn", "costtogo_traj")], Ls=[64], seeds=[0], max_steps=400)
    if mode == "finish":   # the slow dynamics + the policy isolations, reduced budget
        return dict(exps=[("dyn", "costtogo_traj"), ("repr", "costtogo_policy"), ("pred", "costtogo_policy")],
                    Ls=[64, 128], seeds=list(range(4)), max_steps=2000)
    return dict(exps=EXPERIMENTS, Ls=[64, 128], seeds=list(range(5)), max_steps=4000)


def matched_hiddens(obs, L, A):
    """The param-matched dense hidden for each component (the fixed-dense reference uses these too)."""
    h = {}
    for comp, builder in (("repr", lambda: comp_params(t_repr, obs, L)),
                          ("pred", lambda: comp_params(t_pred, L, A)),
                          ("dyn", lambda: comp_params(t_dyn, L, A))):
        H, dp, pct = match_component(comp, obs, L, A, builder())
        assert pct <= PARAM_TOL, f"param-match {pct:.3%} for {comp} L={L}"
        h[comp] = dict(hidden=H, trop=builder(), dense=dp, pct=pct)
    return h


def run(mode):
    os.makedirs(RESULTS, exist_ok=True)
    g = grid_for(mode)
    out = os.path.join(RESULTS, f"component_runs_{mode}.jsonl")
    n = 0
    with open(out, "w") as fh:
        for comp, target in g["exps"]:
            for seed in g["seeds"]:
                data = get_data(target, seed)
                obs, A = data["obs_dim"], data["A"]
                for L in g["Ls"]:
                    mh = matched_hiddens(obs, L, A)
                    hiddens = {c: mh[c]["hidden"] for c in mh}
                    for algebra in ["T", "D"]:                  # isolated component tropical vs dense
                        kinds = {"repr": "D", "dyn": "D", "pred": "D"}
                        kinds[comp] = algebra
                        net = ComponentNet(kinds, obs, L, A, hiddens)
                        m = train_component(net, data, 3e-3, g["max_steps"], seed)
                        row = dict(isolated=comp, target=target, mode=data["mode"], algebra=algebra,
                                   latent_dim=L, seed=seed,
                                   comp_trop_params=mh[comp]["trop"], comp_dense_params=mh[comp]["dense"],
                                   comp_pct=mh[comp]["pct"], **m)
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        n += 1
                        print(f"[{n}] iso={comp:5s} {target:16s} {algebra} L={L:3d} s{seed} -> "
                              f"train={m['train_loss']:.4f} TEST={m['test_loss']:.4f} "
                              f"rmse={m['test_rmse']:.3f} top1={m['test_top1']:.3f} conv={m['converged']}")
    summarize(out, mode)
    print(f"\nDONE: {n} rows -> {out}")


def summarize(out, mode):
    rows = [json.loads(l) for l in open(out)]
    pick = {}
    for r in rows:
        pick[(r["isolated"], r["target"], r["latent_dim"], r["seed"], r["algebra"])] = r
    lines = ["# Component-isolation probe — which net component is the expressivity bottleneck?", "",
             "_Each row: isolate ONE component as tropical vs dense (others = param-matched dense). "
             "Gap = (tropical_X − dense_X) held-out TEST; >0 ⇒ making X tropical HURTS. RMSE in decoded "
             "value units (cells for cost-to-go) or policy top-1 agreement._", "",
             "| isolated | target | L | n | dense TEST (med) | tropical TEST (med) | gap (mean±std) | trop worse |",
             "|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["isolated"], r["target"], r["latent_dim"]) for r in rows})
    for (comp, tgt, L) in keys:
        seeds = sorted({r["seed"] for r in rows if r["isolated"] == comp and r["target"] == tgt and r["latent_dim"] == L})
        pol = "policy" in tgt
        T = [pick.get((comp, tgt, L, s, "T")) for s in seeds]
        D = [pick.get((comp, tgt, L, s, "D")) for s in seeds]
        pairs = [(a, b) for a, b in zip(T, D) if a and b]
        if not pairs: continue
        key = "test_top1" if pol else "test_loss"
        tt = np.array([a[key] for a, _ in pairs]); dd = np.array([b[key] for _, b in pairs])
        gap = (dd - tt) if pol else (tt - dd)            # >0 = tropical worse (lower top1 / higher loss)
        worse = int((gap > 0).sum())
        lines.append(f"| {comp} | {tgt} | {L} | {len(gap)} | {np.median(dd):.4f} | {np.median(tt):.4f} | "
                     f"{gap.mean():+.4f}±{gap.std():.4f} | {worse}/{len(gap)} |")
    path = os.path.join(RESULTS, f"COMPONENT_FINDINGS_{mode}.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"\n[summary] -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "finish"], default="full")
    run(ap.parse_args().mode)
