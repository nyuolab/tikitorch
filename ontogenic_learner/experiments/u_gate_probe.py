"""U-gate probe (G2_SPEC amendment A2): does RMS-normalized settling condition at width?

The paradigm's end-state is train == inference: one settling computation, target clamped
(learn) or free (infer). G2a-P1 showed plain fixed-step settling fails at width 768. This
probe tests the minimal conditioning fix — per-layer RMS-normalized state updates — without
touching the library (standalone settle/update functions over a stock DyadicPCNet).

Run (repo root): PYTHONPATH=. python ontogenic_learner/experiments/u_gate_probe.py
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import two_speed_learner as tsl
from two_speed_learner import BASE_DOMAIN, onehot, val_bpc, windows

import torch
import torch.nn.functional as F
from tikitorch.nn.predictive_coding import DyadicPCNet

OUT = os.path.join(tsl.REPO, "ontogenic_learner/results/g2_rule_probe")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
CTX, B, HID, DEPTH = 12, 256, 768, 8


def settle(net, x0, target, n_steps, gamma, mode):
    """Replicates DyadicPCNet.infer() but with optionally RMS-normalized state updates.
    target=None -> free settling (pure inference); clamped -> learning configuration."""
    states = [s.detach() for s in net.feedforward(x0)]
    free = list(range(1, net.depth)) if target is not None \
        else list(range(1, net.depth + 1))
    if target is not None:
        states[net.depth] = target
    for p in net.parameters():
        p.requires_grad_(False)
    for i in free:
        states[i] = states[i].clone().requires_grad_(True)
    for _ in range(n_steps):
        grads = torch.autograd.grad(net.energy(states), [states[i] for i in free])
        with torch.no_grad():
            for i, g in zip(free, grads):
                if mode == "mu":                 # RMS-normalize: bounded per-element step
                    g = g / (g.pow(2).mean().sqrt() + 1e-8)
                states[i] -= gamma * g
        for i in free:
            states[i] = states[i].detach().requires_grad_(True)
    for p in net.parameters():
        p.requires_grad_(True)
    return [s.detach() for s in states]


def mu_train_step(net, xb, yb, opt, n_steps, gamma, mode):
    target = net._make_target(yb, ref=xb)
    states = settle(net, xb, target, n_steps, gamma, mode)
    opt.zero_grad(set_to_none=True)
    net.energy(states).backward()
    opt.step()
    return all(torch.isfinite(p).all() for p in net.parameters())


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    enc, V = tsl.build_corpora()
    val = {d: windows(enc[d], CTX, len(enc[d]) - len(enc[d]) // 10, 4096, 3)
           for d in [BASE_DOMAIN, "shakespeare"]}
    dims = [CTX * V] + [HID] * (DEPTH - 1) + [V]

    print("===== U-gate: phylogeny pretrain =====", flush=True)
    torch.manual_seed(0)
    base = DyadicPCNet(dims, residual=True, gamma=0.0625).to(DEV)
    opt = torch.optim.Adam(base.parameters(), lr=1e-3)
    data = enc[BASE_DOMAIN]
    gen = torch.Generator().manual_seed(0)
    hi = int(0.9 * len(data)) - 4000
    for _ in range(3000):
        i = torch.randint(hi, (1,), generator=gen).item()
        xi, yi = windows(data, CTX, i, B, 1)
        base.train_step_cascade(onehot(xi, V).to(DEV), yi.to(DEV), opt, mode="dyadic")
    b_wiki = val_bpc(base, val[BASE_DOMAIN][0].to(DEV), val[BASE_DOMAIN][1].to(DEV), V)
    b_shak = val_bpc(base, val["shakespeare"][0].to(DEV), val["shakespeare"][1].to(DEV), V)
    print(f"  base: wiki={b_wiki:.3f} shak={b_shak:.3f}", flush=True)
    import copy
    rows = []

    # cascade-600 reference for stage 2
    ref = copy.deepcopy(base)
    ro = torch.optim.Adam(ref.parameters(), lr=1e-4)
    sdata = enc["shakespeare"]
    for step in range(600):
        xi, yi = windows(sdata, CTX, step * B, B, 1)
        ref.train_step_cascade(onehot(xi, V).to(DEV), yi.to(DEV), ro, mode="dyadic")
    r_shak = val_bpc(ref, val["shakespeare"][0].to(DEV), val["shakespeare"][1].to(DEV), V)
    r_wiki = val_bpc(ref, val[BASE_DOMAIN][0].to(DEV), val[BASE_DOMAIN][1].to(DEV), V)
    print(f"  cascade-600 ref: shak={r_shak:.3f} wiki={r_wiki:.3f}", flush=True)

    for mode, gamma, n_steps in [("plain", 0.0625, 16), ("mu", 0.0625, 16),
                                 ("mu", 0.125, 16), ("mu", 0.0625, 32),
                                 ("mu", 0.125, 32)]:
        tag = f"{mode}_g{gamma}_s{n_steps}"
        net = copy.deepcopy(base)
        o = torch.optim.Adam(net.parameters(), lr=1e-4)
        t0, ok = time.time(), True
        # stage 1: 100-kick stability
        for step in range(100):
            xi, yi = windows(sdata, CTX, step * B, B, 1)
            ok = mu_train_step(net, onehot(xi, V).to(DEV), yi.to(DEV), o,
                               n_steps, gamma, mode)
            if not ok:
                break
        wiki = val_bpc(net, val[BASE_DOMAIN][0].to(DEV), val[BASE_DOMAIN][1].to(DEV), V)
        shak = val_bpc(net, val["shakespeare"][0].to(DEV), val["shakespeare"][1].to(DEV), V)
        drift = wiki - b_wiki
        survived = ok and math.isfinite(wiki) and drift < 0.5
        row = {"config": tag, "finite": ok, "wiki_100": round(wiki, 3),
               "shak_100": round(shak, 3), "drift_100": round(drift, 3),
               "survived_stage1": survived}
        print(f"  [{tag}] stage1: finite={ok} wiki={wiki:.3f} (drift {drift:+.3f}) "
              f"shak={shak:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if survived:                              # stage 2: 600-kick stream vs cascade ref
            for step in range(100, 600):
                xi, yi = windows(sdata, CTX, step * B, B, 1)
                if not mu_train_step(net, onehot(xi, V).to(DEV), yi.to(DEV), o,
                                     n_steps, gamma, mode):
                    row["stage2_diverged"] = True
                    break
            s2_shak = val_bpc(net, val["shakespeare"][0].to(DEV),
                              val["shakespeare"][1].to(DEV), V)
            s2_wiki = val_bpc(net, val[BASE_DOMAIN][0].to(DEV),
                              val[BASE_DOMAIN][1].to(DEV), V)
            row.update({"shak_600": round(s2_shak, 3), "wiki_600": round(s2_wiki, 3),
                        "gap_to_cascade": round(s2_shak - r_shak, 3)})
            print(f"  [{tag}] stage2: shak={s2_shak:.3f} (cascade {r_shak:.3f}, "
                  f"gap {s2_shak-r_shak:+.3f}) wiki={s2_wiki:.3f}", flush=True)
        rows.append(row)

    rows.append({"config": "cascade600_ref", "shak_600": round(r_shak, 3),
                 "wiki_600": round(r_wiki, 3),
                 "base_wiki": round(b_wiki, 3), "base_shak": round(b_shak, 3)})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ugate_probe.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    go = [r for r in rows if r.get("survived_stage1") and
          not r.get("stage2_diverged") and abs(r.get("gap_to_cascade", 9)) <= 0.15]
    print(f"\nU-GATE verdict: {'GO — ' + go[0]['config'] if go else 'not yet'} "
          f"(criteria: finite, drift<0.5, |gap to cascade|<=0.15)", flush=True)


if __name__ == "__main__":
    main()
