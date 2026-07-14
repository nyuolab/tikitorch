"""V4b — the fleet rider: weights-only Fisher merge of two disjoint-shard lives.

Riders A (wiki,docs) and B (shakespeare,code) each lived 2.5M tokens on their
shard (st4_life). This script merges them WEIGHTS-ONLY using each learner's
own Adam second moment as per-parameter importance (the G3-F2 recipe that cut
the fleet-oracle gap 97%):

    theta = (vA * thetaA + vB * thetaB) / (vA + vB)

and probe-evaluates {base, riderA, riderB, naive-average, fisher-merged} on
all four domains. Pass (G3 precedent): merged ~= each solo on ITS OWN shard
while inheriting the other's — the merge gap << naive.

Usage::

    python ontogenic_learner/experiments/st4b_merge.py
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from st1_streams import DOMAINS, OUT, load_assets, load_ckpt   # noqa: E402
from st4_life import probe_windows, run_probes                 # noqa: E402

BASE = REPO / "results/pc_convrnn/ckpts/m1_s1_cascade_s0.pt"
CK = OUT / "ckpts"


def rider_state(tag):
    st = torch.load(CK / f"v4_life_{tag}.pt", map_location="cpu")
    sd = st["model"]
    # opt.state_dict(): param ids follow named_parameters order
    names = list(sd.keys())
    opt_state = st["opt"]["state"]
    v = {}
    for i, name in enumerate(names):
        s = opt_state.get(i, {})
        v[name] = s.get("exp_avg_sq")
    return sd, v


def merge(sd_a, v_a, sd_b, v_b, mode):
    out = {}
    for k in sd_a:
        if mode == "naive" or v_a.get(k) is None or v_b.get(k) is None:
            out[k] = (sd_a[k] + sd_b[k]) / 2.0
        else:
            wa, wb = v_a[k], v_b[k]
            denom = (wa + wb).clamp_min(1e-12)
            out[k] = (wa * sd_a[k] + wb * sd_b[k]) / denom
    return out


def main():
    assets = load_assets()
    probes = probe_windows(assets, DOMAINS, 96)
    sd_a, v_a = rider_state("riderA")
    sd_b, v_b = rider_state("riderB")
    arms = {
        "base": None,
        "riderA": sd_a,
        "riderB": sd_b,
        "naive_avg": merge(sd_a, v_a, sd_b, v_b, "naive"),
        "fisher": merge(sd_a, v_a, sd_b, v_b, "fisher"),
    }
    rows = []
    for arm, sd in arms.items():
        model = load_ckpt(BASE)
        if sd is not None:
            model.load_state_dict(sd)
        model.eval()
        res = run_probes(model, probes)
        row = dict(gate="V4b", arm=arm,
                   **{f"ce_{d}": round(res[d][0], 4) for d in DOMAINS})
        row["ce_mean"] = round(sum(res[d][0] for d in DOMAINS) / len(DOMAINS), 4)
        rows.append(row)
        print(row)
    with (OUT / "v4b_merge.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # verdict: fisher gap vs naive gap on each rider's OWN shard
    by = {r["arm"]: r for r in rows}
    own = {"riderA": ("wiki", "docs"), "riderB": ("shakespeare", "code")}
    for rider, doms in own.items():
        for d in doms:
            solo, fi, na = by[rider][f"ce_{d}"], by["fisher"][f"ce_{d}"], \
                by["naive_avg"][f"ce_{d}"]
            print(f"[V4b] {d:>11} ({rider}): solo {solo:.3f} | fisher "
                  f"{fi:.3f} (gap {fi-solo:+.3f}) | naive {na:.3f} "
                  f"(gap {na-solo:+.3f})")


if __name__ == "__main__":
    main()
