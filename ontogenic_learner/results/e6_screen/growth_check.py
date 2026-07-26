"""Derivation for the second-half vs first-half era-mean growth rates
quoted in the 2026-07-26 compounding discussion (peer-review provenance
rule: every quoted number has a committed derivation).
Input: the per-life tele mirrors (S3 s<seed>/tele.jsonl per arm).
Caveat (Sol): raw %-slope is fragile across arms with different baselines;
superseded by forward-transfer AUC / normalized competence in E6-D."""
import json, glob, os, sys, statistics as st

D = sys.argv[1] if len(sys.argv) > 1 else "."
out = {}
for arm in ("base", "ent-pull", "ent-floor", "anchor", "head-renorm"):
    growth = {"foraging": [], "minatar": []}
    for p in glob.glob(f"{D}/{arm}/s*/tele.jsonl"):
        rows = [json.loads(l) for l in open(p)]
        eras = {}
        for r in rows:
            eras.setdefault(r["era"], []).append(r)
        for env in growth:
            seq = [st.mean(x["ret1k"] for x in v) for e, v in sorted(eras.items())
                   if v[0]["env"] == env and len(v) >= 200]
            n = len(seq)
            if n >= 4:
                growth[env].append(
                    100 * (st.mean(seq[n//2:]) / max(st.mean(seq[:n//2]), 1e-9) - 1))
    out[arm] = {e: dict(mean=round(st.mean(v), 1), n=len(v),
                        per_seed=[round(x, 1) for x in v])
                for e, v in growth.items() if v}
print(json.dumps(out, indent=1))
