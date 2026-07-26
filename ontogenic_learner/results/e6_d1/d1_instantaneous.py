"""E6-D D1 shadowing test — instantaneous per-bucket CE from the cumulative
tele 'ce' field: ce_inst[t] = (ce[t]*t - ce[t-1]*(t-1000)) / 1000.
Committed derivation (provenance rule, PEER_FEEDBACK_20260726)."""
import json, sys, statistics as st

def inst(rows):
    out = []
    for a, b in zip(rows, rows[1:]):
        out.append((b["t"], (b["ce"] * b["t"] - a["ce"] * a["t"]) / (b["t"] - a["t"]),
                    b["ret1k"], b.get("frozen", 0)))
    return out

for path, era, name in ((sys.argv[1], 3, "foraging-revisit"),
                        (sys.argv[2], 5, "minatar-revisit")):
    rows = [json.loads(l) for l in open(path)]
    seq = inst([r for r in rows if r["era"] in (era - 1, era)])
    e = [x for x in seq if x[3] == 1]
    post = [x for x in seq if x[3] == 0 and x[0] > e[-1][0]][:50] if e else []
    fro_ce = [x[1] for x in e]
    print(f"{name}: frozen-window instantaneous CE "
          f"first5={[round(c,2) for c in fro_ce[:5]]} "
          f"last5={[round(c,2) for c in fro_ce[-5:]]} mean={st.mean(fro_ce):.2f}")
    print(f"  post-unfreeze first10 CE={[round(x[1],2) for x in post[:10]]}")
    print(f"  frozen ret1k first/last: {e[0][2]:.0f} / {e[-1][2]:.0f}")
