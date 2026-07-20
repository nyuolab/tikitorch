"""Goal-B forgetting-at-revisit, derived offline from v4_probes_<tag>.jsonl.

Forgetting is NOT logged. This maps era_end_<segidx> -> domain via the rotating
life_schedule, then per domain compares probe CE at each revisit to the best
end-of-own-era CE (pass = non-increasing after the anchor engages). Also reports
the birth->final generalist trade and the norm slope.

Usage: python forget_analysis.py <tag> [<tag> ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
from st4_life import life_schedule                                  # noqa: E402
from st1_streams import OUT                                         # noqa: E402


def verdict(tag):
    for l in (OUT / "v4_life.jsonl").open():
        r = json.loads(l)
        if r.get("tag") == tag:
            last = r
    return last  # last matching row


def analyze(tag):
    v = verdict(tag)
    if v is None:
        return None
    domains = tuple(v["domains"].split(","))
    sched = life_schedule(domains, v["era_tokens"], v["cycles"])    # seg -> (domain, _)
    seg_domain = [s[0] for s in sched]
    rows = [json.loads(l) for l in (OUT / f"v4_probes_{tag}.jsonl").open()]

    birth = next((r for r in rows if r.get("where") == "birth"), None)
    era_rows = [r for r in rows if str(r.get("where", "")).startswith("era_end_")]

    own_best = {d: None for d in domains}        # best CE at end of d's own era
    forget = {d: [] for d in domains}            # (revisit CE - own_best) trajectory
    for r in era_rows:
        seg = int(r["where"].split("_")[-1])
        trained = seg_domain[seg]
        ce_now = r.get(f"ce_{trained}")
        if ce_now is None:
            continue
        if own_best[trained] is None or ce_now < own_best[trained]:
            own_best[trained] = ce_now
        # forgetting = how much each domain has drifted above its own-era best
        for d in domains:
            if own_best[d] is not None and f"ce_{d}" in r:
                forget[d].append(round(r[f"ce_{d}"] - own_best[d], 3))

    final = era_rows[-1] if era_rows else {}
    mean_final = (sum(final.get(f"ce_{d}", 0) for d in domains) / len(domains)
                  if final else None)
    mean_birth = (sum(birth.get(f"ce_{d}", 0) for d in domains) / len(domains)
                  if birth else None)
    # peak forgetting per domain (max drift above own-era best, over the life)
    peak_forget = {d: (max(forget[d]) if forget[d] else None) for d in domains}
    norm0 = birth["norm"] if birth else None
    norm1 = final.get("norm")
    return dict(v=v, domains=domains, mean_birth=mean_birth, mean_final=mean_final,
                peak_forget=peak_forget, norm0=norm0, norm1=norm1,
                agree_final={d: final.get(f"agree_{d}") for d in domains})


if __name__ == "__main__":
    tags = sys.argv[1:] or ["main"]
    print(f"\n{'TAG':22s} {'HC':>6s} {'birth→final':>13s} {'peak-forget (max drift/dom)':>30s} {'norm':>12s}")
    print("-" * 96)
    for tag in tags:
        a = analyze(tag)
        if a is None:
            print(f"{tag:22s}  (no data)"); continue
        hc = a["v"].get("headroom_captured")
        pf = " ".join(f"{d[:4]}:{a['peak_forget'][d]}" for d in a["domains"])
        trade = (f"{a['mean_birth']:.2f}→{a['mean_final']:.2f}"
                 if a["mean_birth"] and a["mean_final"] else "n/a")
        norm = f"{a['norm0']:.0f}→{a['norm1']:.0f}" if a["norm0"] and a["norm1"] else "n/a"
        hcs = f"{hc:.3f}" if hc is not None else "  n/a"
        print(f"{tag:22s} {hcs:>6s} {trade:>13s}   {pf:<30s} {norm:>12s}")
    print("-" * 96)
    print("peak-forget = worst probe-CE drift above each domain's own-era best "
          "(lower = better retention). birth→final = generalist trade (mean probe CE).")
