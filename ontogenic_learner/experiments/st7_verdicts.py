"""E5 Goal-B verdicts (pre-registered bars, E5_SPEC.md).

Reads the life's tele.jsonl + probes.jsonl + ledger/*.jsonl and computes:
  RETAIN     Q1(revisit) >= 0.8 * Q4(previous visit) per env, with the
             recovery cost (min prefix length, in 1k-step buckets, whose
             prefix-mean reaches the target) NON-INCREASING across cycles
             (Theil-Sen slope <= 0). Implementation choices declared:
             quarter = n/4 buckets of the era's telemetry; cost = smallest
             prefix with prefix-mean >= target (inf if never; capped at
             the era length); the bar's "within the first 25%" = cost <=
             n/4. Quarters map to the biome rotation, so a same-quarter
             robustness table (Q_i vs Q_i) is printed beside the primary.
  NO-RELAPSE no DARKROOM ledger incident lasting >48h without rescue.
  STABILITY  nan/POISON_SKIP count <= 3 (ledger); weight-norm plateau is
             computed separately from checkpoint files (--ckpt-glob).
  TRANSFER   reported as measured: zero-shot era-entry returns per env
             first visit, and the arrival record (era 14).
  OPS        ledger seq strictly monotonic across day files; resume count.

Usage: python st7_verdicts.py --dir <dir with tele.jsonl/probes.jsonl/ledger/>
"""
import argparse, glob, json, math, os


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def theil_sen(ys):
    pts = [(i, y) for i, y in enumerate(ys) if math.isfinite(y)]
    if len(pts) < 2:
        return float("nan")
    slopes = [(y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(pts)
              for (x2, y2) in pts[i + 1:]]
    slopes.sort()
    m = len(slopes)
    return slopes[m // 2] if m % 2 else 0.5 * (slopes[m // 2 - 1] + slopes[m // 2])


def era_series(tele):
    eras = {}
    for r in tele:
        eras.setdefault(r["era"], []).append(r)
    return {e: v for e, v in sorted(eras.items())}


def quarters(buckets):
    n = len(buckets)
    q = max(1, n // 4)
    segs = [buckets[0:q], buckets[q:2 * q], buckets[2 * q:3 * q], buckets[3 * q:]]
    return [sum(x["ret1k"] for x in s) / len(s) if s else float("nan") for s in segs]


def recovery_cost(buckets, target):
    """Smallest prefix (bucket count) whose prefix-mean >= target."""
    if target <= 0:
        return 0
    acc = 0.0
    for i, b in enumerate(buckets, 1):
        acc += b["ret1k"]
        if acc / i >= target:
            return i
    return float("inf")


def retain(eras):
    by_env = {}
    for e, v in eras.items():
        by_env.setdefault(v[0]["env"], []).append((e, v))
    out = {}
    for env, visits in by_env.items():
        rows, costs = [], []
        full = max(len(v) for _, v in visits)
        for (ea, va), (eb, vb) in zip(visits, visits[1:]):
            if len(vb) < 0.95 * full or len(va) < 0.95 * full:
                continue  # in-progress/truncated era: excluded, declared
            qa, qb = quarters(va), quarters(vb)
            target = 0.8 * qa[3]
            vacuous = target <= 0  # zero-return prior visit: bar undefined
            cost = recovery_cost(vb, target)
            n = len(vb)
            ok = qb[0] >= target
            within = cost <= max(1, n // 4)
            rows.append(dict(prev_era=ea, era=eb, prevQ4=round(qa[3], 2),
                             Q1=round(qb[0], 2), target=round(target, 2),
                             pass_q1=ok, cost=cost, within_25pct=within,
                             vacuous=vacuous,
                             sameQ=[round(b - a, 1) for a, b in zip(qa, qb)]))
            if not vacuous:
                costs.append(cost)
        slope = theil_sen([c if math.isfinite(c) else len(visits[0][1]) * 2
                           for c in costs]) if costs else float("nan")
        out[env] = dict(revisits=rows, cost_series=costs,
                        cost_slope=slope,
                        non_increasing=(not math.isnan(slope)) and slope <= 0)
    return out


def ledger_events(ledger_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(ledger_dir, "ledger-*.jsonl"))):
        rows.extend(load_jsonl(p))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    tele = load_jsonl(os.path.join(args.dir, "tele.jsonl"))
    probes = load_jsonl(os.path.join(args.dir, "probes.jsonl"))
    led = ledger_events(os.path.join(args.dir, "ledger"))
    eras = era_series(tele)

    print("=== RETAIN ===")
    ret = retain(eras)
    for env, r in ret.items():
        print(f"[{env}] cost series {r['cost_series']} slope {r['cost_slope']:.3f} "
              f"non-increasing={r['non_increasing']}")
        for row in r["revisits"]:
            print("   ", row)

    print("\n=== NO-RELAPSE ===")
    dark = [e for e in led if "DARK" in str(e.get("event", "")).upper()]
    print(f"DARKROOM incidents: {len(dark)}  -> {'PASS' if not dark else dark}")

    print("\n=== STABILITY (ledger part) ===")
    bad = [e for e in led if any(k in str(e.get("event", "")).upper()
                                 for k in ("NAN", "POISON", "INCIDENT"))]
    print(f"nan/poison/incident events: {len(bad)} (bar <=3) "
          f"-> {'PASS' if len(bad) <= 3 else 'FAIL'}; wnorm plateau from ckpts separately")
    for e in bad:
        print("   ", e)

    print("\n=== TRANSFER (as measured) ===")
    firsts = {}
    for e, v in eras.items():
        firsts.setdefault(v[0]["env"], (e, v))
    for env, (e, v) in firsts.items():
        head = v[:25]
        print(f"[{env}] first era {e}: entry mean ret1k "
              f"{sum(x['ret1k'] for x in head)/len(head):.2f} over first {len(head)}k")

    print("\n=== OPS ===")
    seqs = [e.get("seq") for e in led if "seq" in e]
    mono = all(b > a for a, b in zip(seqs, seqs[1:]))
    resumes = [e for e in led if "RESUME" in str(e.get("event", "")).upper()]
    print(f"ledger events {len(led)}, seq strictly monotonic={mono}, "
          f"resume events={len(resumes)}, probes suites={len(probes)}")


if __name__ == "__main__":
    main()
