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


# ---------------------------------------------------------------------------
# E6 endpoints (E6_SPEC; importable — st7_screen.py consumes these)
# ---------------------------------------------------------------------------

def retain_era_mean(eras):
    """P3 primary (fixed metrology): revisit era-mean >= 0.8 * previous
    visit's era-mean, per env. No quarter/game confound by construction."""
    by_env = {}
    for e, v in eras.items():
        by_env.setdefault(v[0]["env"], []).append((e, v))
    out = {}
    for env, visits in by_env.items():
        full = max(len(v) for _, v in visits)
        rows = []
        for (ea, va), (eb, vb) in zip(visits, visits[1:]):
            if len(vb) < 0.95 * full or len(va) < 0.95 * full:
                continue
            ma = sum(x["ret1k"] for x in va) / len(va)
            mb = sum(x["ret1k"] for x in vb) / len(vb)
            rows.append(dict(prev_era=ea, era=eb, prev_mean=round(ma, 2),
                             mean=round(mb, 2), vacuous=(ma <= 0),
                             ok=(mb >= 0.8 * ma)))
        real = [r for r in rows if not r["vacuous"]]
        out[env] = dict(rows=rows,
                        passes=sum(r["ok"] for r in real),
                        total=len(real))
    return out


def retain_rotation_aligned(eras):
    """P3 robustness: per-(env, biome) means compared across consecutive
    visits — kills the biome/game rotation confound by construction.
    Requires the E6 tele 'biome' field; returns {} on E5-era tele."""
    if not any("biome" in r for v in eras.values() for r in v[:1]):
        return {}
    by_env = {}
    for e, v in eras.items():
        by_env.setdefault(v[0]["env"], []).append((e, v))
    out = {}
    for env, visits in by_env.items():
        full = max(len(v) for _, v in visits)
        rows = []
        for (ea, va), (eb, vb) in zip(visits, visits[1:]):
            if len(vb) < 0.95 * full or len(va) < 0.95 * full:
                continue
            def bm(v):
                d = {}
                for x in v:
                    d.setdefault(x.get("biome", -1), []).append(x["ret1k"])
                return {b: sum(xs) / len(xs) for b, xs in d.items()}
            ba, bb = bm(va), bm(vb)
            common = sorted(set(ba) & set(bb))
            per = {b: dict(prev=round(ba[b], 2), now=round(bb[b], 2),
                           ok=(bb[b] >= 0.8 * ba[b]) if ba[b] > 0 else None)
                   for b in common}
            judged = [p for p in per.values() if p["ok"] is not None]
            rows.append(dict(prev_era=ea, era=eb, biomes=per,
                             passes=sum(p["ok"] for p in judged),
                             total=len(judged)))
        out[env] = rows
    return out


def arrival_ignition(eras, bar=1.4):
    """P4: any babyai era with era-mean ret1k >= bar (50% of the E5b birth
    cohort's 2.75)."""
    b = [(e, sum(x["ret1k"] for x in v) / len(v))
         for e, v in eras.items() if v[0]["env"] == "babyai"]
    if not b:
        return dict(applicable=False)
    ign = [(e, round(m, 3)) for e, m in b if m >= bar]
    return dict(applicable=True, bar=bar, ignite=bool(ign),
                first_ignite_era=(ign[0][0] if ign else None),
                max_era_mean=round(max(m for _, m in b), 3),
                era_means={e: round(m, 3) for e, m in b})


def seesaw_amplitude(probes):
    """P1: per suite, mean non-resident probe obs-CE minus mean resident;
    plus the life-mean non-resident CE per env (the consolidation target)."""
    rows, nonres = [], {}
    for s in probes:
        res_env = s.get("env")
        cells = s.get("a", {})
        res = [c["obs_ce"] for k, c in cells.items()
               if k.startswith(res_env)]
        non = [c["obs_ce"] for k, c in cells.items()
               if not k.startswith(res_env)]
        if not res or not non:
            continue
        rows.append(dict(t=s["t"], era=s.get("era"), env=res_env,
                         resident=round(sum(res) / len(res), 3),
                         nonresident=round(sum(non) / len(non), 3),
                         amp=round(sum(non) / len(non)
                                   - sum(res) / len(res), 3)))
        for k, c in cells.items():
            env = k.split(":")[0]
            if env != res_env:
                nonres.setdefault(env, []).append(c["obs_ce"])
    return dict(suites=rows,
                mean_amp=round(sum(r["amp"] for r in rows)
                               / max(len(rows), 1), 3),
                nonresident_mean={e: round(sum(v) / len(v), 3)
                                  for e, v in nonres.items()})


def pih_floor(tele, collapse_eps=0.01, recover_eps=0.1):
    """P6: per-env fraction of 1k-buckets with piH < eps; the era of first
    UNRECOVERED collapse (no later bucket > recover_eps); min era-mean piH.
    piH_act reported beside when present (ent-floor arm)."""
    out = {}
    for key in ("piH", "piH_act"):
        envs = {}
        for r in tele:
            if key in r:
                envs.setdefault(r["env"], []).append(r)
        res = {}
        for env, rows in envs.items():
            vals = [r[key] for r in rows]
            frac = sum(v < collapse_eps for v in vals) / len(vals)
            collapse_era = None
            for i, r in enumerate(rows):
                if r[key] < collapse_eps and all(
                        x[key] <= recover_eps for x in rows[i:]):
                    collapse_era = r["era"]
                    break
            era_means = {}
            for r in rows:
                era_means.setdefault(r["era"], []).append(r[key])
            res[env] = dict(
                frac_collapsed=round(frac, 4),
                first_unrecovered_collapse_era=collapse_era,
                min_era_mean=round(min(sum(v) / len(v)
                                       for v in era_means.values()), 4))
        if res:
            out[key] = res
    return out


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

    print("\n=== E6 ENDPOINTS ===")
    rem = retain_era_mean(eras)
    for env, r in rem.items():
        print(f"[P3 era-mean {env}] {r['passes']}/{r['total']} revisits pass")
    rra = retain_rotation_aligned(eras)
    for env, rows in rra.items():
        tot = sum(r["total"] for r in rows)
        ok = sum(r["passes"] for r in rows)
        print(f"[P3 rotation-aligned {env}] {ok}/{tot} biome-visits pass")
    ai = arrival_ignition(eras)
    if ai.get("applicable"):
        print(f"[P4 ignition] ignite={ai['ignite']} "
              f"first_era={ai['first_ignite_era']} "
              f"max_era_mean={ai['max_era_mean']}")
    sa = seesaw_amplitude(probes)
    print(f"[P1 seesaw] mean_amp={sa['mean_amp']} "
          f"nonresident_mean={sa['nonresident_mean']}")
    pf = pih_floor(tele)
    for key, envs_ in pf.items():
        for env, r in envs_.items():
            print(f"[P6 {key} {env}] frac<0.01={r['frac_collapsed']} "
                  f"unrecovered_collapse_era="
                  f"{r['first_unrecovered_collapse_era']} "
                  f"min_era_mean={r['min_era_mean']}")


if __name__ == "__main__":
    main()
