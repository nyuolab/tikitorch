"""E6 screen comparator — walks <root>/<arm>/s<seed>/{tele.jsonl,
probes.jsonl,ledger/} (the S3 mirror layout), computes the pre-registered
endpoints (st7_verdicts) per life, aggregates per arm, and prints deltas vs
the 'base' arm. Output: table + <root>/e6_screen_endpoints.jsonl.

Usage: python st7_screen.py --root DIR [--base base]
"""
import argparse, glob, json, os

from st7_verdicts import (load_jsonl, era_series, ledger_events,
                          retain_era_mean, retain_rotation_aligned,
                          arrival_ignition, seesaw_amplitude, pih_floor)


def life_endpoints(d):
    tele = load_jsonl(os.path.join(d, "tele.jsonl"))
    probes = load_jsonl(os.path.join(d, "probes.jsonl"))
    led = ledger_events(os.path.join(d, "ledger"))
    if not tele:
        return None
    eras = era_series(tele)
    rem = retain_era_mean(eras)
    rra = retain_rotation_aligned(eras)
    ai = arrival_ignition(eras)
    sa = seesaw_amplitude(probes)
    pf = pih_floor(tele).get("piH", {})
    # late-life resident returns: mean of each env's final full era
    late = {}
    for e, v in sorted(eras.items()):
        if len(v) >= 0.95 * max(len(x) for x in eras.values()):
            late[v[0]["env"]] = round(sum(x["ret1k"] for x in v) / len(v), 2)
    bad = [e for e in led if any(k in str(e.get("event", "")).upper()
                                 for k in ("NAN", "POISON", "INCIDENT"))]
    return dict(
        retain_em={env: f"{r['passes']}/{r['total']}"
                   for env, r in rem.items() if r["total"]},
        retain_rot={env: (sum(r["passes"] for r in rows),
                          sum(r["total"] for r in rows))
                    for env, rows in rra.items()},
        ignite=ai.get("ignite"), ignite_era=ai.get("first_ignite_era"),
        ignite_max=ai.get("max_era_mean"),
        seesaw_amp=sa["mean_amp"],
        nonres=sa["nonresident_mean"],
        pih_frac={env: r["frac_collapsed"] for env, r in pf.items()},
        pih_collapse={env: r["first_unrecovered_collapse_era"]
                      for env, r in pf.items()},
        late_ret=late,
        incidents=len(bad),
        n_suites=len(probes), n_buckets=len(tele),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", default="base")
    args = ap.parse_args()
    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(arm_dir):
            continue
        arm = os.path.basename(arm_dir)
        for sd in sorted(glob.glob(os.path.join(arm_dir, "s*"))):
            ep = life_endpoints(sd)
            if ep is None:
                print(f"!! {arm}/{os.path.basename(sd)}: no tele — skipped")
                continue
            rows.append(dict(arm=arm, seed=os.path.basename(sd), **ep))
    out = os.path.join(args.root, "e6_screen_endpoints.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def agg(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    base = arms.get(args.base, [])
    base_amp = agg([r["seesaw_amp"] for r in base])
    base_late = {}
    for r in base:
        for env, v in r["late_ret"].items():
            base_late.setdefault(env, []).append(v)
    base_late = {e: agg(v) for e, v in base_late.items()}

    print(f"{'arm':14s} {'n':>2s} {'seesaw':>8s} {'dAmp%':>7s} "
          f"{'ignite':>7s} {'pih_frac(min)':>14s} {'late_ret (d% vs base)':>28s} "
          f"{'inc':>4s}")
    for arm, rs in sorted(arms.items()):
        amp = agg([r["seesaw_amp"] for r in rs])
        d_amp = (round(100 * (amp - base_amp) / base_amp, 1)
                 if base_amp and amp is not None else None)
        ign = sum(bool(r["ignite"]) for r in rs)
        pf = min((min(r["pih_frac"].values()) for r in rs
                  if r["pih_frac"]), default=None)
        lr_parts = []
        for env in sorted(base_late):
            vals = agg([r["late_ret"].get(env) for r in rs])
            if vals is not None and base_late[env]:
                lr_parts.append(
                    f"{env[:4]}:{vals} ({round(100 * (vals - base_late[env]) / base_late[env], 1):+}%)")
        inc = sum(r["incidents"] for r in rs)
        print(f"{arm:14s} {len(rs):>2d} {amp!s:>8s} {d_amp!s:>7s} "
              f"{ign}/{len(rs):<5d} {pf!s:>14s} {' '.join(lr_parts):>28s} "
              f"{inc:>4d}")
    print(f"\nwrote {out} ({len(rows)} lives)")


if __name__ == "__main__":
    main()
