#!/usr/bin/env python3
"""a2_reduce.py — reduce the A2 iso-model serve-vs-update power run.

Joins each arm's 1 Hz Vccint sample log (power_<arm>_r<R>.log) with the arm's
JSON emit (arm_<arm>_r<R>.log) by unix timestamp, and computes per arm+rep:
    mean / median / max Vccint watts (core rail), tok/s, kicks, learn-kicks,
and the cross-arm numbers the paper cites:
    serve-vs-update ratio (total AND idle-subtracted dynamic), and the
    optimizer-inclusive "learning below the instrument floor" delta
    (update_W - serve_W), i.e. the extra core-rail watts the SignSGD step costs.

HONESTY: watts are the Vccint CORE rail read by fpga-describe-local-image -M,
NOT true full-card PMBus. Report as core-rail; a real board-watt number needs
external card instrumentation that has never been done (SILICON_FINDINGS.md).

The sampler wrote, per second:
    TS=<unix>
    <lines from `fpga-describe-local-image -M` matching Vccint|Last measured|...>
We take each "Last measured" line's first numeric value as that second's sample
(mW auto-converted to W when the line says mW), stamped with the most recent TS=.

Usage: a2_reduce.py [results_dir]   (default: results/fpga_silicon/a2power)
"""
import glob
import json
import os
import re
import statistics
import sys

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_power(path):
    """Return list of (ts:int, watts:float) from 'Last measured' lines."""
    samples = []
    cur_ts = None
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("TS="):
                try:
                    cur_ts = int(s[3:].split()[0])
                except ValueError:
                    cur_ts = None
                continue
            if "last measured" in s.lower():
                m = NUM.search(s)
                if not m or cur_ts is None:
                    continue
                v = float(m.group())
                # unit heuristic: explicit mW, or an implausibly large core-rail
                # value (> 1000) that must be milliwatts.
                if "mw" in s.lower() or v > 1000.0:
                    v /= 1000.0
                samples.append((cur_ts, v))
    return samples


def parse_arm(path):
    """Return (t_start, t_end, meta) from an arm_*.log. Prefer the python JSON
    emit (precise kick-loop window + tok/s); fall back to run_timed's
    ARM= T_START / T_END."""
    t_start = t_end = None
    meta = {}
    with open(path, errors="replace") as f:
        txt = f.read()
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("{") and '"arm"' in s:
            try:
                j = json.loads(s)
            except ValueError:
                continue
            meta = j
            t_start = float(j.get("t_start", t_start) or t_start or 0) or t_start
            t_end = float(j.get("t_end", t_end) or t_end or 0) or t_end
    if t_start is None:
        m = re.search(r"T_START=(\d+)", txt)
        if m:
            t_start = float(m.group(1))
    if t_end is None:
        m = re.search(r"T_END=(\d+)", txt)
        if m:
            t_end = float(m.group(1))
    return t_start, t_end, meta


def stat_block(samples, t0, t1):
    """mean/median/max/n of watts within [t0, t1] (inclusive)."""
    if t0 is None or t1 is None:
        w = [v for _, v in samples]
    else:
        lo, hi = min(t0, t1), max(t0, t1)
        w = [v for ts, v in samples if lo <= ts <= hi]
    if not w:
        return None
    return {"mean_w": statistics.mean(w), "median_w": statistics.median(w),
            "max_w": max(w), "n": len(w)}


def main(argv):
    d = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "fpga_silicon", "a2power")
    d = os.path.abspath(d)
    rows = {}  # name -> dict
    for p in sorted(glob.glob(os.path.join(d, "power_*.log"))):
        name = os.path.basename(p)[len("power_"):-len(".log")]  # e.g. serve_r1
        armlog = os.path.join(d, "arm_%s.log" % name)
        samples = parse_power(p)
        t0 = t1 = None
        meta = {}
        if os.path.exists(armlog):
            t0, t1, meta = parse_arm(armlog)
        blk = stat_block(samples, t0, t1)
        if "_r" in name:
            base, rep = name.rsplit("_r", 1)
            rep = "r" + rep
        else:
            base, rep = name, "r1"
        rows[name] = {"arm": base, "rep": rep, "power": blk, "meta": meta,
                      "n_samples": len(samples), "t0": t0, "t1": t1}

    if not rows:
        print("no power_*.log found under %s" % d)
        return 1

    # ---- per arm+rep table ----
    print("=== A2 per arm+rep (Vccint CORE rail — NOT full-card PMBus) ===")
    hdr = ("%-14s %-4s %8s %8s %8s %6s %10s %8s %10s" %
           ("arm", "rep", "mean_W", "med_W", "max_W", "nS", "tok/s", "kicks", "upd_kicks"))
    print(hdr)
    for name in sorted(rows):
        r = rows[name]
        b = r["power"] or {}
        m = r["meta"]
        print("%-14s %-4s %8s %8s %8s %6s %10s %8s %10s" % (
            r["arm"], r["rep"],
            _f(b.get("mean_w")), _f(b.get("median_w")), _f(b.get("max_w")),
            b.get("n", "-"), _f(m.get("tok_s"), 0), m.get("kicks", "-"),
            m.get("n_update_kicks", "-")))

    # ---- aggregate per base-arm (mean of per-rep means) ----
    agg = {}
    for name, r in rows.items():
        if not r["power"]:
            continue
        agg.setdefault(r["arm"], {"w": [], "tok": []})
        agg[r["arm"]]["w"].append(r["power"]["mean_w"])
        tk = r["meta"].get("tok_s")
        if tk:
            agg[r["arm"]]["tok"].append(tk)
    A = {a: {"mean_w": statistics.mean(v["w"]),
             "tok_s": statistics.mean(v["tok"]) if v["tok"] else 0.0,
             "reps": len(v["w"])}
         for a, v in agg.items()}

    print("\n=== A2 aggregate (mean across reps) ===")
    for a in ("idle", "serve", "update", "gated"):
        if a in A:
            print("  %-7s mean_W=%.4f  tok/s=%.0f  (reps=%d)"
                  % (a, A[a]["mean_w"], A[a]["tok_s"], A[a]["reps"]))

    # ---- the paper numbers ----
    summary = {"results_dir": d, "aggregate": A}
    if all(a in A for a in ("idle", "serve", "update")):
        idle_w = A["idle"]["mean_w"]
        serve_w = A["serve"]["mean_w"]
        update_w = A["update"]["mean_w"]
        serve_dyn = serve_w - idle_w
        update_dyn = update_w - idle_w
        learn_cost_w = update_w - serve_w      # cost of applying the optimizer step
        summary["ratios"] = {
            "idle_w": idle_w, "serve_w": serve_w, "update_w": update_w,
            "serve_dyn_w": serve_dyn, "update_dyn_w": update_dyn,
            "update_over_serve_total": _safe(update_w, serve_w),
            "update_over_serve_dynamic": _safe(update_dyn, serve_dyn),
            "learn_cost_w": learn_cost_w,
            "learn_cost_frac_of_serve_dyn": _safe(learn_cost_w, serve_dyn),
        }
        if A["serve"]["tok_s"]:
            summary["ratios"]["serve_mJ_per_tok"] = 1000.0 * serve_w / A["serve"]["tok_s"]
        if A["update"]["tok_s"]:
            summary["ratios"]["update_mJ_per_tok"] = 1000.0 * update_w / A["update"]["tok_s"]
        print("\n=== 'learning below the instrument floor' (core-rail) ===")
        print("  idle=%.4f W  serve=%.4f W  update=%.4f W" % (idle_w, serve_w, update_w))
        print("  update/serve (total)   = %.4f" % summary["ratios"]["update_over_serve_total"])
        print("  update/serve (dynamic, idle-subtracted) = %s"
              % _f(summary["ratios"]["update_over_serve_dynamic"]))
        print("  LEARN COST = update - serve = %+.4f W core-rail (the optimizer-inclusive delta)"
              % learn_cost_w)
        print("  -> as fraction of serve dynamic power: %s"
              % _f(summary["ratios"]["learn_cost_frac_of_serve_dyn"]))
        print("  NOTE: Vccint core rail only; not full-card PMBus. serve is a")
        print("        frozen-mode-3 proxy (incl. backward) — a true fwd-only")
        print("        serve would draw LESS, so this LEARN COST is conservative.")

    out = os.path.join(d, "a2_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote %s" % out)
    return 0


def _f(x, nd=4):
    if x is None:
        return "-"
    try:
        return ("%.*f" % (nd, float(x)))
    except (TypeError, ValueError):
        return str(x)


def _safe(a, b):
    try:
        return float(a) / float(b) if b else float("nan")
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
