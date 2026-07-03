"""Lifetime report: probes.jsonl + ledger + samples -> report.md + PNG plots.
Usage: python -m ontogenic_learner.organism.report <life_dir> [--out report_dir]"""
import argparse
import glob
import gzip
import json
import os


def load_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("life_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.life_dir, "report")
    os.makedirs(out, exist_ok=True)

    probes = []
    pp = os.path.join(a.life_dir, "probes", "probes.jsonl")
    if os.path.exists(pp):
        probes = load_jsonl(pp)
    ledger = []
    for p in sorted(glob.glob(os.path.join(a.life_dir, "ledger", "ledger-*"))):
        ledger.extend(load_jsonl(p))

    # ---------------- plots
    plots = []
    if probes:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [r["life_step"] for r in probes]
        fig, ax = plt.subplots(figsize=(7, 4))
        for d, c in [("wiki", "#1f77b4"), ("shakespeare", "#2ca02c"),
                     ("code", "#d62728"), ("docs", "#9467bd"), ("digits", "#ff7f0e")]:
            k = f"bpc_{d}"
            ys = [r.get(k) for r in probes]
            if any(y is not None for y in ys):
                ax.plot(steps, ys, "o-", ms=3, label=d, color=c)
        ax.set_xlabel("life_step"); ax.set_ylabel("held-out bpc")
        ax.legend(fontsize=8); ax.set_title("lifetime quality per domain")
        fig.tight_layout(); fig.savefig(os.path.join(out, "bpc.png"), dpi=120)
        plots.append("bpc.png")

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(steps, [r["gate"]["rolling_rate"] for r in probes], "o-", ms=3,
                label="rolling fast rate")
        ax.plot(steps, [r["plasticity"]["adapt_gain"] for r in probes], "s-", ms=3,
                label="clone plasticity gain (bpc)")
        ax.set_xlabel("life_step"); ax.legend(fontsize=8)
        ax.set_title("gate rate + plasticity over life")
        fig.tight_layout(); fig.savefig(os.path.join(out, "gate.png"), dpi=120)
        plots.append("gate.png")

    # ---------------- markdown
    lines = ["# Organism lifetime report", ""]
    incidents = [e for e in ledger if e.get("event") == "INCIDENT"]
    resumes = [e for e in ledger if e.get("event") == "RESUME"]
    feeds = [e for e in ledger if e.get("event") == "FEED"]
    fast = sum(1 for e in ledger if e.get("event") == "FAST")
    sleeps = sum(e.get("kicks", 0) for e in ledger if e.get("event") == "SLEEP")
    seqs = [e["seq"] for e in ledger]
    monotonic = all(b > a for a, b in zip(seqs, seqs[1:]))   # gaps = crash losses (ok);
    gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b - a > 1)  # duplicates = bug (not ok)
    lines += [f"- ledger events: {len(ledger)} (monotonic: {monotonic}, gaps: {gaps})",
              f"- fast kicks: {fast} · sleep kicks: {sleeps}",
              f"- resumes: {len(resumes)} · incidents: {len(incidents)} · feeds: {len(feeds)}",
              f"- probes: {len(probes)}", ""]
    if probes:
        last = probes[-1]
        lines += ["## Latest probe", "",
                  "| domain | bpc |", "|---|---|"]
        for d in ["wiki", "shakespeare", "code", "docs", "digits"]:
            lines.append(f"| {d} | {last.get(f'bpc_{d}', '—')} |")
        lines += ["", f"plasticity: {json.dumps(last['plasticity'])}",
                  f"gate: {json.dumps(last['gate'])}", ""]
    for p in plots:
        lines.append(f"![{p}]({p})")
    samples = sorted(glob.glob(os.path.join(a.life_dir, "probes", "samples-*.json")))
    if samples:
        s = json.load(open(samples[-1]))
        lines += ["", "## Latest samples", "", "```", s["text"][:400], "```", "",
                  "digit 3 (conditional):", "```", s["digits"].get("3", ""), "```"]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}/report.md ({len(probes)} probes, {len(ledger)} ledger events, "
          f"monotonic={monotonic}, gaps={gaps})")


if __name__ == "__main__":
    main()
