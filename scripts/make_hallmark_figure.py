"""The hallmark LM figure: tropical multiplier-free LM vs nanoGPT, head to head.

Panel A — validation loss (bits/byte) vs training tokens, tiny-shakespeare
BPE-1024, all runs measured on the host (MPS).
Panel B — final quality bars with closure level.
Panel C — measured, within-family F2 replica throughput, with timing and
observable-record caveats.

Honesty notes baked into the figure:
- "strict" = Level-B machine-audited multiplier-free TRAINING (forward, loss,
  backward, update) — the same update rule the fabric kernels execute.
- FPGA numbers are configuration-specific. The 2-layer LM kernel's 772 tok/s
  is measured; its ~9.0 mJ/tok is a derived estimate, not measured full-card
  energy, and its negative WNS makes it preliminary. The D128 MorphSSM replica
  family measured 14,236 / 26,522 / 53,001 tok/s at N=1/2/4: 1.86x and 3.72x
  aggregate scaling relative to N=1, about 93% replica efficiency. Its logged
  observable records matched the host references, not every internal signal.
  The headline builds have negative WNS and are not timing-closed release
  artifacts. These within-family ratios are not GPU speedups or parity claims.
  The CE exp2 table is host-staged at a 16-word-aligned offset (the synthesized
  ROM read zero, and a misaligned wide burst read it shifted — see
  fpga/silicon/SILICON_FINDINGS.md). Full-budget from-scratch FPGA training
  at quality is open;
  the deployment story is warm-handoff (host-train strict, fabric serve+tune).

Usage: PYTHONPATH=. python scripts/make_hallmark_figure.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "manuscript" / "results" / "lm_shakespeare"
TOK_PER_STEP = 24 * 96

ARMS = {  # tag -> (label, color, style)
    "bpeA_dense": ("nanoGPT (dense GPT, AdamW)", "#1f77b4", "-"),
    "bpeA_denseconv": ("dense-conv twin (AdamW)", "#17becf", "--"),
    "bpeA_morphssm": ("MorphSSM tropical (float-trained,\ndeploys multiplier-free)", "#ff7f0e", "-"),
    "strictG_bpe_anneal_puretrop": ("MorphSSM STRICT (multiplier-free\ntraining, 1.5k steps)", "#d62728", "--"),
    "strictH_bpe_anneal_s9decay_6k": ("MorphSSM STRICT + lr-decay\n(multiplier-free training, 6k steps)", "#d62728", "-"),
}


def load_curve(tag):
    rows = [json.loads(l) for l in open(OUT / f"{tag}.jsonl") if '"final"' not in l]
    xs = [r["step"] * TOK_PER_STEP / 1e6 for r in rows]
    ys = [r.get("val_bpb", r.get("val_ce", r.get("val_ce_hard", 0)) / math.log(2))
          for r in rows]
    return xs, ys


def main():
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.15], wspace=0.33)

    # ---------------- Panel A: loss curves ----------------
    ax = fig.add_subplot(gs[0])
    for tag, (label, color, style) in ARMS.items():
        xs, ys = load_curve(tag)
        ax.plot(xs, ys, style, color=color, lw=1.8, label=label)
    ax.axhline(3.581, color="gray", lw=0.8, ls=":")
    ax.text(13.6, 3.60, "bigram table", fontsize=7, color="gray", ha="right")
    ax.axhline(3.087, color="gray", lw=0.8, ls=":")
    ax.text(13.6, 3.10, "trigram table", fontsize=7, color="gray", ha="right")
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation loss (bits / byte)")
    ax.set_ylim(2.1, 4.4)
    ax.set_xlim(0, 14)
    ax.set_title("A  Quality head-to-head (tiny-shakespeare, BPE-1024)", fontsize=10, loc="left")
    ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)

    # ---------------- Panel B: final quality bars ----------------
    ax = fig.add_subplot(gs[1])
    bars = [
        ("nanoGPT", 2.273, "#1f77b4", "float\ntrain+deploy"),
        ("conv\ntwin", 2.351, "#17becf", "float\ntrain+deploy"),
        ("Morph\nfloat", 2.498, "#ff7f0e", "mult-free\nDEPLOY"),
        ("STRICT\n6k", 2.793, "#d62728", "mult-free\nTRAINING"),
        ("STRICT\n1.5k", 3.071, "#e377c2", "mult-free\nTRAINING"),
    ]
    xs = range(len(bars))
    for i, (name, v, c, note) in enumerate(bars):
        ax.bar(i, v, color=c, width=0.66)
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i, 0.18, note, ha="center", fontsize=6.0, color="white",
                va="bottom", fontweight="bold")
    ax.axhline(3.087, color="gray", lw=0.8, ls=":")
    ax.text(4.45, 3.12, "trigram", fontsize=6.5, color="gray", ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax.set_ylabel("final val loss (bits / byte)")
    ax.set_ylim(0, 3.6)
    ax.set_title("B  Reported endpoints (unequal token budgets)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)

    # ---------------- Panel C: measured within-family FPGA replication ----------------
    # Ratios use the measured rw64 N=1 run as the denominator. All three builds
    # have small negative WNS, so this is preliminary scaling evidence rather
    # than timing-closed accelerator performance. There is no N=8 measurement.
    ax = fig.add_subplot(gs[2])
    sys_bars = [
        ("N=1", 14236, "1.00x", "100%", "WNS -0.020 ns"),
        ("N=2", 26522, "1.86x", "93.1%", "WNS -0.097 ns"),
        ("N=4\n(SLR-local max)", 53001, "3.72x", "93.1%", "WNS -0.173 ns"),
    ]
    x = list(range(len(sys_bars)))
    tput = [b[1] for b in sys_bars]
    colors = ["#2ca02c", "#2ca02c", "#9467bd"]
    ax.bar(x, tput, width=0.58, color=colors, alpha=0.9,
           label="measured aggregate")
    ideal = [14236, 2 * 14236, 4 * 14236]
    ax.plot(x, ideal, color="#555555", marker="o", ms=3.5, lw=1.1, ls="--",
            label="ideal N x measured N=1")
    for i, (name, t, ratio, efficiency, wns) in enumerate(sys_bars):
        ax.text(i, t + 1700, f"{t:,.0f}\n{ratio}  ({efficiency} eff.)",
                ha="center", fontsize=7.1, fontweight="bold")
        ax.text(i, max(3500, t * 0.45), f"record matched\n{wns}",
                ha="center", va="center", fontsize=5.6, color="white",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in sys_bars], fontsize=7.0)
    ax.set_ylabel("aggregate training throughput (tokens / s)")
    ax.set_ylim(0, 65000)
    ax.set_title("C  D128 F2 replica scaling (preliminary)", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.0, loc="upper left")
    ax.text(0.98, 0.04,
            "Ratios vs measured N=1; all WNS < 0\n"
            "No timing-closed, N=8, GPU-parity, or energy claim",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
            color="#8b0000",
            bbox=dict(facecolor="white", edgecolor="#8b0000", alpha=0.9, lw=0.6))
    ax.grid(axis="y", alpha=0.25)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.29, top=0.86, wspace=0.33)
    fig.suptitle(
        "Multiplier-free (tropical) language modeling vs nanoGPT — "
        "0.5–1M params, tiny-shakespeare BPE-1024; strict arms are Level-B "
        "machine-audited (zero multiplies in forward+loss+backward+update)",
        fontsize=9.5, y=0.97)
    fig.text(
        0.01, 0.02,
        "Provenance: quality curves/bars are measured on MPS; endpoints use unequal "
        "budgets (STRICT 6k: 13.8M tokens; nanoGPT: 3.46M). Dense quality was reproduced "
        "on A10G CUDA (3.5563 vs 3.5561 val CE). Panels A/B compare matched "
        "parameters/data/tokenizer; the strict arms are Level-B audited multiplier-free training.\n"
        "Panel C: D128 MorphSSM fused training (forward + CE loss + backward + SignSGD) "
        "measured on F2 at 14,236 / 26,522 / 53,001 tok/s for N=1/2/4. Aggregate ratios "
        "are 1.86x and 3.72x relative to measured N=1 (~93% replica efficiency).\n"
        "Each instance's logged observable record matched its host reference; this is not "
        "every-internal-signal exactness. Build reports show DSP=0, but WNS is "
        "-0.020 / -0.097 / -0.173 ns, so these are preliminary, non-timing-closed results.\n"
        "The ratios are within-family scaling, not accelerator speedups. No N=8 result, "
        "matched GPU comparison, or full-card FPGA power measurement is claimed.",
        fontsize=6.1, va="bottom")

    out = OUT / "fig_hallmark_lm.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
