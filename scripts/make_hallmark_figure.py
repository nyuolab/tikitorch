"""The hallmark LM figure: tropical multiplier-free LM vs nanoGPT, head to head.

Panel A — validation loss (bits/byte) vs training tokens, tiny-shakespeare
BPE-1024, all runs measured on the host (MPS).
Panel B — final quality bars with closure level.
Panel C — systems: training throughput and energy per token, with provenance
labels (measured-host / measured-silicon / WNS-estimate / CSIM / projected).

Honesty notes baked into the figure:
- "strict" = Level-B machine-audited multiplier-free TRAINING (forward, loss,
  backward, update) — the same update rule the fabric kernels execute.
- FPGA numbers: the 2-layer LM kernel trained on F2 silicon (params evolved
  on-chip, DSP=0); its 772 tok/s is measured, its 9.0 mJ/tok is WNS-derived.
  The fully-tropical morph-LM kernel now TRAINS ON F2 SILICON, MEASURED and
  bit-exact to the host CSIM golden (AFI cl_forge_morph_lm_v6_bufram,
  agfi-006127c2d7aaabb42, v2 buffer-RAM, DSP=0): ~3,732 tok/s, learning
  (acc 0 -> 0.23 in 30 steps), 3 W Vccint core -> ~0.80 mJ/tok core-rail
  (NOT total-card; not directly comparable to the GPU's board power). It runs
  the perceptron loss because the LUT-CE exp2 ROM reads zero on this AFI
  (root-caused; CE needs a ROM-init fix — see fpga/silicon/SILICON_FINDINGS.md);
  the rest of the datapath is bit-exact. Full-budget from-scratch FPGA training
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
OUT = REPO / "results" / "lm_shakespeare"
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
    fig = plt.figure(figsize=(13.5, 4.6))
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
    ax.set_title("B  Final quality & closure level", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)

    # ---------------- Panel C: systems ----------------
    ax = fig.add_subplot(gs[2])
    sys_bars = [
        ("nanoGPT\nA10G CUDA", 107091, "1.46 mJ/tok\nMEASURED (72 W)", "#1f77b4"),
        ("Morph strict\nhost MPS", 1072, "≈23 mJ/tok\n(rough, 25 W)", "#ff7f0e"),
        ("attn-LM\nF2 SILICON", 772, "≈9.0 mJ/tok\n(WNS-derived)", "#2ca02c"),
        ("morph-LM\nF2 SILICON", 3732, "MEASURED\nbit-exact, DSP=0", "#2ca02c"),
    ]
    x = range(len(sys_bars))
    tput = [b[1] for b in sys_bars]
    ax.bar(x, tput, width=0.6, color=[b[3] for b in sys_bars], alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("training throughput (tokens / s, log)")
    for i, (name, t, etxt, c) in enumerate(sys_bars):
        ax.text(i, t * 1.3, f"{t:,.0f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.text(i, t * 0.06, etxt, ha="center", fontsize=6.0, va="top",
                bbox=dict(facecolor="white", alpha=0.8, lw=0))
    ax.set_xticks(list(x))
    ax.set_xticklabels([b[0] for b in sys_bars], fontsize=7.2)
    ax.set_ylim(8, 9e5)
    ax.set_title("C  Systems: throughput & energy (training)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Multiplier-free (tropical) language modeling vs nanoGPT — "
        "0.5–1M params, tiny-shakespeare BPE-1024; strict arms are Level-B "
        "machine-audited (zero multiplies in forward+loss+backward+update)",
        fontsize=9.5, y=1.02)
    fig.text(0.005, -0.04,
             "Provenance: quality curves/bars measured (MPS; dense quality reproduced "
             "bit-close on A10G CUDA: 3.5563 vs 3.5561 val CE). nanoGPT A10G: tok/s and "
             "energy MEASURED (nvidia-smi board power, 5 Hz, active avg 72 W; 1.46-1.49 "
             "mJ/tok active/whole-window bounds; g5.xlarge). F2 throughput measured on "
             "silicon (2-layer attention-LM kernel, on-chip training validated, DSP=0); "
             "9.0 mJ/tok is WNS-derived, not power-rail. Fully-tropical morph-LM kernel "
             "TRAINS ON F2 SILICON (MEASURED, bit-exact to host golden; agfi-006127c2d7aaabb42, "
             "v2 buffer-RAM, DSP=0): ~3,732 tok/s, acc 0->0.23/30 steps, 3 W Vccint core "
             "(~0.80 mJ/tok core-rail, NOT total-card). Perceptron loss (CE exp2 ROM reads zero "
             "on this AFI -- root-caused, see fpga/silicon/SILICON_FINDINGS.md). Morph-MPS energy is a rough package "
             "estimate. Full-budget from-scratch FPGA training at quality remains open "
             "(integer-lattice dichotomy; tikitorch/fpga/LM_FPGA_VS_GPU_PLAN.md); the "
             "deployment design is warm-handoff: host-train strict, stage int16, serve + "
             "short-horizon fine-tune on fabric.",
             fontsize=6.4, va="top")

    out = OUT / "fig_hallmark_lm.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
