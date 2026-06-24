"""Build the distributed-PC scaling-law table/figure from the F2 sweep logs (train_L<n>.txt the runner emits)
plus the honest GPU references. Parses throughput / comms / per-chip kick / accuracy out of each chip-count run.

MEASURED:  distributed correctness, per-chip kick times, comms bytes/step, steps/s, accuracy, Gate-C.
PROJECTED: pipeline-overlap speedup, DSP-adder array, multi-F2 farm (clearly labelled, never mixed in).

  python scripts/pc_scaling_law.py results/predictive_coding/shard_train   # dir of train_L*.txt
"""
import os, re, sys, glob, json

A10G_NANOGPT_TOKS = 107091     # measured, A10G nanoGPT (tok/s)   [reference]
A10G_MJ_PER_TOK   = 1.46
MORPH_F2_N4_TOKS  = 53001      # measured, morph-LM N=4 replicated on F2 (tok/s) [reference]


def parse(path):
    t = open(path).read()
    g = lambda rx: (re.search(rx, t) or [None, None])[1]
    d = {}
    m = re.search(r"(\d+)-layer net MODEL-PARALLEL across (\d+) FPGAs", t)
    if m: d["L"] = int(m.group(1)); d["chips"] = int(m.group(2))
    d["residual"] = "[RESIDUAL skips]" in t
    sps = g(r"([\d.]+) steps/s")
    d["steps_per_s"] = float(sps) if sps else None
    bps = g(r"(\d+) bytes/step")
    d["comms_bytes_per_step"] = int(bps) if bps else None
    sps_w = re.search(r"cross-chip comms ([\d.]+)%", t) or re.search(r"cross-chip-comms ([\d.]+)%", t)
    d["comms_pct"] = float(sps_w.group(1)) if sps_w else None
    d["acc_traj"] = [float(x) for x in re.findall(r"epoch \d+: test acc = ([\d.]+)", t)]
    d["acc_peak"] = max(d["acc_traj"]) if d["acc_traj"] else None
    d["kicks"] = re.findall(r"chip (\d+) \(slot \d+\): fwd-kick ([\d.]+) ms, bwd-kick ([\d.]+) ms", t)
    gc = re.search(r"WEIGHTS crossed chips=(\d+) bytes \| GRADIENTS crossed=(\d+)", t)
    d["gateC"] = (int(gc.group(1)), int(gc.group(2))) if gc else None
    return d


def main(d):
    runs = []
    for p in sorted(glob.glob(os.path.join(d, "train_L*.txt"))):
        try:
            runs.append(parse(p))
        except Exception as e:
            print(f"skip {p}: {e}")
    runs = [r for r in runs if r.get("chips")]
    runs.sort(key=lambda r: r["chips"])
    print("\n## Distributed PC scaling-law (MEASURED on F2)\n")
    print("| chips | residual | peak acc | acc trajectory | steps/s | comms B/step | comms % | Gate-C (W,G) |")
    print("|------:|:--------:|---------:|----------------|--------:|-------------:|--------:|:------------:|")
    for r in runs:
        print(f"| {r['chips']} | {'yes' if r['residual'] else 'no'} | "
              f"{r['acc_peak'] if r['acc_peak'] is not None else '-'} | "
              f"{r['acc_traj']} | {r['steps_per_s']} | {r['comms_bytes_per_step']} | "
              f"{r['comms_pct']} | {r['gateC']} |")
    print(f"\nReferences (MEASURED, not this work): A10G nanoGPT {A10G_NANOGPT_TOKS} tok/s @ {A10G_MJ_PER_TOK} mJ/tok; "
          f"morph-LM N=4 on F2 {MORPH_F2_N4_TOKS} tok/s.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        chips = [r["chips"] for r in runs]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(chips, [r["acc_peak"] for r in runs], "o-"); ax[0].set(title="Fashion-MNIST peak acc vs chips", xlabel="physical FPGAs", ylabel="test acc")
        ax[1].plot(chips, [r["comms_bytes_per_step"] for r in runs], "s-"); ax[1].set(title="cross-chip comms / step", xlabel="physical FPGAs", ylabel="bytes/step")
        ax[2].plot(chips, [r["steps_per_s"] for r in runs], "^-"); ax[2].set(title="throughput (serial host relay)", xlabel="physical FPGAs", ylabel="steps/s")
        for a in ax: a.grid(alpha=0.3)
        fig.tight_layout(); out = os.path.join(d, "pc_scaling_law.png"); fig.savefig(out, dpi=110)
        print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"(no figure: {e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/predictive_coding/shard_train")
