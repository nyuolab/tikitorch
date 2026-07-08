#!/usr/bin/env python3
# compare_fwd.py — per-stage bit-level diff of the S2 PC-transformer FORWARD activations:
#   silicon dumps (fwd_<chip>_<name>.f32 from forge_pc_xformer_s2_shard.cpp FWD_DUMP=1, matched init + FIXEDOFF=0)
#   vs the bit-faithful shim's source dumps (s2_twochip_shim.cpp FWD_DUMP=1, same init + FIXEDOFF=0).
#
# For each chip/stage it prints: n, maxabs(sil), maxabs(shim), relerr (L2 of the diff / L2 of source),
# max per-element abs delta, cosine, exact-bit-match %, and flags the FIRST stage (in forward order) whose
# relerr jumps out of fp-noise (default 1e-4) -> that localizes the first broken forward op. Under the
# DDR-streaming hypothesis, R should be maxabs-faithful (relay-in) but the first STREAMED-op stage
# (X1=LN, or Q/K/V=gemm, or ATTN, or HD=FFN gemm) is where silicon drifts from source.
#
#   usage: python3 compare_fwd.py <SIL_DIR> <SHIM_DIR> [--chips 0,1] [--thresh 1e-4]
#   e.g.:  python3 compare_fwd.py ./fwd_sil ./fwd_shim
import sys, os, argparse
import numpy as np

# forward order per chip (the kernel's fwd order); XF/LG only exist on the OUTPUT chip.
ORDER      = ["R","X1","Q","K","V","ATTN","M","X2","HD","RP"]
ORDER_OUT  = ORDER + ["XF","LG"]
STAGE_DESC = {
    "R":   "block input R (embed+pos or relayed down-msg; NOT a streamed compute)",
    "X1":  "LN1 out            (layernorm, DDR-streamed)",
    "Q":   "WQ proj + qact     (gemm, DDR-streamed weights)",
    "K":   "WK proj + qact     (gemm, DDR-streamed weights)",
    "V":   "WV proj + qact     (gemm, DDR-streamed weights)",
    "ATTN":"attention + qact   (scores from DDR-cached Q,K; P@V)",
    "M":   "residual mid M=R+AO(WO gemm + add, DDR-streamed)",
    "X2":  "LN2 out            (layernorm, DDR-streamed)",
    "HD":  "FFN hidden         (WUP gemm + relu + qact, DDR-streamed)",
    "RP":  "final residual R'=M+FO (WDN gemm + add, DDR-streamed)",
    "XF":  "LNF out            (OUTPUT: final layernorm)",
    "LG":  "logits             (OUTPUT: LM head gemm)",
}

def load(d, chip, name):
    p = os.path.join(d, f"fwd_{chip}_{name}.f32")
    if not os.path.exists(p): return None
    return np.fromfile(p, dtype=np.float32)

def stats(sil, shim):
    n = sil.size
    ds = sil.astype(np.float64); dh = shim.astype(np.float64)
    diff = ds - dh
    l2src = np.linalg.norm(dh)
    relerr = float(np.linalg.norm(diff) / l2src) if l2src > 0 else float(np.linalg.norm(diff))
    maxad  = float(np.abs(diff).max())
    denom  = (np.linalg.norm(ds) * np.linalg.norm(dh))
    cos    = float(np.dot(ds, dh) / denom) if denom > 0 else 1.0
    # exact bit match: reinterpret both as uint32 and compare
    bit = float((sil.view(np.uint32) == shim.view(np.uint32)).mean() * 100.0)
    return dict(n=n, mxs=float(np.abs(sil).max()), mxh=float(np.abs(shim).max()),
                relerr=relerr, maxad=maxad, cos=cos, bit=bit)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sil_dir"); ap.add_argument("shim_dir")
    ap.add_argument("--chips", default="0,1")
    ap.add_argument("--thresh", type=float, default=1e-4,
                    help="relerr above this = 'diverged' (fp-noise floor is ~1e-6..1e-7 for bit-faithful)")
    a = ap.parse_args()
    chips = [int(c) for c in a.chips.split(",") if c != ""]

    print(f"# silicon dir : {a.sil_dir}")
    print(f"# shim   dir : {a.shim_dir}")
    print(f"# divergence threshold (relerr): {a.thresh:.1e}\n")
    hdr = f"{'chip':>4} {'stage':<5} {'n':>8} {'maxabs_sil':>12} {'maxabs_shim':>12} {'relerr':>10} {'max_abs_d':>11} {'cosine':>10} {'bitmatch%':>9}  note"
    print(hdr); print("-"*len(hdr))

    first_div = {}   # chip -> (stage, relerr)
    any_missing = False
    for chip in chips:
        order = ORDER_OUT if chip == max(chips) else ORDER   # OUTPUT chip = highest index
        for name in order:
            sil = load(a.sil_dir, chip, name); shim = load(a.shim_dir, chip, name)
            if sil is None or shim is None:
                miss = "silicon" if sil is None else "shim"
                print(f"{chip:>4} {name:<5} {'--':>8} {'--':>12} {'--':>12} {'--':>10} {'--':>11} {'--':>10} {'--':>9}  MISSING ({miss} dump absent)")
                any_missing = True; continue
            if sil.size != shim.size:
                print(f"{chip:>4} {name:<5} SIZE MISMATCH sil={sil.size} shim={shim.size} -- SKIP")
                any_missing = True; continue
            s = stats(sil, shim)
            note = ""
            if chip not in first_div and s["relerr"] > a.thresh:
                first_div[chip] = (name, s["relerr"]); note = "<== FIRST DIVERGENCE"
            print(f"{chip:>4} {name:<5} {s['n']:>8} {s['mxs']:>12.4e} {s['mxh']:>12.4e} "
                  f"{s['relerr']:>10.2e} {s['maxad']:>11.2e} {s['cos']:>10.6f} {s['bit']:>9.2f}  {note}")
        print()

    print("="*60)
    for chip in chips:
        if chip in first_div:
            nm, re = first_div[chip]
            print(f"chip {chip}: FIRST DIVERGING STAGE = {nm}  (relerr {re:.2e})  --  {STAGE_DESC.get(nm,'')}")
        else:
            print(f"chip {chip}: no stage exceeded relerr {a.thresh:.1e} (silicon == source to fp-noise on every dumped stage)")
    if any_missing:
        print("\nNOTE: some dumps were missing/mismatched; run BOTH silicon (FWD_DUMP=1) and the shim (FWD_DUMP=1)")
        print("      at the SAME matched init + FIXEDOFF=0 so every fwd_<chip>_<name>.f32 exists on both sides.")
    print("\nInterpretation: the first stage whose relerr jumps from ~fp-noise (<=1e-6) to ~1e-2 is the broken op.")
    print("If R matches but X1/Q/ATTN/HD is first to drift -> the LN/qact/gemm DDR-STREAM read/write is the bug.")

if __name__ == "__main__":
    main()
