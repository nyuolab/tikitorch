"""lm_energy_projection.py — op-count + ENERGY PROJECTION for the off-GPU tropical LM vs the
dense (NanoGPT) baseline. This is the quantitative core of the NanoGPT comparison: bits-per-byte
alone says "dense wins by ~0.4"; the tropical pitch is "give up ~0.4 bpb to ELIMINATE the entire
multiplier datapath," which only shows up in energy.

HONEST SCOPE: this is a PROJECTION from per-op energy figures, NOT a measured FPGA number. The
*measured* fact is that the tropical DEPLOY forward is fully multiplier-free (l1 score + argmax-gather +
dyadic linears + fine-dyadic norm + ReLU MLP + dyadic head): every multiply in the dense model is
replaced by a shift+add. The energy RATIO depends on the per-op assumptions below; we report several.
A real number needs HLS synthesis of the LM (cf. the project's morph-LM / MuZero F2 deployments).

Per-op energy (pJ). Defaults = Horowitz, "Computing's Energy Problem", ISSCC 2014, 45nm:
  fp32 mult 3.7, fp32 add 0.9 ; int32 mult 3.1, int32 add 0.1 ; int8 mult 0.2, int8 add 0.03.
A barrel-shift / abs / compare ~ an int add (we charge it as such; on an FPGA a constant shift is
free routing). Tropical replaces each MAC (mult+add) with a shift+add (= ~2 adds worth, charged as
add+add). The dense softmax exp is charged as ~5 adds (a small LUT/poly on fabric).
"""
import argparse


def transformer_macs(d, layers, heads, seq, vocab, d_ff_mult=4):
    """Multiply-accumulates per GENERATED TOKEN (incremental decode, KV-cached) for a decoder LM.
    Per layer: QKVO = 4 d^2 ; MLP = 2 * d * (d_ff_mult*d) = 8 d^2 ; attention (vs seq keys) =
    scores d + weighted-sum d, per key => 2 * seq * d. Plus final unembedding d*vocab (once)."""
    d_ff = d_ff_mult * d
    per_layer_proj = 4 * d * d              # Q,K,V,O
    per_layer_mlp = 2 * d * d_ff           # up + down
    per_layer_attn = 2 * seq * d           # QK scores + softmax@V over `seq` cached keys
    per_layer = per_layer_proj + per_layer_mlp + per_layer_attn
    total = layers * per_layer + d * vocab
    return {
        "proj": layers * per_layer_proj, "mlp": layers * per_layer_mlp,
        "attn": layers * per_layer_attn, "unembed": d * vocab, "total_mac": total,
    }


def project(cfg, e, label):
    d, L, h, seq, V = cfg["d"], cfg["layers"], cfg["heads"], cfg["seq"], cfg["vocab"]
    mac = transformer_macs(d, L, h, seq, V, cfg.get("d_ff_mult", 4))
    n_mac = mac["total_mac"]
    n_softmax = L * seq                                  # exp evaluations / token (one per attn key)

    # DENSE: each MAC = 1 mult + 1 add ; softmax exp ~5 adds.
    dense_E = n_mac * (e["mult"] + e["add"]) + n_softmax * 5 * e["add"]
    # TROPICAL DEPLOY (fully mult-free): each MAC -> shift+add (charged as 2 adds); l1 score = sub+abs+add
    # (~MAC count, charged as adds); argmax-gather = compares (charged as adds); NO softmax, NO mult.
    trop_E = n_mac * (2 * e["add"])
    mult_elim = n_mac                                    # multiplies removed per token (all of them)
    return {
        "label": label, "d": d, "L": L, "seq": seq, "vocab": V,
        "MAC/token": n_mac, "mult_removed/token": mult_elim,
        "dense_pJ": dense_E, "tropical_pJ": trop_E, "energy_x": dense_E / trop_E,
        "breakdown_MAC": {k: mac[k] for k in ("proj", "mlp", "attn", "unembed")},
    }


CONFIGS = {
    "text8_d384": dict(d=384, layers=6, heads=12, seq=256, vocab=27),          # this session's scale-up
    "gpt2_small": dict(d=768, layers=12, heads=12, seq=1024, vocab=50257),     # NanoGPT GPT-2 (124M) projection
}
ENERGY = {
    "fp32_45nm": {"mult": 3.7, "add": 0.9},      # Horowitz fp32
    "int8_45nm": {"mult": 0.2, "add": 0.03},     # Horowitz int8 (the realistic quantized-inference regime)
    "int32_45nm": {"mult": 3.1, "add": 0.1},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS) + ["all"], default="all")
    p.add_argument("--quality", type=str, default="dense 1.87 bpb vs minimal-multiply l1_gather 2.29 (text8 d384, 1 seed)")
    args = p.parse_args()
    cfgs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    print("LM ENERGY PROJECTION — tropical (off-GPU, mult-free deploy) vs dense (NanoGPT)")
    print("PROJECTION from per-op energy, NOT a measured FPGA number. Quality (measured):")
    print(f"  {args.quality}\n")
    for cname, cfg in cfgs.items():
        print(f"=== {cname}: d{cfg['d']} L{cfg['layers']} h{cfg['heads']} seq{cfg['seq']} vocab{cfg['vocab']} ===")
        for ename, e in ENERGY.items():
            r = project(cfg, e, cname)
            print(f"  [{ename:11s}] MAC/tok={r['MAC/token']:,}  mult_removed/tok={r['mult_removed/token']:,}"
                  f"  dense={r['dense_pJ']/1e6:.2f}uJ  tropical={r['tropical_pJ']/1e6:.2f}uJ"
                  f"  -> {r['energy_x']:.1f}x less compute energy")
        bd = project(cfg, ENERGY["fp32_45nm"], cname)["breakdown_MAC"]
        tot = sum(bd.values())
        print(f"  MAC breakdown: " + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in bd.items()))
        print()
    print("Reads: tropical-deploy removes 100% of the multiplies (all MACs -> shift+add). The energy")
    print("multiple is dominated by the mult:add cost ratio (fp32 ~5x, int8 ~7x at the MAC level); the")
    print("REAL win on FPGA is freeing the DSP multiplier array (LUT shift-adds instead) + no softmax.")
    print("Caveat: a measured ratio needs HLS synthesis of the LM (see morph-LM/MuZero F2 deploys).")


if __name__ == "__main__":
    main()
