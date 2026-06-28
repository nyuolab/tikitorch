# BF16 Model-Parallel PC-Transformer Kernel — Build Status

**Kernel:** `forge_pc_xformer_mp_block.cpp` (BF16 fwd + bwd + dyadic-Adam, model-parallel block, no-global-sync)
**Date:** 2026-06-28
**Verdict:** **NEEDS-CSYNTH-ON-VITIS** — the kernel is WRITTEN and csim-MATCHED to the verified BF16 golden; the 1-SLR fit is ESTIMATED-tight (BRAM-bound), not measured. Real Vitis csynth on EC2 is the gate before the ~5h AFI.

---

## 1. Executive summary

The BF16 model-parallel kernel is **(a) written, (b) csim-matched to the verified BF16 golden, (c) NOT yet measured for 1-SLR fit** — it is *estimated* to fit one SLR but tightly, with **BRAM (not LUT/DSP) as the swing factor**. The arithmetic is done; the synthesis gate is the only thing standing between here and AFI.

| State | Result | Evidence class |
|---|---|---|
| (a) WRITTEN | YES — kernel + config + 2 testbenches + synth tcl all present | files on disk |
| (b) CSIM-MATCHED to BF16 golden | YES — fwd bit-exact, bwd within 1 bf16 ULP | **MEASURED (local g++ csim)** |
| (c) FITS ONE SLR | **TIGHT** — estimated, BRAM-bound | **ESTIMATED (no Vitis here)** |

---

## 2. The kernel (WRITTEN)

`forge_pc_xformer_mp_block.cpp` implements one model-parallel PC-transformer block end-to-end on fabric:
- **Deploy int16 forward** (the production multiplier-free path, dyadic shift-add).
- **Wide BF16 forward-for-backward** (fp32 accumulation, bf16 round at tagged points) — the cache the backward consumes.
- **BF16 backward** producing all weight-grads (`gWQ/gWK/gWV/gWO/gWUP/gWDN/gWHEAD`) and activation-grads (`gQ/gK/gV/gR`), plus the inter-chip up-message `gRout`.
- **Dyadic-Adam optimizer step** (mult-free per-param 1/|g| pow2 shift, dyadic m/v EMA) updating a float master, with the int16 deploy weights re-derived.

The model-parallel contract is preserved: the only cross-chip traffic is the neighbour message `gRout` (down) and the activation `R` (up) — **no global gradient, no global sync.**

Supporting files (canonical tree `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/`):
- `forge_pc_xformer_mp_block.cpp` — kernel. Non-debug/synth build is byte-unchanged; a `-DMP_DEBUG_GRADS` macro-guarded gradient-dump region was added for the gradcheck tb only.
- `forge_pc_xformer_mp_config.hpp` — config; an `MP_DG_*` reserved debug-dump region was appended (`MP_BUFLEN` grows; base `XO_*` and int layout untouched).
- `forge_pc_xformer_mp_block_tb.cpp` — main tb (post-optimizer weight check).
- `forge_pc_xformer_mp_gradcheck_tb.cpp` — **NEW** direct raw-gradient KAT tb (all-ones + random-warmed).
- `synth_pc_xformer_mp_block.tcl` — Vitis csynth driver (xcvu47p-fsvh2892-2L-e, 4.0ns).
- `run_local_csim_mp.sh` / `/tmp/run_mp_gc.sh` — local g++ csim runners.

LLM-instance parallel-WIP pattern files (`forge_pc_xformer_mp_golden_block.hpp/_test.cpp`, `mb_*`) were **not touched** — git status confirms only the three `mp_block`/`mp_config`/`mp_gradcheck` files changed.

---

## 3. CSIM match to the BF16 golden (MEASURED — local g++ csim)

Built locally with g++ + **authentic Xilinx open-source `ap_int` headers** (`/tmp/apreal`) + a `<cmath>` shim for `hls_math`. This validates the **arithmetic only**, not resources/timing.

**Main tb (3 checks, all PASS):**
- **(C) Deploy int forward** vs int64 golden `g_forward`: `XO_LOGIT` max|diff| = **0 → BIT-EXACT** (satisfies "forward bit-exact to the int kernel forward").
- **(A) Wide BF16 forward** vs golden `bf_forward`: WLG 6.913e-4, WRP 1.651e-6, WX1 5.895e-5, WATTN 0.0 → PASS.
- **(B) BF16 backward + dyadic-Adam** updated master vs float-precision Adam reference: max|diff| ≤ 2.6e-7, golden moved 53,248 weights → PASS.

**NEW gradcheck tb (raw bf16 grads compared DIRECTLY to golden `bf_backward`, which the old tb never did):**
- **KAT all-ones:** every tensor (`gWQ/gWK/gWV/gWO/gWUP/gWDN/gWHEAD` + `gQ/gK/gV` + `gR/gRout`) max|diff| = **0.0 → BIT-EXACT.**
- **KAT random-warmed (seed 7, 3 warm steps):** `gWQ` 1.863e-9 (1/4096 entries), `gWV` 9.537e-7 (1/4096), all others 0.0; `gRout/gQ/gK/gV` **BIT-EXACT**; **max relerr 1.093e-4 ≤ 1 bf16 ULP (7.8e-3).**

**The ~2-element random-KAT gap is BY DESIGN, not a bug.** Localized to the wgrad/forward accumulator *width*: the golden computes its entire fwd+bwd in **double** (round to bf16 only at tagged points); the kernel accumulates in **fp32** — the realistic bf16-backward DSP datapath. For ~2 sums sitting on a bf16 round-to-even tie, fp32-vs-fp64 rounding flips one ULP. Confirmed by patching the kernel's `bf_linear_bwd` accumulators to double → `gWQ` went to 0. This must **not** be "fixed" to fp64 — chasing the golden bit-for-bit would mean synthesizing double-precision MACs. The inter-chip `gRout` up-message and all activation-grads are bit-exact regardless.

**Bottom line:** forward bit-exact; backward bit-exact on KAT and within 1 bf16 ULP on random; the residual is fp32-vs-fp64 non-associativity, expected for the hardware datapath.

---

## 4. csynth / budget gate (ESTIMATED — Vitis NOT run here)

> **Vitis HLS is absent on this machine (Darwin; vitis_hls / v++ / vivado_hls all absent).** The MP-kernel DSP/LUT/FF/BRAM/URAM/WNS are **ESTIMATED**, scaled from the int kernel's real csynth report. **The EC2 csynth gate is required before AFI.**

### Measured baseline — int kernel, REAL Vitis csynth
(`forge_pc_xformer_block_csynth.rpt`, xcvu47p, per-SLR budget)

| Resource | int kernel (MEASURED) | per-SLR avail | % SLR |
|---|---|---|---|
| BRAM18 | 764 | 1344 | 56% |
| DSP | 185 | 3008 | 6% |
| FF | 104,110 | 869,120 | 12% |
| LUT | 190,508 | 434,560 | 44% |
| URAM | 42 | 320 | 13% |
| Timing | 3.192 ns (target 4.0, **+1.08 slack**) | — | met |

→ int kernel fits 1 SLR comfortably; **BRAM at 56% is already the dominant resource.**

### Estimated — BF16 MP kernel (base / operator-instance case)

| Resource | est. | % SLR | comfort |
|---|---|---|---|
| DSP | 293 (base) → 505 (high) | ~10% → ~17% | comfortable |
| LUT | 51.2% (base) → 59.9% (high) | — | comfortable |
| FF | 16–20% | — | comfortable |
| **BRAM18** | **base/packed ~1264 / 1344 = 94% (TIGHT, fits)** | **high/naive ~1667 / 1344 = 124% (OVER)** | **SWING FACTOR** |
| URAM | 13% → can absorb mitigation | — | headroom |

**Biggest risk = BRAM, not LUT/DSP.** The BF16 variant adds a full float master + a wide float fwd-for-bwd cache + float Adam m/v on top of the existing int16 arrays (~1.86 MB new on-chip static float, 91 arrays). The FFN-width float arrays dominate (SEQ×FFN and FFN×D, ~28–57 BRAM18 each; FFN=256=4·D). Estimated +500 BRAM18 (packed) to +903 (naive).

**DSP** is comfortable even high: the new fp32 muls live in `bf_matmul`/`bf_linear_bwd` route+wgrad, the softmax Jacobian (`gATTN·WV`, `p·gATTN`, `gs·WQ/WK`), `bf_adam_one`, `bf_qact`/`bf_layernorm`, plus `hls::exp/log2/ldexp`. The kernel stores `float`+bf16_round (Vitis binds a full fp32 mul = 3 DSP each), so the DSP estimate is already conservative-high. Secondary DSP risk: if HLS unrolls the DH=32 attention dot loops, the attention multipliers replicate (the high case).

**Timing is the one number that could not be bounded well.** The int forward closed at 3.192 ns, but the new fp32 mul/exp/log2 chains are deeper than int dyadic shift-add; long fp32 reduction chains (SEQ=64 / FFN=256 accumulators) could threaten WNS if HLS pipelines them poorly. **EC2 csynth must confirm WNS ≤ 4.0 ns** — second only to the BRAM check.

### Mitigation if BRAM is over (no kernel redesign needed)
1. **URAM steering (primary, near-free):** URAM is nearly empty (42/320 = 13%, ~278 URAM ≈ 1.1 MB free). The int kernel uses no `BIND_STORAGE` pragmas (AUTO already steered int16 arrays to URAM). Add `#pragma HLS BIND_STORAGE … impl=uram` to the wide-cache + `FWUP/FWDN/WHD/WRMASK` + Adam m/v float arrays → moves ~400–600 BRAM18 into URAM, landing ~280–450 BRAM18 (21–33% SLR) with URAM ~25–40%. Comfortably under 1 SLR.
2. **FWD/BWD as two CFG-selected kernels sharing one AFI (clean structural fix):** the FWD-phase static arrays and the BWD-phase grad accumulators are in mutually exclusive code paths, never simultaneously live; splitting halves peak footprint. The wide cache already round-trips through DDR (`MP_*` offsets) between kicks, so no on-chip persistence is needed.
3. **Lower-effort levers:** cast the wide cache to bf16/`ap_fixed<16>` on-chip (halves its BRAM, ~3× less mul DSP, and is the eventual production type anyway); or stream FFN-width arrays via dataflow instead of full static buffers.

None of these touch the no-global-sync architecture or the verified arithmetic.

---

## 5. AFI-readiness verdict + remaining steps to D=64 AFI

**Verdict: NEEDS-CSYNTH-ON-VITIS.** Arithmetic is verified (csim match). The kernel is one Vitis csynth run away from AFI-ready; the only open technical question is the **estimated-tight BRAM fit**, with a known, no-redesign mitigation (URAM steering) if it overflows.

Remaining steps to the **D=64 distributed AFI** (USER runs these on F2/EC2):

1. **CSYNTH GATE (blocking):** `vitis_hls -f /tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/synth_pc_xformer_mp_block.tcl` (xcvu47p-fsvh2892-2L-e, 4.0 ns). Confirm: DSP_MULT=0 outside attention on the deploy fwd; the new bf16-backward DSP count; **BRAM ≤ 1344 (1 SLR)**; **WNS ≤ 4.0 ns**. If BRAM over → apply URAM-steering pragmas (§4) and re-run.
2. **CL package:** clone `cl_forge_pc_xformer` for the MP kernel (wide cache DDR offsets `MP_*` already defined). ~5h AFI build.
3. **Distributed driver:** neighbour-message transport (`gRout` down / `R` up only) + **per-chip gradient accumulation** (the measured staleness bound = ~1 step; accumulate per chip rather than async-drop). No global gradient.
4. **Runner:** F2 launch/kick/collect across the D=64 chip set, persisting the float master + int16 deploy weights in DDR between kicks.

---

**STATUS: NEEDS-CSYNTH-ON-VITIS** — next action: run `vitis_hls -f synth_pc_xformer_mp_block.tcl` on EC2 to convert the estimated-tight BRAM/WNS into measured numbers (apply URAM-steering pragmas if BRAM > 1344) before the ~5h AFI.
