# tok/s projection — 124M GPT-2-small training rig on ONE AWS F2

**Grounded in MEASURED Vitis-HLS csynth kernels** (xcvu47p-fsvh2892-2L-e @ 4.0 ns / 250 MHz),
not hand-waving. Every number traces to a report in `results/fpga_training_rig/` or to
`forge_buffer_gpt2.hpp`. Arithmetic is shown; assumptions + risks are explicit (§5).

Device: 3 SLRs. One SLR = 434,560 LUT / 3,008 DSP / 869,120 FF / 320 URAM18 / 1,344 BRAM18.

---

## 0. The single MEASURED anchor kernel

`forge_wgrad_wide.cpp` — the rig's best weight-grad GEMM `gW = q(gO)ᵀ @ q(x)` at d768
(O=I=768, N=16 microbatch contraction, OT8×IT64 = **512-lane** dyadic-MAC PE).
From `forge_wgrad_wide_csynth.rpt`:

| metric | measured value |
|---|---|
| clock | **2.920 ns** estimated ≤ 4.0 ns target (closes; run @ 250 MHz) |
| **total latency** | **111,972 cycles = 0.448 ms** |
| DSP | **0** |
| LUT | **218,554 = 50% of one SLR** (16% device) |
| FF | 72,642 = 8%/SLR |
| BRAM18 | 120 = 8%/SLR |
| URAM | 0 |

Latency breakdown (from the per-loop table):
- burst_load gO ≈ 12,300 + burst_load x ≈ 12,300
- quantize QO 12,293 + QX 12,292
- **OTILE compute ≈ 75,072** (96 otiles × ~782; each itile = NRED 20 + WB 41 + overhead)

So **steady-state compute = 75,072 cycles**; the other ~37k is one-time load+quantize per
kernel invocation (amortizable across a larger on-chip working set / by overlapping load with
compute — not yet done).

This kernel computes one **768×768 × N=16** weight-grad. Its useful work =
768 × 768 × 16 = **9,437,184 dyadic-MACs**.

---

## 1. Per-token MAC count for ONE TRAINING STEP (GPT-2-124M)

A training step = forward + backward(input-grad) + backward(weight-grad). For a linear layer
y=Wx with W ∈ ℝ^{O×I}, per token:
- forward: O·I MACs
- input-grad gX = Wᵀ gO: O·I MACs
- weight-grad gW = gO xᵀ: O·I MACs

⇒ **≈ 3 × (forward MACs)** per linear, per token. (Attention score/context MACs add on top;
counted separately below.)

### 1a. Per-token forward linear MACs (from forge_buffer_gpt2.hpp counts)

Per transformer layer (d=768, FFN=3072), per token:

| linear | MACs/token | source |
|---|---|---|
| QKV proj (3·d·d) | 1,769,472 | P_QKV |
| O proj (d·d) | 589,824 | P_O |
| FFN up (d·4d) | 2,359,296 | half of P_FFN |
| FFN down (4d·d) | 2,359,296 | half of P_FFN |
| **per-layer linears** | **7,077,888** | = P_LAYER − P_LN |

× 12 layers = **84,934,656 MACs/token** in the block linears.

LM head (tied embedding, d·V = 768 × 50257): forward = **38,597,376 MACs/token**.
(Token + positional *embedding lookups* are gathers, not MACs — excluded, correct.)

**Forward linear MACs/token = 84,934,656 + 38,597,376 = 123,532,032 ≈ 1.235e8.**
(This equals the trainable matmul-param count ≈ 123.5M — as expected: 1 forward MAC per weight
per token. The +0.8M to reach 124.4M is LN γ/β and pos-emb, negligible MAC-wise.)

### 1b. Attention score/context MACs/token

Per layer, per token at seq S=1024, 12 heads × 64 head-dim:
- scores QKᵀ: H · S · d_h = causal-avg ≈ ½·S·d per token → use full S·d = 1024×768 = 786,432
- context P·V: same ≈ 786,432
⇒ ~1,572,864 forward attention MACs/token/layer × 12 = **18,874,368 MACs/token** forward.

Attention backward ≈ 2× forward (grad wrt scores + grad wrt V/Q/K paths) ⇒ ~37.7M.
Note: in this rig the **P·V soft-blend is the ONE irreducible true multiply** (on DSPs,
measured 128 DSP in `forge_attn`); QKᵀ score path is L1/dyadic. Attention is **seq-quadratic**
and grows with context; at S=1024 it is ~15% of forward linear MACs.

### 1c. Total MACs per training step per token

| component | fwd | + bwd (×3 linears / ×3 attn) | total /token |
|---|---|---|---|
| block + head linears | 1.235e8 | ×3 | **3.706e8** |
| attention score/ctx | 1.887e7 | ×3 | **5.66e7** |
| **TOTAL** | | | **≈ 4.27e8 MACs/token-step** |

**Headline denominator: ~3.7e8 MACs/token from the linears (the compute that maps to the
measured PE), ~4.27e8 including attention.** This refines the roadmap's "~3×124M = 3.7e8":
the 3.7e8 is exactly the linear path; attention adds ~15%.

I use **3.706e8 (linear-only)** for the primary tok/s projection because the measured kernel IS
the linear PE; attention runs on a *separate* (DSP-bearing) datapath in parallel and is treated
as a §5 risk, not loaded onto the same lanes. A conservative all-in cross-check uses 4.27e8.

---

## 2. Effective MACs/cycle and GMAC/s for ONE 512-lane PE instance

The kernel does 9,437,184 dyadic-MACs in a measured latency.

| basis | cycles | MACs/cycle | @250 MHz |
|---|---|---|---|
| **full latency** (incl. load+quantize) | 111,972 | 9,437,184 / 111,972 = **84.3** | 84.3 × 2.5e8 = **21.1 GMAC/s** |
| **steady-state OTILE only** | 75,072 | 9,437,184 / 75,072 = **125.7** | 125.7 × 2.5e8 = **31.4 GMAC/s** |

Peak theoretical of a 512-lane PE @ II=1 = 512 × 2.5e8 = 128 GMAC/s. Steady-state achieves
125.7/512 ≈ **24.5% lane utilization** — the inner NRED is II=1 but only N=16 deep per tile, so
the OT×IT lanes are filled 16 cycles then drained by a 41-cycle writeback (NRED 20 : WB 41 ≈ the
~32% efficiency). **This is the top throughput lever** (deeper N reuse / overlap WB with NRED →
toward the 128 GMAC/s ceiling). The numbers below use the *measured* 21.1 / 31.4, not the ceiling.

---

## 3. Device parallelism → device GMAC/s (floor / mid / best)

`forge_wgrad_wide` = **50% LUT/SLR, DSP=0, BRAM 8%/SLR, URAM 0, FF 8%/SLR**. LUT is the binding
resource → **~2 instances/SLR** (leaves 0% LUT headroom for glue — realistically 2 is tight; see
§5 risk). Across 3 SLRs that is a nominal **6 instances**. Real ceiling discounts:
- one SLR's worth of LUT/URAM is consumed by attention (DSP soft-blend), norms, CE-LUT, DDR
  streaming choreography, dyadic-Adam → call it **effective ~5 PE instances**, but I report the
  nominal 6 for the mid/best to match the roadmap band and flag the discount as a risk.

| scenario | instances | per-inst GMAC/s | device GMAC/s |
|---|---|---|---|
| **FLOOR** | 1 | 21.1 (full latency) | **21.1** |
| **MID** | 6 | 31.4 (steady-state) | **188.5** |
| **BEST** | 6 | 31.4 × 2 (DSP-as-adder doubles lanes) | **377** |

BEST lever: ~9,000 idle DSP48s lit as signed accumulators/adders lets each PE roughly **double**
its lane count (the DSP-as-adder gate, currently unused — kernel is DSP=0). 2× is a deliberately
conservative read of that lever (the DSP48 can do add+shift; full exploitation could be more, but
2× is defensible from "doubling the accumulate ALUs at fixed LUT").

---

## 4. tok/s = device GMAC/s ÷ MACs-per-token-step

Denominator = **3.706e8 MACs/token** (linear path; primary). All-in cross-check uses 4.27e8.

| scenario | device GMAC/s | tok/s (÷3.706e8) | all-in (÷4.27e8) |
|---|---|---|---|
| **FLOOR** (1 inst, full latency) | 21.1 | **57** | 49 |
| **MID** (6 inst, steady-state) | 188.5 | **509** | 441 |
| **BEST** (6 inst, +DSP-adder 2×) | 377 | **1,017** | 883 |

### Cross-checks

- **Roadmap band (44 floor / 900 planned / 3,500 best):** FLOOR 57 ≈ roadmap 44 (slightly
  higher — the wide kernel beat the morph-scaled estimate). MID 509 is below the 900 "planned"
  but in the same order; the 900 assumes 15–50% DSP-ALU util which my MID does **not** include
  (MID is DSP=0). BEST ~1,000 reaches the lower planned/upper-mid; the roadmap's 3,500 "best"
  requires both the DSP-adder array AND pushing lane utilization from 24.5% toward the ceiling
  (the §2 lever) — i.e. ~4× more than my conservative 2×, plausible but unmeasured. **My numbers
  are a notch more conservative than the roadmap and consistent with it.**
- **Morph anchor (53,000 tok/s @ 102k params, N=4 replicated, DSP=0):** scale by params.
  53,000 × (102e3 / 124.4e6) = **43.5 tok/s** — almost exactly the roadmap's 44 floor, and within
  1.3× of my measured FLOOR 57. The wide-kernel FLOOR being a bit *higher* than the morph-scaled
  floor is expected (the wide kernel is a tuned d768 GEMM, not a naive linear scaling). **Two
  independent measured anchors agree at the floor — strong confidence in the floor.**

### Flagship wall-clocks at the MID number (509 tok/s, linear-path)

| target | tokens | wall-clock @509 tok/s |
|---|---|---|
| flagship-A: ~30M model to convergence (Chinchilla 20×/param ≈ 0.6B tok) | 6.0e8 | **13.6 days** |
| 124M fine-tune, 1B tokens | 1.0e9 | **22.7 days** |
| Chinchilla-optimal 124M, 2.5B tokens | 2.5e9 | **56.8 days** |
| from-scratch 124M, 300B tokens | 3.0e11 | **18.7 years** (infeasible single-card → multi-F2 farm) |

(At BEST 1,017 tok/s these halve: ~30M ≈ 6.8 d, fine-tune-1B ≈ 11.4 d, Chinchilla-2.5B ≈ 28.4 d.)

**Note on flagship-A:** the roadmap's "~2 d to convergence" assumed a *smaller* (~10M) model
and/or the 900+ planned tok/s. At a measured-grounded MID of 509 tok/s, a 30M model to a real
Chinchilla budget is ~2 weeks, not 2 days. A ~10M model at a lighter token budget (e.g. 0.2B)
is ~4.5 days. The "2 day" figure is only reachable at the BEST tok/s on the smallest model — I
flag the roadmap's 2 d as optimistic vs the measured kernel.

---

## 5. ADVERSARIAL self-check — every assumption + its risk

**A1. "6 PE instances fit (2/SLR)."** RISK: HIGH-MEDIUM. The wide kernel is 50% LUT/SLR with
**zero** headroom for a 2nd copy's glue, AXI crossbar, and the rest of the training block
(attention, norm, CE, dyadic-Adam, DDR streaming). 2/SLR is a *resource-arithmetic* ceiling, not
a placed-and-routed result; the chess-train kernel's history (184%/SLR, timing miss) shows LUT
estimates balloon at P&R. **Effective instances are likely 4–5, not 6** → MID/BEST should be read
~15–30% lower (MID ~400–450, BEST ~800–900). I report 6 to stay comparable to the roadmap and to
not understate the *kernel's* density, but the honest planning number is ~5.

**A2. "Steady-state 75,072 cycles is the right per-call latency."** RISK: MEDIUM. The 75k excludes
load+quantize (37k). Achieving steady-state requires overlapping DDR load with compute (double-
buffering) and amortizing the on-chip quantize across many tiles — **not yet implemented** (the
measured 111,972 includes it). FLOOR correctly uses the full 111,972. MID/BEST assume the overlap
optimization lands; if it doesn't, MID falls to 188.5 × (75,072/111,972) ≈ 126 GMAC/s → ~340 tok/s.

**A3. "Lane utilization 24.5% (NRED:WB = 20:41)."** This is a *measured* inefficiency, not an
assumption — and it means there is real upside (the §2 lever) AND the §3 numbers are honest about
not yet capturing it. BEST's 2× from DSP-adders is *separate* from closing the WB-stall gap;
combining both is where the roadmap's 3,500 lives (unmeasured).

**A4. "DSP-as-adder doubles lanes (BEST)."** RISK: MEDIUM. DSP=0 today; lighting 9,000 DSP48s as
signed accumulators is *plausible* (the dyadic-MAC's accumulate maps to a DSP adder) but **unbuilt
and un-csynth'd** — `forge_attn` uses 128 DSP as multipliers, proving DSPs route, but the adder-PE
variant has not been synthesized. 2× is conservative; the lever could be larger or could cost LUT
(net wash). Treat BEST as a *target*, FLOOR/MID as *grounded*.

**A5. "DDR bandwidth is non-binding."** RISK: LOW — this one holds. From `forge_buffer_gpt2.hpp`
`BYTES_PER_STEP ≈ 994 MB` (fwd W 124 + bwd W 124 + opt R/W 746, int8-dyadic). This traffic is
**per optimizer step, batch-independent** (amortized over the microbatch). At even a single DDR
channel ~16 GB/s, 994 MB = **62 ms/step** of pure DDR. At MID 509 tok/s and a microbatch of, say,
512 tokens, a step is 512/509 ≈ 1.0 s of compute >> 62 ms DDR → **~16× headroom**; at batch ≥1k,
≥30×. F2 has multiple DDR channels (more BW), so the rig is **compute-bound, not bandwidth-bound**,
confirming the roadmap. The only way DDR binds is a tiny microbatch (<~64 tok), which one would not
run. RESIDUAL RISK: activation-checkpoint recompute traffic and attention KV streaming at S=1024
add per-token DDR not in BYTES_PER_STEP — but these are <100 MB/step and don't change the verdict.

**A6. "3.706e8 MACs/token denominator."** RISK: LOW. Derived directly from the param counts in
`forge_buffer_gpt2.hpp` (it equals the matmul-param count, a hard identity: 1 fwd-MAC/weight/token,
×3 for the training step). The only judgment call is excluding attention from the *primary* number
(justified: separate datapath). All-in 4.27e8 is the conservative bound and is in the table.

**A7. "Attention runs in parallel and doesn't steal PE cycles."** RISK: MEDIUM. The attention
soft-blend is on DSPs (128 measured) and the L1-score path on a separate datapath, so in principle
it overlaps the linear PEs. But it consumes LUT/URAM that competes with A1's instance count, and
its seq-quadratic cost grows with context. Folded into A1's "effective ~5 instances" discount.

**A8. "250 MHz (4 ns) achieved."** RISK: LOW. Measured estimated 2.920 ns ≤ 4.0 ns with 1.08 ns
uncertainty already absorbed (closes at csynth). P&R could erode it, but the precedent kernels
(fwd_gemm, dyadic_adam, attn all 2.92 ns) and the forward AFI that closed on real silicon
(WNS +0.123 ns) support 250 MHz.

### Net honesty verdict

- **FLOOR ~57 tok/s** is the most defensible single number — two independent measured anchors
  (the wide kernel @ full latency, and the morph 53k-scaled) agree at ~44–57.
- **MID ~509 tok/s** is grounded but leans on two un-landed optimizations (instance count ≥5–6
  AND load/compute overlap). Honest planning band: **~340–510 tok/s**.
- **BEST ~1,017 tok/s** requires the DSP-adder array (unbuilt). The roadmap's 3,500 needs that
  *plus* closing the 24.5%-utilization gap — possible, unmeasured, a target not a result.
- **Verdict: the rig is compute-bound (DDR has ~16–30× headroom), and a measured-grounded MID of
  ~500 tok/s puts flagship-A (~30M to convergence) at ~2 weeks and a 124M 1B-token fine-tune at
  ~3 weeks on ONE F2 — the single-card full-124M from-scratch (300B tok, ~19 yr) remains a
  multi-F2-farm task, exactly as the roadmap states.**
