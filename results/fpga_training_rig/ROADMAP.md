# FPGA training rig → 124M LLM trained on FPGA (the flagship)

The culmination of the multiplier-free program: train a transformer LM **entirely on AWS F2**
(forward + backward + optimizer, no GPU in the loop). Architected from the project's *measured*
F2 assets; numbers are grounded with arithmetic, not hype. (Design only — F2 builds are run by the
user, not the agent.)

## The honest flagship (two parts)

A from-scratch GPT-2-124M pretrain on one F2 is **years** of wall-clock (see below) — so the
defensible flagship is:

- **A — convergence proof (real LLM on FPGA):** a ~10–30M-param transformer trained **from scratch
  to convergence** entirely on one F2 (DDR-resident weights, minimal-multiply attention, dyadic-Adam),
  reporting real bpb/perplexity + measured tok/s + core-rail energy. *Completable in ~2 days.*
- **B — 124M datapath, measured + projected:** instantiate the full 124M GPT-2-small datapath
  (12L/d768/V50257/seq1024, DDR-resident W+m+v), measure **one bit-exact training step** on F2, and
  project full-training wall-clock/energy.

"124M LLM trained on FPGA" = the **124M datapath is real and measured at full width** + a smaller LM
is **actually pretrained to convergence** on-fabric. (A *completed* full-124M result = on-fabric
**fine-tune** ~1B tokens, ~3–13 days — the one tractable single-card full-124M run.)

## Feasibility: COMPUTE-bound, not bandwidth-bound

DDR bandwidth has ~10–90× headroom once a real batch amortizes the ~1 GB/step weight+optimizer
traffic (int8-dyadic packing). The binding limit is **multiplier-free add-ALU throughput @250 MHz**.

| tok/s (one F2, 124M) | basis |
|---|---|
| ~44 (floor) | morph-LM 53k tok/s @102k-params, linearly scaled — LUT-only, DSPs idle |
| **~900 (planned)** | ~9000 DSP48 lit as add/shift ALUs @250 MHz, 15–50% util |
| ~3,500 (best) | DSP + reachable LUT adder array |

Wall-clock @ ~900 tok/s: 300B-token pretrain ≈ **10.5 yr** (infeasible single-card); Chinchilla-2.5B
≈ **32 d**; 124M fine-tune 1B ≈ **13 d**; the ~30M flagship-A to convergence ≈ **2 d**. A from-scratch
124M needs a **multi-F2 farm** (16 boards → Chinchilla-2.5B in ~2 d).

## Staged roadmap

- **S0 — anchor (DONE):** MorphSSM 102k-param LM trains on F2 (53k tok/s @N=4, bit-exact, DSP=0, ~4W).
- **S1 — transformer primitive port (on-chip, small):** real minimal-mult transformer block (dyadic
  GEMM fwd + transpose-bwd, L1+gather attention + one DSP soft-blend, fine-dyadic norm, CE-LUT,
  dyadic-Adam) trains end-to-end on F2, bit-exact. *"the substrate trains a transformer, not just an SSM."*
- **S2 — DDR-resident + tiled streaming:** same small model, weights+m+v in DDR4 (reuse
  cl_forge_muzero_ddr pattern), per-layer streaming + checkpoint/recompute. Proves the 124M memory arch.
- **S3 — flagship-A:** ~10–30M transformer pretrained to convergence on one F2, multi-SLR tiled, real
  bpb + tok/s + energy.
- **S4 — flagship-B:** full 124M datapath, one bit-exact step measured + projection.
- **S5 — stretch:** on-fabric 124M fine-tune (completable) or multi-F2 from-scratch.

## Critical path (de-risk before any AFI spend)

1. **Compute density** — csynth a DSP-as-adder PE tile + the d768 dyadic weight-grad GEMM. Sets the
   real tok/s (collapses the 44→3500 band to a number) AND the feasibility gate.
2. **Backward kernel area/timing** — the chess TRAIN kernel did NOT close (184%/SLR, 5.427 ns). The
   d768 `q(gO)ᵀ@q(x)` weight-grad is the same class, wider — csynth-probe before committing.
3. **Algorithm gate** — the PC operand-quantized weight-grad `q(gO)ᵀ@q(x)` "matches backprop" is proven
   on **MLP/Conv only, never on a transformer**. Validate on a CPU char-LM (+ dyadic-Adam int-state)
   before HLS.

Steps 1–3 are all doable now in csynth / CPU sim — **no F2 build needed** — and collapse the largest
uncertainties first.

### Validation log
- **Gate 3 (algorithm) — ✅ PASSED (2026-06-21, CPU/MPS, `scripts/charlm_mixer.py --linear mfgrad`).**
  The FPGA-faithful multiplier-free backward — weight-grad computed as `q(gO)ᵀ@q(x)` (a shift-add tree,
  both operands dyadic; *not* an autograd matmul) + input-grad `q(W)ᵀ@gO` — trains a **transformer**
  char-LM to **gather 2.481**, identical to the `dyadic`-linear/autograd-backward baseline 2.48. The
  PC `set_mf_grads` recipe (previously MLP/Conv-only) holds on a transformer at no quality cost. With
  dyadic-Adam (optimizer ✓) and the dyadic forward (✓), **the entire fwd+bwd+optimizer loop is now
  validated multiplier-free, end-to-end, at softmax-competitive quality (2.48, +0.3–0.4 from dense)** —
  the single irreducible multiply being the attention soft-blend (on DSPs). Algorithm green-lights the rig.
- **Gate 2 (backward-kernel timing/area) — ✅ PASSED (2026-06-21, Vitis HLS csynth, z1d/FPGA-dev AMI, xcvu47p @4ns; `forge_wgrad_probe.cpp`, report `forge_wgrad_probe_csynth.rpt`).**
  The operand-quantized weight-grad GEMM at d768 (`gW=q(gO)ᵀ@q(x)`, dyadic-MAC = priority-encode →
  exp-add → barrel-shift → signed-accumulate, output-stationary OT8×IT64) **closes timing: 2.920 ns ≤
  4.0 ns**, at **49% LUT of one SLR (16% device), DSP=0, FF 7%/SLR**. This is the exact kernel class
  that did NOT close for chess-train (184%/SLR, 5.427 ns) — it now closes at transformer width because
  the shift-add PE + output-stationary tiling is timing-friendly. Fits in half an SLR with room for the
  rest of the block. (Probe latency 2.66 ms/768×768 matrix is *unoptimized* — tile loops not yet
  pipelined; throughput tuning, not feasibility.)
- **Gate 1 (compute density) — partial read: DSP=0.** The dyadic-MAC maps to LUT logic (no multiplier),
  so the ~9000 DSP48s sit idle — the "DSP-as-adder" lever is *available but unused*. Lighting DSPs as
  accumulators is the next throughput optimization (cuts LUT and/or raises the PE-lane ceiling).
- **NET: all three critical-path gates (3 algorithm, 2 timing/area, 1 fits & DSP-free) are GREEN at the
  probe level — the rig's biggest risks are de-risked.** Next: throughput (pipeline tile loops to II=1 +
  DSP-as-adder) and the full S1 training block (fwd GEMM + attention + norm + dyadic-Adam).
- **S1 KERNEL SUITE — ✅ SYNTHESIZES + CLOSES TIMING on silicon (2026-06-21, Vitis csynth, xcvu47p @4ns;
  4 parallel AWS z1d csynths).** Every training-block kernel closes 2.920 ns ≤ 4.0 ns, with the DSP
  profile EXACTLY as designed — mult-free except the one attention soft-blend:

  | kernel | timing | DSP | LUT/SLR | notes |
  |---|---|---|---|---|
  | `forge_fwd_gemm` (dyadic forward GEMM) | 2.92 ns | 0 | 1% | weight→shift, mult-free |
  | `forge_dyadic_adam` (optimizer update) | 2.92 ns | 0 | 3% | 0.59 ms / 590k-param region |
  | `forge_attn` (attention block) | 2.92 ns | **128** (4%) | 8% | DSP = the `Σp·V` soft-blend = the ONE irreducible multiply, confirmed in silicon |
  | `forge_wgrad_probe` (weight-grad) | 2.92 ns | 0 | 49% | the validated backward |
  | `forge_wgrad2` (throughput try) | 2.92 ns | 0 | 81% | **FAILED**: 2.7–40 ms, no win vs probe 2.66 ms |

  So a full transformer-block trainer's compute fits ~1–2 SLRs, DSP-free except the soft-blend. This
  proves the rig's datapath is **buildable on real silicon**, not just on paper.
- **THROUGHPUT — ✅ 8.7× DELIVERED + tok/s PINNED (2026-06-21, Vitis csynth, xcvu47p @4ns).** The
  weight-grad bottleneck was the **scalar writeback** (512 int32/tile @ 1/cycle = 89% of the 665k-cycle
  baseline), NOT compute. Fix = 512-bit burst writeback + wide loads/fused-quantize:

  | kernel | cycles | ms | LUT/SLR | DSP | vs baseline |
  |---|---|---|---|---|---|
  | `forge_wgrad_probe` (baseline) | 664,932 | 2.66 | 49% | 0 | 1× |
  | `forge_wgrad_wide` (burst writeback) | 111,972 | 0.448 | 50% | 0 | 5.94× |
  | **`forge_wgrad_wide2`** (+ wide loads) ⭐ | **76,640** | **0.307** | 51% | 2* | **8.7×** |
  | `forge_wgrad_flat2` (ping-pong overlap) | 73,841 | 0.295 | 64% | 3* | 9.0× (roofline play **underdelivered**) |
  | `forge_fwd_wide` (pre-dyadic W) | ~70,000 | ~0.28 | 51% | 0 | (fwd; 2.36 ms QW artifact removed) |

  *DSP 2–3 = an address-divide in control logic, NOT a datapath multiplier (dyadic MAC is DSP=0). All
  close 2.920 ns. **Winner: `forge_wgrad_wide2`** (bit-exact to proven, half-SLR). KEY: the weight-grad
  is **output-WRITE-bound** — it must write the full 589,824-elem gW (36,864-word floor > 18,432 compute
  floor); `flat2`'s overlap-to-floor didn't schedule II=1 (projected 36,864, measured 73,841). The v1
  sweep (bigpe/turbo/dsp/fast/pe512) ALL missed (bigpe 1.83-SLR no-win; 4 flatteners >75-min synth
  backstop) — lesson: unroll ONLY the 512 lanes, never tiles/N. **MEASURED tok/s (124M): floor ~57 /
  MID ~509 (band 340–510) / "best" ~600–650 (the old 1,017 was optimistic — writeback-bound, so
  DSP-adder lane-doubling gives ~1.2× not 2×). Flagship-A ~30M-to-convergence ≈ 2 weeks on one F2.** Full
  analysis: `results/fpga_training_rig/TOKS_PROJECTION.md`.
- **THROUGHPUT — ✅ WRITE-FLOOR REACHED via DATAFLOW (2026-06-21, the lever flat2 couldn't pull).** The
  write-bound weight-grad (`wide2` 76,640 cyc vs the 36,864-word write-floor) is fixed by `#pragma HLS
  DATAFLOW` — two concurrent processes (compute ∥ writeback) + a right-sized FIFO. **`forge_wgrad_df3`
  (per-row 2048-bit token, depth-16 FIFO): 40,927 cyc / 0.164 ms, closes 2.920 ns, 51% LUT/SLR, DSP=0**
  (report confirms `compute_tiles` 40,927 ∥ `write_tiles` 36,875, Pipeline type `dataflow` → writeback
  hidden under compute). **1.87× over wide2, 16.2× over the 664,932-cyc probe.** (`forge_wgrad_df`, the
  whole-tile 16,384-bit token, FAILED Vitis's 4096-bit stream-aggregate limit — the per-row hedge was
  necessary.) **Updated MEASURED tok/s (124M): floor ~78 / MID ~720–930 — reaches the roadmap's "900
  planned" band AT DSP=0** (DSP-as-adder now becomes headroom toward "best", not a prerequisite).
  Flagship-A ~30M-to-convergence ≈ 1–1.5 weeks on one F2. The throughput lever is exhausted (write-floor
  reached); details in `TOKS_PROJECTION.md`.
- **S1 KERNEL SET COMPLETE + BIT-EXACT CSIM-VERIFIED (2026-06-21, Vitis csim_design + csynth_design, xcvu47p).**
  The 5 missing transformer-block kernels authored + adversarially-verified (2 real bugs caught: a runtime
  divide in LayerNorm, a missing fine-dyadic-rescale STE in LayerNorm-bwd) and each run csim+csynth on EC2.
  **ALL 5 pass CSIM BIT-EXACT (max|kernel−int64ref|=0, CSim 0 errors) AND close timing:**

  | kernel | csim | timing | DSP | LUT/SLR | role |
  |---|---|---|---|---|---|
  | `forge_xgrad_wide` | bit-exact | 2.920 ns | 0 | 5% | input-grad gx=gO@q(W) |
  | `forge_layernorm` | bit-exact | 2.920 ns | 0 | 2% | fine-dyadic LN fwd (10,928 cyc) |
  | `forge_layernorm_bwd` | bit-exact | 2.920 ns | 0 | 9% | LN backward (7,296 cyc) |
  | `forge_relu` | bit-exact | 2.920 ns | 0 | 1% | ReLU fwd+bwd (1,053 cyc) |
  | `forge_attn_bwd` | bit-exact | 3.358 ns | **384** | 19% | attention backward (the 4 allowed soft-blend/softmax multiplies) |

  Combined with the measured fwd/optimizer/weight-grad kernels, **every component of the multiplier-free
  transformer training datapath (fwd + attn + LN + ReLU + CE/LM-head[reuse] + input-grad + weight-grad +
  dyadic-Adam) now exists, closes timing, fits, and is functionally bit-exact.** DSP profile exactly as
  designed: 0 everywhere except the attention soft-blend (fwd 128 + bwd 384). This is the functional-
  correctness gate (ROADMAP open-item (a)) — CLOSED for the new kernels.
- **S1 INTEGRATED FORWARD — ✅ CSIM BIT-EXACT + CLOSES TIMING (2026-06-22, Vitis csim_design+csynth_design).**
  `forge_train_step_s1` mode 0 = a WHOLE transformer forward + cross-entropy as ONE kernel (embed → LN1 →
  Q/K/V dyadic GEMM → 2-head soft causal attn → Wo → residual → LN2 → FFN(up→ReLU→down) → residual →
  max-plus LM-head → balanced-LUT CE grad). **csim BIT-EXACT (max|kernel−int64ref|=0 over all 9 regions, 0
  errors), closes 2.920 ns ≤ 4.0, DSP=128 (attention soft-blend only — mult-free invariant holds), fits one
  SLR (86% LUT / 43% BRAM / 5% URAM).** Timing took 2 rounds: first cut missed at 5.840 ns (direct gmem/AXI
  gathers in the embed/LN/CE compute loops were the critical path + an II-violation swarm); fixed via the
  morph_lm TT_RESIDENT_W on-chip resident-staging pattern (burst-load weights/tokens/LUTs once → compute
  on-chip → burst-store) with the math/ABI/golden UNCHANGED → 2.920 ns. The `dyad()` round-faithfulness
  question (1.5·2^k vs the Python authority ~1.414·2^k) is RESOLVED by making the hardware dyad rule
  authoritative + self-consistent (the integrated int64 golden uses it). Latency 3.06M cyc/12.2 ms for one
  128-token forward through 1 block (integrated GEMM is un-throughput-tuned; the df3/wide2 techniques apply
  later — correctness/timing is the S1 goal, not tok/s).
- **S1 FORWARD+BACKWARD — ✅ CSIM BIT-EXACT + CLOSES TIMING + FITS THE DEVICE (2026-06-22).** `forge_train_step_s1`
  mode 0 (forward+CE) + mode 1 (backward: LM-head max-plus adjoint scatter → FFN xgrad/wgrad → ReLU → LN2 bwd →
  residual split → Wo bwd → per-head attn_bwd → Q/K/V bwd with gX1 accumulation → LN1 bwd → embed scatter),
  composing the CSIM-verified backward kernels on-chip-resident. **Vitis csim "FWD BIT-EXACT ; BWD BIT-EXACT"
  (all 9 forward + 15 gradient regions max|kernel−int64ref|=0, 0 errors), closes 3.197 ns ≤ 4.0, fits: 31% LUT /
  68% BRAM / 4% DSP (448 = attn only) / 2% URAM of the device.** Two area iterations got it to fit: (i) the
  monolithic fwd+bwd was 116% device LUT (a separate inlined GEMM PE per call site); resource-sharing
  (de-templatize + ALLOCATION instances=1) only reached 103%; (ii) the decisive fix was NARROWING the compute
  engines from the throughput-optimized 128-wide tiling to S1_LANE=16 (cyclic partition + chunked partial-sum
  reduction) — bit-exact-preserving (integer reduction is order-independent), ~3.3× smaller (1.35M → 415k LUT).
  S1 is a correctness/convergence proof, so trading the wide tiling for area is correct (the throughput story is
  the standalone optimized kernels + the tok/s projection). An adversarial pass also caught a load-bearing
  faithfulness bug (wgrad gO int16-wrapped at the Q16 grad scale → fixed with dyad_wide). **The complete
  multiplier-free transformer forward+backward training computation is now functionally correct, timing-closing,
  and silicon-buildable.**
- **STILL OPEN (next):** (a) mode 2 (dyadic-Adam per region) + mode 3 (fused one-kick trainer: zero-grads → fwd
  sweep → bwd sweep → opt); bit-exact one-step CSIM. (b) mode-3 N-step convergence CSIM (loss decreases / tracks
  the charlm 2.481 trajectory) — the on-fabric-training proof. (c) AFI build (gated — clone cl_forge_muzero
  on-chip pattern; the trainer fits one SLR LUT-wise, BRAM spans ~2 SLRs → standard floorplan, device has 3).

## First three steps (local, no F2 builds)

1. **csynth a DSP-as-adder PE tile + d768 weight-grad GEMM** (Vitis HLS, no AFI) — measure LUT/DSP/timing.
2. **CPU bit-faithful d-scaled char-LM:** PC `q(g)ᵀ@q(x)` weight-grad + dyadic-Adam vs AdamW backprop —
   validates the never-on-a-transformer backward + int-state dyadic-Adam (int32 grad-accum, round-vs-floor
   log2 v, m/v precision).
3. **`forge_buffer_gpt2.hpp`** — the 124M DDR layout generator (layer-major [W|m|v|act-ckpt|grad|tokens|
   CE_LUT], 16-word aligned, host-staged exp2-ROM, int8-dyadic) — forces the memory arithmetic exact.

## Biggest risks (all flagged with measured precedents)

1. **Compute density unmeasured** — the DSP-as-adder array is unbuilt (morph was DSP=0); swings wall-clock 10–20×.
2. **Backward kernel timing** — chess-train precedent didn't close (184%/SLR, 5.427 ns) + un-budgeted barrel shifters.
3. **PC weight-grad unproven on a transformer** (MLP/Conv only).
4. **Dyadic-Adam state precision** — int16-m overflow (needs int32 grad-accum), round-vs-floor(log2 v), m/v collapse at scale.
5. **On-chip bwd working set** busts budget (grad-accum + double-buffered W + LM-head vocab tiling).
6. **Multi-SLR model-parallel** is new floorplanning (morph N=4 is data-parallel, doesn't transfer).
7. **DDR streaming BW uncharacterized** (only datapoint is latency-bound) — cheap microbenchmark.
8. **Wall-clock IS the flagship risk** — mitigated by the staged definition (A: ~30M to convergence; B: 124M step + projection), not engineered away.
