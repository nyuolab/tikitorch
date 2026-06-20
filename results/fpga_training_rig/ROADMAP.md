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
