# Build plan — PC-fabric transformer LM (LOCKED 2026-06-26)

**NORTH-STAR (locked):** a **~124M** (GPT-2-small) distributed PC transformer trained on the **multi-F2 fabric in
HOURS**, via a **no-global-sync bubble-free pipeline** that demonstrates the *scalable* architecture → coherent
TinyShakespeare. Honest claim = *the parallel model works and projects favorably*, NOT "beats an A100 on this toy"
(a single F2 < a GPU; an A100 does this in minutes).

**PRECISION DECISION (locked 2026-06-27):** the FULLY-multiplier-free integer recipe CAPS at ~0.20 acc (gibberish)
and does NOT scale away (measured int golden D=64 0.19, D=128 peaks 0.20 then declines; the diffuse integer
realization is a hard, scale-invariant wall — optimizer/backward/precision/block-scaling/attention all ruled out).
So we ADMIT low-precision multiplication: train the bulk in **BF16 (de-risk, robust, FPGA-native) → MXFP8 for
throughput** (4-bit-mantissa multiplies, ~4× cheaper than BF16, with a per-block POWER-OF-2 scale that stays
mult-free — the tropical-adjacent flavor we keep). The NOVEL thesis is now SOLELY the **PC-distributed-no-global-
sync architecture + bubble-free pipeline** — all precision-INDEPENDENT, fully intact under BF16/FP8. Total
multiplier elimination is the documented RESEARCH DIRECTION (the diffuse-int-gap problem). Novel angle to measure:
PC's *local* errors may tolerate lower precision (FP8/FP6) than backprop needs (no whole-depth dynamic range).

## Why PC-on-FPGA can do this (the parallelism — measured-grounded projection in scripts/pc_pipeline_throughput.py)

1. **Bubble-free pipeline parallelism** — PC has NO global backward, so no pipeline-flush bubble (the thing that
   bleeds GPU model-parallel). Projected **8–17× over the serial driver**; ~2× the utilization of GPipe at M~L.
2. **All layers train simultaneously** (local credit assignment) — the whole depth is live, continuous microbatch
   stream. 3. **Async / staleness-tolerant** (no global sync; measured ~0.6pp at 50% stale). 4. **Weight-stationary
   dataflow + the DSP-as-adder array** (mult-free bulk → ~9000 idle DSP48 become a parallel shift-add fabric).
   5. **Replication × token-parallel** (measured 4.00× linear). **Projection: 124M → coherent-overfit
   TinyShakespeare in SINGLE-DIGIT HOURS on one f2.48xlarge-class fabric** (PROJECTED, contingent on the build).

## The recipe (decided on evidence — "#3")

The fully-multiplier-free recipe (L1 attention + integer exp2-LUT softmax) caps a **flat ~0.15 acc behind float
across D=64→256** (measured scale-trend; the gap does NOT shrink with scale) — so it does not reach competitive
quality by scaling. The **integer attention is the wall** (the multiply does essential work there). The recipe
that DOES reach competitive AND scales:

> **Warm-train DOT-PRODUCT attention (the one allowed multiply, ~185 DSP, attention-only)
>  + multiplier-free PC-trained BULK** (FFN, projections, embedding, LM-head, LayerNorm — the *dominant* compute,
>  DSP-free: dyadic ±2^k weights, fine-dyadic LN, int16 activations, dyadic-Adam optimizer)
>  + per-width LR tuning + warmup (the dyadic optimizer is less stable with width — measured).

Measured (torch twin): D=128 → **0.492 acc / 2.428 bpb**, matching the original de-risk's 2.384 bpb and tracking
the float baseline (0.544) up in scale (D=64 0.377 → D=128 0.492). This is the validated, scaling, working-LM path.

**Honest framing (every artifact):** *a transformer trained on the PC fabric, multiplier-free except the
attention, distributed model-parallel with no global gradient synchronization.* The multiply is a small fraction
(attention ≪ FFN), so the DSP-free-bulk throughput story holds. Fully-mult-free is a soft-ceiling research
direction (might narrow with much more compute, unproven) — NOT the deliverable path.

## Phase A — working LM (quality first)

- **A1 [in progress]** Lock the #3 recipe as a clean, scalable char-LM reference with **sampling**
  (`scripts/pc_xformer_charlm.py`). Train at a coherent-text scale; sample; confirm recognizable Shakespeare.
  GPU/EC2 for the ~350M run.
- **A2** Per-width LR/warmup schedule for the dyadic-Adam at scale (measured instability with width).
- **A3** Get the on-fabric kernel onto **dot-product attention** (the forge_pc_xformer kernel currently uses L1 —
  the deviation; the LLM instance's kernel already does warm attention + mult-free bulk + 185-DSP-attention-only,
  likely reusable). csim bit-exact vs the int64 golden → free csynth gate → AFI.
- **A4** Distributed driver + multi-chip run on the #3 recipe; sample text from the on-fabric-trained model.

## Phase B — throughput (after quality; no rebuild)

Burst the readback (kill the unbursted half of the 54% host-relay) → **pipeline-parallel driver** (flat-throughput
-with-chips; PC pipelines cleanly — no global-backward bubble) → replication (proven 4× linear) → DSP-adder array.

## Phase C — scaling (structural)

On-fabric chip-to-chip interconnect (replaces host-relay — the real throughput-scaling enabler) → multi-instance
TCP (structural cap, not faster). Comms is local/linear; with pipelining + on-fabric links throughput is
flat-to-scaling, NOT the regression the un-pipelined host-relay baseline showed.

## Status & immediate build order (toward the 124M north-star)

Done: recipe #3 decided + validated (torch D=128 → 0.492/2.428bpb); Phase A1 char-LM reference (train+sample);
throughput projection (single-digit hours). Infra: SSO live, no F2 running; distributed-PC-transformer AFI NOT
built; forge_pc_block 8-FPGA AGFI exists.

NEXT, in dependency order (BF16-first):
1. **BF16 reference + recipe lock:** validate end-to-end that the **BF16 PC** path (cascade backward, not autograd)
   samples coherent TinyShakespeare at a real scale (PC-cascade ≈ backprop is established; this confirms it holds
   in BF16). scripts/pc_xformer_charlm.py already supports the float/BF16 recipe (MULTFREE=0).
2. **BF16 transformer kernel:** matmuls + PC cascade backward + dyadic/BF16-Adam in BF16 (DSP multiplies). The LLM
   instance's s1-trainer lineage already does warm-precision attention on silicon → reuse patterns. csim → csynth.
3. **D=64 BF16 AFI build** (~5h) → train across 8 FPGAs = the PC-transformer-on-fabric pipeline on silicon.
4. **Bubble-free pipeline-parallel driver + burst the host-relay** (NO rebuild) — the first MEASURED no-sync
   speedup; quantify serial-vs-pipe (8–17× projected).
5. **On-fabric chip-to-chip interconnect** — removes the 54% host-relay; the real throughput-scaling enabler.
6. **Scale the kernel D=64 → D=768** (the 124M layer): bigger matmul tiles, DDR weight staging, multi-SLR +
   replication → **train 124M → coherent Shakespeare**, instrumented (tok/s, comms, energy).
7. **Then MXFP8 for throughput** (4-bit-mantissa multiplies + pow2 block scale) + measure the FP8/FP6 floor PC
   tolerates (the novel precision angle) — the efficiency/scaling pass once quality is in hand.
