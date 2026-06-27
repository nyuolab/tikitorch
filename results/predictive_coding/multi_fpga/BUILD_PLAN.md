# Build plan — PC-fabric transformer LM (LOCKED 2026-06-26)

**NORTH-STAR (locked):** a **~124M** (GPT-2-small) distributed PC transformer trained on the **multi-F2 fabric in
HOURS**, via a **no-global-sync bubble-free pipeline** that demonstrates the *scalable* architecture → coherent
TinyShakespeare. Honest claim = *the parallel model works and projects favorably*, NOT "beats an A100 on this toy"
(a single F2 < a GPU; an A100 does this in minutes). Multiplier-free where it pays (the bulk), warm-trained where
the multiply is essential (attention).

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

NEXT, in dependency order:
1. **Kernel → dot-product attention** (#3): forge_pc_xformer currently uses L1 attn (the capped deviation). Swap
   the L1 score for the warm dot-product QK (the one allowed multiply); re-csim bit-exact vs golden; free csynth.
2. **D=64 #3 AFI build** (~5h, ~$20) → train across 8 FPGAs = the PC-transformer-on-fabric pipeline on silicon.
3. **Bubble-free pipeline-parallel driver + burst the host-relay** (NO rebuild) — the first MEASURED speedup;
   this is where the no-global-sync edge becomes a number. Quantify serial-vs-pipe.
4. **On-fabric chip-to-chip interconnect** — removes the 54% host-relay; the real throughput-scaling enabler.
5. **Scale the kernel D=64 → D=768** (the 124M layer): bigger matmul tiles, DDR weight staging, multi-SLR.
6. **DSP-adder array + replication** (the compute-density + data-parallel multipliers) → **train 124M → coherent
   Shakespeare**, fully instrumented (tok/s, comms, energy), with the honest MEASURED-vs-PROJECTED scaling figure.
