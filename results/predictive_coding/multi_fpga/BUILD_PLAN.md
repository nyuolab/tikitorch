# Build plan — PC-fabric transformer LM (LOCKED 2026-06-26)

**Goal:** a transformer trained on the PC fabric that produces **sampling-coherent TinyShakespeare text** at
**~350M params** (GPT-2-Medium / NanoGPT-Med scale), trains quickly, distributed model-parallel with no global
gradient sync. Multiplier-free where it pays (the bulk), warm-trained where the multiply is essential (attention).

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

## Status

Diagnosis + decision committed (72f5b2a, c34cfe0). Recipe validated in torch. Next: A1 reference + run.
