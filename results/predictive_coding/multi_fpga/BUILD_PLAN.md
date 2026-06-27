# Build plan — PC-fabric transformer LM (LOCKED 2026-06-26)

**NORTH-STAR (locked):** a **~124M** (GPT-2-small) distributed PC transformer trained on the **multi-F2 fabric in
HOURS**, via a **no-global-sync bubble-free pipeline** that demonstrates the *scalable* architecture → coherent
TinyShakespeare. Honest claim = *the parallel model works and projects favorably*, NOT "beats an A100 on this toy"
(a single F2 < a GPU; an A100 does this in minutes).

**PRECISION DECISION (locked 2026-06-27):** the FULLY-multiplier-free integer recipe CAPS at ~0.20 acc (gibberish)
and does NOT scale away (measured int golden D=64 0.19, D=128 peaks 0.20 then declines; the diffuse integer
realization is a hard, scale-invariant wall). So we ADMIT low-precision multiplication: train in **BF16 (de-risk,
robust, FPGA-native) → MXFP8 for throughput** (4-bit-mantissa multiplies, ~4× cheaper than BF16, with a per-block
POWER-OF-2 scale that stays mult-free — the tropical-adjacent flavor we keep). The NOVEL thesis is now SOLELY the
**PC-distributed-no-global-sync architecture + bubble-free pipeline** — all precision-INDEPENDENT, fully intact
under BF16/FP8. Total multiplier elimination is the documented RESEARCH DIRECTION.

**WHERE the wall actually is (triangulated 2026-06-27, torch char-LM D=256, TinyShakespeare):** the cap is NOT the
multiplier-free *forward* — it's the fully-*integer backward + optimizer*. **★ HONESTY: the 2.016 / 1.732 numbers
below are AUTOGRAD witnesses (torch autograd), NOT the PC-cascade datapath the kernel transcribes** — they witness
that the *precision recipe* is sound; the cascade-fix16 confirmation is PENDING Gate G2 (the golden-cascade training
experiment running now). Do not present them as proving the kernel's cascade datapath. Measured, same recipe, same
scale:
- mult-free **forward** (dyadic weights + int16 acts + dot attn) + **float** backward/optimizer → **2.016 bpb,
  COHERENT** (autograd witness; recognizable Shakespeare: *"One word and death... Lest honour him; all harm-on
  faired come"*).
- **full BF16** (forward + backward + optimizer) → **1.732 bpb, MORE coherent** (autograd witness; character names,
  dialogue).
- **fully-integer** golden (int cascade backward + int16-master dyadic-Adam) → **caps 0.20, gibberish**.
So the diffuse-int wall lives in the **int backward + int16-master optimizer**, not the forward. **Refined recipe:
keep the forward multiplier-free (validated coherent + the throughput-critical bulk → DSP-adder array survives) and
spend BF16 only on the backward + optimizer** (where the int realization actually fails). Build BF16-everywhere
first (most robust, 1.73 bpb), then reclaim the mult-free forward as a throughput lever (≈+0.28 bpb cost, keeps the
thesis). Novel angle still open: PC's *local* errors may tolerate lower precision (FP8/FP6) than backprop's
whole-depth dynamic range — measure on the backward/optimizer where the int wall sits.

## Why PC-on-FPGA can do this (the parallelism — measured-grounded projection in scripts/pc_pipeline_throughput.py)

1. **Pipeline parallelism, all chips busy** — the whole depth is live on a continuous microbatch stream (local
   credit assignment: each chip updates from its OWN error, no global all-reduce / no global gradient).
   Projected **8–17× over the serial driver** (the serial driver runs one chip at a time; pipelining fills all L).
2. **No global gradient / no all-reduce** — the genuine structural win: weight updates are per-chip-local. This is
   precision-independent and intact under BF16/FP8.
3. **Weight-stationary dataflow + the DSP-as-adder array** (mult-free bulk → ~9000 idle DSP48 become a parallel
   shift-add fabric). 4. **Replication × token-parallel** (measured 4.00× linear).

**⚠ STALENESS — corrected (measured 2026-06-27, torch char-LM D=128 L=4, TinyShakespeare).** There are TWO distinct
"staleness" claims and they do NOT both hold; the BUILD must respect this:
- **Error-relaxation staleness (TYPE 1, TOLERATED):** using *stale neighbour activation/error messages* in the
  local error computation — the iterative-PC relaxation is stable under this (the old MLP "~0.6pp at 50% stale"
  result). This is the real async win: the backward error sweep need NOT drain in strict lockstep.
- **Weight staleness (TYPE 2, NOT tolerated):** applying a gradient computed `k` weight-updates ago (the naive
  per-microbatch async update across a deep pipeline → staleness = pipeline depth). MEASURED bpb vs sync 2.457:
  stale-1 **2.639 (+0.18, OK)**, stale-2 3.209, stale-4 3.747, stale-8 4.478. **Only ~1 step is tolerable**, and
  **LR-scaling does NOT recover it** (matched LR=1.2e-3: stale-0 2.482 vs stale-8 3.747). Self-consistent grads
  (weight-stashed, PipeDream-2BW best case) still degrade — it's the *magnitude*, not consistency.
- **⇒ BUILD RULE:** the pipeline MUST bound weight-staleness to ~1 via **per-chip gradient accumulation** (freeze
  weights over an aligned window W, accumulate the local error, apply once, the pipeline stays full, cost = a W×
  larger effective batch — standard). This is *pipelined gradient accumulation with a local backward*, NOT "drop a
  round, train fully async." The earlier "async/staleness-tolerant" framing conflated TYPE 1 with TYPE 2 and is
  REFUTED for weight updates. The throughput projection (8–17× over serial) SURVIVES — it comes from filling all
  chips, which accumulation does not impede.
  **⚠ STALENESS DEPENDS ON W vs L — it is NOT "~1 independent of L".** Effective staleness is `ceil(2(L−1)/W)`;
  `staleness ≤ 1` requires **`W ≥ 2(L−1) ≈ 2L`** (a microbatch's forward and backward are ~`2(L−1)` slots apart in
  a depth-`L` pipeline). So **`W=4` is valid ONLY for `L ≤ 3`; the L=8 gate needs `W ≥ 16`, L=12 needs `W ≥ 24`**.
  The driver MUST **assert `W ≥ 2*L` host-side**. Cost at depth: **L=12 → `W ≥ 24` → ~1536-token effective batch**
  (may need an LR re-tune). And: the **serial de-risk driver (1 microbatch in flight) does NOT exercise staleness** —
  a **1F1B host-sim test is required before AFI**. Finally, the OUTPUT-chip LM-head update (`gWHEAD`,
  `block.cpp:559-565`) is a SECOND apply site and MUST fold into the same windowed accumulation or the head desyncs
  from the body. (See KERNEL_DESIGN_SPEC §4.3/§4.5 for the full treatment.)
   5. **Replication × token-parallel** (measured 4.00× linear). **Projection: 124M → coherent-overfit
   TinyShakespeare in SINGLE-DIGIT HOURS on one f2.48xlarge-class fabric** (PROJECTED, contingent on the build).

## The recipe (decided on evidence — "#3")

The fully-multiplier-free recipe (L1 attention + integer exp2-LUT softmax) caps a **flat ~0.15 acc behind float
across D=64→256** (measured scale-trend; the gap does NOT shrink with scale) — so it does not reach competitive
quality by scaling. The **integer attention is the wall** (the multiply does essential work there). The recipe
that DOES reach competitive AND scales:

> **Warm-train DOT-PRODUCT attention (the allowed multiplies — see DSP note below)
>  + multiplier-free PC-trained BULK** (FFN, projections, embedding, LM-head, LayerNorm — the *dominant* compute,
>  DSP-free: dyadic ±2^k weights, fine-dyadic LN, int16 activations, dyadic-Adam optimizer)
>  + per-width LR tuning + warmup (the dyadic optimizer is less stable with width — measured).

**DSP attribution note (corrected vs the csynth report + source — the "185 DSP attention-only" framing was wrong).**
The int `forge_pc_xformer_block` uses **185 DSP**, but this is **NOT the forward `p·V` blend** (that blend is **only
2 DSP**, `block.cpp:436`). The other **183 DSP are the attention-BACKWARD int64 multiplies** (`block.cpp:590-601`;
csynth loops `LOOP_590`=128, `LOOP_591`=11, `LOOP_592`=44). The forward stays effectively mult-free (2 DSP); the
DSP cost is in the **attention backward**, which the mp kernel **keeps unchanged**. ⇒ the mp **DSP delta** vs the 185
baseline is **only the new fix-wgrad multiplies**, not "adding a backward" (the backward already exists at DSP=0 for
every dyadic op).

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
  the deviation; the LLM instance's kernel already does warm attention + mult-free bulk, with the DSP spent on the
  attention forward+backward — 2 DSP fwd `p·V` blend + 183 DSP attn-bwd int64 muls, NOT a "185-DSP attention-only
  forward"; likely reusable). csim bit-exact vs the int64 golden → free csynth gate → AFI.
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

**REFRAME (the int kernel is ALREADY a full train kernel — the mp change is a NARROW precision swap, NOT "adding a
backward").** `forge_pc_xformer_block` already synthesizes the **complete** forward + backward cascade + dyadic-Adam
optimizer, with **DSP_MULT=0 on every dyadic op** (route, wgrad, adam, LN-bwd) and the attention backward as int64
muls. So the mixed-precision sibling is NOT building a backward from scratch; it is a **narrow precision swap on two
operations**: (a) the wgrad outer product `pow2×pow2 → fixed-point-mul wgrad`, and (b) the optimizer master/state
`int16-master → fix16-master`. Everything else (forward, error routes, attn-bwd, LN-bwd, window accumulation) is
structurally carried over. The steps below should be read as that precision swap, not a from-scratch backward.

NEXT, in dependency order (BF16-first):
1. **BF16 reference + recipe lock:** validate end-to-end that the **BF16 PC** path (cascade backward, not autograd)
   samples coherent TinyShakespeare at a real scale (PC-cascade ≈ backprop is established; this confirms it holds
   in BF16). scripts/pc_xformer_charlm.py already supports the float/BF16 recipe (MULTFREE=0).
2. **mp transformer kernel (narrow precision swap, NOT a new backward):** swap the existing int kernel's wgrad
   (`pow2×pow2 → fixed-point-mul`) and optimizer master (`int16 → fix16`); the cascade backward + dyadic-Adam +
   attn-bwd int64 muls already exist. The LLM instance's s1-trainer lineage already does warm-precision attention on
   silicon → reuse patterns. csim → csynth.
3. **D=64 BF16 AFI build** (~5h) → train across 8 FPGAs = the PC-transformer-on-fabric pipeline on silicon.
4. **Bubble-free pipeline-parallel driver + burst the host-relay** (NO rebuild) — the first MEASURED no-sync
   speedup; quantify serial-vs-pipe (8–17× projected).
5. **On-fabric chip-to-chip interconnect** — removes the 54% host-relay; the real throughput-scaling enabler.
6. **Scale the kernel D=64 → D=768** (the 124M layer): bigger matmul tiles, DDR weight staging, multi-SLR +
   replication → **train 124M → coherent Shakespeare**, instrumented (tok/s, comms, energy).
7. **Then MXFP8 for throughput** (4-bit-mantissa multiplies + pow2 block scale) + measure the FP8/FP6 floor PC
   tolerates (the novel precision angle) — the efficiency/scaling pass once quality is in hand.
