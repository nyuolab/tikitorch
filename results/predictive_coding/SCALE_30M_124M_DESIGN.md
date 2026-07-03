# Scaling the Distributed Predictive-Coding Transformer on AWS F2: D=64 → 30M → 124M

**Track:** model-parallel predictive coding (`forge_pc_xformer_mp_block`, ONE transformer block per
physical FPGA, no-global-sync neighbour messages). Distinct from the single-card DDR-streaming rig
(`forge_train_step_s1` / `TOKS_PROJECTION.md`).

**Status of this document:** DESIGN / analysis only. No kernel or config edits. Every quantitative
claim is tagged **[MEASURED]** (read off silicon or a csynth report), **[COMPUTED]** (a closed-form
count from the layout), or **[PROJECTED]** (a model-based extrapolation whose inputs are themselves
measured or computed). The reader should trust MEASURED, plan against COMPUTED, and gate PROJECTED.

**Provenance anchors (all verified for this doc):**
- Kernel: `/private/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_block.cpp`
- Config: `forge_pc_xformer_mp_config.hpp` (MP_* float layout) + `forge_pc_xformer_config.hpp`
  (base X_* dims: `X_D=64`, `X_H=2`, `X_DH=32`, `X_FFN=256`, `X_SEQ=64`, `X_VOCAB=64`; wide-cache
  layout incl. `XO_P = H*SEQ*SEQ` PRAW) — **[MEASURED from source]**
- Driver: `/private/tmp/tt-pc/tikitorch/fpga/silicon/forge_pc_xformer_mp_shard.cpp`
- CPU twin: `/tmp/mp_chain_sched.py` (exact-kernel numerics, glibc order, INIT_DIR)
- csynth: `/private/tmp/tt-pc/results/predictive_coding/multi_fpga/mp_csynth/mp_csynth.rpt`
- Cross-box fabric: `/private/tmp/tt-pc/tikitorch/fabric/tcp_link.py` (bit-exact NeighborLink backend)

**Do-not-touch invariant:** the design below assigns all datapath rewrites to a NEW kernel file
distinct from `forge_train_step_s1*`, `forge_attn*`, `forge_layernorm*`, `forge_fwd_gemm*`,
`forge_wide*`, `forge_wgrad*`, `forge_dyadic_adam*`, `cl_forge_morph_lm*`. The shard driver
(`forge_pc_xformer_mp_shard.cpp`) is in-scope for the TcpLink relay; the block KERNEL/config are not.

---

## 0. Executive summary

The current chain is **[MEASURED]** at D=64: 8 chips fully-pipelined **0.59 steps/s / 303 tok/s**,
bpb converged **3.093** (L=3, run #3), Gate-C clean (no global gradient). Resource at D=64: BRAM 61%,
URAM 48%, **DSP 15% (2553 of 3008 idle)**, LUT 74% per SLR.

Scaling to 30M (D=512) and 124M (D=768) turns on **one central constraint** and is unlocked by **one
central lever**:

- **The constraint:** per-block on-chip footprint grows **0.9 MB → ~50 MB (D=512) → ~235 MB (D=768)**
  against a **6.75 MB SLR BRAM** budget (18.25 MB incl. URAM). At D≥512 essentially *nothing* fits
  on-chip. **DDR/HBM weight+optimizer+activation streaming is MANDATORY.** This is the single biggest
  change and it lives inside the block kernel.
- **The lever:** 2553 idle DSPs. The current narrow MAC is **cyc/output ∝ I²** **[MEASURED from
  csynth]** and would take *tens of seconds* per kick at D=512. A **K-wide contraction engine**
  (K=128 at D=512) turns cyc/output into `ceil(I/K)`, projecting a **~44 ms fwd / ~87 ms bwd** kick
  and restoring today's per-step rate at 8× the parameters.

The headline de-risk finding: at every scale the workload is **compute-bound by 10–50×**, so DDR
bandwidth is *not* the bottleneck — weight streaming is a free background prefetch. Design for
compute-lane occupancy, not for bandwidth.

Honest scope: **30M on ONE f2.48xlarge is the high-value, low-risk milestone.** 124M is a two-box
TcpLink run whose per-kick time is *seconds*, whose wall-clock is *multi-week* for Chinchilla, and
whose per-token **energy LOSES to a GPU by 1–2 orders of magnitude**. The 124M win is **structural**
(no-global-sync, neighbour-only, constant-message-size-in-chip-count), not energy.

---

## 1. Config ladder + chip allocation

All parameter counts **[COMPUTED]** with tied embed/head, FFN=4D, per-block = `4·D² + 2·D·FFN`.

| Target | D | L | H | FFN | VOCAB | SEQ | Blocks params | Embed | **Total** | Chips | Boxes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| current | 64 | 3 | 2 | 256 | 64 | 64 | — | — | ~0.1M | 3–8 | 1 (partial) |
| **30M** | 512 | 8 | 8 | 2048 | 8192 (BPE) | 256 | 25.2M | 4.2M | **29.5M** | **8** | **1 × f2.48xlarge** |
| **124M** | 768 | 12 | 12 | 3072 | 50257 (GPT-2 BPE) | 1024 | ~120M | 4.2M* | **124.3M** | **12** | **2 × f2.48xlarge** |

\* embed at V=50257, D=768 ≈ 38.6M raw, tied; the 124.3M total is the task-supplied computed figure.

### 1.1 30M — one box, eight chips (unchanged topology, scaled dims)

| Chip | Role (CFG word) | Contents |
|---|---|---|
| 0 | `X_RL_INPUT` | token+pos embed, block 0 |
| 1..6 | `X_RL_HID` | blocks 1..6 |
| 7 | `X_RL_OUT` | block 7, final-LN, LM head, CE, sampling |

This is **exactly today's D=64 topology** with D/FFN/VOCAB/SEQ scaled — **zero new fabric wiring**,
one AGFI flashed to all 8 slots, role/phase from the CFG word. The *only* structural change is that
each block's on-chip footprint (~50 MB) exceeds one SLR, so the block kernel must run DDR-streaming
internally (§2). That is a kernel-internal dependency, not an allocation change.

### 1.2 124M — two boxes, twelve chips, one TcpLink boundary

**Recommended split: 8 + 4** (box A = chips 0..7 = INPUT + 7 HIDDEN; box B = chips 8..11 = 3 HIDDEN
+ OUTPUT). Rationale:
- Keeps ONE box **fully populated and identical to the 30M box** you already run (de-risk reuse).
- Puts the OUTPUT chip (head + CE + bpb readback + sampling, all host-orchestrated) on the *less
  loaded* box B.
- **Exactly ONE TcpLink boundary** (chip 7 ↔ chip 8), one TCP socket per direction.

A 6+6 split balances host DMA more evenly and is a fine alternative — the boundary count and wire
traffic are **identical** either way (a linear chain across two boxes always has exactly one
boundary). Pick by host-CPU staging balance, not by comms.

**Cross-box message size = SEQ·D floats/hop/direction, CONSTANT in chip count** — the PC advantage.
**[COMPUTED]**: D=768/S=1024 = 3.146 MB fp32, **1.573 MB bf16** (messages are already bf16-wide in
the kernel). 124M per-step boundary traffic (one boundary, ACCUM=8, fwd-down + bwd-up = 16 crossings):
**50.3 MB fp32 / 25.2 MB bf16**. Wire time at 100 Gbps cluster networking: **2–4 ms/step** —
**<1–3% of a multi-hundred-ms kick**, confirming "the boundary is never the bottleneck."

---

## 2. THE CENTRAL CONSTRAINT — memory hierarchy redesign (the single biggest change)

### 2.1 The footprint wall (why the current kernel does not scale)

**[MEASURED from source]** the current kernel `bload`s EVERY weight (`FWQ..FWDN`) and the whole wide
activation cache into on-chip `static` arrays up front, then runs `bf_matmul` / `bf_linear_bwd` /
`bf_adam_one` against those resident arrays. This works **only** because a D=64 block is ~1.05 MB.

Per-block DDR-resident state (fp32 masters + fp32 adam m/v + fp32 GACC + PRAW `H·SEQ²` attn cache +
wide act cache) **[COMPUTED]**:

| | D=64 / S=64 | D=512 / S=256 | D=768 / S=1024 |
|---|---|---|---|
| masters (fp32) | — | 12.6 MB | 28 MB |
| adam m/v (fp32) | — | 25.2 MB | 57 MB |
| **masters+adam** | ~0.9 MB total | **37.8 MB (5.6× SLR BRAM)** | **84.9 MB (12.6×)** |
| PRAW `H·SEQ²·4` | tiny | 2.1 MB | 48 MB |
| wide act cache | — | 10.5 MB | 100 MB |
| **block total** | **1.05 MB** | **~59.8 MB** | **~207.6 MB** |
| **one SLR fast RAM** | 18.25 MB (6.75 BRAM + 11.5 URAM) | — | — |

At D≥512 **essentially nothing fits on-chip → DDR/HBM streaming is MANDATORY.** F2 cards carry
16–64 GB DDR/HBM, so 207.6 MB/block × 12 blocks = **2.5 GB for the whole 124M model — well under
16 GB [COMPUTED]. Green.**

### 2.2 The headline de-risk: it is COMPUTE-BOUND, so streaming is cheap

**[PROJECTED, roofline]** with K-wide lanes (64 @ D512, 128 @ D768), fclk 250 MHz, and a conservative
single 512-bit AXI port at ~12 GB/s:

| Phase | DMA | Compute | Verdict |
|---|---|---|---|
| D512 FWD | 1.05 ms | 54.5 ms | **compute-bound 52×** |
| D512 BWD | 8.39 ms | 107 ms | **compute-bound 13×** |
| D768 FWD | 2.36 ms | 277 ms | **compute-bound 117×** |
| D768 BWD | 18.9 ms | 528 ms | **compute-bound 28×** |

Arithmetic intensity (fwd) = **69 MAC/byte (D512) → 313 MAC/byte (D768)** because each streamed
weight is reused across ALL SEQ tokens (256 / 1024). **Design consequence:** treat weight DMA as a
free background prefetch and optimize for compute-lane occupancy. This conclusion is *robust to the
shell's DDR wiring* — even one single 512-bit port is 10–50× over-provisioned.

### 2.3 The redesigned datapath — three double-buffered streams

Keep the existing numeric primitives (dyad-hoist forward, 16-lane partial-sum II=1 reduction, bf16
STE backward, dyadic-Adam). **Change only residency**, in a NEW kernel file.

**(1) Matmul weight stream (weight-stationary).** Instead of `bload_f(FWQ, D*D)` up front, stream W
as **128×128 fp32 tiles** (0.064 MB; double-buffered = 0.13 MB), dyad each tile on arrival, MAC it
against a resident activation strip. Ping-pong `bind_storage=ram_2p`: burst tile(k+1) into buffer B
while the PE consumes tile(k) from buffer A. Prefetch of tile k+1 finishes 50–100× before the PE
drains tile k → no stall.

**(2) The real on-chip pressure is the ACTIVATION STRIPS, not the weight tile.** `SEQ·FFN·4` = 2.1 MB
(D512) / **12.6 MB (D768) — already blows the 6.75 MB SLR BRAM on its own**. So the tiling MUST be
2-D: block over the weight (O,I) dims **AND row-block the SEQ dimension** into strips (e.g.
`SEQ_TILE=128` rows). Process `SEQ_TILE` tokens × a resident weight tile, accumulate
`y[SEQ_TILE × TILE_O]` in BRAM, stream the output strip out, advance. Resident footprint stays
**~0.4–1.0 MB regardless of D/SEQ**. The wide activation cache stays DDR-resident and is re-streamed
row-blocked in bwd (same tiling).

**(3) Adam read-modify-write stream.** `bf_adam_one` already touches each element once (II=1). Stream
it: burst-read [master tile, m tile, v tile], update in-place, burst-write. **Store m/v as bf16 in
DDR** (halves 57 MB → 28 MB at D768) — safe because the golden already rounds m,v through bf16 and
adam is memory-touch-once, so bf16 storage adds no reuse loss. **Master stays fp32** (STE needs the
fine bits). Read bf16 → expand fp32 in-lane → update → round bf16 → write.

**(4) Gradient-accumulator (`MP_GACC`) RMW stream.** For each output tile: read the DDR gacc tile, add
the freshly-computed dW tile, write back (`GZERO`=first-subbatch skips the read). Tile-blocked,
double-buffered, compute-bound.

**SLR placement:** partition the gmem into up to 4 HBM pseudo-channels (weight / activation-cache /
adam / message I/O) so the streams don't serialize — but this is a *nicety, not a necessity* (one
port is already 10–50× over-provisioned). If the shell exposes only one DDR4 channel, the
conclusion holds.

### 2.4 Activation checkpointing — the PRAW recompute win (composes with §2.3)

**[MEASURED from source]** the BWD phase `bload_f`s exactly: 9× `SEQ·D`
{WR,WX1,WQ,WK,WV,WATTN,WM,WX2,WRP}, 2× `SEQ·FFN` {WHD,WRMASK}, **PRAW = `H·SEQ²`**, SEQ-scalar
normalizers {ln1ss,ln2ss}; OUTPUT chip adds WXF (`SEQ·D`) + WLG (`SEQ·VOCAB`). Totals **[COMPUTED]**:
D512/S256 = **19.0 MB**; D768/S1024 = **298 MB** (OUTPUT chip). With S=8 pipeline slots each holding
a full copy it is **8× worse**. Checkpointing must shrink the **per-slot resident set**, not just
total DDR traffic.

**Three-tier policy (size × reuse):**

- **KEEP on-chip (tiny, read every step):** LN normalizers `ln1ss/ln2ss/lnfss` (SEQ scalars, 1–4 KB)
  and the running residual-grad `gR/gM/gRP` (one `SEQ·D` live). These are the analytic-LN-backward's
  only stop-grad-surviving inputs — never worth recomputing.

- **RECOMPUTE PRAW (the headline).** Do NOT cache the `H·SEQ²` softmax-prob tensor. In attention
  backward, recompute each `(h,i)` row of Praw on the fly from the ALREADY-resident WQf/WKf:
  `score = QK/√DH → max → exp → normalize` (the exact fwd order/rounding), giving `p[j]` for the
  gp/gV/Jacobian inner loops. **Zero extra loads.** Cost **[COMPUTED]** = `H·SEQ²·DH/2` QK-MACs +
  `H·SEQ²/2` exps: 16.78 MMAC (D512) = **1.12% of bwd**; 402.7 MMAC (D768) = **2.81% of bwd** (1.88%
  of fwd+bwd). **Saves 2 MB (D512) / 48 MB (D768)** on-chip AND its fwd→DDR write. The cached
  alternative is a 7–13 ms DDR round-trip at D768 — recompute is *cheaper AND removes 48 MB*. This is
  the FlashAttention-style score recompute; unambiguously correct.

- **STREAM-TILE the big linears** (D768) — see §2.3.

**Post-checkpoint on-chip working set [COMPUTED]:**
- **D512:** attn-bwd WS = `7·SEQ·D` = 3.5 MB; FFN-down WS = `2·SEQ·FFN + SEQ·D` = 4.5 MB. **Both
  < 18.25 MB SLR → NO sequence tiling needed at D512.**
- **D768:** attn-bwd WS = 21 MB > SLR. Sequence-tile the query axis (`Tq=256` → 14.25 MB). But gK,gV
  are scatter-accumulated over all keys `j≤i`, so `4·SEQ·D` = **12 MB is an IRREDUCIBLE single-pass
  floor** (K,V + gK,gV accumulators); only query-indexed tensors tile. FFN-down: `Tf=512` → 7 MB.
  All fit after tiling.

**Do NOT recompute the FFN hidden (WHD).** Recompute costs a full up-proj (268 MMAC = 16% of bwd at
D512; 2.42 GMAC = 15% at D768) AND does not shrink the WS (WHD must be materialized in full as the
wgrad operand). Verdict: **recompute PRAW, STREAM WHD from DDR**, and let WRMASK be recomputed as
`sign(WHDpre)` for free during the FFN backward.

**S-slot re-runnability (correctness).** PRAW recompute reads only slot-indexed WQf/WKf
(`MP_WQ+po`/`MP_WK+po`) and persistent single-slot masters — so slot g's backward recomputes slot g's
Praw with **no cross-slot hazard**, the same invariant the GACC RMW already relies on. Dropping PRAW
from the per-slot cache shrinks the replicated span by `H·SEQ²` words → **8× that saving in total
DDR**.

**In-scope flag (larger than PRAW at 124M):** the OUTPUT chip's **WLG logits cache = `SEQ·VOCAB` =
8 MB (D512) but 205 MB (D768!)** — *bigger than PRAW*. The CE backward recomputes softmax from WLG;
if WLG is vocab-tiled the head backward tiles too. This is a *larger* activation-memory problem than
PRAW at 124M and belongs on this axis — see head-sharding (§3.1(4)).

---

## 3. The wide-MAC compute engine (spend the idle DSP)

### 3.1 Root cause and the engine

**[MEASURED from csynth]** the two clean matmul instances at D=64: `bf_matmul_8` (I=64, 16384 outs) =
1,835,061 cyc = **112 cyc/out**; `bf_matmul` (I=256, 4096 outs) = 7,340,087 cyc = **1792 cyc/out**.
cyc/output is **exactly 16× for a 4× I increase → cyc/output ∝ I²** (c2 = 0.02734). The `ps[i&15]`
16-lane partial-sum split does NOT hold II=1 at large I: the output loop is un-pipelined across
outputs, so every output pays a fresh fill + tree-drain, effective inner II ≈ I/16. **This is the
scaling wall.** On the narrow MAC, a D=512 HID kick is **~150 s fwd / ~188 s bwd [PROJECTED]** — the
task's "multiple seconds" framing is optimistic.

**The K-wide engine (one instance/chip, reused across all matmuls):**

1. **Inner contraction unrolled K-wide** → K products/cycle feeding a **balanced adder tree** of depth
   `ceil(log2 K)` (K=128 → depth 7). The tree is FEED-FORWARD (pipelined combinational), NOT a
   loop-carried accumulator, so it pipelines at **inner II=1 for any K**. The only loop-carry is the
   cross-tile accumulator over `ceil(I/K)` tiles (touched once per `ceil(I/K)` cyc ≫ tree latency →
   II=1 safe). Forward = dyad-shift lanes (mult-free, DSP-as-adder in the tree, **DSP_MULT=0
   preserved**); backward route/wgrad = K bf16 multiplies (the sanctioned "new DSP").
2. **cyc/output = `ceil(I/K)`** — kills both the I² term and the per-output re-drain.
3. **Operand feeding (composes with §2):** weight-stationary. Stream an `(O_tile × I)` weight tile
   ONCE into a K-bank-partitioned URAM buffer, reuse K-wide across all M=SEQ rows. One 512-bit gmem
   word delivers 16 fp32 / 32 bf16 operands/cyc; the K-bank URAM partition supplies K operands/cyc.
4. **HEAD is a SECOND wall on the OUTPUT chip:** the `D×VOCAB` head is **57% of OUTPUT fwd MACs at
   D512, 85% at D768** (V=50257). **FIX: shard the head+CE across all L chips** (each computes a
   `VOCAB/L` logit slice), which *also* fixes the 205 MB WLG cache of §2.4. This changes the CE
   reduction and cross-chip message shape → verify bpb-equivalence in the twin first.
5. **Attention reuses the same engine** (a minority of block MACs even at SEQ=1024: 7.8% @ D64, 4.0%
   @ D512, 10% @ D768) — no special treatment; its only scaling need is the PRAW recompute of §2.4.

### 3.2 Resource + projected kicks

**DSP/LUT budget per K [COMPUTED]** (VU47P SLR = 3008 DSP / 434k LUT; D=64 today uses 743 DSP /
232.7k LUT; bf16 mul = 3 DSP, fp-add ≈ 2 DSP or 350 LUT, dyad-shift ≈ 60 LUT):

| K | DSP | LUT |
|---|---|---|
| 16 | ~648 | 126k |
| 32 | ~696 | 133k |
| 64 | ~792 | 146k |
| 128 | ~984 | 172k |

Max K by DSP ≈ 800 lanes; by LUT ≈ 765 lanes. **DSP/LUT is NOT binding for K≤128** (>2300 DSP idle at
K=32). Timing closure at 150 MHz for a single clean engine realistically caps ~K=64–128.

**D=512 (30M) per-chip kick @150 MHz [PROJECTED]** (+3 ms LN/softmax/adam overhead; head sharded):

| K | HID fwd | HID bwd | steps/s (S=8) | tok/s |
|---|---|---|---|---|
| 16 | 353 ms | 695 ms | 1.24 | 317 |
| 32 | 178 | 349 | 2.47 | 632 |
| 64 | 90 | 176 | 4.90 | 1255 |
| **128** | **44** | **87** | **9.66** | **2473** |

**K=128 at 30M ≈ today's per-step rate (0.59 steps/s MEASURED) at 8× the params.** vs narrow MAC
(~150 s / ~188 s) this is a **~1700–2100× reduction**.

**D=768 (124M) pipelined throughput [PROJECTED]** (head sharded across 12 chips):

| K | bottleneck | steps/s | tok/s |
|---|---|---|---|
| 32 | 4647 ms | 0.22 | 220 |
| 64 | 2325 | 0.43 | 440 |
| 128 | 1164 | 0.86 | 880 |
| 256 | 584 | 1.71 | 1755 |

**Honest framing:** the wide-MAC makes **30M viable at K=128 (~tens of ms/kick)**; **124M is
seconds-per-kick**, not tens-of-ms — the 7.2 GMAC forward block is intrinsically too big without a
replicated engine (2–4×) or a 300 MHz clock.

**Backward is the bottleneck** (bwd = dx + wgrad = 2× fwd MACs) — size K to the bwd kick. The
wide-MAC must preserve the Lever-A 2× unroll and the KAT1 bit-exact / KAT2 relerr ≤5e-3 gates, but
the **reduction order changes** (tree vs `[i&15]`), so gradients will NOT be bit-exact to the current
kernel — needs a fresh CPU-twin reconcile (within-1-ULP), not a bit-exact port.

---

## 4. Numerics-at-scale + de-risk go/no-go gates + target bpb

### 4.1 The known numeric premium

**[MEASURED]** at L=8 D=64 the deep-chip forward residual grew to ~1100 (harmless at D=64 fp32, but a
*scaling flag*). bf16 messages + partial-sum reorder + fp32 arith give a **0.14–0.26 bpb gradient-
noise premium** vs the fp64 CPU twin, and the premium **grows as LR decays**. Fixes: **cosine LR
schedule + batch de-noising via replication (ACCUM)**.

At D512/D768 with fp32 masters the residual magnitude still has headroom, but: (a) wider D → more
terms per reduction → the partial-sum lane count may need 16→32 to bound the reorder error; (b) the
wide-MAC's tree reduction *changes the order again*; (c) softmax over 1024 keys in bf16 at SEQ=1024
may need an **online running-max** (FlashAttention-style) rather than the current two-pass row-max.
All are flags, not blockers.

### 4.2 Target bpb (all references honest)

Prior art: S1 mult-free char-LM **[MEASURED] bpc 2.4230**; the substrate carries a **~0.3–0.5 nat
tax** vs a same-param backprop nanoGPT (fig_hallmark / S1 lesson). So the **30M target** is: converge
to within ~0.3–0.5 nat of a nanoGPT(D=512/L=8) on the identical BPE tokenizer + data; the current
chain already reaches **bpb 3.093 [MEASURED] at L=3 D=64**, so descent is established — the question
is whether the premium stays bounded as D and L grow. The **124M target** is *first coherent output*
(readable BPE samples), not Chinchilla-optimal loss.

### 4.3 CPU de-risk go/no-go gates (the twin is the cheap oracle)

The twin `/tmp/mp_chain_sched.py` (or a copy under `/tmp/wf30_*`) is exact-kernel numerics in glibc
order. **Gate order (each blocks the next):**

- **G0 — CSYNTH-ONLY gate (process rule, per the HLS reduction-II lesson).** Synth the streaming
  tiled matmul AND the K-wide engine at D=64 (K=16, K=32) in a NEW kernel file; read the per-loop II
  table. **Require inner II=1 and cyc/output = `ceil(I/K)` on both the m_axi-burst loop and the MAC
  loop before ANY AFI build.** This directly falsifies the I² model if the tree fails to amortize.
- **G1 — streaming bit-exactness.** The tile/row-block reorders the OUTER loop, not the inner 16-lane
  tree → should be **bit-exact** to the resident kernel at D=64, then D=512. Prove it in the twin.
- **G2 — PRAW-recompute numerics.** Flip PRAW from bload to recompute in the D=64 kernel; require
  **KAT1 bit-exact / KAT2 relerr ≤5e-3** against the cached-PRAW golden (the `MP_DEBUG_GRADS` dump
  already exists). Then `sync == recompute` bit-exact in the twin at D=64/S=64.
- **G3 — wide-MAC reconcile.** Extend the twin to the K-wide tree order; verify wide gradients
  within-1-ULP of the 16-lane kernel; re-run KAT1/KAT2; establish the new gradient-noise premium at
  K=16/32.
- **G4 — 30M convergence trajectory (twin).** Run the twin at D=512/L=8, bf16 messages ON, cosine LR,
  ACCUM≥8; confirm bpb descends and the premium stays bounded, and watch ACTSTATS residual magnitude
  across chips. **Go/no-go for the 30M silicon spend.**
- **G5 — adam/gacc bf16-in-DDR.** Unit-test read-expand-update-round-write vs `bf_adam_one`; require
  bit-exact m,v evolution (the golden already bf16-rounds → should be identical). Confirm II=1.
- **G6 — 124M premium (twin).** Twin at D=768/L=12, SEQ=1024 (or 512); confirm the premium does not
  compound with the deep-chain residual growth before committing multi-week silicon.

---

## 5. 124M convergence + energy + output-assessment methodology

### 5.1 Convergence run

**Token budget [COMPUTED]:** Chinchilla ~20 tok/param → 124M = **2.5B tokens** compute-optimal; a
**0.3B-token** budget is the honest *first-coherent-output* milestone (readable samples appear well
before Chinchilla).

**Per-block step MACs [COMPUTED]** (D=768, FFN=3072): fwd = `4D² + 2D·FFN` = 7.08 M MACs/tok; ×3
(fwd + input-grad + weight-grad) = **21.2 M MACs/tok-step**.

**Wall-clock bands [PROJECTED]** (chain rate ≈ per-block rate because pipelining overlaps L stages;
per-PE GMAC/s reused from `TOKS_PROJECTION.md` MEASURED `forge_wgrad_df3`):

| tok/s (chain) | basis | 2.5B tok | 1B tok | 0.3B tok |
|---|---|---|---|---|
| ~497 | 1-PE floor | 58 d | 23 d | 6.9 d |
| **~739** | **1-PE steady (DEFENSIBLE planning number)** | **39 d** | **16 d** | **4.7 d** |
| ~3697 | 5-PE + overlap (upside, un-landed) | 7.8 d | 3.1 d | 0.9 d |
| ~4436 | 6-PE (upside) | 6.5 d | — | — |

**Plan against ~739 tok/s → ~39 d Chinchilla / ~4.7 d first-coherent-output; treat 5–6 PE as upside.**

**Checkpoint/resume:** each block-chip persists fp32 master + bf16 m/v (+ deploy int16) ≈ **85 MB/chip
at D=768, ~1.0 GB for all 12 [COMPUTED]**. The shard already reads masters back from DDR at run end
(`readf MP_FWQ.. → dump_master_file`) and `INIT_DIR` reloads them → resume = dump every ~30 min +
relaunch with INIT_DIR. Dump ≈ 0.1 s/chip over PCIe (negligible). **Add a step counter + cosine-LR
phase to the dump** so LR resumes correctly.

**Eval:** held-out bpb (OUTPUT chip MP_WLG logits → CE) as a no-backward forward sweep, periodically.

### 5.2 Energy — the honest, unfavorable story

**[MEASURED]** DSP_MULT=0 forward. The forward's dynamic draw is **below the Vccint sensor floor**
(morph ~4 W core, statistically identical idle vs active). The only defensible measured number is
**per-FPGA Vccint core ~4 W (a FLOOR, NOT total-card).**

**[PROJECTED]** per-token board energy — PC-MP spends **12 whole FPGAs on ONE 124M chain**:

| basis | power | mJ/tok @739 | mJ/tok @3697 |
|---|---|---|---|
| Vccint core (measured floor) | 48 W (4×12) | 65 | 13 |
| full-card estimate | 480 W (40×12) | 649 | 130 |
| generous TDP | 900 W (75×12) | 1216 | 243 |

**Reference:** A10G **[MEASURED board] 1.46 mJ/tok.** → **PC-MP LOSES by 9–170×.** Defensible claims
only: (1) DSP_MULT=0 measured; (2) Vccint ~4 W/FPGA measured floor; (3) per-token board energy
PROJECTED and currently unfavorable. **Do NOT claim a board-to-board energy win** — that needs
external PMBus full-card instrumentation. **The win is STRUCTURAL, not energy.**

### 5.3 Output assessment

The shard's SAMPLE path already generates (roll SEQ context through the pipeline, sample from OUTPUT
chip MP_WLG last-token logits, map ids → text host-side). Protocol:
1. Generate N samples from the trained masters.
2. Train a **same-param backprop nanoGPT (D=768/L=12)** on the identical tokenizer+data; report the
   bpb gap (expect the substrate's ~0.3–0.5 nat tax).
3. **Mult-free-DEPLOY quality gate** — the deploy int16/CSD forward must match the training wide-float
   forward (the S1 lesson: pow2-only D=64 was *bigram-class*; the deploy gate caught the train/deploy
   gap).

Assess: held-out bpb; sample coherence (human read + n-gram overlap); deploy-vs-train bpb delta.

### 5.4 TcpLink cross-box port (124M)

**[MEASURED]** `tcp_link.py` already provides the bit-exact primitive: `make_tcp_chain(num_stages=12,
boundary=8, role='left'|'right')` — box B (right) LISTENS, box A (left) CONNECTS with retry (framed
protocol, MAGIC `0x54504C31`, crc32 trailer, per-message re-verify; `tests/test_tcp_link.py` proves
the 2-process split is bit-exact to ThreadLink in sim).

**The silicon port is a ~40-line relay local to the boundary chip.** Today the shard relays boundary
messages via PCIe readback into a host float buffer (`readf(l, MP_GROUT) → put_f(l+1, MP_GRIN)`). At
chip 7 (box A), after the FWD kick, instead of `put_f` into a local chip-8 shadow, run one
`_TcpEndpoint.send` of the `SEQ·D` array; box B's driver `recv`s into chip-8's `MP_GRIN` before its
kick. Symmetric for the backward up-message. **Reuse `encode_msg`/`_decode_stream` verbatim** (pure
numpy↔bytes). **Add a keepalive + resume-from-last-tag wrapper** — TcpLink retries on connect but NOT
mid-run, so a transient TCP drop over a multi-day run would kill it. The PIPE=1 event loop and ACCUM
window are UNAFFECTED (the boundary is just two neighbour hops; `fdone`/`bdone` treat a hop as
opaque). *Note: this edits the shard driver, which is in-scope; the forbidden `forge_*` files are the
KERNEL/config.*

**Staleness gate.** Silicon-measured weight-staleness tolerates only ~1 step (stale-8 = +2.0 bpb,
unrecoverable). The runner needs `W ≥ 2·(L-1) = 22` at L=12. At ACCUM=8 the window is **below 22** —
the accumulation window / slot depth may need to grow with L. **Confirm W ≥ depth holds at L=12
before committing the run.**

---

## 5.9 PHASE-0 RESULT (2026-07-03, MEASURED): the recipe's numerics HOLD at 30M scale

D=512/L=8/FFN=2048/SEQ=256/V=256(byte) = **25.56M params**, exact-kernel CPU twin (dyadic fwd + bf16 STE bwd
+ dyadic-Adam + cosine + ACCUM=8), enwik8 20M-token bin, 500 steps (~1M tokens seen — deliberately
undertrained; the gate is trajectory shape):
- **bpb 8.42 → ema 4.48 (batches 3.6–4.2), monotone through the full cosine decay** — no plateau, no
  divergence, no NaN. D=64 regression passed first (recipe intact after parameterization).
- **Residual BOUNDED + self-regulating at depth 8 / D=512**: peak ~476 @step 225 → declines to ~120–170.
  The depth-blowup risk is REFUTED at this scale.
- **No dyadic-Adam underflow**: |W|max grows 0.044→0.24; descent continues at LR=3e-4.
- **★G4 COMPLETE — float control (MULTFREE=0, same everything): the mult-free recipe WINS at matched HPs.**
  ema 4.476 vs float 4.753 (batches 3.588 vs 3.866) at 500 steps; float's residual blew to ~1600 (vs the
  mult-free's stable ~120–170) — the dyadic/qact stack acts as a REGULARIZER at this scale. Honest scope:
  single seed, 1M tokens, HPs tuned for neither — the claim is "NO early-training quality tax at 30M;
  ahead at matched HPs", not "beats float at convergence". **PHASE 0 = GO for silicon spend.**

## 6. Phased execution plan

Each phase names its **build**, its **gate**, and whether it touches silicon. **No phase advances
until its gate passes.** CSYNTH_ONLY precedes every perf rebuild (process rule).

### Phase 0 — CPU recipe de-risk (no silicon, cheapest, highest leverage)
- **Build:** extend/copy the twin under `/tmp/wf30_*` with (a) tiled/row-blocked reduction order,
  (b) PRAW-recompute flag, (c) K-wide tree order, (d) bf16 adam/gacc in-DDR, (e) head-sharding CE
  split, (f) D=512 and D=768 configs with bf16 messages + cosine LR + ACCUM≥8.
- **Gate:** G1 (streaming bit-exact D64→D512), G2 (recompute KAT1/KAT2), G3 (wide-MAC within-1-ULP),
  **G4 (30M bpb descends, premium bounded)**, G5 (adam bf16 bit-exact), and head-sharding
  bpb-equivalence. **Go/no-go for any silicon spend.**

### Phase 1 — the streaming + wide-MAC kernel redesign at D=512 (csynth-gated, no AFI)
- **Build:** a NEW kernel file (not the do-not-touch `forge_*`) implementing the three double-buffered
  streams (§2.3), PRAW recompute (§2.4), and the K-wide engine (§3) at D=512 dims. Sweep
  K∈{16,32,64,128}.
- **Gate G0:** per-loop csynth table shows **inner II=1** on the m_axi-burst loop and the MAC loop,
  **cyc/output = `ceil(I/K)`**, attn-bwd/FFN-down WS inside 18.25 MB SLR, and the resource sweep picks
  the **max K that closes 150 MHz** (LUT/DSP say ≥128 fits; timing is the real cap). **No AFI until G0
  is green.**

### Phase 1.5 — D=256 stepping stone on ONE FPGA (validate the roofline on silicon)
- **Build:** the redesigned kernel at D=256 (4–6M params), one block, one FPGA (the S1-isolated F2
  box, never the shared `tiki-f2-silicon`).
- **Gate:** measure fwd/bwd kick; back out achieved MAC/s and HBM GB/s; **confirm the point sits on
  the compute roof (dma ≪ compute)**, validating the 50× margin before D=512. This is the first
  MEASURED replacement for the PROJECTED roofline.

### Phase 2 — 30M one-box run (8 chips, no TcpLink) — the high-value milestone
- **Build:** flash the D=512 K=128 AGFI to all 8 slots of ONE f2.48xlarge; roles from the CFG word;
  DDR-streaming block; head sharded across 8 chips.
- **Gate:** measured fwd/bwd kick vs the **44/87 ms projection**; pipeline throughput vs the **9.66
  steps/s** target; **bpb converges to within ~0.3–0.5 nat of nanoGPT(D=512/L=8)**; deploy-vs-train
  bpb delta acceptable. **This convergence is the go/no-go for 124M.**

### Phase 3 — TcpLink silicon port (2-box, sim-verified first)
- **Build:** port the ~40-line relay into the boundary chip of `forge_pc_xformer_mp_shard.cpp`; add
  keepalive + resume; wire box A (connect) ↔ box B (listen).
- **Gate:** run `make_tcp_chain(12, boundary=8)` across two processes with the twin numerics and
  **assert bit-exact to the single-process 12-block ThreadLink chain**; then the ported relay's wire
  bytes match the Python twin **frame-for-frame** (same crc32/message); measured TCP wire time in a
  100 Gbps placement group **< 3% of a kick**. Confirm cluster-placement + 100 Gbps SKU. Confirm
  **W = 2·(L-1) = 22** staleness window (grow MP_SLOTS/ACCUM if needed; verify it fits DDR).

### Phase 4 — 124M two-box run + convergence + assessment
- **Build:** 8+4 split across two f2.48xlarge; the D=768 kernel (K=128–256, replicated engine or
  300 MHz if tens-of-ms is required — else accept ~0.86–1.71 steps/s); ~30-min checkpoint cadence.
- **Gate:** measured per-block kick at D=768/S=1024 (the single biggest remaining uncertainty —
  drives 6.5 d..58 d); **first-coherent-output at ~0.3B tokens (~4.7 d @ 739 tok/s)** as the primary
  milestone (Chinchilla-full is the stretch, not required — energy does not improve with a longer
  run); output assessment per §5.3; energy logged at the instance level and reported as PROJECTED +
  unfavorable with the Vccint ~4 W/FPGA measured floor.

---

## 7. Consolidated MEASURED vs PROJECTED ledger

**[MEASURED]:** current 0.59 steps/s / 303 tok/s (8 chips, D=64); bpb 3.093 (L=3); resource D=64
(BRAM 61% / URAM 48% / DSP 15% / LUT 74%); FWD kick 53.7 ms / BWD 52.2 ms; cyc/output ∝ I² (112 vs
1792 cyc/out from csynth); S1 bpc 2.4230; A10G 1.46 mJ/tok board; Vccint ~4 W/FPGA floor;
weight-staleness stale-8 = +2.0 bpb; L=8 D=64 residual ~1100; 0.14–0.26 bpb gradient-noise premium;
`tcp_link.py` 2-process split bit-exact in sim.

**[COMPUTED]:** the entire parameter/footprint ladder; message sizes; PRAW-recompute MAC fractions
(1.12% / 2.81%); working-set sizes and tiling floors (12 MB irreducible attn-bwd @ D768); checkpoint
sizes (~85 MB/chip, ~1.0 GB); per-step MACs (21.2 M/tok-step); DSP/LUT per K; boundary traffic
(25–50 MB/step); WLG cache 205 MB @ D768.

**[PROJECTED]:** all kick times at D512/D768 (K-wide model); throughput bands (steps/s, tok/s);
roofline compute-bound ratios (10–117×); wall-clock (6.5–58 d); per-token board energy (9–170× worse
than A10G); the ~150 s narrow-MAC baseline. **The single biggest uncertainty is the MEASURED D=512 /
D=768 per-block kick once DDR-streaming is in the kernel** — Phases 1.5 and 2 exist to convert it from
PROJECTED to MEASURED before the multi-week 124M spend.

## 8. Top open questions gating the plan

1. **Measured D512/D768 per-block kick** with DDR streaming — drives the entire wall-clock; retire in
   Phases 1.5/2.
2. **What clock does the K=128 wide engine close at?** All projections use 150 MHz; S1 fell to
   62.5 MHz under BRAM congestion. A K=128 tree may force <150 MHz, scaling every number linearly —
   pblock the wide cone to one SLR, get the timing report (G0).
3. **Single K=128 engine vs replicated (e.g. 4× K=32)?** Same aggregate MAC/cyc; replication may
   close timing better (smaller cones). Decide from the resource+timing sweep.
4. **SEQ=1024 vs SEQ=512 at 124M** — param count is set by D/L/FFN/VOCAB, *not* SEQ. Halving SEQ drops
   PRAW 4× (48→12 MB) and lets attn-bwd fit without Tq-tiling. Cheapest single lever; needs a
   product-side answer.
5. **F2 HBM channel count / sustained per-bundle bandwidth** — sets whether the 4-stream split is
   needed and the recompute-vs-stream tradeoff at D768. Compute-bound conclusion holds either way;
   micro-benchmark in Phase 1.5.
6. **Staleness window at L=12** — does MP_SLOTS=8 + ACCUM cover the needed W=22, and does growing it
   fit DDR? Gate in Phase 3.
7. **Budget/authority** for a 6.5–58 d two-box run — is the milestone Chinchilla-full or the ~0.3B /
   ~5-day coherent-output proof? The energy story does not improve with a longer run, so the
   coherent-output proof is the higher-value, lower-cost scope.
