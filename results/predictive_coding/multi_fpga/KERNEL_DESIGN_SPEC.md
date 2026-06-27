# KERNEL DESIGN SPEC — Mixed-Precision Distributed PC-Transformer Block (`forge_pc_xformer_mp_block`)

**Status:** BUILD-READY. Synthesizes the 4 recon passes, the measured precision floor, and the 4 design lenses (numerics / hardware / accumulation-pipeline / file-plan+verify) into a single contract for the next kernel + driver.
**Date:** 2026-06-27. **Worktree:** `/tmp/tt-pc` (branch `tropical-predictive-coding`, unpushed). **User runs all F2 builds.**

> **Naming honesty note.** Files carry the `_mp_` ("mixed-precision") infix, NOT `_bf16_`. The measured precision floor (below) makes the datapath **wide fixed-point (`fix16`), not IEEE BF16**. The kernel exposes the backward/optimizer grid width as a single `#define XM_GBITS` (default 16) so `bf16` survives as a *golden-only* witness mode (R1.d wants BF16-everywhere demonstrated), but the **shipping** datapath is fixed-point. The infix avoids a filename that lies about the chosen format.

---

## 1. GOAL + THE VALIDATED RECIPE + WHAT'S DECIDED

### 1.1 Goal
A **~124M** distributed PC transformer trained on the multi-F2 fabric in hours, via a **no-global-sync, bubble-free pipeline** (one transformer block per FPGA, model-parallel across `L` chips; neighbour hidden-down / error-up messages only; no all-reduce, no global gradient). The novel, precision-independent thesis is the **distributed-no-global-sync architecture**. This spec covers the next stage: the **mixed-precision sibling** of the on-silicon integer block.

> **Reframe (the int kernel is ALREADY a full train kernel).** The int `forge_pc_xformer_block` already synthesizes the **complete** forward + backward cascade + dyadic-Adam optimizer, with **DSP_MULT=0 on every dyadic op** (route, wgrad, adam, LN-bwd). The mp change is therefore **NOT "adding a backward"** — the backward and optimizer already exist and are csynth-clean. The mp change is a **narrow precision swap on two operations**: (a) the wgrad outer product `pow2×pow2 → fixed-point-mul`, and (b) the optimizer master/state `int16-grid → fix16-grid`. Everything else (forward, error routes, attn-bwd, LN-bwd structure, window accumulation) is structurally identical to the int kernel. The DSP **delta** is only the new wgrad multiplies (see §3) — the int kernel's existing 185 DSP are carried over unchanged.

### 1.2 The validated recipe (measured, cite numbers)
Triangulated on the torch char-LM twin (`scripts/pc_xformer_charlm.py`, TinyShakespeare, same recipe/scale), per `BUILD_PLAN.md` (locked 2026-06-27):

| recipe | precision | measured bpb | verdict |
|---|---|---|---|
| mult-free forward + **float** backward/optimizer | mixed | **2.016, COHERENT** (autograd witness) | recognizable Shakespeare |
| **full BF16** (fwd+bwd+opt) | BF16 | **1.732, MORE coherent** (autograd witness) | character names, dialogue |
| **fully-integer** (int cascade bwd + int16-master dyadic-Adam) | int | **caps ~0.20, GIBBERISH** | the wall |

> **★ HONESTY (these are AUTOGRAD witnesses, not the cascade).** The **2.016 / 1.732 / 2.898** numbers (and the §1.3 floor numbers) come from **torch AUTOGRAD**, NOT from the PC-cascade datapath the kernel transcribes. They are **witnesses** that the *precision recipe* is sound — they do NOT yet prove the kernel's **cascade** backward/optimizer datapath reaches the same place. **The cascade-fix16 result is PENDING Gate G2** (the standalone golden-cascade training experiment, running now). Do not cite these as proof of the kernel's cascade datapath; treat G2 as the load-bearing cascade proof.

**Triangulated conclusion (the load-bearing fact, autograd-witnessed):** the wall is the **fully-integer backward + int16-master optimizer**, NOT the multiplier-free forward (which is independently coherent at 2.016 bpb *in autograd*). The int16 master is clamped ±32767 with `X_LR_SHIFT=0` forced (any LR shift floors `dw` to zero → frozen weights) — that coarseness IS the wall.

### 1.3 The measured PRECISION FLOOR (decides format: fixed-point, NOT BF16)
Measured (mult-free forward `MULTFREE=1`, 1200 steps, TinyShakespeare, quantizing every `p.grad` post-backward + the dyadic-Adam `m,v,p.data` each step) — harness `/tmp/precfloor_charlm.py` (the tracked `scripts/pc_xformer_charlm.py` was NOT modified). **★ These floor numbers are AUTOGRAD witnesses** (quantized-autograd, NOT the PC cascade) — they establish the *format* floor; the cascade-fix16 confirmation is pending Gate G2.

| fmt | bpb | |
|---|---|---|
| float | 2.941 | reference |
| bf16 | 2.987 | works but **strictly dominated** |
| fix24 | 2.876 | coherent |
| **fix16** | **2.898** | coherent, **= or beats bf16** |
| fix14 | 2.885 | coherent (**true floor**) |
| fix13 | 5.8e9 | **DIVERGED** (by step 1199) |
| fix12 | NaN | **DIVERGED** (~step 750) |

**Verdict: FIXED-POINT SUFFICES.** `fix16` equals-or-beats `bf16`, so floating-point's wide dynamic range buys nothing here — the mult-free recipe already keeps every tensor on a per-tensor power-of-2 scale, and the dyadic-Adam step only needs `sign(m)` and the pow2 ratio `|m|/v`, so 14+ grid bits is plenty. The cliff is **sharp** (fix14 fine → fix13 explodes) and lives in the **optimizer-state dynamic range** — exactly the measured wall, NOT the mult-free forward.

### 1.4 What's DECIDED (BF16 vs fixed-point, per the floor)

| decision | choice | rationale |
|---|---|---|
| **Backward/optimizer format** | **`fix16` wide fixed-point** (default `XM_GBITS=16`; `fix24` for extra margin if a region is cheap; `bf16` golden-witness only) | floor: fix16 = or > bf16; an int multiply binds **1 DSP + ~0 normalize-LUT** vs BF16's **1 DSP + ~250–400 LUT/FF**; cheap INT DSPs not costly float DSPs |
| **Width margin** | ship **fix16** (2 bits over the fix14 cliff); `XM_GBITS` is one `#define` to widen to fix24 if scale re-confirm demands. **The fix16 cliff is a 2-bit, single-seed, 0.2M margin — `XM_GBITS` MUST be re-confirmed at real D/L with ≥3 seeds; DEFAULT TO `fix24` if the fix16 margin does not hold at scale.** | floor measured on a TINY config (0.2M, D=64, L=3, 1200 steps, single seed) — re-confirm at real D/L with ≥3 seeds before locking below 16; fix24 is the safe fallback |
| **Accumulator width** | **int64 (`acc_t`) throughout the cascade**, requantize to `XM_GBITS` only at the consumer (into m/v, into the grad-accumulator). Reserve int64 specifically for attn-bwd `pn*gp` (reaches ~2^63) | "accumulate wide, store narrow" — the floor's explicit recommendation |
| **Forward** | **UNCHANGED, bit-identical** to `forge_pc_xformer_block.cpp` (int16 acts + dyadic ±2^k weights). DSP_MULT=0 preserved; the one allowed forward multiply is the `p·V` blend at line 436 — **only 2 DSP** (NOT the 185-DSP anchor; see §3.2) | the forward is independently coherent (2.016, an autograd witness — §1.2); changing it would break the thesis and the on-silicon contract |
| **Neighbour messages** | **UNCHANGED int16 bands** (down `X_ABITS=9`, up `X_EBITS=12`) | activations/errors, not optimizer state; TYPE-1 tolerates coarse messages; keeps the new chip wire-compatible with the int chip |
| **One-SLR fit** | **single fused AFI** (no FWD/BWD CFG-split needed at D=64) | budget below; FWD/BWD split held as mitigation #4 |

---

## 2. THE MIXED-PRECISION DATAPATH (precise types per stage)

Core types (`forge_pc_xformer_mp_config.hpp`, copied from `forge_pc_xformer_config.hpp` + the deltas):

```cpp
#define XM_GBITS  16                          // fixed-point grid for grads + optimizer state (14=floor, 16=ship)
#define XM_QMAX   ((1<<(XM_GBITS-1))-1)       // symmetric clamp
typedef ap_int<16>      data_t;  // FORWARD activations + the forward-facing dyadic weight snapshot (UNCHANGED)
typedef ap_int<6>       exp_t;   // dyadic exponent (UNCHANGED)
typedef ap_int<64>      acc_t;   // ALL grad/error accumulators — STAYS int64 (pn*gp ~2^63; the width landmine)
typedef ap_int<32>      prob_t;  // Q16 attention probs (UNCHANGED)
typedef ap_int<XM_GBITS> fix_t;  // NEW: master-weight + Adam m/v mantissa, one per-tensor scale_exp per matrix
```

### 2.1 Forward — mult-free integer, UNCHANGED (do NOT touch)
Every op F0–F13 (`forge_pc_xformer_block.cpp:368-518`) is bit-identical. Activations `data_t`=int16; weights applied as `dyad(W)` → `±(x<<e)` signed shift (DSP_MULT=0). The single forward multiply is **F5 `p·V` blend** (`block.cpp:436`): `blend += (acc_t)p[j] * (acc_t)V[...]` (Q16 prob × int16 V) — this is **only 2 DSP**, NOT the 185-DSP anchor. The bulk of the int kernel's 185 DSP is the **attention-BACKWARD int64 multiplies** (`block.cpp:590-601`; 183 DSP), which are carried over UNCHANGED in the mp kernel (see §3.2). **The forward blend is unchanged.** LN (`layernorm`/`fdyad`), L1/dot scores, exp2-LUT adaptive softmax, residuals, relu, CE-grad, `maxshift` down-message: all shift-add/compare/LUT, DSP_MULT=0, unchanged.

The only new forward wrinkle is the **master→forward boundary** (STE, §2.4): the forward reads `dyad(snap(masterfx, scale_exp))` instead of a raw int16 load. Still DSP_MULT=0.

> **MX precondition for the forward guard (HARD).** The golden's forward MX-block-rounding path (`g_block_round`/`g_mxbs`) exists ONLY in the golden, **not in the int kernel**. Therefore `g_forward(mp) == g_forward(int)` holds **only at `MX_BS=0`** (MX disabled). **Pin `MX_BS=0` as a hard precondition of the forward guard** — the TB must assert `MX_BS==0` before checking `g_forward(mp)==g_forward(int)`, or the guard is comparing a golden-only code path against a kernel that doesn't have it.

### 2.2 Backward cascade — the new fixed-point multiplies
Authority forks `g_backward` → `g_backward_mp` (`forge_pc_xformer_mp_golden_block.hpp`). Same cascade STRUCTURE (FFN→LN2→attn→QKV→LN1, `block.cpp:572-610`); all intermediate grads stay `acc_t`=int64 (the cascade was never the dynamic-range wall — `gRP/gM/gHD/gX2/gR/gX1` are already int64 end-to-end, narrowed only at `EOUT`).

| stage | int kernel | **mp sibling** | mult? |
|---|---|---|---|
| error route `Wᵀ·g` (`dyad_route`, `block.cpp:265`) | dyadic shift-add | **UNCHANGED, dyadic** (route weights are pow2 by construction → exact shift, no quantization loss). Ablation flag `XM_ROUTE_MUL` to widen if the cascade is found to clip | no |
| **wgrad outer product** (`dyad_wgrad`, `block.cpp:279`) | pow2×pow2 (`gdv`) coarse | **`fix_wgrad`: real `go·x` MAC, accumulate int64, requantize to `XM_GBITS`.** Operand `go` narrowed to the int16 error band before the multiply → **int16×int16 = 1 DSP/lane** (more precise than pow2×pow2, cheaper than int64 mult). Result summed into `XOM_GACC` (§4), NOT applied | **yes (new)** |
| attn-bwd `gp`/`ex`/`gs`/`gV` (`block.cpp:593-596`) | int64 mult | **UNCHANGED int64 mult**, operands gridded to `XM_GBITS`. ★ `pn*gp` computed in `__int128`→`>>X_AT_PFRAC`→`acc_t` (the width-landmine order preserved exactly) | yes |
| attn-bwd L1 subgrad `gQ/gK ± gs·sgn(Q−K)` (`block.cpp:597-601`) | conditional ±gs | **UNCHANGED, mult-free** (keep it — saves DSP) | no |
| LN bwd (`ln_bwd`, `block.cpp:294-326`) | dyadic, DSP=0, **MISSING the variance-coupling third term** | **MUST ADD the vc term** (dyadic, DSP=0) — see ★ below | no |

★ **LN-bwd is NOT bit-exact to the golden TODAY.** The kernel's `ln_bwd` (`block.cpp:294-326`) is **missing the variance-coupling third term** that the golden's `g_ln_bwd` (`golden_block.hpp:130-133`) HAS — the golden added it in commit `c34cfe0` and the kernel was never updated. So `kernel != golden` on LN-bwd as of today. The mp kernel's `ln_bwd` **must ADD the vc term**: the variance-coupling dyadic-dot `vc`, the `vcm >> X_LOG2_D` mean term, and the `- g_dyadval(chat)*g_dyadval(vcm)` subtraction (all dyadic, DSP=0). **Re-establishing LN-bwd bit-exactness vs the golden is a G3 prerequisite** — without it, G3 `max|diff|==0` cannot pass on any LN-touching region.

### 2.3 Optimizer — fix16 dyadic-Adam (the highest-value change; removes the wall)
Forks `g_adam`/`dyadic_adam` (`block.cpp:329`) → `fix_adam` / `dyadic_adam_mp`. Math FORM is the validated reference (m,v dyadic-EMA `β=1-2^-3=0.875` via `>>X_ADAM_BSH`; step `= sign(m)·2^floor(log2(|m|/v))` via `fix_msb(m)-fix_msb(v)`, two priority encoders, **no divide, no multiply → DSP=0**). The changes that remove the wall:

1. **Master = `fix_t`=ap_int<16> on a per-tensor `2^scale_exp` grid** (not int16 grid-1). Grid step `2^(scale_exp-(XM_GBITS-1))` is `2^15×` finer → `XM_LR_SHIFT > 0` is now LEGAL (int kernel forced `X_LR_SHIFT=0`). A fractional-master companion (`XOM_WFRAC`) carries the sub-integer remainder so an actual LR shift moves weights.
2. **m, v are `fix_t`, requantized each step** (mirrors the char-LM `quantize(st['m'])` lines that produced 2.898 coherent — an **autograd witness**; the cascade equivalent is pending G2).
3. **`X_WD_SHIFT` decay re-enabled** (sub-integer room now exists).
4. **DSP=0 preserved** (shifts/compares/priority-encodes only). The fixed-point choice keeps the optimizer multiply-free — a bonus.
5. **LN γ/β stay FROZEN** (SignSGD on them explodes logits, golden:236); embed/pos stay host SignSGD ±1.

### 2.4 The dyadic-quantize STE boundary (precise, the heart of the design)
Two weight views, two STE boundaries — this is the *measured-good* path (the char-LM does both every step → 2.898 coherent):

```
  master  (fix_t @ per-tensor scale_exp)            <-- fix_adam updates THIS  (fix24/16 grid)
     |  snap(): round(master·2^scale_exp) -> int16, clamp ±32767   [STE forward boundary, NEW]
     v
  fwd weight (data_t = int16)                        <-- forward reads THIS
     |  dyad(): round-to-nearest ±2^e               [STE inner boundary, UNCHANGED, fwd-only]
     v
  applied as ±(x << e)                               shift-add, DSP_MULT=0
```

`snap()` is round-shift + clamp (no multiply); `scale_exp` per matrix = `ceil(log2(maxabs(master)))` (one max-reduce + priority-encode, DSP=0), stored in a small `XOM_SCALE` region. **Soundness:** the char-LM recipe is fix-master → DyadSTE (pow2) and is coherent (**autograd witness** — the cascade transcription is proven only once G2 passes); the kernel inserts an int16-snapshot intermediary. This is lossless while `|master·2^scale_exp| < 32767` — and trained dyadic weights are ±2^k with small k, so |w| stays well under 32767. **The clamp at `snap` is the thing to monitor** (a KAT asserts `snap(masterfx)` then `dyad` equals `dyad(int16master)` for the equivalent value → proves the forward datapath is byte-identical and DSP_MULT stays 0).

---

## 3. DSP / TIMING / AREA BUDGET + ONE-SLR FIT

### 3.1 Available budget per SLR — **MEASURED** (csynth report `forge_pc_xformer_block_csynth.rpt:155`, "Available SLR" row, part `xcvu47p-fsvh2892-2L-e`)
| DSP48 | LUT | FF | BRAM_18K | URAM |
|---|---|---|---|---|
| **3008** | **434,560** | **869,120** | **1344** | **320** |
*(Caveat — ESTIMATED: the F2 shell + `sh_ddr` + 2→1 crossbar consume static logic; realistic user budget ≈ 70–85% of the raw row. Treat the raw row as a ceiling.)*

### 3.2 Integer kernel current usage — **MEASURED** (`csynth_block.rpt:153`, Total row)
| | DSP48 | LUT | FF | BRAM_18K | URAM | timing |
|---|---|---|---|---|---|---|
| used | **185** | **190,508** | 104,110 | **764** | 42 | **3.192 ns** est |
| % of SLR | 6% | **43%** | 11% | **56%** | 13% | < 4.00 ns target (+0.808 ns WNS, PASS) |

**The binding constraint is BRAM (56%), then LUT (43%) — NOT DSP** (2823 DSP free in-SLR). **DSP=185 attribution (corrected vs the csynth report + source):** the forward `p·V` blend (`block.cpp:436`) is **only 2 DSP**. The other **183 DSP are the attention-BACKWARD int64 multiplies** (`block.cpp:590-601`; csynth loop instances `LOOP_590`=128, `LOOP_591`=11, `LOOP_592`=44 DSP). These **183 are KEPT unchanged in the mp kernel** (attn-bwd stays int64 mult). Every `dyad_matmul`/`dyad_route`/`dyad_wgrad`/`dyadic_adam`/`layernorm`/`ln_bwd` instance reports **DSP=0** (confirmed per-instance Detail table, e.g. `dyadic_adam_16` DSP=0 at `:201`). DSP_MULT=0 on the dyadic bulk is **verified**. ⇒ the mp **DSP delta vs the 185 baseline is ONLY the new fix-wgrad multiplies** (§3.4); the 185 carry over.

### 3.3 BF16/fix backward op count — **ESTIMATED** (from D=64/FFN=256/VOCAB=64/SEQ=64 dims)
Per kick: wgrad ≈ 53,248 MAC/token × SEQ=64 ≈ **3.41M** + attn-bwd ≈ **16k** ≈ **3.43M real fix-MAC-ops/kick**. Routes (≈3.41M), LN-coupling (≈12k), and the entire forward STAY shift-add (DSP=0). This is an *op* count; DSP count = ops / (lanes × clocks) — a design knob.

### 3.4 Projected mp-kernel usage — **ESTIMATED** (64-lane fix16 MAC array)
| Resource | int now (MEASURED) | + fix backward (ESTIMATED) | projected | % SLR |
|---|---|---|---|---|
| DSP48 | 185 (=2 fwd blend + 183 attn-bwd int64, both KEPT) | +64–80 (NEW fix-wgrad muls — the ONLY delta) | **~250–265** | **~9%** |
| LUT | 190,508 (43%) | +~11k (fix MAC array) +~8k (wider Adam RMW) +~6k (ctrl) | **~215k–225k** | **~50–52%** |
| FF | 104,110 (11%) | +~6k (int48 accum) +~10k (wider state) | **~120k** | **~14%** |
| BRAM_18K | 764 (56%) | +~0 (fix16-in-int32 m/v does NOT grow the region; GACC streams 1 matrix slice) | **~764–840** | **~56–63%** |
| URAM | 42 (13%) | +0–16 | ~42–58 | ~13–18% |

### 3.5 One-SLR fit VERDICT
**FITS one SLR as a single fused AFI.** Binding constraint stays **BRAM (~56–63%)**, LUT second (~50–52%); DSP a non-issue (~9%). **Timing expected to HOLD ≤4 ns (~3.3–3.5 ns):** the fix-MAC chain (1 DSP + int48 add + de-scale) is far shallower than BF16's leading-zero-count + renormalize, and the fixed-point choice *removes* a known hazard — the int wgrad's `gdv` dual priority encoder (the documented 4.266→<4 ns miss) disappears (we just multiply). The remaining `fix_msb` encoders in the optimizer split into two II=1 passes.

### 3.6 Mitigation ladder (if a fresh csynth shows LUT >~75% or BRAM >~85%)
1. **Fewer MAC lanes** (64→32→16) — wgrad is NOT the throughput bottleneck (the dyadic forward loops are), so halving lanes barely moves wall-clock. **First lever.**
2. **Time-multiplex one shared fix-MAC array across matrices** via `#pragma HLS ALLOCATION function instances=fix_wgrad limit=1` (~32 DSP total).
3. **MXFP8 the wgrad** (E4M3 + per-block pow2 scale) — ~4× LUT-cheaper normalize than BF16, keeps tropical-adjacent pow2. The golden has the path (`g_mxbs`).
4. **CFG-selected FWD/BWD split sharing ONE AFI** — `PHASE`-gated; FWD/BWD never co-execute so HLS time-shares LUT/DSP → roughly halves peak LUT. **Heavy hammer, held in reserve.**

---

## 4. PER-CHIP GRADIENT ACCUMULATION + NO-GLOBAL-SYNC PIPELINE (bound weight-staleness to ~1)

### 4.1 The measured constraint
`STALENESS_MEASURED.md` (D=128, L=4, 1500 steps): weight staleness tolerates only **~1 update-step**.

| weight staleness (update-steps) | bpb | Δ vs sync |
|---|---|---|
| 0 (sync) | 2.457 | — |
| **1** | **2.639** | **+0.18 ✓ tolerable** |
| 2 | 3.209 | +0.75 |
| per-layer (avg lag 5) | 3.754 | +1.30 |
| 8 | 4.478 | +2.02 |

LR-scaling does NOT recover it (matched LR=1.2e-3: stale-0 2.482 vs stale-8 3.747). A naive per-microbatch async update across a deep pipeline gives staleness = pipeline depth (≈`2·(L−l)` → ~24 at early layers of a 12-layer model → catastrophic). **FORBIDDEN.**

### 4.2 Two staleness TYPES (keep distinct — conflating them was the REFUTED over-claim)
- **TYPE 1 — neighbour-message staleness — TOLERATED.** Stale neighbour activation/error messages in the local error sweep (the "~0.6pp at 50% stale" relaxation result). **The backward error sweep need NOT drain in lockstep — the driver MAY exploit this.** This is the genuine async win.
- **TYPE 2 — weight staleness — NOT TOLERATED** (only ~1 step). **Weight updates MUST be window-bounded.** The "async / drop-a-round / staleness-immune" framing for *weights* is REFUTED (that result was TYPE 1).

### 4.3 BUILD RULE — pipelined gradient accumulation with a local backward
Each chip **freezes its weights over an aligned window of `W` microbatches**; during the window it runs forward + local-backward + **accumulate** (no optimizer apply); at the window boundary it applies dyadic-Adam **once** from the accumulated grad. Cost = a `W×` larger effective batch (standard, accepted). The pipeline stays full → the 8–17×-over-serial throughput projection SURVIVES (accumulation does not impede filling chips).

**Effective TYPE-2 staleness is `ceil(2(L−1)/W)`, NOT "~1 independent of L".** Within a window the `W` in-flight microbatches see the SAME frozen weights (the freeze IS the weight-stash), but boundary-straddling microbatches cross a window apply. In a depth-`L` pipeline a microbatch's forward and its eventual backward are separated by ~`2(L−1)` slots, so an apply that lands mid-flight can be `ceil(2(L−1)/W)` update-steps stale. **`staleness ≤ 1` therefore requires `W ≥ 2(L−1) ≈ 2L`**, NOT any fixed `W`. Consequences:
- **`W=4` is valid ONLY for `L ≤ 3`.** The **L=8 gate needs `W ≥ 16`**; **L=12 needs `W ≥ 24`**.
- **Host-side assert `W ≥ 2*L`** (refuse to launch otherwise — silently undersized `W` reintroduces the measured stale-`k` degradation, e.g. stale-8 = +2.0 bpb).
- **Throughput / effective-batch cost:** the staleness bound forces a large window — **L=12 → `W ≥ 24` → ~1536-token effective batch** (24 × SEQ=64). This **may need an LR re-tune** at scale (larger effective batch typically wants a larger LR; sweep in the G1/G2 twin).
- **De-risk step 1 (the synchronous sweep, 1 microbatch in flight) does NOT exercise staleness at all** — it has no pipeline overlap, so it cannot validate the `W ≥ 2(L−1)` bound. A **1F1B host-sim test (multiple microbatches in flight) is REQUIRED before AFI** to confirm the window actually holds staleness ≤ 1 at the target `L`.

### 4.4 CFG + DDR changes
**CFG word** (`XO_CFG`, lanes via `cfgw.range`). The current 3 lanes are `PHASE`(31:0)/`ROLE`(63:32)/`DO_UP`(95:64); the shard already writes a zero into lane 3 (`hbuf[XO_CFG+3]=0`, `forge_pc_xformer_shard.cpp:39`) — **lane 3 is free.**

| lane | name | values | meaning |
|---|---|---|---|
| 0 | `PHASE` | `X_PH_FWD=0`/`X_PH_BWD=1` | unchanged |
| 1 | `ROLE` | INPUT/HID/OUT | unchanged |
| 2 | `XM_GMODE` | `XM_GM_ACCUM=0`/`XM_GM_APPLY=1` | **NEW** (refines `DO_UP`). ACCUM = run bwd, sum wgrad into `XOM_GACC`, do NOT touch weights/m/v. APPLY = fold this kick's grad, drain `XOM_GACC`→dyadic-Adam→master, **re-zero `XOM_GACC`** |
| 3 | `XM_WIN_IDX` | uint32 | **NEW** — window index; kernel echoes into `XOM_WSTATE` (seqlock fence, §4.6) |

**DDR layout** — new regions placed AFTER `XO_VAR` so the layout up to `XO_VAR` stays byte-identical (replay/inspection tooling unaffected). Real anchors: `X_NW = X_D*X_D*4 + X_FFN*X_D + X_D*X_FFN + X_VOCAB*X_D = 53248` (`config:103`); per-matrix offsets `X_OFF_WQ..X_OFF_MHEAD` (`config:105-111`); `XO_MOM`/`XO_VAR` at `config:112-113`.

```cpp
// forge_pc_xformer_mp_config.hpp — appended after XO_VAR
static const int XOM_GACC  = XO_VAR + X_NW;             // [X_NW] window-summed wgrad (fix grid, int64 on-chip via 2-lane RMW)
static const int XOM_WFRAC = XOM_GACC + X_NW;           // [X_NW] fractional-master low bits (lets XM_LR_SHIFT>0 move weights)
static const int XOM_SCALE = XOM_WFRAC + X_NW;          // [16]   per-matrix scale_exp (7 matrices; 1 aligned word)
static const int XOM_WSTATE= XOM_SCALE + X_WL;          // [16]   {WIN_IDX, accum_count} seqlock publish (host APPLY-drop detector)
// ... every region below XO_VAR (XO_TOK at config:115 onward) shifts down by (2*X_NW + 2*X_WL) words ...
//     (XOM_GACC[X_NW] + XOM_WFRAC[X_NW] + XOM_SCALE[X_WL] + XOM_WSTATE[X_WL] = 2*X_NW + 2*X_WL)
```
`XOM_GACC` is zeroed at window start (free at init — the driver `calloc`s the whole buffer), `+=` each microbatch (wide-RMW, DSP=0), consumed once at APPLY, re-zeroed by APPLY. On-chip it streams one matrix slice at a time (≤128 KB, same as current m/v handling) → **no new BRAM pressure**. DDR cost = the four appended regions = `XOM_GACC[X_NW]` + `XOM_WFRAC[X_NW]` + `XOM_SCALE[X_WL]` + `XOM_WSTATE[X_WL]` = **2·`X_NW` + 2·`X_WL` words ≈ 2·`X_NW`·4 B ≈ 426 KB/chip** (the two `X_WL` regions are negligible; trivial vs DDR). **m/v store as fix16-in-int32 → the `XO_MOM`/`XO_VAR` region width does NOT change.**

### 4.5 Kernel BWD-tail dispatch (the ONLY control-flow change)
The forward and the backward error cascade (which fills the on-chip `gWQ..gWHEAD`, `block.cpp:572-610`) run **every** BWD kick regardless of `XM_GMODE` — the local error must propagate to the neighbour each kick. Only the **disposition** of `gW*` changes (replaces the `if(DO_UP){...dyadic_adam...}` block at `block.cpp:612-620`):
- **ACCUM:** `accum_region(buffer, XOM_GACC+X_OFF_*, gW*, n)` — wide-RMW `gacc += gW`, single-read+single-write same loop body, II=1. Weights/m/v untouched.
- **APPLY** (window's last kick): fold this grad (one `accum_region`), then per matrix `apply_region`: load `XOM_GACC` slice → `dyadic_adam_mp(W, gacc, m, v, n)` → store W+m+v → **zero the gacc slice in the same pass** → publish `{WIN_IDX, 0}` to `XOM_WSTATE`.

> **★ Fold the OUTPUT-chip LM-head update into the SAME windowed accumulation.** The kernel has a **SECOND apply site** on the OUTPUT chip: the LM-head gradient `gWHEAD` (`block.cpp:559-565`) is applied separately. If it stays a per-kick apply while the body matrices accumulate over a window, the **head desyncs from the body** (head at staleness 0, body at the window boundary) → mixed-staleness training, a silent recipe break. **`gWHEAD` MUST accumulate into `XOM_GACC` (at `X_OFF_MHEAD`) and apply at the window boundary like every other matrix** — one consistent window, one apply site. The ACCUM/APPLY dispatch must cover `gWHEAD` identically.

### 4.6 Driver schedule (no global barrier)
New `forge_pc_xformer_mp_shard.cpp` (clone of `forge_pc_xformer_shard.cpp`; keeps `stage`/`kick`/`readw`/`setcfg`/`role_of`/`RV` verbatim). `setcfg` gains the `XM_GMODE`+`XM_WIN_IDX` lanes. Host keeps `int bwd_cnt[MAXL]`; on chip `l`'s BWD kick: `gmode = (bwd_cnt[l] % WIN_W == WIN_W-1) ? APPLY : ACCUM`, `win = bwd_cnt[l]/WIN_W`. **All chips share a common `W`, and the host MUST assert `W ≥ 2*L`** (§4.3) → only then does every chip's APPLY land at staleness ≤ 1; an undersized `W` silently degrades. There is **NO barrier and NO cross-chip handshake** — the only "global" thing is a host loop variable counting kicks the host already issues. Gate-C `xchip_weight`/`xchip_grad` stay 0 (the accumulator lives in each chip's own DDR, never crosses a chip). **Mean-by-shift:** make `W` a power of two and add `>>log2(W)` in `apply_region` so the optimizer sees a batch-mean grad and the existing LR transfers directly.

**Seqlock fence (fault detection, NOT a barrier):** after each BWD kick the kernel echoes `{WIN_IDX, accum_count}` to `XOM_WSTATE`; the host reads it (one cheap `readw`) on APPLY kicks and `exit(7)`s if the apply silently no-op'd (the documented RTL write-drop). This catches a dropped apply instead of training on stale weights forever.

**De-risk in two independent steps:** (1) validate windowed accumulate on the EXISTING synchronous sweep (confirm apply-every-`W` + accumulator-zero + seqlock) — **note this 1-microbatch-in-flight sweep does NOT exercise weight-staleness** (no pipeline overlap), it only checks the accumulate/apply mechanics; (2) **a 1F1B host-sim test (multiple microbatches in flight) to actually exercise and confirm the `W ≥ 2(L−1)` staleness bound at the target `L` — REQUIRED before AFI** — then swap in the real 1F1B bubble-free schedule.

---

## 5. FILE PLAN (NEW files, distinct names) + CONSTRAINT NOTES

All under `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/` unless noted. **`derives from` = OUR file copy-and-modify; LLM-pattern = READ-ONLY copy of an idea. NEVER edit any source listed.** Distinct `extern "C"` symbol `forge_pc_xformer_mp_block` (never collides with `forge_pc_xformer_block`).

| NEW file | derives from | genuinely new |
|---|---|---|
| `forge_pc_xformer_mp_config.hpp` | ours `forge_pc_xformer_config.hpp` | `XM_GBITS`/`fix_t`, `XOM_GACC`/`XOM_WFRAC`/`XOM_SCALE`/`XOM_WSTATE`, `XM_GMODE`/`XM_WIN_IDX` CFG lanes, `XM_LR_SHIFT`/`XM_WD_SHIFT` |
| `forge_pc_xformer_mp_golden.hpp` | ours `forge_pc_xformer_golden.hpp` | `g_qfix` (the floor's symmetric per-tensor quantizer), `g_bf16` (witness mode); forward helpers byte-identical |
| `forge_pc_xformer_mp_golden_block.hpp` | ours `forge_pc_xformer_golden_block.hpp` | `g_backward_mp`/`g_wgrad_mp`/`g_accum`/`g_adam_mp`/`fix_msb`/`snap`; **`g_forward` byte-identical** |
| `forge_pc_xformer_mp_block.cpp` | ours `forge_pc_xformer_block.cpp`; LLM-patterns `forge_wgrad_df3.cpp` (DATAFLOW wgrad tiling), `forge_dyadic_adam.cpp`/`forge_train_step_s1.cpp` (commit-safe wide RMW) | `fix_wgrad`, `accum_region`, `apply_region`, `dyadic_adam_mp`, `snap`, `XM_GMODE` dispatch |
| `forge_pc_xformer_mp_block_tb.cpp` | ours `forge_pc_xformer_block_tb.cpp` | window/accum KAT, GACC/WFRAC region compares, near-2^62 `pn*gp` KAT, `g_forward(mp)==g_forward(int)` guard |
| `forge_pc_xformer_mp_golden_test.cpp` | ours `forge_pc_xformer_golden_test.cpp` | the G2 "golden trains" harness with `MP_OPT∈{fix,bf16,int}` |
| `synth_pc_xformer_mp_block.tcl` | ours `synth_pc_xformer_block.tcl` | `set top forge_pc_xformer_mp_block`; gate banner reworded (forward-mult-free, NOT total≤185) |
| `run_local_csim_mp.sh` | ours `run_local_csim.sh` | fix16 path (authentic ap_int, NO float shim needed); optional bf16 shim |
| `cl_forge_pc_xformer_mp/` (under `tikitorch/hardware/hdk/`) | ours `cl_forge_pc_xformer/` (reuses proven `cl_forge_muzero_ddr` DDR/DMA-PCIS wiring) | kernel instance name; buffer-size BAR4 addr check (buffer grows by 2·X_NW + 2·X_WL words ≈ 426 KB, well within DDR/crossbar reach) |
| `forge_pc_xformer_mp_shard.cpp` (under `tikitorch/fpga/silicon/`) | ours `forge_pc_xformer_shard.cpp` | window loop, `XM_GMODE` gating, seqlock verify, 1F1B schedule |
| `run_pc_xformer_mp_shard_f2.sh` (under `scripts/`) | ours `run_pc_xformer_shard_f2.sh` | mp build target, `W` passthrough; distinct `STATE_DIR=.tiki_xformer_mp_shard_state` |

### Constraint notes (HARD)
- **NEW files only, distinct names.** Never edit/touch the LLM-instance pattern files under `/Users/ekoermann/Documents/tikitorch/.../forge_kernels/` (`forge_attn*`, `forge_layernorm*`, `forge_dyadic_adam`, `forge_fwd_gemm`, `forge_wgrad*`, `forge_train_step_s1*`) or `cl_forge_morph_lm` — read for patterns, copy ideas into new names. Our own `forge_pc_xformer_*` files are also READ-ONLY references (copy-and-modify into `_mp_` siblings).
- **Two goldens now coexist:** the int `…_golden_block.hpp` (authority for the int AFI on silicon) and the mp `…_mp_golden_block.hpp`. They MUST share `g_forward` byte-for-byte — the TB asserts `g_forward(mp)==g_forward(int)` as a guard, or the "forward unchanged" thesis breaks silently.
- **Invariants preserved:** DSP_MULT=0 on the forward bulk; attention `p·V` is the one allowed forward multiply; BF16/fix multiplies ONLY in wgrad + attn-bwd; attn-bwd accumulators int64; optimizer writeback uses the single-read+single-write fused-wide shape (the 5-write fan-out drops on silicon); distributed = one block/chip, neighbour-only messages, no global gradient.

---

## 6. VERIFICATION LADDER (strict order; gate N must pass before N+1)

Gates G1–G4 are FREE (CPU/EC2, no F2); G5–G6 are on silicon (**USER runs the AFI / instance**).

| gate | what | pass criterion |
|---|---|---|
| **G1 — recipe parity (torch witness)** | `pc_xformer_charlm.py`, `MULTFREE=1` forward + `BWD_FMT=fix16` (the precision-floor setup) at the real D/L | **fix16 ≤ ~2.0 bpb, coherent Shakespeare-like samples** (NOT 0.20-capped gibberish); reproduce the cliff (fix14 coherent / fix13 diverges) so the chosen 16-bit width has its measured margin |
| **G2 — golden lifts the wall (LOAD-BEARING, the CASCADE proof)** | `forge_pc_xformer_mp_golden_test.cpp` trains `g_forward`+`g_backward_mp`+`g_adam_mp` over TinyShakespeare, `MP_OPT∈{fix16,bf16,int}` | **fix16/bf16 golden → ~1.7–2.0 bpb coherent; int baseline caps ~0.20 / gibberish** — the empirical proof, in the EXACT **PC-cascade** arithmetic the kernel transcribes (NOT autograd), that mixed-precision bwd+opt lifts the int wall. **This is the experiment that converts the §1.2/§1.3 autograd witnesses into a cascade-datapath result** — until G2 is green the 2.016/1.732/2.898 numbers are witnesses only. If this fails, transcribing it bit-exactly only reproduces a broken result |
| **G3 — csim bit-exact (transcription)** | `run_local_csim_mp.sh` (local authentic ap_int) + Vitis `csim_design`; TB diffs ALL DDR regions incl. `XOM_GACC`/`XOM_WFRAC`. **PREREQUISITE: the mp kernel's `ln_bwd` must FIRST ADD the variance-coupling vc term to match the golden's `g_ln_bwd` (§2.2 ★, `golden_block.hpp:130-133`, added in c34cfe0) — the int kernel's `ln_bwd` is missing it, so kernel≠golden on LN-bwd TODAY and G3 cannot pass on LN-touching regions until this is fixed.** Also assert `MX_BS=0` (forward guard precondition, §2.1) | **`max|diff|==0` every region**; 3 KATs pass: (a) zero grad_logit → zero GACC delta + zero master delta after apply; (b) all-ones tokens → bit-exact; (c) **window KAT**: W=4 ACCUM + 1 APPLY == sum 4 local grads then one adam step (bit-exact), AND `XOM_GACC` reads back zero post-APPLY. ★ highest-probability failure = the `pn*gp` near-2^62 width-landmine — isolated by a dedicated single-op KAT |
| **G4 — free csynth gate (pre-AFI)** | `vitis_hls -f synth_pc_xformer_mp_block.tcl` (EC2 CPU), read `forge_pc_xformer_mp_block_csynth.rpt` | (1) **DSP_MULT=0 on the dyadic bulk** (per-instance Detail: every `dyad_matmul`/`dyad_route`/`dyad_wgrad`/`dyadic_adam`/`layernorm`/`ln_bwd` DSP=0). The only DSP are: the 2-DSP forward `p·V` blend + the 183-DSP attention-BACKWARD int64 muls (carried from the int kernel) + the NEW fix-wgrad MAC array. **Total DSP > 185 EXPECTED** (185 baseline + the fix-wgrad delta) — gate is dyadic-bulk-mult-free, NOT total≤185. (2) **WNS ≤ 4.0 ns** (expect ~3.3–3.5). (3) **one-SLR fit** (DSP/LUT/FF/BRAM/URAM each <100% of the SLR row 3008/434560/869120/1344/320; watch LUT then BRAM). If LUT>~75% or BRAM>~85% → mitigation §3.6, re-csynth |
| **G5 — AFI + on-silicon one-step bit-exact (USER)** | USER builds AFI from `cl_forge_pc_xformer_mp/` (~5h), loads a slot; driver runs ONE kick (FWD + one BWD APPLY) fixed-seed; host reads `XOM_GACC` + master `W` + `XO_EOUT` | **on-silicon one-step `max|diff|==0` vs the golden replayed with the silicon's EXACT init + token order.** ★ lesson: a one-step verify ≠ the recipe (norm_shift off-by-one + `PC_INITPERM` order-fragility bit prior runs) — exercise the full FWD+accumulate+apply kick |
| **G6 — the distributed run (signature)** | USER runs `run_pc_xformer_mp_shard_f2.sh` on f2.48xlarge (L chips, one block each, window `W ≥ 2*L` — e.g. `W ≥ 16` for L=8, asserted host-side, §4.3, TinyShakespeare) | **trains to ~2.0 bpb coherent** (matches G1/G2 within noise), OUTPUT-chip per-window bpb descends from chance; **no-global-sync instrumented** (cross-chip traffic = only hidden-down/error-up, Gate-C `xchip_weight/grad`=0); per-window grads bit-exact-replayable by the golden over the first N windows. Honesty flags to record: W× larger effective batch (accepted); throughput host-relay-bound at small L (prior shard 25.8→4.0 steps/s); the honest bpb target is the **mult-free-forward 2.016**, NOT the full-BF16 1.732 (we ship the mult-free forward as the throughput lever) |

---

## 7. OPEN RISKS + RECOMMENDED FIRST STEP

### 7.1 OPEN QUESTIONS (lens disagreements preserved, NOT papered over)
1. **Format: fixed-point vs BF16 (RESOLVED but with a caveat).** All three design lenses (numerics, hardware, file-plan) and the measured floor agree: **ship `fix16`, not BF16** (fix16 = or > bf16; cheaper DSP/LUT; removes a timing hazard). BUT the BUILD_PLAN's R1.d still wants **BF16-everywhere demonstrated first** (most robust, 1.732 target). **Resolution kept open:** `bf16` survives as a golden-only witness mode (`MP_OPT=bf16` in G2) to demonstrate the 1.732 lift; the *kernel datapath* ships fixed-point. These are not in conflict once BF16 is scoped to the witness, not the silicon.
2. **Width: fix16 vs fix24 (UNRESOLVED — needs a measurement).** The floor measured fix14 = true cliff, fix16 = 2-bit margin — but the **fix16 cliff is only a 2-bit, single-seed, 0.2M margin** (TINY config: D=64, L=3, 1200 steps, single seed, shared batch order). The numerics lens recommends **fix24** (8 bits of margin, near-free in fabric: a 24-bit add is one LUT carry-chain); the hardware/file-plan lenses ship **fix16** (natural int16 width, LUT-friendlier, 2 bits over the cliff). **OPEN: `XM_GBITS` MUST be re-confirmed at the real D/L with ≥3 seeds and a longer schedule before locking below 16; DEFAULT TO `fix24` if the fix16 margin does not hold at scale.** `XM_GBITS` is one `#define` so this is a one-token change once measured. Default ships fix16; widen to fix24 if G1/G2 at scale drift toward the cliff or the ≥3-seed margin is thin.
3. **`fix_wgrad` operand narrowing (UNRESOLVED).** Narrowing `go` to the int16 error band before the wgrad multiply gives int16×int16=1 DSP and is *more* precise than the int kernel's pow2×pow2 but *less* than a full int64×int16. The floor quantized grads to fix16 anyway, so int16-narrowed wgrad is within the measured-good regime — but **OPEN: if accuracy regresses at scale, widen `go` to fix24 (24×16 mult ≈ 1–2 DSP).** Knob `XM_WGRAD_NARROW`.
4. **`XM_LR_SHIFT` value (UNRESOLVED).** With a fix master, `XM_LR_SHIFT>0` is now legal (int kernel forced 0), but too large re-creates the "step rounds to zero" freeze at the fix grid. The floor run used continuous lr; the discretized shift needs its own confirm. **OPEN: sweep `XM_LR_SHIFT∈{0,1,2}` in the G1 char-LM twin before AFI.**
5. **Route precision (minor, UNRESOLVED).** Recon says keep `dyad_route` dyadic (route weights are pow2 → exact shift); the numerics lens agrees but flags it MAY need widening if the int64-acc rescale bands clip the cascade. **OPEN: ablation flag `XM_ROUTE_MUL`, default OFF; flip only if G2 shows cascade clipping.**

### 7.2 Highest-severity risks (honest, ranked)
1. **The `pn*gp` width-landmine in csim (G3) — HIGH probability.** `pn*gp` reaches ~2^63; the kernel's int64-accumulate→`>>X_AT_PFRAC`→requant order MUST mirror the golden's `__int128`→shift→truncate order exactly. A mismatch looks like "training diverges," not a localized bug — defended by a dedicated near-2^62 single-op KAT.
2. **RMW write-drop on silicon (G5) — HIGH severity, LOW visibility.** The documented landmine (`forge_train_step_s1.cpp:1997-2008`): a write fan-out the RTL scheduler can't prove live → the optimizer/accumulator step silently no-ops while csim/csynth PASS. Defended by (a) single-read+single-write fused shape in `accum_region`/`apply_region`/`dyadic_adam_mp`, (b) the `XOM_WSTATE` seqlock + host `exit(7)` detector, (c) the window-accum KAT asserting GACC reads back zero post-APPLY.
3. **`snap()` int16-clamp clipping the master — MED.** New intermediary vs the char-LM (which goes fix-master→pow2 directly). Lossless while `|master·2^scale_exp|<32767`; trained dyadic weights are ±2^k small-k so clipping isn't expected — but monitor it (KAT + runtime check).
4. **G4 one-SLR fit is LUT-gated — MED.** Choosing fixed-point is the mitigation baked in (BF16 normalize logic is the LUT risk). If fix16 still tips LUT >100%, escalate the §3.6 ladder (fewer lanes → time-mux → MXFP8 → FWD/BWD split).
5. **Window schedule must NOT serialize the apply — MED.** If the driver waits for all chips before the next window's FWD, it reintroduces a bubble and breaks the throughput projection. G6 instruments steps/s to catch this.

### 7.3 RECOMMENDED FIRST IMPLEMENTATION STEP
**Run Gate G2 first — but stage it as the very first build artifact.** Concretely:

> Create `forge_pc_xformer_mp_golden.hpp` + `forge_pc_xformer_mp_golden_block.hpp` (copy ours, fork `g_backward`→`g_backward_mp` with `fix_wgrad`/`g_qfix` and `g_adam`→`g_adam_mp` with the fix16 master + `XOM_GACC` accumulation + `snap()`), plus `forge_pc_xformer_mp_golden_test.cpp`, and run the **G2 standalone training loop** over TinyShakespeare in `MP_OPT∈{fix16,bf16,int}`.

Rationale: the golden is the bit-exact authority for everything downstream (G3 transcription, G5/G6 silicon). G2 is the **load-bearing proof** that the mixed-precision backward+optimizer actually lifts the int wall (~0.20 → ~1.7–2.0 bpb) *in the exact integer/fixed arithmetic the kernel will transcribe* — not just in the torch twin (which G1 already showed). It is pure CPU C++ (no HLS, no Vitis, no F2), it directly validates Open Questions #2/#4 (sweep `XM_GBITS` and `XM_LR_SHIFT` here, cheaply, at the real D/L), and if it fails the whole build stops before any HLS or ~5h AFI is spent. It also forces the `g_forward(mp)==g_forward(int)` guard early, locking the "forward unchanged" thesis. Build the HLS kernel (`forge_pc_xformer_mp_block.cpp`) only after G2 is green.

---

*Sources (all paths absolute, verified 2026-06-27): kernel `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_block.cpp` (the one multiply at line 436; CFG lanes :351-354; dyad_wgrad :279; dyadic_adam :329; apply block :612-620); config `…/forge_pc_xformer_config.hpp` (X_NW :103; X_OFF_* :105-111; XO_MOM/XO_VAR :112-113; bands X_ABITS=9/X_EBITS=12 :35/37; X_LR_SHIFT=0 :62); csynth budget `/tmp/tt-pc/results/predictive_coding/multi_fpga/forge_pc_xformer_block_csynth.rpt` (Total :153, Available SLR :155, Util% :157); driver `/tmp/tt-pc/tikitorch/fpga/silicon/forge_pc_xformer_shard.cpp` (setcfg :39 with free lane 3, sweep :89-101); staleness `…/STALENESS_MEASURED.md`; recipe `…/BUILD_PLAN.md`; precision floor `/tmp/precfloor_charlm.py` (tracked `scripts/pc_xformer_charlm.py` NOT modified).*
