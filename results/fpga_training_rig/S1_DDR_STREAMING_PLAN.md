# S1 Transformer Trainer — DDR-Streaming Rearchitecture Plan

**Goal:** make `forge_train_step_s1.cpp` (D=128, kept at full size) ROUTE + close P&R on F2
by converting `do_fwd_loss` (mode 0) and `do_bwd` (mode 1) from ON-CHIP-RESIDENT (whole working
set in BRAM, random access) to DDR-STREAMING (each op holds only its per-tile working set on-chip,
streams tiles from DDR -> compute -> writes back). `do_opt` (mode 2) is ALREADY tiled (S1_OPT_TILE=4096)
and is NOT touched. The math, dyad() rule, and the int64 golden are UNCHANGED — only the memory access
pattern changes (the same kind of change that the on-chip-staging fix and the do_opt-tiling fix were).

**Status this plan rearchitects:** kernel is functionally complete + Vitis csim bit-exact (all 4 modes),
converges under real ap_int, csynth closes 3.197ns. AFI P&R FAILED: WNS=-4.111ns / TNS=-853,446ns at
125MHz — pervasive routing congestion driven by ~507 BRAM18K (>504 device) of large static arrays
(R_all_oc, X1/X2, attnp_oc, relu_oc, the weight/grad arrays) all needing interconnect at ~83% BRAM.

---

## 0. Verified facts grounding this plan (read from the real files)

- `forge_s1_config.hpp`: D=128, H=2, DH=64, FFN=512, L_SEQ=128, V=128, N_LAYERS=1. Every region base
  is `S1_AL16`-aligned (16-word / 512-bit). `s1_off_*` helpers give layer-major addressing.
- `forge_train_step_s1.cpp`:
  - GEMM dim bounds: `S1_GEMM_TMAX=128` (=L_SEQ), `S1_GEMM_IMAX=512`, `S1_GEMM_OMAX=512` (=FFN).
    These size the `fwd_gemm`/`xgrad_gemm`/`wgrad_gemm` dyadic-quant scratch (`xE/xS/xNZ`, `goE/...`).
  - `fwd_gemm(xin,wcode,w_off,yout,M,I,O)` (lines 353–406): output-stationary, IRED contraction,
    `S1_LANE=16` cyclic banks, II=1. Already reads W from an on-chip `code_t` array via `w_off`.
  - `attn_head` (468–543): per-query causal; `p_save[i*L_SEQ+j]`, `jstar_save[i]` written per row.
    The ONE multiply is the `p[j]*V[j][d]` blend.
  - `do_fwd_loss` (569–901): burst-loads all weights/LUTs (603–622) -> EMB gather -> LAYER loop
    (LN1/QKV/HEAD/Wo/RES1/LN2/FFN/RES2/SAVER) -> LNF -> HEADLOOP (LM head + CE) -> burst-store
    activation-saves (883–896).
  - `do_bwd` (1211–1659): re-burst-loads weights + reads mode-0 activation saves (1280–1290),
    zeroes the gradient accumulators (gWQ_oc … gWHEAD_oc), runs the reverse chain into per-token
    scratch (gR, gX1_oc, gDn, gHd, gU, gAcat, gQv…), burst-stores g_* at end (1636+).
  - `do_opt` / `dyadic_adam_region` (1778–1888): the PROVEN tiled template — `static int32_t
    g_t/m_t/v_t/wfp_t/dst_t[S1_OPT_TILE]` with `BIND_STORAGE ram_2p impl=bram`; `OPT_TILE:` loop
    `tbase += S1_OPT_TILE`, per-tile `burst_load_i32` -> PASS A/B with FIXED `[i]` indexing II=1 ->
    `burst_store_i32`. **This is the structure we copy for the activation streams.**
  - Dispatch (2038–2072): mode 0/1/2/3; `#pragma HLS ALLOCATION function instances=… limit=1` on
    do_fwd_loss/do_bwd/do_opt AND on the GEMM/attn/LN helpers — ONE physical engine each.

Key structural truth (confirmed): **the residual stream `R[t][d]` (int16, on the stack, L_SEQ*D=16K
elt) is the SEQUENTIAL dependency carrier.** It is NOT one of the huge static BRAM arrays — it is small
(32KB) and must stay resident. The huge arrays are the **activation SAVES** written for the backward
(`R_all_oc`, `X1/X2`, `attnp_oc`, `relu_oc`, `argh/glog/logit`) plus the **gradient accumulators** in
do_bwd. Those are write-once-stream-out (fwd) / read-once-stream-in (bwd) and are what we move to DDR.

---

## 1. Which arrays move to DDR-streamed-per-tile vs stay resident

### STAY RESIDENT (small; weights/LUTs/tokens + per-op working scratch)
These are already small or are the genuine working set of one op; keeping them on-chip is required for
II=1 and avoids variable-offset AXI on the critical path.

| Array(s) | BRAM18K | Why resident |
|---|---|---|
| All weight codes/values: `embed_oc, wq/wk/wv/wo_oc, wup/wdn_oc, ln1g/ln1b/ln2g/ln2b/lnfg/lnfb_oc, whead_oc` | ~130 | Read-only, sequential GEMM/gather reads; already on-chip; total ~130 BRAM is fine. (Optionally reload `wup/wdn` per-op via RELOAD if 130 still crowds — see §4 fallback.) |
| `ce_lut, at_lut` (256 each) | ~2 | Host-staged ROMs (AFI static-init-zero bug); must stay staged + resident. |
| `toks, tgts` (128 each) | ~2 | Per-position index metadata; tiny. |
| `R[L_SEQ][D]` residual (int16, on stack) | ~ (LUT/small BRAM) | THE sequential dependency carrier; small (32KB); must persist across the whole block. |
| Per-op block scratch already on stack: `X1, Qm/Km/Vm, Qh/Kh/Vh/Ah, Acat, AO, X2, U, Hd, Dn, Rn` | small | These ARE the per-op working tiles; reused per layer. Keep. |
| `fwd_gemm`/`xgrad`/`wgrad` dyadic-quant scratch `xE/xS/xNZ` etc. (`TMAX*IMAX`) | bounded by tile | These already define the GEMM's working set; bound them to the tile (see §2). |
| LN stats `ln1st/ln2st/lnfst_oc`, `attnj_oc` | ~7 | Tiny per-(layer,token) metadata; cheap to keep resident, stream-out once at end (or keep — they fit). |

### MOVE TO DDR (stream per tile; the BRAM hogs)
These are the large activation SAVES (fwd) and gradient accumulators / saved-activation reloads (bwd).
They are written/read sequentially and never need the whole array on-chip at once.

| Array | BRAM18K freed | Tile granularity | Direction |
|---|---|---|---|
| `relu_oc` `[N_L*L_SEQ*FFN]` | **114** | per-token row `[FFN]=512` (or per t-tile `[TT*FFN]`) | fwd write-out |
| `R_all_oc` `[(N_L+1)*L_SEQ*D]` | **57** | per-token `[D]=128`, per layer-snapshot | fwd write-out |
| `attnp_oc` `[N_L*H*L_SEQ*L_SEQ]` | **57** | per-query row `[L_SEQ]=128`, per (layer,head) | fwd write-out |
| `X1_oc` + `X2_oc` `[N_L*L_SEQ*D]` each | **28+28** | per-token `[D]=128` | fwd write-out |
| `argh_oc` + `glog_oc` + `logit_oc` `[L_SEQ*V]` each | **29+29+29** | per-position `[V]=128` | fwd write-out |
| (bwd) gradient accumulators `gWUP_oc/gWDN_oc` `[N_L*FFN*D]` | **29+29** | per output-tile `[OT][I]` (wgrad output-stationary) | bwd accumulate-out |
| (bwd) `gEMBED/gWQ/gWK/gWV/gWO/gWHEAD` | ~70 total | per-region tile (do_opt-style) or output-tile (wgrad) | bwd accumulate-out |
| (bwd) saved-activation RELOADS (`R_all/X1/X2/relu/attnp/glog/argh`) | (same as fwd) | per-tile stream-IN matching the consuming op | bwd read-in |

**Net:** moving just `relu_oc` (114) drops 507->393 (already below 504 and likely below the congestion
knee). Moving `relu_oc + R_all_oc + attnp_oc + X1/X2` drops to ~263 BRAM18K (~52% device). Adding the
head trio (`argh/glog/logit`) and bounding the GEMM dyadic scratch to a tile reaches the **<40% device**
target. The exact landing depends on the GEMM-scratch tile size (§2, §4).

---

## 2. Per-op tile-stream restructure

General template (copied verbatim in structure from `dyadic_adam_region`): for each op, a tile array
sized to the op's working slice, `BIND_STORAGE ram_2p impl=bram`; an outer tile loop over positions (or
output rows for GEMM); per tile a `burst_load_i32` of only the inputs this tile needs, an II=1 compute
loop with FIXED `[i]` indexing, a `burst_store_i32` of the outputs. Burst granularity is the 512-bit wide
word (16 int32/cycle, II=1) — `burst_load_i32`/`burst_store_i32` already do this. All tile bases stay
16-word aligned because region bases are `S1_AL16` and tile strides are multiples of D=128 / FFN=512 / V=128
(all multiples of 16).

**EMBED gather (EMB loop):** unchanged compute. `R_all_oc` save -> stream out the R_0 snapshot per-token
tile (`[D]` per t) instead of writing the resident `R_all_oc`. `R[t][d]` stays resident (the carrier).
- Tile = `[TT][D]` (TT positions, e.g. 32). II=1; the embed read is on-chip; only the save burst-stores.

**LN1 / LN2 / LN_F (per-token):** compute `layernorm_tok` per token into the resident `R`/`X1`/`X2`/`Rn`
stack scratch (UNCHANGED — these are small and needed by the immediate next GEMM). The CHANGE: the
`X1_oc`/`X2_oc` *saves* and the LN-stats saves stream out per-token-tile. LN stats are tiny (keep resident,
single store at end is fine). So LN1/LN2/LNF: keep X1/X2/Rn resident for the next op; burst-store X1_oc/X2_oc
per t-tile.
- Tile = `[TT][D]`. II=1 (the SUMX/CEN/RESC inner loops are already II=1, D-bounded).

**Q/K/V GEMM (fwd_gemm, M=L_SEQ I=D O=D):** **no structural change** — `fwd_gemm` is already
output-stationary, reads on-chip W and on-chip `X1` (resident scratch), writes on-chip `Qm/Km/Vm`
(resident scratch). It does NOT touch a DDR-bound array. The only tiling lever here is the dyadic-quant
scratch `xE/xS/xNZ[TMAX][IMAX]` — for the projections I=D=128 so IMAX-sized at 512 is over-allocated;
bound the scratch second dim to the actual `I` per call (it already iterates `i<I`, so reduce the array
dim to `S1_GEMM_IMAX` only where FFN is needed — see FFN below). Keep as-is for QKV.

**Attention per-head (attn_head):** Q/K/V heads are resident scratch `[L_SEQ][DH]` (small, 64-wide) —
keep resident; the per-head extraction loop stays. The CHANGE: `attnp_oc` p_save and `jstar_save` stream
out per-query-row. The MAP's key insight holds: `p_save[i*L_SEQ+j]` is per-query-row streaming reuse,
NOT random — once query i+1 starts, query i's row is dead. So: keep a SINGLE on-chip `p_row[L_SEQ]`
(128 int32, ~1 BRAM) per query, write it in SAVEP, then `burst_store_i32(p_row, buffer,
s1_off_attn_p(l,h)+i*L_SEQ, L_SEQ)` per query. `jstar` is 1 word/query -> keep `attnj_oc` resident
(tiny) and store once.
- The blend `p[j]*V[j][d]` (the ONE multiply) is unchanged; p[] is the resident per-query working vector.
- Tile = one query row `[L_SEQ]`. II=1 preserved (SAVEP is already II=1 over j; the burst is per-row,
  amortized over the SCORE/EXP/BLEND work of that query).
- **attn all-positions access:** the SCORE/EXP/BLEND read `Q[i][d]`, `K[j][d]`, `V[j][d]` over all j<=i —
  these are the resident `Qh/Kh/Vh[L_SEQ][DH]` scratch (whole sequence, 64-wide, ~small), so all-positions
  access stays ON-CHIP. We do NOT stream K/V from DDR (that would break II=1 with per-j gather). Only the
  p_save OUTPUT streams. This is the safe choice (see risks §6).

**Wo (output projection GEMM):** same as QKV — `fwd_gemm` reads on-chip `Acat`, writes on-chip `AO`.
No DDR array involved. RES1 residual add into resident `R`. Unchanged.

**FFN up / ReLU / down:** 
- up `fwd_gemm(X2 -> U)`: M=L_SEQ I=D O=FFN. U is resident scratch `[L_SEQ][FFN]` — this is the LARGEST
  resident scratch (128*512 int16 = 128KB). Keep resident (it is the working tile of the FFN op and feeds
  ReLU immediately). The fwd_gemm dyadic scratch here is `xE[TMAX][IMAX=512]` — but for UP, I=D=128, so
  the IMAX=512 allocation is the wasteful one; size the GEMM input-scratch to `max(D, FFN)` per call is
  unavoidable with one shared engine, so this is a fixed cost (~few BRAM), accept it.
- ReLU: compute `Hd[t][k]=max(U,0)` into resident `Hd` scratch; the CHANGE: `relu_oc` mask streams out
  per-token-tile `[TT][FFN]`. This frees the 114-BRAM hog. Tile = `[TT][FFN]`; II=1 over k (already II=1).
- down `fwd_gemm(Hd -> Dn)`: M=L_SEQ I=FFN O=D. Hd resident; Dn resident; RES2 into resident R. The
  contraction is over FFN=512 — `xE[TMAX][IMAX=512]` is correctly sized here. Unchanged.

**LM-head + CE (HEADLOOP, per position):** logits/argmax computed per position into resident `logits[V]`,
`ad_head[V]`, `e_row[V]` (small, V=128). The CHANGE: `argh_oc`, `glog_oc`, `logit_oc` stream out
per-position tile `[V]`. Keep a per-position on-chip `[V]` working vector for each (already there:
`logits`, `ad_head`, `e_row`) and burst-store per position (or batch a few positions per burst for fewer,
wider bursts — tile = `[TP][V]`, TP e.g. 8).
- Tile = `[TP][V]`. II=1 (HEAD_V/CE_E/CE_P/CE_G already II=1 over v).

**Backward xgrad / wgrad / attn_bwd / LN_bwd / scatter:**
- The saved-activation READS (`R_all`, `X1/X2`, `relu`, `attnp`, `glog`, `argh`) stream IN per-tile,
  matched to the consuming op's tile (per-token `[D]`, per-token `[FFN]`, per-query `[L_SEQ]`,
  per-position `[V]`). Mirror the fwd write tiles.
- `xgrad_gemm` (gx = sum_o gO * q(W)): output-stationary on (t,i), already II=1; reads on-chip gO scratch
  + on-chip W; writes on-chip gx scratch. Stream the gO IN per-token-tile and gx OUT per-token-tile if
  those map to DDR-bound grad-activation regions (gR/gX1/gX2/gAcat/gQ/gK/gV are scratch grad regions in
  the offset map — stream them like fwd activations).
- `wgrad_gemm` (gW = sum_t q(gO)*q(x)): contraction over t (all L_SEQ). It needs the WHOLE t-range of gO
  and x for one (o,i). Strategy: **output-stationary tile on (o)** — for an output-tile of OT rows, hold
  `goE/goS/goNZ[L_SEQ][OT]` and `xE/...[L_SEQ][I]` on-chip (stream gO and x IN once per output-tile),
  accumulate `gW[OT][I]` on-chip, burst-store the gW output-tile OUT. This is exactly the
  `forge_wgrad_df3` OTILE/ITILE pattern. The big grad accumulators `gWUP/gWDN` (29 BRAM each) are written
  per output-tile, never fully resident. Tile = `[OT][I]` for gW + `[L_SEQ][OT]`/`[L_SEQ][I]` for the
  streamed-in dyadic operand caches. Pick OT so `L_SEQ*OT + L_SEQ*I + OT*I` int32 stays < the budget
  (OT=32, I=512 -> ~64K+256K+64K ... bound I to the per-call value; for FFN wgrad I=128 (down) or O=512
  (up) — tune OT per call, default OT=32).
- `attn_bwd`: gQ/gK/gV — reads `attnp` (stream IN per-query-row, the SAME reuse pattern as fwd), the soft-
  blend/softmax multiplies stay on DSP. Per-query working set on-chip; II governed by the existing bwd
  loops (note do_bwd currently has II=4 data-dependent scatters in places — see §6 risk).
  L1 sign(Q-K) is mult-free.
- `LN_bwd`: per-token; reads X1/X2 (stream IN per-token) + LN stats (resident) via `s1_apply_invq`;
  STE gamma pass-through; DSP=0. Per-token `[D]` tile.
- scatter (gEMBED, gWHEAD): sparse RMW via saved indices (argh/toks). Stream the index arrays + the target
  grad region per-tile; the scatter itself is the existing data-dependent write (II>1 acceptable for the
  correctness proof). Accumulate into a do_opt-style tile and burst-store.
- **do_opt: NO CHANGE** (already tiled).

---

## 3. Bit-exact strategy

Identical discipline to the two proven fixes (on-chip-staging closed mode 0; do_opt-tiling closed mode 2):

1. **Math, dyad() rule, loop bodies UNCHANGED.** Only the memory ACCESS PATTERN changes (resident array
   -> per-tile burst in/out). Every logical element reads/writes the IDENTICAL buffer offset; only the
   ORDER/STAGING of the DDR transfer changes.
2. **The transformations are provably order-independent:**
   - Activation SAVES are pure WRITES of already-computed values to fixed offsets — tiling the writes
     cannot change any value.
   - GEMM contractions are integer adds with int48 headroom; partial-tile partial sums summed in sequence
     yield the IDENTICAL total (this is the same argument the existing `TT`/`WG_TT` LANE-narrowing and the
     do_opt tiling already rely on). No new overflow: acc_t=int48 (Vitis) / int64 (g++ golden) unchanged.
   - The residual stream `R` stays a single resident array updated in place exactly as today — the
     sequential dependency is NOT reordered.
   - attn p_save per-query-row streaming uses the SAME `p[j]` values; only the destination is a DDR burst
     instead of a resident array slot.
3. **The int64 golden (`#else` g++ path) stays the authority and is UNTOUCHED** — the `burst_load_i32`/
   `burst_store_i32` helpers already have a scalar `#else` path that is a plain `for` copy, so the golden
   sees the same buffer contents. New tile loops get the same `#else` scalar fallthrough.
4. **Validation gate:** after EACH op conversion, run csim and assert `max|kernel - int64 golden| == 0`
   over ALL exposed regions (the existing TB check). A single nonzero -> revert that op. This is the
   non-negotiable gate (same as the staging/do_opt fixes).
5. **Alignment invariant:** every tile base = region_base (`S1_AL16`) + k*(D|FFN|V|L_SEQ), all multiples
   of 16 -> wide bursts stay aligned (the OFF_CE_LUT-class corruption bug is avoided). Assert this with a
   compile-time `static_assert` on the tile stride % 16 == 0 per region.

---

## 4. BRAM target + DDR traffic/step

**BRAM target: < ~40% device (< ~200 BRAM18K of 504).**

Landing estimate after streaming the listed DDR arrays + bounding GEMM scratch:
- Weights/LUTs/tokens resident: ~130 BRAM18K.
- Per-op working tiles (one at a time, ALLOCATION limit=1 already serializes engines): the largest single
  resident working set is the FFN op (`U`+`Hd` resident `[L_SEQ][FFN]` int16 = 2*128KB ≈ ~28 BRAM) OR the
  wgrad output-tile operand caches (`[L_SEQ][I]` dyadic caches, OT-bounded ≈ ~20–40 BRAM). Streamed-out
  arrays now cost only their tile (`[TT][FFN]` etc. ≈ a few BRAM each).
- LN stats + attnj + small metadata: ~10 BRAM.
- **Total resident ≈ 130 + ~40 (worst single op) + ~10 ≈ ~180 BRAM18K ≈ 36% device.** Meets <40%.

Fallback levers if it lands >40% or still congests (in order):
1. RELOAD `wup_oc`/`wdn_oc` (58 BRAM) per-FFN-call instead of holding both layers resident (morph RELOAD
   idiom) — at N_LAYERS=1 they're one layer, but if N_LAYERS bumps to 2 this is the first lever.
2. Bound the shared `fwd_gemm`/`wgrad_gemm` dyadic scratch to a position-tile (TT<L_SEQ) so
   `xE/...[TT][IMAX]` shrinks (TT=64 halves it; TT=32 quarters). Adds <1% loop overhead.
3. Lower `S1_LANE` 16->8 (halves the cyclic-partition bank count -> fewer BRAM, more cycles).
4. Smaller TT/OT tiles everywhere (more, smaller bursts; II unchanged).

**DDR traffic per step (matches the MAP estimate):**
- Forward activation-SAVES streamed out: R_all(32.8K) + X1/X2(32.8K) + attnp(32.8K) + relu(65.5K) +
  argh/glog/logit(3*16.4K) + LN stats(~0.8K) ≈ ~215K int32 ≈ ~860 KB OUT.
- Backward: same activations streamed IN (~860 KB) + gradient regions streamed OUT (~1.6M? no — grads are
  one int32/weight ≈ ~200K weights ≈ 800 KB OUT).
- Total ~0.8–1.7 MB transferred per fwd+bwd step. At ~8 GB/s DDR ≈ ~100–215 µs, **fully hidden** by the
  II=1 pipelined compute (bursts issued once per tile, latency overlapped). DDR BW is NOT the bottleneck
  (the kernel is compute-bound; this matches the morph-LM and do_opt experience). The single-shot proof
  is latency-insensitive anyway.

---

## 5. Staged BUILD ORDER (convert one op, csim after each, then re-AFI)

Convert in INCREASING risk / DECREASING BRAM-win-per-effort, csim bit-exact after EACH step (gate: max
diff == 0). Do NOT batch conversions — one op per commit so a regression is bisectable.

1. **`relu_oc` -> DDR (forward).** Biggest single win (114 BRAM, 507->393). Sequential per-token write,
   trivially order-independent. Lowest risk. csim. *This alone may already route* — optionally do a
   speculative re-AFI here to check the congestion knee.
2. **`R_all_oc` -> DDR (forward).** 57 BRAM (393->336). Per-token per-layer-snapshot write. csim.
3. **`X1_oc` + `X2_oc` -> DDR (forward).** 56 BRAM (336->280). Per-token write. csim.
4. **`attnp_oc` -> DDR (forward), per-query-row.** 57 BRAM (280->223). The per-query-row reuse pattern;
   slightly more care (verify the row stride + alignment). csim.
5. **`argh_oc` + `glog_oc` + `logit_oc` -> DDR (forward).** 87 BRAM (223->~136). Per-position write. csim.
   **-> RE-AFI #1: forward-only mode-0 P&R.** Expect WNS positive, BRAM ~<30% for mode 0 alone. This is
   the first real routing milestone.
6. **Bound GEMM dyadic scratch (TT tile) if needed** to hit <40% with do_bwd present. csim.
7. **Backward saved-activation RELOADS -> per-tile stream-IN** (mirror the fwd write tiles). csim after
   each consuming op (xgrad, then LN_bwd, then FFN bwd, then attn_bwd).
8. **Backward gradient accumulators -> output-tiled wgrad (gWUP/gWDN first, then gWQ/.../gWHEAD).** csim.
9. **Backward scatter (gEMBED/gWHEAD) -> tiled accumulate + burst-store.** csim.
   **-> RE-AFI #2: full mode-0+mode-1 P&R** at 125 MHz. Target WNS>0, BRAM<40%, TNS=0.
10. **do_opt: untouched.** Mode 3 (`do_train_run`) sequences modes 0/1/2 — re-run the fused csim to
    confirm end-to-end bit-exact + convergence. **-> RE-AFI #3 (full kernel) if 0/1 were AFI'd separately
    via a build define, else #2 is already the full kernel.**

CSIM command per step: build the standalone g++ path + the Vitis csim via `synth_train_step_s1.tcl` and
assert the TB's `max|kernel-ref|==0`. Csynth after steps 5 and 9 to confirm II=1 held and BRAM dropped.

---

## 6. Risks + effort

**Risks (concrete + honest):**
1. **II=1 on streamed loops — MEDIUM.** The do_opt tiling proved II=1 survives per-tile bursts because the
   burst is issued ONCE per tile (not per element) and compute uses FIXED `[i]` indexing. The risk is a
   tile loop where the burst-store of tile k overlaps the compute of tile k+1 and HLS cannot resolve the
   shared on-chip tile array dependency -> II>1 or a ping-pong requirement. Mitigation: separate input and
   output tile arrays (no in-place), or accept the do_opt schedule (sequential burst-in / compute /
   burst-out per tile, II=1 within each phase). The single-shot proof tolerates the inter-tile bubble.
2. **DDR latency — LOW.** Hidden by wide II=1 bursts; the kernel is compute-bound; single-shot is latency-
   insensitive. Already validated on F2 by morph-LM RELOAD and do_opt.
3. **attn all-positions access — MEDIUM (and the reason we DON'T stream K/V).** SCORE/EXP/BLEND read
   `K[j][d]`,`V[j][d]` over all j<=i. Keeping Qh/Kh/Vh resident `[L_SEQ][DH]` (small, 64-wide) avoids a
   per-j DDR gather that would shatter II=1. We ONLY stream the p_save OUTPUT. If even Qh/Kh/Vh resident
   is too much (it is not at DH=64), the fallback is a K/V row-tile with double-buffering — but that is the
   one place a naive "stream everything" would break II, so the plan explicitly keeps it resident.
4. **Sequential dependency of the residual stream — LOW (by design).** `R[t][d]` stays a single resident
   in-place array; we never tile or reorder it. RES1/RES2 add into it exactly as today. The streamed arrays
   are the SAVES of R (R_all snapshots), not R itself — saving is a pure write, no dependency.
5. **Backward scatter II>1 — LOW for correctness.** do_bwd already has data-dependent scatters (II=4) into
   gEMBED/gWHEAD. Tiling does not worsen this; the area win in bwd comes from the wgrad output-tiling and
   the saved-activation reload tiling, not from the scatters. Accept the existing II on scatters.
6. **BRAM landing uncertainty — MEDIUM.** The ~180 BRAM / ~36% estimate depends on the shared-GEMM dyadic
   scratch (IMAX=512) and the wgrad operand-cache tile size. If it lands 40–55%, apply the §4 fallback
   levers (bound GEMM scratch TT, RELOAD wup/wdn, lower S1_LANE). The do_opt precedent shows the tiling
   itself is cheap; the residual risk is the shared engines' fixed scratch, which the TT-bound addresses.
7. **Alignment regression — LOW but corrupting if missed.** Every new tile base must be 16-word aligned.
   Mitigated by `static_assert` on strides and the existing `S1_AL16` region bases; csim catches any
   misalignment as a nonzero diff immediately.

**Effort estimate:** ~3–5 focused days for an HLS engineer who knows this kernel.
- Forward conversion (steps 1–5): ~1–1.5 days (the relu/R_all/X1/X2/head trio are all the same
  per-token-tile template copied from do_opt; attnp per-query-row is the one bespoke piece). csim is fast.
- RE-AFI #1 + iterate on BRAM if needed: ~0.5 day of wall-clock per build (USER runs the F2/Vivado build;
  agent designs only).
- Backward conversion (steps 6–9): ~1.5–2 days (wgrad output-tiling is the most involved; saved-activation
  reloads mirror the fwd tiles; scatters unchanged).
- RE-AFI #2 + close: ~0.5–1 day wall-clock per build iteration.
The dominant uncertainty is build wall-clock (P&R), not design — the forward path likely routes after
step 1 or 5, de-risking the rest.

---

## Appendix — reusable tile template (structure copied from `dyadic_adam_region`)

```cpp
// per-op DDR stream-out of an activation save (forward); mirror with burst_load_i32 for bwd stream-in.
static int32_t tile_oc[TILE_ELTS];            // sized to one op's slice, e.g. [TT*FFN] or [L_SEQ]
#pragma HLS BIND_STORAGE variable=tile_oc type=ram_2p impl=bram
SAVE_TILE: for (int tbase = 0; tbase < N; tbase += TILE_ELTS) {
    int tn = N - tbase; if (tn > TILE_ELTS) tn = TILE_ELTS;
    COMPUTE: for (int i = 0; i < tn; i++) {     // FIXED indexing; the existing II=1 op body writes tile_oc[i]
#pragma HLS PIPELINE II=1
        tile_oc[i] = /* the value the resident array held, computed verbatim */;
    }
    burst_store_i32(tile_oc, buffer, REGION_OFF + tbase, tn);   // 512-bit wide, II=1, aligned
}
```
Backward reads use the inverse: `burst_load_i32(buffer, REGION_OFF + tbase, tile_oc, tn)` then the
existing II=1 consumer body indexes `tile_oc[i]`. do_opt is left exactly as-is.
