# Next-rebuild bundle — remaining throughput levers (post-2.94×)

State (2026-07-01): kick = **~121 ms measured** (was 355; partial-sum II=1 fix, AFI
`agfi-0846ef96674df0815`). csynth says 44–65 ms of that is kernel compute → **~60 ms/kick is
unattributed** (DDR access latency / data-dependent bound). Rule: **measure first** (memory:
hls-reduction-ii-lesson) — every item below states its measurement gate. Bundle candidates in
priority order; all compose; one AFI build.

## 0. MEASURE the ~60 ms residual first (no rebuild)
Driver-side per-phase timer in `kick()` (`forge_pc_xformer_mp_shard.cpp:56` already tight-spins
AP_DONE — wrap with `now_s()` and print per-chip FWD vs BWD ms). One 50-step run gives the
fwd/bwd split and whether the residual is per-kick-fixed (→ items 2,3 pay) or data-dependent
compute (→ csynth max-bound, only item 1 pays). **Do this in the next silicon run regardless.**

## 1. Cross-sub-batch pipelining via a double-buffered activation cache (~2×, kernel+driver)
The ACCUM window is weight-frozen (updates only on the last sub-batch) → sub-batches are
**staleness-free to pipeline**. Blocker today: ONE wide-forward cache (`MP_W*`) per chip — chip
l's `fwd(k+1)` would clobber the cache `bwd(k)` needs.
- Kernel: index the `MP_W*` cache region by a cfg PARITY bit (double the region; ~300 KB × 2
  fits `MP_BUFLEN` headroom); fwd stores to slot `k&1`, bwd reads slot `k&1`.
- Driver: split `kick()` into `kick_async(l)` (poke AP_START) + `wait(l)` (poll AP_DONE);
  schedule: chip l runs `fwd(k+1)` concurrently with chip l+1 `fwd(k)` / bwd sweep of `k-1`.
  Per-chip sub-batch order is preserved → per-chip gradient accumulation order unchanged
  (same sums as serial; still within-ULP semantics).
- Ideal: critical path per sub-batch 6 kicks → ~2 (each chip: 1 fwd + 1 bwd, overlapped
  across chips). Realistic ~1.8–2.2×. Gate: csim parity-slot bit-exact vs serial; csynth Δarea≈0.

## 2. Persistent on-chip weights + lazy moments (attacks the fixed share of the residual)
Weights are `static` but reloaded from DDR every kick (`mp_block.cpp` ~404–418: int + float
masters ≈ 386 KB/kick); Adam m/v (~1.15 MB) loaded+stored every BWD even when `DO_UP=0`.
- Load masters once (cfg INIT kick or a "clean" flag the driver sets after staging); reload
  only when the host restages. Load/store m/v **only on DO_UP kicks**; skip the `MP_GACC`
  DDR round-trip on `GZERO` (store-only) and on non-boundary sub-batches once item 1's
  on-chip accumulator lands.
- Payoff bounded by the item-0 measurement (if the 60 ms is mostly these loads → up to
  ~1.5×; if not → skip). URAM check: masters+moments ≈ 721 KB ≪ per-SLR URAM.

## 3. Gate the int-deploy forward during training (small, free)
The byte-identical int deploy forward + `rescale_band2` + int-cache stores run every FWD kick
(`mp_block.cpp` ~426+) but feed nothing in the training loss (bpb reads the wide `MP_WLG`).
Add a cfg TRAIN flag to skip it (~8 ms/fwd-kick per csynth: dyad_matmul 2×3.1 + 2×0.8 ms).
Keep it enabled for deploy-eval runs (it IS the mult-free deploy product).

## 4. Batch dimension inside the kick (Stage-2, revisit AFTER item 0)
Refuted as the *primary* fix when compute was 350 ms/seq; at ~65 ms/seq compute and a
possibly-fixed ~60 ms/kick overhead, B sequences/kick amortizes the fixed part:
`(60 + B·65)/B` vs `125` → B=8 ≈ 72 ms/seq (~1.7×) **if** item 0 shows the residual is fixed.
Also collapses the ACCUM kick count and retires the `MP_GACC` DDR round-trip. Interleaved
(1× cache reused) per the prior design; composes with item 1 (parity slots become B slots).

## Not bundled
- Pipelined **driver-only** overlap without item 1: impossible with one cache slot (the fwd/bwd
  sweep of one sub-batch is a strict 6-kick dependency chain).
- Kernel replication / DSP-as-adder array: post-bundle scaling levers (see fpga-training-rig).

## Order of operations
1. Item 0 timer → run alongside the current coherence run's successor.
2. Implement 1+2+3 (+4 if measurement says fixed-dominated) → csim gates (gradcheck ≤1 ULP,
   accum test, NEW parity-slot test) → `CSYNTH_ONLY=1` per-loop check → one AFI build.
3. Expected combined: kick ~121 → ~50–65 ms AND ~2× pipeline overlap → **~4–6× step rate**
   on top of today's 1.38 steps/s (projection; every number above is labeled measured/projected).
