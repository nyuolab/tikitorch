# S2 PLAN — DDR-resident + tiled streaming toward Flagship-A (~10–30M-param transformer trained to convergence on ONE F2)

> Supersedes/extends [`S1_DDR_STREAMING_PLAN.md`](S1_DDR_STREAMING_PLAN.md) (which did the per-op tile-stream
> restructure for the D=128 *on-chip* kernel to free BRAM). S2 generalizes that discipline to make the on-chip
> footprint **dim-independent** by making **weights + optimizer-state + activations DDR-resident**, so D and L can
> grow without busting BRAM. Authored by a 5-agent design workflow (4 lenses + synthesis) reading the real kernel
> / CL / buffer files. Estimates are flagged vs measured.

## 0. Where S1 actually landed (the real starting line)

S1 is **DONE on silicon, but as a hybrid and at toy scale** — S2 is designed against that real state:

- **Config that closed + ran:** D=64 / H2 / DH32 / FFN256 / L_SEQ64 / V64 / **1 layer** (halved from D=128, which
  was stuck in an area↔timing bind, WNS −4.6 ns across 5 AFI builds). The whole working set lives in **one
  monolithic on-chip buffer** (`S1_OFF_*`), host-staged once.
- **What trains on F2 today** ([`onfabric_train/S1_HOSTOPT_RESULT.md`](onfabric_train/S1_HOSTOPT_RESULT.md),
  agfi-0179e963c146d39f7 / v17): fwd+bwd on fabric, **host-side dyadic-Adam**, **FREEZE_CODES** (only EMBED +
  W_HEAD value-weights update; dyadic backbone frozen at residual-identity init). CE 4.149→2.095, acc 24→64/64,
  85.6 ms/step incl. host round-trips. Kernel clock **62.5 MHz (A0 recipe)**.
- **Three hard facts S2 inherits:**
  1. **D=128 monolithic does NOT route on F2** — the S1 AFI only closed by *shrinking to D=64*. So S2 cannot
     "bump dims"; the path to 10–30M is **DDR-streaming the working set so on-chip footprint is dim-independent**.
  2. **The crossbar SLR-crossing is the timing killer** ([`S1_AFI_FLOORPLAN_NOTES.md`](S1_AFI_FLOORPLAN_NOTES.md)):
     the failing path is `XBAR_DDR`'s wready mux spreading SLR1↔SLR2 (89% route), *not* the kernel (halving the
     kernel made it worse). **S2 makes the crossbar bigger-traffic → floorplan is on the S2 critical path.**
  3. **`do_opt` write wall is real** — on-fabric optimizer m_axi writes pass cosim, never reach board DDR; fwd/bwd
     writes commit. The working path is **host-optimizer**. S2's default optimizer = **host-side, streamed**.

**Mission, precisely:** take the D=64/1-layer/FREEZE hybrid that runs, and turn it into a D≈256–512 / 6–12-layer /
TRAIN-ALL trainer whose on-chip footprint stays ~constant by streaming W+m+v+activations from DDR4 — proving the
124M memory architecture at a scale that actually converges to real bpb.

## 1. The S2 model — sizing ladder (build smallest that proves the arch, then scale within S2)

The binding S2 risk is **multi-SLR P&R**, not tok/s, so size the *first* AFI to almost-certainly route, then scale:

- **S2a (de-risk):** **D=256 / L=4** / H8 / DH32 / FFN1024 / seq128 / V1024 ≈ **4–6M params** — first DDR-streaming
  AFI; proves the memory arch + floorplan at a size that routes.
- **S2b (flagship-A target):** **D=256 / L=12** / H8 / DH32 / FFN1024 / seq256 / V256(byte) ≈ **10.6M params** — the
  convergence run. (Maps onto the nanoGPT char-LM parameter class; see §2.)
- **S2c (stretch):** D=512 / L=8 ≈ **30M** — only if S2b's floorplan has headroom.

### 1a. ★ Required constraint — LayerNorm needs D = power of two
S1's LN uses `mean = sum >> LOG2_D` (no divider; `forge_s1_config.hpp:54`). **D MUST be a power of two**, so
**D=256 (LOG2_D=8)** is the model of record — *not* D=384 (=6×64), which breaks the no-divider LN. D=256 also
reuses the S1 **DH=32** attention head (`forge_attn` AT_DH) **verbatim** — only `L`, `seq`, and the DDR-streaming
wrapper are new. (If a non-pow2 D is ever wanted, replace the mean with a dyadic shift-add reciprocal
`sum·(2^-8+2^-9+…)` — ~5 shifts, still DSP=0 — but there is no reason to pay that here.)

**Param arithmetic (S2b, D=256/L=12, FFN=1024, V=256 byte, seq=256), `forge_buffer_gpt2.hpp` formula:**
```
P_LAYER  = 3·D² + D² + 2·D·FFN + 4·D = 196608 + 65536 + 524288 + 1024 = 787,456
P_LAYERS = 12 × 787,456                                                = 9,449,472
P_EMBED(tied, V·D)=256·256 = 65,536 ; P_POS(seq·D)=256·256 = 65,536
TOTAL ≈ 9.58M  (block/non-embed ≈ 9.45M — capacity lives in the dyadic backbone, not the embedding)
```

## 2. Trainability — grounded, not a toy

- **nanoGPT char-LM regime** (published, reproducible): a ~10M-param char transformer (6L/384 ≈ 12L/256 class)
  trains to **val ≈ 1.47 nats/char ≈ 2.12 bpb** on Shakespeare-char.
- **The multiplier-free recipe is already validated on a transformer char-LM** (ROADMAP Gate-3,
  `scripts/charlm_mixer.py --linear mfgrad`): `q(gO)ᵀ@q(x)` weight-grad + dyadic-Adam reaches gather **2.481** vs
  dense 2.48 — the substrate loses ~0.3–0.4 nats vs float, no more.
- **Demo task / dataset:** **enwik8** (90M/5M/5M byte-level char-LM; not memorizable at 10M params over 0.2B+
  tokens) for the flagship; **Shakespeare-char** for the fast smoke.
- **Target metric:** **test bits/byte**. Bar = **< 1.7 bpb enwik8** (a 10M char-model's honest reach) or **≤ 2.2
  bpb Shakespeare-char** (nanoGPT baseline within the substrate's ~0.3-nat tax). Report with tok/s + core-rail
  energy. This is a *real* LM result, the lift over S1's 64-token memorization.

## 3. Wall-clock to convergence (measured-grounded, honest)

Recovered device throughput from [`TOKS_PROJECTION.md`](TOKS_PROJECTION.md) (124M MID 720–930 tok/s, DSP=0) ≈
**267–345 GMAC/s** (df3 write-floor). S2b per-token denominator ≈ **3× the ~9.5M forward-linear MACs ≈ 28M**, ~13×
smaller than 124M.

| regime | eff tok/s | 0.22–0.3B-tok char budget |
|---|---|---|
| FLOOR (1 PE, no host amortization) | ~780–870 | **~3–4 days** |
| MID (host-opt hidden under a ≥4096-tok microbatch) | ~6,000 | **~10–14 hours** |

**S2-to-convergence ≈ sub-day to ~2 days on one F2** — faster than the roadmap's flagship-A figure precisely
because S2 is the lighter 10M model and char-LM convergence needs only ~0.2–0.3B tokens. Honest expectation for the
silicon run including overheads/restarts: **~3–7 days** of F2 wall-clock.

## 4. The memory architecture (DDR layout)

**Adopt `forge_buffer_gpt2.hpp`'s layer-major arithmetic verbatim, parameterized down** → `forge_buffer_s2.hpp`
with the *identical* region/stride scheme, so the 124M layout (S4) is literally the same header at `G_D=768/G_L=12`.
**S2 and S4 share one DDR layout generator.**

| region | encoding | size @ S2b (~9.5M) | S4 (124M) analog |
|---|---|---|---|
| `OFF_W` | int8 dyadic code (sign+exp) | ~9.5 MB | 124 MB |
| `OFF_M` / `OFF_V` | int8 dyadic-Adam m / v | 9.5 MB each | 124 MB each |
| `OFF_WFP` | int16 master | ~19 MB | 248 MB |
| `OFF_ACTCK` | int16, **per-layer block INPUT only** (recompute the rest in bwd) | ~1.5 MB | 18.9 MB |
| `OFF_GRADT` | int32, **one output-stationary tile, flushed** (NOT full-resident) | ~0.4 MB | 0.79 MB |
| LUTs/tokens | host-staged, **16-word (512-bit) aligned** (the morph `OFF_CE_LUT` alignment fix is mandatory) | tiny | same |

Resident W+m+v+master ≈ **47 MB** (F2 DDR4 is GBs → trivial). Key invariants carried from the gpt2 header: 3
bytes/param resident; grad **tiled output-stationary and flushed** (never full-resident — this is what keeps
on-chip footprint dim-independent); activation checkpoints = block-inputs only + within-block recompute in bwd.

## 5. The streaming kernel (restructure `forge_train_step_s1.cpp`)

Generalize the S1 per-op tile-stream discipline (proven twice on silicon: the on-chip-staging fix closed mode-0,
the do_opt tiling closed mode-2) so footprint is dim-independent. Each op copies the proven `dyadic_adam_region`
tile template (`static tile[TILE]` BRAM, outer `tbase += TILE` loop, `burst_load_i32` → II=1 fixed-`[i]` compute →
`burst_store_i32`). Three streaming classes:

1. **Weights stream IN per-layer/per-tile** (NEW for S2): the forward reads each layer's W as one contiguous
   layer-major burst (`layer_W(l)`), decodes the int8 dyadic code inline in the GEMM PE, computes, advances.
   **Double-buffer**: prefetch layer l+1's weights while computing layer l (the morph `TT_RESIDENT_W` reload
   idiom, silicon-validated). This lets L grow without L× BRAM.
2. **Activations**: forward **streams out** the per-layer block-input checkpoint only; backward **recomputes**
   within-block activations (GPT-2 act-checkpointing) rather than streaming the full save-set — cheaper in BRAM.
   The residual stream `R[t][d]` stays the **single resident in-place dependency carrier** (never tiled/reordered
   — the load-bearing invariant from the S1 plan).
3. **Weight-grad** uses the **`forge_wgrad_df3` output-stationary DATAFLOW pattern** (compute ∥ writeback, write-
   floor reached, 40,927 cyc, DSP=0, half-SLR): hold `gW[OT][I]` on-chip for an output-tile, stream gO+x in once
   per tile, accumulate, burst-store the tile out, never full-resident.

The one irreducible multiply (attention soft-blend Σp·V) stays on DSPs. Q/K/V heads stay **resident** per-block
(`[L_SEQ][DH]`, 32-wide — small); do **not** stream K/V (per-j DDR gather shatters II=1) — only the p_save streams.

**Bit-exact gate (non-negotiable, per-op):** math / `dyad()` / int64 golden UNCHANGED; only the access pattern
moves. After each op conversion, csim and assert `max|kernel − int64 golden| == 0`; a single nonzero reverts that
op. All transforms are provably order-independent (saves are pure writes; integer contractions have int48 headroom
so tiled partial sums are identical). **De-risk entirely in csim/CPU before any AFI.** Target clock **125 MHz (A1)**
— the streaming restructure removes the BRAM congestion that forced S1's 62.5 MHz; the achievable clock is set by
the crossbar floorplan (§7), not the compute.

## 6. The optimizer strategy on silicon (respect the `do_opt` write wall)

**Default = HOST-SIDE optimizer, streamed** (the only thing proven to work on silicon — the exact mechanism behind
S1's convergence). Per step: (1) mode-0 fwd on F2 → logits; (2) mode-1 bwd on F2 → grads (these writes commit on
silicon); (3) **dyadic-Adam step on host** (bit-exact to the kernel `#else` path, 0 mismatch / 8M element-steps),
updating W-code/m/v masters + live deploy; (4) BAR4 restage updated weights.

**S2 scaling fix:** S1 streamed back only EMBED+W_HEAD (FREEZE). **TRAIN-ALL at D=256/L=12 round-trips ~47 MB of
W+m+v/step** → ~4–5 ms restage at PCIe ~8–12 GB/s — fine, but it must be **streamed + double-buffered against the
next forward**, not synchronous per the S1 toy loop. Stream the grad readback **layer-major** so the host begins
the opt step on layer-0 grads while layer-11's are still DMA'ing back; the optimizer is batch-independent (one step
per microbatch) so at microbatch ≥512 tok the host-opt fully hides.

**Parallel HW-debug track (NOT a dependency): crack the `do_opt` write wall** — it is `do_opt`-specific (fwd/bwd
commit) → an RTL/AXI handshake bug in the optimizer's m_axi write path, below cosim's behavioral-AXI fidelity.
cosim/RTL candidates: (a) write-burst alignment / `wstrb`; (b) a missing write-response (B-channel) wait before
AP_DONE; (c) crossbar arbitration starving the opt-write channel. If cracked → on-fabric optimizer (a 3rd mode) and
the host round-trip disappears. **S2 ships on host-opt; on-fabric-opt is a stretch.** (See the (B) on-silicon
diagnostic that probes whether the m/v/wfp *master* writes land or everything drops — that result steers this.)

## 7. Floorplan + AFI (the actual critical path)

The S1 miss was *categorically* the `XBAR_DDR` wready mux crossing SLR1↔SLR2 (89% route), proven by control (a
smaller kernel made it worse). S2 adds DDR traffic + a bigger streaming kernel that now spans multiple SLRs
(**model-parallel — NEW; morph N=4 was data-parallel and does not transfer**). Apply the fixes proactively:

1. **Clone `cl_forge_muzero_ddr`, not the S1 CL.** muzero_ddr CLOSED at +0.123 ns with *empty* constraints (same
   crossbar+sh_ddr+DMA-PCIS structure). The S1 CL crammed `s1_axi_cdc` (512-bit AXI async FIFO) into the narrow X5
   DDR column = the crowding source. **S2 rule: keep the X5 DDR column muzero-clean**; pblock the kernel-mem CDC
   *out* of X5 into the SLR0 compute block.
2. **Constraints in `small_shell_cl_pnr_user.xdc` ONLY** — the HDK flow reads *only* `${SHELL_MODE}_cl_pnr_user.xdc`
   (`cl_timing_user.xdc`/`cl_synth_user.xdc` are silent no-ops; that is why S1 v9≡v10). No Tcl control-flow in XDC.
3. **Confine the crossbar to SLR1** (X0Y4:X3Y7 + X5Y4:X5Y7) so it cannot spread to SLR2; pin the streaming kernel
   to **SLR2 + SLR0-compute (NOT SLR1)**.
4. **Multi-SLR model-parallel partition (NEW work):** **op-partition within a layer** — GEMM PEs in SLR2, attention
   soft-blend (DSP) + LayerNorm in SLR0-compute, with the residual stream `R` (a narrow int16 `[L_SEQ][D]` bus) as
   the only cross-SLR hand-off → minimizes SLR-crossing nets (the thin residual crosses, not the wide weight bus).
5. **★ Prototype the time-disjoint 2:1 mux FIRST.** Host-stage (DMA-PCIS→DDR) and kernel-compute (m_axi→DDR) are
   **time-disjoint** per the F2-run sequencing → a 2:1 mux can replace the SASD crossbar and **delete the exact
   wready-mux cells that fail** (geometry-free, no kernel re-csim). Highest-leverage CL change; may remove the
   floorplan problem entirely. Also resurfaces the S1 AXI register slice on the DMA-PCIS↔mux boundary.

**AFI build order (USER runs builds; agent de-risks in csynth/cosim first):**
- **Build 0 — S2a (D=256/L=4) forward-only DDR-streaming AFI**: prove the DDR-stream CL routes + closes. The
  floorplan milestone — retire the binding risk at the smallest size before committing to D=256/L=12.
- **Build 1 — S2a full fwd+bwd**, host-opt, FREEZE → confirm the convergence trajectory reproduces on silicon.
- **Build 2 — S2b (D=256/L=12) TRAIN-ALL**, host-opt streamed+pipelined → the flagship-A convergence run; report
  real bpb + tok/s + core-rail energy.

## 8. Bandwidth sanity (compute-bound verdict carries to S2)

W+m+v ≈ 28 MB; `BYTES_PER_STEP ≈ readW + readW-again + opt-R/W(2×) ≈ ~75 MB/step`, **batch-independent**
(amortized over the microbatch). At one DDR channel ~16 GB/s → ~5 ms/step pure DDR; at MID tok/s a 512-tok
microbatch is ~1 s compute → **>100× headroom**. **S2 stays compute-bound** (DDR only binds at microbatch <~64,
which one would not run). This is the whole justification for the DDR-resident arch: streaming costs nothing once a
real batch amortizes it.

## 9. Sequenced deliverables (all local/csim-first; AFI gated to user)

| # | deliverable | where | gate |
|---|---|---|---|
| 1 | `forge_buffer_s2.hpp` — gpt2 header parameterized to S2a/b/c (layer-major W/m/v/actckpt/gradtile, 16-aligned) | `hls/forge_kernels/` | constexpr param-count asserts; identical scheme to gpt2 |
| 2 | Per-op DDR-stream restructure of `forge_train_step_s1.cpp` (weights-in double-buffered, act-ckpt+recompute, wgrad df3) at parameterized D | `hls/forge_kernels/` | **csim `max|kernel−int64golden|==0` per op**; csynth closes 2.92 ns, BRAM dim-independent |
| 3 | CPU bit-faithful convergence run at S2b dims (TRAIN-ALL, dyadic-Adam, real char-LM, ~10M) | `scripts/` | real bpb beats frozen baseline; TRAIN-ALL densification doesn't saturate (LN_F + code-weight-decay scale) |
| 4 | S2 DDR CL: clone `cl_forge_muzero_ddr`; X5-clean, CDC-out-of-X5, crossbar-confined-SLR1, kernel-SLR2+SLR0; **prototype the time-disjoint 2:1 mux** | `hdk/cl_forge_s2_ddr/` | constraints in `small_shell_cl_pnr_user.xdc`; **user AFI Build 0 routes + closes** |
| 5 | Streamed+pipelined host-optimizer driver (layer-major grad readback, double-buffered restage vs next forward) | driver | step time → compute-bound, not round-trip-bound |
| 6 | **Parallel track:** `do_opt` write-wall RTL/AXI debug in cosim | — | optional on-fabric-opt; not a dependency |

**Critical-path order:** (1)+(3) first (pure CPU — collapses convergence + memory-arithmetic risk), then (2)
op-by-op csim, then (4) the floorplan AFI. **The floorplan is the dominant uncertainty — prototype the 2:1-mux CL
early and run Build 0 at the smallest S2a size to retire it before the D=256/L=12 convergence run.**

## 10. Honest risk ledger

1. **Multi-SLR model-parallel floorplan — HIGH (binding).** NEW (morph N=4 was data-parallel). The crossbar already
   failed at S1's smaller footprint; S2 adds traffic. Mitigated: clone muzero_ddr, the 2:1-mux that deletes the
   failing cells, op-partition with only the thin residual crossing SLRs, smallest-first Build-0. **Do not commit
   to D=256/L=12 until S2a routes.**
2. **D scaling vs the D=128-doesn't-route precedent — MEDIUM.** DDR-streaming is *meant* to make footprint
   dim-independent so this stops being a wall — but it must be **measured** (csynth BRAM/LUT flat vs D), not assumed.
3. **Host-opt round-trip bandwidth at TRAIN-ALL — MEDIUM.** ~47 MB/step round-trip; mitigated by layer-major
   streaming + double-buffering against the next forward. If it binds, the on-fabric-opt track rises in priority.
4. **TRAIN-ALL densification at S2 scale — MEDIUM.** S1 only proved FREEZE on silicon; dense-backbone training
   needed LN_F + code-weight-decay (which exist) but was validated only in csim at toy scale. Deliverable (3)
   de-risks this on CPU before AFI spend. *(The (A) TRAIN_ALL host-optimizer run is the on-silicon first look.)*
5. **Wall-clock — flagship risk, mitigated by scope.** ~3–7 days F2 wall-clock for the convergence run is what
   makes flagship-A completable on one card.

**Bottom line:** S2 = `forge_buffer_gpt2.hpp` layout at S2 dims + the `S1_DDR_STREAMING_PLAN` per-op restructure
generalized to dim-independent footprint + the host-optimizer-streamed loop that already converged on silicon + a
`cl_forge_muzero_ddr`-cloned CL whose floorplan proactively kills the crossbar SLR-crossing (ideally via the
time-disjoint 2:1 mux). Build smallest-first (S2a forward-only AFI) to retire the floorplan risk, then scale to
D=256/L=12 for the flagship-A convergence proof.
