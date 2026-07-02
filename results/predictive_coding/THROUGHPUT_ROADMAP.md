# THROUGHPUT ROADMAP — Distributed PC-Transformer Training on F2

**Status date:** 2026-07-03 · **AGFI:** `agfi-0a08046c7d64090ff` · **Config:** D=64, H=2, FFN=256, SEQ=64, VOCAB=64, L=3 chips, 150 MHz CDC (6.667 ns/cyc), PIPE=1 (2 slots), ACCUM=8
**Kernel:** `tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_block.cpp` · **Driver:** `tikitorch/fpga/silicon/forge_pc_xformer_mp_shard.cpp` · **Config:** `forge_pc_xformer_mp_config.hpp`

**Label convention (used everywhere):**
- **[M]** = MEASURED on silicon or read from the csynth report of the shipped kernel.
- **[E]** = COMPUTED-ESTIMATE (csynth cycles × 6.667 ns, discrete-event schedule sim, or analytic scaling law). All cycle→ms conversions use the 150 MHz CDC period; the synth tcl's 3.0 ns constraint is an over-constraint and is never used for wall-clock.
- Csynth source: `/tmp/csynth3.rpt` (de-flatten-only kernel, pre-parity — validated close enough for loop ranking). Schedule sim: `/tmp/wf3_sched.py`, a discrete-event replica of the driver reap/dispatch loop (lines 346–379 of the shard driver).

**Anchors [M]:** FWD kick 53.5–57.6 ms · BWD kick 91.3–99.0 ms · serial 0.28 steps/s · PIPE=1 0.50 steps/s (258 tok/s), per-step wall ~2.0 s · pipelined masters bit-identical to serial · BRAM_18K 97%/SLR, LUT ~69%, DSP 15% (455/3008), URAM 3%, FF 31%, P&R WNS +0.051 ns · quality run #3 bpb 3.093 vs same-schedule CPU twin 2.834 → **0.26 bpb silicon noise premium** (gradient-noise mechanism, grows as LR decays).

---

## 1. What fills the kicks NOW (reconciled)

The csynth per-module model reproduces the measured kicks almost exactly, with one calibration: both attention loops are HLS triangular worst-case; the realistic factor is 0.52 (= Σ(i+1)/64² plus per-row overhead).

**Reconciliation [M vs E]:**

| Kick | Modeled [E] | Measured [M] |
|---|---|---|
| FWD, INPUT/HID chip | 54.1 ms | 53.5 ms (lightest) |
| FWD, OUTPUT chip | 58.2 ms | 57.6 ms (heaviest) |
| BWD, INPUT/HID chip | 91.7 ms | 91.3 ms |
| BWD, OUTPUT chip | 99.5 ms | 99.0 ms |

The measured 3-chip spread **is** the role spread: the OUTPUT chip carries head-fwd/CE/head-bwd. At TRAIN=1 the deploy-int forward (dyad_matmul, int LN, int attention) contributes **0 ms** — only the WIDE float forward runs in a training kick.

### 1.1 FWD kick breakdown (OUTPUT chip, ms @150 MHz) [E, csynth-derived]

| # | Op | Dims | Cycles | ms | Notes |
|---|---|---|---|---|---|
| 1 | wide attention `FA_*` | H=2, SEQ=64 | — | **15.92** | realistic ×0.52; worst-case 30.6. `FA_BD` P@V blend is II=8 (~17 cyc/MAC) |
| 2 | 4× D² matmul (Q,K,V,O) | 64×64 | 4×516,269 | **13.77** | shared clones `bf_matmul_6/_8` |
| 3 | FFN-UP `bf_matmul` | I=64→O=256 | 2,064,557 | **13.76** | |
| 4 | FFN-DN `bf_matmul` | I=256→O=64 | 1,314,989 | **8.77** | |
| 5 | head matmul | D×VOCAB | 516,269 | **3.44** | OUTPUT chip only |
| 6 | `bf_layernorm` ×2 (+LNF) | — | — | 1.25 (+0.62) | |
| 7 | `bf_qact` ×5 | — | — | 0.66 | |

### 1.2 BWD kick breakdown (OUTPUT chip, ms @150 MHz) [E, csynth-derived]

The backward is **~88% linear-backward**:

| # | Op | Dims | Cycles | ms | Notes |
|---|---|---|---|---|---|
| 1 | 4× D² bwd (WO,WQ,WK,WV) | 64×64 | 4×1,073,454 | **28.63** | shared clones `bf_linear_bwd_12/13/14` |
| 2 | FFN-DN bwd `bf_linear_bwd` | I=256→O=64 | 4,276,526 | **28.51** | single biggest op in the kernel |
| 3 | FFN-UP bwd `bf_linear_bwd_4` | I=64→O=256 | 3,543,726 | **23.63** | |
| 4 | attention bwd `ABWD` | — | — | **10.35** | realistic ×0.52; worst 19.9. `AB_JAC` II=4, `AB_GP` unrolled-serial |
| 5 | head bwd `bf_linear_bwd_5` | — | 1,069,358 | **7.13** | OUTPUT only |
| 6 | `bf_ln_bwd` ×2, CE, LNF-bwd | — | — | 0.59 + 0.36 + 0.30 | |

**Root cause of the BWD/FWD 1.7× imbalance:** each linear backward computes TWO matmul-equivalents (dx = dyad(W)ᵀ@gy AND wgrad = gyᵀ@x), run **sequentially** inside `bf_linear_bwd`.

**Why the linears are slow (the real lever):** the inner reduction is already II=1 with 16 partial-sum lanes — but every middle iteration pays a ~67-cycle pipeline **drain** against only ~64 cycles of useful reduction ≈ 51% overhead, paid M·I + O·I times. FFN-DN bwd is worst because I=256 gives 16,384 middle iterations each draining a short O=64 inner. **Widening the ps[] lanes 16→32/64 buys ~nothing** (lanes only break the fp-add recurrence); the correct move is unrolling the MIDDLE loop.

### 1.3 Per-step wall budget (schedule level) [E, sim; validated against M]

Sim reproduces all three anchors: serial 3.58 s (0.279 vs 0.28 [M]), PIPE=1 1.982 s (0.504 vs 0.50 [M], 258 vs 257 tok/s [M]), chip utilization 58–63%.

| Component | ms/step | Share | Verdict |
|---|---|---|---|
| Bottleneck-chip busy: 8×(57.6+99.0) | 1,253 | 63% | irreducible without kernel work; contains 461 ms of F/B single-engine serialization + 63 ms chip imbalance |
| **Slot-depth-2 idle** (period 230 vs 157 ms/seq) | **584** | **29%** | the dominant recoverable gap → S=3 fix |
| Window fill/drain at the update boundary | 145 | 7% | inherent to the weight update, NOT the barrier (barrier itself 0.3 ms); cross-step pipelining gains **0.0** |
| Host staging exposed (post-READBURST) | ~5 | 0.25% | already solved; ~8.8 ms serialized, mostly hidden |
| Unexplained vs measured | ~20 | 1% | |

Key structural fact: the dependency cycle F₀→F₁→F₂→B₂→B₁→B₀→F₀(g+2) closes through the full ~446 ms round trip with only 2 tokens in flight → period = 446/2 = 230 ms/seq instead of the engine bound max(f+b) = 157 ms.

---

## 2. Ranked future-throughput levers

**Summary table** (rank = recommended attack order = leverage per risk):

| # | Lever | Projected gain | Resource cost | Risk | Effort | When |
|---|---|---|---|---|---|---|
| 1 | ACCUM 16–32 (env knob) | 193→166 ms/seq on top of S=3; +50% cum. steps/s-equiv; **quality co-fix** for the 0.26 premium [E] | none | low (validate LR/trajectory) | hours, no build | **now** (next run) |
| 2 | Parameterized slot depth S (run S=3) | 0.50→0.647 steps/s (+28%), 258→331 tok/s [E, sim] | DDR only ~380 KB/slot; no BRAM/LUT | low (bit-identical gate, static_asserts) | ~5 h AFI + driver | **next build** (#1) |
| 3 | URAM rebind + inline `bstore_f/bload_f` | 0 direct; **enabler** for #4,#6,#7,#8 (BRAM 97%→≤40% gate) [E] | URAM 3%→~15–40% | med (timing at WNS +0.051) | ~1 day + AFI | build #2, **alone** |
| 4 | BWD rebalance: 2× middle-unroll of `bf_linear_bwd` (dx+wgrad) | BWD 95→~53–58 ms; engine bound 157→~111–116 ms/seq; with S=3 → ~0.9–1.0 steps/s-equiv [E] | +250–350 DSP (→26%), +45–70k LUT (→81–86%) | med (P&R at 2×; 4× may miss timing) | 1–2 days + AFI | build #3, after #3 |
| 5 | Attention II=1 (fwd II=8, bwd II=4 → bf_matmul recipe) | fwd attn 15.9→~3.2, bwd 10.3→~2.6 ms [E]; **the O(SEQ²) de-risk** | +5–10k LUT | low now, high value later | 1–2 days | fold into build #3; **mandatory before SEQ=256** |
| 6 | 250 MHz native (drop CDC; co-fixes = #3) | kicks ×0.60–0.65; ~100–108 ms/seq at S=3/ACCUM=32 [E] | none new beyond #3 | med (was the deferred miss; #3 removes the mux-spans-SLR cause) | ~1 day + AFI | after #3 (and ideally #4) |
| 7 | In-chip N-lane batch replication (N=2→4) | throughput mode: ~×3.6 steps/s at N=4 (2.0→~0.6 s/step); OR de-noise mode: batch ×N, premium 0.26→~0.13 (N=4) / ~0.09 (N=8) [E] | URAM 54% (N=4)/78% (N=8); DSP ~60% at N=4; LUT +8–15%/lane (the unknown) | med-high (start N=2, read the delta) | staged builds | with **(b)**, current D=64 tier only — **off the table at 30M** |
| 8 | dx‖wgrad DATAFLOW (Lever B) | ~2×/linear, composable with #4 → ~4× | ~+4k LUT/clone | med | 1 day | only if BWD is still the floor after #4+#5 |
| 9 | FWD matmul 2× middle-unroll (Lever D) | fwd matmuls 36.3→19.1 ms [E] | LUT/DSP like #4 | low | 1 day | only after #4+#5, for balance — likely superseded by the 30M wide-MAC |
| 10 | Dual-engine F/B concurrency + S≥5 | asymptote 99 ms/seq = 1.26 steps/s; 580 tok/s at ACCUM=32 [E] | duplicated working arrays; gated on #3; LUT-limited | high (days of restructure) | days | last resort at this tier — **#6 has equal payoff at lower risk** |
| 11 | More chips (b): S≥L + ACCUM≥32 | L=8 at S=2 collapses to 24% util; S≈8 + ACCUM≥32 restores it [E, sim] | none (build #2's parameterized S covers it) | low if #2 done | driver env | **(b)** step |
| 12 | 30M musts: wide MAC K=8–16 + URAM banking + DDR weight streaming; PRAW recompute; BPE head | see §2.3 | see §2.3 | high | weeks | **(c)** rebuild |

**Composition caveat [E]:** the naive "Lever A alone → 0.80 steps/s" (analysis 1) assumed steady state gated by max(fwd,bwd); the schedule sim shows the true single-engine gate is (f+b) per chip, reachable only with S≥3. After #4, the engine bound is ~111–116 ms/seq; S=3 + ACCUM≥16 reaches it → **~0.9–1.0 steps/s-equivalent** combined. The literal 8×99 ms = 0.79 s/step is **NOT a scheduling target** — max(fwd,bwd) gating requires on-chip F/B concurrency (#10) plus S≥5 plus ACCUM≥32.

### 2.1 Non-levers (measured ~zero — do not spend builds here)

- **Per-step barrier:** 0.3 ms [E, sim]. Skip.
- **Cross-step pipelining / async driver:** gain 0.0 — the weight-update boundary makes B₀(last)→F₀(first) drain inherent; sim shows S=2 cross-step == S=2 barrier to the millisecond [E]. Skip.
- **Host staging:** ~5 ms exposed of ~8.8 ms serialized (post-READBURST) [E, calibrated]. Skip at this tier; re-measure at 30M (messages grow as SEQ·D).
- **Partial-sum lane widening 16→32/64:** inner loop already II=1; buys ~nothing [E, csynth]. The middle-loop unroll (#4) is the correct realization of "widen the MAC datapath".
- **DDR bandwidth / inter-chip volume:** tens of MB/kick vs ~9.6 GB/s per 512-bit master = a few ms vs seconds of compute [E]. The design is and stays COMPUTE-bound.
- **Per-chip kernel replication at 30M:** zero BRAM room at D=64 (97%), and a D≥128 block overflows one SLR — scale via more chips for L, not replication [E]. (Replication remains live at the current D=64 tier, #7.)

### 2.2 Lever details and risks

**#2 Parameterized S (parity rework).** Parity state is DDR-resident (MP_PAR mirror of [MP_WR..MP_GROUT_END)), NOT the 97%-BRAM working arrays — a slot costs only DDR. Kernel edit is mechanical (po = slot·MP_PAR_OFF as a slot STRIDE, S× tok/tgt lanes); driver slot = g%S with S-deep hmsg2/emsg2. S≥4 buys nothing with a single engine at L=3 (engine bound 157 ms). **Must be parameterized, not a hardcoded 3rd slot** — L=8 needs S≈8. Gates: preserve GACC strict-ascending-bwd (bnext monotonicity), metric-read slot (ACCUM−1)%S, regenerate the 16-word-alignment static_asserts per slot (else burst I/O corrupts silently), re-run the 20-step bit-identical-to-serial gate on silicon. Before committing: add a 10-line timestamped dispatch/reap trace to confirm the 584 ms slot-idle attribution [M-check].

**#3 URAM rebind (the shared prerequisite).** BRAM 97% is a HARD wall: any middle-unroll needs 2–4 concurrent operand reads (array-partition 2–4) which spills BRAM and fails P&R. Rebind masters+moments+wide-cache (bind_storage impl=uram; URAM 3%→~15–40%) + inline/register the shared `bstore_f/bload_f` helpers — the same change is the known 250 MHz co-fix (helper mux spans SLRs because BRAM saturation scatters buffers). Csynth gate: BRAM ≤40%/SLR, timing unchanged at 150 MHz. Land **alone** with a serial-mode regression; one variable at a time.

**#4 BWD middle-unroll.** Attacks the top-3 BWD costs directly (28.5 + 28.6 + 23.6 = 80.7 of the ~92–99 ms kick). 2× ≈ 1.9×/linear (backward-linear total 87.9→46.3 ms); 4× ≈ 3.4× if LUT allows (~90%+, timing risk at WNS +0.051 — validate P&R, keep 150 MHz CDC). Note the D² clones are SHARED (2–3 physical) — widening a shared clone speeds every call but costs area once; don't double-count. Projections are csynth-based and exclude host relay — **re-measure steps/s after landing** rather than assuming the full gain; host relay became the bottleneck at 8-chip scale before.

**#5 Attention II=1.** Cheap now, but this is the **scaling lever**: attention is O(SEQ²) and grows 64× to SEQ=256 — it must be at the bf_matmul II=1 recipe before the 30M rebuild or it dominates. Caveat: triangular worst-case is ~2× the realistic figure; validate short-row overhead doesn't dominate at II=1.

**#6 250 MHz.** Equal payoff to dual-engine (#10) with far less area risk; est. 60–75% close per the deferred note, unlocked by #3. Do not combine with a widen build in one AFI.

**#7 N-lane replication (current tier).** Share masters (~242 KB) + Adam moments (~480 KB, read-only during fwd/bwd, weight-streaming broadcast to N lane MAC groups); replicate the wide cache (~370 KB/slot); avoid per-lane grad buffers via a fixed-order shared accumulator (~61k adds, <1 ms) which **exactly reproduces the serial ACCUM summation order** → the bit-exact-to-CPU-twin story survives. N-lane top loop, NOT N kernel instances (single AXI master, one FSM). Risks: LUT/lane unknown (start N=2, read csynth delta); fixed lane order mandatory (tree-sum reassociation breaks the twin match); re-verify pipelined-bit-identical with lanes. Its **quality co-benefit** is first-class: grad noise /√N attacks the measured 0.26 bpb premium directly (→~0.13 at N=4 [E]).

### 2.3 30M-tier musts (the (c) rebuild: D=256, SEQ=256, L=6–12, BPE VOCAB 4–16k) — all [E]

Work scaling: linears SEQ·D² (64×), attention SEQ²·D (64×, fastest in SEQ), head SEQ·D·VOCAB (**1024×** at VOCAB=4k; 4096× at 16k — the head becomes the largest single matmul and lives on ONE chip).

**What breaks first, in order:**

1. **BRAM — breaks immediately (the structural wall).** Already 97% at D=64 with everything resident. A single D=256 FFN weight = 512 BRAM_18K; ~17 FFN·D-class resident arrays → ~8000+ BRAM_18K needed vs 4032/FPGA. Resident set at D=256 (~3.9M weight/opt words + ~5.7M wide-cache-with-parity) saturates ALL on-chip memory (BRAM+URAM ≈ 9.9M words); D=768 is ~10× over. **Verdict: the model will not fit resident past D≈96–128. Weight streaming from DDR (per-matrix/per-tile) is mandatory, not optional.** DDR footprint is fine (~40–56 MB at D=256, ~200–400 MB at D=768).
2. **Compute at 1 MAC/cycle.** D=256 HID kick ≈ 5.6–9 s (fwd+bwd), OUTPUT chip 13–20 s head-dominated [E, ±1.5× band]. **Wide MAC K=8–16** (unroll the contraction, adder-tree reduce) cuts the ~90%-linear share by K: K=4 → 1.5–2.7 s, K=16 → 0.6–1.3 s (attention+LN then become 40–50% — diminishing returns past ~8× unless #5 also lands). DSP is NOT the limiter (K=16 ≈ 50–150 of 3008); the limiter is **operand read ports** → wide-MAC and URAM banking (2 ports/bank, ~960 banks idle) are ONE joint redesign with DDR streaming.
3. **PRAW/Pcache (H·SEQ² attention probs): 32×/copy explosion.** 16 BRAM_18K/copy today → 512 at D=256/H=4 (1024 at H=8), ×~4 live copies (fwd Praw, DDR MP_WPRAW, bwd Praw, parity) → up to ~2048 BRAM_18K ≈ 1.5 SLRs of BRAM for softmax probs. **Fix: recompute-in-backward** (attention-prob checkpointing) from the cached WQf/WKf — cost ~34M MACs ≈ 5% of block work. Cheap, high-leverage, do at the SEQ=256 rebuild.
4. **BPE head:** weight-tie embed↔head (halves VOCAB·D storage; changes the backward — tied grad accumulates both paths, re-verify bit-exact vs golden); vocab-TILE + stream the head weight; consider sharding head/CE across chips (compute-imbalanced on OUTPUT alone).
5. LUT (69%) secondary; DSP/DDR/host-relay NOT walls — re-measure host relay once the MAC is ~10–30× wider.

**124M point (D=768/L=12/VOCAB≈50k):** ~40–80 s/HID-block-kick at 1 MAC/cycle; head at 50k the dominant single op — confirms the flagship framing (convergence-train ~10–30M on F2, 124M datapath measured+projected). The levers set whether a kick is ~1 s or ~10–80 s.

**Gate before committing the (c) plan:** run ONE D=256 csynth to confirm these analytic projections (they are extrapolations off the D=64 report with a D-dependent fill-overhead factor ~2×→~1.25×).

---

## 3. Do this next (one page, consistent with the approved (a)→(b)→(c) plan)

**(a) Current tier — squeeze the 3-chip rig:**

1. **Now, no build:** raise ACCUM to 16–32 (env only) — throughput AND the planned batch de-noising fix for the 0.26 bpb premium from one knob; validate quality on a short run (LR may need rescaling). Add the timestamped dispatch/reap trace to the driver to empirically confirm the 584 ms slot-idle attribution. **[expected: ~0.50→0.55-equiv + quality]**
2. **Build #1 — parameterized slot depth S** (run S=3 at L=3): mechanical, ~5h AFI, DDR-only cost. Re-run the bit-identical-to-serial silicon gate. **[0.50→0.647 steps/s, 331 tok/s; with ACCUM=32 → 166 ms/seq ≈ 0.75-equiv, 386 tok/s]** This build also future-proofs (b): S must scale ≥L.
3. **Build #2 — URAM rebind + helper inline, ALONE**, serial regression, csynth gate BRAM ≤40%/SLR. Zero direct gain; unlocks everything after.
4. **Build #3 — BWD rebalance (2× middle-unroll of `bf_linear_bwd`) + attention II=1** in one kernel pass; stage csynth after each change; hold 150 MHz. **[engine bound 157→~111–116 ms/seq; with S=3+ACCUM≥16 → ~0.9–1.0 steps/s-equiv, ~460–510 tok/s]** Re-measure on silicon — if host relay surfaces, stop widening and fix the relay.
5. **Build #4 (optional) — 250 MHz native** (co-fixes already landed in #2). **[kicks ×0.60–0.65]** Skip dual-engine unless a measured gap remains — 250 MHz has equal payoff at lower risk.

**(b) More chips (8/instance, then TcpLink 2-box):** run S≥L (=8) and ACCUM≥32 — at S=2, L=8 collapses to 24% utilization. Re-measure host relay per-call (sim constants are calibrated estimates; the L=8 projections assume same-box PCIe; TCP is unmodeled). Optionally land **N=2 lane replication** here if the quality premium (not throughput) is binding — it composes multiplicatively and halves-ish the noise premium.

**(c) 30M rebuild (D=256/SEQ=256/BPE):** ONE joint kernel redesign = **wide MAC (K=8–16) on URAM-banked, DDR-streamed weights** (fit-enabler AND throughput-enabler) + **PRAW recompute-in-backward** + **BPE head** (tie/tile/shard, bit-exact-vs-golden gate). Do NOT carry replication or host-relay work into this tier. Run a single D=256 csynth first to confirm the projections before committing.

---

## 4. Provenance ledger

| Claim class | Label | Source |
|---|---|---|
| Kick times 53.5–57.6 / 91.3–99.0 ms; 0.28/0.50 steps/s; 257–258 tok/s; bit-identical pipelining; bpb 3.093 vs 2.834; WNS +0.051; BRAM 97%/LUT 69%/DSP 15%/URAM 3% | **MEASURED** | silicon runs on agfi-0a08046c7d64090ff; csynth top-module report |
| Per-op ms tables (§1.1–1.2); attention ×0.52 factor; drain-overhead diagnosis | **COMPUTED-ESTIMATE** (csynth cycles × 6.667 ns, reconciled to measured kicks to ~1%) | /tmp/csynth3.rpt (de-flatten-only, PRE-parity kernel) |
| Step budget (§1.3), S=3/ACCUM ladder, L=8 collapse, dual-engine asymptote | **COMPUTED-ESTIMATE** (discrete-event sim reproducing 3 measured anchors; host constants calibrated not per-call-measured; <3% sensitivity to kick attribution) | /tmp/wf3_sched.py |
| Lever gains (#4–#10), area deltas, replication budgets | **COMPUTED-ESTIMATE** (csynth extrapolation; one measured calibration point: DSP-bound 16-lane II=1 linear = +27 DSP/+7k LUT vs DSP-free twin) | csynth + kernel source |
| 30M/124M kick times, BRAM/PRAW verdicts, fit wall D≈96–128 | **COMPUTED-ESTIMATE** (analytic scaling laws off the D=64 report, ±1.5× band; BRAM counts ±1 block; confirm with one D=256 csynth) | §2.3 |

---

## UPDATE 2026-07-03: Lever A + S=8 built (csynth-gated)

- **S=8 pipeline slots** (rebuild #4, decoupled): DDR-only, csynth free-fit (BRAM 97%/URAM 3% == shipped
  de-flatten AFI). Fixes the S=2-at-L=8 utilization cliff (~24% → full). AFI building.
- **Lever A + budgeted URAM** (rebuild #5): csynth-CONFIRMED **BRAM 61% / URAM 48% / LUT 74%, one SLR**, and
  **bf_linear_bwd 4.28M→2.19M cyc = 1.95×** (the backward-linear 2× win, II=1 preserved). Budget lesson from
  #4: bind ONLY high-fill (≥4096 words/bank) arrays — Wd/HDpre/gHD/HDa/WRMASK — not tiny high-bank arrays
  (which wasted URAM to 237%). AFI building concurrently. Projected BWD kick 91→~58 ms → pipeline steady-state
  (gated by max(fwd,bwd)) ~1.6× on top of S=8.

