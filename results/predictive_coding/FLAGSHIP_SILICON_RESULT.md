# Distributed PC Transformer on AWS F2 — Flagship Result & Status

**Honest one-liner:** a multiplier-free-forward + BF16-backward **predictive-coding transformer,
trained model-parallel across 3 physical AWS F2 FPGAs with no global gradient, trains on silicon
(bpb 6.9 → ~4.0)** — the *structural* flagship holds. **It does not yet reach coherent quality**
(it plateaus at bpb ~4.0 / acc 0.16 vs the CPU recipe's 3.07 / 0.37); that gap is **silicon-side
fidelity**, precisely localized below, not a recipe failure.

Everything here is labeled **MEASURED** (on real silicon / real Vitis / real CPU runs) vs
**PROJECTED** (modeled). AGFI `agfi-05ec62437c0bf8669` (afi-09b98dec099565b36).

---

## 1. What WORKED — MEASURED

### On silicon (f2.48xlarge, 3 FPGAs, `mpxfshard-20260629T234059Z`)
- **Distributed PC transformer training runs end-to-end.** One BF16 transformer block per chip
  (D=64, H=2, FFN=256, seq=64, vocab=64), 3 chips model-parallel, **neighbour-only messages, no
  global gradient / all-reduce**. The kernel **computes correctly across the clock-domain crossing
  on real hardware** — no crash, no NaN, genuine learning.
- **bpb 6.925 (step 0) → ~4.0 (plateau, min 3.797); accuracy 0.031 → ~0.16** over ~2400 of 3000
  steps (hit the 90-min instance deadline). The mult-free dyadic forward + BF16-STE backward +
  dyadic-Adam datapath produces real gradient descent on silicon.
- **0.46 steps/s — host-PCIe-relay-bound** (the predicted bottleneck; see §5).

### The kernel is silicon-ready (Vitis csynth + real P&R)
- **mp kernel csynth** (`forge_pc_xformer_mp_block`): mult-free dyadic forward DSP_MULT=0; BF16
  backward binds DSP as expected. At `factor=2` unroll + 3.0 ns over-constrain: **LUT 62%, BRAM
  80%, DSP 12%, est 2.668 ns** — fits one SLR.
- **Timing CLOSED** (build #8/#9): **0 violated paths / 113 met**, clean `post_route.dcp`, via the
  CDC (kernel at 150 MHz) + the reset false-path. `create-fpga-image` accepted it → AGFI.

### On CPU — the recipe and the fabric twin
- **The BF16 recipe reaches coherence**: the CPU golden (`forge_pc_xformer_bf16_golden_test`,
  BF16_BWD=2) reaches **bpb 3.07 / acc 0.37** (2500 steps, ACCUM=4) — coherent TinyShakespeare.
- **`tikitorch.fabric`** (the distributed runtime, CPU `ThreadLink` backend, 13/13 tests):
  - pipeline overlap **5.60×** (vs 5.57× theoretical max);
  - comms-cost model calibrated to the MEASURED 61 % host-relay fraction → transport choice
    **word→burst 3.9×**, plus **double-buffering +1.27×** on the slow path;
  - **1F1B schedule** with depth-proportional staleness `[5,4,3,2,1,0]` (the real HW mechanism);
  - **staleness→quality** measured: penalty 1.45× at W=1, recovered at W ≥ depth;
  - a **gradient-checked (~1e-9) real transformer char-LM** trains on the fabric (bpb 4.04 → 0.04).

---

## 2. The timing-closure saga — what DIDN'T work, and the root cause of each

Nine AFI builds. The kernel arithmetic and area were never the problem (csim bit-exact, fit one
SLR throughout); it was all build/timing **integration**. Each was root-caused, not flailed:

| # | Failure | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | CL synth: `module ..._ip not found` | the BF16 backward's float ops compile to Xilinx floating_point IP; `csynth` emits only the wrappers + companion `*_ip.tcl` generators, never runs them (the int kernel had zero float ops) | source the `*_ip.tcl` in `synth_cl` → synthesize the FP cores inline |
| 2 | Routed but **timing VIOLATED −1.688 ns @ 250 MHz** | the BF16 float backward fills 79 % of one SLR; **structural** routing delay (~2.3 ns) the csynth estimate didn't see | (attempted congestion fixes, then CDC) |
| 3 | `--clock_recipe_a A0` fast-fail | **F2 hard-locks `clk_main_a0` at 250 MHz** for every recipe (125 MHz is `#NOT SUPPORTED#` in the shell); custom recipes also need `aws_clk_gen` | abandon the recipe path — timing must close at 250 MHz or on a derived clock |
| 4 | `HDK_IP_SRC_DIR not set` | `hdk_setup.sh` needs git-lfs to pull the F2 shell; the Ubuntu AMI's auto-installer uses `yum` → fails | install git-lfs deterministically (apt → static binary) |
| 5 | Still VIOLATED (−1.025 ns) | congestion relief (LUT 79 %→62 %, unroll 8→2 + over-constrain) helped WNS but the **routing penalty is structural**, not congestion | → CDC (slower kernel clock) |
| 6/7 | CDC routed but VIOLATED; diagnostic | `clk_kernel` IS 150 MHz and the **kernel compute paths MEET (+0.206 ns)** — the CDC works; the **sole** violation is a **reset-CDC recovery check** (−3.270 ns on `rst_kernel_pipe/CLR`), an async-assert/sync-deassert synchronizer that needs a false path. ALSO: the CL's constraint files were mis-named — the F2 shell reads `${SHELL_MODE}_cl_pnr_user.xdc`, so user constraints were silently ignored | `set_false_path` on the reset crossing, in the correctly-named XDC |
| 8 | **CLOSED** (0 violated) but no tarball | this F2 HDK writes `Developer_CL.tar` to `checkpoints/` directly, not `checkpoints/to_aws/` (the F1 convention the upload assumed) | grab the tar wherever it is, upload only if timing met |
| 9 | **CLOSED + tarball captured** → AGFI | — | — |

**Dead ends that are now documented refutations:** the fully-integer backward (caps at gibberish,
refuted earlier → BF16 is the fix); lowering the F2 main clock by recipe (impossible); closing the
BF16 backward at 250 MHz by congestion alone (the penalty is structural).

---

## 3. OPEN PROBLEM #1 — the silicon quality plateau (the headline gap)

**Symptom:** silicon plateaus at **bpb ~4.0 / acc 0.16**; the CPU recipe reaches **3.07 / 0.37**.

**Diagnosis so far (MEASURED):**
- **ACCUM is REFUTED as the cause.** The CPU golden with the silicon's *exact* config (ACCUM=1,
  LR=3e-3, warmup=120, BF16_BWD=2, seed=0) reaches **bpb 3.65 / acc 0.28 at 1500 steps and is still
  falling** — it does **not** plateau. ACCUM=4 is only marginally better (3.69). So the silicon
  underperforms the **identical-config** CPU reference → the gap is **silicon-specific**, not the
  recipe / hyperparameters.
- A mild, systematic shortfall (4.0 vs 3.65, not garbage/NaN) is the signature of **precision loss**,
  not a catastrophic CDC data bug.

**Hypotheses tested → REFUTED:**
- *int16/bf16 inter-chip MESSAGE precision* — REFUTED. `scripts/pc_fabric_msg_precision.py` trains
  the same multi-block char-LM varying ONLY the inter-chip message encoding (exact-float vs
  int16-per-tensor vs bf16): **Δ = 0.0000 bpb**. Message quantization is benign.
- *coarse attention softmax* — REFUTED by inspection: the mp kernel forward is the **precise
  mult-free exp + warm dot attention**, not the exp2-LUT ceiling.
- *the BF16 backward / dyadic-Adam* — NOT the difference: the kernel's backward + optimizer are
  **csim-bit-exact to the bf16 golden** (transcribed from it), and that golden reaches 3.07.

## ★ RESOLVED (2026-06-30): the plateau is **batch = 1**, not a bug

After an exhaustive elimination (a 5-agent root-cause Workflow + a dozen CPU/silicon probes), the cause is
standard and the kernel is **sound**:

**1. The kernel is sound — proven on silicon.** A fixed-batch overfit (`FIXEDOFF=0`, pin every step to one
64-token sequence) drives the silicon to **bpb 0.103 / acc 0.984** — it memorizes one batch near-perfectly,
smoothly and deterministically. So there is **no kernel fault, no CDC corruption, no relay bug**; the
datapath computes + updates + is reproducible.

**2. Root cause = effective batch size 1.** The silicon updates weights from **one sequence per step**
(ACCUM=1). The kernel **zeroes its gradient accumulators every backward kick**, so it *cannot* accumulate —
the driver's `ACCUM>1` just issues extra kicks and applies only the last sub-batch's gradient. Batch-1
training explains **both** silicon symptoms at once: noisy single-sequence gradients → the per-step bpb
**bounce**, and a high gradient-noise floor → the **~4.0 plateau**. Confirmed on the exact bf16 golden:

| effective batch | golden 1-block bpb | |
|---|---|---|
| **1** (= silicon today) | **3.74** | noisy, plateaus |
| **8** | **3.20** | crosses the coherence bar (~3.1) |
| 48 (= the torch recipe) | 2.86 | coherent |

**3. Everything else was refuted** (each on a faithful CPU testbed): bf16 forward *and* backward
(`Δ0.024`), int16/int8 error quant, LR/warmup/depth, message precision, the relays (code-recon: forward is
**raw float32**, backward bf16 — both clean), and the workflow's headline **frozen-LN / residual-runaway**
(decisively disproved — freezing the LN on the real recipe stays smooth at 2.838, and the torch residual
grows to 327 ≫ silicon's 35.8 *without* any instability). The "3-chips-worse-than-1-block" comparison was an
artifact of comparing a batch-1 silicon run to mismatched references — at equal batch the kernel recipe is
fine. **The flagship structural claim is fully intact: the distributed PC transformer trains *correctly*
across 3 FPGAs with no global gradient; it just trains at batch 1.**

### The fix — one rebuild delivers coherent **and** rapid
- **(A) Coherence — true gradient accumulation in the kernel.** Persist `gW` in DDR; zero on the first
  sub-batch of an ACCUM window, **add** on each, apply dyadic-Adam on the **last** over the averaged
  gradient. All ACCUM sub-batches see the same (un-updated) weights → correct accumulation → effective
  batch 8–32 → **coherent ~3.0**. (The driver already sequences `fwd(k); bwd(k)` per sub-batch with
  `do_update` only on the last — only the kernel's zero-every-kick needs to change.)
- **(B) Rapid — revert `factor=2` → `factor=8` unroll.** The unroll was cut only to close 250 MHz timing;
  the AFI runs at **150 MHz** (the CDC), which has the slack back → ~4× faster kick, no other change.
- Verify in csim (bit-exact to a golden-with-accumulation) → free csynth gate (DSP_MULT=0, timing@150 MHz,
  one-SLR) → **one AFI build** → re-run at `ACCUM=16`.

**BATCH vs DATA disentangled (the ACCUM sweep ran at fixed steps, so larger ACCUM also saw more sequences):**
at a *matched* 9600 sequences, ACCUM=1 (batch 1) descends to ~3.35 then **wanders back up to 3.52** (noisy,
floors ~3.4–3.5), while ACCUM=8 (batch 8) reaches **3.20** smoothly. So it is **both**: the old 3000-step
runs were *undertrained* (→4.0), but batch-1 has an intrinsic ~3.4–3.5 noise floor regardless of data — only
a real batch converges below the coherence bar. **Batch genuinely matters → the accumulation kernel is the
right fix, not "just train longer."** (Fully refutes the data-only alternative.)

### IMPLEMENTED + csim-VERIFIED (2026-06-30, commit `72bddd6`)
- config `MP_GACC_*` persistent DDR accumulators (WQ..WDN, head, embed, pos) per chip; kernel reads `GZERO`
  (cfg bit 159:128) and accumulates at all three grad sites (block weights, OUTPUT head, INPUT embed/pos);
  driver `setcfg` gains `gzero=(k==0)`; runner carries `ACCUM`. All 16 unroll pragmas `factor=2`→`8`.
- **Verified (local g++ csim, authentic ap_int + hls shims):** the gradcheck is still **bit-exact at ACCUM=1**
  (KAT all-ones 0 / KAT random ≤1 bf16 ULP), and a new `forge_pc_xformer_mp_accum_tb` confirms the
  accumulators **sum exactly** (acc1=g, acc2=2g, max|diff|=0.00e+00). DSP_MULT=0 preserved (adds + memory only).
- **Next (user runs the build):** `bash scripts/run_mp_afi_build_f2.sh` (csynth gate inside it: DSP_MULT=0,
  timing@150 MHz, 1-SLR) → new AGFI → `AGFI=<new> ACCUM=16 STEPS=large bash scripts/run_mp_xformer_shard_f2.sh
  launch` → the coherent (~3.0) + ~4×-faster-kick flagship run.

---

## 4. OPEN PROBLEM #2 — coherence ceiling: the attention training wall

Long-standing across the project (`pc-xformer-softmax-ceiling`, `lm-shakespeare-bakeoff`): tropical
/ mult-free attention is the hard piece. The BF16 recipe reaches 3.07 with **warm dot attention +
precise mult-free exp**; whether the silicon attention trains to that quality (vs the integer-LUT
ceiling) is entangled with Open Problem #1 and must be checked once the message-precision question
is settled.

---

## 5. THROUGHPUT — ★SOLVED 2026-07-01: 2.94× measured (the II=80 story)

**MEASURED ON SILICON: 1.38 steps/s (was 0.47) = 2.94×; kick 355 → ~121 ms.**
AFI `agfi-0846ef96674df0815` (partial-sum kernel, timing MET WNS +0.003 ns @150 MHz CDC).

**The road there is the lesson.** Three plausible levers each failed on silicon (each a ~5 h build):
1. ~~host-relay-bound~~ → bursting the readback: 0.47 vs 0.46 (I/O is bursted, ~0.2 ms/kick).
2. ~~inner-compute-bound~~ → `factor=2→8` unroll: no change.
3. ~~dyad transcendental in the inner MAC~~ → hoisting it (bit-exact, correct, 64× redundancy): no change.

The **csynth per-loop table** (a `CSYNTH_ONLY=1` ~15-min pass, no P&R — now built into
`run_mp_afi_build_f2.sh`, which uploads every module's `*_csynth.rpt`) gave the real answer in one shot:
the dominant reductions (`bf_matmul` BM, `bf_linear_bwd` LR/WG, trip 131k) were pipelined at **II=80**
(target 1) — the `factor=8` UNROLL chains 8 dependent fp-adds into ONE accumulator (~10 cyc each). The
unroll *created* the stall; that's why widening it (and everything else) was a no-op.

**Fix:** 16 independent partial-sum lanes (`ps[i&15] += w·x`) + tree-reduce → **II 80→1** (csynth-confirmed;
top-fn 355 → 44–65 ms; `bf_matmul` 31→12 ms, `bf_linear_bwd` 63→10 ms). Commits `db51373`/`20f9a41`.
- **Honest deltas:** silicon ~121 ms vs csynth 44–65 ms → ~60 ms/kick residual (DDR access latency /
  data-dependent bound) not yet attributed — the next csynth-vs-silicon comparison target.
- **Precision trade-off:** the reduction reorder makes grads/acts **within 1 bf16 ULP** of the golden
  instead of bit-exact (standard for a fast GEMM; deterministic; benign for dyadic-Adam). The gradcheck
  gate is re-aligned to ≤1 ULP and still fails on any real bug.
- Remaining levers (unbuilt): pipelined driver / 1F1B across chips (~3×), persistent on-chip weights
  (scaling-critical as D grows), gating the dead int-deploy forward during training (~24 ms/fwd-kick).

**PROCESS RULE (memory: hls-reduction-ii-lesson): no perf-motivated rebuild without the csynth per-loop
report first.** At 1.38 steps/s, ACCUM=8×1500 steps ≈ 2.4 h — the coherence run is practical.

---

## 5b. COHERENCE RUN #1 (2026-07-02, MEASURED): batch-8 lands at 3.825 — better, structured, not yet coherent

**Run:** `ACCUM=8 STEPS=1500` on the fast AFI (`agfi-0846ef96674df0815`), 3 chips, 2.4 h.
**Final: ema 3.825, acc 0.240 (15× chance). Gate-C clean: 0 weights / 0 gradients crossed chips
(786 MB of pure neighbour messages).** The 400-token sample decodes to word-shaped proto-English with
Shakespeare *structure* (line breaks, speaker-colon headers, dialogue punctuation: `S: GRkmeushe…`,
`Whpuindo,` `My :`) — consistent with ~3.8 bpb: real structure, not yet coherent words.

**Honest read:** real accumulation moved the plateau 4.0-4.5 → **3.83 flat** (from ~step 600) — but the
CPU references DESCEND past it. With kernel-exact ACCUM semantics (grads summed over the window, one
dyadic-Adam on the sum):

| @step 600 → 700, ACCUM=8 | ema | trend |
|---|---|---|
| 1-block exact-kernel repro | 3.56 → 3.46 | descending (→ golden's 3.196@1200) |
| **3-block exact-kernel repro** | 3.61 → 3.53 | **descending** |
| **silicon 3-chip** | 3.80 → 3.83 @1400 | **flat** |

So the plateau is NOT depth/frozen-LN recipe (the 3-block repro descends).

**PLATEAU ROOT-CAUSED (2026-07-02): the constant-LR noise floor.** Full elimination, each on a
faithful twin with kernel-exact ACCUM semantics: depth ✗ · exact init ✗ (bit-exact `init_master` regen)
· exact data order ✗ (glibc `srand(123)` reimplemented + verified against the canonical vector; real
token bin) · **fp32 weight/moment state ✗** (the fp32-adam twin descends 3.564@699 — refuting my own
registered underflow hypothesis, as did the silicon: predicted FLAT, measured DESCENDS). The decisive
silicon measurement: **resuming the plateaued masters at LR 1e-3 descends immediately, 3.83 → ema
3.649 in 500 steps** (the best silicon number of run #1's lineage), approaching a new ~3.6 floor.

**Mechanism:** SGD-family training floors at ≈ LR × gradient-noise. The silicon's effective gradient is
noisier than the CPU references (bf16 inter-chip messages + the 1-ULP reduction reorder + fp32
arithmetic vs the refs' fp64), so the same constant LR floors higher (3.82 vs ~3.4). Calibration:
floor(3e-3) ≈ 3.82, floor(1e-3) ≈ 3.6–3.65; crude linear extrapolation → asymptote ~3.5, i.e. the
schedule closes most of the gap and the *residual* vs CPU is the noise itself — which the in-chip
batch-replication lever (N datapaths/SLR sharing URAM weights, gradient summed locally, no global
sync; morph-LM-proven N=4 linear) attacks as batch de-noising.

**Fix (host-side, no rebuild, commit `4f680f8`):** warmup → cosine LR decay to `LR×LR_MINFRAC`
(`LR_DECAY=1`, driver + runner).

## ★ 5c. COHERENCE RUN #2 (2026-07-02, MEASURED): **bpb 3.187 — converged on the fabric**

**From scratch, ACCUM=8 × 2000 steps (16,000 sequences ≈ 1.0M tokens), cosine 3e-3→3e-4, 3.2 h, 88 tok/s,
3 physical FPGAs, Gate-C clean.** Monotone descent the whole run (3.80@600 → 3.58@1000 → 3.32@1400 →
**ema 3.187 @2000**, still creeping down), **acc 0.348 ema / 0.422 final batch (22–27× chance)**.

| same init / order / data / schedule | ema @2000 |
|---|---|
| exact CPU twin (fp64 wide) | 3.049 |
| **silicon, 3 chips** | **3.187** (**gap 0.14** — was 0.4+ at constant LR) |
| (single-block golden, constant 3e-3, for scale) | 3.196 |

**Sample (decoded):** real English-like words + Shakespeare structure — `ING LAT:` (a mangled speaker
header), `And kne, wher, thy lot's ge shele;`, `Noeed for a tring to kis se hat` — vs run #1's letter
salad. The remaining 0.14 vs the twin is the silicon gradient-noise premium (bf16 messages + reduction
reorder + fp32), which shrinks with batch — the in-chip batch-replication lever's job.

**This is the "trains performantly and converges on the fabric" milestone:** the distributed PC
transformer — one block per physical FPGA, no global gradient, on-chip batch accumulation, host-side
LR schedule — converges to within 0.14 bpb of its exact CPU reference at 2.94×-improved throughput.

---

## 6. PATH FORWARD — to a coherent model, trained rapidly

Two orthogonal tracks (the quality gap and the throughput gap are independent):

**A. Close the quality gap (coherence):**
1. Build the distributed-faithful BF16 NBLK=3 CPU golden (int16 inter-chip messages) → isolate
   message-precision vs CDC vs depth (Open Problem #1).
2. If message-precision: widen / rescale the inter-chip error encoding (driver/kernel), re-verify
   on CPU, then re-run silicon → expect bpb toward ~3.1.
3. Confirm the attention trains to BF16 quality on silicon (Open Problem #2).

**B. Make it rapid (throughput):**
1. Burst the host readback (Stage-0, no rebuild) → first measured relay-bursted steps/s.
2. `HostRelayLink` / `PcieP2PLink` fabric backends driving the real AGFI; double-buffering.
3. Then replication + scale (D=64 → larger; more chips).

**Definition of done for "coherent + rapid":** silicon bpb ≤ ~3.2 with recognizable Shakespeare,
at a steps/s no longer relay-bound — both now have a concrete, measured-on-CPU plan.

---

## Reproduce / artifacts
- AGFI: `agfi-05ec62437c0bf8669`. Build: `scripts/run_mp_afi_build_f2.sh`. Run:
  `AGFI=... scripts/run_mp_xformer_shard_f2.sh launch`. CPU golden:
  `g++ -O2 -std=c++14 tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_bf16_golden_test.cpp`
  then `BF16_BWD=2 ACCUM=1 STEPS=1500 ...`.
- Fabric: `PYTHONPATH=. python3 scripts/pc_fabric_{sim_demo,comms_sweep,staleness_quality,charlm}.py`;
  tests `pytest tests/test_fabric.py tests/test_transformer_stage.py` (13/13).
- Build saga commits: `7ceaea5` (FP-IP) → `d1ea535` (unroll/over-constrain) → `26d883e` (CDC) →
  `b97c30a` (reset false-path) → `7f2da08` (tarball capture). Fabric: `807bce7`, `352480c`,
  `ea3815b`, `2a64440`, `2f42e18`, `b41af98`.
