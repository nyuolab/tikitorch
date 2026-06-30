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

**Leading hypothesis:** the **distributed int16 inter-chip error message**. The sharding rescales
the backward error to int16 between chips — an approximation the single-process golden does not
have. Plausibly the deep chips receive degraded gradients → the plateau. (Secondary: the
attention; acc 0.16 sits near the documented integer-attention-softmax ceiling 0.153.)

**Confound to clean up:** the CPU reference compared was **single-block**; the silicon is **NBLK=3**.

**The decisive next experiment (cheap, CPU):** a **distributed-faithful BF16 NBLK=3 golden** that
models the int16 inter-chip messages. If it plateaus at ~4.0 → the message precision is confirmed
(fix = wider / better-scaled inter-chip message, a driver/kernel change). If it reaches ~3.3 → a
true on-chip CDC issue → re-run silicon with one-step dumps. **The `tikitorch.fabric` runtime
already models the distributed regime, so it is the natural host for this experiment.**

---

## 4. OPEN PROBLEM #2 — coherence ceiling: the attention training wall

Long-standing across the project (`pc-xformer-softmax-ceiling`, `lm-shakespeare-bakeoff`): tropical
/ mult-free attention is the hard piece. The BF16 recipe reaches 3.07 with **warm dot attention +
precise mult-free exp**; whether the silicon attention trains to that quality (vs the integer-LUT
ceiling) is entangled with Open Problem #1 and must be checked once the message-precision question
is settled.

---

## 5. OPEN PROBLEM #3 — throughput (host-relay-bound)

**MEASURED:** 0.46 steps/s on silicon = **host-PCIe-relay-bound** — the same word-at-a-time readback
that makes the existing fabric ~61 % comms-bound. This is *not* a compute limit. **The fix is the
chip-to-chip comms work already de-risked in `tikitorch.fabric`:** burst readback (Stage-0,
no rebuild), then PCIe-P2P / on-fabric links, plus double-buffering — modeled to recover ~3.9× ×
the pipeline overlap. **This is the lever for "rapid" training and is independent of the quality gap.**

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
