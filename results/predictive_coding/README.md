# Predictive-Coding on AWS F2 — Master Documentation Hub

Navigational index for the distributed predictive-coding (PC) transformer effort.
New here? Start at section 2 (the stack) then jump via the file map (section 3).

> **★ Flagship status → [FLAGSHIP_SILICON_RESULT.md](FLAGSHIP_SILICON_RESULT.md)** — the
> consolidated result: a distributed BF16 PC transformer **trains on 3 physical F2 FPGAs**
> (bpb 6.9 → ~4.0, no global gradient), the 9-build timing-closure saga (root-caused), and the
> open problems (the silicon quality plateau — *silicon-specific, ACCUM refuted*; throughput;
> attention). Honest: structural flagship holds, **coherent quality not yet**.

---

## 1. What this is

A **distributed predictive-coding transformer trained on AWS F2 FPGAs with no global
synchronization** — one transformer block per physical chip, credit assigned locally
(PC cascade / error-relaxation), neighbours exchange only hidden-state (down) and error
(up). The north star is a **~124M coherent-Shakespeare LM** trained entirely on multi-F2
via a bubble-free, no-all-reduce pipeline.

**Honest framing:** the *architecture* works and *projects* favorably — distributed PC
training is **measured bit-exact on 8 physical FPGAs** at D=64 (Fashion-MNIST), replication
scales linearly, and the PC-transformer kernels pass real Vitis csynth. It is **NOT** a
"beats GPUs" claim: the 124M figure is a **projection**, no 124M model is trained, and
**no AFI exists yet** for the PC-transformer path. The competitive char-LM rests on
admitting *low-precision* (BF16) multiplies in the backward + optimizer only — the deploy
forward stays multiplier-free.

---

## 2. The stack, top to bottom

| Layer | Where | One-liner |
|---|---|---|
| **Recipe (char-LM)** | [`../../scripts/pc_xformer_charlm.py`](../../scripts/pc_xformer_charlm.py), [`../../scripts/pc_xformer_ablate.py`](../../scripts/pc_xformer_ablate.py) | Mult-free bulk (dyadic ±2^k weights, fine-dyadic LN, int16 acts, dyadic-Adam) + warm dot attention (the one allowed multiply); `MULTFREE=0` = float NanoGPT control. |
| **Library** | [`../../tikitorch/nn/predictive_coding.py`](../../tikitorch/nn/predictive_coding.py), [`../../tikitorch/fabric/`](../../tikitorch/fabric/), [`../../tikitorch/optim/`](../../tikitorch/optim/) | PC layers (`TropicalPCNet` / `DyadicPCNet` / `SparseCodingPC`); transport-agnostic no-sync runtime (`fabric`); mult-free optimizers. |
| **HLS kernels** | [`../../tikitorch/hardware/hls/forge_kernels/`](../../tikitorch/hardware/hls/forge_kernels/) | INT block + **BF16 mixed-precision block** (one transformer block = fwd + PC-cascade/STE bwd + dyadic-Adam) and their int64/bf16 goldens. |
| **CL / HDK** | [`../../tikitorch/hardware/hdk/cl_forge_pc_xformer/`](../../tikitorch/hardware/hdk/cl_forge_pc_xformer/) | DDR CL wrapping the BF16 kernel; adds clk_wiz 150MHz CDC + axi_clock_converters (kernel missed the shell's locked 250MHz). |
| **Silicon drivers + build runners** | [`../../tikitorch/fpga/silicon/forge_pc_xformer_*_shard.cpp`](../../tikitorch/fpga/silicon/), [`../../scripts/run_*_f2.sh`](../../scripts/) | Model-parallel host drivers (no global gradient) + EC2 csynth / AFI-build / shard-run orchestration. |
| **Results / docs** | [`multi_fpga/`](./multi_fpga/) | Flagship scaling writeup, raw silicon logs, csynth reports, the cascade-backward de-risking trail. |

---

## 3. File & directory map

### Recipe & ablation (scripts/) — CPU/torch float-proxy, NOT the kernel
| Path | What | Status |
|---|---|---|
| [`../../scripts/pc_xformer_charlm.py`](../../scripts/pc_xformer_charlm.py) | The char-LM recipe trainer; `STALE_DELAY`/`STALE_PERLAYER` simulate the no-sync bubble | WORKS-cpu |
| [`../../scripts/pc_xformer_ablate.py`](../../scripts/pc_xformer_ablate.py) | Per-axis bpb ablation (ATTN/LN/ACT/OPT/WQUANT/softmax/…) | WORKS-cpu |
| [`../../scripts/pc_pipeline_throughput.py`](../../scripts/pc_pipeline_throughput.py) | 124M throughput **MODEL** — self-labels MEASURED anchors vs PROJECTED timing | DOC |
| [`../../scripts/run_charlm_gpu.sh`](../../scripts/run_charlm_gpu.sh) | GPU runner to scale the recipe (D=384/350M-class) | SCRIPT |
| [`../../scripts/make_mp_shakespeare_bins.py`](../../scripts/make_mp_shakespeare_bins.py) | Builds golden-vocab Shakespeare bins + itos | SCRIPT |

### Library (tikitorch/) — `fabric/` is the only PC code physically on disk; the rest is in git HEAD only (see caveat below)
| Path | What | Status |
|---|---|---|
| [`../../tikitorch/nn/predictive_coding.py`](../../tikitorch/nn/predictive_coding.py) | `TropicalPCLayer/Net` (iterative settle), `DyadicPCNet` (cascade/error-relax/settle-nudge + mult-free dyadic bwd), `SparseCodingPC` (generative) | WORKS-cpu† |
| [`../../tikitorch/optim/__init__.py`](../../tikitorch/optim/__init__.py) | Mult-free optimizers: SignSGD, TropicalSGD/Lion/Adam, FrexpMagnitudeSGD, ExponentEMAAdam, … | WORKS-cpu† |
| [`../../tikitorch/models/__init__.py`](../../tikitorch/models/__init__.py) | Tropical architectures (MLP/SSM/Transformer/CNN) — general, not PC-specific | WORKS-cpu† |
| [`../../tikitorch/backend/`](../../tikitorch/backend/) | FPGA-first device/backend registry seam — general infra | WORKS-cpu† |
| [`../../tikitorch/fabric/__init__.py`](../../tikitorch/fabric/__init__.py) | Package entry: `NeighborLink, ThreadLink, PipelineRunner, run_serial` (numpy-only) | WORKS-cpu |
| [`../../tikitorch/fabric/link.py`](../../tikitorch/fabric/link.py) | Abstract 4-op nearest-neighbour boundary + `LinkStats` | WORKS-cpu |
| [`../../tikitorch/fabric/thread_link.py`](../../tikitorch/fabric/thread_link.py) | ThreadLink backend (threads + blocking queues) — CI/CPU sim | WORKS-cpu |
| [`../../tikitorch/fabric/runner.py`](../../tikitorch/fabric/runner.py) | `PipelineRunner`: windowed local-update loop, overlap + staleness + byte instrumentation; `run_serial` = W=M baseline | WORKS-cpu |
| [`../../tikitorch/fabric/stages.py`](../../tikitorch/fabric/stages.py) | `LinearStage` whose local backward **is** the PC cascade | WORKS-cpu |
| [`../../tikitorch/fabric/README.md`](../../tikitorch/fabric/README.md) | Fabric design doc + backend roadmap + CPU-measured numbers | DOC |

> †**Worktree caveat:** the paths above are the **committed** (git HEAD) tree; a fresh checkout of the
> branch has them all. *This particular worktree* currently has the importable Python library — `tikitorch/__init__.py`
> plus `nn/`, `optim/`, `models/`, `backend/` (≈536 files) — absent as **uncommitted parallel WIP**, so
> `import tikitorch.nn.predictive_coding` fails here until you run **`git checkout -- tikitorch/`** (the broad
> restore; a narrower checkout leaves `tikitorch/__init__.py` missing and the package unimportable). The
> `tikitorch/fabric/` module is present and tests 4/4 regardless (it imports as a PEP-420 namespace package).
> `forward_forward` / `pc_*_probe` files in the session git header are separate parallel WIP, not part of this stack.

### HLS kernels (tikitorch/hardware/hls/forge_kernels/) — the `forge_pc_xformer_*` family
| Path | What | Status |
|---|---|---|
| [`../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_block.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_block.cpp) | **INT** block: fwd + hand-rolled PC-cascade bwd + int dyadic-Adam (mult-free except attn soft-blend) | VALIDATED-csynth-csim |
| [`../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_block.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_block.cpp) | **BF16 mixed-precision** block: int mult-free deploy fwd + textbook backprop+STE in bf16 on a float master | VALIDATED-csynth-csim |
| [`…/forge_pc_xformer_config.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_config.hpp) / [`…_mp_config.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_config.hpp) | Dims D=64/H=2/FFN=256/SEQ=64/VOCAB=64 + DDR layout (mp appends float-master/Adam region) | DOC |
| [`…_golden.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_golden.hpp) / [`…_golden_block.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_golden_block.hpp) | Plain-int64 reference arithmetic / full single-block fwd+bwd+Adam golden (THE int csim golden) | WORKS-cpu |
| [`…_bf16_golden_block.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_bf16_golden_block.hpp) | Clean BF16/wide STE backward golden (the de-risked `BF16_BWD=2` the mp kernel reproduces) | WORKS-cpu |
| [`…_mp_golden_block.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_block.hpp) / [`…_mb_golden_block.hpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mb_golden_block.hpp) | Multi-precision (int/fix16/fix24/dbl) and multi-block (L-stack, `__int128` attn-bwd) golden forks | WORKS-cpu |
| [`…_block_tb.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_block_tb.cpp) / [`…_mp_block_tb.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_block_tb.cpp) / [`…_mp_gradcheck_tb.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_gradcheck_tb.cpp) | CSIM testbenches: bit-exact vs golden (acts + post-update weights); gradcheck isolates raw bf16 grads (max\|diff\|=0) | VALIDATED-csynth-csim |
| [`…_golden_test.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_golden_test.cpp), [`…_bf16_golden_test.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_bf16_golden_test.cpp), [`…_mp_golden_test.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mp_golden_test.cpp), [`…_mb_golden_test.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_mb_golden_test.cpp) | Golden training proofs (loss falls / hits ~3.1 bpb bar / per-MP_OPT BPB sweep) | WORKS-cpu |
| [`…_sim.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_sim.cpp) | Local multi-chip sim of the distributed transformer via the **real ap_int kernel** (bpb falls before any F2 run) | WORKS-cpu |
| [`torch_check.py`](../../tikitorch/hardware/hls/forge_kernels/torch_check.py), [`attn_unit.cpp`](../../tikitorch/hardware/hls/forge_kernels/attn_unit.cpp), [`attn_check.py`](../../tikitorch/hardware/hls/forge_kernels/attn_check.py), [`forward_hash.cpp`](../../tikitorch/hardware/hls/forge_kernels/forward_hash.cpp), [`bf16_dump.cpp`](../../tikitorch/hardware/hls/forge_kernels/forge_pc_xformer_bf16_dump.cpp) | Cross-checks: BF16 STE vs torch autograd; isolated attn backward; forward-identity guard (bf16 must not perturb deploy fwd) | WORKS-cpu |
| [`synth_pc_xformer_block.tcl`](../../tikitorch/hardware/hls/forge_kernels/synth_pc_xformer_block.tcl), [`synth_pc_xformer_mp_block.tcl`](../../tikitorch/hardware/hls/forge_kernels/synth_pc_xformer_mp_block.tcl) | Vitis HLS csim+csynth gates (DSP_MULT=0 outside attn, timing, 1-SLR) | SCRIPT |
| [`run_local_csim.sh`](../../tikitorch/hardware/hls/forge_kernels/run_local_csim.sh), [`run_local_csim_mp.sh`](../../tikitorch/hardware/hls/forge_kernels/run_local_csim_mp.sh) | Local authentic-ap_int csim (no Vitis/EC2) | SCRIPT |

### CL / HDK (tikitorch/hardware/hdk/cl_forge_pc_xformer/) — DDR scaffold, AFI build in flight
| Path | What | Status |
|---|---|---|
| [`…/design/cl_forge_pc_xformer.sv`](../../tikitorch/hardware/hdk/cl_forge_pc_xformer/design/cl_forge_pc_xformer.sv) | DDR CL top: BF16 kernel on 150MHz clk_kernel + clk_wiz CDC + axi_clock_converters; 2→1 crossbar onto sh_ddr; reuses cl_forge_muzero_ddr wiring | VALIDATED-csynth-csim |
| [`…/build/scripts/gen_ip.tcl`](../../tikitorch/hardware/hdk/cl_forge_pc_xformer/build/scripts/gen_ip.tcl), [`…/synth_cl_forge_pc_xformer.tcl`](../../tikitorch/hardware/hdk/cl_forge_pc_xformer/build/scripts/synth_cl_forge_pc_xformer.tcl) | CL IP gen (reg-slice, crossbar, clk_wiz, axi_cc) + OOC synth driver (sources staged HLS FP `*_ip.tcl`) | SCRIPT |
| [`…/design/*.vh`](../../tikitorch/hardware/hdk/cl_forge_pc_xformer/design/), [`…/build/constraints/*.xdc`](../../tikitorch/hardware/hdk/cl_forge_pc_xformer/build/constraints/) | CL defines, PCIe IDs, synth/timing user constraints (clk_kernel CDC false-paths) | SCRIPT |

### Silicon drivers + F2 runners (tikitorch/fpga/silicon/, scripts/)
| Path | What | Status |
|---|---|---|
| [`../../tikitorch/fpga/silicon/forge_pc_xformer_shard.cpp`](../../tikitorch/fpga/silicon/forge_pc_xformer_shard.cpp) | INT model-parallel driver: one block/chip across L F2 chips, neighbour hidden+error only | WORKS-cpu |
| [`../../tikitorch/fpga/silicon/forge_pc_xformer_mp_shard.cpp`](../../tikitorch/fpga/silicon/forge_pc_xformer_mp_shard.cpp) | BF16 mp driver: float master + bf16 dyadic-Adam per chip, host relays only wide neighbour msgs | WORKS-cpu |
| [`../../scripts/run_mp_csynth_f2.sh`](../../scripts/run_mp_csynth_f2.sh) | z1d Vitis csynth gate — **HAS RUN**, produced [`mp_csynth/mp_csynth.rpt`](./multi_fpga/mp_csynth/mp_csynth.rpt) | SCRIPT |
| [`../../scripts/run_mp_afi_build_f2.sh`](../../scripts/run_mp_afi_build_f2.sh) | HLS→CL→DCP→AGFI build for the BF16 kernel — **launched Jun 29, no AGFI yet** | SCRIPT |
| [`../../scripts/run_mp_xformer_shard_f2.sh`](../../scripts/run_mp_xformer_shard_f2.sh), [`../../scripts/run_pc_xformer_shard_f2.sh`](../../scripts/run_pc_xformer_shard_f2.sh) | Load AGFI on f2.48xlarge slots + run the shard driver — **need an AGFI, never run** | SCRIPT |

### Results / docs (results/predictive_coding/multi_fpga/)
See section 7 for the full deep-dive list.

---

## 4. Status: what's MEASURED vs PROJECTED

**MEASURED on silicon (real F2):**
- Distributed PC training **bit-exact** to the Python reference, **D=64**, **8 physical FPGAs**
  (f2.48xlarge), Fashion-MNIST ~0.73 depth-flat, no global sync — W max\|diff\|=0 every chip,
  all 7500 steps. → [`multi_fpga/MULTI_FPGA_SCALING.md`](./multi_fpga/MULTI_FPGA_SCALING.md),
  [`train_L8_rerun.txt`](./multi_fpga/train_L8_rerun.txt), [`bitexact_replay_clean.txt`](./multi_fpga/bitexact_replay_clean.txt).
- Linear **×4 kernel replication** and the no-sync no-all-reduce contract (from prior PC-MLP silicon work).

**MEASURED on CPU (real threads, not silicon):**
- Fabric runtime: **5.60× overlap** (8-stage), staleness-vs-window W, **W=M bit-exact to backprop**
  (tests 4/4). → [`../../tikitorch/fabric/README.md`](../../tikitorch/fabric/README.md).

**VALIDATED at csynth/csim (Vitis HLS, no AFI):**
- INT PC-transformer block: timing 3.192ns<4.0, DSP=185 attention-only (DSP_MULT=0 on all dyadic ops), 1 SLR.
- BF16 mp block: csynth timing 3.392ns<4.0 target (Fmax ≈294 MHz, derived 1000/3.392), DSP=743 bf16 bwd-only, LUT ~345k / 1 SLR. NOTE: this is the csynth *estimate*; real post-route P&R does NOT yet close (the timing-closure work in progress).
  → [`multi_fpga/mp_csynth/mp_csynth.rpt`](./multi_fpga/mp_csynth/mp_csynth.rpt).

**IN-FLIGHT / NOT YET (do not assume these exist):**
- **No AFI for the PC-transformer.** The D=64 BF16 AFI build was *launched* (Jun 29) but produced
  no AGFI; CL timing closure (kernel missed 250MHz, CDC added) is **in progress**.
- **Neither shard driver has executed on F2** (no run-state dirs).
- **Everything at d768 / 124M is PROJECTED** — [`pc_pipeline_throughput.py`](../../scripts/pc_pipeline_throughput.py)
  is a model; the 124M timing is not silicon.
- **Silicon fabric backends** (HostRelayLink/PcieP2PLink/TcpLink/FabricLink) are **unbuilt** — only ThreadLink exists.

---

## 5. How to run the common things

> **User runs F2 builds.** The `run_*_f2.sh` scripts launch real EC2/F2 instances and cost money/time.

**Fabric demo + tests (CPU, free, fast):**
```
python scripts/pc_fabric_sim_demo.py        # overlap speedup + staleness-vs-W
pytest tests/test_fabric.py                 # W=M bit-exact gate (4/4)
```

**Char-LM recipe + ablation (CPU/torch, float-proxy):**
```
python scripts/pc_xformer_charlm.py         # MULTFREE=0 for float NanoGPT control
python scripts/pc_xformer_ablate.py         # per-axis bpb cost
```
> First restore the library if importing PC layers in this worktree: `git checkout -- tikitorch/` (broad — see §3 caveat).

**Local kernel csim (authentic ap_int, no Vitis):**
```
tikitorch/hardware/hls/forge_kernels/run_local_csim.sh      # INT block vs int64 golden
tikitorch/hardware/hls/forge_kernels/run_local_csim_mp.sh   # BF16 mp block vs bf16 golden
```

**F2 runners (names only — user-invoked):**
`run_mp_csynth_f2.sh` (csynth gate, has run) · `run_mp_afi_build_f2.sh` (AFI build, in flight) ·
`run_mp_xformer_shard_f2.sh` / `run_pc_xformer_shard_f2.sh` (distributed shard run, need AGFI) ·
`run_charlm_gpu.sh` (GPU scale-up).

---

## 6. Roadmap / next

> A consolidated `ROADMAP.md` does **not** currently exist under this tree — the plan of record lives in
> [`multi_fpga/BUILD_PLAN.md`](./multi_fpga/BUILD_PLAN.md) and [`multi_fpga/KERNEL_DESIGN_SPEC.md`](./multi_fpga/KERNEL_DESIGN_SPEC.md).

Immediate next steps:
1. **Close the D=64 BF16 AFI** — finish CL timing closure (clk_wiz 150MHz CDC) and confirm the AGFI from `run_mp_afi_build_f2.sh`.
2. **First on-silicon PC-transformer run** — once the AGFI exists, run `run_mp_xformer_shard_f2.sh` (BF16) / `run_pc_xformer_shard_f2.sh` (int) on f2.48xlarge; confirm bpb falls + bit-exact.
3. **Build `HostRelayLink`** on the existing AGFI so the `tikitorch.fabric` runtime drives real FPGAs (today only ThreadLink).
4. **Bound staleness** via per-chip grad accumulation ([`STALENESS_MEASURED.md`](./multi_fpga/STALENESS_MEASURED.md)) and then **scale** toward d768/124M (replication + DSP-adder array per [`pc_pipeline_throughput.py`](../../scripts/pc_pipeline_throughput.py)).

---

## 7. Deep-dive docs

**Flagship / measured silicon:**
- [`multi_fpga/MULTI_FPGA_SCALING.md`](./multi_fpga/MULTI_FPGA_SCALING.md) — 2→8 FPGA scaling; deep cascades wash out → host-side residual skip fixes it; Fashion-MNIST depth-flat, bit-exact, no global sync.
- [`multi_fpga/train_L8_rerun.txt`](./multi_fpga/train_L8_rerun.txt) — raw 8-FPGA run log + signature throughput/comms/Gate-C/energy breakdown.
- [`multi_fpga/bitexact_replay_clean.txt`](./multi_fpga/bitexact_replay_clean.txt) — W max\|diff\|=0 vs Python ref, every chip, all 7500 steps (L=4, L=8).

**Kernel csynth (HLS-validated, no AFI):**
- [`multi_fpga/forge_pc_xformer_block_csynth.rpt`](./multi_fpga/forge_pc_xformer_block_csynth.rpt) — INT block, 3.192ns, DSP=185 attn-only, 1 SLR.
- [`multi_fpga/mp_csynth/mp_csynth.rpt`](./multi_fpga/mp_csynth/mp_csynth.rpt) + [`mp_vitis_hls.log`](./multi_fpga/mp_csynth/mp_vitis_hls.log) — BF16 mp block, 3.392ns, DSP=743 bf16-bwd, 1 SLR.

**Design / build contracts:**
- [`multi_fpga/BUILD_PLAN.md`](./multi_fpga/BUILD_PLAN.md) — locked north-star plan; precision DECISION = admit low-prec multiply (BF16→MXFP8); int recipe caps ~0.20.
- [`multi_fpga/KERNEL_DESIGN_SPEC.md`](./multi_fpga/KERNEL_DESIGN_SPEC.md) — mixed-precision datapath contract (narrow swap on wgrad + optimizer only).

**The backward-precision crack (read this to understand *why* BF16):**
- [`multi_fpga/BF16_BACKPROP_RESULT.md`](./multi_fpga/BF16_BACKPROP_RESULT.md) — **BUILD-GO**: clean BF16/wide STE backward hits the autograd bar, grads bit-identical to torch.
- [`multi_fpga/BF16_KERNEL_RESULT.md`](./multi_fpga/BF16_KERNEL_RESULT.md) — kernel csim-matched to BF16 golden (its "needs-csynth" verdict is **stale** — csynth subsequently ran + passed).
- The de-risking trail (mostly NEGATIVE/PARTIAL — the int hand-rolled cascade caps ~0.15 vs autograd ~0.30):
  [`STE_CASCADE_RESULT.md`](./multi_fpga/STE_CASCADE_RESULT.md), [`COMPLETE_STE_RESULT.md`](./multi_fpga/COMPLETE_STE_RESULT.md),
  [`WIDE_BACKWARD_RESULT.md`](./multi_fpga/WIDE_BACKWARD_RESULT.md), [`CASCADE_BATCHING_ROOTCAUSE.md`](./multi_fpga/CASCADE_BATCHING_ROOTCAUSE.md),
  [`CASCADE_RECIPE_ALIGN_RESULT.md`](./multi_fpga/CASCADE_RECIPE_ALIGN_RESULT.md),
  [`G2_CASCADE_PRECISION_RESULT.md`](./multi_fpga/G2_CASCADE_PRECISION_RESULT.md), [`G2_RETRY_RESULT.md`](./multi_fpga/G2_RETRY_RESULT.md),
  [`G2_DISAMBIGUATION_RESULT.md`](./multi_fpga/G2_DISAMBIGUATION_RESULT.md), [`G2_WARM_CONFIRM_RESULT.md`](./multi_fpga/G2_WARM_CONFIRM_RESULT.md).

**Recipe / pipeline diagnosis:**
- [`multi_fpga/SOFTMAX_RECIPE_DIAGNOSIS.md`](./multi_fpga/SOFTMAX_RECIPE_DIAGNOSIS.md) — weak training is the **integer attention softmax**, not the mult-free substrate (float baseline 0.489).
- [`multi_fpga/STALENESS_MEASURED.md`](./multi_fpga/STALENESS_MEASURED.md) — no-sync pipeline tolerates ~1 step of weight staleness; bound via per-chip grad accumulation.

---

*Honesty contract: distributed PC training is real on silicon at D=64 (MLP/Fashion-MNIST, bit-exact, 8 FPGAs).
The PC **transformer** is HLS-validated only — no AFI exists, no 124M model is trained, and the competitive
char-LM depends on low-precision (BF16) multiplies in the backward + optimizer. The 124M figure is a projection.*
