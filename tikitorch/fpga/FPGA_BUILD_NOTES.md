# TikiTorch FPGA build & silicon notes

Single source of truth for taking `model.fpga()` to real AWS F1/F2 silicon: the
on-board **run** flow (load an AFI, train over PCIe), the **AFI build** flow (HLS →
DCP → AFI), the validated AGFI registry, the timing-failure history + root cause, and
the operational gotchas. Consolidates what was scattered across the source monorepo
build scripts and the project memory.

> Status legend: **[validated]** proven on silicon · **[blocked]** known failure ·
> **[infra]** monorepo-coupled tooling (see "Where the scripts live").

---

## 0. TL;DR

- **On-silicon training works** [validated]: `model.fpga()` binds a tropical model to
  an F2 board (`bound=True`, **dsp=0**) and runs the whole fwd→bwd→sign-SGD step on the
  board. A 2026-06 run evolved **99.5 % of params** (229872/231040) over 50 `train_step()`s.
- **The on-board optimizer is the hardware sign-SGD kernel**, baked into the AFI — you
  cannot swap optimizers (e.g. frexpmag) without building a new AFI.
- **A clean *live loss curve* is blocked** [blocked]: every AFI rebuild fails post-route
  timing. Root cause (below): the monolithic kernel's critical path needs **~14.3 ns
  (~70 MHz)**, below the slowest standard shell clock (125 MHz) → no clock recipe closes
  it. The fix is HLS pipelining of a 512-bit select, not a clock tweak.

---

## 1. The two flows

### A. Run on silicon (`model.fpga()` over PCIe) — [validated]
Runs **on** an f1/f2 instance with the `aws-fpga` SDK. Loads a pre-built AGFI, compiles
the `fpga_pci` native bridge (`tikitorch/fpga/bridge.cpp`, `-DTT_FPGA_PCI`), and drives
the same `AcceleratedModel.train_step()` chain proven in LOCAL mode — now over PCIe.

```
load AGFI (fpga-load-local-image) → build _libtikifpga.so (bridge.cpp, TT_FPGA_PCI)
  → sudo python: m.fpga(strict=True).train_step(x, y, lr=1, threshold=0)
```
Result on success: `BOUND on silicon: True | kernel: forge_train_step | dsp: 0` and the
parameter-evolution count. The Python side is unchanged; only the transport differs.

### B. Build an AFI (HLS → DCP → AFI) — [infra]
Runs **on** a z1d build instance (Xilinx tools). Pipeline:
```
vitis_hls -f synth.tcl  (forge_train_step.cpp → RTL, 512-bit m_axi)
  → hls_to_cl.py  (auto-generate the CL project)
  → aws_build_dcp_from_cl.py  (Vivado synth+place+route, ~2h43m → DCP tarball)
  → aws s3 cp <tarball> s3://...    → aws ec2 create-fpga-image  → AGFI
```
**Pre-route gate:** the generated `m_axi_gmem` port must be **512-bit** wide
(`C_M_AXI_GMEM_DATA_WIDTH=512`); narrower silently masks write strobes (the v1–v5
silicon failure mode) — the build aborts before the slow Vivado step if not.

---

## 1b. MuZero dyadic kernels — HLS-validated on the F2 part [validated, 2026-06-09]

The zero-DSP dyadic MuZero kernels (`forge_train_step_muzero.cpp` forward-inference +
`forge_train_step_muzero_train.cpp` on-chip SignSGD training) **pass Vitis HLS 2024.1
C-synthesis on the real F2 part `xcvu47p-fsvh2892-2L-e` at the 4.0 ns / 250 MHz target:**

| kernel | DSP | est. clock | latency (cycles) | LUT | FF | BRAM |
|---|---|---|---|---|---|---|
| `forge_train_step_muzero` (fwd) | **0** | **2.920 ns** | **bounded: 25,539** (~102 µs) | 109165 (8% dev) | 60552 | 2 |
| `forge_train_step_muzero_train` (fwd+bwd+SignSGD) | **0** | 2.920 ns (clock est. only) | **UNBOUNDED — HLS reports `?`** | 96330 (7% dev) | 39860 | 2 |

The vendor synthesizer's DSP operation table is **empty** for both — every "multiply" mapped to
shift+add LUT logic, and the `/MZ_L` (=/32) normalization mapped to LUTs (not a DSP divide).
**DSP=0 is solid for both.** But read the rest carefully (corrected 2026-06-09 after artifact re-audit):

- **Forward kernel: genuinely measured** — DSP=0, latency *bounded* at 25,539 cycles, clock 2.920 ns.
  This is a real, schedulable, complete initial-inference datapath.
- **Training kernel: NOT a complete/bounded learner yet.** DSP=0 holds and 2.920 ns is reported, but
  that is only the **AXI-wrapper clock estimate** — HLS could not **bound the latency** (every
  cycle/interval field is `?`, because `nearest_pow2`'s leading-bit `while` loop has a data-dependent
  trip count). It is also **initial-inference only** (returns −1 for the dynamics stage; no K-step
  unroll, no reward head). So it is a validated **zero-DSP arithmetic primitive**, not a finished
  on-fabric learner. To make it real: bound `nearest_pow2` to a fixed ≤15-iteration loop, add the
  dynamics unroll + reward head, re-synthesize, and report the *actual* bounded latency.
- **"Low-risk P&R" was an overstatement — corrected.** A clean HLS *C-synthesis* estimate is **not** a
  P&R timing guarantee: per §3, the only MuZero-family kernel ever placed-and-routed (the char-LM)
  **FAILED post-route timing** (WNS −1.557 ns) despite plausible HLS numbers. The HLS estimate is
  encouraging, but **P&R timing closure is a genuine open risk** until an AFI is actually built.

Reports: `results/fpga_silicon/forge_train_step_muzero{,_train}_csynth.rpt`.

GOTCHA found here: Vitis HLS rejects ambiguous `?:` that g++ accepts — `cond ? x : -x` where `-x`
widens an `ap_int<32>` to `ap_int<33>`, and `relu`'s `data_t vs int 0`. Fix is an explicit cast on
the negative/zero branch (`(acc_t)(-v)`, `(data_t)0`); semantics-preserving under the g++ CSIM shim.
Synthesize early — the CSIM (g++) build does NOT catch these.

STILL TODO: the full HLS→DCP→AFI build (flow B, ~3 h P&R + ~1–2 h AFI gen on z1d) then the
f2.6xlarge on-board run. Build infra (`aws_build_dcp_from_cl.py` etc.) is monorepo-coupled — adapt
`KERNEL_DIR`/top to `forge_train_step_muzero{,_train}` and reuse this part/clock.

---

## 2. Validated AGFI registry (account 739966866637, us-east-1)

| AGFI | name | notes |
|---|---|---|
| `agfi-0a845113b088cf62e` | `cl_forge_train_step_v7_monolithic` | **the default** for `model.fpga()` char-LM; bound, dsp=0, 99.5 % params evolved |
| `agfi-0507f7d451ea7c338` | `cl_forge_train_step_tier1_d128_v128` | tier-1 D=128 V=128 |
| `agfi-03098a84aba9aba04` | `cl_forge_train_step_2layer` | 2-layer LM |
| `agfi-042a4d0cf6f63b3fd` | `cl_forge_train_step_3layer` | 3-layer LM |
| `agfi-094acffc290c6c1dd` | `cl_forge_morph_ecg` | MorphSSM-ECG (separate track) |

List them: `aws ec2 describe-fpga-images --profile <p> --region us-east-1 --query 'FpgaImages[].{agfi:FpgaImageGlobalId,name:Name,state:State.Code}'`.

---

## 3. Timing-failure history + ROOT CAUSE  — [blocked]

The on-silicon **parameter-evolution proof is exact**, but a **live loss curve** is not
yet achievable, because the loss readout reads `grad_logit` at the bundled
`forge_buffer.hpp` offset, and the currently-registered AFIs are layout-rebuilds whose
`grad_logit` offset differs from the originally-validated (now-deregistered) AGFI — so
the accuracy proxy pins at a constant. The fix is to **rebuild a layout-matched AFI**…
but every rebuild has failed post-route timing:

| attempt | target | result |
|---|---|---|
| original (deregistered) | — | validated, but its layout no longer matches the bundled buffer |
| rebuild ×2 (prior) | 4.0 ns / 250 MHz | post-route **timing failed** |
| rebuild (2026-06-08) | 4.0 ns / 250 MHz | **WNS −1.557 ns** ("functionalities NOT guaranteed") |
| slower-clock retry (2026-06-08) | 8.0 ns / 125 MHz + recipe A0 | **HLS est. 14.345 ns (~70 MHz)** |

**ROOT CAUSE (the actionable finding):** at an 8 ns HLS target, `vitis_hls` estimates the
monolithic kernel needs **14.345 ns (~70 MHz)**, gated by a **512-bit `select` operation**
(`w_new`, `forge_train_step.cpp:552`). That is *below the slowest standard shell-clock
floor* (clk_main_a0 = 125 MHz / 8 ns), so **no clock recipe can ever close it** — the
slower-clock lever is exhausted. Closing timing requires **HLS pipelining / registering the
512-bit critical path** (a kernel change), not a build-flag change.

Fix candidates (next work): (a) `#pragma HLS PIPELINE`/register the wide `w_new` select to
break the 14.3 ns path; (b) split the 512-bit op into staged narrower ops; (c) retarget the
kernel to a slower auxiliary clock (clk_extra_*). Each needs a rebuild (~3 h) + re-verify.

---

## 4. Operational gotchas (each cost real time at least once)

- **F2 AMI is Python 3.8 with NO pip.** `ami-092fc5deb8f3c0f7d` → `sudo apt-get install -y
  python3-pip`, then install deps **system-wide under sudo** (the bridge runs under sudo for
  PCI BAR access, so `--user` installs are invisible to root):
  `sudo python3 -m pip install "numpy==1.24.4" "typing-extensions==4.12.2"` and
  `sudo python3 -m pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu`.
- **PYTHONPATH must point at the package root.** `python3 scripts/x.py` puts `scripts/` on
  `sys.path`, not the repo root → `import tikitorch` fails. Export `PYTHONPATH=<repo root>`.
- **PYTHONUNBUFFERED=1** to see epoch/step logs through `tee` (Python block-buffers when piped).
- **`--clock_recipe_a` requires `--aws_clk_gen`** in `aws_build_dcp_from_cl.py`, or the build
  aborts immediately.
- **512-bit AXI gate** (§1.B): verify `C_M_AXI_GMEM_DATA_WIDTH=512` before the ~3 h Vivado step.
- **HLS target clock** lives in `synth.tcl`: `set target_period_ns 4.0` (env-overridable pattern:
  `[expr {[info exists ::env(TARGET_PERIOD_NS)] ? $::env(TARGET_PERIOD_NS) : 4.0}]`).

---

## 5. Instances / costs

| role | type | AMI | ~cost | notes |
|---|---|---|---|---|
| run on silicon | `f2.6xlarge` | `ami-092fc5deb8f3c0f7d` | ~$13.65/hr | aws-fpga SDK; terminate after the run |
| AFI build | `z1d.6xlarge` | `ami-092fc5deb8f3c0f7d` | ~$1.1/hr | Vivado P&R ~2h43m + AWS AFI gen ~1–2 h |
| GPU benchmarks | `g5.4xlarge` (A10G) | `ami-05603a42e5254c4bb` | ~$1.6/hr | image/seq closed benchmarks |

Profile `NYUMCResearchAdminFullAccess-739966866637`, key `mnemo-dev`, SG `sg-0efe5c4b6eeb6fda6`,
region `us-east-1`. **Always terminate after** (these are billable).

---

## 6. Where the scripts live

In this repo (`scripts/`), standalone:
- `run_fpga_f2_verify.sh` — the on-silicon `model.fpga()` verify (flow A).
- `run_tikitorch_f2_silicon.sh` — provision an f2.6xlarge + run the verify (launch/monitor/download/terminate).
- `run_tikitorch_gpu_benchmarks.sh` — provision a g5.4xlarge + run the closed image/seq benchmarks.

Still in the source monorepo (kernel-build infra, forge-coupled — flow B): `run_hdk_build_train_step.sh`,
`run_hdk_build_remote.sh` (HLS→CL→DCP→S3), `create_afi.sh` (DCP→AGFI). The HLS kernel source is
vendored here at `tikitorch/hardware/hls/forge_kernels/` (identical to the monorepo's `forge/`
copy), so a standalone build flow is a clean follow-up — the only adaptation is `KERNEL_DIR` and
the rsync source path.
