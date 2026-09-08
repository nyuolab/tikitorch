# TikiTorch FPGA build & silicon notes

Technical notes for the historical `model.fpga()` AWS F2 experiments: the
on-board **run** flow (load a compatible AFI, train over PCIe), the **AFI build**
flow (HLS → DCP → AFI), the account-specific AGFI inventory, the recorded
timing-failure history, and operational gotchas. These notes consolidate
project records; they are not a promise of public AFI availability or generic
board support.

> Status legend: **[validated]** observed for the named historical silicon
> configuration · **[blocked]** known failure · **[infra]** monorepo-coupled
> tooling (see "Where the scripts live").

---

## 0. TL;DR

- **A configuration-specific on-silicon path ran** [validated]: a 2026-06 F2
  run used the FPGA-PCI bridge, the one-layer runner, and a compatible
  account-specific AFI to execute the whole fwd→bwd→sign-SGD step. The logged
  before/after observation found **99.5% of parameters changed**
  (229872/231040) over 50 `train_step()` calls. The associated build record
  reports DSP=0. `bound=True` or the `KernelSpec.dsp` field alone is not board
  evidence: LOCAL host-reference mode can also bind, and the DSP field is
  metadata rather than a live synthesis probe.
- **The on-board optimizer is the hardware sign-SGD kernel**, baked into the AFI — you
  cannot swap optimizers (e.g. frexpmag) without building a new AFI.
- **A clean *live loss curve* is blocked** [blocked]: the AFI rebuilds recorded
  below failed post-route timing. Root cause (below): the monolithic kernel's
  critical path needs **~14.3 ns (~70 MHz)**, below the slowest standard shell
  clock (125 MHz) → no standard shell-clock recipe closes this unpipelined
  build. The fix is HLS pipelining of a 512-bit select, not a clock tweak.

---

## 1. The two flows

### A. Run on silicon (`model.fpga()` over PCIe) — [validated configuration]
Runs **on** an F2 instance with the `aws-fpga` SDK. It loads a compatible AGFI,
compiles the `fpga_pci` native bridge (`tikitorch/fpga/bridge.cpp`,
`-DTT_FPGA_PCI`), and drives the same `AcceleratedModel.train_step()` call
sequence exercised in LOCAL host-reference mode—now through PCIe to the named
image. LOCAL does not validate the synthesized path.

```
load AGFI (fpga-load-local-image) → build _libtikifpga.so (bridge.cpp, TT_FPGA_PCI)
  → sudo python: m.fpga(strict=True).train_step(x, y, lr=1, threshold=0)
```
The historical log prints `BOUND on silicon: True | kernel: forge_train_step |
dsp: 0` plus the parameter-change count. Here, board execution is established
by the loaded FPGA-PCI transport and run context, not by `bound=True`; `dsp: 0`
echoes kernel metadata and must be paired with the archived build report.

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

## 1b. MuZero dyadic kernels — HLS C-synthesis only [2026-06-09]

The dyadic MuZero kernels (`forge_train_step_muzero.cpp` forward inference and
`forge_train_step_muzero_train.cpp` SignSGD path) have Vitis HLS 2024.1
C-synthesis reports targeting the F2 device part `xcvu47p-fsvh2892-2L-e` at
4.0 ns / 250 MHz. They have not completed DCP/AFI build or silicon validation:

| kernel | DSP | est. clock | latency (cycles) | LUT | FF | BRAM |
|---|---|---|---|---|---|---|
| `forge_train_step_muzero` (fwd) | **0** | **2.920 ns** | **bounded: 25,539** (~102 µs) | 109165 (8% dev) | 60552 | 2 |
| `forge_train_step_muzero_train` (fwd+bwd+SignSGD) | **0** | 2.920 ns (clock est. only) | **UNBOUNDED — HLS reports `?`** | 96330 (7% dev) | 39860 | 2 |

The HLS reports' DSP operation tables are **empty** for both: operations map to
shift/add LUT logic, and the `/MZ_L` (=/32) normalization maps to LUTs rather
than a DSP divide. Thus DSP=0 is an HLS-reported property of these named builds,
not a routed or on-board result.

- **Forward kernel:** HLS scheduled a bounded 25,539-cycle latency and reported
  DSP=0 with a 2.920 ns clock estimate. This is a C-synthesis result for the
  initial-inference datapath, not measured silicon execution.
- **Training kernel: NOT a complete/bounded learner yet.** HLS reports DSP=0 and 2.920 ns, but
  that is only the **AXI-wrapper clock estimate** — HLS could not **bound the latency** (every
  cycle/interval field is `?`, because `nearest_pow2`'s leading-bit `while` loop has a data-dependent
  trip count). It is also **initial-inference only** (returns −1 for the dynamics stage; no K-step
  unroll, no reward head). So it is an HLS-synthesized **zero-DSP arithmetic
  primitive**, not a finished on-fabric learner. To make it real: bound
  `nearest_pow2` to a fixed ≤15-iteration loop, add the
  dynamics unroll + reward head, re-synthesize, and report the *actual* bounded latency.
- **"Low-risk P&R" was an overstatement — corrected.** A clean HLS *C-synthesis* estimate is **not** a
  P&R timing guarantee: per §3, a related monolithic char-LM build **FAILED
  post-route timing** (WNS −1.557 ns) despite plausible HLS numbers. The HLS estimate is
  encouraging, but **P&R timing closure is a genuine open risk** until an AFI is actually built.

Reports: `manuscript/results/fpga_silicon/forge_train_step_muzero{,_train}_csynth.rpt`.

GOTCHA found here: Vitis HLS rejects ambiguous `?:` that g++ accepts — `cond ? x : -x` where `-x`
widens an `ap_int<32>` to `ap_int<33>`, and `relu`'s `data_t vs int 0`. Fix is an explicit cast on
the negative/zero branch (`(acc_t)(-v)`, `(data_t)0`); semantics-preserving under the g++ CSIM shim.
Synthesize early — the CSIM (g++) build does NOT catch these.

STILL TODO: the full HLS→DCP→AFI build and an F2 on-board run. The historical
build infrastructure is monorepo-coupled; porting it to these kernel tops needs
its own validation rather than assuming a parameter-only adaptation.

---

## 2. Historical AGFI inventory (account 739966866637, us-east-1)

These identifiers record the private account/region used for the experiments.
They are not distributed with TikiTorch, may be deregistered or inaccessible to
other accounts, and are not release-supported defaults. Reproduction requires
building or obtaining an authorized, layout-compatible image.

| AGFI | name | notes |
|---|---|---|
| `agfi-0a845113b088cf62e` | `cl_forge_train_step_v7_monolithic` | historical one-layer script default; FPGA-PCI run observed 99.5% of parameters change; associated record reports DSP=0 |
| `agfi-0507f7d451ea7c338` | `cl_forge_train_step_tier1_d128_v128` | tier-1 D=128 V=128 |
| `agfi-03098a84aba9aba04` | `cl_forge_train_step_2layer` | 2-layer LM |
| `agfi-042a4d0cf6f63b3fd` | `cl_forge_train_step_3layer` | 3-layer LM |
| `agfi-094acffc290c6c1dd` | `cl_forge_morph_ecg` | MorphSSM-ECG (separate track) |

List them: `aws ec2 describe-fpga-images --profile <p> --region us-east-1 --query 'FpgaImages[].{agfi:FpgaImageGlobalId,name:Name,state:State.Code}'`.

---

## 3. Timing-failure history + ROOT CAUSE  — [blocked]

The archived on-silicon observable is a before/after parameter-change count
(229872/231040), not a bit-exact internal trajectory or learning-quality result.
A **live loss curve** is not yet achievable, because the loss readout reads
`grad_logit` at the bundled
`forge_buffer.hpp` offset, and the currently-registered AFIs are layout-rebuilds whose
`grad_logit` offset differs from the originally-validated (now-deregistered) AGFI — so
the accuracy proxy pins at a constant. The fix is to **rebuild a layout-matched AFI**…
but each rebuild recorded in this history failed post-route timing:

| attempt | target | result |
|---|---|---|
| original (deregistered) | — | validated, but its layout no longer matches the bundled buffer |
| rebuild ×2 (prior) | 4.0 ns / 250 MHz | post-route **timing failed** |
| rebuild (2026-06-08) | 4.0 ns / 250 MHz | **WNS −1.557 ns** ("functionalities NOT guaranteed") |
| slower-clock retry (2026-06-08) | 8.0 ns / 125 MHz + recipe A0 | **HLS est. 14.345 ns (~70 MHz)** |

**ROOT CAUSE (the actionable finding):** at an 8 ns HLS target, `vitis_hls` estimates the
monolithic kernel needs **14.345 ns (~70 MHz)**, gated by a **512-bit `select` operation**
(`w_new`, `forge_train_step.cpp:552`). That is *below the slowest standard shell-clock
floor* (clk_main_a0 = 125 MHz / 8 ns), so **no standard shell-clock recipe can
close this unpipelined build**—the slower-clock lever is exhausted. Closing timing requires **HLS pipelining / registering the
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

These are dated June 2026 environment notes and rough historical prices, not
current quotes. Check current AWS availability and pricing before use.

| role | type | AMI | ~cost | notes |
|---|---|---|---|---|
| run on silicon | `f2.6xlarge` | `ami-092fc5deb8f3c0f7d` | ~$13.65/hr | aws-fpga SDK; terminate after the run |
| AFI build | `z1d.6xlarge` | `ami-092fc5deb8f3c0f7d` | ~$1.1/hr | Vivado P&R ~2h43m + AWS AFI gen ~1–2 h |
| GPU benchmarks | `g5.4xlarge` (A10G) | `ami-05603a42e5254c4bb` | ~$1.6/hr | image/seq closed benchmarks |

Use a profile, key pair, security group, region, and FPGA image authorized in
your own account. **Always terminate after** (these instances are billable).

---

## 6. Where the scripts live

In this repo (`scripts/`), standalone:
- `run_fpga_f2_verify.sh` — the on-silicon `model.fpga()` verify (flow A).
- `run_tikitorch_f2_silicon.sh` — provision an f2.6xlarge + run the verify (launch/monitor/download/terminate).
- `run_tikitorch_gpu_benchmarks.sh` — provision a g5.4xlarge + run the closed image/seq benchmarks.

The historical flow-B orchestration remains in the source monorepo:
`run_hdk_build_train_step.sh`, `run_hdk_build_remote.sh` (HLS→CL→DCP→S3), and
`create_afi.sh` (DCP→AGFI). Kernel sources are vendored here under
`tikitorch/hardware/hls/forge_kernels/`, but a standalone AFI build from this
release has not yet been demonstrated. Port and verify the orchestration,
constraints, CL package, paths, and target-specific checks before claiming a
standalone build.
