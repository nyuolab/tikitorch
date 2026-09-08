# S1 Transformer Trainer — AFI Build + Run Plan

Goal: build an Amazon FPGA Image (AFI) for the S1 transformer-block trainer kernel
`forge_train_step_s1` and RUN it on a real AWS F2, so a multiplier-free transformer block
trains end-to-end on FPGA silicon (forward + backward + dyadic-Adam, no GPU).

This plan was written against the REAL files (paths + line refs below). It corrects two
load-bearing errors in the upstream MAP results:

1. **Modes 1/2/3 are NOT stubbed.** The actual kernel tail of
   `tikitorch/hardware/hls/forge_kernels/forge_train_step_s1.cpp` implements `do_bwd` (mode 1),
   `do_opt` (mode 2, dyadic-Adam + re-quantize), and `do_train_run` (mode 3, the **one-kick
   fused trainer**: `do_opt(INIT)` then an `n_steps` loop of fwd→bwd→opt). Mode 3 with `a1=n_steps`
   is exactly the "kick mode-3 N steps" the goal asks for. This is the whole reason the run is
   feasible without a per-step host round-trip (see RISKS).
2. **The S1 control map is 13-bit, like muzero_ddr, NOT 5-bit like pc_train_ddr.** The S1
   `s_axilite control` bundle carries `buffer` offset (0x18/0x1c) + `mode` (0x24) + `a1/a2/a3`,
   so the OCL address slice must be `[12:0]`, matching `cl_forge_muzero_ddr.sv` line 102/105.

Verified kernel facts:
- csynth: `ap_clk` target 4.00 ns, **Estimated 3.197 ns** (margin +0.803 ns), uncertainty 1.08 ns
  (`results/fpga_training_rig/forge_train_step_s1_csynth.rpt` lines 18-35).
- Device util (xcvu47p, all SLRs): **LUT 74% (966,350/1,303,680), BRAM 71% (2,894/4,032),
  DSP 9% (896/9,024), FF 10%, URAM 2%** (rpt lines 53-75). Per-SLR rows: LUT 222%, BRAM 215%,
  **DSP only 29%** — the pressure is LUT/BRAM, never DSP.
- Per-mode latency @250 MHz (rpt lines 41-42): **do_fwd_loss 18.3-18.8 ms, do_bwd 64.5 ms**.
  do_opt + do_train_run report "?" (variable trip count) — a full mode-3 step ≈ fwd+bwd+opt
  ≈ **~85-90 ms/step**; this dominates the run-time and cost estimates below.
- ABI (`extern "C" int forge_train_step_s1(int32_t* buffer, int mode, int a1, int a2, int a3)`):
  m_axi `bundle=gmem`, `max_widen_bitwidth=512` → wide 512-bit bursts, wires straight to the
  proven 512b crossbar with no width converter.
- **S1_BUFLEN = 1,611,280 int32 words = 6.146 MB** (recomputed by walking the offset map in
  `forge_s1_config.hpp`; matches the MAP's "~1.6M words / ~6.4 MB"). All regions 16-word
  (512-bit) aligned via `S1_AL16`.

---

## 1. CL ADAPTATION — clone muzero_ddr → `cl_forge_train_step_s1`

### Which template to clone

**Clone `cl_forge_muzero_ddr`** (the validated AFI, agfi-085dca07f374aa1cd, runs on F2), NOT
pc_train_ddr and NOT morph_lm:
- muzero_ddr has the **13-bit OCL slice** the S1 control map needs (pc_train_ddr is 5-bit → a
  silent register miss on S1's buffer-ptr/mode/args). This alone makes muzero_ddr the right base.
- muzero_ddr is the ONLY one of the three that has been **on silicon** (DDR P&R closed WNS
  +0.154 ns, AFI loads, 108 inf/s measured) — least integration risk.
- pc_train_ddr proves the same DDR topology works for a kernel that **writes back** to DDR
  (S1 also writes grads/W/loss back) — we borrow its `forge_pc_train_step` → m_axi write-master
  wiring confidence, but the SV is byte-identical to muzero_ddr except the kernel name + slice width.
- morph_lm (N=4 replicated, URAM, per-instance pblocks) is overkill — S1 is a SINGLE kernel,
  on-card DDR (not URAM staging), so its fan-out/broadcast OCL and pblock machinery are unneeded.

### Files to create (new package `tikitorch/hardware/hdk/cl_forge_train_step_s1/`)

Copy the muzero_ddr package, then make these EXACT edits:

```
cl_forge_train_step_s1/
├── design/
│   ├── cl_forge_train_step_s1.sv          [from cl_forge_muzero_ddr.sv; 3 edits below]
│   ├── cl_forge_train_step_s1_defines.vh  [rename guard CL_FORGE_TRAIN_STEP_S1_DEFINES]
│   ├── cl_id_defines.vh                    [VERBATIM copy — keep CL_SH_ID0/ID1 literals]
│   └── ip/                                 [empty; gen_ip.tcl populates it on the build box]
├── build/
│   ├── scripts/
│   │   ├── gen_ip.tcl                       [from muzero_ddr; rename OCL slice → s1_ocl_slice]
│   │   └── synth_cl_forge_train_step_s1.tcl [from muzero_ddr; rename CL + OCL-slice IP refs]
│   └── constraints/
│       ├── cl_synth_user.xdc               [EMPTY copy — muzero_ddr's is empty; none needed]
│       └── cl_timing_user.xdc              [EMPTY copy]
└── README.md                               [document the mapping + the 3 SV edits]
```

### The 3 exact SV edits (`cl_forge_train_step_s1.sv`)

Everything else in `cl_forge_muzero_ddr.sv` (lines 1-213) is reused VERBATIM: the reset
synchronizers (lines 49-56, with `rst_main_n_sync` — the name sh_ddr expects), the OCL→kernel
control path, the 2→1 `axi_crossbar_ddr` (lines 137-187, SI0=kernel ID-widened 1→16, SI1=DMA-PCIS
16-bit, M00=DDR base 0), the `krn_bid/krn_rid = *_16[0]` slice (lines 189-190), and the
`sh_ddr #(.DDR_PRESENT(1))` instance (lines 196-211).

1. **Module name + include** (lines 25-31): `cl_forge_muzero_ddr` → `cl_forge_train_step_s1`;
   include `cl_forge_train_step_s1_defines.vh`. PCIM stays tied off
   (`unused_pcim_template.inc`, line 37) — S1 uses on-card DDR.

2. **OCL slice instance name** (line 66): `muzero_ocl_slice OCL_REG_SLC` → `s1_ocl_slice OCL_REG_SLC`
   (so it does not collide with the HDK's `axi_register_slice_light`, same reason muzero renamed it).
   All port connections unchanged.

3. **Kernel instance** (lines 100-118): `forge_train_step_muzero u_kernel` →
   `forge_train_step_s1 u_kernel`. Keep the **13-bit** control address slice
   (`.s_axi_control_AWADDR (ocl_aw_addr_pre[12:0])`, `.s_axi_control_ARADDR (ocl_ar_addr_pre[12:0])`)
   — IDENTICAL to muzero_ddr, do NOT narrow to [4:0]. All m_axi_gmem.* port names are the same
   (HLS emits the standard `m_axi_gmem_*` set for the `bundle=gmem` interface), so the existing
   `krn_*` net hookup is unchanged. The S1 kernel's gmem is 512-bit (`max_widen_bitwidth=512`),
   matching the 512b crossbar → no width converter.

### gen_ip.tcl edits (`build/scripts/gen_ip.tcl`)

From muzero_ddr's `gen_ip.tcl` (lines 1-24), change ONLY the OCL slice module name:
- `-module_name muzero_ocl_slice` → `-module_name s1_ocl_slice` (line 12).
- Keep the AXI4LITE 32b config (line 13) and the **`axi_crossbar_ddr`** crossbar VERBATIM
  (lines 17-23: NUM_SI=2, NUM_MI=1, AXI4, DATA_WIDTH 512, ADDR_WIDTH 64, **ID_WIDTH 16**,
  R_REGISTER 1, M00 base 0x0, ADDR_WIDTH 34). ID_WIDTH must stay 16 (matches cl_sh_ddr_axi;
  15 → 15-bit master, deadlock).

### synth tcl edits (`build/scripts/synth_cl_forge_train_step_s1.tcl`)

From muzero_ddr's `synth_cl_forge_muzero_ddr.tcl` (lines 1-42), change ONLY two IP names:
- line 31: `muzero_ocl_slice/muzero_ocl_slice.xci` → `s1_ocl_slice/s1_ocl_slice.xci`
- line 33: `synth_ip [get_ips muzero_ocl_slice]` → `synth_ip [get_ips s1_ocl_slice]`
- Keep `read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]` (line 13) — it picks up the
  flattened HLS RTL + our CL .sv. Keep the sh_ddr dependency-IP `read_ip` list VERBATIM
  (lines 20-27: cl_ddr4, axi_register_slice[_light], cl_axi3_256b_reg_slice,
  cl_axi_clock_converter[_light]) — **NEVER `read_verilog` sh_ddr.sv** (encrypted envelope error).
  Keep the OOC `synth_design` + header/footer sources.

### How forge_train_step_s1 wires to the shell (the data path)

```
host (BAR0 OCL AXI4-Lite) --> s1_ocl_slice --> kernel s_axi_control[12:0]
       writes: R_BUFFER_0/1=DDR addr 0, R_MODE, R_ARG1..3, R_AP_CTRL=AP_START; polls AP_DONE
host (BAR4 DMA-PCIS, 512b, BURST_CAPABLE) --\
                                             >-- axi_crossbar_ddr (2->1, SASD) --> sh_ddr --> DDR4
kernel m_axi_gmem (512b reads AND writes)  --/   (BUFFER ptr reg = DDR addr 0)
```
DMA staging and kernel m_axi NEVER overlap in time (host stages fully → kicks → kernel runs),
so the SASD crossbar needs no concurrent arbitration and the master ID stays 16-bit.

---

## 2. BUILD STEPS — clone run_pc_afi_build.sh → run_s1_afi_build.sh

The build harness clones the proven `scripts/run_muzero_afi_build.sh` (run_pc_afi_build.sh is the
same script with the CL/KERNEL params swapped). Create `scripts/run_s1_afi_build.sh` identical to
run_muzero_afi_build.sh but with the head env vars:

```bash
CL="${CL:-cl_forge_train_step_s1}"
PKG_REL="${PKG_REL:-tikitorch/hardware/hdk/cl_forge_train_step_s1}"
KERNEL_TCL="${KERNEL_TCL:-synth_train_step_s1.tcl}"   # already committed, top=forge_train_step_s1
KERNEL_PROJ="${KERNEL_PROJ:-forge_train_step_s1_proj}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                    # FPGA Developer AMI (proven)
MAX_MINUTES="${MAX_MINUTES:-360}"                      # see time est: bump 300->360 for the bigger kernel
ROLE="${ROLE:-tiki-sweep-runner}"
STATE_DIR=".tiki_s1_afibuild_state"                   # isolate state from other AFI builds
```

The user-data.sh body is reused unchanged; its steps (lines 59-156 of run_muzero_afi_build.sh):

1. **git archive HEAD → S3** (the launch-time guard at line 38 must check
   `$PKG_REL/design/$CL.sv` — i.e. `cl_forge_train_step_s1/design/cl_forge_train_step_s1.sv`;
   commit S1 the package before launch or the guard aborts).
2. **Launch z1d.2xlarge** (FPGA Developer AMI, IAM `tiki-sweep-runner`, shutdown-on-terminate,
   200 GB gp3), continuous 60s build.log uploader, `shutdown -h +MAX_MINUTES` backstop.
3. On the box: `export HOME=/root`, **install git-lfs then `git lfs pull` in aws-fpga** (shell
   DCP/manifest are LFS objects; without it P&R links pointer stubs → fail), source Vivado/Vitis
   settings, `source hdk_setup.sh`.
4. **HLS the kernel**: `vitis_hls -f synth_train_step_s1.tcl` (this also runs csim — bit-exact
   golden gate before the 3 h P&R) → RTL in `forge_train_step_s1_proj/solution1/syn/verilog/*.v`.
5. **Scaffold CL from HDK CL_TEMPLATE** (cp -rL to dereference dangling symlinks), overlay our
   `design/` + `build/scripts/*.tcl` + `build/constraints/*.xdc`, **flatten the HLS *.v into
   `design/`**, `mkdir design/ip`.
6. **gen IP**: `vivado -mode batch -source build/scripts/gen_ip.tcl -tclargs design/ip`
   (creates s1_ocl_slice + axi_crossbar_ddr).
7. **synth + P&R**: `aws_build_dcp_from_cl.py --cl cl_forge_train_step_s1 --no-encrypt`
   (OOC synth ~20-30 min FAIL-FAST, then P&R). Full log streamed to S3.
8. **Collect** `checkpoints/to_aws/*.Developer_CL.tar` + post_route DCP + `*.rpt` (timing/util)
   → S3; write `DONE`; self-terminate.

**Register the AFI** (locally, after `fetch`):
```bash
aws ec2 create-fpga-image \
  --input-storage-location Bucket=tiki-sweep-<acct>,Key=<tag>/results/to_aws/<...>.Developer_CL.tar \
  --logs-storage-location  Bucket=tiki-sweep-<acct>,Key=<tag>/logs \
  --name s1-trainer --description "forge_train_step_s1 fused trainer DDR" --region us-east-1
# poll: aws ec2 describe-fpga-images --filters Name=fpga-image-global-id,Values=agfi-... \
#       --query 'FpgaImages[0].State.Code'  → until "available"
```

Instance/AMI/cost: z1d.2xlarge @ ~$0.74/hr on-demand (us-east-1), FPGA Developer AMI
`ami-092fc5deb8f3c0f7d`. Wall-clock: HLS+csim ~10-15 min, OOC synth ~20-30 min, P&R ~2.5-3 h
→ **~3-3.5 h, ~$2.5-3 build compute**. AFI registration is free, completes in ~30-60 min.

---

## 3. RUN STEPS — new driver forge_train_step_s1_f2.c + clone run_pc_train_f2.sh

### New host driver: `tikitorch/fpga/silicon/forge_train_step_s1_f2.c`

Model it on `tikitorch/fpga/silicon/muzero_ddr_power.c` (staging + control) and
`forge_pc_train_f2.c` (train loop + readback). Compile as C (`-std=gnu11`, fpga_dma.h needs C99)
on the F2: `gcc -O2 -std=gnu11 -I$SDK/include -I<hls>/forge_kernels forge_train_step_s1_f2.c
-lfpga_mgmt -o s1_f2`. Driver logic:

```c
#include "forge_s1_config.hpp"   // S1_BUFLEN, S1_OFF_* (compile with a C-safe shim or hardcode)
#define R_AP_CTRL 0x00, R_AP_RETURN 0x10, R_BUFFER_0 0x18, R_BUFFER_1 0x1c
#define R_MODE 0x24, R_ARG1 0x2c, R_ARG2 0x34, R_ARG3 0x3c   // S1 args a1/a2/a3
#define AP_START 0x01, AP_DONE 0x02, DDR_BUF 0x0ULL

// 1. attach BAR0 (control) + BAR4 BURST_CAPABLE (DMA-PCIS stage) + BAR4 plain (readback)
// 2. build host buffer hbuf[S1_BUFLEN]: stage INITIAL W_fp masters into S1_OFF_WFP_*,
//    EMBED/W_HEAD/LN-beta values, the matching CODE regions (= s1_encode(dyad(W_fp))),
//    CE_LUT + ATTN_LUT (exp2 ROMs — MUST be host-staged, BRAM static-init reads zero on AFI),
//    and the TOKENS/TARGETS for the training batch. (The one-step bit-exact TB shows the exact
//    contents to stage; reuse gen_buffer_init.py to produce them.)
// 3. stage whole buffer: fpga_pci_write_burst(bar4, DDR_BUF+off*4, hbuf+off, n) in 256K-word chunks
// 4. point kernel at DDR: poke R_BUFFER_0=0, R_BUFFER_1=0
// 5. KICK fused trainer: poke R_MODE=3; R_ARG1=N_STEPS; R_ARG2=S1_OPT_TRAIN_ALL(0); poke R_AP_CTRL=AP_START
//    poll R_AP_CTRL for AP_DONE with a timeout >= N_STEPS*~90ms + margin (e.g. N_STEPS*0.2s)
// 6. read back: peek bar4r at S1_OFF_LOSS (loss proxy), S1_OFF_WFP_* / S1_OFF_W_HEAD (trained W),
//    S1_OFF_LOGITS / S1_OFF_GRAD_LOGIT for the bit-exact check vs the CSIM golden.
```

### Run on F2: clone `scripts/run_pc_train_f2.sh` → `scripts/run_s1_train_f2.sh`

Same structure (AGFI guard, f2.6xlarge, `ami-092fc5deb8f3c0f7d`, IAM role, 40 min backstop).
user-data: `source sdk_setup.sh`, `fpga-clear-local-image -S 0`,
`fpga-load-local-image -S 0 -I $AGFI`, `fpga-describe-local-image -S 0`, fetch src, gcc the
driver, `sudo ./s1_f2 0 <N_STEPS>`, upload results + DONE, self-terminate. Set
`AGFI=agfi-...`, `STATE_DIR=.tiki_s1_train_f2_state`, results →
`results/fpga_training_rig/onfabric_train/`.

### Confirming the CSIM trajectory reproduces (the silicon-proof)

Two complementary checks (mirror morph_lm `golden_traj.jsonl` + `silicon_*` traj):
- **One-step bit-exact**: kick mode-0 (or mode-3 with N_STEPS=1, TRAIN_ALL), read back LOGITS,
  GRAD_LOGIT, LOSS, and the post-step W_fp/codes; require `max|silicon - CSIM golden| == 0`
  (the TB `forge_train_step_s1_tb.cpp` uses the IDENTICAL dyad() rule, so it is authoritative).
- **Trajectory**: kick mode-3 with N_STEPS=K, read the LOSS region; compare the loss curve and the
  final weights to a host CSIM run of the same K steps from the same seed/init. Because mode 3 just
  re-sequences the same fwd/bwd/opt functions through gmem, it is bit-exact to the per-mode golden →
  the silicon loss must equal the CSIM loss word-for-word. The TRAIN_ALL block in the TB already
  reports CE 4.848 → 2.067 over its horizon; reproducing that drop on silicon is the success bar.

Run cost: f2.6xlarge ~$1.65/hr; a few-hundred-step mode-3 run finishes in minutes (see RISKS for
the ~90 ms/step throughput). **< $1 per F2 run.**

---

## 4. FLOORPLAN VERDICT — does the 74% LUT / 71% BRAM kernel co-fit with the shell?

**Verdict: YES, it should co-fit, but it is TIGHTER than muzero (the only honest caveat).**

Reasoning, with the numbers:
- The csynth util (74% LUT / 71% BRAM / 9% DSP, FF 10%, URAM 2%) is the **device-aggregate** view
  (LUT 966,350/1,303,680; BRAM 2,894/4,032). The per-SLR "222% / 215%" rows are NOT an
  over-subscription verdict — they are the single-SLR denominator (434,560 LUT / 1,344 BRAM per
  SLR), and the xcvu47p has **3 SLRs**, so the kernel spanning ~2.2 SLRs of LUT and ~2.15 SLRs of
  BRAM fits in the 3-SLR aggregate with the shell taking the rest.
- The F2 shell + DDR path (sh_ddr MIG + clock converters + DMA-PCIS + our OCL slice + crossbar) is
  the SAME shell muzero_ddr shipped with. Empirically that shell DDR path cost the muzero build
  ~400-500 BRAM and ~30-40k LUT. Add to S1: **BRAM ~2,894 + ~450 ≈ 3,350 / 4,032 ≈ 83%**;
  **LUT ~966k + ~35k ≈ 1,001k / 1,304k ≈ 77%**. DSP is a non-issue (896 + shell ≪ 9,024).
- Clock: 250 MHz (4.0 ns). csynth Estimate is 3.197 ns with 1.08 ns uncertainty already folded in;
  muzero_ddr's much-smaller kernel closed post-route at only +0.154 ns. S1 is ~5× the LUT/BRAM of
  the muzero forward, so **post-route WNS is the realistic risk**, not raw area capacity.

**Mitigations if P&R is tight (in order, cheapest first):**
1. **Lower the user clock to 200 MHz (5.0 ns)** — the kernel closes csynth at 3.197 ns, so a 5.0 ns
   target gives ~1.8 ns of routing slack with zero RTL change (set the clock recipe in the build,
   or relax `create_clock` in synth). This is the single highest-leverage knob; latency grows 25%
   (still ~110 ms/fused step) which is fine for a convergence proof.
2. **Add pblocks** to confine the kernel to SLR0+SLR1 and keep the shell/DDR in SLR2 (the morph_lm
   `cl_pnr_user_floorplan.xdc` pattern). muzero_ddr needed none; add only if the first P&R misses.
3. **Drop to FREEZE_CODES at runtime (a2=1)** if a future area-trim is needed — this is a runtime
   arg, not a build change, but the kernel area is mode-agnostic so it does not help P&R; listed
   only as the fallback if the fused build itself won't fit (then build a fwd-only or
   fwd+bwd variant by `#if`-gating do_train_run — last resort).
4. **S1_LANE 16→8** (config rebuild) halves GEMM parallelism for ~20% LUT at 2× latency — only if
   1-2 fail.

The empty `cl_synth_user.xdc` / `cl_timing_user.xdc` (muzero_ddr shipped them empty) are the right
starting point; add constraints only on a measured P&R miss.

---

## 5. RISKS

| # | Risk | Likelihood | Mitigation |
|---|------|-----------|------------|
| R1 | **PCIM not drivable on EC2 Nitro** (host-DRAM IOMMU wall — the Stage 3b finding). | Certain if PCIM is used | Use the **DDR path** (sh_ddr + DMA-PCIS staging) — already the chosen architecture; PCIM stays tied off. NON-NEGOTIABLE. |
| R2 | **Post-route timing miss** — S1 is ~5× muzero's LUT/BRAM; routing congestion can blow the +0.8 ns csynth margin. | Medium | Build at **200 MHz** (Mitigation #1 §4); then pblocks. csim already passed so it's a placement problem, not a math one. |
| R3 | **BRAM tightness** (~83% post-shell). | Low-Medium | xcvu47p has 4,032 BRAM aggregate; 83% fit. If P&R can't place, fall to a fwd+bwd-only build (drop do_opt/do_train_run residency) or S1_LANE=8. |
| R4 | **CE_LUT / ATTN_LUT static-init reads ZERO on AFI** (F2 shell ROM bug). Wrong/zero LUT → silent uniform softmax, low accuracy, no NaN. | Certain if not staged | Host **always stages** S1_OFF_CE_LUT + S1_OFF_ATTN_LUT (exp2 ROMs) in the buffer before the kick. The kernel `burst_load`s them at entry. Verify with a one-step bit-exact check. |
| R5 | **Per-step host round-trip would be slow** (stage 6 MB + kick + poll, per step). | N/A — avoided | Use **mode 3 (do_train_run)**: ONE kick runs N_STEPS fwd→bwd→opt with weights/state resident in DDR; no per-step host traffic. This is why mode 3 exists. |
| R6 | **Throughput / wall-time**: do_fwd 18.3 ms + do_bwd 64.5 ms + do_opt ≈ **~85-90 ms/step** @250 MHz. 1k steps ≈ 90 s; a real convergence run is minutes-to-tens-of-minutes. | Inherent | Honest: S1 is a **correctness/convergence proof**, not a throughput kernel (per the kernel's own header). For tok/s, project from the standalone replicated kernels (TOKS_PROJECTION.md). Keep N_STEPS modest; checkpoint LOSS periodically by reading back between kicks. |
| R7 | **Kick poll timeout too short** → false "hung". A mode-3 kick can take N_STEPS×90 ms. | Medium | Set the AP_DONE poll timeout to `N_STEPS * 0.2 s + 5 s` (muzero used 10 s for one fwd; S1 fused is far longer). |
| R8 | **Host buffer staging size** (6.15 MB) over BAR4. | Low | 256K-word (1 MB) bursts as in muzero_ddr_power.c; ~6 bursts, sub-second. After the first stage, mode-3 keeps state in DDR — re-stage only TOKENS/TARGETS per new batch (the forge_pc_train_f2.c partial-stage pattern). |
| R9 | **Control address width mismatch** (using pc_train's 5-bit slice). | Low (called out) | Use **[12:0]** (muzero_ddr width). S1's control bundle = buffer+mode+a1/a2/a3. |
| R10 | **W_fp init contract**: the forward reads CODE/VALUE regions, not W_fp. If only W_fp is staged, first forward reads garbage codes. | Medium | Either (a) host stages BOTH W_fp and the matching codes/values (= s1_encode(dyad(W_fp)) / W_fp), OR (b) the mode-3 entry already runs `do_opt(S1_OPT_INIT)` which re-quantizes codes/values from staged W_fp before step 0 — so staging W_fp alone is sufficient for mode 3. Use (b); for standalone mode-0 testing use (a) (the TB stages both). |
| R11 | **git-lfs not pulled on build box** → P&R links shell-DCP pointer stubs. | Low (handled) | The user-data installs git-lfs + `git lfs pull` in aws-fpga (already in the runner). |
| R12 | **Encrypted sh_ddr read** → "encrypted envelope" synth error. | Low (handled) | NEVER `read_verilog` sh_ddr.sv; only `read_ip` its deps (synth tcl lines 20-27). Instantiate it in the .sv only. |

---

## 6. TIME + COST ESTIMATE

| Phase | Wall-clock | Cost | Notes |
|-------|-----------|------|-------|
| Create CL package + driver + runners (local) | ~1-2 h human | $0 | 3 SV edits + 2 IP renames + new .c driver + 2 cloned runners; commit S1 package. |
| Local csim/csynth re-verify (optional, on a dev box w/ Vitis) | ~15-30 min | $0 | Gate before spending build money; the rpt already exists. |
| AFI build (z1d.2xlarge) | ~3-3.5 h | ~$2.5-3 | HLS+csim ~15 min, OOC synth ~20-30 min FAIL-FAST, P&R ~2.5-3 h. Budget MAX_MINUTES=360. |
| AFI registration (create-fpga-image) | ~30-60 min | $0 | Async; poll for "available". |
| F2 run (f2.6xlarge) | ~10-20 min wall (a few min of FPGA) | < $1 | Load AGFI, stage 6 MB, mode-3 N_STEPS kick (~90 ms/step), read back loss + W, bit-exact check. |
| **Total** | **~half a day** (mostly P&R + AFI reg, both unattended) | **~$3-4** | Plus 1 contingency rebuild at 200 MHz if R2 hits → +$3, +3.5 h. |

**Bottom line:** every piece is a small, mechanical diff off a CL that already ran on F2; the kernel
already csim-bit-exact + csynth-closes with margin; modes 1/2/3 are real so the one-kick fused
trainer is the run path. The only genuine unknown is post-route timing at 250 MHz on a ~5×-bigger
kernel — de-risked by building at 200 MHz if the first P&R misses.
