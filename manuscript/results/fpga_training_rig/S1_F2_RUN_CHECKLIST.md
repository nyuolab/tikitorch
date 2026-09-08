# S1 on-F2 GO / NO-GO run checklist (forge_train_step_s1, clkgen+CDC AFI)

First-ever on-silicon run of the multiplier-free transformer-block TRAINER with the new
**AWS_CLK_GEN + dual-AXI-CDC** architecture (kernel @ clk_extra_a1=125 MHz, shell @ clk_main_a0=250 MHz).
Success = reproduce the TB MODE-3 convergence on real fabric: stage the seed=4242 init, kick mode 3,
read back LOGITS, see the true CE drop below the log-V floor.

Files: driver `tikitorch/fpga/silicon/forge_train_step_s1_f2.c`; CL `tikitorch/hardware/hdk/cl_forge_train_step_s1/design/cl_forge_train_step_s1.sv`;
runner `scripts/run_s1_train_f2.sh`; plan `results/fpga_training_rig/S1_CLKGEN_PLAN.md`; spec `/tmp/clkref_local/clkref/AWS_CLK_GEN_spec.md`.

---

## Pre-flight (off-silicon, before launch)

- [ ] **AFI is clkgen-enabled.** The DCP/AFI MUST have been built with `--aws_clk_gen --clock_recipe_a A1`
      (S1_CLKGEN_PLAN.md sec 5). Verify the build manifest references `aws_clk_gen` and recipe A1.
      *Failure symptom if wrong:* MMCM_LOCK_REG never reads bit0 set (no MMCM_A in the image) → bring-up
      times out at 5 s → `MMCM_A did NOT lock` → abort before any kick. (Plan risk #5.)
- [ ] **AGFI is `available`.** `aws ec2 describe-fpga-images ... State.Code == available` (runner guards this).
      *Failure:* load fails / wrong image.
- [ ] **Build flag present.** Runner gcc line includes `-DHAVE_FPGA_CLKGEN=0` (FIXED — uses the explicit-register
      bring-up fallback; polls only the group-A MMCM bit; no `-lfpga_clkgen` dependency).
      *Failure if missing:* default `HAVE_FPGA_CLKGEN=1` path → link error `undefined reference to
      aws_clkgen_deassert_resets` (lib not bundled in `-lfpga_mgmt`) → `/tmp/s1_f2` never built → run dies at build.
- [ ] **Canonical config header on the -I path.** Runner uses `-I/opt/tiki/.../hardware/hls/forge_kernels`
      (git-archive, D=64) — NOT a stale `/private/tmp/forge_s1_config.hpp` (D=128). `S1_BUFLEN` must be 404496
      (1.543 MB). *Failure:* a D=128 header would change the staged layout/size and break byte-exactness vs the kernel.

## Step 0 — AFI load (runner user-data)

- [ ] `fpga-clear-local-image -S 0` then `fpga-load-local-image -S 0 -I $AGFI`; `sleep 4`; `fpga-describe-local-image -S 0`.
- [ ] **Verify:** describe shows the AGFI `loaded`, AFI state OK, shell version present.
      *Failure symptom:* load error / image not `loaded` → all PCI attaches below will see a bare shell.

## Step 1 — PCI attach

- [ ] Driver attaches BAR0 (`FPGA_APP_PF/APP_PF_BAR0`, OCL AXI4-Lite control), BAR4 burst
      (`APP_PF_BAR4/BURST_CAPABLE`, DMA-PCIS staging), BAR4 plain (`APP_PF_BAR4`, readback peeks).
- [ ] **Verify:** all three attaches return 0. *Failure:* `attach BAR0/BAR4 failed` → abort (return 3/4).

## Step 2 — AWS_CLK_GEN bring-up (THE NEW, NEVER-RUN-ON-SILICON STEP)  [driver clkgen_bringup()]

Order is load-bearing (spec anti-pattern: never release resets before MMCM lock — AWS_CLK_GEN_spec.md:165-167).

- [ ] **2a. Attach mgmt BAR.** `fpga_pci_attach(slot, FPGA_MGMT_PF=1, MGMT_PF_BAR4=4)` — the SDA/PF1-BAR4 path
      the CL wires to AWS_CLK_GEN's `s_axil_ctrl_*` (cl_forge_train_step_s1.sv:106-110). AWS_CLK_GEN self-decodes
      the full `0x0005_xxxx` space (aws_clk_gen.sv:218-268), so the driver's `CLKGEN_BASE_REG=0x58000 + off`
      absolute addressing reaches BASE_REG with NO external SDA xbar.
      *Failure symptom:* `attach mgmt BAR ... failed` → return -1. (If this CL ever lost the SDA wiring, the
      mgmt poke/peek would target a tied-off slave and reads would return 0xBAAD_DEC0 / never lock — see 2b.)
- [ ] **2b. Poll MMCM_LOCK_REG (0x58000+0x20) for bit0 (MMCM_A locked).** Group-A-only enable → only bit0 can set;
      mask = `CLKGEN_MMCM_A_LOCKED=0x01` (NOT the all-groups 0x0151). 5 s timeout.
      *Verify:* lock acquired within 5 s. *Failure symptom:* `MMCM_A did NOT lock (MMCM_LOCK_REG=0x........)` →
      return -2. If readback is `0xBAAD_DEC0` the SDA path is wrong/decoding undefined space; if `0x0` the clkgen
      registers aren't in the image (AFI not built with `--aws_clk_gen`).
- [ ] **2c. Clear SYS_RST_REG (0x58000+0x14) = 0** (deassert cl_rst_a1_n bit1 + all writable resets) — ONLY after 2b.
      *Verify:* log prints `writable bits[9:0]=0x000` (the writable window; reserved [31:10] STAY SET per
      AWS_CLK_GEN_spec.md:121 — a non-zero RAW readback is EXPECTED and is NOT a failure). *Failure symptom:*
      `SYS_RST_REG deassert write failed` → return -3.
- [ ] **2d. Bring-up returns 0** → `clk_extra_a1` (125 MHz) live, kernel + both CDC kernel-sides out of reset.
      *Failure:* nonzero → main aborts (return 9) BEFORE stage/kick.

> Residual timing note (not a blocker): there is no explicit settle after bring-up. The kernel-domain reset
> sync needs a few 125 MHz cycles to propagate before the first OCL control transaction crosses the OCL CDC;
> this is masked by the calloc + ~1.5 MB BAR4 stage (Step 3, many ms) that runs before the first kick. If the
> stage path is ever shortened, add a ~1 ms sleep after bring-up.

## Step 3 — Stage the working set into DDR over BAR4

- [ ] Driver builds `hbuf` (seed=4242, `stage_opt_buffer_memorize` = TB-identical init) then `stage(0, S1_BUFLEN)`
      via `fpga_pci_write_burst` in 256K-word (1 MB) chunks to DDR byte addr `o*4` (DDR_BUF=0 = crossbar M00 base).
- [ ] **Verify:** `staged 1.543 MB into DDR via BAR4 in ... s` printed, no `write_burst rc` errors. All staged
      regions are 16-word/512-bit aligned by construction (S1_AL16). *Failure symptom:* `write_burst @word N rc=..`
      → return 6 (DMA/crossbar fault).
- [ ] *(optional insurance, not in driver)* peek-back a few staged input words and compare to `hbuf` to localize a
      CDC/crossbar staging corruption before kicking (proven pattern: forge_pc_ddr_verify.c). Worthwhile on this
      first run because a staging fault would otherwise present only as non-convergence with no isolation.

## Step 4 — Warm kicks (INIT + initial forward) through the OCL CDC

- [ ] `kick(MODE_OPT, S1_OPT_INIT, a2, 10 s)` re-quantizes codes/values from staged W_fp; then
      `kick(MODE_FWD, 0, 0, 10 s)` → forward + CE → LOGITS.
- [ ] **Verify:** both return 0 (AP_DONE seen). MMIO order is safe across the async 250→125 CDC: every poke blocks
      on BRESP and every peek on RVALID, so BUFFER/MODE/ARG writes flush before AP_START, and AP_START flushes
      before the AP_DONE poll. *Failure symptom:* `KICK HUNG (mode=2/0 ...)` → return 7. A hang HERE (vs a hang at
      the mode-3 kick) most likely means the clkgen bring-up didn't actually release the kernel (revisit Step 2) or
      a CDC handshake deadlock (CL/IP-config issue → rebuild, see deferred).
- [ ] **Verify:** `INITIAL CE = ...` printed (near the log-V floor pre-training, ~CE 4.x; floor=log V).

## Step 5 — THE training kick: mode 3 (fused trainer)

- [ ] `kick(MODE_TRAIN, n_steps, a2, n_steps*0.2+5 s)` — ONE kick runs n_steps fwd→bwd→dyadic-Adam resident in DDR.
- [ ] **Verify:** `mode-3 AP_DONE in ... s (.. ms/step)`. *Failure symptom:* `KICK HUNG (mode=3 ...)`. If steps run
      slower than ~90 ms/step the default N_STEPS=200 (45 s budget) could trip a FALSE hang — measure ms/step from
      the warm forward / a small N_STEPS first and scale the budget before a long run (tunable, not a bug).

## Step 6 — Readback + CE gate

- [ ] Readback (AFTER AP_DONE, separate non-burst `bar4r` handle, sign-preserving uint32→int32): LOGITS, TARGETS,
      GRAD_LOGIT, LOSS[16], W_HEAD, EMBED, final-layer R.
- [ ] **Verify the four un-fakeable convergence gates** (collapse sits AT the floor and fails all four):
      - `CE-drops >= 0.5`  (true float CE, CE_TEMP=0.01, == TB metric)
      - `below-floor`      (`ce1 < log V - 0.25`)
      - `acc-improves`     (`acc1 > acc0`)
      - `no-saturation`    (`peak|R| < 32767` — not pinned at the int16 ceiling)
- [ ] **PASS:** `ON-FABRIC TRAINER CONVERGES (silicon == CSIM)` + `[S1 RESULT] ... drop=...`.
      *Failure symptom:* `NO-CONVERGENCE (inspect)`. If CE sits exactly at `log V` with peak|R|==32767 → int16
      saturation/collapse; if CE is flat but not at floor → check `max|gradlogit rowsum|` (CE adjoint rows must
      sum ≈0) and re-confirm the staged init (seed/order) is byte-identical to the TB.

---

## Residual risks (watch, not blockers)

1. **SDK lock-mask unverifiable.** Resolved by `-DHAVE_FPGA_CLKGEN=0`: the explicit fallback polls only the
   MMCM_A bit, so we do NOT depend on whether `aws_clkgen_deassert_resets()` hard-polls 0x0151. If a future run
   reverts to the SDK path (`HAVE_FPGA_CLKGEN=1` + `-lfpga_clkgen`), first confirm the SDK reads the enabled set
   (CLKS_AVAIL) rather than 0x0151, OR enable all four clock groups in the CL to match the cl_mem_perf reference.
2. **CDC config / deadlock (rebuild-class).** If a warm kick hangs with bring-up confirmed good, suspect a CDC
   IP-config mismatch (ID width / AXI4 vs AXI4LITE / ACLK_RATIO) — this is a CL/IP fix, not a host fix.
3. **Mode-3 timeout tightness.** Tune `N_STEPS` budget from a measured ms/step (see Step 5).
4. **No post-stage sanity readback in the driver.** Consider the optional peek-back (Step 3) on this first run.
5. **Cosmetic only:** runner banner / driver say "~6.15 MB" (old D=128); actual staged size is 1.543 MB (D=64).
   The driver computes the real byte count from compile-time S1_BUFLEN, so the run is correct — comment is stale.
