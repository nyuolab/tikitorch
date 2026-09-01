# TikiTorch Hardware: HLS Kernels and the AFI / Silicon Build Guide

This directory ships the **High-Level Synthesis (HLS) C++ kernels** that
implement the TikiTorch tropical learning loop directly in FPGA fabric, plus
the synthesis scripts and a guide to taking them all the way to **silicon** on
an AWS EC2 F2 instance (AFI = Amazon FPGA Image).

The reported D128 and D512 MorphSSM configurations execute a complete tropical
language-model **training step** (forward + loss + backward + sign-SGD update)
on FPGA fabric. The archived D128 build reports use **zero DSP blocks**. The
D512 archive contains its execution record and source recipe but not a final
routed-resource report, so this guide does not assign D512 a physical zero-DSP
result. Build-specific evidence is not a blanket statement about every kernel
or future build, and a release claim also needs the operator/source audit
described below—DSP=0 alone does not rule out multiplication synthesized into
LUTs.

---

## 1. What is here

```
hls/
  forge_hls_types.hpp                  # ap_int<16>/<32> under Vitis, stdint fallback for g++
  forge_kernels/
    forge_buffer.hpp                   # 1-layer flat int32 buffer layout (D=128, V=128, L=128)
    forge_buffer_2layer.hpp            # 2-layer buffer layout
    forge_buffer_3layer.hpp            # 3-layer buffer layout
    forge_train_step.cpp               # 1-layer fused fwd+loss+bwd+update kernel
    forge_train_step_2layer.cpp        # 2-layer fused training-step kernel
    forge_train_step_3layer.cpp        # 3-layer fused training-step kernel
    forge_train_step*_tb.cpp           # standalone g++ testbenches (functional verification)
    synth.tcl, synth_2layer.tcl, synth_3layer.tcl   # Vitis HLS csynth scripts
```

- **`data_t` = `ap_int<16>`, `acc_t` = `ap_int<32>`** under Vitis HLS. All
  arithmetic is fixed-point integer; the tropical ops (max/min/add) need no
  multipliers, so the datapath is multiplier-free by construction.
- The training kernels are **fused**: a single `forge_train_step` does the
  whole per-token loop (embed -> tropical attention -> tropical FFN ->
  head -> loss -> backward -> sign-SGD weight update) over one window of
  tokens, writing updated weights back to the shared buffer.

These kernels correspond to the `tikitorch.models` tropical LM
(tropical-shift attention + dilation/erosion FFN) trained with
`tikitorch.optim.SignSGD` / `PureTropicalSGD`.

---

## 2. Functional verification on the dev host (no FPGA needed)

The `*_tb.cpp` testbenches compile with plain `g++` (the `forge_hls_types.hpp`
shim swaps `ap_int<N>` for `int16_t`/`int32_t` when `__VITIS_HLS__` is not
defined). Use this to compare selected integer-kernel outputs with a host
reference before spending synthesis time:

```bash
cd tikitorch/hardware/hls/forge_kernels
g++ -O2 -std=c++17 -I.. forge_train_step.cpp forge_train_step_tb.cpp -o tb
./tb        # prints per-step loss / accuracy; should track the tikitorch run
```

Match it against the PyTorch model run with hard tropical ops (`tau=0`,
`use_soft=False`) and integer (INT16) quantized weights.

> **WARNING — the host golden does NOT cover the synthesis path.** The `*_tb.cpp`
> / `golden_*.cpp` harnesses compile the `#else` scalar path (no `__VITIS_HLS__`/
> `__SYNTHESIS__`). A green golden proves only the **host-path observables that
> the harness actually compares**, not the complete math or **hardware**. It does
> NOT exercise the `wide_t`/`ap_int<>` burst path, so it
> misses the silicon-only bug classes below. Treat the golden as necessary, not
> sufficient — re-run a CSIM-vs-silicon check (or HLS C/RTL co-sim) after any
> change to a burst, an offset, an `ap_int<>` width, or a table.

### Silicon-only bug classes (CSIM-invisible) — checklist for new kernels

Three classes bit only on the synthesized AFI; each cost a debug cycle on the
fully-tropical morph-LM (see `tikitorch/fpga/silicon/SILICON_FINDINGS.md`):

1. **Misaligned wide burst.** `burst_load_words`/`burst_store_words` truncate the
   offset to `(word_off >> 4)` on the wide path, so a base that is not a multiple
   of 16 reads/writes SHIFTED on silicon while the scalar golden stays correct.
   *Guard:* the `static_assert` 16-alignment block in the buffer header
   (`forge_buffer_morph_lm.hpp` / `forge_buffer_morph_ecg.hpp`) — keep every
   burst-reached `OFF_*`/`WORDS_*` a multiple of 16; it fails the BUILD, not the bitstream.
2. **Dead `static const` ROM.** A table declared `static const` can read ALL-ZERO
   on the packaged AFI (its BRAM init does not survive encrypt + place); CSIM
   reads it fine. *Fix / pattern:* **host-stage the table into the buffer** and
   read it like data — reserve a 16-aligned `OFF_<TABLE>` region, have the host
   write the values once, `burst_load_words` it into a local array. Never index a
   `static const` ROM on the data path. Reference: `OFF_CE_LUT`/`WORDS_CE_LUT`
   (staged in `fpga/silicon/gen_buffer_init.py`, read as `ce_lut[]` in
   `do_fwd_loss` — NOT `FT_CE_EXP2_Q16[]`, which is kept only as the CSIM source).
3. **`ap_int` width truncation.** An `int` arg truncated to `ap_int<16>` on the
   synthesis path while the host `int` path is fine (the signsgd `threshold` bug:
   CE Q16 grads exceed 32767 → use `ap_int<32>`).

---

## 3. HLS synthesis (Vitis HLS)

Each kernel has a `synth_*.tcl` that targets the F2 part and a 250 MHz clock
(`target_period_ns 4.0`):

```bash
cd tikitorch/hardware/hls/forge_kernels
vitis_hls -f synth.tcl            # 1-layer
vitis_hls -f synth_2layer.tcl     # 2-layer
vitis_hls -f synth_3layer.tcl     # 3-layer
```

Check the synthesis report and generated operator tables for the gates that
matter:

- **DSP = 0 for the named build**, plus an empty multiplier/operator report and
  source-level closure audit. Zero DSP is an important physical observation,
  but by itself it does not exclude LUT-based multipliers.
- **WNS (Worst Negative Slack) >= 0** at the target period, or the AFI build
  misses timing. Several archived experimental builds landed around WNS
  -0.15 to -0.20 ns; those are preliminary measurements, not timing-closed
  release artifacts. Rebuild and re-verify after any timing fix.

---

## 4. From kernel to silicon (AWS EC2 F2 / AFI)

High-level path (uses the AWS FPGA HDK + Vitis flow):

1. **Synthesize** the kernel to RTL (section 3). The generated Custom Logic
   (CL) wraps the kernel with a 512-bit AXI master (PCIM) — see ABI notes
   below.
2. **Build the AFI**: package the CL with the HDK `aws_build_dcp_from_cl`
   flow, upload the tarball to S3, and call `create-fpga-image`. AWS returns
   an **AFI id** and an **AGFI id**.
3. **Load and run**: on the F2 instance, `fpga-load-local-image` the AGFI,
   then drive the kernel from the host (a small C driver poking the OCL
   AXI-Lite registers and DMA'ing the int32 buffer over PCIM).

### ABI / wiring notes (carved from the silicon-validated build)

These are the load-bearing details that make a working AFI rather than a
hung one:

- **One flat int32 buffer in device memory.** All tensors (weights, grads,
  activations, tokens, targets) live at fixed byte offsets in a single int32
  buffer — see `forge_buffer.hpp`. Each kernel reads only the regions it needs
  and writes only what it produces. The host owns the per-window loop.
- **Use PCIM, not the DDR bridge.** Move the buffer over the 512-bit PCIM AXI
  master. `max_widen_bitwidth=512` forces a 512-bit AXI master that the
  auto-generated CL's hardcoded 512-bit nets expect.
- **No multi-kernel mux.** Fuse the whole training step into one kernel rather
  than wiring several kernels behind a selector — the multi-kernel mux path
  does not map cleanly through the CL on this flow.
- **Watch int-return register shifts.** A kernel that returns an `int` shifts
  the OCL AXI-Lite register map; account for the offset when the host reads
  back status/return values, or you will read the wrong register.
- **Scalar args via s_axilite**, the buffer pointer via the wide AXI master.
- **Commit the whole HDK CL package.** `hdk/cl_<kernel>/design/*.sv|*.vh`,
  `design/ip/*.xci`, `build/scripts/gen_ip.tcl` + `synth_cl_*.tcl`, and
  `build/constraints/*.xdc` are build INPUTS, not regenerable artifacts. An
  uncommitted CL build is a multi-hour from-scratch reconstruction (it happened
  once). Build artifacts (checkpoints/.dcp/to_aws tarballs) stay gitignored; the
  sources do not. See `hdk/cl_forge_morph_lm/README.md`.

### Historical operating points (configuration-specific)

- 1-layer tropical LM, D=128 / V=128 / L=128, ~1200 tok/s with the historical
  C host driver. The archive does not define a sufficiently controlled Python
  denominator, so this is not a host-language or accelerator speedup claim.
- 2-layer tropical LM trained end-to-end on F2 silicon: ~770 tok/s, with a
  derived ~9 mJ/token estimate and WNS around -0.197 ns. The energy value is
  not a measured full-card power result, and the negative WNS means this build
  missed timing; retain it as historical experimental context, not a
  timing-closed performance or priority claim.

Your numbers will vary with part, timing closure, and host driver. Treat these
as archived operating points, not guaranteed performance or release support.

---

## 5. Mapping kernels to TikiTorch Python

| HLS kernel                  | TikiTorch equivalent                                              |
|-----------------------------|------------------------------------------------------------------|
| tropical attention block    | `tikitorch.nn.TropicalShiftAttention` (tau=0, hard)              |
| dilation -> erosion FFN     | `tikitorch.nn.TropicalDilation` -> `tikitorch.nn.TropicalLinear`|
| sign-SGD weight update      | `tikitorch.optim.SignSGD` / `tikitorch.optim.PureTropicalSGD`   |
| fused `forge_train_step`    | `tikitorch.fpga.AcceleratedModel.train_step(...)` for a matched runner |

Train and validate in PyTorch with `tikitorch`, anneal `tau -> 0`, quantize to
INT16, compare the selected observables against the `*_tb.cpp` testbench, then
synthesize and build the AFI. A host checkpoint can seed hard deployment, but
the native runner must quantize and marshal parameters into the kernel-specific
INT16 layout, synchronize updated values back when needed, and re-verify the
target image. Ordinary `compute_loss(); backward(); opt.step()` remains the
PyTorch path; use `AcceleratedModel.train_step(...)` for the fused native path.
