# TikiTorch Hardware: HLS Kernels and the AFI / Silicon Build Guide

This directory ships the **High-Level Synthesis (HLS) C++ kernels** that
implement the TikiTorch tropical learning loop directly in FPGA fabric, plus
the synthesis scripts and a guide to taking them all the way to **silicon** on
an AWS EC2 F2 instance (AFI = Amazon FPGA Image).

The headline result: a complete tropical language-model **training step**
(forward + loss + backward + sign-SGD update) runs on the FPGA using **zero DSP
blocks** — only comparators, adders, and bit-shifts. This is the physical
realization of TikiTorch's algebraic-closure claim.

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
defined). Use this to verify the integer kernel matches the PyTorch reference
before spending synthesis time:

```bash
cd tikitorch/hardware/hls/forge_kernels
g++ -O2 -std=c++17 -I.. forge_train_step.cpp forge_train_step_tb.cpp -o tb
./tb        # prints per-step loss / accuracy; should track the tikitorch run
```

Match it against the PyTorch model run with hard tropical ops (`tau=0`,
`use_soft=False`) and integer (INT16) quantized weights.

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

Check the synthesis report for the two things that matter:

- **DSP = 0** — confirms the multiplier-free closure claim physically.
- **WNS (Worst Negative Slack) >= 0** at the target period, or the AFI build
  will fail timing. (In practice the validated builds land at WNS around
  -0.15 to -0.20 ns and required one or two AFI rebuilds to close timing; see
  the lessons below.)

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

### Reference operating point (from the validated 1- and 2-layer builds)

- 1-layer tropical LM, D=128 / V=128 / L=128, ~1200 tok/s with a C host
  driver (about 15x over the Python host baseline).
- 2-layer tropical LM trained end-to-end on F2 silicon: ~770 tok/s, about
  9 mJ/token, WNS around -0.197 ns — the first multilayer NN training on FPGA
  with **zero DSPs**.

Your numbers will vary with part, clock closure, and host driver; treat these
as the order of magnitude to expect.

---

## 5. Mapping kernels to TikiTorch Python

| HLS kernel                  | TikiTorch equivalent                                              |
|-----------------------------|------------------------------------------------------------------|
| tropical attention block    | `tikitorch.nn.TropicalShiftAttention` (tau=0, hard)              |
| dilation -> erosion FFN     | `tikitorch.nn.TropicalDilation` -> `tikitorch.nn.TropicalLinear`|
| sign-SGD weight update      | `tikitorch.optim.SignSGD` / `tikitorch.optim.PureTropicalSGD`   |
| fused `forge_train_step`    | one `loss.backward(); opt.step()` over a token window           |

Train and validate in PyTorch with `tikitorch`, anneal `tau -> 0`, quantize to
INT16, verify against the `*_tb.cpp` testbench, then synthesize and build the
AFI. No parameter copying is needed between the soft-trained and hard-deployed
forms — the same weight tensors drive both paths (Maslov-dequantization
curriculum).
