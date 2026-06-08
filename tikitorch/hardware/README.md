# TikiTorch Hardware: HLS Kernels and the FPGA / Silicon Path

This directory ships the **High-Level Synthesis (HLS) C++ kernels** that
implement the TikiTorch tropical learning loop directly in FPGA fabric, the
Vitis HLS synthesis scripts, host-side testbenches, and a guide to taking the
kernels all the way to **silicon** on an AWS EC2 **F2** instance (AFI = Amazon
FPGA Image).

The headline claim of TikiTorch — *algebraic closure of the learning loop* — is
that a complete training step (forward + loss + backward + weight update) can be
expressed entirely in the tropical (max-plus / min-plus) semiring, whose only
primitive operations are **compare, add, and bit-shift**. None of those needs a
multiplier. These kernels are the physical proof: a full on-chip training step
that synthesizes to **zero DSP blocks** — only LUTs, carry chains, and
flip-flops.

> **Measured headline (PTB-XL ECG MorphSSM core, `forge_train_step_morph_ecg`):**
> **1,207 tok/s**, **5.8 mJ/token**, **DSP = 0**, **WNS = +0.009 ns** (timing
> closes with positive slack at the 250 MHz target). See the manuscript for the
> full table; numbers are quoted in §6.

---

## 1. The closure -> hardware mapping (why there are no multipliers)

A conventional NN training step is dominated by `y = W @ x` and its transpose in
the backward pass — i.e. multiply-accumulate, which on an FPGA means **DSP
slices**. TikiTorch replaces multiply-accumulate with the tropical semiring:

| Conventional (ring `(+, ×)`) | Tropical (`(max, +)` / `(min, +)`)        | FPGA primitive            |
|------------------------------|-------------------------------------------|---------------------------|
| `sum_k W[i,k] * x[k]`        | `max_k (W[i,k] + x[k])` (dilation)        | adder + comparator tree   |
| `sum_k W[i,k] * x[k]`        | `min_k (W[i,k] + x[k])` (erosion)         | adder + comparator tree   |
| `dL/dW = x ⊗ δ` (dense)      | scatter `δ` to the **argmax/argmin** tap  | index store + add (coeff 1)|
| `W -= lr * grad`             | `W -= lr * sign(grad)`, `lr = 2^-s`       | sign bit + bit-shift add  |

Three consequences make the datapath multiplier-free **by construction**:

1. **Forward.** A tropical "matmul" is a tree of adders feeding a tree of
   max/min comparators. No partial products, so no DSPs — just carry chains and
   LUT comparators.
2. **Backward.** Because max-plus / min-plus is piecewise-linear, the local
   Jacobian is a **selection matrix**: the gradient at an output routes, with
   coefficient exactly 1, to the single input that won the max/min. The kernels
   *save the winning index* (`argmax`/`argmin`) on the forward pass and, on the
   backward pass, **sparse-scatter** the incoming gradient to that index — an
   indexed add, never a multiply.
3. **Update.** Weights are updated with **sign-SGD** (`W -= lr·sign(grad)`).
   With a power-of-two learning rate `lr = 2^-s`, the update is a sign test
   plus an arithmetic shift-add. `PureTropicalSGD` keeps even the momentum and
   noise integer (an int counter + an LFSR), so the *entire optimizer* is
   shift/add/compare too.

The only `*` that survives in the C++ is **address arithmetic** on small loop
counters (`c*K`, `l*D*D`, the ring-buffer base, the dilated time index
`t - k*dil`) — index computation, not a data-path multiply. The synthesis
report confirms `DSP = 0`.

These kernels correspond directly to TikiTorch Python:

| HLS kernel family            | TikiTorch equivalent                                                          |
|------------------------------|------------------------------------------------------------------------------|
| tropical attention block     | `tikitorch.nn.TropicalShiftAttention` (hard, `tau=0`)                         |
| dilation / erosion FFN       | `tikitorch.nn.TropicalDilation` -> `tikitorch.nn.TropicalLinear`             |
| morph depthwise conv + proj  | `tikitorch.nn.MorphConv1d` (`DepthwiseMorphConv1d`) + tropical projection     |
| MorphSSM core (`*_morph*`)   | `tikitorch.models.MorphSSM` (`task_type="classification"`, mean-pool head)    |
| sign-SGD weight update       | `tikitorch.optim.SignSGD` / `tikitorch.optim.PureTropicalSGD`                 |
| fused `forge_train_step*`    | one `loss.backward(); opt.step()` over a token window / sequence             |

The workflow is: train and validate in PyTorch with TikiTorch, anneal `tau -> 0`
(soft tropical -> hard tropical), quantize weights to INT16, verify against the
`*_tb.cpp` testbench (§3), then synthesize (§4) and build the AFI (§5). The
*same* weight tensors drive the soft-trained and the hard-deployed forms — no
parameter copying between them (the soft relaxation is just a temperature on the
identical max/min).

---

## 2. What is in `hls/`

```
hls/
  forge_hls_types.hpp                     # data_t=ap_int<16>, acc_t=ap_int<32> under Vitis;
                                          #   stdint fallback (int16_t/int32_t) for plain g++
  forge_kernels/
    # ---- tropical LM (shift-attention + dilation/erosion FFN) ----
    forge_buffer.hpp                      # 1-layer flat int32 buffer layout (D=128, V=128, L=128)
    forge_buffer_2layer.hpp               # 2-layer buffer layout
    forge_buffer_3layer.hpp               # 3-layer buffer layout
    forge_train_step.cpp                  # 1-layer fused fwd+loss+bwd+sign-SGD kernel
    forge_train_step_2layer.cpp           # 2-layer fused training-step kernel
    forge_train_step_3layer.cpp           # 3-layer fused training-step kernel
    forge_train_step_tb.cpp               # 1-layer g++ testbench
    forge_train_step_2layer_tb.cpp        # 2-layer g++ testbench
    forge_train_step_3layer_tb.cpp        # 3-layer g++ testbench
    synth.tcl / synth_2layer.tcl / synth_3layer.tcl   # Vitis HLS csynth scripts

    # ---- MorphSSM core (depthwise morph conv + tropical projection + residual) ----
    forge_buffer_morph.hpp                # small classification config (D=64, L_LAYERS=4, K=4, L_SEQ=64, DIL_CAP=16) buffer
    forge_train_step_morph.cpp            # monolithic morph-core fwd/bwd/update kernel (fast C-sim config)
    forge_train_step_morph_tb.cpp         # morph-core g++ testbench (toy task learns)
    synth_morph.tcl                       # Vitis HLS csynth script (small config)

    # ---- MorphSSM core scaled to PTB-XL ECG (the headline result) ----
    forge_buffer_morph_ecg.hpp            # ECG config: D=128, L_LAYERS=8, K=7, L_SEQ=5000, DIL_CAP=128
    forge_train_step_morph_ecg.cpp        # STREAMING morph-core kernel (DDR-resident big tensors)
    forge_train_step_morph_ecg_tb.cpp     # bit-exact-vs-CPU-reference testbench
    synth_morph_ecg.tcl                   # Vitis HLS csynth script (ECG config)
```

### Per-kernel description

- **`forge_train_step` (1-layer LM).** The original silicon-validated kernel.
  A single fused pass over one window of tokens: embed -> tropical shift
  attention -> dilation/erosion FFN -> head -> cross-entropy loss -> backward
  (argmax-routed sparse scatter) -> sign-SGD update, all writing back into the
  shared flat int32 buffer (`forge_buffer.hpp`). Dispatched by `mode`
  (0 = fwd+loss, 1 = bwd, 2 = sign-SGD) so the host owns the per-window loop.

- **`forge_train_step_2layer` / `forge_train_step_3layer`.** The same fused
  training step stacked to 2 and 3 tropical layers (their own
  `forge_buffer_{2,3}layer.hpp` layouts). The 2-layer kernel was the first
  *multilayer* NN trained end-to-end on FPGA silicon with zero DSPs; the 3-layer
  kernel showed depth saturates at this D=128/V=128 operating point (width is
  the next binding constraint).

- **`forge_train_step_morph` (MorphSSM core, small config).** A monolithic
  single kernel implementing the **morphological** core instead of attention:
  per layer, `batch_shift` mean-subtract normalization -> causal depthwise
  morph conv (alternating max-plus *dilation* on even layers / min-plus
  *erosion* on odd layers, dilation `d_l = min(2^l, cap)`) -> tropical
  projection `y[j] = max_i(P[j,i] + MC[i])` -> residual add; then mean-pool to
  a feature vector. **Hybrid design:** the host owns embed + classification head
  + cross-entropy; the kernel owns the morph core fwd/bwd/update. Backward
  sparse-scatters through the saved conv tap index `k*` and projection index
  `i*`. This is the fast-C-sim companion to the ECG kernel (same math, tiny
  shapes). Maps to `tikitorch.models.MorphSSM` /
  `tikitorch.nn.DepthwiseMorphConv1d`.

- **`forge_train_step_morph_ecg` (MorphSSM core, PTB-XL ECG — headline).**
  The morph core scaled to the manuscript ECG config
  (`D=128, L_LAYERS=8, K=7, L_SEQ=5000, dilation cap 2^7 = 128`). Identical
  math to `forge_train_step_morph`, but **streaming**: the big per-(layer,t)
  tensors (`X`, saved `TAP_IDX`/`PROJ_IDX`, `POOLED`, `GRAD_POOLED`, weights
  `MH`/`MP` and grad accumulators) live in DDR at fixed `OFF_*` offsets and are
  moved with wide-AXI burst load/store, so the on-chip working set stays
  `O(L_LAYERS · D · WPOW)` (a per-layer sliding window) instead of
  `O(L_SEQ · D)`. Forward streams `t = 0..L_SEQ-1`; backward streams in reverse
  `t = L_SEQ-1..0` with a symmetric on-chip grad window (the conv routes a grad
  at output `(c,t)` to input `(c, t - k·dil)` — a *future* timestep under
  reverse iteration). The residual-stream activation *values* are never stored:
  because the core is piecewise-linear, backward routes purely through the saved
  argmax **indices** with coefficient 1. This is the kernel behind the §6
  headline numbers.

---

## 3. Functional verification on the dev host (no FPGA needed)

The `*_tb.cpp` testbenches compile with plain `g++`. The `forge_hls_types.hpp`
shim swaps `ap_int<N>` for `int16_t` / `int32_t` when `__VITIS_HLS__` /
`__SYNTHESIS__` is not defined, so the *exact same kernel source* runs as an
ordinary C++ program. Use this to verify the integer kernel matches the
reference before spending synthesis time:

```bash
cd tikitorch/hardware/hls/forge_kernels

# tropical LM kernels (1 / 2 / 3 layer)
g++ -O2 -std=c++17 -I.. forge_train_step.cpp        forge_train_step_tb.cpp        -o tb1   && ./tb1
g++ -O2 -std=c++17 -I.. forge_train_step_2layer.cpp forge_train_step_2layer_tb.cpp -o tb2   && ./tb2
g++ -O2 -std=c++17 -I.. forge_train_step_3layer.cpp forge_train_step_3layer_tb.cpp -o tb3   && ./tb3

# MorphSSM core (small config — toy task should learn, acc >> chance)
g++ -O2 -std=c++17 -I.. forge_train_step_morph.cpp     forge_train_step_morph_tb.cpp     -o tbm  && ./tbm

# MorphSSM core, ECG config (TB overrides L_SEQ small for a fast BIT-EXACT C-sim)
g++ -O2 -std=c++17 -I.. forge_train_step_morph_ecg.cpp forge_train_step_morph_ecg_tb.cpp -o tbe  && ./tbe
```

Expected: the LM and morph TBs print per-step loss/accuracy and a `PASS`; the
ECG TB checks the streaming forward `POOLED`, the saved indices, and the
backward gradients (`g_MH`, `g_MP`, `grad_X`) **bit-exact against a dense CPU
reference** (`mism=0` on every tensor). Match the LM/morph results against the
PyTorch model run with **hard** tropical ops (`tau=0`, `use_soft=False`) and
INT16-quantized weights.

---

## 4. HLS synthesis (Vitis HLS)

Each kernel has a `synth_*.tcl` targeting the F2 part `xcvu47p-fsvh2892-2L-e`
at a 250 MHz clock (`target_period_ns 4.0`):

```bash
cd tikitorch/hardware/hls/forge_kernels
vitis_hls -f synth.tcl             # 1-layer LM
vitis_hls -f synth_2layer.tcl      # 2-layer LM
vitis_hls -f synth_3layer.tcl      # 3-layer LM
vitis_hls -f synth_morph.tcl       # MorphSSM core, small config
vitis_hls -f synth_morph_ecg.tcl   # MorphSSM core, ECG config (headline)
```

Each script writes
`<proj>/solution1/syn/report/<top>_csynth.rpt`. Check the two things that
matter:

- **DSP = 0** — confirms the multiplier-free closure claim *physically* in the
  utilization table. (Every kernel in this directory synthesizes to 0 DSPs.)
- **WNS (Worst Negative Slack) >= 0** at the target period, or the downstream
  AFI build will fail timing. The ECG kernel closes with **WNS = +0.009 ns**;
  some of the earlier LM builds landed slightly negative (around -0.15 to
  -0.20 ns) and needed an AFI rebuild or two to close — see the lessons in §5.

---

## 5. From kernel to silicon (AWS EC2 F2 / AFI)

High-level path (AWS FPGA HDK + Vitis flow):

1. **Synthesize** the kernel to RTL (§4). The generated Custom Logic (CL) wraps
   the kernel with a 512-bit AXI master (PCIM) plus an AXI-Lite (OCL) register
   block — see the ABI notes below.
2. **Build the AFI.** Package the CL with the HDK `aws_build_dcp_from_cl` flow,
   upload the design tarball to S3, and call `create-fpga-image`. AWS returns an
   **AFI id** and an **AGFI id**.
3. **Load and run.** On the F2 instance, `fpga-load-local-image` the AGFI, then
   drive the kernel from a small C host driver: poke the scalar args / `mode`
   over the OCL AXI-Lite registers, DMA the flat int32 buffer over PCIM, kick
   the kernel, poll done, and loop the window/epoch on the host.

### ABI / wiring notes (carved from the silicon-validated builds)

These are the load-bearing details that make a *working* AFI rather than a hung
one:

- **One flat int32 buffer in device memory.** All tensors (weights, grads,
  activations, tokens, targets, saved indices) live at fixed byte offsets in a
  single int32 buffer — see the `forge_buffer*.hpp` layouts. Each kernel reads
  only the regions it needs and writes only what it produces; the host owns the
  per-window / per-epoch loop.
- **Use PCIM, not the DDR bridge.** Move the buffer over the 512-bit PCIM AXI
  master. `max_widen_bitwidth=512` forces a 512-bit AXI master matching the
  auto-generated CL's hardcoded 512-bit nets (this is why the burst helpers pack
  16 int32 lanes per `ap_uint<512>` word).
- **No multi-kernel mux.** Fuse the whole training step into one `mode`-dispatched
  kernel rather than wiring several kernels behind a selector — the multi-kernel
  mux path does not map cleanly through the CL on this flow.
- **Watch int-return register shifts.** A kernel that returns an `int` shifts the
  OCL AXI-Lite register map; account for the offset when the host reads back the
  status/return value, or you will read the wrong register.
- **Scalar args via `s_axilite`**, the buffer pointer via the wide AXI master.

---

## 6. Reference operating points (measured)

From the silicon-validated builds (your numbers vary with part, clock closure,
and host driver; treat these as the order of magnitude):

| Kernel                          | Config                         | Throughput   | Energy        | DSP | WNS        |
|---------------------------------|--------------------------------|--------------|---------------|-----|------------|
| `forge_train_step` (1-layer LM) | D=128, V=128, L=128            | ~1,200 tok/s | —             | 0   | (closed)   |
| `forge_train_step_2layer`       | 2-layer LM, D=128, V=128       | ~770 tok/s   | ~9.0 mJ/token | 0   | -0.197 ns  |
| `forge_train_step_3layer`       | 3-layer LM, D=128, V=128       | (depth-saturated, ~same as 2-layer) | — | 0 | (closed) |
| **`forge_train_step_morph_ecg`**| **D=128, L=8, K=7, L_SEQ=5000**| **1,207 tok/s** | **5.8 mJ/token** | **0** | **+0.009 ns** |

Notes:

- The **1-layer LM** was the first end-to-end LM training step on F2 silicon
  (~1,200 tok/s with a C host driver, ~15x over the Python-host baseline).
- The **2-layer LM** was the first *multilayer* NN trained end-to-end on FPGA
  with zero DSPs (~770 tok/s, ~9.0 mJ/token, WNS -0.197 ns — it closed timing on
  an AFI rebuild).
- The **morph_ecg** kernel is the manuscript headline: a full MorphSSM training
  step over a 5,000-sample PTB-XL ECG sequence at **1,207 tok/s, 5.8 mJ/token,
  DSP = 0, and WNS = +0.009 ns** (positive slack — timing closes outright).

Every entry has **DSP = 0**: the tropical learning loop is closed, end to end,
with nothing but comparators, adders, and bit-shifts.

---

## 7. Locating these files from Python

After installation the kernels ship as package data. From Python:

```python
import tikitorch.hardware as hw
hw.hls_dir()            # absolute path to the bundled hls/ directory
hw.build_guide_path()   # absolute path to BUILD_GUIDE.md
```

(See also `BUILD_GUIDE.md` in this directory for the condensed
HLS -> AFI -> F2 walkthrough.)
