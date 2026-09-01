# TikiTorch Hardware: HLS Kernels and the FPGA / Silicon Path

This directory ships **High-Level Synthesis (HLS) C++ kernels** for specific
hard, quantized TikiTorch recipes, together with Vitis HLS scripts, host-side
testbenches, and a guide to taking those kernels to **silicon** on an AWS EC2
**F2** instance (AFI = Amazon FPGA Image).

Selected hard recipes can express their declared forward, routed reverse, and
dyadic update using **compare, add, and bit-shift** operations. Closure is a
property of that complete recipe and executed path, not of every TikiTorch
operator. Soft routing, ordinary losses or heads, host arithmetic, and other
optimizer settings can leave the closed subset. A configuration-matched HLS
operator/resource report and routed implementation are therefore the physical
evidence for a compiled kernel; software structure alone does not prove DSP
utilization.

> **Historical PTB-XL MorphSSM-core operating point
> (`forge_train_step_morph_ecg`):** **1,207 tok/s**, a **derived 5.8 mJ/token
> estimate** (not measured full-card energy), **DSP = 0** for the reported
> build, and **WNS = +0.009 ns** (positive slack at the 250 MHz target). This is
> configuration-specific core evidence; the host-owned embed, head, and loss
> remain outside that kernel. See §6 for the preserved record and its limits.

---

## 1. Recipe-scoped closure -> hardware mapping

A conventional NN training step is dominated by `y = W @ x` and its transpose
in the backward pass. In the supported hard kernel recipes below, the
corresponding data path uses tropical reductions and selected-gradient routing:

| Conventional (ring `(+, ×)`) | Tropical (`(max, +)` / `(min, +)`)        | FPGA primitive            |
|------------------------------|-------------------------------------------|---------------------------|
| `sum_k W[i,k] * x[k]`        | `max_k (W[i,k] + x[k])` (dilation)        | adder + comparator tree   |
| `sum_k W[i,k] * x[k]`        | `min_k (W[i,k] + x[k])` (erosion)         | adder + comparator tree   |
| `dL/dW = x ⊗ δ` (dense)      | scatter `δ` to the **argmax/argmin** tap  | index store + add (coeff 1)|
| `W -= lr * grad`             | `W -= lr * sign(grad)`, `lr = 2^-s`       | sign bit + bit-shift add  |

For those declared recipes, three choices avoid data-dependent multiplication:

1. **Forward.** A tropical "matmul" is a tree of adders feeding a tree of
   max/min comparators, so the algorithm has no partial products. The synthesis
   report, rather than the source-level algebra, determines whether a particular
   FPGA build uses DSP blocks for any arithmetic.
2. **Backward.** Because max-plus / min-plus is piecewise-linear, the local
   Jacobian is a **selection matrix**: the gradient at an output routes, with
   coefficient exactly 1, to the single input that won the max/min. The kernels
   *save the winning index* (`argmax`/`argmin`) on the forward pass and, on the
   backward pass, **sparse-scatter** the incoming gradient to that index — an
   indexed add, never a multiply.
3. **Update.** These recipes update weights with **sign-SGD**
   (`W -= lr·sign(grad)`).
   With a power-of-two learning rate `lr = 2^-s`, the update is a sign test
   plus an arithmetic shift-add. `PureTropicalSGD` keeps even the momentum and
   noise integer (an int counter + an LFSR), giving a shift/add/compare update
   path for the corresponding non-adaptive configuration.

In the listed kernel sources, remaining `*` expressions are intended for
**address arithmetic** on loop counters (`c*K`, `l*D*D`, the ring-buffer base,
and related index calculations), rather than model-data products. Confirm that
intent for each exact configuration in its HLS operator and utilization reports;
a report from one configuration does not transfer to another.

These kernels correspond directly to TikiTorch Python:

| HLS kernel family            | TikiTorch equivalent                                                          |
|------------------------------|------------------------------------------------------------------------------|
| tropical attention block     | `tikitorch.nn.TropicalShiftAttention` (hard, `tau=0`)                         |
| dilation / erosion FFN       | `tikitorch.nn.TropicalDilation` -> `tikitorch.nn.TropicalLinear`             |
| morph depthwise conv + proj  | `tikitorch.nn.MorphConv1d` (`DepthwiseMorphConv1d`) + tropical projection     |
| MorphSSM core (`*_morph*`)   | `tikitorch.models.MorphSSM` (`task_type="classification"`, mean-pool head)    |
| sign-SGD weight update       | `tikitorch.optim.SignSGD` / `tikitorch.optim.PureTropicalSGD`                 |
| fused `forge_train_step*`    | the corresponding recipe-specific reverse/update over a token window / sequence |

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

    # ---- MorphSSM core scaled to the historical PTB-XL ECG reference ----
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
  training-step structure stacked to 2 and 3 tropical layers (their own
  `forge_buffer_{2,3}layer.hpp` layouts). The archived 2-layer configuration
  ran training updates on FPGA and its matching record reports DSP = 0; the
  3-layer experiment showed depth saturation at this D=128/V=128 operating
  point. These observations do not establish priority or generalize beyond the
  listed configurations.

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

- **`forge_train_step_morph_ecg` (MorphSSM core, PTB-XL ECG reference).**
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
  argmax **indices** with coefficient 1. This is the kernel behind the
  historical §6 operating point.

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
vitis_hls -f synth_morph_ecg.tcl   # MorphSSM core, ECG reference config
```

Each script writes
`<proj>/solution1/syn/report/<top>_csynth.rpt`. Inspect the HLS operator and
resource tables for the exact source, flags, and dimensions being built. A
reported **DSP = 0** supports only that compiled datapath; it is not a proof
that every kernel, TikiTorch operator, or host-orchestrated training loop uses
zero DSP blocks.

Timing closure is established separately by the downstream routed report:
**WNS (Worst Negative Slack) must be >= 0** at the stated target. The archived
ECG configuration reports **WNS = +0.009 ns**. Earlier LM artifacts around
-0.15 to -0.20 ns miss the target and must not be called timing-closed. A
different rebuild closes timing only if its own nonnegative routed report is
preserved and cited; closure cannot be inferred from a negative-WNS artifact.

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

## 6. Historical reference records (mixed measured and derived values)

The table preserves earlier configuration-specific operating points. Throughput,
reported utilization, and numeric WNS entries are historical observations;
energy entries are derived estimates rather than measured full-card energy.
Your numbers will vary with part, clock closure, and host driver, so treat these
as order-of-magnitude context rather than release guarantees.

| Kernel                          | Config                         | Throughput   | Derived energy estimate | DSP in listed build | Routed WNS |
|---------------------------------|--------------------------------|--------------|-------------------------|---------------------|------------|
| `forge_train_step` (1-layer LM) | D=128, V=128, L=128            | ~1,200 tok/s | —                       | 0                   | not listed here |
| `forge_train_step_2layer`       | 2-layer LM, D=128, V=128       | ~770 tok/s   | ~9.0 mJ/token           | 0                   | -0.197 ns (miss) |
| `forge_train_step_3layer`       | 3-layer LM, D=128, V=128       | (depth-saturated, ~same as 2-layer) | — | 0 | not listed here |
| **`forge_train_step_morph_ecg`**| **D=128, L=8, K=7, L_SEQ=5000**| **1,207 tok/s** | **5.8 mJ/token** | **0** | **+0.009 ns (pass)** |

Notes:

- The historical **1-layer LM** run reports ~1,200 tok/s with a C host driver
  and ~15x relative to its Python-host baseline. That comparison is specific to
  those two host paths, not a general accelerator speedup or priority claim.
- The listed **2-layer LM** build reports ~770 tok/s and DSP = 0. Its
  ~9.0 mJ/token value is derived, and **WNS = -0.197 ns means this artifact
  missed timing**. A claim about any later rebuild requires that rebuild's own
  routed report.
- The **morph_ecg core** record reports a 5,000-sample PTB-XL configuration at
  **1,207 tok/s**, a **derived 5.8 mJ/token estimate**, **DSP = 0** for the
  listed build, and **WNS = +0.009 ns**. It covers the fused morph-core
  forward/reverse/update; embed, classification head, and loss are host-owned.

All four rows record **DSP = 0 for the listed compiled configurations**. That
does not establish zero DSPs for unlisted builds or multiplier-free closure for
their host boundaries, other recipes, or TikiTorch as a whole.

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
