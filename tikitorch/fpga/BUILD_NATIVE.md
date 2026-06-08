# Building the TikiTorch FPGA native driver (`_libtikifpga`)

`tikitorch.fpga` talks to the silicon-validated kernels through a small
native shared library, `_libtikifpga`. **This library is not shipped in the
wheel** (platform/ABI mismatch, and it needs the Xilinx toolchain to build).
Without it, `tikitorch.fpga` transparently falls back to CPU — everything still
runs, just not on a board. This guide is how to build it and run on AWS F2.

> You cannot build this in a generic sandbox/CI: it requires the bundled HLS
> sources, a C++17 compiler, and (for the board path) the Xilinx XRT runtime and
> an F2 instance. The Python side is complete and tested without it.

## What the loader expects

`tikitorch/fpga/_native.py` `dlopen`s the library and binds these extern-C
symbols (see `_BRIDGE_SYMBOLS` there):

```c
void* tt_fpga_open(const char* xclbin_or_null);          // open device/context
void  tt_fpga_close(void* h);
int   tt_fpga_board_present(void* h);                     // 0/1
void* tt_fpga_alloc(void* h, long words);                // int32 buffer
void  tt_fpga_free(void* h, void* buf);
void  tt_fpga_load_weights(void* h, void* buf, const int16_t* w, long n);
void  tt_fpga_save_weights(void* h, void* buf, int16_t* w, long n);
int   tt_fpga_run(void* h, const char* entry, void* buf, int mode,
                  int a1, int a2, int a3, int layer_id);  // rc==0 on success
```

`tt_fpga_run` dispatches by `entry` (the kernel symbol, e.g.
`forge_train_step_2layer`) and the canonical mode codes
`FWD_LOSS=0, BWD=1, SIGNSGD=2`. The trailing `layer_id` is consumed only by the
6-int (`KernelABI.FUSED_6`) kernels (2-/3-layer LM); the 5-int (`FUSED_5`)
kernels (`forge_train_step`, `forge_train_step_morph[_ecg]`) ignore it.

## The two ABIs and the kernel buffer

The bundled HLS sources (locate them with `tikitorch.hardware.hls_dir()`) export:

| entry | ABI | buffer-words const (`.hpp`) | shape |
|---|---|---|---|
| `forge_train_step` | 5-int | `FT_BUFFER_WORDS` | V=128 D=128 D_FF=512 L=128, 1 layer |
| `forge_train_step_2layer` | 6-int | `FT2_BUFFER_WORDS` | …, 2 layers |
| `forge_train_step_3layer` | 6-int | `FT3_BUFFER_WORDS` | …, 3 layers |
| `forge_train_step_morph` | 5-int | `MD_BUFFER_WORDS` | D=64 L=4 K=4 Lseq=64 |
| `forge_train_step_morph_ecg` | 5-int | `FT_BUFFER_WORDS_E` | D=128 L=8 K=7 Lseq=5000 |

These are exactly the entries seeded into `KernelCache.default()`.

## Build — LOCAL mode (no board, bit-exact validation)

The kernels compile as plain C++ (the `forge_hls_types.hpp` shim maps `ap_int<>`
to `int16_t`/`int32_t` under g++). Wrap them with a `bridge.cpp` that implements
the symbols above against the kernel C functions and a heap buffer:

```bash
HLS=$(python -c "import tikitorch.hardware as h; print(h.hls_dir())")/forge_kernels
g++ -O2 -std=c++17 -fPIC -shared \
    -I"$HLS" \
    bridge.cpp "$HLS"/forge_train_step_2layer.cpp \
    -o _libtikifpga.so
export TIKITORCH_FPGA_LIB=$PWD/_libtikifpga.so
```

LOCAL mode runs the kernel math on the host (no `xrt`), so `board_present()`
returns 1 and you can validate your **on-board runner** (see below) against the
bundled `forge_train_step_*_tb.cpp` testbench — which is the silicon-validated
reference for the fused sequence — before ever touching real silicon.

## Build — XRT mode (AWS F2 silicon)

On an F2 instance with an `.xclbin` in hand, build `bridge.cpp` in XRT mode
(implement the symbols with `xrt::device`/`xrt::kernel`/`xrt::bo`, mirroring the
`host_training_xrt.cpp` flow in the source monorepo — that file is *not*
bundled in this package):

```bash
g++ -O2 -std=c++17 -fPIC -shared -DTT_FPGA_XRT \
    -I/opt/xilinx/xrt/include -I"$HLS" \
    bridge.cpp \
    /opt/xilinx/xrt/lib/libxrt_coreutil.so \
    -o _libtikifpga.so
export TIKITORCH_FPGA_LIB=$PWD/_libtikifpga.so
```

Point `tt_fpga_open` at the `.xclbin` path.

## The last mile: an on-board runner

Transport + ABI are wired. The remaining step is the **on-board runner** — a
registered plug-in that owns the whole on-board step for one kernel: it marshals
the model's (float) parameters into the kernel's INT16 weight order, **drives the
exact fused sequence the kernel's testbench prescribes**, syncs trained weights
back, and returns a loss proxy. This is deliberately one unit because the fused
sequence is *buffer-layout-coupled* (e.g. the K/V cache regions are zeroed
directly in the buffer between the forward and backward phases) — it is not a
generic loop. It is kernel-specific and **none ship by default**; a model is only
ever *bound* (runs on board) once a runner exists for its kernel.

Author the runner to mirror the bundled, silicon-validated testbench for the
kernel exactly. The kernel arg slots are `(buffer, mode, arg1, arg2, arg3[, layer_id])`
with, for FWD/BWD, **arg1 = position index `m`**, **arg2 = input token**, **arg3 =
target** (FWD only); for SIGNSGD, arg1 = `lr`, arg2 = `threshold`
(see `forge_train_step_2layer.cpp` arg docs). For the 2-/3-layer LM kernels
(`forge_train_step_2layer_tb.cpp`): clear caches → forward all positions per layer
(layers `0..n-1`) passing `m` in arg1, the **input token in arg2 for layer 0 only**
(`-1` for later layers), and the **target in arg3 for the final layer** (`-1`
otherwise) → clear caches → backward per layer in reverse (`n-1..0`) over positions
in reverse (same per-layer token-in-arg2-for-layer-0 convention, arg3 = 0) → one
`SIGNSGD(lr, threshold, 0, -1)`. The morph kernels (`forge_train_step_morph*_tb.cpp`)
instead run a single whole-sequence `FWD(0,0,0)` → `BWD(0,0,0)` → `SIGNSGD(lr, th, 0)`
with the sequence pre-staged in the buffer (no per-token loop).

```python
import tikitorch.fpga as fpga

def run_2layer(handle, spec, model, tokens, *, targets, lr, threshold):
    # tokens/targets are the batched (B, L) inputs from train_step.
    # 1) marshal model weights -> kernel INT16 buffer order
    #    [embed, W_head, (s_q, s_k, W1, b1, W2, b2) x n_layers]
    # 2) drive the validated forge_train_step_2layer_tb.cpp sequence via handle
    # 3) sync trained INT16 weights back into `model`; return the loss proxy
    ...

fpga.register_board_runner("forge_train_step_2layer", run_2layer)
```

With a board present, a matching kernel, and a registered runner,
`fpga.accelerate(model).bound` becomes `True` and `train_step` drives the fused
loop on silicon. Validate the runner against LOCAL mode and the testbench first,
then move to XRT. See `tikitorch/hardware/README.md` for the closure→silicon mapping.
