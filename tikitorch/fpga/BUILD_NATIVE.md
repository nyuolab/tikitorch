# Building the TikiTorch FPGA native driver (`_libtikifpga`)

`tikitorch.fpga` is an experimental integration boundary. The wheel does not
ship `_libtikifpga`, a configuration-matched FPGA image, or a safe bundled
training-step runner. Without an explicitly completed binding contract, an
accelerated wrapper uses its ordinary PyTorch fallback.

> A loaded library, a catalog entry, a structural match, and physical FPGA
> execution are different facts. LOCAL is a host-reference endpoint, not a
> board. Neither LOCAL nor a native-library load establishes synthesis, timing,
> resource use, silicon exactness, or performance.

## Catalog inspection is not model matching

The default `KernelCache` records these fixed HLS artifacts:

| entry | ABI | buffer-words constant | fixed recipe |
|---|---|---|---|
| `forge_train_step` | `FUSED_5` | `FT_BUFFER_WORDS` | V=128, D=128, DFF=512, Lmax=128, 1 LM layer |
| `forge_train_step_2layer` | `FUSED_6` | `FT2_BUFFER_WORDS` | same dimensions, 2 LM layers |
| `forge_train_step_3layer` | `FUSED_6` | `FT3_BUFFER_WORDS` | same dimensions, 3 LM layers |
| `forge_train_step_morph` | `FUSED_5` | `MD_BUFFER_WORDS` | D=64, N=4, K=4, L=64, dilation cap 16 |
| `forge_train_step_morph_ecg` | `FUSED_5` | `FT_BUFFER_WORDS_E` | D=128, N=8, K=7, L=5000, dilation cap 32 |

`lookup(model_type, config)` returns a record for a compact catalog key. It does
not inspect a model and is never permission to execute:

```python
import tikitorch.fpga as fpga

cache = fpga.KernelCache.default()
record = cache.lookup(
    "TropicalTransformer",
    fpga.LMKernelConfig(V=128, D=128, n_layers=1),
)
print(record.entry if record else "no catalog record")
```

`match(model)` first performs that lookup and then checks the complete
model-observable structural recipe: secondary dimensions, hard/soft operator
variants, attention configuration, normalization, heads, biases, and named
parameter regions. Ambiguity returns `None`. Stock model constructions currently
fail closed. The one-layer LM entry is also forced to miss because the retired
flat-parameter runner cannot safely marshal its layout.

Even a non-`None` structural match does not cover facts that only a runner and
build manifest can establish:

- runtime tensor shapes, sequence lengths, dtypes, and value ranges;
- learning-rate, threshold, loss, and optimizer-mode contracts;
- named parameter packing, including every bias region, and synchronization back;
- quantization scales, saturation, and retained scale metadata;
- native symbol, ABI arity, buffer-word count, source/image identity, and transport;
- agreement with the functional testbench and the target synthesized image.

## What loading and availability mean

On import, `_native.load()` searches for `_libtikifpga`, calls `dlopen`, and binds
the expected ctypes signatures. It does **not** call `tt_fpga_open`, probe LOCAL,
or probe a board. A returned `NativeHandle` establishes only that the library and
expected symbols loaded.

The later `fpga.is_available()` probe calls the bridge's legacy-named
`tt_fpga_board_present`. LOCAL returns 1 for a usable host endpoint, so
`is_available()` and `device_count()` describe native endpoints, not necessarily
physical FPGA devices.

`AcceleratedModel.bound` becomes true only when the backend has a usable endpoint,
`KernelCache.match(model)` succeeds, and an application explicitly registered a
runner for that entry. It means the wrapper will call that runner. It does not
prove the runner's runtime/build checks or a physical-board path. `strict=True`
only turns a missing binding into `NativeUnavailable` instead of fallback; it
does not strengthen a custom runner.

## Native bridge ABI

The bridge must export:

```c
void* tt_fpga_open(const char* transport_arg_or_null);
void  tt_fpga_close(void* h);
int   tt_fpga_board_present(void* h);  /* endpoint usable; LOCAL may return 1 */
void* tt_fpga_alloc(void* h, long words);
void  tt_fpga_free(void* h, void* buf);
void  tt_fpga_load_weights(void* h, void* buf, const int16_t* w, long n);
void  tt_fpga_save_weights(void* h, void* buf, int16_t* w, long n);
int   tt_fpga_run(void* h, const char* entry, void* buf, int mode,
                  int a1, int a2, int a3, int layer_id);
void  tt_fpga_write_region(void* h, void* buf, long offset_words,
                           const int32_t* src, long n);
void  tt_fpga_read_region(void* h, void* buf, long offset_words,
                          int32_t* dst, long n);
```

The bridge uses canonical modes `FWD_LOSS=0`, `BWD=1`, and `SIGNSGD=2`.
`FUSED_6` entries consume `layer_id`; `FUSED_5` entries ignore it. The current
FPGA-PCI implementation is narrower: it ignores `entry`, `buf`, and `layer_id`
and targets one fixed 5-int AXI-Lite register map and BAR4 convention. It cannot
drive the 2-/3-layer ABI without an extension.

## Build a LOCAL host-reference library

LOCAL needs the bundled HLS sources and a C++17 compiler. The helper builds the
one-layer kernel library and exports its path:

```bash
eval "$(bash scripts/build_fpga_local.sh --export)"
python -m examples.fpga_quickstart
```

The quickstart intentionally demonstrates catalog inspection and fail-closed
fallback. Building LOCAL no longer auto-registers the retired one-layer runner.

The equivalent manual build is:

```bash
HLS=$(python -c "import tikitorch.hardware as h; print(h.hls_dir())")/forge_kernels
g++ -O2 -std=c++17 -fPIC -shared -DCSIM_ONLY -DTT_KERNEL_1LAYER \
    -I"$HLS" \
    tikitorch/fpga/bridge.cpp "$HLS"/forge_train_step.cpp \
    -o _libtikifpga.so
export TIKITORCH_FPGA_LIB=$PWD/_libtikifpga.so
```

Use the matching `-DTT_KERNEL_2LAYER`, `-DTT_KERNEL_3LAYER`,
`-DTT_KERNEL_MORPH`, or `-DTT_KERNEL_MORPH_ECG` define and source for another
artifact. Without `-DCSIM_ONLY` and a `-DTT_KERNEL_*` selection, LOCAL dispatch
has no compiled kernel.

Validate a prospective runner against the corresponding
`forge_train_step_*_tb.cpp`. LOCAL exercises host-compiled kernel math and bridge
marshalling only; it does not exercise routing, PCIe, board state, or silicon.

## Physical-FPGA transports

The archived AWS F2 route uses the AWS FPGA SDK's `fpga_pci`/`fpga_mgmt`
interfaces and the bridge's `-DTT_FPGA_PCI` mode:

```bash
SDK=/path/to/aws-fpga/sdk/userspace
g++ -O2 -std=c++17 -fPIC -shared -DTT_FPGA_PCI \
    -I"$SDK/include" \
    tikitorch/fpga/bridge.cpp \
    -L"$SDK/lib" -lfpga_mgmt -lrt \
    -o _libtikifpga.so
export TIKITORCH_FPGA_LIB=$PWD/_libtikifpga.so
```

This requires an accessible AFI that exactly matches the bridge register map,
buffer layout, runner, and BAR convention. The repository does not ship a public
AFI or guarantee that a historical AGFI is accessible across accounts or regions.

For a supported XRT device and configuration-matched xclbin:

```bash
g++ -O2 -std=c++17 -fPIC -shared -DTT_FPGA_XRT \
    -I/opt/xilinx/xrt/include \
    tikitorch/fpga/bridge.cpp \
    /opt/xilinx/xrt/lib/libxrt_coreutil.so \
    -o _libtikifpga.so
export TIKITORCH_FPGA_LIB=$PWD/_libtikifpga.so
export TT_FPGA_XCLBIN=/path/to/configuration-matched.xclbin
```

The XRT route is an alternate transport, not the FPGA-PCI path used for the
archived F2 records.

## Implementing the runner last mile

No safe runner is bundled or registered. The compatibility function
`register_reference_runners()` raises `NativeUnavailable`; it cannot reactivate
the historical one-layer marshaller, whose generic parameter flattening used the
wrong named-region order and omitted `B1`/`B2` seed/sync.

A custom runner has the signature:

```python
def run_kernel(handle, spec, model, tokens, *, targets, lr, threshold):
    validate_runtime_inputs(tokens, targets, lr, threshold, spec)
    validate_loaded_build_manifest(handle, spec)
    # Allocate spec.buffer_words_const words for the verified build.
    # Pack every parameter by explicit name -> header offset, including biases.
    # Drive the exact testbench sequence and reject every transport error.
    # Read every updated region back by name and preserve quantization metadata.
    return loss_proxy

fpga.register_board_runner("forge_train_step_2layer", run_kernel)
```

Registration is an explicit trust boundary, not validation performed by
TikiTorch. Before enabling `strict=True`, add tests that sentinel-fill every
named parameter region, reject off-by-one runtime lengths and invalid dtypes or
ranges, verify the ABI/buffer manifest, and compare the observable record first
with C-simulation/LOCAL and then with the exact synthesized image.
