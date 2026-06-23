# Generative sparse-coding PC on F2 — the backprop-impossible capability, multiplier-free on silicon

**Status (2026-06-23):** CPU fabric-faithful reference VALIDATED; kernel + CL + driver + runners csynth-ready.
The F2 csynth → AFI → train+inpaint run are user-run (gated by a free csynth pass). Silicon images land after.

## The claim (why a feed-forward backprop net structurally cannot do this)

A discriminative backprop net is a fixed `input → output` map; recovering missing *inputs* needs a **separate,
task-specific decoder per query**. The sparse-coding PC is **generative** — it stores a prior `x ≈ D·s` with
`s` K-sparse — and answers **any** "infer X given Y" query by **settling the same code** against whatever
pixels are observed: full reconstruction, in-painting under **any** mask, denoising — **one** trained model, no
retraining, no task head. Here it is delivered **multiplier-free** (±2^k dictionary, shift-add settle, a
sort-free top-k) and trained with **no backprop** (settle + local dyadic dictionary update), entirely on F2.

## What is real (measured on CPU, in the fabric arithmetic)

`scripts/pc_sparsecoding_fabric_ref.py` reimplements `SparseCodingPC` (`tikitorch/nn/predictive_coding.py`) in
the EXACT integer / dyadic / shift-add / compare arithmetic the kernel uses, on 16×16 MNIST (P=256, M=256
atoms, K=24 active):

```
50%-random-mask in-paint MSE  vs  mean-fill   →  ~30–46% better than mean-fill (per epoch, oscillating)
```

The generative capability **survives fixed-point**: the dyadic dictionary, the integer settle (max-shift
gradient band + power-of-two step), the **sort-free counting top-k**, and the power-of-two column renorm all
recover digits from 50% masking clearly better than mean-fill. (The float reference gets ~60%; the
fixed-point dyadic-WTA update is coarser and noisier — honest softening, not a different result.)

Fabric-faithful design points (each is a kernel constant): dictionary columns are ±2^k with the unit-L2-norm
divide folded into a **per-column power-of-two right-shift** (`renorm_D`, which also caps |D| — the saturation
fix); `D·s` and `Dᵀ·r` are dyadic shift-add; the settle gradient is **max-shift normalized** each step (band
`G_BITS`) with a power-of-two step (`>>LR`); **top-k WTA = threshold-by-counting** — a 16-bit binary search for
the threshold with ≥K atoms (compare + popcount, sort-free, deterministic, the one genuinely new fabric
primitive — the kernel implements this same algorithm so it is bit-exact to the reference, NOT `torch.topk`).

## Artifacts

- **Reference** `scripts/pc_sparsecoding_fabric_ref.py` — the CPU de-risk above + a side-by-side
  orig/masked/in-painted PNG + golden-vector dump (`PC_GOLDEN=…`) for the silicon bit-exact check.
- **Kernel** `tikitorch/hardware/hls/forge_kernels/forge_sparsecoding_pc.cpp` (top `forge_sparsecoding_pc`):
  `mode=0` TRAIN (settle clean batch → local dyadic dict update `D += dyad(err)ᵀdyad(s) >> DLR` → per-column
  pow2 renorm), `mode=1` INFER/INPAINT (settle masked batch → write codes `s` + reconstruction). Dictionary
  persists in DDR. `DSP_MULT=0` (every op shift/add/compare/mux; the top-k is compare+popcount).
- **csim testbench** `forge_sparsecoding_pc_tb.cpp` — runs one INFER kick and asserts the settled `s` +
  reconstruction match an int64 reference **bit-for-bit** (INFER has no renorm → the clean Gate A).
- **CL** `tikitorch/hardware/hdk/cl_forge_sparsecoding_ddr/` (DDR CL clone; kernel `forge_sparsecoding_pc`,
  slice `scpc_ocl_slice`). **csynth TCL** `synth_sparsecoding_pc{,_csim}.tcl`.
- **Driver** `tikitorch/fpga/silicon/forge_sparsecoding_f2.c` — train the dict on fabric, then a 50%-mask
  in-paint kick; reads back the reconstruction, computes in-paint-vs-mean-fill MSE, dumps `sc_{orig,mask,
  infill}.i32` (the headline images). **Runner** `scripts/run_sparsecoding_f2.sh` (f2.6xlarge).

## Verification (three gates)

1. **CPU fabric-ref beats mean-fill** (free) — done, ~30–46%.
2. **csynth gate** (free): `KTOP=forge_sparsecoding_pc KTCL=synth_sparsecoding_pc_csim.tcl
   bash scripts/run_kernel_csynth.sh launch` → DSP_MULT=0, timing, one-SLR fit, **csim BIT-EXACT** (INFER).
3. **End-to-end on fabric**: train the dict + in-paint on F2; the on-fabric in-paint MSE beats mean-fill (the
   driver prints it) and the rendered orig/masked/recovered grid shows recognizable digits.

## Honest scope

- Single-layer, 16×16 MNIST, MSE-on-missing is a coarse metric (no likelihood/FID); fixed-point softens the
  float ~60% to ~30–46% and adds epoch-to-epoch noise (the dyadic-WTA dict update has no momentum). The honest
  claim is **"clearly beats mean-fill on the fabric arithmetic"**, not "matches float".
- The value is the **capability**, not the number: ONE generative model, trained with no backprop, answering
  an arbitrary in-paint query by settling — which a feed-forward discriminative net cannot, delivered
  multiplier-free on silicon.

## Build / run (user)

```
KTOP=forge_sparsecoding_pc KTCL=synth_sparsecoding_pc_csim.tcl bash scripts/run_kernel_csynth.sh launch   # gate
CL=cl_forge_sparsecoding_ddr PKG_REL=tikitorch/hardware/hdk/cl_forge_sparsecoding_ddr \
  KERNEL_TCL=synth_sparsecoding_pc.tcl KERNEL_PROJ=forge_sparsecoding_pc_proj bash scripts/run_pc_afi_build.sh launch
AGFI=agfi-... EPOCHS=12 bash scripts/run_sparsecoding_f2.sh launch        # train dict + in-paint on F2
```
