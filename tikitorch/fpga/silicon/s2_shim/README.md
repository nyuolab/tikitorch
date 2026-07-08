# s2_shim — bit-faithful CPU reference + silicon-vs-source forward diff for the S2 PC-transformer

This harness **found the D=512/30M F2 training-divergence root cause** (2026-07): the LN dyadic
quantizer used `hls::log2`/`hls::round`, which Vitis HLS **silicon approximates** (giving
`s = min(quant(scale), scale)` — up to 11% per-token LN error at `S2_MBITS=2`), while every
csim/shim check mapped those intrinsics to exact libm and saw nothing. Fixed by bit-exact
IEEE-754 integer extraction in the kernel (`s2_ilog2_floor/round`, `s2_dyad_quant`,
`s2_round_ne`), confirmed by this harness: silicon forward became **byte-for-byte identical**
to the shim at every stage. Full story: `results/predictive_coding/S2_30M_ONFABRIC.md`.

## Contents
- `s2_twochip_shim.cpp` — drives the REAL kernel (`forge_pc_xformer_s2_block.cpp`, compiled
  locally via the shims) as chip0(HID + host-embed) + chip1(OUT) for one training step from a
  byte-identical matched init. `FWD_DUMP=1` dumps every forward stage
  (R,X1,Q,K,V,ATTN,M,X2,HD,RP,+XF,LG) as `fwd_<chip>_<stage>.f32`.
- `_apshim/ap_int.h`, `_mpshim/hls_math.h` — minimal shims so the unmodified kernel compiles
  with g++. **Blind spot:** `hls_math.h` maps `hls::exp/log2/round` to exact libm; silicon
  approximates them. The kernel's dyadic paths are integer-exact now; the remaining
  shim-blind intrinsic is softmax `hls::exp`.
- `build_and_run.sh` — compile + one-step run. `SHIM_DEFS="-DS2_K=... -DS2_VOCAB=..."` **must
  match the AFI's build flags** (changing `S2_K` changes the reduction-tree order).
- `compare_fwd.py <SIL_DIR> <SHIM_DIR>` — per-stage relerr/cosine/bitmatch table; flags the
  FIRST diverging stage.
- `analyze_delta.py` — per-tensor weight-delta cosine/sign-agreement (step-1 update compare).
- `SILICON_CMD.sh` — the exact F2 launch recipe for the matching silicon FWD_DUMP
  (NL=2, STEPS=1, matched `INIT_S3`/`ENWIK_S3`).

## The method (reusable for any silicon-vs-source numeric bug)
1. Byte-identical init + fixed data window on both sides (S3: `s2_locinit{,_data}`).
2. `FWD_DUMP=1` on silicon (SILICON_CMD.sh) and here (`build_and_run.sh`).
3. `compare_fwd.py` → first diverging stage.
4. **Per-token ratio analysis** on that stage: if a per-ROW scalar fit has ~zero residual, the
   bug is a per-token normalizer (LN 1/std class); reverse-engineer the closed form against
   the intended math — the shim CANNOT see hls:: intrinsic approximations, silicon-vs-math can.

## Verdict thresholds (measured 2026-07)
- reduction-order fp-noise floor (same math, different tree order): X1 ~5e-8, ATTN ~5e-4
- the old LN bug: X1 relerr 5.4e-2 (≈1000× above floor)
- post-fix silicon: 0.00 every stage (100% bitmatch)
