# Distributed PC training of a TRANSFORMER on TinyShakespeare — the LLM-on-fabric flagship (small scale)

**Status (2026-06-23):** the novel claim (block-sharding) is CPU-de-risked + the kernel is designed; the full
`forge_pc_xformer_block` HLS kernel is the remaining build (the largest single artifact — it composes
attention + layernorm + FFN + PC backward). This doc is the de-risk + design + honest scope.

## The claim

Train a transformer MODEL-PARALLEL across F2 FPGAs: each chip holds **one transformer block's** weights, and
the chips exchange ONLY a neighbour hidden-state (downstream) + a neighbour error (upstream) — no global
gradient, no all-reduce, no barrier. The on-thesis, small-scale realization of the documented flagship
("train a transformer LM entirely on F2 → a ~124M LLM sharded across a fabric"). The per-chip unit is Phase
1's distributed "block", upgraded from a dense layer to a full transformer block.

## What is de-risked (measured on CPU)

Two independent results combine to validate the approach:

1. **PC trains a transformer (already validated, elsewhere):** `charlm_mixer.py --linear mfgrad`
   (operand-quantized `q(gO)ᵀq(x)`) = **2.481 bpb, identical to the dyadic-autograd baseline** — the
   FFN/projection stack trains fully multiplier-free at zero quality cost.

2. **Block-sharding preserves training (this work):** `scripts/pc_xformer_shard_ref.py` trains a small char
   transformer (D=64, 2 heads, 2 blocks, seq=64) two ways — full-precision block boundaries vs **int16-
   quantized neighbour messages at every block boundary** (the exact cross-chip rescale). The curves are
   nearly identical:

   ```
   step    full-precision boundary    int16 neighbour-message boundary (sharded)   bpb
      0          6.279                          6.279   (-0.000)
    200          3.253                          3.250   (-0.003)
    400          2.940                          2.933   (-0.007)
    599          2.674                          2.663   (-0.012)
   ```

   The int16 inter-block messages train the transformer to **within -0.012 bpb** of full-precision (sharded
   marginally better, within noise) — the Phase-1 message-rescale result (sign/scale-invariant under uniform
   rescale) carried to transformer blocks. Distributed block-sharding does not cost accuracy.

Together: PC trains the transformer's stack mult-free, and the cross-chip int16 messages are benign → a
transformer can be trained block-sharded across chips by PC.

## The two honest caveats (carried from the plan)

- **Coordination, not collision.** The single-card transformer trainer (`forge_train_step_s1`) and its kernels
  (`forge_attn`, `forge_layernorm`, `forge_fwd_gemm`, `forge_wgrad_df3`, `forge_dyadic_adam`) are the **parallel
  LLM instance's active flagship** — NOT modified here. The distinct contribution is the **distributed** angle
  (block-sharded PC, neighbour-only inter-block credit). The kernel reuses those as **patterns copied into NEW
  files** (`forge_pc_xformer_block.*`), distinct CL/state/tags.
- **The attention training multiply.** Mult-free attention *deploy* is lossless (L1-score + argmax-gather,
  ~2.54 bpb), but competitive *training* needs the soft `Σ p·V` blend — the **one irreducible multiply**
  (DSP-allowed; strictly-zero-multiply attention training caps at ~3.4 bpb). Honest claim: **the
  FFN/projection/embedding stack is fully multiplier-free PC-trained; attention is warm-trained with the soft
  blend and deploys multiplier-free** → competitive ~2.5 bpb. NOT a fully-multiply-free transformer trainer.

## Kernel design (the remaining build) — `forge_pc_xformer_block.cpp`

ONE transformer block as a distributed shard unit, role (HIDDEN/OUTPUT) + phase (FWD/BWD) + update from a DDR
CFG word (the Phase-1 split applied to a block; one AFI loads on every slot). Reuses the LLM instance's kernel
patterns copied into new files; built off-by-one-correct (`norm_shift = bit_length−target`) from the start.

- **FWD:** LN → attention (L1-score + soft `Σ p·V` blend) → residual → LN → FFN (dyadic GEMM + ReLU) →
  residual; cache LN stats + relu mask + attention probs. (HIDDEN: emit the hidden state downstream; OUTPUT:
  LM-head + CE.)
- **BWD:** build top error (OUTPUT block: CE/LM-head; HIDDEN block: incoming neighbour error e_in) → PC cascade
  backward through FFN + projections (mult-free `q(gO)ᵀq(x)`, the validated recipe) + attention backward (the
  `forge_attn_bwd` soft-blend pattern, the one multiply) + layernorm backward → SignSGD / dyadic-Adam update;
  emit e_down upstream (max-shift int16). Weights persist per chip.
- **DSP_MULT = 0** everywhere except the attention soft-blend (the one irreducible, DSP-allowed multiply).

**Supporting build (clone the Phase-1 pattern):** `cl_forge_pc_xformer_ddr/` (DDR CL clone, kernel
`forge_pc_xformer_block`), `synth_pc_xformer_block{,_csim}.tcl`, `forge_pc_xformer_shard.c` (generalize
`forge_pc_train_shard.c`: per-chip block, neighbour messages = hidden states (fwd) + errors (bwd), LM-head/CE
on the output chip, per-N-step bpb eval, the Gate-C cross-chip audit), `run_pc_xformer_shard_f2.sh`
(f2.12xlarge → f2.48xlarge), `shakespeare_to_bin.py`.

## Staging (each step a result; the 124M LLM is the north star)

1. **2-block char-transformer on TinyShakespeare across 2 F2s** (f2.12xlarge): each chip = one block,
   neighbour-only; bpb falls on-fabric; Gate-C audit shows only hidden-state/error messages cross.
2. **Deeper/wider across f2.48xlarge (8 slots)** → competitive ~2.5 bpb, trained model-parallel on silicon.
3. **The ~124M LLM sharded across a fabric** (multi-box TCP neighbour messages — the roadmap's "true
   no-shared-host messaging"): honest milestone = measured datapath + projected convergence (300B-token
   from-scratch = multi-F2-years; the aspirational endpoint, not a near-term deliverable).

## Build / run (after the kernel is built)

```
KTOP=forge_pc_xformer_block KTCL=synth_pc_xformer_block_csim.tcl bash scripts/run_kernel_csynth.sh launch
CL=cl_forge_pc_xformer_ddr PKG_REL=tikitorch/hardware/hdk/cl_forge_pc_xformer_ddr \
  KERNEL_TCL=synth_pc_xformer_block.tcl KERNEL_PROJ=forge_pc_xformer_block_proj bash scripts/run_pc_afi_build.sh launch
AGFI=agfi-... bash scripts/run_pc_xformer_shard_f2.sh launch
```
