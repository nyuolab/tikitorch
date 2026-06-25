# Scaling distributed predictive-coding training: 2 → 8 → 16 FPGAs, and a transformer

**Date:** 2026-06-24. Branch `tropical-predictive-coding`. Builds on the **signature result** (a 2-layer net
trained model-parallel across 2 FPGAs, bit-exact, 0.9114 MNIST — `SIGNATURE_RESULT.md`). This extends the
distributed PC rig toward the flagship goal: *show the potential to someday compete with transformers on GPUs*.
Staged exactly as the user asked — 8-FPGA Fashion-MNIST first, then the 16-FPGA cross-box transformer path.

**The honest north star (stated up front).** Today an A10G runs nanoGPT at ~107k tok/s @ 1.46 mJ/tok; the best
*measured* F2 LM throughput (morph-LM N=4 replicated) is ~53k tok/s = 2× slower. The distributed PC rig is
**host-PCIe-relay-bound** (word-at-a-time readback), so spreading a net over more chips is **structural, not a
wall-clock win**. The claim here is: *mult-free, weight-transport-free, neighbour-only PC training is
structurally realizable on a multi-FPGA fabric, scales with chip count, removes backprop's global sync, and —
with the residual stream — trains deep and trains a transformer.* Every number below is labelled MEASURED or
PROJECTED. This is **not** "beats GPUs."

---

## 1. A real finding: deep dyadic+ReLU cascades wash out — residuals fix it (no rebuild)

Generalizing the 2-chip recipe to an L-chip deep net (`pc_shard_train_ref.step_L`, bit-exact to the 2-layer
`step()` at L=2) surfaced a clean negative result, then its fix:

- **Plain deep MLP collapses at depth ≥ 4.** A cascade of dyadic-matmul → ReLU → max-shift layers **washes out
  the input signal in the forward pass**: at depth 4 and 8, on Fashion-MNIST, the net collapses to chance
  (0.10) for **every** init we tried — random, identity-biased, and a *sticky dyadic-identity* (diagonal = 64 →
  `dyad`=2⁶, which snaps back across [48,96) so SignSGD can't erase it). Freezing the interior layers didn't
  help either: a deep stack of fixed dyadic+ReLU layers still maps everything to a near-constant. (CPU-measured,
  `pc_init_probe.py` / `pc_depth8_probe.py`.) This is a **forward signal-washout**, not just an optimizer issue.

- **A host-side residual skip fixes it — and the kernel never changes.** Because the driver *relays* the
  neighbour activations and errors, it can add a residual itself: forward `out_l = relu_block(in_l) + in_l`;
  backward `grad-wrt-in_l = routed_l + e_out_l` (max-shifted back to int16 — a uniform error rescale is benign
  for SignSGD, the signature's invariance). With the skip, the deep net trains at every depth:

  | depth (chips) | 2 | 3 | 4 | 8 |
  |---|---|---|---|---|
  | plain deep MLP | 0.70 | collapse | **0.10** | **0.10** |
  | **+ residual skip** (CPU) | **0.73** | 0.61 | 0.60 | **0.65–0.73** |

  (Fashion-MNIST 16×16, dyadic MLP, MSE+SignSGD; `residual_scaling_cpu.txt`.) Depth is **accuracy-flat to
  slightly declining** — this simple task doesn't *need* depth — but the net now trains at any depth without
  collapse. This is exactly why the flagship is a **transformer**: its residual stream is the same mechanism,
  and it is what lets depth pay off.

---

## 2. Stage 1 — 8 FPGAs train Fashion-MNIST, model-parallel (silicon, existing AGFI, no rebuild)

One f2.48xlarge, the existing `forge_pc_block` AGFI (`agfi-0a9578c54e77ba4d9`, DSP=0) on all 8 slots, the
residual deep-MLP driver. The **same instance sweeps 2 → 4 → 8 chips** for a silicon scaling-law. Each chip
holds one layer's weights+momentum in its own DDR; only a neighbour activation (down) + error (up) cross; the
host adds the residual skip. Run `pcshard-20260624T230512Z`.

| chips | peak test acc | acc trajectory (4 ep) | steps/s | comms bytes/step | comms % wall | per-chip fwd-kick | Gate-C (W, ∇) |
|------:|--------------:|-----------------------|--------:|-----------------:|-------------:|-------------------|:-------------:|
| 2     | **0.729**     | .658 .639 .707 .729   | 25.8    | 65,536           | 57.4 %       | hidden 8.92ms / out 0.77ms | 0, 0 |
| 4     | **0.730**     | .513 .716 .730 .716   | 9.1     | 196,608          | 60.3 %       | 3×hidden 8.92ms / out 0.77ms | 0, 0 |
| 8     | **0.728**     | .606 .707 .728 .660   | 4.0     | 458,752          | 61.0 %       | 7×hidden 8.92ms / out 0.77ms | 0, 0 |

**MEASURED (run `pcshard-20260624T230512Z`):** model-parallel training across **8 physical FPGAs**. The per-chip
kick times make the split visible on silicon — every hidden chip does a full 256×256 dyadic matmul (8.92 ms),
the output chip does the small 256×10 readout (0.77 ms). Comms grows exactly linearly: 2(L−1) neighbour
messages/step (2→6→14 for L=2→4→8). **Gate-C: no weight and no gradient ever crosses a chip** (static guarantee
+ live `xchip_msg` counter). Energy: per-slot Vccint ~4 W, dynamic at/below the sensor floor (a DSP_MULT=0
core; honest — shell-dominated, not a power record). **Accuracy is depth-flat (~0.73)** — this task doesn't need
depth; the structural achievement (8-way model-parallel, no global gradient) is the result, and the *depth
headroom* the residual unlocked is what the transformer exploits. Throughput honestly drops 25.8→4.0 steps/s
(host word-at-a-time PCIe relay; a chip-to-chip link or DMA-burst readback would collapse the 61 % comms).

**Bit-exactness (`pc_shard_perm_replay_L.py`).** Replaying the single-process reference from the silicon's exact
init+order: **L=2 fully bit-exact** (both chips `max|diff|=0` over 7500 steps). For L=4/L=8, **every chip is
bit-exact except slot-1's *dumped file*** — root-caused (by the adversarial audit) to a **host dump bug**: a
back-compat alias overwrote `shardW1.i32` with the *output* layer `W[L-1]` for L≥4, so the replay compared
chip-1's hidden weights against the output layer's. **The training itself is bit-exact** — all other chips match
AND chip-1's *neighbours* (0 and 2) match exactly, which is only possible if chip-1's forward/backward messages
(hence its weights, via `dyad(W)`) matched the reference at every step. The alias is fixed; a clean re-run
(`pcshard-20260625T004340Z`) confirms all chips bit-exact directly.

---

## 3. Stage 2 — a distributed PC TRANSFORMER: kernel forward csynth-ready, recipe partially de-risked

The flagship: one transformer block per chip, 8 blocks across 8 FPGAs, trained on TinyShakespeare with
neighbour-only messages. A CPU de-risk (`pc_xformer_dyadic_ref.py`, 8 blocks, int16 block boundaries) probes
**ONE axis** of the fabric recipe — pow2 (dyadic) quantization of the projection/FFN/head **weights**, with a
straight-through gradient — inside an otherwise fully-float, AdamW, dot-product-attention PyTorch model:

| 8-block distributed transformer (TinyShakespeare) | final-step single-batch **train** bpb (no held-out eval) |
|---|---|
| float-weight projections + warm-trained dot-attention | 2.248 |
| **pow2 (dyadic) projection/FFN/head WEIGHTS** + same float attention | 2.384 (+0.136) |

**Read this narrowly.** It shows pow2-quantizing the projection/FFN/head *weights* (plus the int16 cross-chip
boundaries) is survivable at +0.136 bpb. It does **NOT** validate the rest of the fabric recipe — the de-risk
still uses float `nn.LayerNorm` (not the kernel's fine-dyadic integer LN), float **dot-product** attention (not
the kernel's L1-score + exp2-LUT softmax — so it tests *nothing* about the kernel's attention math), float
activations everywhere (not int16 + maxshift bands), a **straight-through identity backward** (not the kernel's
integer PC-cascade `q(gO)ᵀq(x)`), and **AdamW** (not mult-free dyadic-Adam). The two bpb are single-batch
final-step *train* losses (noisy — the float twin was 2.166 at step 1400, 2.248 at 1599), not validation bpb,
and are **not** comparable to the established charlm_mixer 2.481 (a val bpb at d256/seq256 with the real integer
backward). **The single hardest, least-tested component — the integer PC-cascade backward + dyadic-Adam — is
exactly what this de-risk does not exercise, and is the primary remaining algorithmic risk** (to be checked
against the int64 golden tb + ideally a CPU integer-backward reference before the AFI build).

**Kernel:** `forge_pc_xformer_block.cpp` (+ `forge_pc_xformer_config.hpp`, D=64/H=2/FFN=256/seq=64/vocab=64) —
role/phase/CFG distributed block reusing the proven `forge_pc_block` + S1/attn/LayerNorm/wgrad/dyadic-Adam
patterns (copied into new files; originals untouched). The **forward path** (embed/HIN → LN1 → attention[the
one multiply] → WO+residual → LN2 → FFN → residual → OUTPUT head+CE | downstream message) is written and the
forward bugs the audits found are fixed.

**The backward is BUILT and the kernel is CSIM-BIT-EXACT to the golden — validated LOCALLY (§6).** A plain-int64
golden of the full block (fwd + PC-cascade backward + mult-free dyadic-Adam) was built and **validated locally**
— it trains a tiny char-LM on TinyShakespeare to ~**4–5× chance** with bounded logits, proving the integer
backward + optimizer are sound. The HLS kernel transcribes it, and a minimal `ap_int` shim (`_apshim/ap_int.h`)
compiles kernel + testbench with plain g++ so the **csim runs locally — and passes at max|diff| = 0** across
every region (activations, the updated weights with ~30k genuinely moved, the whole input-gradient cascade
`EOUT`, + zero-grad and all-ones KATs). Getting there caught **nine bugs invisible on paper**: 3 recipe bugs
(LR rounded steps to zero; decay 32× too strong; LN-gammas must be frozen), a workflow's 6 HLS bugs (two
wouldn't compile; matmul clipped to int16 before rescale; int48 overflowed the attention backward), and 3 the
local csim exposed (`norm_shift` scanned only 47 bits so ~2^55 backward grads left `EOUT` un-rescaled; the decay
drifted negative weights; the OUTPUT-role HIN contract). **Status: csim bit-exact (local, `run_local_csim.sh`);
the EC2 Vitis run (`synth_pc_xformer_block.tcl`) is the authoritative csim + the free csynth gate — next step.**

---

## 4. Stage 3 — 16 FPGAs across two boxes (cross-instance TCP), the structural cap

`forge_pc_train_shard_net.c` trains ONE residual net across **two f2 instances** (head = layers 0..L-1, tail =
layers L..2L-1) with exactly **one boundary message each way crossing the VPC network** (TCP); every intra-box
message stays on PCIe. `run_pc_shard_net_f2.sh` launches head+tail in a self-allowing security group;
`pc_shard_net_eval.py` scores the combined 2L-layer net and bit-exact-replays it. Runs on the **existing AGFI**
— a cheap proof at 2×f2.12xlarge (4 FPGAs) or the full 2×f2.48xlarge (16 FPGAs). The adversarial audit (two
independent passes) found + fixed a **bit-exactness blocker** here before any run: the tail's boundary error
crossing to the head omitted the boundary layer's residual skip + max-shift rescale; it now sends exactly
`mshift(routed_L + e_in_L, EDN_BITS)`, matching `step_L(res=True)`. **Honest:** TCP < PCIe < fabric → this is
*slower* than one box; the value is that the no-global-coordination scheme runs across two physically separate
accelerators with a single network link, Gate-C clean (weights/grads never cross instances). Code is ready; the
2-instance run is the user's to launch (cost-gated like the AFI builds).

---

## 5. GPU-competition framing (measured vs projected)

- **MEASURED:** distributed correctness + bit-exactness at 2/4/8 chips (training bit-exact to the reference);
  comms-scaling and per-chip throughput; Gate-C (no global model/gradient); a *narrow* transformer-recipe
  de-risk (pow2 weights + int16 boundaries survive, +0.136 bpb train-loss — NOT the full mult-free stack or the
  integer backward); linear kernel replication from earlier work (morph-LM N=4 = 4.00×, 53k tok/s).
- **PROJECTED (clearly not measured):** pipeline-overlap of the host relay (PC has no backward lock — modelled
  2.3–3.9× in `throughput_model`); the ~9000 idle DSP48s re-used as an add-max ALU array (the top throughput
  lever, unbuilt); a multi-F2 farm; 124M-parameter scale. Energy stays the honest caveat — a DSP_MULT=0 core's
  dynamic draw is below the Vccint sensor floor; the board figure is shell-dominated, not a power record.
- **The claim:** structurally realizable + scales with chips + removes backprop's global sync + trains deep
  (residual) + trains a transformer + *projects* toward GPU-competitive throughput. **Not** "beats GPUs."

## 6. The transformer kernel backward — designed, golden-validated, transcribed, csim-ready

(Design from the audit workflow; **now implemented** — `forge_pc_xformer_golden{,_block}.hpp` is the locally
validated int64 reference, `forge_pc_xformer_block.cpp`'s BWD phase is its HLS transcription, and
`forge_pc_xformer_block_tb.cpp` is the bit-exact csim testbench.) It mirrors the forward in reverse, reusing three validated
patterns verbatim in arithmetic: `forge_layernorm` (LN-bwd, DSP=0), `forge_attn_bwd` (the only multiplies),
`forge_wgrad_df3` (mult-free `gW = Σ q(gO)ᵀq(x)`), `forge_dyadic_adam` (the optimizer). Stages: **(0)** seed the
top error — OUTPUT: head+LNF backward of the cached Q16 grad_logit; HIDDEN/INPUT: `EIN`; **(1)** FFN-bwd (route
+ relu-mask + two mult-free wgrads); **(2)** LN2-bwd; **(3)** attention-bwd — *the multiplies* `gV/gp/gs` per
head (fed the cached `P`), `gQ/gK` an L1 `sign(Q−K)` subgradient, all WO/QKV routes+wgrads dyadic; **(4)**
LN1-bwd; **(5)** dyadic-Adam RMW over every weight; **(6)** `EOUT = maxshift(gR, EBITS)` (INPUT scatters
`gEmbed`). **DDR-cache check:** all activations the backward needs are already cached **except R′** — the one
mandatory forward change (now added: `XO_RP`); Q/K/V and the LN `c_hat` are recomputed bit-identically from
cached tensors (no extra DDR). **Optimizer:** dyadic-Adam (not SignSGD) — the project's own measurement shows
the attention/shift path needs Adam's adaptivity (dyadic-Adam gather 0.929 > AdamW 0.850; plain SGD stalls).
**Validation:** an int64 golden tb that ports the *integer* arithmetic (not the float de-risk), per-stage
`max|diff|=0` with the multiply outputs (`gQ/gK/gV`) checked first, plus KAT corners (all-ones, single-token,
causal `j>i`, zero-seed). This is the bit-exact bar the 2-FPGA and on-fabric runs met.

## 7. Reproduce

```
# Stage 1 (8-FPGA Fashion-MNIST scaling-law, existing AGFI):
AGFI=agfi-0a9578c54e77ba4d9 INSTANCE_TYPE=f2.48xlarge NSLOTS=8 RESIDUAL=1 SWEEP="2 4 8" EPOCHS=4 \
  BINS_S3=s3://.../pc_fashion_bins INITPERM_S3=s3://.../pc_initperm_fashion8res \
  bash scripts/run_pc_shard_train_f2.sh launch        # then fetch; python scripts/pc_scaling_law.py <dir>
# Stage 2 (recipe de-risk, CPU): PYTHONPATH=. XF_BLOCKS=8 python scripts/pc_xformer_dyadic_ref.py
#         (backward golden, locally trains): cd tikitorch/hardware/hls/forge_kernels &&
#            g++ -O2 -I. forge_pc_xformer_golden_test.cpp -o /tmp/gtest && /tmp/gtest
#         (kernel csim bit-exact + csynth gate): vitis_hls -f synth_pc_xformer_block.tcl
# Stage 3 (16 FPGAs, 2 boxes): AGFI=... NSLOTS=8 INSTANCE_TYPE=f2.48xlarge INITPERM_S3=s3://.../pc_initperm_net16 \
#         bash scripts/run_pc_shard_net_f2.sh launch   # then python scripts/pc_shard_net_eval.py
```
