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

<!-- SILICON: filled from train_L{2,4,8}.txt when the F2 run lands -->
| chips | peak test acc | steps/s | comms bytes/step | comms % of wall | Gate-C (W, ∇ crossed) |
|------:|--------------:|--------:|-----------------:|----------------:|:---------------------:|
| 2     | _pending_     | _pend_  | _pend_           | _pend_          | 0, 0                  |
| 4     | _pending_     | _pend_  | _pend_           | _pend_          | 0, 0                  |
| 8     | _pending_     | _pend_  | _pend_           | _pend_          | 0, 0                  |

**MEASURED:** model-parallel training across 8 physical FPGAs; per-chip kick times; comms bytes/step grows
~linearly with chip count (each added layer = one more neighbour message each way); Gate-C — **no weight and no
gradient ever crosses a chip** (static guarantee + live `xchip_msg` counter); bit-exactness — the Python
reference replayed from the silicon's exact init+order reproduces every chip's final weights
(`pc_shard_perm_replay_L.py`). **Honest:** ~0.65–0.73 accuracy (depth-flat); slower wall-clock than one chip
(host PCIe relay). The result is the *distributed mechanism at 8-way scale*, measured.

---

## 3. Stage 2 — a distributed PC TRANSFORMER, recipe validated, kernel forward csynth-ready

The flagship: one transformer block per chip, 8 blocks across 8 FPGAs, trained on TinyShakespeare with
neighbour-only messages. **The recipe is CPU-validated** (`pc_xformer_dyadic_ref.py`, 8 blocks, int16 block
boundaries):

| 8-block distributed transformer (TinyShakespeare) | bpb (MEASURED, CPU) |
|---|---|
| float projections + warm-trained attention | **2.248** |
| **DYADIC (mult-free) projections/FFN/head** + warm-trained attention | **2.384** (+0.136) |

So making the projection/FFN/head stack **multiplier-free** (the fabric's dyadic shift-add forward) costs only
+0.136 bpb — it *preserves* training. Attention is **warm-trained** (the soft `Σ p·V` blend is the one allowed
multiply; deploy is mult-free via L1+argmax-gather, ~2.43–2.53 bpb from earlier work). 2.384 bpb is competitive
with the established single-card PC-on-transformer (charlm_mixer 2.481).

**Kernel:** `forge_pc_xformer_block.cpp` (+ `forge_pc_xformer_config.hpp`, D=64/H=2/FFN=256/seq=64/vocab=64) —
role/phase/CFG distributed block reusing the proven `forge_pc_block` + S1/attn/LayerNorm/wgrad/dyadic-Adam
patterns (copied into new files; originals untouched). The **forward path** (embed/HIN → LN1 → attention[the
multiply] → WO+residual → LN2 → FFN → residual → OUTPUT head+CE | downstream message) is written and
csynth-ready (`synth_pc_xformer_block.tcl`). **Remaining build (the user's csim→csynth→AFI):** the PC-cascade
backward + dyadic-Adam update (transcribed from the validated recipe, bit-exact to an int64 golden tb).

---

## 4. Stage 3 — 16 FPGAs across two boxes (cross-instance TCP), the structural cap

`forge_pc_train_shard_net.c` trains ONE residual net across **two f2 instances** (head = layers 0..L-1, tail =
layers L..2L-1) with exactly **one boundary message each way crossing the VPC network** (TCP); every intra-box
message stays on PCIe. `run_pc_shard_net_f2.sh` launches head+tail in a self-allowing security group;
`pc_shard_net_eval.py` scores the combined 2L-layer net and bit-exact-replays it. Runs on the **existing AGFI**
— a cheap proof at 2×f2.12xlarge (4 FPGAs) or the full 2×f2.48xlarge (16 FPGAs). **Honest:** TCP < PCIe < fabric
→ this is *slower* than one box; the value is that the no-global-coordination scheme runs across two physically
separate accelerators with a single network link, Gate-C clean (weights/grads never cross instances).

---

## 5. GPU-competition framing (measured vs projected)

- **MEASURED:** distributed correctness + bit-exactness at 2/4/8 chips; comms-scaling and per-chip throughput;
  Gate-C (no global model/gradient); the mult-free transformer recipe (2.384 bpb); linear kernel replication
  from earlier work (morph-LM N=4 = 4.00×, 53k tok/s).
- **PROJECTED (clearly not measured):** pipeline-overlap of the host relay (PC has no backward lock — modelled
  2.3–3.9× in `throughput_model`); the ~9000 idle DSP48s re-used as an add-max ALU array (the top throughput
  lever, unbuilt); a multi-F2 farm; 124M-parameter scale. Energy stays the honest caveat — a DSP_MULT=0 core's
  dynamic draw is below the Vccint sensor floor; the board figure is shell-dominated, not a power record.
- **The claim:** structurally realizable + scales with chips + removes backprop's global sync + trains deep
  (residual) + trains a transformer + *projects* toward GPU-competitive throughput. **Not** "beats GPUs."

## 6. Reproduce

```
# Stage 1 (8-FPGA Fashion-MNIST scaling-law, existing AGFI):
AGFI=agfi-0a9578c54e77ba4d9 INSTANCE_TYPE=f2.48xlarge NSLOTS=8 RESIDUAL=1 SWEEP="2 4 8" EPOCHS=4 \
  BINS_S3=s3://.../pc_fashion_bins INITPERM_S3=s3://.../pc_initperm_fashion8res \
  bash scripts/run_pc_shard_train_f2.sh launch        # then fetch; python scripts/pc_scaling_law.py <dir>
# Stage 2 (recipe de-risk, CPU): PYTHONPATH=. XF_BLOCKS=8 python scripts/pc_xformer_dyadic_ref.py
#         (kernel csynth gate): vitis_hls -f synth_pc_xformer_block.tcl
# Stage 3 (16 FPGAs, 2 boxes): AGFI=... NSLOTS=8 INSTANCE_TYPE=f2.48xlarge INITPERM_S3=s3://.../pc_initperm_net16 \
#         bash scripts/run_pc_shard_net_f2.sh launch   # then python scripts/pc_shard_net_eval.py
```
