# S1 — (A) full-backbone TRAIN_ALL on silicon + (B) do_opt write-wall characterized

**Date:** 2026-06-26 · **AFI:** agfi-0179e963c146d39f7 (v17) · **Instance:** f2.6xlarge
**Driver commit:** 9f04103 · **Raw log:** [`s1_hostopt_trainall_train.txt`](s1_hostopt_trainall_train.txt)

One F2 run, `REGIME=trainall`, did two things: (A) trained the *full multiplier-free backbone* on real-silicon
FPGA gradients via the host-side optimizer, and (B) ran a precise diagnostic of the on-fabric `do_opt` write wall.

## (A) Full-backbone TRAIN_ALL — measured on silicon

```
INITIAL  CE = 4.1556   acc = 3/64
step  20: CE = 3.7284   acc = 8/64
step 100: CE = 2.5622   acc = 8/64
step 200: CE = 2.1582   acc = 8/64
FINAL    CE = 2.1582    acc = 8/64    peak|R| = 311
CODE backbone: 17385 / 49344 code bytes moved
CE 4.1556 -> 2.1582  (drop 1.9975 nats; floor 4.1589)   acc 3 -> 8/64
[gates] CE-drops>=0.5 · below-floor · acc-improves · no-saturation  → all pass
```

**What's new vs the FREEZE run:** the host optimizer now steps **every** learnable region — including the dyadic
**CODE** backbone (Wq/Wk/Wv/Wo/W_up/W_down + LN gammas) via the `encode_fine` code path — on FPGA-computed
gradients. **17,385 of 49,344 live code bytes moved on real silicon**: the full multiplier-free transformer
backbone's optimizer ran on hardware, not just the embed/head value path. No saturation collapse (peak|R|=311 «
32767) — the LN_F + code-weight-decay fixes hold at D=64. 144.7 ms/step (restages all 14 live regions vs 2 for
FREEZE).

**Cross-validation (silicon == the csim gate, near-exact):** the g++ `s1_converge_gate` (drives the real kernel)
predicted CE 4.158→2.169 with codes_moved ~17,376; silicon delivered CE 4.156→**2.158** with codes_moved
**17,385**. The host optimizer (bit-exact to the kernel `#else` path) reproduces the csim trajectory on hardware.

**HONEST CAVEAT — TRAIN_ALL does NOT beat FREEZE at D=64.** acc stays at chance (~8/64 = 1/8) the whole run while
CE drops to ≈log(8)=2.08. The dense backbone learns the **symbol marginal** (only 8 of 64 symbols occur), **not
the successor mapping** that FREEZE's clean residual-identity path learns (FREEZE reached acc 64/64). This is robust
across init densities (2–35%, all acc-chance in csim) and is consistent with the banked verdict: the dense-backbone
*quality* win needs **scale**, not a D=64/1-layer toy task. **The result here is a capability measurement — the
full mult-free backbone optimizer runs end-to-end on real-silicon FPGA gradients (codes move, CE drops, no
collapse) — not a claim that dense > freeze on this task.** Lifting acc at scale is exactly what S2 + the integer-
training recipe work (stochastic rounding, etc.) targets.

## (B) do_opt write-wall — precisely characterized

Sentinel probe (stage known `wfp=100, m=0, v=0, dst=12345` into EMBED; kick the kernel `do_opt(STEP)`; read back):

```
wfp_master : staged=100   got=100   pred=99   -> UNCHANGED (dropped)
m          : staged=0     got=0     pred=32   -> UNCHANGED (dropped)
v          : staged=0     got=0     pred=32   -> UNCHANGED (dropped)
dst_live   : staged=12345 got=12345 pred=99   -> UNCHANGED (dropped)
VERDICT: do_opt STEP drops ALL m_axi writes (wfp/m/v/dst all UNCHANGED) -> kernel m_axi STEP path dead on silicon
```

**This overturns the earlier hypothesis.** The wall is **not** a cross-cast/RAW-dependency dst-only drop (the prior
theory: the scheduler drops the one store whose only reader is a different-cast future kick). **ALL FOUR** do_opt
writes — including the in-instance state slots wfp/m/v — drop on silicon. So the entire `do_opt` STEP **m_axi write
channel is dead on hardware**, while `do_fwd_loss`/`do_bwd` writes on the *same* gmem bundle commit (forward runs;
grads read back `max|g_W_HEAD|=507904`). The differentiator is that `do_opt` delegates its writes to the
`dyadic_adam_region` sub-function — the **3rd `#pragma HLS ALLOCATION instances=1`-shared** re-entry of the m_axi
write path.

**Refined fix hypothesis for a fully-on-fabric optimizer (kernel rebuild, gated — user runs builds):** the shared
`limit=1` m_axi writeback for `do_opt` either (a) never drains through the real shell+CDC+crossbar+sh_ddr before
AP_DONE (no write-response/B-channel fence on that path), or (b) the 3-way ALLOCATION muxing of the write channel
breaks the `do_opt` write on the physical interconnect. The fix: **give `do_opt` its own non-shared m_axi write
path** (un-share `dyadic_adam_region`'s writeback from the fwd/bwd gmem instance) and/or add an explicit
write-response fence so AP_DONE gates on the burst landing. cosim does not catch this (behavioral-AXI fidelity); it
needs the real interconnect — so verify with a host probe like this one after any rebuild, not cosim alone.

## Bottom line

- **Banked, measured on silicon:** the *complete* multiplier-free transformer training loop — FPGA forward + FPGA
  backward + dyadic-Adam over the **full backbone** (codes + values) on FPGA-computed gradients — runs on real F2
  and reduces CE 2 nats with no saturation; 17,385 backbone code bytes trained on hardware. (Optimizer host-side;
  FREEZE remains the stronger *task* result at D=64.)
- **The on-fabric `do_opt` wall is now precisely characterized:** all writes drop (not just the dst) → it's the
  shared m_axi write channel, fixable by un-sharing it + a write-response fence in a kernel rebuild. This is the
  concrete next step for the fully-on-fabric single-kick flagship.
