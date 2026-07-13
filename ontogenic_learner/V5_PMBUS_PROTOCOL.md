# V5 — silicon energy protocol (full-card PMBus, serve vs serve+learn on F2)

Pre-registered measurement protocol for the north-star capstone
(NORTHSTAR_VALIDATION.md gate V5 = strategy note M5). **User-run session**,
scheduled after V0–V2 pass (V0 ✅, V1 ✅, V2 in flight). This document is the
runbook; fill the RESULT blocks during the session.

## What is being measured (and what is explicitly NOT)

The **marginal board-level energy of learning while serving**: full-card
mJ/token for (a) forward-only serving vs (b) the fused serve+learn kick, same
board, same checkpoint, same tokens. NOT core-rail Vccint (the ~4W / ~0.3
mJ/tok figures are disclaimed as not-a-board-claim in fig_hallmark_lm.md);
NOT a comparison against a frozen GPU (strawman in our favor — the incumbent
is an ADAPTING GPU, i.e. LoRA-periodic).

## Fairness invariants (all five must hold or the number doesn't count)

1. **Full-card power** (PMBus / `fpga-describe-local-image-slots` power query
   covering shell + DDR + transceivers), sampled ≥1 Hz for the whole window.
2. **Iso-quality**: both arms run the SAME checkpoint at its measured bpb
   (record it; must be ≥ trigram quality).
3. **Same tokens**: identical token stream, identical count (≥ 10 min of
   sustained throughput per arm; report tokens served).
4. **Marginal delta reported**: mJ/tok(serve+learn) − mJ/tok(serve), plus the
   amortized figure at a measured gate-fire rate (V0/V2 fire-rates; G1b's
   silicon precedent: 72.8% fire with 136 kicks never issued).
5. **Adapting GPU baseline**: GPU-LoRA on A10G (board power via nvidia-smi,
   1.46 mJ/tok serving anchor already measured in results/lm_gpu_baseline),
   same stream/tokens/quality-tier.

## Arm matrix (per board, in one session)

| arm | AGFI | knobs | measure |
|---|---|---|---|
| S (serve) | forward-only morph-LM AGFI (exists; fig_hallmark_lm.md) | DO_UP/learn disabled | W(t), tok/s → mJ/tok_S |
| SL (serve+learn, all-fire) | fused-train AGFI (fwd+CE+bwd+signSGD one kick; exists) | learn enabled every kick | mJ/tok_SL |
| SL-gated | fused-train AGFI + host gate (G1b machinery) | gate at V2's measured z* | mJ/tok_gated (should ≈ S + fire_rate×(SL−S)) |
| idle | either AGFI loaded, no kicks | — | board idle floor W_0 (subtract-reported AND raw-reported) |

## Procedure (per arm)

1. Load AGFI; verify bit-exact kick vs host golden (existing KAT).
2. Warm 2 min. Start power logger (1 Hz minimum, timestamped).
3. Run ≥10 min sustained at max sustainable kick rate; log tokens + kicks.
4. Stop; record: mean W, tok/s, tokens, kicks, mJ/tok = mean_W / tok_per_s.
5. Repeat ×3 (report mean ± spread).

RESULT S: ______ W, ______ tok/s, ______ mJ/tok
RESULT SL: ______ W, ______ tok/s, ______ mJ/tok
RESULT SL-gated: ______ W, ______ tok/s, ______ mJ/tok, fire-rate ______
RESULT idle: ______ W

## Derived numbers (the deliverables)

- **Marginal learning energy**: mJ/tok_SL − mJ/tok_S (and ×fire-rate amortized).
- **The on-silicon multiplier**: mJ/tok_SL ÷ mJ/tok_S — compare with V0's
  op-volume prediction (1.38×) and wall prediction (1.23×).
- **tokens/J** serve and serve+learn, vs GPU-LoRA board tokens/J at matched
  tokens/quality-tier.
- Calibration constant for the off-silicon op-count energy proxy
  (scripts/lm_energy_projection.py) — back-propagated into the V3 jsonl.

## Kill / honest-negative (pre-registered)

No measurable S→SL delta (instrumentation floor) ⇒ report the floor, not a
win. Board-joules(SL) ≥ GPU-LoRA board-joules at matched tokens+quality ⇒ the
energy claim fails at board level and the number feeds the ASIC study as the
FPGA-tax input (the card is EXPECTED to lose board power; that loss is an
input, not a defeat).
