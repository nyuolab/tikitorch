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

## Session scoping amendment (2026-07-14, pre-registered BEFORE the run)

Executed by the assistant (user delegated). Decisions:
- **Kernel/AGFI:** `agfi-050b219d4090fb8d6` — the morph-LM config-B fused
  train kernel (fwd+CE+bwd+signSGD on fabric), chosen because it is the
  BIT-EXACT-certified one (wcrc KAT vs host golden gates the session). The
  L=2 xformer bundle path is avoided: its known wrong-direction-update
  silicon defect (G1B_RESULT) would violate fairness invariant 2.
- **Arms mapping:** S = repeated mode-0 forward sweeps; SL = repeated mode-3
  fused single-step kicks; SL-gated = serve sweeps + one fused kick per 61
  sweeps (the V4-measured 1.65% life fire-rate, replayed as a duty cycle);
  idle = loaded image, no kicks. 3 interleaved reps x >=10 min.
- **Instrument:** `fpga-describe-local-image -S 0 -M` at 1 Hz — the F2 image
  power/rail telemetry named in this runbook. We report EXACTLY what the
  tool exposes, labeled; if it exposes rails but not a full-card 12V edge,
  the number is reported as rail/image power, NOT board power, and the
  full-card claim is scoped accordingly (this is the honest limit of F2
  telemetry without external instrumentation).
- **GPU baseline:** g5.xlarge (A10G) board power via nvidia-smi at 1 Hz
  during the SAME streams: frozen serve, LoRA-periodic (V3 center cell),
  and online-BP — run by `scripts/_pc_v5gpu_onbox.sh`.
- **Iso-quality note:** the F2 kernel is the small config-B LM (its banked
  quality tier), not the d256 stream model; the cross-substrate tokens/J
  comparison is therefore reported at each substrate's own measured quality
  tier with the capability gap STATED, per the honesty rung. The marginal
  serve->serve+learn multiplier (the N2 silicon half) is within-substrate
  and unaffected.
