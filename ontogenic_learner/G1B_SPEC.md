# G1b — silicon certification of the two-speed learner (turnkey protocol for the fabric instance)

**Status:** spec'd 2026-07-03, ready to execute whenever the fabric instance has a slot.
**Why this gate is the critical path:** the program goal is a continuously learning, low-energy
system **running on fabric**. Every gate so far ran on the CPU/MPS twin; G1b is the bridge —
and it needs **zero kernel rebuild**, because the two-speed learner was designed as a host
policy over per-kick knobs that already exist in the shipped kernel.

## 1. What already exists (code-verified 2026-07-03, worktree /tmp/tt-pc)

| learner primitive | fabric mechanism | where |
|---|---|---|
| fast kick / skip (ε-gate) | per-kick `DO_UP` (apply optimizer or not) | `forge_pc_xformer_mp_block.cpp:439–449` (CFG word) |
| two LRs (wake 1e-3 / sleep 3e-4) | per-kick `LRBITS` (lr = LRBITS·2⁻²⁰), restaged every kick | same CFG word |
| discard-vs-accumulate windows | `GZERO` | same CFG word |
| surprise signal ε | bursted BAR4 readback of grad-accumulator / gRout norms between kicks (or host-side batch CE) | `forge_pc_xformer_mp_shard.cpp:69–87` |
| replay reservoir / sleep batches | tokens are host-staged per kick (`MP_TOK`) | `shard.cpp:308–310` |
| warm-start (phylogeny) | float-master init from host file | `shard.cpp:219` |
| audit | bit-exact spot-replay of any kick vs the CPU twin (~1% overhead, measured) | existing Gate-C machinery |

AGFI: reuse `agfi-0a08046c7d64090ff` (the measured S=2 pipelined bundle). **Isolation:** run on
the PC program's own box per the standing two-box rule; never load onto the S1 instance's F2.

## 2. Protocol (in order; each step gates the next)

1. **Golden-sim replay (CPU, no silicon).** Implement the two-speed driver policy
   (~100–150 lines in the shard host runner: CE-EMA surprise gate → DO_UP; LRBITS schedule;
   host reservoir with surprise-weighted sampling; sleep blocks every K kicks) and run the G1
   protocol at the kernel config (D=64, H=2, FFN=256, SEQ=64, char-64) through the bf16
   golden sim: arms frozen / cascade / rehearsal / two-speed, seeds ≥2.
   **Gate:** the twin verdict reproduces in kernel numerics (two-speed Pareto-beats rehearsal).
2. **One-kick certification (silicon).** Standard one-step gate: a gated kick and an ungated
   (DO_UP=0) kick, bit-exact vs the golden.
3. **Short silicon stream (~500 kicks, one domain shift).** Measure: gate fire-rate, updates
   applied vs budget, wall-time/kick with gating (a skipped kick should cost ~a forward),
   final bpb vs the golden twin (expect within the established noise premium).
4. **The audit artifact.** Spot-replay ≥1% of *applied* updates bit-exact against the twin and
   publish the ledger — "certifiable lifelong learning on deployed silicon," the claim no GPU
   stack can match. This is the paper's fabric section and the beachhead demo.

## 3. Pre-registered go/no-go

**GO:** step 1 verdict reproduces + step 2 bit-exact + step 3 completes with the two-speed
schedule behaving as on the twin (gate selectivity within ±15pp of twin's fire-rate; quality
within the known silicon premium). **Anything else:** the divergence is itself the finding —
localize (numerics vs schedule-timing vs relay) exactly as the batch-size/LR-floor
root-causes were done.

## 4. Effort estimate

Step 1: one session on this instance (golden binary build permitting) or the fabric instance.
Steps 2–4: one fabric session, existing AGFI, ~1–2 F2 spot-hours. No csynth, no AFI build.
