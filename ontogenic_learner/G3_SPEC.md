# G3 — fleet consolidation: merged-beats-solo (pre-registered spec, v1)

**Status:** spec'd 2026-07-03, frozen before any run. **Owner:** this instance (local MPS;
silicon certification of the winning config is the fabric follow-up, G3b).

## 1. Question and stakes

The paradigm's performance endgame is **fleet-experience scaling**: N deployed learners, each
living its own stream, periodically consolidating — the fleet is the training run, and only
weights ever move (never data: privacy + bandwidth + the situated-data moat). G3 tests the
minimal mechanism: **two ontogenic learners on disjoint domain shards + DiLoCo-style outer
weight averaging. Does the merged model hold BOTH competences?** If yes, the fleet mechanism
is real at the twin level and G3b (silicon) + scaling (N>2) are licensed. If naive averaging
destroys competence, the amendment path is smarter merging (outer momentum, Fisher-weighted).

## 2. Design

Scale: the G2a configuration (hidden 768, depth 8, ~4.1M params, MPS). Phylogeny: shared wiki
checkpoint. Shards: **learner A lives on Shakespeare, learner B lives on code** (maximally
distant domains). Every learner runs the **two-speed schedule** (the 3×-validated winner:
ε-gate z>0.25, fast lr 1e-3 ≤450, sleep every 150 × 15 @ 3e-4, surprise-weighted reservoir;
reservoirs and optimizer moments stay LOCAL — only weights are exchanged at merges).

Arms (seeds {0,1}):
| arm | description | budget |
|---|---|---|
| `solo_A` / `solo_B` | one learner, its shard only | 600 each |
| `fleet_m150` / `fleet_m500` | A+B, outer averaging every 150 / 500 steps (+ final merge) | 600+600 |
| `fleet_final` | A+B, single merge at the end (ablation: does periodic merging matter?) | 600+600 |
| `pooled` | ORACLE: one learner on the interleaved union (centralization) | 1200 |

Total learner-compute is matched between the fleet (2×600) and the oracle (1200).

## 3. Metrics

Final-model bpc on held-out A, held-out B, and wiki (base retention); comms ledger (MB of
weights exchanged vs MB of raw shard data that never moved); per-merge transfer trajectories.

## 4. Pre-registered go/no-go (frozen 2026-07-03)

- **GO:** for the best merge schedule, the final merged model
  (i) **retains home competence**: bpc(A) ≤ solo_A's bpc(A) + 0.05 AND bpc(B) ≤ solo_B's
  bpc(B) + 0.05; and (ii) **gains away competence through weight-space only**:
  beats each solo on its away shard by ≥ 0.30 bpc. Secondary: mean gap to the pooled oracle
  ≤ 0.15 bpc; wiki drift no worse than the solos'.
- **PARTIAL:** home competences retained only at one merge frequency, or oracle gap > 0.15 →
  fleet real but schedule-sensitive; sweep before G3b.
- **NO-GO:** averaging costs > 0.15 bpc on BOTH home shards at every frequency → naive
  averaging insufficient; amendment to smarter merging before any fleet claim.

## 5. Execution log

- [x] Harness `experiments/g3_fleet.py` (two-speed policy as fleet learner; weights-only
      merges; local reservoirs/EMAs/moments; comms ledger). Smoke (tiny budgets): shape
      correct — fleet 3.919/4.453 sits between solos (3.764/5.704, 4.272/4.290) and the
      pooled oracle (3.934/4.156); away-domain gain ≈ 1.25 bpc through weight-space alone.
- [x] Full run (`g3_full.jsonl`, 6 arms × 2 seeds): **PARTIAL** per §4. Mean finals
      (A=shak / B=code / wiki): solo_A 2.903/7.079/3.00 · solo_B 4.254/3.043/3.25 ·
      fleet_m150 3.092/3.727/2.89 · fleet_m500 3.140/3.599/2.89 · fleet_final
      3.317/4.214/2.78 · pooled 3.184/3.020/3.09. Away-transfer through weight-space is
      large (−1.16 / −3.35 bpc vs the solos' away scores) and periodic ≫ final-only
      merging; but the +0.05 home-retention bar fails — **and the pooled oracle itself
      fails it** (A: 3.184 vs specialist 2.903): at 4.1M, dual-competence has an
      irreducible capacity cost the original bar ignored.

## 6. Amendment A1 (logged 2026-07-03, before any further runs)

1. **Recalibrated primary criterion** — the honest reference for a *single shared model* is
   the centralized pooled oracle, not the specialist: GO iff a fleet variant's
   mean-over-shards bpc is within **0.15** of pooled's, AND beats both solos on
   mean-over-shards, AND its per-shard home cost ≤ pooled's home cost + 0.05.
   (Current best, fleet_m500: mean 3.370 vs pooled 3.102 → gap 0.27 — fails as-is.)
2. **New mechanism arm, two-speed-native:** `fleet_mc{150,500}` — **merge-then-consolidate**:
   after every outer merge, each learner runs one sleep block (15 slow-LR kicks on its LOCAL
   surprise-weighted reservoir) before resuming its stream — re-anchoring the merged weights
   to local experience (biologically: sleep after learning from others). Update-budget
   overhead (~+270 updates/learner-pair) logged and reported next to pooled's 1,200.

## 7. Final verdict (2026-07-03): PARTIAL — mechanism real, capacity-limited

Amendment run (`g3_full_mc.jsonl`): **merge-then-consolidate is a null** — fleet_mc500
3.136/3.596/2.888 ≡ fleet_m500 3.140/3.599/2.89. The residual gap to centralization is
*capacity*, not merge-anchoring (the pooled oracle shows the same dual-specialist cost).

Against the amended criterion: fleet_mc500 **beats both solos as a single model** (mean 3.366
vs 3.65/4.99 ✓), **beats the pooled oracle on shard A** (3.136 vs 3.184 ✓) and on base
retention (wiki 2.888 — best of ALL arms including the oracle ✓), but concedes the distant
shard (code 3.596 vs 3.020) → oracle gap 0.264 > 0.15 ✗.

**What is established:** competence transfers through weight-space alone (−1.2/−3.4 bpc away
gains, zero raw-data exchange); periodic merging ≫ one final merge; the merged model is the
best generalist in the experiment. **What is not:** specialist-parity at 4.1M with naive
averaging. **Forward path (new work, properly gated):** (a) capacity scaling — does the
fleet–oracle gap shrink with model size (paper-grade scaling question); (b) structured
merging that preserves specialist subspaces (Fisher/TIES-style, or modular
experts-per-domain — connects to the tropical-MoE / hot-add-a-chip thread). G3b silicon
certification proceeds with the validated mechanism when the fabric instance has a slot.

## 8. Follow-up F1 — capacity scaling of the fleet–oracle gap (pre-registered 2026-07-03)

Sizes: **~0.9M** (hidden 384, depth 6), **4.1M** (hidden 768, depth 8 — reuse today's runs),
**~15M** (hidden 1536, depth 8). Arms per size: solo_A, solo_B, fleet_m500 (best schedule),
pooled. Seeds {0,1}. Metric: **Δ(size) = mean-over-shards(fleet) − mean-over-shards(pooled)**,
plus per-shard home costs vs the oracle.

Pre-registered read-out: **Δ shrinking monotonically with size** → the capacity hypothesis is
confirmed; the fleet path is scale + naive averaging, and the 4.1M gap is a small-model
artifact. **Δ flat or growing** → structured merging (F2) becomes mandatory before any fleet
scale-up. Either result is a paper section.

- [x] F1 complete (`g3_full_cap384.jsonl`, `g3_full_cap1536.jsonl`, 2026-07-03).
      **Read-out: Δ is FLAT** — oracle gap +0.290 / +0.267 / +0.290 at 0.9M / 4.1M / 15M
      (a 17× capacity range). The pre-registered conclusion applies: **structured merging
      (F2) is mandatory before any fleet scale-up** — capacity alone will not close the gap.
      Sharper characterization: the gap is *entirely* the distant shard at every scale
      (B-side +0.64/+0.58/+0.58; A-side ≈ 0, reaching parity with the oracle at 15M), and
      fleet's base retention stays best-in-class at every size (wiki 3.003/2.893/2.832 —
      better than the oracle's). Naive averaging systematically erodes the most
      out-of-distribution competence, scale-invariantly → F2 must protect the distant
      specialist's subspace (asymmetric/Fisher/TIES-style merging, or modular
      experts-per-domain à la the hot-add thread).

## 9. Follow-up F2 — structured merging (pre-registered 2026-07-03, before any run)

At 4.1M, m500 schedule, seeds {0,1}; baselines (solos, fleet_m500, pooled) reused from
`g3_full.jsonl` (identical protocol/seeds). Three mechanisms, all local-state-only
(fabric-plausible — nothing requires data or gradients to cross learners):

| arm | merge rule |
|---|---|
| `fleet_ties500` | sign-resolved deltas vs the post-previous-merge reference: agree → mean; disagree → keep the larger-magnitude delta (TIES-style interference resolution) |
| `fleet_fish500` | per-parameter importance-weighted: θ = (v_A·θ_A + v_B·θ_B)/(v_A+v_B), with v = the learner's OWN Adam second moment (synaptic importance lives in local optimizer state — zero extra compute or comms) |
| `fleet_asym500` | fixed asymmetric weights favoring the distant specialist: θ = 0.3·θ_A + 0.7·θ_B |

**Pre-registered read-out** (vs naive Δ = +0.267 at 4.1M): **GO** = best variant cuts the
oracle gap ≥ 50% (Δ ≤ 0.13) while keeping base retention better than the oracle and mean ≤
naive fleet's. **PARTIAL** = 25–50% cut. **NO-GO** = < 25% → the modular/experts-per-domain
path (F3, the hot-add thread) becomes the requirement, not an option.

- [x] F2 complete (`g3_full_f2.jsonl`, 2 seeds, 2026-07-03). **Verdict: GO — decisive.**
      Mean finals (A / B / wiki / mean / oracle-gap / gap-cut vs naive +0.267):
      **fish 3.011 / 3.207 / 2.891 / 3.109 / +0.007 / 97%** ·
      asym 3.390 / 3.186 / 2.912 / 3.288 / +0.186 / 30% ·
      ties 3.135 / 3.918 / 2.979 / 3.526 / +0.425 / −59% (worse than naive).
      **Adam-second-moment importance weighting closes the fleet–oracle gap to noise**
      (+0.007) while beating the oracle on the near shard (3.011 vs 3.183) and base retention
      (2.891 vs 3.102), replicated across seeds (fish s0/s1 within 0.02 everywhere), with
      zero data movement and zero extra communication — the importance signal is each
      learner's own local optimizer state. TIES (sign resolution) reliably *hurts* the
      distant shard; fixed asymmetric weights protect one side by taxing the other. The
      per-parameter, locally-computed importance is the mechanism. Fisher-merging is now the
      fleet's default consolidation rule for G3b and all scale-ups.
