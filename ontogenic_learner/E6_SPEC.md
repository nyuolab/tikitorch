# E6 — The Ratchet Arc, Tier 1 (PRE-REGISTERED; frozen before launch)

*Question: can the organism ACCUMULATE? E5 delivered the life but not the
ratchet: world-model storage seesaws (probe CE 0.2–2 resident vs ~19–21
non-resident), the action head saturates (greedy argmax bitwise-frozen from
2.2M steps, margins 19–37 logits), acting entropy collapses, and a novel
world arriving at maturity never ignites its policy. E6 screens three
intervention families against these failures in ONE unified design, with
n=3 everywhere. IDBD (user direction): Sutton-1992 Incremental
Delta-Bar-Delta as meta-learned per-weight plasticity — the hypothesis is
that it DISCOVERS the stable/plastic split E5 observed emergently.*

User decisions (2026-07-22): one unified screen; all three IDBD flavors;
budget ~$200+ incl. a replicated E5 baseline. Plan approved same day.

## Tiers

### Tier 1 — E6-R: replicated E5 baseline (~$16)
Two additional FULL E5-config lives (12M steps, 3 worlds), seed/world_seed
(1,1) and (2,2); with E5 (seed 0) → n=3 for every E5 life-course claim:
seesaw amplitude, entropy-lock timing, arrival failure, late-life ratchet,
zero-shot asymmetry. `retain_era_ckpts` on → all era-boundary ckpts to S3
(the substrate for Tier-3 branch controls). Launch: `ARM=base LIFE=full
SEEDS="1 2"`.

### Tier 2 — E6-C: the unified screen (~$75-110)
Compressed life: era_steps=250k (biome_steps=62.5k), 10 cycles, SAME 24-era
structure, babyai arrival at cycle 6 (= 3.5M), 6M steps, probes 3h,
all-era ckpts, margin telemetry. 8 arms × 3 seeds (seeds 0,1,2 =
world_seeds), one c7i.2xlarge per arm (3 packed lives, OMP 2):

| arm | overlay | family / attacks |
|---|---|---|
| base | {} | the screen's own n=3 baseline |
| anchor | anchor=adam, lam .01 | consolidation: EWC-precision protection |
| idbd-adam | optimizer=idbd-adam, lr_decay=none | consolidation: meta-learned per-weight α (Adam-scaled) |
| idbd-dyadic | optimizer=idbd-dyadic, lr_decay=none | same, shift-class α=2^q (substrate-native) |
| idbd-sgd | optimizer=idbd-sgd, lr_decay=none | classic Sutton-92 gradient form |
| ent-floor | act_eps=0.05 | entropy: acting-time ε-mix floor |
| ent-pull | ent_k=-3 | entropy: training-time action-head uniform pull |
| head-renorm | head_renorm_M=4.0 | entropy: bound the head's margin growth |

Declared: IDBD arms run lr_decay=none (α owns the schedule — also
de-confounds the E5 ratchet); ε-mix makes the stream mildly off-policy
(that IS the arm's mechanism); anchor prelife runs unanchored (birth
identity), θ re-anchors post-birth; anchor tracks per-step.

### Pre-registered endpoints (identical for every arm; screen references =
the archived E5 life run through the same code: seesaw amp 18.6,
non-resident CE 19.3–21.2, ignition none/max 0.0, minatar collapse era 20,
era-mean RETAIN 6/9 both envs)

- **P1 SEESAW**: life-mean non-resident probe obs-CE (and amp).
  CONSOLIDATION-WIN = arm mean ≥30% below the base arm's mean, with P2.
- **P2 PLASTICITY-INTACT**: babyai probe obs-CE ≤ 2.0 on ≥1 cell by the end
  of its first lived era (protection must not kill acquisition).
- **P3 RETAIN (fixed metrology)**: era-mean revisit bar (mean ≥ 0.8·prev)
  primary + rotation-aligned per-biome table (tele `biome` field) as
  robustness. The E5 quarter-based bar is a documented design error; not
  reused.
- **P4 ARRIVAL-IGNITION**: any babyai era-mean ret1k ≥ 1.4 (50% of the E5b
  birth cohort). ENTROPY-WIN = ignition in ≥2/3 seeds with P5.
- **P5 NO-TAX**: per-env final-full-era mean return ≥ 90% of the base arm's.
- **P6 ENTROPY-FLOOR**: frac of 1k-buckets with piH<0.01; first unrecovered
  collapse era; min era-mean piH (piH raw; piH_act beside it for ent-floor).
- Winners = family-WIN + P5; ≤2 advance to Tier 3. No winner → E6-C
  publishes the negative (weights-only consolidation / entropy control
  insufficient at this tier) and Tier 3 runs controls only.

### Tier 3 — E6-V: confirm + controls (~$65-80)
- Winners ×3 seeds at FULL scale vs the Tier-1 n=3 baseline (same bars at
  full scale; SUSTAINED-style: endpoint holds in ≥2/3 seeds).
- **Ckpt-branch novelty-attribution control** (the E5 skeptic's falsifier):
  from a Tier-1 replicate era ckpt at an entropy lock, three branches —
  novel-world next (natural), extra-familiar residency, scrambled-obs
  pseudo-novelty (codec obs-symbol permutation; env code untouched).
  Bar: entropy restoration in the novel branch only ⇒ the E5
  novelty-restoration attribution stands; restoration in familiar/scrambled
  ⇒ attribution REFUTED (trunk-movement suffices), published either way.
- **lr-matched ratchet control**: one full life, lr_decay=fixed (lr0/8
  from birth). Bar: late-life era means still climb ⇒ ratchet survives
  de-confounding; flat/declining ⇒ the E5 ratchet was schedule-artifact,
  published.

## Metrology & ops
st7_verdicts.py E6 endpoints (retain_era_mean, retain_rotation_aligned,
arrival_ignition, seesaw_amplitude, pih_floor) — fixture-tested; E5-archive
regression run gives the reference values above. st7_screen.py aggregates
arms × seeds → endpoint table + delta-vs-base (fixture: planted −51% seesaw
/ +10% returns / 2-seed ignition all recovered). Launcher run_e6_ec2.sh:
one box per (arm, life), templated e6life@seed units, per-seed workdirs +
S3 subprefixes, per-life watchdog (restart 1800s; reboot only if ALL stale
5400s), per-seed sync incl. ckpt_era/. Probe suites write status
state="probe" (packing-safe). Single-box sentinel before the fleet:
ARM=base LIFE=screen with era_steps=2000, 30 min — measures packed sps
(fallback c7i.4xlarge if <60), verifies units/watchdog/sync end-to-end.

## Local verification (all green before launch; house rule)
st7_test_optim.py: θ_meta=0 ⇒ AdamW-parity; sign-direction + |Δβ|≤θ bound;
dyadic hysteresis no-flap + single-crossing; burst stability (95% zero +
heavy-tail spikes) + h-leak drought decay; bitwise state roundtrip incl.
int8 q; _AnchorOpt pull contract; action_uniform_pull (LM grads untouched,
head entropy 0.018→0.355). G0 gradcheck flags-off PASS (282 checks).
FLAGS-OFF SENTINEL: smoke life on E6 code ≡ pre-E6 commit f139b45 —
weights sha256 identical, (ret1k, ce, fire, piH) tele sequences identical
(new tele fields declared metrology additions). Kill-resume twin drill per
arm (9 configs). Branch drills: novel ≡ uninterrupted twin; familiar
diverges only from branch_start_era; scrambled permutes symbols with env
digests unchanged.

## Cost ledger (band authorized ~$200+)
Tier-1 2 × c7i.xlarge × ~44h ≈ $16 · Tier-2 8 × c7i.2xlarge × ~25-30h ≈
$75-110 · Tier-3 ≈ $65-80 → total ≈ $160-210.

## Pre-launch review amendments (12 confirmed findings, 2026-07-23; all
frozen BEFORE launch — the bars below supersede ambiguous phrasings above)

- **P1 FROZEN**: the bar quantity is the unweighted mean over {foraging,
  minatar} of the env-level life-mean NON-RESIDENT probe obs-CE (babyai
  non-resident CE reported separately, pre-/post-arrival split; the amp
  number is descriptive only). WIN = arm mean ≥30% below base, with P2.
- **P2 FROZEN + computable**: primary = suite-A obs-CE on the babyai cells
  from any suite landing in [first babyai era, +1 era] (st7_screen p2ce);
  where no suite lands there, the offline eval on the first-babyai-era END
  ckpt (ckpt_era/, windows regenerated deterministically, n_windows=16).
  Bar: ≤ 2.0 on ≥1 babyai cell.
- **P4 DECLARED**: 1.4 is an absolute bar calibrated on the E5b 100k-step
  birth cohort, applied to 250k-step era-means (dilution direction:
  conservative). Complete eras only (0.95 × all-era max buckets).
- **P6 FROZEN**: descriptive endpoint (entropy-family WIN runs through
  P4+P5); collapse = piH<0.01, recovery = any later bucket >0.1; screen
  table reports WORST-case (max) collapse fraction across seeds.
- **IDBD birth divergence DECLARED**: IDBD arms run the prelife with
  meta-adaptation active (the arm is the whole treatment); birth weights
  are arm-specific; birth identity holds only among adamw-based arms
  (base/anchor/ent-*/head-renorm share it bitwise). PRE-COMMITTED: any
  Tier-3 confirm of an IDBD winner forces a plain-AdamW prelife so the
  confirm attributes the win to the in-life mechanism.
- **Ops fixes landed**: watchdog boot-grace + missing-status-not-stale
  (the dropped E5 guard would have reboot-looped fresh boxes — BLOCKER);
  branch cfg auto-adoption on restart (branch lives are restart-safe with
  the original unit line); probe wall-clock persisted in the ckpt (restart
  churn cannot starve suites) + mid-suite status touches; atomic
  status.json writes; nonfinite-grad skip in all IDBD steps; nan_guard
  gates the time-cadence save (a poisoned state can never be persisted);
  incident restore now re-asserts BOTH the era's env (the wrong-world fix
  covers nan_guard too) and the schedule lr (halvings=-1 → apply_lr).
