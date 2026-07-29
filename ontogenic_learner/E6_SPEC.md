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

## Launch-time amendments (2026-07-22, user-authorized expanded budget)
- Sentinel fired: 3-packed c7i.2xlarge ran 18-25 sps < the 60 bar → screen
  fleet on c7i.4xlarge (OMP 4); the slow base box was terminated and base
  relaunched on the same class as every other arm (probe suites are
  wall-clocked — box-class uniformity keeps suite density comparable
  across arms; the seed-extension boxes run 2 lives vs the primaries' 3,
  declared as a within-arm density difference, second-order).
- SEEDS EXPANDED: screen n=5 per arm (seeds 0-4; family bars re-read as
  ignition ≥3/5 etc. — the ≥2/3 phrasing scales as ≥60% of seeds);
  Tier-1 n=4 baseline (seeds 0[=E5],1,2,3); lrfixed control launched
  immediately (independent of screen winners).
- Cost restatement: screen 16 boxes ≈ $190 + Tier-1 3 boxes ≈ $40 +
  lrfixed ≈ $18 + Tier-3 remainder ≈ $50-70 → ≈ $300-320 total
  (user: "we can spend more if faster or more robust", 2026-07-22).

## Post-launch mechanism clarification (2026-07-24; explanatory only)

This note changes no arm, overlay, endpoint, threshold, seed rule, or kill
branch. It makes the already-launched entropy-family contrast explicit.

For the `ent-pull` overlay, `ent_k=-3` sets
`lambda = 2**ent_k = 1/8`. Let the union action head have `K` rows and the
current environment have `K_e` valid actions in rows `[0, K_e)`. Define
`U_e(a)=1/K_e` on those rows and zero on the remaining rows, and let
`p_theta=softmax(z)` over all `K` union-head rows. At each action position in
a fired update, the added per-position objective and logit gradient are

$$
L_{\mathrm{pull}}
= \lambda\,\mathrm{CE}(U_e,p_\theta)
= \lambda\left[\log K_e+D_{\mathrm{KL}}(U_e\|p_\theta)\right],
\qquad
\nabla_z L_{\mathrm{pull}}=\lambda(p_\theta-U_e).
$$

The softmax is over all `K` head rows. Thus valid rows receive
`lambda * (pi - 1/K_e)` and invalid rows receive `lambda * pi`, suppressing
probability outside the current action set. The implementation adds this term
after the dyadic advantage sign/zero mux, so it remains active when the
quantized advantage is zero. It changes the action-head gradient and the
shared-trunk error entering the single cascade sweep; the direct LM-output-head
gradient is bitwise unchanged. "Action-head pull" names the injection point,
not a head-only parameter update. The registered unit check verifies all three
properties and raises mean head entropy from 0.018 to 0.355 in its saturated
fixture.

This learner-side term is distinct from both nearby controls:

- `ent-floor` acts with
  $\widetilde\pi_e=(1-\epsilon)\pi_{\theta,e}+\epsilon U_e$ at
  $\epsilon=0.05$, where
  $\pi_{\theta,e}=\mathrm{softmax}(z_{[0,K_e)}/\tau_\pi)$ is renormalized on
  the current environment's valid rows. It guarantees sampling support and
  changes the experienced stream, but adds no direct parameter gradient.
- Standard maximum-entropy regularization minimizes, up to a constant,
  $D_{\mathrm{KL}}(\pi_{\theta,e}\|U_e)$. The `ent-pull` arm reverses the
  direction and applies $D_{\mathrm{KL}}(U_e\|p_\theta)$ to the union-head
  learner distribution, which weights every valid action by `1/K_e`, diverges
  as any valid-action probability approaches zero, and suppresses invalid rows.
  It is therefore specifically an anti-saturation pressure, not merely a
  generic entropy bonus.

The registered screen tests prevention across a life. `ent-floor` without
`ent-pull` would implicate insufficient action exposure; `ent-pull` without
`ent-floor` would implicate learner-side saturation; both succeeding would not
by itself identify which route is necessary; neither succeeding would reject
entropy control at this tier. Because treatment begins before saturation, even
a positive `ent-pull` endpoint would not establish that the same term can
reopen an already locked policy. That stronger recovery claim requires a
separately declared checkpoint-onset intervention and is not an E6-C endpoint.

---

## E6-C SCREEN RESULTS (2026-07-25, 40/40 lives, 0 incidents; data results/e6_screen/)

### Mechanical verdicts against the FROZEN bars

- **CONSOLIDATION: NO WINNER (published negative).** Best arm = anchor at
  −10.0% non-resident CE vs the −30% bar (P2 also missed: 4.16 > 2.0 — the
  protection-taxes-plasticity risk, measured). All three IDBD arms WORSEN
  the seesaw (+8/+9%) and tax residents −35→−54% (θ=0.01 traverses an
  octave per 100 aligned kicks — the peer-review diagnosis; see
  PEER_FEEDBACK_20260724). idbd-sgd's −67.6% "seesaw win" is degenerate
  (P2 3.18 fail, residents −45/−57%: an unspecialized model is uniformly
  mediocre everywhere — the P5 gate exists for exactly this).
- **ENTROPY: NO AUTOMATIC WINNER — by 0.7pp.** ent-pull: ignition **5/5
  seeds** (max era-means 2.31–2.69 vs birth cohort 2.75; base 0/5 at
  0.000), P6 collapse largely prevented (worst-case frac 0.26 vs base
  1.0), foraging late +19.6% — but minatar late = 89.3% of base vs the
  ≥90% P5 bar. Distributions overlap heavily (ent-pull best seed 42.8 >
  base mean 40.7); the miss is within seed noise, but the bar is applied
  as frozen. head-renorm: 4/5 ignition, P5 hard fail (minatar −47%).
  ent-floor: 0/5 ignition at zero tax — exploration noise alone cannot
  ignite (the forward-KL learner-desaturation mechanism confirmed by
  contrast).
- **Base arm: arrival failure 5/5 at 0.000/0.000** — with the full-scale
  and lr-fixed replications, the critical-period-like result stands at
  n=8 lives; the lr-fixed arrival failure kills the lr-schedule
  alternative.

### DECLARED post-prereg decisions (deviations stated, not hidden)

1. **ent-pull advances to Tier-3 full-scale confirm (3 seeds) as
   MARGINAL** — deviation from the letter of "family-WIN + P5" declared:
   the P5 miss is 0.7pp on one env against +19.6% on the other, within
   n=5 noise, and is not the buy-ignition-by-wrecking-residents failure
   mode P5 was frozen to exclude. Tier-3 full scale decides.
2. **Dose-response arm ent-pull-weak (ent_k=−4, 5 seeds, screen scale)**
   — the forward-KL frame predicts weaker pull ⇒ less resident tax; does
   ignition survive at half the regularizer?
3. **Branch controls launch NOW from Tier-1 s3's end-of-era-13 ckpt**
   (the replicate reproduced the full E5 sequence: era-13 entropy lock →
   era-14 arrival failure → era-15 cross-task restoration, same eras as
   E5 — restoration now n=2). Familiar + scrambled branches run; the
   novel arm = the source life's own lived continuation.

## BRANCH CONTROL RESULT (2026-07-25, three arms from Tier-1 s3's
end-of-era-13 entropy-lock ckpt; the pre-registered falsifier)

era-15 minatar acting entropy (the restoration readout), after era 14 as:
- **natural babyai** (novel, learnable): piH 0.668 — RESTORED (ret era14 = 0.0)
- **familiar foraging** (extra residency): piH **0.016 — NOT restored**
  (despite healthy era-14 performance, ret 37.5, and despite era-15 minatar
  still scoring 42.8 on its locked deterministic policy)
- **scrambled babyai** (permuted obs symbols — novel statistics, broken
  semantics, unlearnable: era-14 ret 0.0, piH 0.0): piH **0.779 — RESTORED**,
  ≈ the natural branch.

**VERDICT (per the frozen bar): the "novelty restores entropy" attribution
in its semantic form is REFUTED-AS-REGISTERED and REFINED** — restoration
occurred in a non-learnable branch, so learnable novel structure is NOT
required; mere passage/era-boundary is ALSO insufficient (familiar arm).
The causal ingredient is the SURPRISE LOAD itself: a high-CE input stream
driving large world-model (trunk) adaptation re-mixes the head's
context-conditioned logits and restores exploration elsewhere. This
CONFIRMS the trunk-movement mechanism hypothesized in the E5 analysis and
kills the romantic reading. n=1 per arm (one source ckpt); replicable from
the other Tier-1 seeds' ckpts if the claim is elevated. Language rule:
"surprise-driven entropy restoration", not "novelty therapy", and
"homeostasis" remains banned (peer-review point 2 upheld by data).

## LR-MATCHED RATCHET CONTROL RESULT (2026-07-25, full 12M life at fixed
lr0/8; the second pre-registered E5-claim control)

- Foraging era means: 74.3(birth!), 40.4, 28.4, 47.2, 40.1, 59.9, 53.7,
  49.5, 64.5, 39.6 — second-half +16% vs first-half, but the BIRTH era is
  the lifetime max; mid-life dip then partial recovery. Minatar: second
  half −17% (era-13-class collapse then partial recovery). Babyai: 0.0 in
  all four eras (arrival failure schedule-independent, again).
- **Combined verdict across n=3 full lives (E5, Tier-1 s3, lrfixed): the
  E5 "behavioral ratchet" is NOT ROBUST — one clean late-life climb (E5
  seed 0), one mid-life peak (s3), one mixed (lrfixed).** The lr schedule
  is not the sole driver (a mild second-half foraging gain survives at
  fixed lr), but "capability climbs with age" dies as a general claim.
  What survives, and is still non-trivial: NO late-life collapse under
  12M steps of fire-always plasticity in any life — stability without
  decay — plus seed-dependent trajectories. Language: "late-life
  capability is MAINTAINED; accumulation remains undelivered" — the E6-C
  conclusion (consolidation family negative) restated from the
  behavioral side, and the honest setup for what E6-S must actually
  measure.

---

## E6-D — THE COMPOUNDING DISCRIMINATOR (prereg DRAFT->frozen at D1 launch;
peer-review round 2 design, PEER_FEEDBACK_20260726)

Sequenced BEFORE E6-S (which is deferred until D1-D3 classify the
interference). Subsystem-specific causal frame adopted: WM seesaw =
ordering first; policy compounding = task structure first; lifetime
ratchet = negative (bffc62c).

- **D1 SHADOWING TEST (launching now, ~$5):** freeze_first_steps=50_000
  (serve-only window at REVISIT era starts; pure era-clock function,
  resume/branch-safe; flags-off bitwise + drills green). Branches from the
  COMPLETED Tier-1 s3 life: era002-end ckpt -> era-3 foraging revisit
  frozen; era004-end -> era-5 minatar revisit frozen. The source's own
  lived eras 3/5 are the unfrozen controls (zero cost). BAR: live CE
  recovery to <=2x resident level WITHIN the frozen 50k (no updates) =>
  SHADOWED (contextual recalibration); no recovery until unfreeze =>
  FAST RELEARNING (features preserved, mapping rebuilt). Return read
  beside CE; frozen buckets tagged in tele.
- **D2 ORDER-BY-WIDTH:** recorded-stream WM-only replay (exact multiset,
  blocked vs interleaved) x d{128,256,512} x 3 seeds + on-policy
  era-granularity sweep beside it (multiset cannot be held fixed
  on-policy; declared). Build after the ent-pull confirms land.
- **D3 MECHANISM PROBES:** cross-world gradient cosine, top-k route
  overlap, hidden lin-probe retention, recovery half-life.
- **D4 TASK RICHNESS:** compositional XLand subset (later tasks require
  earlier skills); acquisition AUC vs from-scratch, worst-skill floor,
  breadth, zero-shot. THEN E6-S: >=3 widths, held-out width/lifetime
  prediction = the bar for the word "law"; normalized composite primary.
- **Replay-as-diagnostic-oracle arm** accepted (replay stays excluded from
  the organism; the arm prices the constraint).
- Metric rule (Sol): %-slope retired for compounding claims; quoted growth
  numbers now have committed derivations (results/e6_screen/growth_check).

## D1 SHADOWING TEST RESULT (2026-07-26, two branches from Tier-1 s3;
derivation results/e6_d1/)

Instantaneous CE through the 50k-step SERVE-ONLY revisit windows:
- foraging revisit: flat 18.6 the whole window (zero drift); on unfreeze
  -> 1.1 -> 0.7-0.8 within ~2k steps.
- minatar revisit: flat 22.3 the whole window; on unfreeze -> 8.1 -> 0.4
  within ~3k steps.
- Behavior split rides again: minatar's frozen policy scored 167/1k
  constant (the frozen skeleton runs open-loop); foraging's decayed 13->1
  (its scoring depends on online WM adaptation).

**VERDICT (per the frozen bar): NOT SHADOWING — FAST RELEARNING.**
Serve-only exposure recovers NOTHING (the displacement is weight-level,
not contextual masking), yet gradient-on recovery erases a ~20-nat
displacement in 1-3k steps (~100x faster than initial acquisition:
features/subspace preserved, output mapping displaced, remapping nearly
free WITH updates and impossible WITHOUT). Kills the cheap remedy class
(context-conditioned calibration/temperatures — registered map) and
selects the structural class: slow-trunk/fast-adapter splits, sparse
experts, surprise-gated route allocation, or low-rank historical-subspace
protection. D2 (order-by-width) + D3 (mechanism probes) now discriminate
WITHIN that class.

## DOSE-RESPONSE RESULT (2026-07-26, ent_k=-4 x 5 seeds):
## ent-pull-weak = THE E6-C WINNER, under the ORIGINAL frozen bars

ignition 5/5 (P4 ✓) · late returns foraging +26.1%, minatar 93.0% of base
(P5 ✓ both envs) · piH worst-case collapse frac 0.43 (vs base 1.0) ·
non-resident CE unchanged (-0.2%: entropy repair does NOT touch the
seesaw — orthogonal mechanisms, as the family split predicted).
**ENTROPY-FAMILY WIN, no deviation required** — the forward-KL theory's
dose prediction (weaker pull => smaller tax, ignition retained) confirmed;
taxes monotone in pull strength, foraging actually improves at the weaker
dose. Note: the arrival-window babyai probe read (p2ce 16.2) is a suite-
cadence artifact to resolve against the era-ckpt offline eval in the final
analysis (P2 gates P1, not the entropy win). Tier-3: 3 full-scale
ent_k=-4 confirms launch alongside the running ent_k=-3 fulls (which now
double as full-scale dose-response).

## TIER-3 FULL-SCALE IGNITION CONFIRM (2026-07-27): 3/3 UNANIMOUS

ent-pull (ent_k=-3) full-scale lives (12M steps, arrival at 7M with 2x the
screen's saturation history): era-14 means 2.346 / 2.251 / 2.289 vs the
1.4 bar (birth cohort 2.75; base full-scale lives 0.000 x4 + lrfixed
0.000), acting entropy 1.83-1.88 throughout the arrival. **The SUSTAINED
bar (>=2/3 seeds at full scale) is met unanimously: a mature agent under
the forward-KL anti-saturation pull acquires a novel sparse-reward world
at ~82-85% of newborn efficiency.** The causal chain is complete and
each link independently replicated: pathology (n=8) -> mechanism
(saturation; surprise-load restoration) -> cure (dose-calibrated
forward-KL) -> boundary (acquisition, not accumulation — the seesaw is
untouched by the cure and belongs to E6-D/E6-S). Remaining: ent_k=-4
full-scale P5 reads (~2 days) for the winner's tax at scale; then E6
RESULTS + arc close.

## WINNER FULL-SCALE ARRIVAL (2026-07-28): ABOVE-NEWBORN, 3/3

ent-pull-weak (ent_k=-4) full-scale arrival eras: 3.117 / 3.367 / 3.019
(mean 3.17 = ~115% of the birth cohort's 2.75; every seed above newborn;
every seed above every ent_k=-3 seed's 2.25-2.35; piH 1.72-1.80).
**POSITIVE MATURITY TRANSFER, measured**: with exploration unblocked at
the calibrated dose, seven million steps of mature representation makes
the agent a BETTER learner of the novel sparse-reward world than a
newborn — the inversion of the arrival pathology, and the first measured
instance of the maturity advantage the program's thesis always implied.
Dose-response at full scale is monotone in the right direction (weaker
pull => higher arrival return: less regularizer drag on the freshly
learnable head). Remaining: winner lives complete (~1 day) for the
full-scale P5 tax + revisit series; then E6 RESULTS.

---

# E6 RESULTS — THE RATCHET ARC, CLOSED (2026-07-29)

~65 lives, ~$650, 7 days launch-to-close. Every verdict below is
pre-registered (bars frozen before launch; the two declared deviations are
marked), replicated, and derived from committed artifacts
(results/e6_screen/, results/e6_tier3/, results/e5_full/, E6_SPEC sections
above in chronological order).

## The causal chain (the arc's spine)

1. **PATHOLOGY (n=8 lives):** a mature agent under the E5 recipe NEVER
   ignites a novel sparse-reward world — return exactly 0.0, acting
   entropy exactly 0.000, at half AND full scale, cycle-decayed AND fixed
   lr. Mechanism located: action-head logit saturation (E5: margins
   19-37, argmax bitwise-frozen from 2.2M steps).
2. **MECHANISM of the one natural recovery (branch falsifier):** cross-
   task entropy restoration is caused by SURPRISE LOAD driving trunk
   adaptation — scrambled-obs (unlearnable) restores it (0.779 ≈ natural
   0.668), extra-familiar residency does not (0.016). "Novelty therapy"
   refuted-as-registered.
3. **CURE (forward-KL anti-saturation, ent_k):** the training-time
   uniform-target pull on the action head — gradient of λ·CE(U,π), the
   FORWARD KL direction that punishes near-zero action probabilities —
   ignites the mature agent; the acting-time ε-floor (0/5) proves the
   desaturation must reach the LEARNER, not the actor.
4. **DOSE (ent_k=-4 = the winner):** screen ENTROPY-WIN under the original
   frozen bars (ignition 5/5; P5 both envs; piH collapse frac 0.43 vs
   base 1.0); taxes monotone in pull strength.
5. **INVERSION (the headline):** at full scale the treated adult BEATS the
   newborn: arrival eras 3.12/3.37/3.02 (mean ~115% of the 2.75 birth
   cohort), and **12/12 winner babyai eras above the newborn mean** across
   all revisits — positive maturity transfer, measured, stable, slightly
   ratcheting within the third world (ws0 ends at its lifetime-best 3.38).
6. **FULL-SCALE P5 (SUSTAINED >=2/3):** foraging 3/3 (47.0/42.4/47.8 vs
   baseline 38.1 = ~120%); minatar 2/3 (104%/106%; ws0's 36% is an era-22
   rotation outlier — its own era-20 = 57.0; baseline seeds' own finals
   range 19-51). Residual life-mean minatar tax ~10-15% = the cure's
   declared price at this dose.
7. **BOUNDARY:** the cure buys ACQUISITION, not ACCUMULATION — seesaw
   untouched (-0.2%); D1 branch-freeze: the seesaw is weight-level
   displacement with preserved features (serve-only recovers nothing;
   gradient-on recovers ~20 nats in 1-3k steps). Consolidation family:
   pre-registered negative (anchor -10% vs -30% bar + P2 plasticity tax;
   IDBD all-flavors negative at theta=0.01, octave-per-100-kicks
   diagnosis). Ratchet: NOT ROBUST across n=3 baseline lives; the robust
   claim is MAINTENANCE WITHOUT COLLAPSE.

## Language locked for the papers
"critical-period-LIKE pathology, fully inverted by a forward-KL
anti-saturation regularizer"; "surprise-driven entropy restoration";
"late-life capability maintained; cross-world accumulation open (E6-D)";
never "novelty therapy"/"homeostasis"/"biologically plausible rule".
External review (GPT Sol) materially shaped: the KL-direction frame, the
dose-response design, the D1 discriminator, the retraction hygiene.

## What E6 hands to E6-D / E6-S
Structural remedy class selected by D1 (slow-trunk/fast-adapter, sparse
experts, surprise-gated routes, low-rank subspace protection); D2
order-by-width recorded-stream replay + D3 mechanism probes next; E6-S
deferred behind them, with the normalized-competence composite and the
held-out-scale bar. The organism now: durable, cure-carrying, above-
newborn on arrival — still does not compound across worlds. That is the
next arc's single question.
