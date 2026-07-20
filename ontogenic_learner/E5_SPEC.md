# E5 — the second life (prereg DRAFT; ratios frozen now, absolute steps frozen
# after the pilot via the single-factor rescale rule; per the approved plan)

*The persistence gate: a weeks-long, kill-9-resumable, always-on embodied
agent living alternating foraging/minatar eras with revisits, with a
pre-registered third-world (BabyAI) arrival late in life. Goal-B posture:
retention is TESTED, not assumed. Claim language locked in the plan's
goal-alignment section ("biologically constrained/inspired"; systems-
integration novelty; no non-BP-rule claims).*

## Configuration (recipe + E4d winners)

- Agent: st6_reward.Agent recipe (token-policy + dyadic advantage + intr
  β=0.5 + adv-norm), d128/4L, window 96, tau=0 serve. Firing schedule =
  ALWAYS (E4d: no sparse winner — fa25 fails maintenance, advf degenerate).
  Mechanism lever = HEADOBSK (PRE5-A WIN: head + obs-k -2 compose — minatar
  3/3 sustained at 36.5 > streaming-AC, foraging 58.3 incl. the 78.1 record;
  wmv NULL at 4x cost, does not join).
- UnifiedCodec: shared [ACT][OBS][RWD] + shared action block K=7 (union incl.
  MiniGrid; each env renormalizes over its own K_env) + shared reward block +
  DISJOINT obs-symbol blocks: foraging(3) | minatar(11) | babyai(reserved,
  ~20). Vocab fixed at birth INCLUDING the reserved BabyAI block.
- Eras: era_steps=500k (biome_steps=125k → 4 rotations/era), cycle =
  (foraging, minatar), 10 cycles = 20 eras = 10M env steps; start-env
  rotates per cycle; envs NEVER reset (revisits are genuine re-entries);
  done=1 on the outgoing era's final frame; ctx rolls across boundaries.
- ONE birth (foraging, tau-annealed prelife port, BC per arm, pi0_ent
  audit, birth ckpt). No per-era rebirth (zero-shot minatar entry IS the
  transfer readout). mb_warmup=0; tau_pi fixed for life.
- lr decay: cycle-clock (halve per 2 cycles, floor lr0/16). Anchor: NONE
  primary; drift tripwire (own-env probe CE rising across 4 consecutive
  suites at fire≈1) engages anchor-all-weak lam=.01, ledgered ANCHOR_ON.
- Box: c7i.xlarge-class, systemd Restart=always + cron watchdog, NO
  self-terminate; budget cap ≈$75 → graceful end-of-life ckpt + idle.

## Files (per the Plan-agent architecture, verified against source)

- `experiments/st7_life.py` — E5Config(+hash), UnifiedCodec,
  LifeAgent(Agent) with state()/restore() + per-env frame_len + K-renorm
  acting + era-boundary done, birth(), era loop (Life class), main().
- `experiments/st7_probes.py` — frozen probe windows (8+ cells: 3 foraging
  biomes, 5 games, +1-2 BabyAI on arrival), Suite A obs-CE replays, Suite B
  clone-return probes (2k steps at life tau_pi, NOT greedy), Suite C
  action-agreement vs end-of-cycle-1 snapshot; state-hash isolation assert.
- `experiments/st7_ops.py` — s3 wrapper, heartbeat/status writer,
  load_latest_fallback (walks older ckpts on torch.load failure).
- `experiments/st7_test_resume.py` — Stage-0 local drills: kill-9 hash-equal
  resume (params+opt+env bytes+counters+tele tail), probe isolation, env
  roundtrips.
- `scripts/run_e5_ec2.sh` — organism-launcher port (systemd, watchdog,
  pinned minatar/torch versions, no data tarball).
- Modified: `st5_envs.py` get/set_state (DONE, committed fbdb4cc,
  bit-exact 300-step verified).

## Checkpoint (15-item blob, save_atomic keep-8 + LATEST + S3 latest.pt)

model · opt · ctx/ctx_r/ctx_d · gate(mu_f,mu_s) · sal_mu · adv_baseline/
var_ema/ce_ema/intr_sum · t_step+life counters+ledger_seq · win_tail ·
BOTH env states (opaque bytes + sha256) + current env id · lr machinery ·
anchor state · RNG (global torch + probe/audit gens) · probe bookkeeping +
snapshots · hashes (config, env-code sha256 of st5_envs.py+st5_world.py+
minatar/torch versions+commit) · tag+config. Cadence: 50k steps OR 900s +
forced era-boundary ckpt. Resume: config-identity + env-code-hash asserts
(hard fail on mismatch); telemetry dedupe; declared RESUME_NONEXACT splice
branch for hotfixes.

## Probes (every 6h wall, life-clock frozen, clone-isolated + state-hash assert)

Transfer/interference matrix rows every 6h × 8 cells × {obs-CE, reward-CE,
return/1k, action-agree}. Pre-registered readouts: interference slopes
(other-env residency), zero-shot transfer (first-entry values), per-revisit
savings (the Goal-B non-increasing-forgetting criterion, per env), drift
(agree vs cycle-1 snapshot).

## Pre-registered Goal-B bars (verdicts at life end; honest branches)

- **RETAIN:** per-revisit re-entry recovery — era-A′ mean return over its
  final quarter reaches ≥80% of era-A's final quarter within the first 25%
  of the revisit era, with the recovery cost (steps to 80%) NON-INCREASING
  across cycles 3→10. KILL branch: increasing recovery cost = forgetting
  dominates; published as bounded-horizon adaptation.
- **NO-RELAPSE:** no DARKROOM incident (behavioral-entropy + obs-entropy
  floor, calibrated in pilot) lasting >48h without the ledgered rescue.
- **STABILITY:** weight-norm plateau (st4 criterion); no nan incidents
  beyond POISON_SKIP count ≤ 3; audit coverage on sampled kicks 100%.
- **TRANSFER (novel readouts, reported as measured):** zero-shot minatar at
  birth+era-2 entry; cross-env interference slopes; BabyAI arrival (below).
- **OPS:** ledger integrity across all resumes; ≥1 real kill-9 survived in
  the life (we will drill one deliberately in week 1).

## The third world (E5b gate + arrival)

BabyAI adapter (fixed-task GoTo* levels, no language channel, 7×7 obs,
K=7, object-type symbolization) + gate screen (~$8): recipe ×3 seeds +
random/frozen floors; inclusion bar = beats random floor on return OR clean
obs-CE learning (policy traction NOT required — mid-life acquisition is the
readout). If passed: BabyAI eras join from cycle 6 (pre-registered arrival,
capacity reserved at birth); probes gain BabyAI cells (frozen at pilot
time). If failed: 2-env life, reserved block inert (declared).

## Pilot (mandatory gates before the life)

Stage 0 (local, free): st7_test_resume all-pass. Stage 1 (EC2 ~$8-17):
compressed life (era 100k × 4 cycles) + 7 drills — kill-9 mid-era (hash-
equal resume), kill-during-ckpt-write (LATEST fallback), SIGSTOP (watchdog
<15 min), S3-only resume, probe duty <5%, telemetry shakedown + dark-room
floor calibration, throughput → single-factor era rescale → freeze this
spec's absolute numbers. All gates pass → launch the life.

---

## E5b RESULT (2026-07-19, 9/9 cells, 0 failures, ≈$3): BABYAI JOINS THE LIFE

Unanimous on BOTH inclusion conditions, all 3 seeds: recipe (headobsk)
2.74/3.05/2.46 ret/1k (mean 2.75 = 2.4× the random floor 1.14 — real policy
traction on sparse terminal-only reward through windowed REINFORCE) and
obs-CE 0.081–0.091 vs frozen 0.50–0.59 (clean world-model learning). Birth
calibration at K=7 confirmed (pi0H ≈ ln 7). Data:
`results/st_reward/e5b_screen.jsonl`. The full life runs babyai_enabled with
the arrival at cycle 6: cycles 0–5 = (foraging, minatar) pairs, cycles 6–9 =
rotated triples → 24 eras, 12M steps (~+20% budget, within cap).

## PILOT (running, i-087583fd254577a2f) — drill scoreboard so far

kill-9 mid-era ✓ (systemd resume, RESUME event, tele continuity) ·
SIGSTOP watchdog ✓ (restart <12 min, resumes=2) · S3 recovery artifact ✓
(latest.pt uploads + torch-loads after the live-patched heredoc fix,
launcher fixed dfcc810) · probe duty ✓ (87s/suite at 1h cadence ≈ 2.4%) ·
throughput 41–79 sps blended (full life ≈ 2–3 days ≈ $20–30, NO era
rescale needed) · FIRST SCIENCE: era-0→2 foraging catastrophically
interfered (probe CE ≈ 21, clone returns ≈ 0–3) while minatar excels
(CE 0.27–2.3, clones to 118) — the era-3 revisit recovery is the live
RETAIN readout; the full life measures whether revisits cheapen.

## Mid-life declared intervention (2026-07-20 ~19:40 UTC, ledgered via INCIDENT + this note)

At ~3.9M steps the probe-isolation assert FIRED and crashed the process
(systemd resumed cleanly; INCIDENT in journal). Root cause, reproduced
locally: the full life's probe suites include BabyAI cells (the pilot's
did not) and the canonical digest's deep walk reached (a) the SHARED
gymnasium EnvSpec registry object, which gym.make mutates when
constructing probe envs, and (b) ctypes library loaders where hasattr
probing dlopens. Fix: BabyAIEnv digested by FUNCTIONAL state only
(counters, grid encode, agent pos/dir, np_random bit-generator state) —
implemented in st7_probes.py ONLY, so the env-code hash (st5_envs+
st5_world) is untouched and the running life's resume assert holds.
Battery re-verified locally with babyai cells (witness idempotent +
invariant + post-probe continuation). Hot-patched on-box via SSM +
service restart (resume from ckpt); suites 1-2 unaffected (pre-arrival);
suite 3 was lost to the crash. The assert did its job: fail-loud beats
silent contamination.
