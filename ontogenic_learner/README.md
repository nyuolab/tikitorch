# ontogenic_learner

> **2026-07-11 — PAUSE LIFTED.** The recurrent substrate this line was waiting
> for exists (branch pc-morphssm: cascade≡BP MorphSSM, zero-multiply audited,
> no activation graph, async-exact). The fused validation program is
> pre-registered in `NORTHSTAR_VALIDATION.md` (V-gates; two-goal framing:
> live-loop vs lifelong retention). All char-61 numbers below are now
> hypothesis-only references — the V-gates run BPE-1024 on the real substrate.

> *The parameters×tokens era belongs to backprop-on-GPU and always will; the experience×time
> era has no incumbent yet — and this program's measured assets (local rule, bit-exactness,
> elasticity, mult-free silicon) are exactly the properties that era prices. The window is
> ~two years, and the whole bet hinged on one cheap experiment: does the two-speed learner
> beat rehearsal-backprop on the retention–plasticity Pareto?*
>
> **It does. G1 verdict: GO (2026-07-03, pre-registered, 3 seeds).**

A definitive architecture for the **experience×time era**: low-energy, always-learning systems
that learn the way biology does — locally, from surprise, in two speeds, for their whole
deployed life. Built on [TikiTorch](../) (the tropical/dyadic multiply-free substrate, the
predictive-coding layer stack, and the FPGA-fabric training rig) but organized as its own
codebase: TikiTorch answers *"what silicon and what learning rule"*; **ontogenic_learner**
answers *"what do you do with a lifetime."*

---

## 1. The goal

Break with the backprop-on-GPU paradigm toward systems that **(a) scale better structurally**
(layer-local credit → model-parallel across weak devices, no global gradient, no all-reduce),
**(b) are resilient** (checkpoint-free elasticity — chips can die and learning continues),
**(c) are biologically plausible** (no weight transport; error-gated plasticity; wake/sleep
consolidation — used as a hypothesis-generator, never as marketing), **(d) learn continuously,
with the end-state being NO difference between training and inference at all** — one PC
settling computation, target clamped (learn) or free (infer); the two-speed learner is the
operational bridge, the settling regime (U-gate) is the destination — and **(e) have a
pathway to being MORE performant *because of* these properties** —
via fleet-experience scaling (N deployed learners × fresh situated data no batch pipeline can
buy), never-stale models on drifting distributions, test-time settling as native adaptive
compute, and updates-per-joule below the GPU power floor.

We do not compete on tokens/$ for batch pretraining — the verified per-device gap (~3,000× at
the projected kernel tier; 40–1,000× in rented dollars) makes that unwinnable forever, and we
say so. We compete on **SPJS: score-per-joule-per-stream** on non-i.i.d. continual experience:
`{final capability, retention, plasticity, adaptation latency, energy, audit coverage}` —
the axis on which a frozen GPU model scores zero and an adapting one pays the LoRA-cycle tax.
GPU baselines are first-class entries; the leaderboard is honest only if the incumbent can
enter and win columns. Full paradigm definition, six principles, honesty ledger, and the
pre-registered protocols: **[PARADIGM.md](PARADIGM.md)**.

Division of labor with the GPU world (and with the fabric instance scaling the on-fabric LLM
in parallel): **GPU = phylogeny** (pretrain the prior once — also the science accelerator for
rule-level questions), **fabric/edge = ontogeny** (auditable, always-on, in-place adaptation),
consolidation across individuals via infrequent outer averaging (no per-step sync).

## 2. The roadmap — gates, in order (kill/promote, pre-registered criteria)

| gate | question | status |
|---|---|---|
| **G1** — two-speed learner | Does ε-gated wake + sleep-consolidation beat rehearsal-backprop on the retention–plasticity Pareto at matched update budget? | ✅ **GO** (2026-07-03) |
| **G1.1** — stabilized settled-PC sleep | Does the iterative (settle-then-update) rule run stably as the sleep phase? | ✅ clean (0 divergences); beats rehearsal, not yet cascade-sleep |
| **G1.2** — stronger controls + energy curve | Does the GO survive a strengthened rehearsal control (50% replay, domain-balanced)? What does the gate-threshold τ sweep say about quality-per-joule? | ✅ **GO** (beats ALL 4 controls; τ = a continuous stability–plasticity–energy dial) |
| **G1b** — fabric certification | Certify the two-speed learner on real F2 silicon — zero kernel rebuild; turnkey protocol ready → [G1B_SPEC.md](G1B_SPEC.md) (knob mapping code-verified, 4-step protocol, go/no-go frozen) | 📋 **spec'd, ready for the fabric instance** — the critical-path gate for "running on fabric" |
| **G2** — the rule-level question | Does non-converged / prospective-configuration PC beat SGD countermeasures at LM scale? | 🟡 **G2a pilot complete — honest null** ([G2_SPEC.md](G2_SPEC.md) §8): async-PC ≡ BP on H1/H2/H3 at 4.1M (all differences attributed by controls); the prospective regime doesn't condition at width (needs µPC-style fixes → research-track); G2b GPU spend paused by amendment A1. **The schedule replicated at 9× scale and dominated cyclic streams.** |
| **G3** — fleet merged-beats-solo | Two learners on disjoint shards, exchanging ONLY weights, merge into one model holding both competences ([G3_SPEC.md](G3_SPEC.md)) | 🟢 **GO via F2**: naive averaging was capacity-limited (scale-invariant 0.27 oracle gap, F1) — but **Fisher-style merging (each learner's own Adam second moment as per-parameter importance) closes the gap to +0.007 (97% cut)**, beats the oracle on near-shard and base retention, replicated across seeds, zero data/extra comms. Fisher-merge is the fleet's default rule |
| **G4** — wall-power | Measured J/stream vs Jetson-Orin / RTX-4090 LoRA at matched adaptation quality. The F2 card is expected to LOSE — that loss quantifies the FPGA tax and is the ASIC study's input | ⬜ |
| **G5** — ASIC option | Fund silicon only after G1+G3 pass and G4 quantifies the tax; gating spec = the integer/multiply-free backward | ⬜ gated |

Standing rule: **every dollar of hardware follows a passed gate.** A NO-GO at G2 contracts the
paradigm claim to auditable-fleet systems economics — still a business, no longer a paradigm —
and we publish the negative next to the infrastructure.

## 3. Results so far

### G1 — GO ✅ (full detail: [results/g1_two_speed/G1_RESULT.md](results/g1_two_speed/G1_RESULT.md))

DyadicPCNet char-LM (ctx 12 · h 256 · d 4 · mult-free ±2^k), phylogeny on 5M chars of
Wikipedia (bpc 2.90), then a strictly sequential non-i.i.d. stream with large ordered shifts —
**Shakespeare → repo docs → repo code** (frozen-base bpc 5.5/6.6/7.4). Matched budget 600
updates/domain, 3 seeds, mean:

| arm | final avg bpc ↓ | forgetting ↓ | base drift ↓ | plasticity ↑ |
|---|
## 2b. The V-gate program (2026-07-12 → 07-14) — the successor ladder, COMPLETE

The pause-lifting program that fused this line with the pc-morphssm substrate.
Full prereg + verdicts: [NORTHSTAR_VALIDATION.md](NORTHSTAR_VALIDATION.md);
one-page visual: `results/st_stream/D2_life_stripchart.html` (also published
as a live artifact).

| gate | verdict |
|---|---|
| V0 fused mechanism | ✅ 1.23× wall / 1.38× ops vs serve; grads bitwise ≡ cascade; zero-multiply closed config |
| V1 streams + oracles | ✅ headrooms 6.7–10 bits/tok off-domain; tokenizer identity exact |
| V2 does-it-help | ✅ regret −1.6…−3.0 nats/tok, HC 0.61–0.69, at 1.13× serve |
| V2b gate law | ✗→✓ z-gates are fixed-quantile rate limiters (can't self-quiet); the two-timescale ABSOLUTE-MARGIN gate replaces them (self-quiets by construction — V4cal: 0.01–0.24% stationary fire) |
| V3 honest incumbents | ✅ frontier held: matching our quality costs tuned LoRA 3.3× our update budget + a 2.7× memory class; online-BP wins unconstrained compute (segmentation as pre-registered); online-BP also HURTS on stationary — gating is universal |
| V4 lifetime (10M tokens, 5 rotated cycles) | ✅ ALL criteria: HC 0.602 at 1.0063× serve; fires 1275→45/cycle (self-quieting); norm equilibrium; non-increasing interference; sign-drift adversary survived 3.5× the collapse horizon; 0/1,236 audit violations |
| V4b fleet merge | ✗ HONEST NEGATIVE: Fisher merge loses to naive averaging under sparse gated firing (stale, fire-rate-mismatched importance; riders converged to overlapping generalist basins) → V4b.1 = firing-normalized importance |
| V5 silicon energy | session in flight (two-image S1 design; scoping in [V5_PMBUS_PROTOCOL.md](V5_PMBUS_PROTOCOL.md)) |

This ladder retires G1b (the loop is now certified end-to-end on the real
substrate at life scale; the L=2 xformer silicon defect remains open on the
fabric track) and delivers G4's compute half (V3) with its energy half in V5.
Paper numbers N1/N2/N3 are banked in the prereg ledger.
---|---|---|---|
| frozen | 6.548 | — | — | 0.00 |
| cascade (naive sequential BP) | 3.529 | +0.610 | +0.712 | 1.00 |
| rehearsal (control) | 3.623 | +0.625 | +0.899 | 0.98 |
| **two_speed** (ε-gated wake + sleep) | **3.485** | **+0.419** | **+0.576** | **0.98** |

The two-speed learner **Pareto-beats the control** (33% less forgetting, better final quality,
98% plasticity) and dominates naive cascade on every retention column — while its ε-gate fired
on only **~30% of stream batches**: the learner ignored 70% of its experience and still won.
That selectivity is the low-energy always-learning mechanism, working.

**Honest ledger:** the win so far is the *schedule* (gating + surprise-weighted reservoir +
low-LR sleep), not the PC rule — G1.1's stabilized settled-PC sleep runs clean and also beats
the control but adds no edge over cascade-sleep at ~1.7× compute (the rule question moves to
G2, in the regime where the literature locates it). The rehearsal control underperformed naive
cascade and must be strengthened before publication (G1.2). Scale is the designed first rung
(0.5M params); it licenses the next rungs, it does not claim LM scale.

## 4. Running things

```sh
# from the tikitorch repo root (tikitorch is the library; PYTHONPATH picks it up)
PYTHONPATH=. python ontogenic_learner/experiments/two_speed_learner.py --smoke   # ~30 s sanity
PYTHONPATH=. python ontogenic_learner/experiments/two_speed_learner.py          # full G1, 3 seeds
```

Corpora are derived once into `data/experience_stream/` (wiki=text8 slice, tinyshakespeare,
repo docs, repo code) and then treated as **frozen caches** for run-to-run comparability.
All results are JSONL event logs — every number in this README is reproducible from them.

### G1.2 — GO ✅ under strengthened controls, and the gate is a *dial*

The GO survives all four rehearsal-control variants (best control: domain-balanced 25% replay,
3.534 final / +0.510 forgetting — still beaten; 50% replay *hurts* at matched budget: 4.010).
And the gate-threshold sweep exposes the paradigm's control surface — **one scalar (τ, the
surprise threshold — a neuromodulatory tone) trades plasticity against retention against
energy**:

| gate τ | final avg ↓ | forgetting ↓ | plasticity ↑ | updates (energy) |
|---|---|---|---|---|
| z>0.00 | **3.480** | +0.311 | 0.96 | 1,755 (98%) |
| z>0.25 (G1) | 3.489 | +0.442 | 0.98 | 1,746 (97%) |
| z>0.75 | 3.611 | +0.253 | 0.91 | 1,212 (67%) |
| z>1.50 | 3.807 | **+0.030** | 0.80 | **717 (40%)** |

At τ=1.5 the learner updates on ~7% of stream batches, spends **40% of the energy budget**,
retains prior domains **essentially without forgetting** (+0.03 bpc), and still delivers 80%
of full plasticity. That is the low-energy always-learning operating point — and the fact that
it's a *smooth dial* rather than a fixed trade is itself the finding.

## 5. Relation to the PC-TikiTorch fabric program (the PC LLM)

Two arms, one organism. The **fabric program** builds the *phylogeny engine* — kernels, AGFIs,
and distributed PC training that answer "can a transformer be trained at all on mult-free
fabric with no global sync" (measured: 8 blocks on 8 FPGAs bit-exact; 3-chip converged
bpb 3.187; 258 tok/s; now building the 30M S2 kernel toward 124M). **This arm** builds the
*ontogeny layer* — what the trained organism does with its deployed lifetime: always-on
ε-gated learning, wake/sleep consolidation, fleet Fisher-merging, bit-exact audit,
train≡inference as the end-state.

| | fabric program | ontogenic_learner |
|---|---|---|
| axis | parameters × tokens, on fabric | experience × time, on the same fabric |
| unit of progress | kernels, AFIs, kick times, bpb-at-convergence | pre-registered gates on the SPJS tuple |
| learning rule | PC cascade (== block-local BP) | same rule; the innovation is the **schedule** + system |
| what it proved | the architecture trains on silicon, bit-exact, no global gradient | the deployed substrate should never stop learning — and wins doing so |
| north star | 124M PC transformer on multi-F2 | re-scoped together: *consolidate fleet experience into a warm-started 124M* |

Concretely interlocked: the two-speed learner is a **host policy over the fabric's existing
per-kick knobs** (`DO_UP`/`LRBITS`/`GZERO` — zero kernel rebuild, certified on silicon at
G1b); Fisher-merging uses the **Adam moments already resident on-chip**; the audit ledger
rides their bit-exactness. Traffic flows both ways: they supplied the substrate and the
measured recipes; this arm returned window-gating (+34% measured throughput), the L=2
backward-defect dossier, and the strategic answer to "why fabric at all" (GPUs own rented
FLOPs forever; the fabric owns deployment-time learning). And this arm's founding move was
taking the fabric program's own *honest negative* (`pc_continual.py`: iterative PC retains
but is 4× less plastic) and turning the trade-off into a design — fast cascade wake + slow
consolidating sleep.

## 6. Organism v0 — the always-on learner (the paradigm as a living service)

[organism/](organism/) is the paradigm deployed: **start it and it keeps learning, forever**
— an ε-gated two-speed learner (the 3×-validated policy, verbatim) living on a scheduled
3-modality world (wiki · Shakespeare · MNIST-as-pixel-tokens), on a persistent EC2 CPU box.
Restart-safe to the bit (Gate-2-tested: kill -9 ≡ uninterrupted), append-only audit ledger
(every weight change attributable), clone-isolated probes every 6 h (per-domain bpc, digit
generation, plasticity), systemd + watchdog resilience, and an S3-only CLI
(`python -m ontogenic_learner.organism.cli status|sample|report|feed|terminate`).
First life: 14-day pilot (plan: `~/.claude/plans/…majestic-riddle.md`; launcher:
[scripts/run_organism_ec2.sh](../scripts/run_organism_ec2.sh); fabric warm-start swap:
[organism/FABRIC_SWAP.md](organism/FABRIC_SWAP.md)). Honest calibration: this is
*operational* train≡inference; the settling-based unification remains the U-gate track.

## 7. Status log (newest first — keep updating this)

- **2026-07-14** · **V-GATE PROGRAM COMPLETE (except V5, in flight).** V0–V4
  all pass as pre-registered; two honest negatives published with mechanisms
  (V2b gate law; V4b Fisher-under-sparse-firing). Headlines: the fused
  serve+learn loop measured at 1.38× ops; a 10M-token single-process life at
  1.0063× serving cost with retention, norm equilibrium and a fully audited
  update stream; the incumbent Pareto held. See §2b and
  NORTHSTAR_VALIDATION.md. All results/telemetry committed on pc-morphssm.

- **2026-07-08 (later)** · **PILOT ENDED EARLY BY DECISION at life-day 3.3** (graceful CLI
  terminate: final checkpoint → ack → instance gone; ~$28 total of the $135 budget). Life
  record preserved in full at `s3://tiki-sweep-739966866637/organism/org-20260704T161956Z`
  (16 probe suites, 4.39M ledger events, gaps=0; final report snapshot in
  `organism_report_sync/`; minor open item: report's monotonicity checker flags
  out-of-order lines with zero gaps — likely a rotation-read-order artifact in report.py,
  check before the next life). What the 3.3 days bought: zero-incident always-on ops incl.
  two abuse recoveries (shakedown), the norm-growth/novelty-starvation finding, the
  fast-adapt-then-overfit and interference curves, the cross-modality monomodal collapse,
  and the G4 mechanism hunt it triggered (anchor GO). The line now waits on the recurrent
  CNN PC architecture; the harness, the anchor, and the pathology catalogue carry forward.

- **2026-07-08 (decision)** · **PAUSE deepening this line pending the real PC architecture.**
  Assessment (user + instance concur): the organism's model is a placeholder (ctx-24 MLP —
  no recurrence/conv/settling; shared output layer), and the deployed rule is BP+gate+buffer
  (cascade==BP), so mechanism *magnitudes* here don't transfer; the cross-modality collapse
  at day 3 is partly architecture artifact. The pilot RUNS TO TERM (day-8 novelty + revisit
  recovery are still-missing headline data; pre-registration holds), but **v0.1 is NOT born
  on the MLP** — it waits for the recurrent CNN PC architecture (other instance), then =
  that architecture + G4 anchor + this harness (FABRIC_SWAP warm-start). G4 phase-2/3
  (layerwise Π, k-WTA) also deferred to land natively in that architecture. What transfers
  regardless: the harness (model-agnostic by design), the G4 design rules (fixed-point
  consolidation; weight-space > buffer-scaling; gate-satiety failures), and the longitudinal
  pathology catalogue any successor must beat.

- **2026-07-08** · **G4 COMPLETE — the anchor is how you coax continual learning out of the
  PC learner** ([G4_SPEC.md](G4_SPEC.md), 18 runs / 12 arms / 4 rounds in one day). The CL
  hypothesis was restated honestly (cascade==BP ⇒ the *rule* can't carry CL; the discarded
  Bayesian terms must) and tested mechanism-by-mechanism on a compressed wiki→shak→wiki life.
  **Winner: the precision-weighted weight-posterior anchor** (`grad += λ·v̂·(θ−θ_EMA)`,
  v̂ = slow-Adam exp_avg_sq — the G3 Fisher ingredient turned inward; the restored
  `p(W)=N(θ_post,1/v)` term): interference −38%, end-of-life retention +10pp (no seed
  overlap), norm halved *and flattened* (first equilibrium observed), at **zero adaptation
  cost**. Add fire-less τ for SPJS mode (−46% interference, 2.6× fewer kicks, ~0.2 bpc tax).
  Three honest negatives with mechanisms: epistemic gate CE−H is *anti*-satiating (miscalibrated
  at ignorant init); per-param grad damping is nullified by Adam's own normalization (consolidation
  must move the fixed point); parity-strength replay through a 4096 buffer *degrades everything*
  (buffer memorization — weight-space memory beats data-space memory at organism scale). The
  workflow's sleep-inversion claim was refuted against the live checkpoint before building
  (reservoir healthy: 77% wiki slots, priorities ≈ uniform). **v0.1 birth config: two-speed +
  AdamW + anchor.** Pilot untouched throughout — it is the control life.

- **2026-07-07 (later)** · **Pilot day 2.2 — the Shakespeare shift, first post-shift probe.**
  Three findings in one measurement (probe 11, ~4 h ≈ 370k steps into the new domain):
  **(1) Fast adaptation** — shakespeare bpc 28.61 → 4.10 in 4 hours of experience.
  **(2) Partial retention** — wiki 2.88 → 4.00 (+1.13 interference despite sleep-cycle
  reservoir rehearsal; well short of collapse; the day-4–6 mixed-revisit phase tests
  recovery speed). **(3) The norm-growth natural experiment: the curve BENT** — slope
  dropped from ~+55/0.5M steps (wiki era) to +16/0.5M post-shift, supporting the
  overconfidence-sharpening hypothesis: unbounded growth is a *novelty-starvation* effect
  (stationary-data sharpening), not intrinsic to the always-on policy. v0.1 still gets
  weight decay, but the pathology self-limits under distribution drift — i.e. exactly the
  regime the organism is built for. Gate behaved as designed: pinned at the 0.30 cap
  through the shift (rate limiter working), 0.294 settling after.

- **2026-07-07** · **Pilot day 1.8 — first longitudinal finding: unbounded weight-norm
  growth.** Ops are spotless (4.07M steps, 0 incidents, 0 resumes, gate steady 0.29, ~$17).
  But the probe series shows mean weight norm growing near-linearly 26 → 404 over 4M steps
  (neither Adam has weight decay — the G1/B1 lives were too short for it to matter), with
  the expected signatures: wiki bpc drifting up 2.47 → 2.81 (overconfidence on the
  surprise-gated hard tail), unseen-domain bpc exploding (digits → 2453 as never-seen pixel
  symbols get pushed to −∞), plasticity adapt_gain inflating 2.1 → 7.0 (gradient-scale
  artifact, not real plasticity). **Decision: observe, don't operate mid-life** — this is
  exactly the "policy stability at 10⁷ steps" risk the pilot was built to measure, and the
  imminent day-2 Shakespeare shift is the natural experiment (fresh error signal may or may
  not bend the norm curve). If growth persists post-shift, the v0.1 phylogeny gets
  AdamW-style decay (a *new birth*, never mid-life surgery). This finding alone justifies
  running lives longer than benchmarks.

- **2026-07-05** · **🌱 THE PILOT IS BORN** — `i-0ea5e986108ac2a00`, life
  `org-20260704T161956Z`, a pristine 14-day life at measured pacing (2.3M steps/day).
  The shakedown before it passed everything: all local gates (bit-resume kill -9 ≡
  uninterrupted; probe isolation state-hash-exact; monotonic crash-safe ledger), full ops
  on a smoke box (live S3 sampling, terminate-ack), and an 18-hour full-scale run with
  **two abuse tests survived** (SSM process kill → systemd restart+resume; full reboot →
  auto-resume), 5 probe suites, wiki bpc 2.65 and healthy clone-plasticity, **zero
  incidents across 610k ledger events**. Shakedown finding worth keeping: at τ=0.25 the
  z-scored gate is self-normalizing — it filters the hardest ~30% of batches even on
  stationary data (starvation lives at higher τ). Two shakedown bugs fixed before the
  pilot (gz-aware ledger recovery; prompted text probe — the space-attractor). Interact:
  `python -m ontogenic_learner.organism.cli status|sample|report|feed`.

- **2026-07-03 (night)** · **B1 benchmark anchoring COMPLETE** ([BENCHMARKS.md](BENCHMARKS.md)):
  **Split-MNIST class-incremental 82.4% (naive collapses to 19.7%; rehearsal 89.7% at 3.5×
  the updates; joint 97.5%)** — rehearsal-class continual learning with the ε-gate declining
  71% of the budget; permuted-MNIST 91.4% at half budget. **TinyShakespeare: i.i.d. ceiling
  2.427 bpc — within 0.004 of the S1 silicon LM (2.4230)**; budgeted single-pass stream arms
  2.62–2.74; generations show real word/line structure. Boundary honestly mapped: the
  machinery pays under distribution shift; on stationary streams it's an energy knob
  (−27% updates for +0.12 bpc). README §5 added: relation to the PC-LLM fabric program.

- **2026-07-03 (later)** · **G1b runs 2–5: the debugging arc — mechanical cert STANDS; an
  L=2 silicon defect found, cornered, and dossier'd.** Window-gating v2 measured a **+34%
  throughput/energy dividend** on silicon (282 vs 211 tok/s; 1,096 BWD + 959 FWD kicks never
  issued at 72.6% window fire-rate). Quality saturation was then root-caused through five
  cheap runs + a free CPU differential: recipe eliminated → topology exonerated (CPU NBLK=2
  converges, bpb 3.76) → AGFI verified → forward healthy at init → **single-batch overfit
  climbs monotonically at ANY LR** = wrong-direction updates in the silicon backward at L=2
  (a configuration never trained before). Full evidence chain + handoff:
  [results/g1b_cert/G1B_RESULT.md](results/g1b_cert/G1B_RESULT.md) — fabric instance to run
  the one-kick L=2 KAT / L=3-vs-L=2 on 48xl. Total spend ≈ $12 / 5 boxes.
- **2026-07-03** · **★ G1b RUN 1 — THE ONTOGENIC LEARNER RAN ON REAL SILICON.** Fresh
  f2.12xlarge (own box, zero collision), AGFI on both slots, 500 gated steps + 500-step
  control, both EXIT=0, self-terminated (~$3). **Mechanical certification PASSED**: ε-gate
  fired 72.8%, **136 backward kicks never issued** (physical energy saving; two-speed ran
  8% faster wall-clock than control), 45 sleep replays from a surprise-weighted reservoir,
  kick anchors reproduce the fabric program's exactly (FWD 53.5/57.6 · BWD 91.5/99.1 ms),
  Gate-C clean, and the **audit ledger exists** (409 applied updates, each attributable and
  replayable — the certifiable-lifelong-learning artifact). Honest catch: both arms diverged
  at XF_LR=3e-3/ACCUM=1 (config, not patch — identical trajectories) → quality leg **run 2
  relaunched at 1e-3** (`i-0f8129d554fe159c5`). Details:
  [results/g1b_cert/G1B_RESULT.md](results/g1b_cert/G1B_RESULT.md).
- **2026-07-03** · **Mission sharpened (user): the end-state is train≡inference** — one PC
  settling computation, clamped=learn / free=infer (now PARADIGM.md principle 1, three
  stages). **U-gate probe run (G2_SPEC A2): "not yet" — stability SOLVED** (plain settle
  NaNs; RMS-normalized µ-settle finite in every config at width 768, the G2a-P1 divergence
  fixed) **but learning quality stays gated** (+1.6–1.8 bpc to cascade) → next iteration =
  Depth-µP-proper reparameterization, research-track.
- **2026-07-03** · **G3-F2 COMPLETE: GO.** Fisher-style merging — per-parameter importance =
  each learner's own local Adam second moment — **eliminates the fleet–oracle gap**
  (+0.007 vs naive's +0.267, a 97% cut), beats the centralized oracle on the near shard
  (3.011 vs 3.183) and base retention (2.891 vs 3.102), replicated across seeds, with zero
  data movement and zero additional communication. Ablations: TIES sign-resolution *hurts*
  (−59%); fixed asymmetric weights help one side by taxing the other (30%). The mechanism is
  per-parameter, locally-computed importance. **Fisher-merge is now the fleet default.**
  In parallel: G1b track opened — isolated worktree `/tmp/tt-g1b`, ε-gated TWOSPEED driver
  patch + own-instance launcher (fresh `f2.12xlarge`, timestamped tag, own S3/state — no
  collision with the fabric instance's boxes) being written; AWS SSO pending user's browser.
- **2026-07-03** · **G3-F1 capacity sweep COMPLETE: the fleet–oracle gap is scale-invariant**
  (+0.29/+0.27/+0.29 bpc at 0.9M/4.1M/15M — a 17× range) and lives entirely on the distant
  shard at every size (near-shard reaches oracle-parity at 15M; fleet base retention beats
  the oracle at every size). Pre-registered conclusion: **structured merging (F2) is
  mandatory** — capacity will not close it. Also: **G1b turnkey silicon-certification
  protocol spec'd** → [G1B_SPEC.md](G1B_SPEC.md) (knob mapping code-verified, zero rebuild,
  ~1–2 F2 spot-hours) — ready for the fabric instance.
- **2026-07-03** · **G3 COMPLETE — PARTIAL: the fleet mechanism is real, naive averaging is
  capacity-limited.** Two two-speed learners on maximally distant shards (Shakespeare / code),
  exchanging only weights: away-competence transfers massively through weight-space
  (solo-away 7.08 → fleet 3.60 on code), periodic merging ≫ final-only, and the merged model
  is the best *generalist* in the experiment — beats both specialists as a single model and
  beats even the centralized pooled oracle on base retention (wiki 2.888). But it concedes
  the distant shard to centralization (code 3.596 vs 3.020; oracle gap 0.26 > the amended
  0.15 bar), and the pre-registered amendment (merge-then-consolidate sleep) came back null —
  the gap is shared *capacity*, not merge damage (the oracle pays the same dual-specialist
  cost). Forward: capacity-scaling of the fleet–oracle gap; structured/Fisher/modular merging
  (→ the hot-add / tropical-MoE thread). Logs: [results/g3_fleet/](results/g3_fleet/).
- **2026-07-03** · **G2a pilot COMPLETE — an honest null for the rule, a third win for the
  schedule.** (i) The prospective/settle regime fails to condition at width 768 in every
  config (P1) — testing it at scale requires µPC-style reparameterization, filed as
  research-track; G2b GPU spend paused (spec amendment A1). (ii) The stable non-converged
  form (async error-relaxation, p=0.5 — the fabric-native regime) is *exactly* BP-equivalent:
  H1 interference, H2 few-shot, and H3 plasticity-decay all indistinguishable, and the one
  apparent difference (low-LR retention) was fully attributed to learning rate by a `bp_lr4`
  control (3.690/−0.097 ≡ 3.696/−0.090). Sync-relax reproduced BP bit-for-bit (machinery
  validated). (iii) **The G1 schedule replicated at 9× scale** — best H1 Pareto point again —
  and on H3 cyclic streams it dominated final quality on *every* domain (wiki 3.666 vs
  bp 4.107): two-speed converts relearning into retention. Also: FLOP-rich BP (9× updates)
  made forgetting *worse* (+1.041 vs +0.750) — more compute is not a continual-learning fix.
  All on local MPS (measured 10× CPU); ~45 min total. Logs:
  [results/g2_rule_probe/](results/g2_rule_probe/).
- **2026-07-03** · **G2a pilot launched** — harness
  [experiments/g2_rule_probe.py](experiments/g2_rule_probe.py) (h1/h2/h3 probes, FLOP-matched
  controls, arm grammar with divergence guards).
- **2026-07-03** · **Preprint skeleton live** — [paper/](paper/) (`main.tex`, compiles clean;
  figures auto-generate from the result logs via `paper/figs/make_figs.py`; `make` rebuilds
  everything). **G2 spec'd and pre-registered** — [G2_SPEC.md](G2_SPEC.md): 4 hypotheses,
  FLOP-matched comparisons, modern-countermeasure baselines, scale ladder G2a (local, $0) →
  G2b (30M GPU decision rung, ~$100) → G2c (124M optional); go/no-go frozen before any run.
- **2026-07-03** · **G1.2 verdict: GO** — two_speed beats ALL four strengthened rehearsal
  controls (z∈{0.00, 0.25} variants); τ sweep maps the retention–plasticity–energy dial
  (near-zero forgetting at 40% energy, τ=1.5). Raw: `results/g1_two_speed/g1_full_g12.jsonl`.
- **2026-07-03** · Project home created: `ontogenic_learner/` (this README = the living doc);
  experiment gains the G1.2 arm grammar (rehearsal[50][_bal], two_speed_*[_zNNN]) and
  repo-root path robustness.
- **2026-07-03** · **G1.1 clean** — settled-PC sleep stabilized (settle=8, lr_slow=1e-4,
  γ=0.0625, snapshot/restore guard): 0 divergences, Pareto-beats rehearsal (+0.458 forget,
  3.522 final, 0.97 plast), but cascade-sleep remains the better Pareto point at ~60% energy.
- **2026-07-03** · **G1 verdict: GO** — two_speed_casc Pareto-beats rehearsal-backprop across
  all 3 seeds at matched budget; ε-gate selectivity ~30%. First attempt at settle=16/γ=0.125
  NaN'd (known iterative-divergence mode) → G1.1.
- **2026-07-03** · Paradigm defined ([PARADIGM.md](PARADIGM.md)); G1 pre-registered and
  implemented; corpora localized (Gutenberg unreachable → wiki/shakespeare/docs/code stream,
  a harder and better-ordered shift).
- **2026-07-03** · Strategy locked upstream: three adversarial panels (15 web-grounded lenses)
  established the experience×time framing, the SPJS metric, the gate ladder, and the ~2-year
  window. See memory + `results/graphics/` figs 01–06 for the program-wide context.
