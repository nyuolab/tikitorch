# Ontogenic Learning — a low-energy, always-learning paradigm for the experience×time era

**Status:** paradigm definition v0 + the G1 decisive experiment (this directory).
**Instance note:** this workstream runs on the main-repo instance (CPU/torch twin); the other
Claude instance continues scaling the on-fabric LLM in `/tmp/tt-pc` — no F2 usage, no worktree
touches from this side. The two meet at G1b (fabric certification of the winning schedule).

---

## 1. The claim

The parameters×tokens era belongs to backprop-on-GPU and always will. The **experience×time**
era — systems that learn continuously from non-i.i.d. streams, for their whole deployed life,
within a power budget — has **no incumbent**: everything shipped today is either session-scoped
fast weights (TTT resets per session) or centralized fleet *data* loops (weights still train in
the datacenter). Biology is the existence proof that the alternative works: local rules,
error-gated plasticity, wake/sleep consolidation, ~20 W.

Ontogenic Learning is the paradigm for that era. It is named for what it optimizes: not the
species prior (phylogeny — batch pretraining, GPUs win it, keep them), but the **lifetime of
one deployed system** (ontogeny).

## 2. The six principles

1. **Learning is a property of deployment, not a phase — and the end-state is stronger: no
   difference between training and inference at all.** The definitional target (user,
   2026-07-03) is ONE computation — predictive-coding settling — where clamping the output
   means learning and leaving it free means inference. Same dynamics, same silicon, same
   energy budget. Shipped in three honesty-graded stages: *(i) operational* unification —
   the deployed substrate trains in place (the two-speed learner; measured; silicon
   certification G1b); *(ii) computational* unification — the settling regime itself, gated
   on conditioning-at-width (G2a-P1 → the U-gate probe, G2_SPEC amendment A2); *(iii)
   capability* payoff — the same weights generating, inpainting, and recalling by clamping
   different subsets (the SparseCodingPC line, the measured backprop-impossible win).
2. **Credit is local.** Per-layer error units, neighbour-only messages, no global gradient, no
   weight transport. (This is what makes principles 1, 4, 5 physically cheap — and it is the
   fabric-native and biologically plausible part.)
3. **Plasticity is error-gated.** Learn from surprise; skip the boring. The error magnitude ε
   is the neuromodulatory signal deciding WHEN to learn and WHAT to remember. Most stream
   batches should cause **no update at all** — that is where "low-energy" comes from.
4. **Two speeds: wake and sleep.** Fast plastic updates on novel data (cascade kicks); slow,
   conservative consolidation over a surprise-weighted replay reservoir (iterative/settled PC
   passes at low LR) — complementary learning systems, on hardware knobs that already exist
   (`DO_UP`/`LRBITS`/`GZERO` per kick).
5. **Every update is auditable.** Bit-exact training means a deployed learner can *prove* what
   it learned and from which data (~1% spot-replay overhead, measured). No other stack offers
   this; regulated/disconnected fleets are the beachhead.
6. **Individuals consolidate into culture.** Fleet-level outer averaging (DiLoCo-style, no
   per-step sync) merges lifetimes; the fleet is the training run, and its situated experience
   is a data moat no batch pipeline can buy.

## 3. The metric: SPJS (score-per-joule-per-stream)

One tuple per system on a standardized drifting stream, no i.i.d. reshuffling:
`{final capability, retention (forgetting matrix), plasticity (adaptation speed), adaptation
latency, energy (updates applied + FLOP-equivalents; watts at the wall when on hardware),
audit coverage (% of updates bit-exactly replayable)}`.
GPU baselines (rehearsal, LoRA, memory layers) are first-class entries: the leaderboard is
honest only if the incumbent can enter and win columns.

## 4. What we are NOT claiming (the honesty ledger)

- The fabric's converged **cascade mode is gradient-checked backprop** — running it sequentially
  and calling it a new paradigm would be fraud. The paradigm content is the *regime and schedule*:
  gating, two speeds, settled consolidation, locality, audit.
- The repo's own prior test (`scripts/pc_continual.py`, permuted-MNIST) logged the honest
  negative: iterative PC retains better (0.21–0.42 vs 0.06–0.27) but is ~4× less plastic and
  loses average accuracy. **The two-speed learner is the designed answer** — fast cascade for
  plasticity, iterative only for consolidation — and G1 tests exactly that.
- The baselines to beat are 2025–26 SGD countermeasures (rehearsal here; sparse-memory-FT and
  TTT in later rounds), not vanilla fine-tuning.

## 5. G1 — the decisive experiment (runs now, CPU twin)

**Question:** does the two-speed learner beat rehearsal-backprop on the retention–plasticity
Pareto at matched update budget — and does ε-gating buy it at lower energy?

**Setup.** Bengio-style context char-LM on the measured local-PC arch family (`DyadicPCNet`,
multiplier-free ±2^k substrate, context 12, hidden 256, depth 4, char-64-compatible vocab).
Phylogeny: pretrain once on a 5M-char Wikipedia slice (`text8`); every arm starts from the
same checkpoint. Ontogeny stream (strictly sequential, non-i.i.d.): **Shakespeare → repo
Markdown docs → repo source code** — three large, cleanly ordered distribution shifts, fully
local/offline-reproducible (Gutenberg was unreachable; prose→technical-prose→code is the
harder and better-instrumented shift anyway). 1,500 stream batches per domain, update budget
**B = 600 per domain** for every learning arm. All replay reservoirs are seeded with
base-domain memories (fairness: every remembering arm starts with the same prior life).

**Arms.**
| arm | wake path | sleep path | replay | gate |
|---|---|---|---|---|
| `frozen` | — | — | — | — |
| `cascade` | cascade @ 600 even slots | — | — | none |
| `rehearsal` | cascade @ 600 even slots | — | 25% uniform reservoir | none |
| `two_speed_casc` | ε-gated cascade (≤450) | low-LR **cascade** on reservoir (150) | surprise-weighted | z > τ |
| `two_speed_iter` | ε-gated cascade (≤450) | low-LR **iterative settle** (`train_step`, n_steps=16) on reservoir (150) | surprise-weighted | z > τ |

`two_speed_casc` vs `two_speed_iter` is the critical ablation: it separates the *schedule*
effect (gating + sleep + prioritized replay) from the *rule* effect (settled-PC consolidation).

**Metrics.** Full transfer matrix (bpc on every domain's held-out tail after every phase),
forgetting per domain, plasticity fraction vs `cascade` (Δbpc on the new domain as a fraction of
cascade's Δbpc), updates applied, forward-pass-equivalents (energy proxy), wall-clock. ≥3 seeds.

**Go/no-go (pre-registered):**
- **GO** — a two-speed arm keeps ≥70% of cascade plasticity AND Pareto-dominates `rehearsal`
  (better retention at ≤ equal plasticity loss and ≤ equal update budget). If only
  `two_speed_iter` achieves it, the PC rule itself is load-bearing → G2 (scale the regime).
- **PARTIAL** — two-speed matches rehearsal's Pareto point at materially lower applied-update
  energy → the low-energy always-on claim survives; the retention claim demotes.
- **NO-GO** — rehearsal dominates → the paradigm contracts to auditable-fleet economics
  (systems claim only); publish the negative next to the positive infrastructure.

**Follow-ups:** G1b — replay the winning schedule through the bf16 golden kernel config
(D=64/H=2/FFN=256/SEQ=64) and certify one silicon kick (the fabric instance's turf, when free).
G2 — prospective-configuration at LM scale on GPUs. G3 — fleet merged-beats-solo on 2–3 chips.

## 6. Biological mapping (the design's provenance, not its proof)

wake/sleep ↔ fast kicks / consolidation passes · ε-gating ↔ neuromodulatory surprise signals ·
surprise-weighted reservoir ↔ hippocampal prioritized replay · slow settled updates ↔ cortical
consolidation · locality/no-weight-transport ↔ the standard biological objections to backprop ·
fleet outer-averaging ↔ culture. Bio-plausibility here is a hypothesis-generator; every claim
above is settled by the SPJS numbers, not the analogy.
