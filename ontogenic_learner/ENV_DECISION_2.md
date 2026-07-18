# ENV_DECISION_2 — the environment portfolio after E4d (2026-07-18)

*Supersedes-and-extends E2_ENV_DECISION.md. Question (user): is MinAtar the
best standard env? XLand-MiniGrid or a classic alternative? Decision
(user-approved): the frozen trio stands for E5; XLand-MiniGrid is designated
the future generalization gate.*

## Requirements an env must meet for THIS program

CPU-fast single-env stepping (one life = one env; no vectorization);
symbolizable obs ≤ ~100 cells (token frames); small constant action set;
continuing-or-episodic with clean done semantics; pip-clean permissive
license; bit-exact serializable (E5 resume bar); ideally published baselines
for honest comparison.

## The portfolio (unchanged for E5)

| world | role | status |
|---|---|---|
| foraging (bespoke) | primary/control — dense-ish reward, biome drift, oracle access | all gates run here; floors + sAC baseline |
| MinAtar (5 games) | standard benchmark — published streaming-RL numbers, constant K=6 raw API, 10×10 symbolic | E2-adopted; head config beats sAC late-life |
| BabyAI/MiniGrid (fixed-mission set) | the third world — sparse episodic reward, mid-life arrival readout | adapter built, bit-exact, 12 syms; E5b gate screen pending |

## Alternatives assessed

- **XLand-MiniGrid** — JAX meta-RL suite; millions of composable rule/goal
  tasks; ~1M+ steps/s via GPU/TPU vectorization (arXiv:2312.12044).
  DESIGNATED the generalization gate (post-E5): task DISTRIBUTIONS test
  systematic generalization/meta-adaptation — a different claim than
  persistence, and the right next claim after a retention result. Not for
  E5: its speed advantage vanishes at single-env CPU stepping; heavy JAX
  dep on the life box; and swapping now would re-open a frozen prereg with
  drilled adapters. Engaged prior art for that future gate: task diversity
  vs continual RL (arXiv:2606.00880) — directly adjacent to our
  interference/transfer readouts.
- **Craftax/Crafter** — achievement-based survival; Craftax is JAX (same
  single-env caveat); Crafter is pixel-first (64×64×3 → symbolization is a
  research project of its own). Rejected for now.
- **MiniHack/NLE** — fast C core, symbolic; but large action spaces (up to
  ~100) break the shared-action-block codec, and task structure is deep.
  Rejected for the token-agent tier.
- **Classic control (CartPole etc.)** — near-continuous obs needs binning,
  trivial dynamics add little credibility. Rejected.
- **Procgen/Atari** — pixel/GPU-oriented; wrong cost class. Rejected.

## Standing decision

E5 lives in the trio. XLand-MiniGrid becomes **E6-G (generalization gate)**
after the life: pre-sampled task distribution, the same agent recipe,
zero-shot + few-episode adaptation readouts against its PPO baselines —
scoped when E5's retention verdict is in.
