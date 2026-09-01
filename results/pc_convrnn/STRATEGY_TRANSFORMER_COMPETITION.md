# Can dynamic activations compete with dynamic weights? (the PREREG-owed analysis)

Three-perspective panel (mechanism / economics / red-team), 2026-07-08, grounded
in the measured record only. Question (user): can PC-settled tropical sequence
models — dynamically computed activations — ultimately compete with the
transformer's dynamically computed weights?

## The one-sentence verdict

The fight is **not symmetric and not single-front**: head-on frontier-LM quality
and one-shot content-addressed recall are structurally the transformer's
(settling inherits, rather than fixes, what the forward can score); the **modes
axis** (bidirectional constraint satisfaction on an unmodified causal
checkpoint) and the **joules axis** (zero-multiply training at BP-twin quality
on zero-DSP silicon, always-on streaming learning) are genuinely contestable —
and everything generative is **gated by base-model quality**, which is the
program's critical path.

## 1. The mechanism asymmetry (why emulation is the wrong goal)

Attention manufactures a per-token linear operator from content — all-pairs
retrieval in ONE forward pass, composing across layers. PC settling computes a
MAP fixed point under the model's density: `z* = argmin_z E(z; clamped
context)`. **Settling adds no connectivity; it inverts the forward it is
given.** Retrieval-by-settling reduces to retrieval-by-density-scoring, and the
stack's recall scoring is measured partial and seed-brittle (G2b induction:
~24%, 2/3 seeds; 1/3 at chance). The G6 smoke makes the gating quantitative:
the clamped objective HAS signal (lightcone-CE truth 1.947 < greedy 2.159), but
coordinate descent drives CE to 1.842 — *below the truth* — while exact-match
falls 0.25 → 0.06. At 1.93-nat base quality the energy landscape ranks
degenerate fills above the answer: **inference-time settling is only as good as
the density it descends.** (The transformer world learned the same lesson about
inference-time compute on weak base models.)

Two things survive this asymmetry with measured support:
- **Selectivity / streaming / length-gen**: selective copy beyond RF (23–68% vs
  chance controls), train-96/eval-1024 degradation +0.032 vs +0.096, O(d)
  streaming state vs the KV cache. The tropical stack owns this axis.
- **The modes claim, narrowly**: bidirectional conditioning on the SAME causal
  checkpoint, no retraining — unique among causal models. A transformer shop
  answers with FIM/masked pretraining (a training-time decision, zero inference
  cost); our capability is post-hoc. Honest phrasing only.

**The strongest repo-native move is hybridization, not emulation:** the
l1-score + argmax-GATHER attention — already measured lossless on induction at
hard deploy, and routing-discrete (L1 adjoint dense-but-shift-add; gather
adjoint = a scatter) — is cascade-compatible and zero-multiply at τ=0. One such
layer inside the MorphSSM stack is the recall organ this family lacks, inside
the closure envelope. Pre-registered as the follow-up arm.

## 2. The economics segmentation (where the fight is winnable)

The repo's own arithmetic closes the datacenter lane: 124M-scale pretraining ≈
decade-per-card (TOKS_PROJECTION.md), and RUNG3 concedes async-backprop
neutralizes the throughput argument on clocked hardware. **Do not position
against batch pretraining/serving.** The contestable cost function is
**experience-per-joule on always-on streams (SPJS)**: local, async-tolerant
credit (G1: exact once hops cover depth), zero-multiply sweeps, O(d) state, and
the two-mode checkpoint map directly onto the ontogenic two-speed learner
(cascade = wake plasticity at forward cost; relaxation = consolidation/
generation on demand — same weights). The transformer's global backward + batch
structure is a *structural* penalty on that scoreboard, not an implementation
one. Critical-path measurement before using this competitively: **board-level
PMBus tokens/J at an iso-quality operating point** (the repo's own "one
remaining measurement"); core-rail projections don't count.

## 3. The bear case, kept in view (trajectory evidence is currently bearish)

(1) No scaling data for the substrate tax, and all three trajectory signals
point the wrong way: the 0.15–0.23-nat tax was *widening* at budget end; BPE
reversed the conv advantage toward dense GPT; the text8 mult-free-attention tax
was scale-invariant. (2) Modern LM capability is recall-shaped (in-context
learning ≈ induction), which is the family's measured edge. (3) The strict
(fully-closed) training rung costs down to ≈ bigram today — the anneal is
load-bearing. (4) The generative mode runs at soft τ (outside the zero-multiply
envelope) and costs ~10–100 forward-equivalents per query. The branch's proven
result is a **cost theorem, not a capability theorem** — it makes the substrate
cheap to train at ITS quality; it does not close the quality gap.

## 4. Pre-registered flip conditions (what would change the verdict)

- **M1 — substrate-tax scaling ladder (decisive; next major spend):**
  {0.5M, 5M, 50M} params × BPE × paired {MorphSSM-cascade, DenseConv-twin,
  dense GPT}, 3 seeds. Flip: paired tax at 50M ≤ half its 0.5M value, monotone.
- **M2 — recall gate:** MQAR/induction at gaps ≥4× RF, task loss only, 3/3
  seeds ≥90% where attention ≥95% — OR the l1+gather hybrid inside MorphSSM
  reaching the same bar at hard deploy.
- **M3 — quality-conditioned settling crossover:** G6b across checkpoints at
  ~1.93/~1.85/~1.75 nats; flip = Δ(settled − matched-compute greedy) EM turns
  positive and grows as base CE improves.
- **M4 — cross-family uniqueness:** matched-param FIM-causal-transformer
  infilling baseline; flip = settling ≥ FIM at matched wall-clock.
- **M5 — iso-quality board energy:** PMBus tokens/J ≥3× vs GPU at an
  iso-quality point ≥ trigram.

## 4b. NORTH STAR (elevated by user 2026-07-10/11; validation pre-registered)

*"A device sits on its stream; every example it serves, it may also learn
from, at about the cost of serving it twice, with no extra silicon, no extra
memory class, and no synchronization with anything else."* Two separate but
related goals: **Goal A** — the live loop (one process, serve + intermittent
in-place learning at ~2×; a mechanism claim) and **Goal B** — lifelong
retention (what is learned persists usefully in the weights; a
continual-learning claim). Validation = the V-gate ladder in
`ontogenic_learner/NORTHSTAR_VALIDATION.md` (V0 fusion/cost-audit → V5
full-card PMBus), fusing this branch's substrate with the ontogenic harness.

## 5. Program implications (adopted)

1. Reframe the competitive claim: *same-checkpoint dual-mode training+inference
   on a zero-multiply substrate for edge/streaming/always-on adaptation* — not
   "compete with frontier transformers." SPJS is the official scoreboard.
2. Sequence base-quality (BPE, larger d, more tokens) BEFORE claiming
   generative-mode wins; G6 is option value priced by M3.
3. Pre-register the l1+gather hybrid arm (the recall organ) and the M1 scaling
   ladder as the next two experiment programs after the current EC2 batches.
4. Close honesty items before economic claims: PMBus; G5 machine-audited
   zero-multiply training; account the generative mode's soft-τ multiplies and
   wall cost inside the closure story.
