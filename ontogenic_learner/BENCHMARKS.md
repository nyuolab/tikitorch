# B1 — benchmark anchoring: real-task capability numbers (pre-registered 2026-07-03)

**Why:** the gate ladder proved *relative* wins (Pareto vs rehearsal at matched budget) on our
own stream. B1 anchors the ontogenic learner to **absolute, recognizable numbers** on the two
tasks this repo has history with, so its capability is legible against both the continual-
learning literature and the fabric program's own bars.

## B1-MNIST — continual learning on the canonical benchmarks

Protocols (both standard):
- **Permuted-MNIST, 5 tasks** (the repo's own `pc_continual.py` benchmark — direct lineage to
  the logged honest negative).
- **Split-MNIST, class-incremental, 5 tasks** (0/1→2/3→…→8/9, single 10-way head, no task
  labels at test — the hard protocol; literature anchors: naive ~20%, EWC ~20%,
  rehearsal/GEM ~80–90%, joint ~98%).

Substrate: DyadicPCNet MLP (784-256-256-10), cascade updates (==BP — the validated fabric
rule). Arms at matched update budget per task: `naive` · `rehearsal_bal` (25% domain-balanced)
· `two_speed` (ε-gated fast + surprise-weighted sleep) · `joint` (i.i.d. oracle). Metrics:
accuracy matrix after every task, final average, forgetting, updates applied (energy). 2 seeds.

**Pre-registered success bar:** two_speed lands in the rehearsal-class band (≥75% final avg on
Split-MNIST class-inc; ≥90% on permuted) while beating rehearsal_bal on forgetting at ≤ its
update budget — i.e., the G1 result generalizes off our own stream onto the canonical tasks.

## B1-Shakespeare — LM quality with absolute bars + generation

RAW TinyShakespeare (full ~65-char vocab, NO normalization — comparable to `pc_charlm.py` and
the golden), ctx 16, DyadicPCNet context-LM ~4.4M params. Bars computed in-run:
**bigram floor** and an **i.i.d. ceiling** (same arch, cascade, 3 epochs of random batches —
the "unconstrained training" reference). Repo context bars: bf16 golden transformer = 3.07 bpb
(coherent); nanoGPT-class ≈ 2.0–2.1 bpc.

Stream arms (the ontogenic protocol: ONE sequential pass, budget 1,500 updates):
`stream_cascade` · `stream_rehearsal` · `stream_two_speed`. Metrics: val bpc (last-10%
held-out), updates applied, and **300-char generation samples** (the capability check a
number can't fake). 2 seeds.

**Pre-registered success bar:** two_speed ≥ stream_cascade on val bpc at ≤ budget, within
0.35 bpc of the i.i.d. ceiling (a single budgeted stream pass vs unconstrained training), and
generations show word/line structure (not bigram babble).

## Results (2026-07-03, 2 seeds, MPS; raw: `results/benchmarks/b1_*_full.jsonl`)

### B1-MNIST — rehearsal-class accuracy at a fraction of the energy

| protocol | naive | rehearsal | **two_speed** | joint oracle |
|---|---|---|---|---|
| Split-MNIST (class-inc) — acc | 0.197 | 0.897 | **0.824** | 0.975 |
| — forgetting | +0.995 | +0.115 | +0.138 | — |
| — updates | 2,000 | 2,000 | **573 (29%)** | 2,000 |
| Permuted-MNIST — acc | 0.872 | 0.945 | **0.914** | 0.959 |
| — updates | 2,000 | 2,000 | **1,026 (51%)** | 2,000 |

Protocol self-validates against the literature anchors (naive 19.7% ≈ the canonical ~20%
collapse; joint 97.5% ≈ ~98%). **Verdict vs the pre-registered bar: accuracy bands PASS
(≥75% split ✓ 0.824, ≥90% permuted ✓ 0.914); the strict beats-rehearsal-on-forgetting clause
does not** (rehearsal edges +0.115 vs +0.138 while spending 3.5× the updates). Honest
headline: **rehearsal-class continual learning with the ε-gate declining 49–71% of the
budget** — the low-energy claim on the canonical benchmark; the τ dial trades the residual
forgetting gap when a deployment wants it.

### B1-Shakespeare — absolute bars, and the domain of advantage mapped

| arm | val bpc ↓ (mean of seeds) | updates |
|---|---|---|
| bigram floor | ~3.9 | — |
| i.i.d. ceiling (3 epochs, unconstrained) | **2.427** | 11,763 |
| stream_cascade (1 pass, budgeted) | 2.624 | 1,500 |
| stream_rehearsal | 2.703 | 1,500 |
| **stream_two_speed** | 2.740 | **1,093 (73%)** |

Anchors: the i.i.d. ceiling (2.427) lands within **0.004 bpc of the fabric program's S1
mult-free transformer on real silicon (2.4230)** — two different mult-free architectures,
same number. **Verdict: within-0.35-of-ceiling PASSES (0.31); generations PASS** (ceiling:
"Unless you have mine amonest out of himself to deed with his hing"; two-speed after one
gated pass: "I have the beard, dies not soars" — word/line structure, not bigram babble);
**the beats-stream-cascade clause FAILS — informatively**: pure Shakespeare is a shift-free
stationary stream, so there is nothing to forget and little to gate; plain cascade wins on
quality and the gate buys only energy (−27% updates for +0.12 bpc). Together with B1-MNIST
and G1: **the ontogenic machinery pays under distribution shift; on stationary streams it is
an energy knob.** That boundary statement goes in the paper verbatim.
