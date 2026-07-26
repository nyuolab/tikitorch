# External peer feedback round 2 (GPT Sol, 2026-07-26) — adjudicated

## Accepted corrections to our own reasoning
1. **No single causal ranking for the whole system** — subsystem-specific:
   WM seesaw = ordering/interference first, width/modularity second; policy
   compounding = task structure first, optimization second, width
   unresolved; lifetime ratchet = a negative result, full stop. ADOPTED as
   the E6 RESULTS framing.
2. **Provenance hit (fair): the ent-pull/base second-half growth rates were
   computed ad hoc and not archived.** Fixed: derivation script + output
   now committed (results/e6_screen/growth_check.*). Standing rule
   reaffirmed: any number quoted to the user must have a committed
   derivation.
3. Raw %-slope is fragile across arms with different baselines/ceilings —
   forward-transfer AUC + normalized competence are the compounding
   measures. ADOPTED into the E6-D/E6-S metric set.
4. Fast revisit recovery SUGGESTS shadowing, does not prove it (fast
   relearning from preserved features is unrefuted). ADOPTED — this is
   exactly what the shadowing test discriminates.
5. Disjoint output rows rule out output-slot competition only — not
   insufficient shared-trunk capacity. ADOPTED (width stays live).
6. "No parameter count restores IID" softened: width can reduce gradient
   conflict / permit emergent modularity without recreating IID. ADOPTED.

## The phased discriminator (E6-D) — adopted as the next gate, BEFORE E6-S
Phase D1 **Shadowing test** (cheapest, launches first): at revisit
boundaries, freeze learning for the first 50k steps. CE/return recovery
WITHOUT updates ⇒ contextual shadowing; recovery requiring updates ⇒ fast
relearning. Implementation: freeze_first_steps cfg + branch mode from the
COMPLETED Tier-1 s3 life's era ckpts — the source's own lived eras are the
unfrozen controls at zero cost.
Phase D2 **Order-by-width**: recorded-stream WM-only replay (the exact
sample multiset, blocked vs interleaved) × widths {128, 256, 512} × 3
seeds — the controlled version of Sol's design; the on-policy variant
(era-granularity sweep) runs beside it, deviation declared (on-policy
streams cannot hold the multiset fixed).
Phase D3 **Mechanism probes**: cross-world gradient cosine, top-k route
overlap, hidden-state linear-probe retention, recovery half-life —
classifying the interference as representational / routing / calibration.
Phase D4 task richness (compositional XLand subset; acquisition AUC vs
from-scratch, worst-skill retention, breadth, zero-shot) — then E6-S with
≥3 widths + held-out width/lifetime prediction as the bar for "law".

## Diagnosis → remedy map (registered so the follow-up is mechanical)
shadowed ⇒ context-conditioned calibration (norms/biases/temperatures,
tiny per-world state); overwritten ⇒ slow-trunk/fast-adapter split, sparse
experts, or surprise-gated route allocation; gradient conflict ⇒ low-rank
historical-subspace protection (Fisher subspace, not per-weight anchors).
**Replay-as-diagnostic-oracle arm ACCEPTED** for E6-D: replay stays
excluded from the organism (memory-class line unchanged) but one replay
arm measures the empirical PRICE of that constraint — a number the thesis
needs anyway.
