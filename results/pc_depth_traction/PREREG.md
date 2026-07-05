# Deep-PC traction experiments (pre-registered 2026-07-05, from the TODO in results/graphics/README.md)

## Exp 1 — cascade-vs-relaxation gradient divergence vs depth (L = 2/4/8/16)

`scripts/pc_depth_divergence.py`. On the char-LM DyadicPCNet family (the machinery where both
rules exist): per-parameter gradient comparison between `cascade_grads` (the gradient-checked
BP-equivalent) and `error_relax_grads` at n_iter ∈ {L, 3L, 10L} sync (sanity: sync must ≡
cascade) and async (participation 0.5, stale, random order — the fabric regime), on fixed
seeded batches at each depth. Metrics: per-layer cosine + relative-L2 vs cascade, attributed
by layer index / weight-vs-bias / readout. **Read-out:** does the async/truncated divergence
stay bounded with depth or compound (worst-layer trend)? Honest scope note: softmax/LN
attribution requires the transformer relaxation port (the G2b gap) — this family localizes
divergence by DEPTH POSITION; the softmax/LN question is flagged for the xformer port.

## Exp 2 — PC-vs-BP bpb gap vs depth (the central traction figure)

`scripts/pc_depth_traction.py` (drives `pc_charlm.run_one_lm`). Depths {2,4,8,16} × seeds
{0,1,2} × arms {**bp_float** (float substrate, BP rule), **bp_dyadic** (dyadic substrate, BP
rule — the disambiguator), **relax_async** (dyadic substrate, local-PC rule)}. Matched data
(same epochs/batches); the relax arm's extra relaxation FLOPs reported, not hidden. Metric:
best val bpc (early-stopped) per (depth, arm, seed). **Read-out:** gap(bp_dyadic − bp_float)
= the substrate tax vs depth; gap(relax_async − bp_dyadic) = the local-rule tax vs depth —
the figure that says whether deep PC has traction.

## 3 — theory note
`THEORY_CASCADE_EQUIVALENCE.md` (same dir): conditions under which the single-sweep cascade
equals settled iterative PC, grounded in the repo's measured gradient checks.
