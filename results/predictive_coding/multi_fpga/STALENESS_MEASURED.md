# No-sync pipeline staleness — measured (2026-06-27)

**Question:** the bubble-free PC pipeline updates each chip from its *local* error with no global barrier. Does
the resulting **weight staleness** (a chip's forward uses weights missing the most-recent in-flight updates) break
training? This is the crux of the "no-global-sync" thesis.

**Setup:** torch char-LM reference (`scripts/pc_xformer_charlm.py`), the real mult-free fabric recipe
(`MULTFREE=1`: dyadic weights + int16 acts + dot attention + dyadic-Adam). D=128, L=4, H=4, block=128,
TinyShakespeare, 1500 steps, 0.8M params. `STALE_DELAY=k` applies each parameter's gradient `k` update-steps
late (self-consistent forward/backward = the PipeDream-2BW weight-stashed best case); `STALE_PERLAYER` scales the
delay with block depth (the faithful pipeline: chip *l* lags ~2·(L−l) microbatches).

## Result — weight staleness tolerance is TIGHT (~1 step)

| weight staleness (update-steps) | final bpb | Δ vs sync |
|---|---|---|
| 0 (synchronous baseline)        | **2.457** | —      |
| 1                               | 2.639     | +0.18 ✓ (tolerable) |
| 2                               | 3.209     | +0.75  |
| per-layer (avg lag 5)           | 3.754     | +1.30  |
| 4                               | 3.747     | +1.29  |
| 8                               | 4.478     | +2.02  |

**LR-scaling does NOT recover it** (isolation control, matched LR=1.2e-3): stale-0 = 2.482, stale-8 = 3.747. A
stale gradient is a bad descent direction regardless of step size — it is the staleness *magnitude* that matters,
not optimizer aggressiveness, and not forward/backward consistency (these grads are self-consistent).

## Interpretation — two different "staleness", only one is tolerated

- **TYPE 1 — error-relaxation staleness (TOLERATED):** stale *neighbour messages* in the local error computation.
  The iterative-PC relaxation is stable under this (the prior MLP "~0.6pp at 50% stale"). The genuine async win:
  the backward error sweep need not drain in strict lockstep.
- **TYPE 2 — weight staleness (NOT tolerated):** applying gradients computed `k` weight-updates ago. Naive
  per-microbatch async across a deep pipeline gives staleness = pipeline depth (≈2·(L−l) per chip) → for a 12-layer
  124M model, ~24-step staleness at early layers → severe degradation. **Tolerates only ~1 step.**

The earlier roadmap conflated these and over-claimed "async / drop-a-round / staleness-tolerant" for weight
updates. **Corrected.**

## Build rule (consequence)

Bound weight-staleness to ~1 via **per-chip gradient accumulation**: each chip freezes its weights over an aligned
window of W microbatches, accumulates its local error, applies once → effective staleness ~1, the pipeline stays
full, cost = a W× larger effective batch (standard). This is **pipelined gradient accumulation with a local
backward** — not fully-async training. What survives intact: no global all-reduce / no global gradient (local
credit assignment), all chips busy, the 8–17×-over-serial pipeline throughput (accumulation does not impede
filling the chips), replication (4.00× linear). The thesis ("trained with no global gradient synchronization")
holds; the "staleness-immune" oversell does not.

Raw logs: `/tmp/stale_sweep.log`, `/tmp/stale_ctrl.log`, `/tmp/stale_thresh.log`.
