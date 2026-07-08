# Results — cascade PC trains fully tropical sequence models (`pc-morphssm`)

Pre-registration: `PREREG.md` (committed before runs). Theory: `THEORY_UNROLLED_CASCADE.md`.
Verdicts below are stated against the pre-registered criteria. [measured] unless noted.

## G0 — adjoint correctness: PASS [measured]

`scripts/pc_convrnn_charlm.py --gradcheck` (`g0_gradcheck.jsonl`, 282 tensor checks):
2 architectures (real MorphSSM, ConvRNNLM) × τ ∈ {0.5, 0} × 3 seeds, d=64 L=3,
T=64 (192 for the conv-RNN so carry chains exceed the conv RF).

- Engine cached-routing forward == `model(x)` **bit-exact** in every cell.
- Cascade grads vs `loss.backward()`: **cos = 1.00000000 on every parameter
  tensor** (embed, conv w/b, out_proj W/b, norm betas, top-k head W/b, λ), both
  taus. The stop-grad pow-2 LN scale needed no exemption in practice (the
  engine reproduces autograd's straight-through convention exactly).
- λ gradients dense (64/64 channels nonzero), match autograd to ≤3e-8;
  carry chains up to 125 steps deliver error (gate required ≥50).

## G1 — cascade ≡ settled/async relaxation on the unrolled stack: PASS [measured]

`scripts/pc_convrnn_gradcheck.py` (`g1_equivalence.jsonl`), conv-RNN, L ∈ {2,4,8},
T=96, 3 seeds, τ=0.5:

- **Sync** reverse-topological relaxation (n_iter = 1 and L+2) equals the
  cascade **bitwise** (maxdiff exactly 0.0) at every depth — the error system is
  triangular/nilpotent, as the theory note requires.
- **Async** (participation p ∈ {0.5, 0.25}, stale snapshots, random order): the
  measured behavior is sharper than the pre-registered "cos ≥ 0.999": once the
  expected hop count (rounds × p) covers the depth, async grads are **exactly**
  the cascade grads (cos = 1.00000000); below the diameter, the un-reached lower
  layers are exactly zero (cos = 0 on `embed.weight`), never a degraded angle.
  **The async "tax" at layer granularity is propagated-or-not.** With stale
  (Jacobi) reads, error advances ≤1 layer-hop per round and a hop lands only
  when the frontier layer participates ⇒ settling diameter ≈ (L+2)/p rounds.
  The PREREG "n_iter = 3·diam" budget is therefore correct only when diam is
  counted in hops (rounds × p) — the raw sub-budget cells are retained in the
  jsonl. Time does NOT enter the diameter (the scan adjoint is within-node at
  this granularity), confirming the λ-not-T diameter claim at layer level.
- Error **delivery** at the embedding: mean = min = 1.0000 (all positions; the
  all-position CE injects error everywhere — "bounded ≠ delivered" satisfied).

## G2a — recurrence drift probe: PASS [measured]

`scripts/pc_convrnn_stability.py --part 2a` (`g2_stability.jsonl`), d=256 L=7
conv-RNN, 20k-char stream, hard τ=0:

| variant | worst \|drift slope\| /step | max carry-frac | h_max | bounds |
|---|---|---|---|---|
| learned λ (log-spaced, λ ≥ 2⁻⁶) | **1.96e-6** (pass < 1e-4) | 0.676 | 4.1 | u ≤ h ≤ runmax(u) ✓ |
| λ=0 ratchet control | 1.0e-5 | **0.999** (state frozen at running max) | 4.1 | n/a |
| negative-leak control (λ=−0.05) | **5.0e-2 = exactly the injected rate** | 1.000 | **1003.7** | n/a |

The tropical-spectral-radius account is quantitative: positive cycle weight ⇒
linear drift at that rate (the min-plus RNN kill, reproduced); zero leak ⇒
monotone ratchet (carry-frac → 1, no forgetting); λ > 0 ⇒ bounded and responsive.

## G2b — synthetic battery (kill gate): pending [running]

## G3 — TinyShakespeare bakeoff: pending [running]

Wall-clock note [measured]: at the full bakeoff config (d=256, L=7, B=24, T=96)
the cascade step is at parity with autograd BP — mps 454 vs 640 ms/step
(morphssm), cpu 1535 vs 1581 — the O(forward) claim is measured, not projected.
The training smoke descends (4.37 → 3.82 nats in 60 steps, 3.8k tok/s ≈ the
published MorphSSM throughput; params 512,065 == the bakeoff MorphSSM exactly).
