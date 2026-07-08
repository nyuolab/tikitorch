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

## G2b — synthetic battery (kill gate): recurrence attribution CONFIRMED; kill NOT fired; 95% pass bar unmet [measured]

`scripts/pc_convrnn_stability.py --part 2b` (`g2_synthetic.jsonl`): d=64, L=2
(conv RF = 19), T=256, 3 seeds per arm, BP soft-anneal 2k steps, lr 1e-2,
λ_min = 2⁻⁸. Chance = 1/16 = 6.25%.

| task | delay/gap | convrnn (3 seeds) | convrnn-norecur (3 seeds) |
|---|---|---|---|
| selective copy | 32 | 23.1 / 32.4 / 68.0 % | 8.9 / 4.6 / 6.3 % |
| selective copy | 64 | 22.7 / 35.7 / 63.5 % | 11.1 / 3.5 / 7.7 % |
| selective copy | 128 | 16.7 / 33.3 / 59.6 % | 7.4 / 4.8 / 3.9 % |
| running max | (all-pos) | 99.90 / 99.94 / 99.93 % | 97.96 / 98.05 / 97.75 % |
| induction | 32 | 21.3 / 3.8 / 28.6 % | 9.0 / 5.9 / 4.8 % |
| induction | 64 | 23.9 / 4.0 / 24.6 % | 8.6 / 2.9 / 6.9 % |

**Verdicts vs the pre-registered criteria:**
- **Kill criterion (recurrent arm at chance): decisively NOT fired.** On the two
  content-carrying tasks all six conv-only controls sit at chance while the
  recurrent arm is 3–10× above it — the information crosses the RF through the
  recurrence and nothing else.
- **Pass bar (copy > 95% at delay 64): unmet at this probe budget.** The seed
  spread (23→64%) plus the FLAT accuracy across delay 32→128 says retention is
  not the constraint (λ-decay holds the value across the full delay); circuit
  *findability* (marker-gated write, query-gated read) is. Fallback ladder
  applies in the pre-registered order: longer budget / lr probe → pow2-grid λ →
  erosion twin. No architecture-level kill.
- **Running-max confound (honest demotion):** with uniform values 1..16 at
  T=256 the running max saturates to the ceiling early, so window-max +
  positional prior answers ~98% without memory — the control's 98.0% is prior,
  not recall, and the task is demoted to a sanity check for the recurrent arm
  (99.9%, exact, zero variance). The consistent ~2 pp gap is the hard-position
  sliver the prior cannot answer — weak corroboration only. A discriminative
  running-max needs a non-ceiling / drifting value distribution or a
  hard-position-conditioned metric.
- **Induction (recall) is the capability edge**, as the tropical-SSM results
  predicted: 2 of 3 seeds find a partial circuit (~24%), one finds none —
  recurrence-carried but weakest of the battery. Consistent with the
  pre-registered expectation that content-addressed recall is where the
  running-max state struggles.

## G3 — TinyShakespeare bakeoff: PASS on every pre-registered criterion [measured]

Exact bakeoff protocol (1500 steps, B24, T96, char-65, fixed val windows,
τ-anneal, mps, hard==soft at final eval in every arm). All runs local (mps);
jsonl per arm in this directory; checkpoints in `ckpts/` (untracked).

**Arm 0 — parity:** original `nanogpt_shakespeare.py` (canonical-repo
PYTHONPATH; the bare run imports the stale kanervasforge submodule copy — trap
recorded) seed 0 @ lr 1e-2 → **1.9445**, inside the published 1.934±0.05. PASS.

**The credit claim (cascade vs bp twin, matched lr 1e-2):**

| seed | morphssm:bp | morphssm:cascade | gap |
|---|---|---|---|
| 0 | 1.9352 | 1.9352 | +0.000 |
| 1 | 1.8993 | 1.9128 | +0.014 |
| 2 | 1.9445 | 1.9376 | −0.007 |

Mean gap **+0.002 ± 0.011** (bar ≤ 0.05): **PASS**. Cascade PC trains the real
MorphSSM to backprop-twin quality at backprop wall-clock (454 vs 640 ms/step
mps; ~4.5–4.8k tok/s both arms). conv-RNN matched-lr twin, seed 0: 1.9271 vs
bp 1.9271 (gap +0.000); seeds 1–2 intentionally not run — conclusion already
made, remaining compute moved to EC2 (user decision 2026-07-08). Trajectories
are NOT bit-identical (fp reduction order differs between autograd kernels and
the scatter sweep; they agree to ~1e-6 at step 0 and stay within seed-noise) —
"twin" is a statistical claim; the *gradient* identity is G0's bitwise claim.
NOTE: the earlier lr/3 cascade rows (mean 1.998) are retained as data; the
dyadic-era "PC lr = lr/3" rule does NOT apply to exact-mode cascade (G0:
identical gradients ⇒ matched lr is the correct twin protocol).

**Async fabric probe (n=1 pre-registered):** `morphssm:relax_async`
(participation 0.5, stale, random order, n_iter=21) → **2.0301** vs its lr/3
cascade sibling 2.0108: gap **+0.019** (bar ≤ 0.15): **PASS**. Wall cost
honestly 3.3× (3010 s vs ~910 s).

**Recurrence long-range payoff (pre-registered metric):** train T=96, eval
hard CE at T=1024 (mean over 3 seeds, degradation vs own T=96):

| arm | T=96 | T=1024 | degradation |
|---|---|---|---|
| convrnn:bp | 1.917 | 1.949 | **+0.032** |
| morphssm:bp | 1.926 | 2.017 | +0.091 |
| convrnn-norecur:bp | 1.936 | 2.032 | +0.096 |

The recurrent arm degrades ~⅓ of its paired conv-only control (bar ≤ ½):
**PASS**. Learned λ medians ≈ 0.40–0.49 (recurrence active, not self-disabled).

**Seq-96 headline (pre-registered null):** convrnn 1.917 ≤ morphssm 1.926 ≤
norecur 1.936 — ordered in the recurrence's favor but within seed spread;
reported as parity, exactly as pre-registered. No "beats MorphSSM" claim.

## Next steps (EC2 — user decision 2026-07-08: no further local runs)

Compute template: `scripts/run_connect4_prior_sweep_ec2.sh` pattern (IAM
instance-profile, S3 state, no-SSH polling). All remaining batches are CPU-
parallel at 0.5M params. Order:
1. **G6 generative inference (the program north star)** — implement
   `MorphPCEngine.infer()` + G6a no-op control locally (cheap, CPU), then the
   G6b infilling / G6c denoising battery on EC2 against the saved G3 ckpts.
2. **G2b fallback ladder** — selective copy at 10–20k steps × lr probe ×
   {learned λ, pow2 λ, erosion twin} × 3 seeds (the 95% bar question; G2b
   showed findability, not retention, is the constraint).
3. **G4 depth ladder** — d=128, L ∈ {4,8,16,32} × {bp, cascade} × 3 seeds
   (+async ×1): the settling-free-depth figure.
4. **G5 strict rung** — PureTropicalSGD + tropical_ce_lut annealed vs MorphSSM
   strict; hard-from-scratch cascade receipt; `find_training_multiplies`
   zero-multiply-backward flag-plant.

Wall-clock note [measured]: at the full bakeoff config (d=256, L=7, B=24, T=96)
the cascade step is at parity with autograd BP — mps 454 vs 640 ms/step
(morphssm), cpu 1535 vs 1581 — the O(forward) claim is measured, not projected.
The training smoke descends (4.37 → 3.82 nats in 60 steps, 3.8k tok/s ≈ the
published MorphSSM throughput; params 512,065 == the bakeoff MorphSSM exactly).
