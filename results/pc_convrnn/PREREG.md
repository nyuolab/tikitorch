# PREREG — cascade PC trains fully tropical sequence models (branch `pc-morphssm`)

Committed before any gate runs. Theory: `THEORY_UNROLLED_CASCADE.md`. Style follows
`results/pc_depth_traction/PREREG.md`.

**PROGRAM GOAL (the north star, per user 2026-07-08):** enable the unique
generative/relaxation behavior of predictive coding, performantly, via tropical
deep learning — *pay one forward-pass-equivalent for training credit (the
cascade), and pay iteration only where you want the generative behavior; the
equivalence theorem guarantees it is the same model in both modes*
(THEORY_UNROLLED_CASCADE.md §1b). G0–G5 establish the training half (cascade =
settled PC, at BP wall-clock, zero-multiply backward at τ=0); G6 establishes
the inference half (infilling / steering / denoising by state relaxation on the
same checkpoint). Decisions locked with the user 2026-07-08:

1. **Headline = credit assignment.** Primary arm: **cascade PC training the real
   `tt.models.MorphSSM`** (bakeoff config d=256/L=7/K=7, exp dilation, tropical
   out_proj, ~0.51M params) against its own BP-trained published numbers
   (float 1.934±0.035 nats / closed-hard 2.1489, `results/lm_shakespeare/
   BAKEOFF_SHAKESPEARE.md`). Parity passes; beating MorphSSM is stretch, not pass.
2. **Recurrence (conv-RNN arm) payoff = long-range metrics** (selective copy
   beyond conv RF, train-96/eval-512-1024 streaming, O(d) state), NOT seq-96 bpc
   (predicted null; pre-registered as not-a-failure).
3. **Scope of this push = G0–G3.** G4 (depth ladder), G5 (strict rung), and G6
   (generative inference — the north-star payoff) are pre-registered below but
   run as follow-ups; G6 is the first follow-up once G3 checkpoints exist.

## Arms and naming

- `bp_float` — autograd + AdamW, τ-annealed (the BP twin; must reproduce ~1.93).
- `pc_cascade` — one forward with cached routing + one reverse-topological sweep
  sets every `.grad`; AdamW at PC-lr (probe around BP-lr/3 = 5e-4); same anneal.
- `pc_relax_async` — iterative error relaxation on the frozen routing,
  participation 0.5 + stale + random order, n_iter = 3·diam; the fabric form.
- `convrnn_*` — MorphSSM layers + per-channel leaky-max recurrence
  h_t = max(u_t, h_{t−1}−λ), λ = 2⁻⁶+|raw|, log-spaced init, HARD at all τ;
  `norecur` twin (recurrence off) is the paired ablation.

Protocol (all G3 cells): exact bakeoff — 1500 steps, batch 24, seq 96, char-65,
90/10 split, fixed val windows (eval seed = seed+1, 8×24×96 = 18,432 tokens),
τ-anneal hold 0.5 → 0 by 90%, soft AND hard eval logged, final hard==soft check,
device mps, all-position CE. Telemetry from step 0 in every arm: modal-char acc pin
(0.1490), branch std, per-layer argmax/carry-win active fraction, state drift
slope, λ distribution. No pass/kill decisions from 300-step screens (mis-ranked
twice before); lr probes prune catastrophes only.

## Gates

### G0 — adjoint correctness (blocks everything)
Small nets (d=64, L=2–3; T=64 and T=192 for the conv-RNN), seeds 0–2, CPU.
(a) engine forward must equal `model(x)` bit-exactly at τ=0.5 and τ=0;
(b) per-parameter cosine of cascade grads vs `loss.backward()` at τ=0.5 AND τ=0,
every tensor (embed, conv w/b, out_proj W/b, norm betas, head W/b, λ);
(c) through-time: carry chains ≥ 50 steps deliver nonzero grad to λ.
**Pass:** cos ≥ 0.999 on every tensor (stop-grad LN scale is the known deviation;
dropout disabled for the check). **Kill:** any miss after tie-break alignment =
implementation bug — fix, never waive. Output `g0_gradcheck.jsonl`.

### G1 — cascade ≡ settled/async relaxation on the unrolled lattice (theory headline)
Frozen routing, fixed batches, L∈{2,4,8} (MorphSSM config), T=96, seeds 0–2.
Sync reverse-topo n_iter=L+2 must equal the cascade **bit-identically**. Async
(participation 0.5, stale, rand_order, n_iter=3·diam): cos ≥ 0.999 mean, worst
(layer, lag) bucket bounded, no compounding trend with L or lag. Report carry-chain
histogram + error **delivery** vs lag (fraction nonzero), rounds-to-0.999 vs
participation. **Retreat (not kill):** async rounds ∝ T ⇒ fabric story weakens;
cascade claims unaffected. Output `g1_equivalence.jsonl`.

### G2 — stability + recurrence-earns-its-keep (architecture kill gate, conv-RNN only)
2a drift: 20k-char stream through random-init and 200-step-trained d=256/L=7
conv-RNN; **λ=0 control must reproduce the known drift-to-extremum** (the old
negative becomes the positive control). Pass: drift slope < 1e-4/step with learned
λ; state obeys u_t ≤ h_t ≤ running-max(u).
2b synthetics (d=64, L=2, 3 seeds, BP soft-anneal ~2k steps): running-max
(sanity, ≈ exact), selective copy T=256 delays {32,64,128} vs `norecur` control,
induction probe at gaps > RF.
**Pass:** selective copy > 95% at delay 64 where `norecur` is at chance.
**KILL:** copy-beyond-RF at chance for all of {learned λ, pow2-grid λ, erosion
twin} while 2a passes ⇒ recurrence dead as an architecture claim; Track 1
(MorphSSM+PC) proceeds unaffected. Output `g2_stability.jsonl`, `g2_synthetic.jsonl`.

### G3 — TinyShakespeare bakeoff entry
Arms: (0) parity rerun `nanogpt_shakespeare.py --runs tropical-float --variant
morphssm` seed 0 → must land 1.934±0.05 (environment insurance); (1) `bp_float`
MorphSSM ×3 seeds via the new harness (must match arm 0 — harness-parity);
(2) `pc_cascade` MorphSSM ×3; (3) `pc_relax_async` ×1 (add seeds only if within
0.05 of a boundary; wall cost reported honestly); (4) conv-RNN `bp_float` +
`pc_cascade` + `norecur` ×3; (5) streaming eval of (4) at T=512/1024.
**Pass:** pc_cascade within **0.05 nats** of its bp_float twin (the credit claim);
async within 0.15 of cascade (dyadic-conv precedent +0.106); conv-RNN seq-96
parity expected (not a kill); recurrence payoff = at T≥512 the recurrent arm
degrades ≤ half the `norecur` arm.
**Kill:** cascade gap > 0.10 nats after an lr probe ⇒ triage (routing cache /
anneal interaction) before any scaling. Reference rows (not rerun): dense-conv
1.7019, dense GPT 1.7388, MorphSSM float 1.934±0.035 / closed 2.1489, trigram
2.1397, bigram 2.4819, best tropical attention 2.4498, dyadic-conv-PC 2.617 bpc.
Output `g3_<arm>.jsonl` (nanogpt row schema + credit/pc_lr/pc_niter/participation/
recurrence/lambda_p50/chain_p95).

### G4 (follow-up, pre-registered) — depth ladder
d=128, L∈{4,8,16} × {bp_float, pc_cascade, pc_relax_async} × 3 seeds (async n=1 at
L=16), optional L=32 bp+cascade. Readouts: (cascade − bp) vs depth must be ~0 at
every L (identical gradients — drift = bug); async tax bounded, non-compounding.
Pre-registered reading: if bp is flat with depth (MorphSSM L3≈L7 precedent) the
figure claims "BP quality at O(depth) settling-free cost", NOT "depth helps".

### G5 (follow-up, pre-registered) — strict-closure rung
(a) strict-annealed: PureTropicalSGD + tropical_ce_lut, anneal→hard, vs MorphSSM
strict (2.231 char / 3.071 bpb BPE); (b) hard-from-scratch cascade receipt —
expected ≈ bigram (2.44–2.48; the anneal is load-bearing +0.21–0.25) — published
honestly as the L3 ladder line; (c) flag-plant: `find_training_multiplies` trace of
forward + τ=0 cascade + closed optimizer with only sanctioned waivers (CE
softmax/LUT boundary, batch-mean divide) — the zero-multiply-backward training-
closed LM.

### G6 (follow-up, pre-registered) — generative inference on the cascade-trained checkpoint
The inference half of the program goal. Implement `MorphPCEngine.infer(clamp)`:
energy descent over STATES with routing re-evaluated each step (DyadicPCNet.infer
pattern; soft τ during relaxation — pow-2 τ preferred — hard eval of the result;
pow-2 settling step γ). Consumes the G3 checkpoints — no retraining.
- **G6a equivalence-in-reverse control:** input-clamped settling must converge
  to the feedforward pass (E → 0, states → forward activations). This is the
  sanity gate and the honest demonstration that naive iterative inference adds
  nothing — the unique behavior REQUIRES a different clamping pattern.
- **G6b infilling (the headline):** TinyShakespeare, clamp prefix + suffix
  latents, relax the middle span (embedding-space states; decode by nearest
  embedding / vocab argmax), spans {8, 16, 32} chars. Baseline on the SAME
  checkpoint: left-to-right resample of the span conditioned on the prefix only
  (the causal LM's native capability), matched compute. Pass: settled infill
  beats the baseline on span CE / exact-match at ≥2 of 3 span lengths, 3 seeds.
- **G6c denoising/completion:** corrupt {10, 25, 50}% of context tokens, clamp
  the trusted rest, relax; measure token-recovery accuracy vs a no-settling
  forward baseline. The deep-tropical analog of the SparseCodingPC win.
- **Stability pre-registration (the C3-at-inference risk):** if fixed-step
  settling diverges at d=256 (the U-gate/settle_nudge failure modes), the
  pre-listed fallbacks are, in order: smaller pow-2 γ, RMS-conditioned state
  steps, settling at larger τ, narrower/shallower checkpoint. If none converge,
  report the negative honestly — the training-half claims (G0–G5) stand alone.
- **Closure disclosure:** relaxation-mode inference at τ>0 is a declared
  boundary (same rung as the training anneal); the feedforward deploy mode
  remains fully closed at τ=0. jsonl: `g6_generative.jsonl`.

## EC2 execution plan (locked with user 2026-07-08 after G0–G3 all-PASS; no further local runs)

One c5.9xlarge via the `run_connect4_prior_sweep_ec2.sh` machinery (S3 state,
no-SSH, hard backstop), driver `scripts/_pc_morphssm_onbox.sh`, batches in
parallel with pinned threads:
- **B1 = G6 battery (north star, first):** on-box cascade-trained checkpoints
  (matched-lr, deterministic seeds), then G6a settle-to-forward control, G6b
  infilling spans {8,16,32} × 64 val windows vs matched-compute left-to-right
  baselines (greedy; best-of-N rescored), G6c denoising {10,25,50}% corruption.
  Reduced-form settling per the DAG corollary: internal states settle to the
  forward in closed form, so generative inference iterates ONLY over the free
  variables (gap/corrupted token embeddings) through the soft forward — the
  same theorem that makes training one-sweep makes generative inference
  iterate-only-where-free. Full-state settling kept as the G6a control.
- **B2 = G2b fallback ladder round 1 (budget/lr only, per the ladder order):**
  selective copy, convrnn, 10k steps × lr {3e-3, 1e-2} × 3 seeds (6 runs).
  pow2-λ / erosion-twin rungs only if round 1 fails.
- **B3 = G4 depth ladder:** d=128, L ∈ {4,8,16,32} × {bp, cascade} × 3 seeds
  (+ relax_async n=1 at L=16).
- **B4 = G5 strict rung:** PureTropicalSGD + tropical_ce_lut annealed ×3;
  hard-from-scratch cascade receipt ×1; `find_training_multiplies` trace.

## The competing-hypothesis question (user, 2026-07-08 — analysis owed once EC2 is running)

Whether the enabled capability can ultimately compete with the transformer:
**dynamically computed activations (PC settling against clamped constraints at
inference) vs the transformer's dynamically computed weights (attention as
input-dependent mixing).** To be analyzed against the measured record — the
tropical-attention bigram floor, MorphSSM's fixed-routing parity, the
recurrence's long-range payoff, and the G6 outcomes — and folded into the
branch's strategy note.

## M2/M3 execution round (pre-registered 2026-07-10, user go "go go go")

**M3 — quality-conditioned settling crossover (the generative north star's
gate).** Quality ladder of cascade-trained (matched-lr) MorphSSM checkpoints:
Q1 = the existing 1500-step/d256 ckpt (1.9352 hard CE; its G6 battery = the v6
rows); Q2 = 6000 steps/d256; Q3 = 6000 steps/d384. G6b (spans {8,16,32}) + G6c
per new checkpoint, 32 windows, best-of-8, identical protocol to v6.
**Readout:** Δ(EM, settled − greedy) and Δ(recovery) as a function of base CE.
**Pass:** Δ turns positive and grows monotonically as CE improves. **Fail is
informative:** flat/negative Δ at ~1.85-nat quality says the crossover (if it
exists) sits above this substrate's reachable quality — feeds M1's priority.

**M2 — recall gate with the l1+gather hybrid (the recall organ).**
`ConvRNNLM(hybrid_attn=True)`: ONE l1-score+argmax-gather attention sublayer
(pre-norm residual, mid-stack; nn.Linear projections for the capability gate,
dyadic later for closure; gather adjoint = scatter, hence cascade-compatible
at τ=0 if the gate passes). Tasks: MQAR (4 kv-pairs, disjoint key/filler
ranges, query gap ∈ {32,64,128} ≫ RF=19) and induction (gaps {32,64,128}),
d=64 L=2 T=256, 4000 steps, task loss only, τ-anneal, 3 seeds.
Arms: convrnn / convrnn-hybrid / convrnn-norecur (mqar control).
**Pass:** hybrid ≥ 90% at every gap, 3/3 seeds, on BOTH tasks, where the pure
conv-rnn arm is materially below (≤ ~35%, its measured band). Passing arms the
next step: the attention layer's frozen-routing cascade adjoint + closure rung.
**Kill:** hybrid ≤ conv-rnn ⇒ the gather mixer does not train from task loss
in this stack (the OFF_GPU 0.831-from-LM-loss precedent fails to transfer) —
the recall wall stands and M1 becomes the program's sole quality lever.

**M4 — cross-family uniqueness baseline (pre-registered 2026-07-10, rides Box C).**
The bear case's challenge: "a transformer shop answers infilling with one
FIM data-augmentation flag at pretraining, zero inference cost — is settling
unique beyond causal models?" Sharpest within-substrate form: **FIM-train the
SAME MorphSSM** (identical arch/params/substrate, cascade credit, vocab 65+3
sentinels ⟨PRE⟩⟨SUF⟩⟨MID⟩, PSM format, 50% FIM / 50% plain-LM batches, 6000
steps d256 = the Q2 budget). Eval: the G6b windows/spans verbatim; FIM arm
generates the middle greedily after [⟨PRE⟩ prefix ⟨SUF⟩ suffix ⟨MID⟩].
**Readout:** span EM — FIM-trained vs the settled-PC and causal-greedy arms at
matched quality (Q2), plus the FIM tax on plain-LM CE. **Interpretation
(pre-registered):** FIM ≫ settled ⇒ "unique" honestly narrows to "unique among
*unmodified* causal checkpoints — retraining beats settling"; settled ≥ FIM ⇒
the no-retraining mode is genuinely competitive with the retraining answer.

**M1 — substrate-tax scaling ladder (pre-registered 2026-07-10, GPU box).**
Corpus text8 (20M chars, committed), BPE-1024 (train-split tokenizer), seq 96,
batch 24, τ-anneal, metric val bpb. Scales: **S1** morph_d=256/L7 (~0.5M) at
20M tokens (8,700 steps); **S2** morph_d=832/L7 (~4.9M) at 40M tokens (17,400
steps). Arms per scale: morphssm:cascade (S1 ×3 / S2 ×2 seeds), morphssm:bp ×1
(twin-parity at scale), DenseConvLM twin ×2, dense GPT ×2 (S1 d128/L3/ff384;
S2 d448/L6/ff1344). **Readout:** paired substrate tax = morphssm − dense-conv
twin (bpb) at S1 vs S2. **Flip (from the strategy note):** tax at S2 ≤ half its
S1 value with a monotone trend; constant additive tax = weak-survive; growing
tax = the text8-precedent kill. **Riders on the same box:** (a) G6 battery on
the S2 cascade checkpoint — the M3 crossover at genuinely better quality;
(b) cross-family M4: FIM-trained DenseGPT (d128/L3, vocab 68, char) vs the
same-substrate FIM and settling arms. Twin-parity secondary check: cascade
within 0.05 bpb of morphssm:bp at both scales.

## Honest claim ladder (every run states its rung)

| rung | training config | claimable |
|---|---|---|
| L0 | float-sim cascade + Adam, τ-annealed | cascade PC (≡ settled iterative PC ≡ BPTT grads) trains fully tropical seq models at BP-twin quality, settling-free; deploy fully tropical (τ=0 closure) |
| L1 | post-anneal hard-routing sweep | zero-multiply credit assignment through depth and time (soft-phase multiplies disclosed) |
| L2 | + PureTropicalSGD + LUT loss | fully multiplier-free training step (sanctioned artifacts only) |
| L3 | hard-from-scratch | ≈ bigram today; the price is stated, not hidden |

## Pre-registered risk register
Recurrence gradient starvation (log-spaced λ init; carry-win telemetry; paired
ablation) · anneal cliff through the recurrence (scan hard from step 0; τ-floor
fallback) · soft out_proj memory (chunked recompute, never materialize (B·T,d,d)) ·
PC-lr mistuning masquerading as a gap (per-arm probes; 3× rule prior) · findability
wall at hard τ (PC arms anneal like BP; hard-from-scratch quarantined in G5) ·
protocol drift (G3 arm 0 parity rerun gate).
