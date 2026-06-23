# TikiTorch roadmap — PyTorch-shaped, for the tropical/FPGA niche

**North star.** TikiTorch is the *verifiable* zero-DSP, multiplier-free, FPGA-deployable
tropical-semiring library — **PyTorch-shaped** so a torch user pays near-zero switching cost. The
product is the **guarantee** (closure verification) and the **deploy artifact** (the bit-exact dyadic
shift schedule), wrapped in torch-familiar ergonomics. "Make it like PyTorch" means make a torch user
productive *in this niche* and make the moat first-class — **not** replicate torch features whose
defining op is a non-pow2 multiply/divide/transcendental, nor general-DL throughput plumbing this
single-board, kernel-bound, built-on-torch library does not need.

TikiTorch is built **on** torch (it subclasses `torch.nn.Module`, reuses autograd / `Parameter` /
`no_grad`); it provides tropical `nn`/`optim`/`models`/`functional`, a closure-audit layer, and the
FPGA path. So the gaps that matter are *ergonomics + the moat*, not core-framework parity.

## NOW (high-leverage, low-effort, niche-serving)

- [x] **Closure GUARD** — `with tt.assert_no_multiply():` / `@tt.closed` (`audit.py`), reusing the
      existing `_make_catcher` `TorchDispatchMode`. Fails on the first hardware multiply/divide/
      transcendental in *your own* forward/training step; powers of two pass as shifts. The
      `torch.no_grad()`-shaped sugar that is the thing a PyTorch user adopts the library *for*. **Done.**
- [x] **Closure-clean activation + dropout modules** (`tt.nn.TropicalReLU/TropicalLeakyReLU/
      TropicalHardtanh/TropicalDropout`). Fixes a real hole: standard `nn.Dropout`'s `1/(1-p)` survivor
      rescale is a non-pow2 multiply at most `p` (it's fine only when `1/(1-p)` is a power of two, e.g.
      `p=0.5`); `TropicalDropout` is a `{0,1}` gate (no rescale) — closure-clean for **all** `p`. Swapped
      into the 7 `nn.Dropout` sites across `nn`/`models`. **Done.**
- [ ] **MIGRATION.md** — side-by-side "your PyTorch loop vs the closed TikiTorch loop" teaching the 4
      non-obvious closure idioms (`lr_shift` not `lr=`, tropical losses, soft→hard `tau` anneal,
      power-of-2 dims). The namespace is already 1:1; the *idioms* are the adoption blocker.
- [ ] **`tt-closure` CLI** (`[project.scripts]`) — point at any `model.py:factory`, print
      `multiplications: 0` / nonzero-exit. Generalizes the `__main__` demo; the CI-gate primitive.
- [x] **Promote `fpga.dyadic_export`** to first-class — `export_net`/`verify_schedule`/`export_layer`/
      `reconstruct_weight` + `MuZeroKernelConfig` now exported in `tikitorch.fpga.__all__`, with tests
      (bit-exact shift-schedule round-trip). **Done.** _(Follow-ons: save/load the schedule to disk; embed
      the `verify_schedule` result in the checkpoint as a "closure passport".)_
- [ ] Cheap credibility: `git tag v0.1.0` + release (CHANGELOG links a missing tag); fix `examples/README`;
      publish the docs site (the mkdocs site itself now **exists** — added alongside this work).

## NEXT (medium)

- [ ] **`TropicalSequential`** that auto-cascades `set_tau`/anneal + `assert_closed` across children
      (a plain `nn.Sequential` silently fails to propagate `set_tau` — a documented footgun, hand-rolled
      in ~25 sites).
- [ ] **Bit-shift LR schedulers** (`ShiftStepLR`/`ShiftWarmupLR`) — there is no scheduler at all; the
      sign-only optimizers can't decay. A scheduled integer `lr_shift` is a decay that is *itself a
      bit-shift* — a tropical-native feature torch's float schedulers can't claim.
- [ ] **CI closure-conformance gate** over the model zoo (green/red "zero multiplies" per push).
- [ ] **Tropical op-counter** — `DSP=0 + N shifts/compares/adds` profiler for any traced model (FLOPs
      are vacuous when there are no FLOPs).

## LATER

- AST lint flagging raw `*`/`@`/`/`/`exp` inside a tropical `forward` (the gap the structural scan
  can't see; the runtime guard already catches it dynamically).
- Generalize `verify_schedule` bit-exactness across the whole HLS kernel set, wired into csim CI.
- `tt.harden(model)` — public "give me the closed deploy graph" transform (drive `tau→0`, swap soft→hard).
- Named `tt` dtype symbols + INT16 deploy-gap preview (after `dyadic_export` promotion).
- `tt.hub` named registry for closure-certified pretrained weights + PyPI trusted-publisher release.
- Capacity gaps as demanded by real models: `TropicalConv1d`/`ConvTranspose`, packaged MHA with K/V +
  masks, optimizer+tau checkpoint-resume, FPGA-layout collate helpers (over stock `torch.utils.data`).

## EXPLICITLY AVOID (these *break* closure or are throughput plumbing this niche doesn't need)

- **BatchNorm/GroupNorm/InstanceNorm** — per-channel variance normalization = data-dependent
  multiply/divide. `TropicalLayerNorm/RMSNorm/BatchShift` already cover normalization with pow2 shifts.
- **Smooth activations** (Sigmoid/Tanh/GELU/SiLU) — transcendental/multiply-heavy, no closed form. The
  tropical nonlinearity *is* max/min + pow2 leaky-relu.
- **LSTM/GRU gates** — sigmoid/tanh multiplicative gating. Use `TropicalRNNCell`/`MorphSSM`.
- **Gradient clipping** — sign optimizers already discard gradient magnitude (clipping is a no-op on the
  update); `clip_grad_norm_`'s global L2 norm is itself a multiply.
- **A `tikitorch.data` Dataset/DataLoader framework + distributed/DataParallel** — the FPGA step is
  kernel- and BAR4-bound on a single board, not host-throughput-bound. Use `torch.utils.data` directly
  (the `Trainer` already accepts a torch `DataLoader`).
- **torch.jit / torch.compile / trace** — the deploy path is the dyadic int16 shift schedule verified
  bit-exact vs HLS csim, not a traced/compiled torch graph.
- **Multi-board / per-op `.to('fpga')` dispatch** — single-board AWS-F2; whole-model binding is the right
  Tier-1 shape (the seam already supports flipping per-op dispatch on with no API change).
- **A standalone `API_SPEC.md`** — `docs/` + mkdocstrings autodoc already exist; don't keep a second
  hand-maintained reference in sync. Spend the effort on `MIGRATION.md` + publishing the docs.
- **Float weight-EMA / SWA / Polyak** — `decay*shadow + (1-decay)*w` is a data-dependent multiply.

_Source: a 6-dimension TikiTorch-vs-PyTorch gap analysis (nn/models, training stack, data/IO,
backend/device, the closure moat, docs/packaging) with anti-cargo-cult + adoption-leverage filtering._

---

# Silicon / FPGA-vs-GPU research track

> Separate from the library roadmap above (which stays deliberately single-board). This is the
> *research* program: can a multiplier-free tropical substrate **compete with GPUs on training**?

## Where we are (measured, 2026-06-15)

The whole tropical LM training loop (fwd + CE loss + backward + signSGD) runs end-to-end on AWS F2
silicon, bit-exact to the host golden, and **replication scales it LINEARLY**:

| config | tok/s | scaling | vs nanoGPT A10G (107,091) |
|---|---|---|---|
| 1 kernel | 14,236 | 1.00× | 13% |
| N=2 | 26,522 | 2.00× | 25% |
| **N=4 (SLR-local max)** | **53,001** | **4.00×** | **49.5%** |
| N=8 | ~106,000 | proj. | ~100% (parity) |

All bit-exact, DSP=0, ~4 W core, wall-clock unchanged (true parallel). N=4 is the SLR-local max
(per-SLR CL URAM 176/192/~320 → 1+1+2). Recipe + ceiling in `tikitorch/fpga/silicon/SILICON_FINDINGS.md`.

## PC local-credit training track — the *no-global-coordination* substrate bet (2026-06-23)

A second silicon pillar, distinct from the replication track above. Replication is **data-parallel**
(independent copies, no comm) and the morph-LM trains by SignSGD-**backprop**. This track instead puts
**local predictive-coding credit assignment** on the multiplier-free fabric — which removes backprop's
**global** ops (ordered backward sweep, **weight transport**, **all-reduce**, **barrier**). That is the
structural lever for sharding **one** model across many chips a backprop cluster can't coordinate.
Docs: `results/predictive_coding/{TRAINING_AT_SCALE,PC_BACKWARD_SILICON,MULTI_F2_SCALE_OUT}.md`.

**Measured on real F2 silicon:**

| result | status | evidence |
|---|---|---|
| **PC backward = bit-exact silicon** | ✅ | `forge_pc_wgrad` (full `cascade_grads` layer-step: routed adjoint + weight grad + bias grad + error route + residual skip), DSP=0, 2.920 ns, AGFI `agfi-063e2dd0d3255dcaa`; on-chip readback == `cascade_grads` `max\|diff\|=0` (known-answer + random). **Transpose-free**: the error route reuses the *forward* `q(W)` in place — no separately-stored/synced transposed buffer (this is *not* the bio "weight-transport-free" property, which would need separate feedback weights; the hardware win is no transpose copy + no cross-chip weight movement in the shard). |
| **Sharded across 2 physical FPGAs** | ✅ | 2-layer backward across an f2.12xlarge's slots 0+1, ONLY a single neighbour error message between them (no global gradient / transport / barrier) → bit-exact to single-process `cascade_grads`. Silicon realization of the gloo `pc_model_parallel.py` proof. |
| **Fully-on-fabric trainer = trains on silicon** | ✅ | `forge_pc_train_step` — one kick = forward + cascade backward + SignSGD-momentum update, weights+momentum persist in DDR; host only stages a batch + kicks. Pipelined forward (FF 20%, HLS ~8 min), **`DSP_MULT=0`** (1 DSP as an adder), timing-closed **WNS +0.196 ns**, AGFI `agfi-0fdc38db9da6add74`. **On-fabric RUN measured:** 16×16 MNIST chance → **0.7475** over 5 epochs (batch 32, uniform init), entirely on the FPGA; an independent recipe-forward eval of the dumped weights = 0.7413 (cross-confirms). Recipe ceiling ~0.81 at this config / ~0.90 best. Bug found+fixed: kernel `norm_shift` off-by-one (1 less shift than the recipe) stalled training at chance despite a bit-exact one-step verify; compensated host-side (loss target 8→16, no rebuild). Details: `ONFABRIC_TRAINING.md` (honesty-audited). |

**Honest scope.** These prove the **substrate** (multiplier-free + no-global-coordination credit
assignment *runs and trains* on real silicon) — **not** a throughput win: per the Rung-4 fabric-sim,
digital PC is break-even-to-slower than backprop (it still pays R relaxation rounds + a weight-grad
pass). The win is **structural** — sharding without global ops — realized only at extreme comm-bound
scale **or** on an analog/in-memory substrate where the O(L) error relaxation is continuous-time
settling. This dovetails with the Cluster track below (PC needs no all-reduce *at all*, vs even 1-bit
SignSGD's 1-bit one).

**Lessons banked (reusable):** on-silicon verification catches synthesis-only bugs C-sim cannot — a
`static const` threshold ROM whose init didn't survive Vivado synth (the all-1s **known-answer test**
localized it in one cheap run; fix = table-free dyad); and inter-layer PC errors **grow** each layer →
a fixed-point rescale of the neighbour message is load-bearing (same activation-scaling the muzero net
needed).

- [x] **On-fabric training RUN** — `run_pc_train_f2.sh` on `agfi-0fdc38db9da6add74`: host stages
      batches + kicks; accuracy climbed **chance → 0.7475** entirely on the FPGA. ✅ (`ONFABRIC_TRAINING.md`)
- [x] **Make the train-step build cheap** — pipelined the forward reduction (was a 256-wide unroll @ 54% FF,
      ~2 h HLS): FF → 20%, HLS → ~8 min, timing closed WNS +0.196 ns, compute unchanged (csim still bit-exact). ✅
- [ ] **Deeper / network-distributed shard** — ≥3 layers across ≥3 FPGAs (f2.48xlarge = 8), then TCP
      neighbour messages across separate hosts (true no-shared-host messaging).
- [ ] **Longer/bigger on-fabric run** — batch 128 + randn init + more epochs to chase the recipe's ~0.90
      ceiling on fabric (this run was a 5-epoch, batch-32 correctness demo at 0.75).

## CONSTRAINT CLARIFICATION — the invariant is MULTIPLIER-FREE, not DSP=0

**DSP blocks are fine — just not in multiply mode.** The closure guarantee is *no hardware multiply*
in fwd/loss/bwd/update. A DSP48 used as an **adder / subtractor / comparator / accumulator / wide-MUX**
(not its multiplier) is fully closure-clean and is exactly what a max-plus PE wants. We over-constrained
the silicon work to literal `DSP=0` as a conservative proxy — which left the VU47P's **~9,024 DSP48s
completely idle**. For throughput, *use them as hardened add-max ALUs* (≈10× more max-plus units than
the LUT-only fabric we measured). Action items:
- [ ] **Throughput build that uses DSPs as add/compare ALUs** (drop the `DSP=0` synthesis constraint;
      keep the *multiply-free* invariant). Re-measure tok/s/kernel — expect a large jump.
- [ ] **Audit fix:** the closure check should flag DSPs in **multiply mode** (report `DSP_MULT=0`), not
      `DSP=0`. Update `audit.py` / the HLS report parse + the manuscript wording ("zero multipliers",
      not "zero DSPs"). This also likely relaxes the MuZero-chess train-kernel area problem (it was
      forced LUT-only; DSP-adders may fix the 184%-of-SLR blowup — see [[muzero-board-scaling]]).

## On-FPGA throughput track (single board → competitive per-watt)

- [ ] **DSP-as-add-max ALUs** (above) — the biggest near-term lever.
- [ ] **Hardened systolic add-max array** (TPU-MXU-shaped, but add-max PEs). Max-plus PE = int16 adder +
      comparator — intrinsically cheaper than an FP16 multiply-add, so pack more PEs at higher util.
- [ ] **HBM weight streaming** for models that don't fit on-chip URAM (we kept tiny models resident).
- [ ] **N=8 GPU-parity** on the current part: staging-buffer compaction (drop the 5 grad regions —
      confirmed dead in the staging buffer) + grads→BRAM rebalance so 2 kernels fit the URAM-poor
      SLR0/SLR1. Projected ~106k from the measured 4× linearity.
- [ ] **Newer silicon:** Versal (VP1902 / AI-Edge) — far more LUTs + AI Engines (VLIW-SIMD maps to max-plus).

## Cluster track (where tropical is structurally advantaged)

Large-model training is increasingly **communication-bound** — tropical's edge:
- [ ] **1-bit SignSGD all-reduce.** The update needs only the *sign* of the summed gradient → 1 bit/
      weight vs 16–32 for dense FP = a **16–32× smaller all-reduce**, attacking the exact multi-GPU
      bottleneck. (signSGD-with-majority-vote, free here because the kernel is already sign-based.)
- [ ] **On-fabric reduction + transceiver pipeline parallelism.** Do the 1-bit majority-vote in hardware
      on the link (no NCCL/host round-trip); stream layer *i* → layer *i+1* across chips over Aurora/GTY.
      FPGAs' home turf. (Note: AWS F2 cross-FPGA is host-PCIe only — this needs on-prem transceiver links.)
- Independent replication (population training / sweeps / serve-many-models) is embarrassingly parallel —
  linear across boards, no comm bottleneck — already true today.

## ASIC endgame

The real prize: a chip with **no FP multipliers**, a massive int8/16 **add-max array** + on-fabric 1-bit
all-reduce. Tropical discards the GPU's moat (the FP multiplier), so a dedicated tropical ASIC could
plausibly beat GPUs on **tokens/joule and tokens/dollar at cluster scale**. The FPGA work is the
silicon-validated de-risking (the algorithm trains, bit-exact, multiplier-free, scales linearly).

## ⚠️ The gating risk — model QUALITY at scale

Everything above is moot unless tropical models reach **GPU-class quality**. Today they don't: strict
multiplier-free MorphSSM = **2.79 bpb** vs dense nanoGPT **2.27 bpb** (tiny-shakespeare BPE-1024), and
tropical *attention* is the known-broken piece. "Compete on large models" **requires closing this gap**,
and whether the gap **closes or widens with scale** is the go/no-go question for the whole track. This is
a research problem, not a hardware one — and it's the next thing to investigate.

### Quality-gap investigation plan

**Framing (2026-06-15):** the target is *our best tropical model vs nanoGPT-on-GPU*, NOT morphssm-internal
ablations. nanoGPT's edge IS attention, so a true tropical *transformer* (working tropical attention) is
the strategic lever — **but** the data says tropical attention is currently the *worst* tropical component,
not the cheapest win:

| model | mixer | char bpb |
|---|---|---|
| dense GPT | real attention | 2.51 |
| MorphSSM | conv/morph (no attention) | ~2.80 |
| best tropical attention (hilbert) | tropical attention | 3.53 |

~6 tropical-attention variants already exist (`hilbert/dot_product/l_inf/asym/value_pull[+topk/min]`,
`tikitorch/nn/__init__.py`); all lose to the conv mixer by ~0.7 bpb — **MorphSSM exists because tropical
attention failed.** The catch: the bakeoff is seq_len 96, where attention has little to do that a conv
can't; attention's real edge (data-dependent global routing, long-range, in-context retrieval) only shows
at long context/scale — exactly where a fixed-receptive-field conv/SSM mixer plateaus and where we'd
actually compete. So tropical attention is plausibly *necessary at scale* yet *invisibly so* on the toy
benchmark.

**Gate before investing in tropical attention (cheap, dense-only, no tropical research):** train
dense-attention vs dense-conv-twin across context length {96, 512, 2048} (and scale). If the gap WIDENS
with context → attention is the load-bearing thing to tropicalize (prize quantified, do the hard work);
if conv keeps pace → MorphSSM is the substrate and tropical attention is a trap.

**If go — the tropical-attention research problem (sharp):** softmax attention = soft convex *averaging*
of values by *dot-product* similarity (both multiplicative). Max-plus makes it hard max-alignment +
winner-take-all selection (picks ~one value, not a blend) with sparse max/argmax gradients (the collapse).
The tau-softmax relaxation trains but degrades at hard deploy. The unsolved core: **recover soft,
multi-value blending + smooth gradients in a multiplier-free op** — what none of the 6 variants cracked.

#### Brainstorm verdict (2026-06-15, 31-agent generate→adversarial-vet→synthesize)

**The decisive finding:** every "broken tropical attention" number (2.45–2.74 bpb) is a **τ=0 (hard
max-plus) number** — `nanogpt_shakespeare.py:tau_at` anneals τ→0 and deploys hard. The SOFT path already
exists (`functional.py: tropical_softmax_*`, `tropical_softmax_value_pull`) but is **thrown away at
deploy**. **Nobody has measured a fixed-nonzero-τ deployed attention LM** — the largest untested lever,
~free to test.

**Purity verdict (ranked purity-first per the north star):** soft value-blending appears to **require
leaving idempotent max-plus** — pure-max-plus blend-recovery ideas were adversarially killed: Gumbel/
stochastic-hardmax (forward still argmax; repo measured injected noise HURTS, breaks bit-exact gate),
top-k dyadic sum-blend (data-INDEPENDENT weights, not input-conditioned; unbounded sum), shift-normalized
soft attention (= the `shift_softmax` row-sum∈[0.5,1] that CAUSED the collapse), entmax/sparsemax (needs
sort+cumsum the fabric lacks). Idempotency (`max(a,a)=a`) is the opposite of averaging; non-idempotency
(`logsumexp(a,a)=a+log2`) is exactly what enables blend. So **purely-tropical attention hits a real wall
on the blend specifically**; the working candidate is multiplier-free but not max-plus.

**Shortlist:** (#1) **fixed-τ log-semiring attention** [floor, not max-plus] — deploy the existing soft
path at τ=2^−s; the prior variants ARE this at τ=0. (#2) tropical linear-attention SSM (log-domain
recurrence, power-of-2 additive decay — mandatory or it saturates). (#3) additive Bahdanau score w/
dyadic w — **ablation only** (finding-3: the blend, not score geometry, is the bottleneck).

- [ ] **DECISIVE FIRST BUILD (free, no new forward code):** clamp `tau_at` to a power-of-2 floor (2^−2)
      for the attention path; eval at that fixed τ (not τ=0); sweep τ∈{2^−1,2^−2,2^−3}, char, 1500 steps,
      ~min on MPS. **Breaks the 2.45–2.74 band → blend was binding, log-semiring is the path (attention
      leaves pure max-plus, rest stays max-plus); doesn't move → finding-3 wins, blend was never the
      bottleneck and purely-tropical attention stays alive via score-side fixes.** Decides whether we're
      forced off pure max-plus at all.
- [ ] **The math that must hold:** `τ·logsumexp_j(z_j/τ)` with τ=2^−s = `{max,+,−,shift, one 1-input
      LUT}`; the LUT is a NEW `g(δ)=log1p(exp(−δ))` table (the only new silicon ask; today only the CE
      exp2 table exists). **Deploy the `logsumexp-of-(V+s)` form, NEVER normalize-then-weighted-sum** —
      the textbook `Σαⱼ·Vⱼ` form hides an anti-log multiply + the row-sum-divide collapse.
- [ ] **Do NOT chase** (vetted fatal): LNS explicit `ΣpⱼVⱼ` Hopfield form (anti-log multiply + divide
      collapse); shift-normalized attention; τ-floor as a *destination* (still the smoothed-max ceiling,
      not a convex blend); entmax/sparsemax; LFSR-Gumbel; iter-top-k sum-blend.

#### PURELY-TROPICAL all-pairs communication — routing, not averaging (2026-06-15, 29-agent vet)

Reframe (user): attention's *function* is data-dependent all-pairs communication to find signal;
its *mechanism* (softmax averaging) is incidental. The tropical (max-plus) semiring **is** the
shortest/longest-path & DP algebra, so all-pairs communication is **native**: all-pairs best-path =
max-plus matrix powers (Bellman–Ford / Floyd–Warshall). This is the **purely-tropical (idempotent)**
path — the north-star ideal, NO relaxation — and it fills MorphSSM's exact gap (fixed receptive field →
no content-addressed long-range retrieval).

**Winner: iterated best-path closure / Bellman–Ford diffusion.** Build `S[i,j]=max_d(Q_id+K_jd)` once;
iterate the STATE `H⁽ᵗ⁺¹⁾[i,d] = max(H⁽ᵗ⁾[i,d], max_j(S[i,j]+H⁽ᵗ⁾[j,d]))` for T rounds. Inner = existing
`TropicalValuePull`; outer = +0 self-loop. All idempotent max-plus (+/max/gather/negate). T rounds =
≤T-hop all-pairs routing; full closure in log T via repeated squaring. The 2-hop induction path
(match-by-content → read-successor) is exact here and **impossible for any conv**.
*Note:* `max_plus_matmul`/`floyd_warshall` already exist (`algorithmic.py`, `planning/`) but were NEVER
wired as a sequence mixer — the closure engine is ready.

**Why it may train where single-hop (2.45) collapsed:** weight-tied S → `grad_S = Σ_t` over all hops
(denser); the `max(H,·)` self-loop is an always-alive residual (non-argmax-gated). Depth densifies the
gradient structurally, not by relaxing the algebra.

**Honest open question (make-or-break):** per-entry backward still routes through ONE argmax; depth
densifies the *aggregate* gradient but not per-entry sparsity (~0.90–0.95, the Viterbi sparsity that
collapsed single-hop). Trainability is ARGUED, not measured.

- [ ] **PROTOTYPE (tiny):** add `n_hops` (T=2–4, k=1) loop around the pull in `TropicalAttention.forward`
      (`nn/__init__.py:~1277`), reusing `_tropical_score` + `_tropical_aggregate`; `h=max(h,pull)` self-loop.
      Screen: char tiny-shakespeare, d=256, 1500 steps, n=3 seeds, vs MorphSSM 1.934 / dense-conv 1.702 /
      single-hop 2.45. Success = clears the 2.45 bigram band; real win = beats MorphSSM 1.934 (routing >
      fixed conv); 2.0–2.2 = first evidence routing ≠ averaging-mimicry.
**MEASURED (2026-06-15, char tiny-shakespeare, d=128/L=3, 1500 steps, this harness — nats):**
- **Fixed-τ (soft-blend): ✗** — τ=0.25 deploy-soft 2.500 / τ=0.5 2.503 vs τ=0 control 2.518 (≈ +0.018,
  negligible) — stays at the bigram floor (2.482). **Blend is not the lever; no log-semiring relaxation
  needed → stay purely max-plus (train+deploy τ=0).**
- **Bellman–Ford diffusion (n_hops, pure max-plus routing): ✗** — n_hops=2 → 2.544, n_hops=4 → 2.548 vs
  single-hop 2.518 (≈, marginally worse). Multi-hop did NOT overcome per-entry argmax sparsity; no gain
  from depth. Not a collapse (acc 0.27 > modal 0.15 = bigram-ish lookup), just no added capacity.
- So tropical attention in EVERY form tested = bigram floor (~2.5); MorphSSM conv mixer (1.934) remains
  the only working tropical sequence mixer. (Code shipped: `--n-hops`, `--tau-floor`, both committed.)

**CAVEAT — wrong regime for attention.** seq_len=96 is where even *dense* attention beats the dense
conv-twin by only +0.078 (no long-range signal to route). We measured tropical routing exactly where
attention has the least to offer → the negative is SCOPED, not fundamental. n_hops=4≈n_hops=2 most likely
because the task needs no long-range routing, not because the mechanism can't.

- [ ] **DISCRIMINATOR (next, dense-only, cheap):** dense-attention vs dense-conv-twin across context
      length {96, 512, 2048}. Attention pulls ahead with context → attention IS the lever at scale, re-test
      tropical routing THERE; conv keeps pace → attention isn't the lever for this data (accept conv
      substrate, or move to a retrieval-flavored task where attention provably matters). Settles whether
      the tropical-attention negatives are scope or substance.
- [ ] **Purity boundary / fallback:** if hard-argmax won't train it, the minimal patch is τ-warmup →
      deploy-hard (τ=0) — keeps the DEPLOYED forward purely tropical (soft crutch at train time only).
      Renormalize per round (additive `x − ½(λmax+λmin)`, closure-clean) to prevent max-cycle saturation
      (`planning/bellman.py:113`). This couples to the fixed-τ sweep above (how much soft-gradient help
      routing needs at train time).

_The morphssm-internal localization below is now a secondary deconfounder (run the cheap matched-budget
float ref + dense-SignSGD control to bank ~0.25–0.30 bpb), not the main thrust. Gap-localization study
(2026-06-15), numbers verified against `results/lm_shakespeare/*.jsonl`:_

**Where the 0.52 bpb actually lives** (corrects two stale leads — attention is NOT the broken piece
at BPE scale, and capacity is NOT binding):

| source | bpb | confidence | evidence |
|---|---|---|---|
| int16 quant + LUT-CE + shift-norm | ~0.00 | HIGH | all sims float32; `diag_lut_balanced` 2.500 ≤ `tropical_ce` 2.523 |
| **substrate** (real→max-plus, AdamW held) | **~0.15** | MED-HIGH | denseconv 2.351 → morphssm-float 2.498 |
| mixer (attn→conv) | +0.078 | LOW (seed noise) | bpeA_dense→denseconv; **net-negative — attention off the critical path** |
| **optimizer** (AdamW→sign update) | **~0.13** | MED | +0.201 nats char (full_morphssm→full_closed) |
| anneal-drop | ~0.16 | MED | +0.245 nats char; **the deployed arm keeps the anneal** |
| **budget + lr-schedule confound** | **~0.15–0.29** | LOW | dense 1.5k/0.77M vs strict 6k/1.0M; strict still descending |

- **Capacity is NOT the limit at this scale:** `full_morphssm_L3` (238k params) 1.9388 *beats* `L7` (512k)
  1.9475 — the morph stack saturates. The L3→L7 saturation is the leading **warning sign** for the
  scale question below.
- The biggest single logged number (+0.573 float→strict at equal 1.5k budget) is the **most confounded**
  in the corpus (budget + params + optimizer all vary) — do not read closure as owning the whole gap.
- The measured 0.52 is an **under-estimate** of the substrate cost: morphssm has 1.3× the params (1.0M vs
  dense 0.77M).

**Decisive first experiment — matched-budget float reference (overnight, ~4 h ×2):** MorphSSM-**float**
+ AdamW + anneal, **6000 steps**, same data/seq/seed/params as `strictH_bpe_anneal_s9decay_6k`, with and
without lr-decay. Deconfounds budget from closure (the largest, most-confounded chunk). Reading: float-6k
≈ 2.30–2.45 → closure tax is small, gap is mostly budget+optimizer (**fixable**); float-6k ≈ 2.498 →
substrate floor ~0.22; float-no-decay also degrades post-anneal → the lr-decay benefit is generic, un-book
the 1.01-bpb no-decay swing from closure.

**Ordered sequence (each gated on the prior):**
1. [ ] Matched-budget float reference (above) — splits budget from closure.
2. [ ] **Dense-net SignSGD control** — `--opt signsgd0` on the dense arm (one CLI flag, ~30 s; the dense
       path is NOT AdamW-hardwired). Is the sign-tax generic or tropical-landscape-specific?
3. [ ] lr-schedule sweep on the strict arm at fixed 6k (`--lr-decay-every {0,250,500,1000,1500}`) —
       deconfounds lr-decay from the wall-clock artifact in the existing uncontrolled pair.
4. [ ] **Dyadic channel-mixer DONE RIGHT** (highest-leverage + the fundamental-vs-fixable settler): add
       `out_proj_type=="dyadic"` → `DyadicTropicalProjection` (2-line `elif` at `tikitorch/nn/__init__.py`
       ~1086; class at nn:610 has `set_tau` parity). The only LM dyadic attempt used the *bare*
       `DyadicLinear` and collapsed (`full_morphssm_dyadic` 3.355) — the MuZero-validated 2-layer
       construction was never tried in the deep-residual LM slot. **Track residual-branch activation std
       per layer** (collapse signature = std explodes ~100× under AdamW, per the bakeoff).
5. [ ] Tropicalize-one-piece-at-a-time on the dense-conv twin (swap only the channel-mix linear → max-plus,
       then only the conv) — isolates which sub-component carries the ~0.15 substrate tax.
6. [ ] Param-match audit — build a ~1.0M dense (or shrink morphssm to ~0.77M) and re-measure.

**Go/no-go — does the gap close or widen with scale?** Paired dense-AdamW vs morphssm-float sweep over
~0.25M / 1M / 4M params (≥2 seeds), each run to near-convergence (matched-budget protocol), plot
**(morphssm_bpb − dense_bpb) vs params**. Shrinks → **GO** (substrate tax amortizes, tropical competes at
scale); constant → conditional go (~0.15 fixed penalty, viable if the DSP-free silicon win pays for it);
widens → **NO-GO for parity** (small-model/edge play only). L3→L7 saturation says this is the real risk.

**Fundamental vs fixable:** the risk concentrates in the **max-plus-linear channel mixer**
(`TropicalDilationModule`, nn:276) — `y_j = max_i(W[j,i]+x[i])`, axis-aligned selectors, no oblique/smooth
normals (the exact primitive the MuZero §2a study showed underfits, and that dyadic ±2^k *fixed* there).
Step 4 settles it: trains-without-collapse-and-closes-gap → **fixable** (pursue dyadic mixers); trains-but-
no-gain → capacity isn't the issue (training-side fix); **collapses again (residual std explodes) →
constraint is STABILITY not expressivity** (max-plus gain-boundedness is load-bearing; need a new
closure-clean normalization admitting oblique normals — the one outcome pointing "fundamental," and it
says exactly what to invent rather than abandon).

**Recommendation:** run #1 (overnight) + #2 (minutes) first — together they cheaply attribute ~0.25–0.30
bpb (budget + sign tax) and will likely shrink the apparent gap well below 0.52; then #4 settles the one
genuinely-fundamental risk for a 2-line edit + a 4 h run.
