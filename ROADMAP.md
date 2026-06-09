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
