# FAQ & troubleshooting

## Training

**My model won't learn / accuracy is stuck at chance.**
Most often the `tau` curriculum, the loss, or the optimizer — in that order:

1. **Annealing too fast.** If you drive `tau → 0` early, you train the sparse hard op
   from scratch and it stalls. Start `tau ≈ 0.5–1.0`, reach `0` only at ~80% of training
   (see [the recipe](choosing-a-model.md#the-tau-softhard-recipe)).
2. **Perceptron-loss floor.** `TropicalPerceptronLoss` is the cheapest closed loss but
   leaves the most on the table. Switch to `nn.TropicalCrossEntropyLoss` — it is the
   single biggest lever (~+17pp on hard char-LM), still multiplier-free with
   `softmax="lut"`.
3. **Optimizer underfit.** `SignSGD` is robust but coarse. Try `TropicalAdam` or the
   magnitude-aware optimizers (`FrexpMagnitudeSGD`, `ExponentEMAAdam`).
4. **Deep collapse.** A *deep* untrained tropical stack can route everything through a
   few cells (its linear-region count drops with depth at random init — see
   [Tropical geometry](tropical-geometry.md)). Use residual/identity-style blocks
   (`MorphSSM` skips, `residual_scaling`) or fewer, wider layers.

**Loss is `inf` / `nan`.**
Tropical identities use `±inf` (the semiring's additive identity). A genuine `nan`
usually means an all-`inf` reduction (e.g. an over-aggressive causal mask, or an empty
top-k). Check masks and `agg_topk_k`; confirm inputs are finite.

**Do I call `set_tau` on the loss?**
No need — but it is a no-op on the loss modules (`TropicalCrossEntropyLoss.set_tau` does
nothing), so a uniform `for m in model.modules(): m.set_tau(t)` loop is safe.

## Closure (`assert_closed`)

**`assert_closed(model)` raises on my `nn.Linear` head.**
That is by design — a dense head is a real multiply. Either:

- accept it as a **declared boundary**: `assert_closed(model, allow={"head"})`, and
  report both the boundary and the fully-closed numbers; or
- set `head_type="tropical"` for a model that passes outright.

Use `find_closure_violations(model)` to see exactly which submodules multiply.

**What counts as a violation?**
A *hardware multiply or divide*: `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm` (variance
divide), the cuDNN RNNs, a `1/(1-p)` dropout rescale, a non-power-of-two scale. Adds,
comparisons, max/min, and **power-of-two** shifts are fine. There are three checkers —
`assert_closed` (modules), `assert_closed_runtime` / `find_runtime_multiplies` (the
traced forward), and `assert_training_closed` (forward + backward + optimizer).

**Is using DSP blocks always bad?**
The invariant is *no multiply*, not *no DSP*. DSP48s used as adders/comparators/
accumulators are allowed and wanted — the audit targets multiplies specifically.

## FPGA / hardware

**Does my code need an FPGA to run?**
No. Everything runs on CPU/GPU in PyTorch. The FPGA path is opt-in: build the LOCAL
bridge (`bash scripts/build_fpga_local.sh`) to exercise the kernels with no board, or
bind a real board with `model.fpga()`. Without hardware it transparently falls back to
the CPU op — same numbers.

**Why does `model.fpga()` run on CPU?**
No board / driver detected, so it falls back (by design). The fallback is bit-faithful
to the kernel; you only need the board for throughput/energy, not correctness.

## Performance / memory

**A wide tropical linear OOMs.**
Max-plus linears have no GEMM equivalent and materialize a `(batch, out, in)`
intermediate; a big FFN at long sequence can be many GB. `nn.TropicalLinear` already
**tiles over the output dim** above a threshold (`_chunked_linear`) so peak memory is
bounded — identical math, chunked. If you still OOM, lower the batch/sequence or the
layer width. (On FPGA/ASIC the reduction streams with no materialization — that is the
point of being multiplier-free.)

**It's slower than `torch.nn` on my GPU.**
Expected. Eager PyTorch has no fused max-plus kernel, so the tropical ops materialize
intermediates a GEMM would not. The speed/energy win is on **FPGA/ASIC**, where there is
no multiplier to begin with — not on a GPU built around dense multiply-accumulate.

## Still stuck?

- Read the runnable [notebooks](notebooks.md) — each isolates one idea.
- Check [CONTRIBUTING](https://github.com/nyuolab/tikitorch/blob/main/CONTRIBUTING.md)
  for the closure invariant and dev setup.
- Open an issue with a minimal repro and the output of `find_closure_violations(model)`.
