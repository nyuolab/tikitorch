# FAQ & troubleshooting

## Training

**My model won't learn / accuracy is stuck at chance.**
Most often the `tau` curriculum, the loss, or the optimizer — in that order:

1. **Annealing too fast.** If hard selected gradients stall, keep `tau > 0` longer and
   compare several schedules. Starting near `0.5–1.0` and reaching zero late in a run
   is a useful initial heuristic, not a universal setting (see
   [the recipe](choosing-a-model.md#use-soft-routing-deliberately)).
2. **Loss mismatch.** Compare the simple routed `TropicalPerceptronLoss`, exact
   host-conditioned cross-entropy, and the LUT-conditioned experimental path. They
   optimize differently. Audit loss value and backward separately before assigning a
   closure label.
3. **Optimizer mismatch.** A sign-only update can be too coarse for a particular
   problem. Establish learnability with an ordinary optimizer, then compare
   `TropicalAdam` or a magnitude-aware optimizer as a controlled change.
4. **Routing collapse.** Inspect route occupancy and compare soft routing, fewer
   layers, residual variants, and an ordinary dense baseline. Do not infer the cause
   from loss alone.

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

- accept it as a **declared boundary**, using the exact registered path (`head` or
  `output_head`, depending on the model), and report that boundary; or
- select a tropical head, then run the structural, executed-path, and full-step checks
  on the remaining recipe.

Use `find_closure_violations(model)` to see exactly which submodules multiply.

**What counts as a violation?**
A *hardware multiply or divide*: `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm` (variance
divide), the cuDNN RNNs, a `1/(1-p)` dropout rescale, a non-power-of-two scale. Adds,
comparisons, max/min, and **power-of-two** shifts are fine. There are three checkers —
`assert_closed` (modules), `assert_closed_runtime` / `find_runtime_multiplies` (the
traced forward), and `assert_training_closed` (forward + backward + optimizer).

**Is using DSP blocks always bad?**
The invariant is *no multiply*, not *no DSP*. DSP48s used as adders/comparators/
accumulators are not evidence of a multiplier by themselves. Software audits classify
operations; HLS and routed resource reports determine how a physical build maps them.

## FPGA / hardware

**Does my code need an FPGA to run?**
No. The normal library surface runs on PyTorch CPU/GPU tensors. The FPGA path is opt-in:
build the local bridge for its supported host-reference paths, or bind a compatible
board with `model.fpga()`. Use `model.fpga(strict=True)` when fallback would invalidate
the run; otherwise inspect the returned wrapper's `bound` property.

**Why does `model.fpga()` run on CPU?**
Some FPGA entry points deliberately fall back when a compatible native session is not
available. Inspect the selected backend and the specific runner's status output. A CPU
fallback is not, by itself, evidence of bit-faithfulness, routed timing, throughput,
energy, or even that the intended board kernel was selected.

## Performance / memory

**A wide tropical linear OOMs.**
Max-plus linears have no GEMM equivalent, and a naive broadcast creates a
`(rows, out, in)` candidate tensor that can be many GB. The hard PyTorch reference tiles
rows, outputs, and input features, retaining only output values and winner indices; its
backward accumulates directly into the `(out, in)` weight gradient. Soft logsumexp
linears still require dense candidates for each bounded output tile. If you still OOM,
lower the batch/sequence or layer width. A streaming compiled kernel has a different
memory path and must be measured separately.

**It's slower than `torch.nn` on my GPU.**
This can happen because eager PyTorch lacks the same optimized kernel ecosystem for
max-plus/min-plus reductions that it has for GEMM. Specialized FPGA/ASIC mapping is a
research motivation, not an automatic speed or energy win. The current claim ledger
explicitly does not support a general end-to-end energy advantage.

## Still stuck?

- Read the runnable [notebooks](notebooks.md) — each isolates one idea.
- Check [CONTRIBUTING](https://github.com/nyuolab/tikitorch/blob/main/CONTRIBUTING.md)
  for the closure invariant and dev setup.
- Open an issue with a minimal repro and the output of `find_closure_violations(model)`.
