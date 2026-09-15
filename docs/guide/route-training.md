# Training with device-resident route tapes

Hard additive operators can determine input and parameter gradients from winner
routes and declared auxiliary state. TikiTorch exposes that property through an
experimental first-order MorphSSM training path:

```python
import torch
import tikitorch as tt

torch.manual_seed(0)
model = tt.models.MorphSSM(
    d_model=16, n_layers=1, kernel_size=3, dropout=0,
    task_type="lm", vocab_size=11,
)
model.set_tau(0)
ids = torch.randint(0, 11, (2, 12))
targets = (ids + 1) % 11
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
optimizer.zero_grad(set_to_none=True)
logits = tt.autodiff.route_logits(model, ids)
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
loss.backward()
optimizer.step()
```

Move **both** model and input/target tensors to CUDA to use that device. There is
no implicit CPU fallback in the route training path. This example uses ordinary
cross-entropy and AdamW: it does not claim whole-step multiplier-free closure.

## Storage and derivative contract

- Forward records each layer's routes immediately; it does not build the research
  engine's activation-cache list. Live per-layer temporaries still contribute to
  peak memory. Output logits, the loss boundary, parameters, and optimizer state
  are separate from route bytes.
- `DeviceRouteTape` uses byte-aligned uint8 fields on the input device. CPU and
  CUDA use the same tensor packing kernels. The kernels are compilable PyTorch
  implementations, not hand-tuned device-specific performance claims.
- `packed=False` keeps owned int64 route fields for a matched unpacked baseline.
  `to_bytes()` explicitly exports the portable `PackedRouteTape` format; export
  may transfer data to CPU. Serialization is unnecessary during ordinary training.
- `route_logits` saves only route byte tensors through PyTorch saved-tensor hooks.
  It supports first-order backward, repeated forward calls, and ordinary losses
  and optimizers. Higher derivatives are explicitly rejected.
- Supported configuration: token-embedding MorphSSM LM, single-scale depthwise
  hard convolution, hard tropical projection, tropical normalization, unit
  residual scale and top-k head. Soft layers, grouped/multiscale convolution,
  recurrent/action-head extensions and aliased parameters require other contracts
  and are rejected. This is not generic activation-free PyTorch training.
  Overridden forwards, module hooks and non-default embedding gradient options
  are also rejected: the streaming path cannot silently ignore their semantics.
- Winner ties use the lowest candidate index; top-k ties retain ascending index
  order. This updates software tie behavior. Historical hardware and archived
  experiment records are **not** retroactively assigned the new convention.
- The fingerprint binds configuration and dtype, not parameter values. Supported
  additive parameter VJPs use recorded routes rather than current weight values.
  Replay still requires the same configuration and correctly shaped cotangent.
  Older `model-produced` fingerprints must be replayed with their archived
  implementation; do not relabel them as first-index records.

For manual or custom-loss replay:

```python
logits, tape = tt.autodiff.morphssm_forward_tape(model, ids)
cotangent = torch.ones_like(logits)  # replace with your loss's dL/dlogits
grads = tt.autodiff.morphssm_vjp(tape, model, cotangent)
```

The returned dictionary does not mutate `.grad`. The historical
`replay_morphssm_vjp` convenience API retains its mean-CE boundary convention.
Exactness refers to the declared selected derivative; floating-point reduction
orders can differ within dtype-appropriate tolerances.

## AdamW numerical diagnosis

The native compiler comparison compiles the model/reverse path, **not AdamW's
update**. On the diagnosed CPU stack, a changed mean-reduction order produces
tiny upstream rounding differences. Cross-entropy cancellation can leave signed
residuals even along a structurally zero-gradient direction. For zero initial
moments, AdamW's adaptive update is `-lr*g/(abs(g)+eps)`: its default `eps=1e-8`
can turn a `1e-8` gradient discrepancy into a learning-rate-sized update.

The untimed diagnostic separates identical-input primitives, identical-logit
loss derivatives, identical-tape VJPs, equal-gradient optimizer updates, and
independent fresh-batch trajectories. The
[source-bound evidence](https://github.com/nyuolab/tikitorch/blob/main/manuscript/research/EVIDENCE.md#adamw-numerical-diagnostic)
preserves a source snapshot recording both default failure and a bounded mitigation: explicit `eps=1e-4`
passes 20 updates across three seeds and compiled autograd/packed/unpacked paths
on the tested small FP32 CPU configuration, without changing tolerances.
This uses **standard PyTorch AdamW**, not `tt.optim.TropicalAdam` (a different,
Rprop-like algorithm). It does not establish an optimal epsilon for task learning.

```bash
# Verify selected historical evidence against its pinned original sources:
python -m manuscript.tools.historical_cpu_evidence --check
# Untimed native diagnostic; choose a new output path for each reproduction:
python -m scripts.route_certificates.optimizer_diagnostic --steps 20 \
  --eps 1e-8 1e-6 1e-5 1e-4 --output manuscript/build/adamw-reproduction.json
# Inspect a separately configured benchmark recipe without starting it:
python -m scripts.route_certificates.training_benchmark --dry-run --adamw-eps 1e-4
```

All benchmark arms and learning controls receive the same recorded epsilon;
the default remains `1e-8`. Validate the intended shape/device/precision and
learning consequences before adopting a recipe. Larger epsilon is not a
long-horizon route-stability guarantee, and failed default cells remain failures.
These historical-source checks are not fresh measurements of the current
library. The original diagnostic's direct `--check` remains strict against the
current checkout and rejects stale records. Strong benchmark comparisons and
LM pretraining are subsequent work.

## Integer error-feedback contract

The experimental `tt.optim.fixed_point_ef_step` exposes the exact integer update
used by the CPU MorphLM diagnostic. For `Q = 2**k`:

```text
a = remainder + raw_gradient
update = trunc_toward_zero(a / Q)
weights_next = weights - update
remainder_next = a - update * Q
```

```python
import torch
from tikitorch.optim import fixed_point_ef_step

w = torch.tensor([10, -10], dtype=torch.int16)
r = torch.zeros(2, dtype=torch.int32)
g = torch.tensor([1, -1], dtype=torch.int32)
for _ in range(4):
    w, r = fixed_point_ef_step(w, r, g, k=2)
assert w.tolist() == [9, -9] and r.tolist() == [0, 0]
```

Run `python -m examples.fixed_point_workflow` for a checked carry/resume example.
The function returns new tensors, accepts matching-shape dense strided CPU
tensors only, rejects noncanonical remainders (`abs(r) >= Q`), int32 accumulator
overflow and int16 weight clipping. Save **weights, remainder and fixed k** in
checkpoints. Do not reset remainders between minibatches or reinterpret existing
remainders with a different k.

What the math buys: `z = w - r/Q` obeys `z_next = z - g/Q` exactly for the supplied
gradient sequence, while the visible integer weight differs from z by less than
one weight unit. Deferred sub-unit updates are retained. This does **not** prove
loss descent or equivalence to a floating model that queries different gradients
along its own trajectory. Upstream gradient accumulation must also be checked.

This is a primitive for callers that already supply integer gradients in declared
units, not a drop-in AdamW replacement or automatic quantizer for autograd. The
MorphLM study supplies summed raw LUT gradients and its own integer surrogate
backward; neither its k nor its quality results transfer unchanged to a mean-CE
PyTorch training loop. The existing `EFTropicalSGD` uses floating residuals and at
most one emitted step per call, with optional adaptive/leaking/momentum behavior;
it is a different algorithm. No optimizer default changes here. CPU parity does
not establish physical FPGA compatibility, speed, energy or a zero-DSP design.

## Matched full-step benchmark

```bash
python -m scripts.route_certificates.training_benchmark --dry-run
python -m scripts.route_certificates.training_benchmark --device cpu --adamw-eps 1e-4 --reps 5
# A matched momentum-SGD recipe, also checked by the native compiler gate:
python -m scripts.route_certificates.training_benchmark --optimizer sgd --reps 5
# Fast graph-capture correctness smoke; not optimized-kernel performance evidence:
python -m scripts.route_certificates.training_benchmark --compile-backend aot_eager --warmup 0 --reps 1
# Run only on an existing authorized CUDA machine:
python -m scripts.route_certificates.training_benchmark --device cuda --batches 2 4 8 --sequences 96 192 --width 128 --layers 4
# Optional real corpus, with a frozen contiguous 85/15 character split:
python -m scripts.route_certificates.training_benchmark --corpus /path/to/corpus.txt --learning-steps 500
```

The six arms are eager autograd, fullgraph-compiled autograd, eager checkpointing,
fullgraph-compiled checkpointing, fullgraph-compiled unpacked routes, and
fullgraph-compiled packed routes. Every
arm starts with the same seeded parameters/data and performs the same number of
complete forward/reverse/optimizer updates (AdamW by default, momentum SGD with
`--optimizer sgd`). `--adamw-eps` is shared across every AdamW arm and both
auxiliary learning controls, including when timing momentum SGD as a separate
recipe. One startup step (including compilation
when applicable) is reported separately from warmup and steady-step timing.

Both checkpoint arms test per-layer, grouped, and selective policies. Selective
checkpointing is unavailable on older PyTorch versions and is reported as such;
it is not silently replaced. The fastest successful checkpoint policy is chosen
within an optional CUDA allocator budget, separately for eager and compiled
checkpointing. The best baseline includes both. A compiler failure makes a missing arm
inconclusive, never a speedup against an omitted baseline. `aot_eager` validates
graph capture but is not an optimized systems result.

The selective policy includes the actual `amax`/`amin` projection reductions.
A separate untimed forward/backward records visited operations and save/recompute
policy decisions. Eager counts are runtime policy calls; compiled counts are
compiler-trace policy calls, **not** physical kernel counts, saved bytes or
measured recomputation savings. Requested but unobserved targets remain visible;
a selective cell without observed coverage cannot win a comparison.

Known local limitation: on the diagnosed PyTorch 2.10 CPU/Inductor stack,
native selective checkpointing passed the one-layer pilot but a two-layer
SGD case terminated with `SIGSEGV`. The normal benchmark isolates this in its
worker and excludes it; the optional native-policy test also runs in a separate,
timeout-bounded process. Do not infer broader selective support from either
the one-layer success or graph-capture-only checks.

Every measured arm also receives a matched gradient-and-update value check:
its measured final parameters, gradients and loss are compared with eager after
the same complete update count, not a separate fresh single-step run. Failed or
timed-out subprocesses cannot supply successful rows even if they wrote JSON.
Compiler rounding differences near zero gradients can be amplified by AdamW's
small default epsilon: a cell may pass a gradient tolerance yet fail the update
tolerance, including ordinary compiled autograd. Such a cell remains inconclusive;
do not relax its tolerances to manufacture a speedup. Optimizer choices are
explicit protocol parameters, not interchangeable evidence.

Each resource cell runs in its own timeout-bounded subprocess. CUDA reports peak
allocated/reserved memory, synchronized full-step latency, and actual tested
batch/context cells. CPU reports process RSS (including interpreter/compiler),
**not** tensor allocator peak. No metric here is measured DRAM traffic or energy.
`measured_fit_cells` reports actual tested fits with a separate parity status;
only validated fits support comparisons, without extrapolating a maximum.
Source hashes include the runner, its executable protocol/provenance helpers and
new library source files as well as the Git revision and patch.
Rows also record the constructed model's causal receptive field separately from
the timed sequence length. With kernel size three, one/two/four layers have
spans of 3/7/31 tokens; long tensor timing is not evidence of long-context learning.
Summaries require equal initialization/data hashes and update counts, and source
changes during the run invalidate completion. An eager validation OOM is recorded
separately from whether the measured packed execution fit; unvalidated fits are
not performance wins. Auxiliary route-byte accounting is explicitly labeled and
does not measure autograd or checkpoint retained state.

The runner also checks a short same-model training trajectory and held-out draws
of a synthetic token-successor task against a dense token baseline. This bounded
learning regression is not a competitive language-modeling result or an
iso-quality hardware comparison. Real-task quality, GPU resource measurements,
and matched physical hardware evidence remain separate research experiments.
Each synthetic run must reduce both models' held-out loss below 75% of its initial
value and exceed 80% accuracy to complete successfully; finite metrics alone do
not pass the learning gate. Very short learning budgets can correctly fail it.
With `--corpus`, all arms use matching next-character windows; evaluation comes
only from the held-out suffix, with a vocabulary learned on the training prefix
and an explicit unknown token. The learning comparison then uses a conventional
GRU baseline and records accuracy, loss, perplexity, update count and parameter
count. Corpus bytes are hashed in the manifest. A single corpus/seed run does not
establish matched-quality superiority; sweep seeds/budgets before making that claim.
Corpus learning results are labeled `observed`, not a passed quality gate; a
completed corpus run establishes execution, parity and recorded metrics only.

The older `compiled_resource_benchmark` remains the historical preregistered
protocol; do not combine its serialized-CPU arm or stored rows with this protocol.
