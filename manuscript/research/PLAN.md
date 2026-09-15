# Research plan: useful learning on FPGA

Status: **FPGA-first language-model pretraining**, reprioritized 2026-09-09 by
the project owner. AWS research spending below **$1,000 cumulative for this
continuation** is authorized; spending above that requires further approval.
This is not a per-job allowance or permission to change unrelated resources.
No public release, authorship decision or submission is authorized. Separate
development tuning from final evaluation; never call choices made after viewing
results preregistered. The [manuscript README](../README.md) remains the handoff.

### Current handoff state (2026-09-15)

The declared same-objective Gram-preconditioned readout producer and separate
independent audit both finish successfully. The solve reaches its unchanged
200-iteration cap: sampled-training/development CE **1.9788 / 2.6039**, versus
**2.0612 / 2.5736** for the earlier affine endpoint. The original-coordinate
numerical gap estimate is **0.01504**, above **1e-5**. Conditioning improves
training fitting but worsens development CE; neither numerical completion nor
the joint quality-progress gate passes. No recipe is promoted. Review this
tradeoff with Xiang before declaring a further bounded experiment; no automatic
solve extension, new cohort or paid scaling follows. See the
[completed evidence](EVIDENCE.md#same-objective-gram-preconditioned-readout-2026-09-15)
and [shared handoff](../README.md#handoff-for-xiang).

Both actual terminal exits and the byte-integrity postcheck completed before
this status update. The saved per-run declarations and all 86 scientific sources
remain immutable. Sections below preserve the chronological protocols and
original decisions; the current handoff takes precedence for what remains next.

### Original execution order (2026-09-09)

The primary hypothesis is that additive hard routing makes **complete learning
efficient on custom FPGA hardware**, even where generic CPU/GPU kernels show no
benefit. CPU/GPU execution remains useful for mathematical references, modeling
controls and cost comparisons; a GPU speedup is not a prerequisite for the FPGA
study. A hardware advantage still requires useful held-out quality and measured
complete-system cost, not only fewer arithmetic operations or zero DSPs.

The immediate sequence is **fresh-window software correctness → existing-image
FPGA training and held-out evaluation → profile → memory/compute redesign →
larger language-model pretraining**. ECG is now a secondary option. The current
D512 image is a 1,196,032-parameter character model, not an LLM result. Its prior
repeated-window training establishes execution feasibility, not pretraining.
Its integer/LUT/SignSGD recipe is distinct from the CPU AdamW epsilon experiment.

Pilot G now passes the first **physical FPGA full-checkpoint gate**: 64 fresh
updates, with identical complete weight payloads at updates 0, 32 and 64 in
C-simulation and silicon. It does not pass the useful-context-learning gate.
The CPU/CSIM head-repair screen and subsequent 1,024-window extension below pass
their gradient-flow and context-use gates for both anchored-head recipes.
Neither beats the matched-data bigram, so larger paid learning remains gated.
The fine-scale recipe gains accuracy but loses NLL quality with more training.
The completed confidence diagnostic finds that fixed dyadic rescaling alone
does not repair that gap. Next test intermediate scales during training, not
post-hoc changes to these failed gates. See the
[paired hardware evidence](EVIDENCE.md#current-code-fpga-checkpoint-equivalence-pilot).

The optimizer gate must distinguish a wrong derivative/update from floating-point
trajectory sensitivity. Compare identical inputs and states, isolate the first
different primitive, check common-cotangent VJPs and equal-gradient optimizer
updates, then examine repeated fresh batches. Any epsilon, precision or optimizer
change defines a separately documented recipe shared by all benchmark arms.
Do not erase the default-recipe failure or select a new tolerance to make it pass.

#### Completed CPU groundwork

The previous continuation established a development pilot, not a final
quality/resource study. Keep FP32 CPU, stock eager AdamW (`lr=1e-3`,
`eps=1e-4`, betas `0.9/0.999`, weight decay `0.01`) and the existing `1e-4`
absolute/relative comparison tolerances. Extend the untimed fresh-batch
diagnostic to `(D,L,T,B,V)=(32,2,24,2,11)`, 20 updates, seeds 0/1/2,
compiled autograd and packed/unpacked routes. Preserve every outcome; do not
choose a new epsilon after a failure and call it the original protocol.

Add compiled-checkpoint layer/grouped/selective candidates alongside eager
checkpointing. Instrument selective-policy observations outside resource
timings and distinguish eager execution observations from compiler-trace
observations. An unsupported policy or unobserved coverage is not a win.
After source freeze, refresh the original optimizer record append-only and
run an initial `(16,1,12,2,11)` all-arm CPU timing pilot with two warmups,
five repetitions and a 60-second per-worker timeout. Report startup separately,
verify the measured final state, and retain failures. This is a single-seed
implementation pilot; CPU RSS is not peak tensor memory and synthetic successor
learning is not language-model quality. No dataset download, accelerator job,
paid compute or submission was included in that earlier pass. See the
[evidence ledger](EVIDENCE.md#current-cpu-comparator-pilot) for its outcomes.

#### First FPGA milestone and spending controls

Reuse the existing D512 image if ownership, availability, shell/toolchain and
exact source identity can be verified. Do not pay for new synthesis just to
repeat the old demonstration. The first experiment stages fresh 128-character
training windows, performs one fused fabric update per window, preserves model
weights across kicks, writes periodic full checkpoints and evaluates frozen
held-out text. Compare full arrays against newly built C-simulation at selected
checkpoints, not only checksums or 16 predictions. The host may stage data and
evaluate snapshots; forward/loss-gradient/backward/update for training must stay
on FPGA. Any host training computation must be disclosed as a changed boundary.

Use the already present `data/tinyshakespeare.txt` for an initial character-LM
development pilot (1,115,394 ASCII bytes; SHA256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`).
On 2026-09-09 the local bytes were independently hashed against the exact
[upstream char-rnn file at commit 370cbcd](https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt)
and matched. Retain that provenance and review corpus terms before upload;
the data remain ignored and are not bundled with library sources.
Freeze contiguous train/validation/test partitions and sequence boundaries;
do not clip unknown bytes to token 127. Validation is for development; leave
the final test untouched. Report loss under the explicitly normalized LUT
distribution, not ordinary softmax perplexity. Add train-fitted unigram/bigram
and context-removal controls before interpreting learning as useful. This
pilot neither establishes general language competence nor a scaled LLM.

Local development follow-up, fixed before execution: after the D16 eight-update
and D512 two-update ABI/reference smokes, run a D512 64-update C-simulation
reference (`seed=0`, `lr=1`, `threshold=2048`, validation every 32 updates on two
windows) for the proposed short silicon pilot. Then examine learning over
**1,024 fresh D512 windows** with the same seed/update recipe, validation every
256 updates on eight windows, and a 180-second local process timeout. This is
one development trajectory, not a multi-seed result, tuning sweep, hardware
measurement or matched-data-budget baseline comparison. Do not choose a new
optimizer after viewing the outcome and call it this protocol. Retain every
outcome append-only; the unigram/bigram controls use the entire training
partition, while these pilots see only their recorded first training windows.
Local concurrent regression tests make CPU timings unsuitable for a systems
comparison. These runs incur no AWS spend and do not evaluate the final test.

**Outcome and next gate:** the 1,024-update run and a subsequent frozen-checkpoint
[input/head diagnostic](EVIDENCE.md#checked-input-sensitivity-and-head-routing-diagnosis)
improve NLL but do not demonstrate context use. On 949 of the 1,024 inspected
validation positions, a shared head winner causes exact cancellation of its
head-input gradient. Before increasing paid pretraining, test route diversity,
head initialization, logit/LUT scaling and threshold effects using independently
declared numeric recipes. Compare simple controls fitted to the same number of
training tokens, and measure input/context sensitivity before and during
training. Do not assume that a lower loss or a nonzero parameter update is
contextual learning. The short existing-image correctness pilot remains useful;
larger hardware training is gated on a meaningful learning improvement.

### CPU head-repair screen v1 (completed)

Keep the existing native sources and silicon-correctness recipe frozen. Use a
new versioned host harness for the following screen. The protocol below was
declared before execution; its outcome and next experiment are recorded afterward.
The mechanism is structural: when all classes choose feature `d*`, the
head-input cotangent is `sum_v g_v = 0` for the balanced loss. Changing only the
learning rate, threshold or temperature cannot recover that exactly cancelled
cotangent while the routes remain shared. Do not remove zero-sum balancing to
manufacture a nonzero gradient.

Use D512/four layers/vocabulary 128/window 128/batch one, integer `lr=1` and
`threshold=2048`. Pair seeds **0, 1, 2** across four arms. Generate the unchanged
initial buffer first so all non-head weights are byte-identical across paired
arms; then modify only the head and declared loss configuration.

| Arm | Initial head weights | `zshift` |
|---|---|---:|
| Reference | Existing deterministic jitter in `[-4,4]` | 4 |
| Wider head | Integers in `[-512,512]` from a separate existing-LCG stream seeded by `seed XOR 0xA5A5A5A5` | 4 |
| Distinct anchors | Keep jitter at `d=(5*v+seed)%512`; subtract 8192 from other entries of each vocabulary row | 4 |
| Anchors / finer LUT index | Exactly the same initial weights as distinct anchors | 0 |

Anchors are a restrictive mechanistic control, not a proposed optimal head.
Measure whether they actually diversify winners; do not assume it. These are
host-staged weight/configuration changes compatible with the existing kernel
interface, not synthesis changes. A new reduction, post-residual normalization,
geometry or per-tensor optimizer would change that boundary and require a new
image.

Run exactly **256 fresh windows per arm/seed** (32,768 supervised character
pairs), in the same order. Evaluate initialization and update 256 on validation
windows **8–23**, leaving the test split untouched. Save every initial/final
checkpoint and failure; no adaptive extension of weak arms. Fit unigram and
bigram controls on exactly those supervised targets/pairs, vocabulary 128 and
Laplace smoothing `alpha=1`; do not use the rest of the training partition or
create transitions across windows. Report the model's LUT-normalized NLL and
per-window paired differences, not conventional softmax perplexity.

Prespecified screening gates, **not statistical-significance claims**:

- Mechanism: for every seed, exact-zero head-input VJP on at most 25% of positions
  with a nonzero logit-gradient vector, at least 16 distinct head winners on at
  least 75% of evaluated positions, no observed int16 saturation, and independent
  native prediction/loss-gradient agreement. Report rounded-zero target masses
  separately and inspect cancellation in the earlier projection layers.
- Context use: every seed beats its matched-data unigram by at least 0.05
  nats/token; removing only past temporal taps worsens final NLL by at least
  0.02, and that intact-context advantage increases by at least 0.02 over
  initialization. Reversed/all-space inputs remain sensitivity probes.
- Before a larger paid learning benchmark: additionally beat matched-data
  bigram by at least 0.02 nats/token in the seed mean, with no seed losing by
  more than 0.02. Otherwise report trainability repair, not LM superiority.

**Outcome (2026-09-09):** all twelve cells completed without a native/reference
mismatch. Both anchored-head arms pass the mechanism and context-use gates in
every seed. The finer LUT (`zshift=0`) is strongest: mean validation NLL **2.7213**,
versus **4.0604** for the original recipe, **3.2254** for matched unigram and
**2.6210** for matched bigram. Removing past taps worsens its NLL by
**0.184–0.328** nats/token across seeds. No arm passes the larger-paid-learning
gate. This is a small-model trainability improvement in C-simulation, not a new
silicon or efficiency result. See the [full screen evidence](EVIDENCE.md#head-routing-repair-screen-v1).
The executed manifest retains the hash of the pre-outcome plan; this updated
document does not replace that historical protocol identity.

### CPU learning-budget extension v1 (completed)

First test whether additional fresh data closes the remaining **0.1003 nats/token
mean gap to the matched bigram** before changing the projection geometry. The
observed projection-branch cancellation differs substantially across layers and
seeds, and does not by itself establish the cause of the remaining quality gap.

- Pair the two anchored-head recipes (`zshift=4` and `0`) across seeds **0,1,2**:
  six fresh-process, from-initialization runs, **1,024** fresh training windows
  each. Keep D512/four layers/128 symbols/window128/batch1, anchor construction,
  `lr=1`, `threshold=2048`, all non-head initialization and native sources fixed.
- Evaluate checkpoints **0,256,1,024** on validation windows **24–39**. These
  windows were not evaluated in v1; they remain development validation, not the
  untouched final test. Do not compare this new cohort's absolute metrics with
  v1's cohort as if they were paired observations.
- At update256 fit controls to exactly **32,768** supervised pairs; at update1,024
  use exactly **131,072**. Keep within-window pairing, vocabulary128, `alpha=1`,
  context-removal probes, per-layer diagnostics and complete checkpoints.
  Initial model evaluation is the no-update control; fitted-baseline comparisons
  are made at the two trained checkpoints. Require update256 full weights to
  match the corresponding v1 arm/seed byte-for-byte, despite the changed
  frozen-weight validation cohort.
- Freeze a new versioned protocol/harness before execution; retain all six cells,
  use a **900-second per-worker timeout**, and do not extend weak cells adaptively.
  Apply the same final mechanism/context/bigram thresholds as v1. The
  initial-to-final context gain is measured on the new fixed validation cohort.
- Do not change projection initialization in this budget experiment. If the
  quality gate still fails, declare a separate paired projection-route control
  with identical head, loss, data and budget to test the proposed mechanism.

No additional AWS allocation or new FPGA image is part of this extension.
Any selected improved recipe must repeat the full FPGA/CSIM checkpoint gate
before a larger hardware benchmark; host initialization/configuration
compatibility alone is not a silicon validation result.

**Outcome (2026-09-09):** all six declared cells completed. On the same new
validation cohort, mean NLL changes **3.0635 → 2.7366** for the coarse recipe and
**2.8045 → 2.9006** for the fine recipe, from update256 to update1,024.
The final matched bigram is **2.5933**; neither recipe passes the paid-learning
gate. The fine recipe's accuracy increases in every seed while rounded-zero
target masses increase to **34, 24 and 30 / 2,048** positions. Head-input
gradients remain nonzero and projection-branch cancellation often decreases.
This does not support treating projection collapse as the sole remaining cause.
The executed manifest preserves the pre-outcome plan hash.

### Post-hoc fixed-checkpoint confidence diagnostic

The observed accuracy/NLL reversal motivates this **exploratory development
diagnostic**, not a retrospectively preregistered experiment or a replacement
for the failed learning-budget gates. Before executing the diagnostic, fix:

- Both anchored training recipes, all three seeds, and both trained checkpoints
  **256 and 1,024**; evaluate every combination of LUT shifts **0,1,2,3,4** on
  the same validation windows **24–39**. No new training or test evaluation.
- Extract saturated logits from the immutable full checkpoints. At the original
  training shift, verify every prediction and rounded loss-gradient entry
  against retained native arrays and the frozen integer reference. Rehash
  sources, artifacts and the completed extension audit before and after.
- Report all 60 checkpoint/shift cells, per-window NLL, accuracy, confidence,
  rounded-zero target masses, last-LUT-bin targets, per-token loss quantiles and
  the loss contribution of those named severe-event groups (not quantile-selected
  tail groups). An evaluation-only shift cannot change
  argmax accuracy or show that training at that shift would work.
- Retain every grid outcome and label any favorable rescaling as post-hoc on
  development validation. Do not use it to open the original paid-learning gate,
  claim an optimizer fix, or promote a selected recipe without a separate
  fixed training/validation protocol and FPGA checkpoint verification.

**Outcome (2026-09-09):** all 60 cells completed with original-shift native-array
agreement. Fine-scale final mean NLL is **2.900561** at its trained shift0 and
**2.906081** at evaluation shift1; larger shifts are worse. Coarse-scale weights
also perform best at their original shift4 among this grid. There is no
matched-bigram repair from these fixed evaluation rescalings. Severe-error
events explain the fine recipe's aggregate NLL reversal, but reducing those
errors by rescaling also weakens confidence on correct predictions. This is
not evidence that a different training scale cannot work, or that projection
collapse is the sole cause. See the
[complete confidence evidence](EVIDENCE.md#fixed-checkpoint-confidence-diagnostic).

### Intermediate-scale training control (completed)

All six new training cells and six historical checkpoint controls completed on
2026-09-09. On the shared validation40–55 cohort, shift2's final mean NLL is
**2.783547**, versus matched bigram **2.660153**: a **0.123394** deficit.
Shift1 deteriorates from mean **2.749688** at256 to **2.862251** at1,024.
All four recipes pass mechanism/context checks but fail the unchanged bigram
gate. The declaration below is retained; a separate optimizer experiment follows.

The two training scales tested so far are endpoints, separated by a factor of
16 in LUT index scale. Test the intermediate training recipes before a geometry
change: the offline confidence sweep does not test how their gradients would
change learning. This follow-up is chosen after the above development results;
it is not part of either completed protocol.

- Two **new training** recipes, anchored head with `zshift=1` and `zshift=2`,
  paired seeds **0,1,2**, from initialization for exactly **1,024 fresh windows**.
  Keep all native sources, head construction, geometry, `lr=1` and
  `threshold=2048` unchanged. Match actual complete initial weight bytes to the
  earlier anchored controls; trained weights are not expected to match across
  different loss scales.
- Evaluate **0,256,1,024** on a new fixed development cohort, validation windows
  **40–55**, always at that recipe's training shift. Re-evaluate the existing
  immutable shift0/4 checkpoints on this same cohort as controls, without
  retraining them or substituting their old-cohort scores. Leave test untouched.
- Refit alpha1 unigram/bigram at exactly 32,768 and 131,072 supervised pairs;
  initialization remains a no-update control. Keep original context-removal,
  native-array/route, int16-clipping and head/projection-gradient diagnostics.
- Freeze a new versioned harness before execution. Retain every cell and full
  checkpoint, use the same 900-second per-training-worker cap, and apply the
  unchanged per-seed mechanism/context and mean/per-seed bigram thresholds.
  No adaptive extension or evaluation-scale selection within this experiment.

The recipe choice is exploratory; improvement would need independent validation
and the physical FPGA/CSIM full-checkpoint gate before a larger paid benchmark.
If neither intermediate scale helps, revisit the separately controlled projection
initialization or update schedule; do not simultaneously change both and assign
causality. No AWS launch or new image is included in this CPU experiment.

### Separate bigram representability control (completed)

All four constructed checkpoints completed native verification on2026-09-09.
Each gives identical LUT NLL **2.660464**, only **0.000311** above the floating
bigram **2.660153**, without observed clipping or use of temporal context.
This is a representability result, not successful gradient learning.
The original declaration and construction follow.

Alongside the intermediate-scale study, test whether the unchanged architecture
and integer LUT can represent the matched bigram itself. This is **host count
fitting and parameter construction**, not gradient learning or an FPGA-pretraining
result, and cannot substitute for a trained recipe passing the quality gate.

Fit alpha1 bigram counts to the first **1,024 fresh training windows / 131,072
supervised pairs**, with no cross-window transitions. Use one deterministic
seed0 anchor layout. Set each head anchor weight to zero, non-anchor head weights
to `-8192`, and temporal/projection/bias weights to zero. For each input token
`t` and output symbol `v`, set its anchor embedding coordinate to
`-2^z * floor(16 * log2(max_u(C[t,u]+1)/(C[t,v]+1)) + 0.5)`
(nearest-integer, ties upward); unused coordinates are zero.
Evaluate all four fixed scales `z=0,1,2,4` on validation windows **40–55**.
The construction is fixed from training counts, with no validation-based choice.

Without saturation, the zero projection matrices add only a common scalar to
all hidden channels, preserving the embedding logit differences. Verify that
property, actual anchor winners, complete unchanged checkpoint bytes, native
prediction/rounded-gradient agreement and absence of clipping; do not assume
them from the formula. Compare the resulting normalized-LUT NLL with the matched
floating bigram and report quantization/tie differences. Keep all four cells,
full constructed checkpoints and source identities, using fresh native processes
with bounded timeouts. Leave final test untouched and make no AWS calls.

A successful representation control would point toward optimization rather than
an unavoidable bigram expressiveness limit; it would not identify a unique cause.
If used as a future warm start, the host fitting pass and subsequent gradient
passes must be separately counted and disclosed. It is not the current
from-initialization one-pass training protocol.

### Integer optimizer repair v1 (completed)

The fixed-gradient diagnostic and all four native CPU learning cells completed.
The count-fitted `k=18` recipe reaches NLL **2.618411**, beating the stronger
bigram **2.660153** by **0.041742**, with context advantage **0.053022**.
Count-fitted `k=20` reaches **2.649575**, with context advantage only **0.007685**.
From-initialization k18/k20 remain worse than bigram at **2.968443/3.240623**.
The positive result is **count-initialized learning on reused development data**,
not from-scratch pretraining or a three-seed/hardware gate. The completed
[evidence record](../evidence/fpga_lm_efsgd_2026-09-09.json) preserves all cells,
full optimizer states and exposure accounting. The original declaration follows.

The completed scale and representability controls motivate a separately
versioned optimizer; no existing scientific/native source or FPGA image is
modified. Thresholded SignSGD discards gradient magnitude. For a single
conditional binary target with fixed surrogate mass `A`, `G=A(q-Y)`, and
`threshold/A < q < 1-threshold/A`, its expected thresholded sign is `1-2p`,
not a quantity vanishing at `q=p`. This toy conditional argument motivates a
test; it does not prove that it is the sole cause in the evolving routed model.

First, run a **fixed-gradient diagnostic**, not training: bind the completed
capacity checkpoint and recompute its embedding/shared-head/bias gradients on
all1,024 original training windows. Compare threshold2048 sign drift with mean
gradient, grouped by context occurrence, and fixed error-feedback shifts16/18/20.
Keep the fitted weights fixed throughout; virtual updates never feed back into
the gradient. Check complete parameter-gradient equality against the frozen
observer on training windows0/1/2. With zero initial remainders, verify the exact
identity `sum(G)=2^k*sum(step)+r_final`, not an asserted equality between the
sum of rounded updates and the rounded total. Do not select `k` using validation.

Then run **four exploratory native CPU cells**: original anchored seed0 or
exact count-fitted initialization, each with `k=18` or `k=20`, always at
`zshift=2`, D512 and the unchanged forward/LUT/backward. Train on windows0–1023
once in order; evaluate full checkpoints0/256/1024 on the same development
validation40–55 cohort. This reused cohort is not a fresh holdout, and one seed
cannot pass the three-seed paid-learning gate. Keep all cells, including failures.

The replacement update is `a=r+G; step=trunc_toward_zero(a/2^k);
w=w-step; r=a-step*2^k`. It preserves accumulated gradient magnitude up to the
explicit remainder, without promising convergence. The two denominators are a
training-unit bracket: one target occurrence contributes about32,768–65,536
units, giving fractional weight motion0.125–0.25 or0.03125–0.0625 respectively.
These are not bounds for arbitrary deep path-merged gradients. Keep nonrecovering
undefined-behavior checks, widened update preflight and failure on any observed
forward or weight-update clipping. A failed native process cannot export a
successful state. No momentum, threshold, loss or geometry sweep is added.

Checkpoints must include allweights, gradient remainders, update count and
optimizer/loss/source identity. Forward-only evaluation must leave all of that
state byte-identical; verify import/resume against uninterrupted training in
native tests, and predictions, loss gradients and route arrays against the
independent frozen observer. Bound each cell to1,200seconds; a timeout is a
retained failed cell, not permission for an adaptive extension.

Count fitting consumes131,072 supervised pairs **before** update0 in that arm;
its gradient pass then replays those pairs. Report262,144 exposures but131,072
distinct pairs at final, versus131,072 for the random arm. At checkpoint256,
fit the matched replay baseline on all1,024 fitting windows plus the replayed
prefix256—not two copies of only the prefix. Report both alpha1 one-pass and
actual-exposure unigram/bigram controls, using the stronger final bigram as
the conservative comparator. Host fitting is not FPGA training.

A favorable exploratory result must be confirmed across seeds/new development
cohort and pass the unchanged quality/context gates before paid scaling.
Although resident gradient arrays can hold remainders without another permanent
on-chip parameter array, the current deployed image clears them on every call.
This repair requires a **new optimizer/state ABI, synthesis and FPGA image**,
full-state physical checkpoint equivalence, and measured resource/timing checks.
The sanitized CPU adapter is not a hardware implementation or speed benchmark.

### Shuffled-order optimizer confirmation v1 (completed)

All three declared order replicates completed and were audited on development
windows 56–71. Final NLL is **2.576359 / 2.620202 / 2.588656**, against the
stronger one-pass bigram **2.644470**: mean advantage **0.049398 nats/token**.
All three pass the mechanism checks, but context benefits are
**0.101865 / 0.015069 / 0.094489**. Seed 1 misses the unchanged **0.02** context
threshold. Therefore **context and combined scaling gates fail**, despite the
positive mean and per-run bigram results. No threshold was changed and no paid
execution followed. See the [complete record](../evidence/fpga_lm_efsgd_confirmation_2026-09-09.json).

The original declaration is retained below. This establishes limited
training-order robustness of the bigram improvement, not a passed all-run
context gate, independent initialization robustness or useful LLM pretraining.

Confirm the successful count-fitted k18 recipe **without changing initialization,
geometry, loss, optimizer, width or budget**. This follow-up is chosen from the
completed exploratory study, not part of its original protocol. The optional
unused-channel initialization ablation is not selected: the actual warm-start
trajectory already learned useful temporal features through those channels.

- Three seeds **0,1,2** select independent `random.Random(seed).shuffle` orders
  of the same training windows0–1023. These are **order seeds, not independent
  weight initializations or data samples**. Store each complete order before
  execution and verify every logged window index against it.
- Every run starts from the exact complete prior `count_fitted_k18` state0:
  all weights, zero remainders, k18, zshift2, LUT and native source identity must
  match byte-for-byte. Require the completed four-cell prior raw run and audited
  public record before scientific execution; preserve and recheck both.
- Train1,024 updates and evaluate complete states0/256/1024 on a **new fixed
  development cohort, validation56–71**,2,048 positions. Leave final test
  untouched. No validation-based scaling, initialization change or extension.
- Each model inherits131,072 supervised pairs of host fitting from its starting
  checkpoint and performs131,072 new gradient-pair exposures. Report262,144
  exposures of **model history**,131,072 distinct pairs, and **zero new model
  count-fitting passes**. Do not relabel inherited fitting as newly executed.
- Fit alpha1 controls on the inherited full1,024-window fit plus the actual
  shuffled prefix at each checkpoint. Report one-pass and replay controls; use
  the lower final NLL for each conservative comparator. Baseline fitting and
  artifact rechecking are verification work, not new model training.
- Retain every cell, full state, native output and failure; use the same
  1,200-second worker limit. Independently recompute native predictions, balanced
  gradients, routes, context probes and exposure controls; require unchanged
  full state through evaluation and correct full-state import.
- Only after all three complete, apply the unchanged mechanism/context gates
  and mean/per-seed bigram thresholds. Keep statistical-significance and paid
  authorization claims false. A positive result establishes limited robustness
  to training order on a new development slice, not broad LLM quality.

This is still CPU-only. The next hardware milestone, conditional on confirmation,
is a newly synthesized optimizer/state interface and exact full-state FPGA/CSIM
equivalence—not simply loading these checkpoints into the existing SignSGD image.

### Matched warm-start SignSGD control v1 (completed)

The declared control completed safely in **654.03 seconds**, within its fixed
1,200-second limit. NLL worsens **2.660464 → 2.741048 → 2.824754**, versus the
matched EF k18 final **2.618411**. Context benefit **0.124364** does not offset
the overall loss degradation. All 1,024 native post-update weight payloads
match the host gradient/update oracle; no clipping or guard failure occurs.
The [audited record](../evidence/fpga_lm_warm_signsgd_2026-09-09.json) supports
retaining the fitted-start EF recipe for the next repair, not generic SignSGD
inferiority or a remainder-only causal claim. The original declaration follows.

The successful exploratory recipe changes both initialization and optimizer.
The existing matched original-anchor comparison actually favors SignSGD over
error feedback, so isolate the missing **fitted-start optimizer comparison**
before committing to an FPGA optimizer redesign. This is a one-cell causal
development control, separate from the completed order confirmation above.

- Import the exact full weights from the completed exploratory
  `count_fitted_k18` state 0. Use the unchanged native kernel, D512/N4/V128/T128,
  zshift 2, SignSGD **lr 1 / threshold 2048**, with cleared per-window gradients.
- Train windows **0–1023 in their original sequential order**, 1,024 updates;
  evaluate full checkpoints **0/256/1024** on original development **40–55**.
  Compare against the completed EF k18 cell with the same initial weights,
  order, cohort and model-history exposure budget. Do not retune either recipe.
- Charge **131,072 inherited host-fitting exposures**, **zero new fitting**,
  and **131,072 new gradient exposures**: 262,144 model-history exposures over
  131,072 distinct supervised pairs. Fit the same alpha1 controls to the full
  inherited fit plus replay prefix; retain the stronger one-pass comparator.
- Declare the complete machine-readable recipe before native building; bind
  the completed prior raw run and audited public record, native build/layout,
  source bytes and corpus. Use new output paths and retain every failure.
- Before each native update, independently compute the frozen host observer's
  full gradient, reject forward clipping and unsafe int32 arithmetic, and
  preflight every SignSGD weight update for int16 clipping. Compare actual
  native predictions, gradients, routes and **every post-update weight**.
- Validation uses the legacy `lr=0` call, which still executes backward; check
  its arithmetic and complete weight invariance. Do not mistake it for the
  forward-only EF evaluation interface. Reject clipped context-removal probes.
- Keep the **1,200-second worker limit**. Recompute checkpoint outputs and
  controls and verify all training-array hashes, route digests, full-weight
  chain receipts and artifact membership after execution. Distinguish runtime
  per-update oracle comparisons from a second posthoc trajectory replay.

No three-seed or paid gate is evaluated. This control compares the complete
fixed optimizers, including scale/deadband, not the remainder mechanism alone.
A healthy learning result may motivate confirmation with the existing FPGA
optimizer; a timeout or guard failure is not a quality verdict. In every case
the failed all-order context gate above remains recorded and does not authorize
hardware scaling. Final test, AWS resources and the manuscript PDF stay untouched.

### Forward-preserving projection diversity v1 (completed)

**Outcome:** all six cells complete and the new arm passes the unchanged
mechanism/context/bigram screen. Its mean NLL is **2.611317** versus stronger
bigram **2.635265**; seed 2 trails bigram by 0.001097, within the existing
0.02 per-seed allowance. All three new context benefits exceed 0.02.
However, historical controls have better NLL in **every paired order**, with
mean **2.579311**; their seed 1 again fails context (0.013775). The intervention
therefore improves consistency of the context diagnostic but worsens predictive
quality by **0.032006 nats/token on average**. Do not select it as the superior
learning recipe merely because it passes a screening conjunction.
The [complete comparison and audit](EVIDENCE.md#forward-preserving-projection-comparison-v1)
retain all outcomes, full states and inherited histories. No final test or
physical FPGA execution followed. The original declaration follows.

The failed order-confirmation context gate has a localized diagnostic signature:
at the final checkpoint, layer 1 retains only **3.93%** of incoming projection
gradient L1 for seed 1, versus **40.75% / 29.87%** for seeds 0 / 2. These are
saved gradient-mass measurements, not inference from route counts. Head
cancellation stays zero and remainders persist; the association does not prove
why that trajectory developed differently. The matched SignSGD control above
does not provide a successful alternative optimizer.

Test **one deterministic, forward-preserving initializer**, without changing
native forward/backward, loss, optimizer, scale, model size or learning budget.
Let anchored coordinates be `A_v=(5v)%512`, and partition the sorted 384 unused
coordinates into three groups `B,C,D` of 128. In every layer, designate
`A_v→B_v`, `B_v→C_v`, `C_v→D_v`, `D_v→B_v`; each projection row has weight 0
at its designated input and −8192 elsewhere. All other weights are unchanged
from the exact count-fitted initial state; optimizer remainders start at zero.

At this fitted point, unused channels have identical values and dominate the
anchored channels through either temporal maximum or minimum. Consequently
the new projection outputs equal the old outputs exactly. The transformation
preserves initial activations, logits and normalized distribution while making
the top three projection scatters injective on their initial backward support.
The bottom can merge gradients; past-time carry, parameter updates, and later
training quality are not guaranteed. Validate the fixed-point preconditions,
margins and native outputs before interpreting learning results.

- **Three new training cells**, order seeds 0/1/2, use the exact saved
  confirmation permutations of windows 0–1023, EF k18, zshift 2 and D512/N4.
  Train 1,024 updates and save complete optimizer states at 0/256/1024.
- **Three historical controls** import the completed original confirmation
  states at those same checkpoints; perform **zero new training**. The failed
  historical gate remains intact. Require the complete audited prior run, not
  a passing prior gate or a selected subset of its seeds.
- Evaluate all six cells on the same **new development windows 72–87**, 2,048
  positions. Compare initial native predictions, logit gradients and head/tap
  routes exactly; projection routes must change to the declared mapping.
  Independently compare every native checkpoint output with the host observer.
- Each new model inherits 131,072 fitting exposures, performs 131,072 gradient
  exposures, and has 262,144 model-history exposures over 131,072 distinct
  pairs. Historical controls inherit both their old fitting and training
  histories but execute neither again. Keep these categories separate and
  report **zero new model fits** throughout.
- Baselines use the inherited full fit plus each actual shuffled replay prefix;
  report one-pass and replay controls and use the stronger final comparator.
  Use the same fixed mechanism/context/bigram thresholds separately per arm,
  only after **all six cells** complete. Report both arms and all seeds.
- Keep the 1,200-second worker limit, exact source/corpus/native-build/state
  bindings, full-state evaluation/import invariance, clipping and arithmetic
  checks, exclusive outputs and retained failures. Preserve all historical
  records and distinguish checkpoint recomputation from full training replay.

This is an initialization and gradient-routing repair, not a changed objective,
pruned FPGA graph, broad optimizer theorem, final-test result, or hardware/cost
claim. No adaptive extension, parameter grid, seed selection or paid launch is
part of this declaration.

### Next quality and implementation gates (not yet run)

**Historical proposal, not the current runnable protocol.** Later diagnostics
below supersede this ordering. Development windows 88–151 have since been used
by the update-scale screen; the still-reserved development cohort is now
**152–863**, and the final test remains untouched. The small fitted-prior GRU
comparison below is complete, but the broader learned-comparator evaluation
proposed here is not. Do not treat the old 88–863 range or these suggested
recipes as a fresh preregistration.

The six-cell result does not justify a larger paid learning run. Keep both
final step-1,024 arms frozen; neither reduced cancellation nor a larger
context-removal penalty substitutes for lower held-out loss.
The owner-requested learning-contract diagnostic below is complete. Its
follow-up recommendation now precedes these broader quality/comparator and
hardware steps; none of those follow-ups has run.

1. **Broaden development evaluation before selecting a recipe.** Declare
   aggregation and decision thresholds before scoring all remaining validation
   windows **88–863** (776 windows, 99,328 positions). Evaluate every order in
   both arms and the same count controls, with unchanged tokenization, resets
   and normalized NLL. Retain per-window paired losses; do not select favorable
   seeds, checkpoints or subranges. Shared evaluation positions and adjacent
   text are not independent samples for an iid-token significance claim.
2. **Add one conventional learned comparator.** A single-layer character GRU
   with 128-dimensional embeddings/hidden state and a 128-way softmax has
   131,968 parameters. Two passes over the same 1,024 training windows give it
   262,144 gradient exposures over 131,072 distinct pairs, versus each routed
   model's inherited 131,072 count-fit plus 131,072 gradient exposures. Report
   its 2,048 updates, different optimizer and lack of host fitting; matching
   model-history exposure counts is not matching compute or update counts.
   Declare its bounded optimizer recipe/selection history using previously
   used development data before inspecting the expanded cohort. An undertrained
   default is not a strong baseline. This is a cheap first comparison, not the
   full optimized CPU/GPU benchmark.
3. **Prototype the stateful accelerator contract locally.** The
   [conditional sparse-path argument](EVIDENCE.md#conditional-sparse-execution-opportunity-not-implemented-or-measured)
   motivates persistent EF state, atomic paged checkpoints, exact multi-window
   parity and checked sparse guards/fallbacks in C-simulation. Preserve the
   existing dense CPU oracle and old FPGA image. Decide the hardware architecture
   from measured guard coverage and state/resource needs, not the quality screen
   alone; a new image and physical full-state parity remain required.

Lock recipe and comparator selection before a final-test evaluation. Any later
change informed by that test needs a new independent holdout. None of these
proposed development steps establishes useful from-scratch LLM pretraining,
authorizes cloud execution or demonstrates a cost advantage.

### Learning-contract diagnostic v1 (completed)

Completed on 2026-09-10 under the declaration below, whose actual pre-run copy
is retained with the immutable run. The [checked results](EVIDENCE.md#learning-contract-diagnostic-v1)
establish FP64 contextual-teacher learnability and exact native gradient/state
parity in six independent D512 probes. Weight quantization suppresses immediate
descent much more than the tested head approximations; error feedback retains
the unapplied motion. These step-zero probes do **not** explain the full
1,024-update validation-quality trade-off or justify a new recipe or paid run.

Run a bounded CPU diagnostic before another recipe repair. Preserve all
historical learning sources, states and gates; no broader validation cohort,
final test, FPGA deployment or cloud operation is part of this study.

1. **Affine and local-learning reference.** Use an independently checked FP64
   implementation of the kernel's excluded causal max/min taps, projection,
   biases, residuals and head, with real mean-centering and exact CE at
   `beta=ln(2)/64`. A deterministic D16/V4/N4/K7/T12 teacher forces the first
   layer's previous-token tap and the other layers' current-token taps, with
   distinct head anchors. Four training and four disjoint evaluation windows
   provide exact soft teacher distributions. Require teacher context KL and
   same-current-token distribution differences above `1e-6`, positive route
   margins, and no adaptive teacher reroll.
2. **Three fixed-budget learning cells.** Compare fixed-route and dynamic-route
   learning from the same nearby teacher perturbation, plus a separate
   structured-cold initialization (not fully random routing). Require the
   nearby teacher/student routes to agree. Each cell gets exactly 64 full-batch
   updates: 192 FP64 updates / 9,216 soft-target position exposures total,
   reusing 48 training positions and 48 synthetic evaluation positions.
   Compute the complete initial affine Jacobian and use the fixed rate `1/L0`,
   where `L0=beta^2*lambda_max(J J^T)/(2*48)`. Check manual/autodiff VJPs,
   directional derivatives within the route cell, affine interpolation and
   common-shift invariance. Fixed affine CE must be nonincreasing. The target
   of halving initial training KL is an observed convergence diagnostic, not
   guaranteed in 64 steps; retain misses. Score final parameters dynamically
   too, so an ineligible frozen-route solution is not called model learning.
3. **D512 numerical ladder at identical points.** For both saved step-zero
   initializations, inspect old training windows 0/37/511. Compare fixed-mass
   exact-softmax, normalized LUT, normalized truncated LUT, and variable-mass
   head gradients with real centering; then change only centering backward to
   the legacy integer floor rule. Use common raw gradient units and apply the
   floor rule before any global rescaling. Compare the legacy full raw gradient to
   the independent frozen observer and six separately reset native EFk18
   updates, including every resulting weight and remainder. These are six
   one-step probes, **not** a learning trajectory.
4. **Separate finite-step effects.** For each direction, use fixed scales
   `1/16,1/4,1` (90 independent candidates). Score ideal weights with frozen
   affine routes/offsets, rounded weights with that same affine map, rounded
   weights with fixed routes/recomputed integer centering, then fully dynamic
   integer forward. Report both exact CE and normalized LUT NLL, changed
   routes/weights and per-region gradient statistics. Only legacy scale1 is
   the actual native EF update; other cells are counterfactual directions,
   not deployable new optimizers. Reject clipping or arithmetic failures.
5. **Stationarity and parameterization controls.** Use the declared nonuniform
   four-class count rows `[40,9,1,0]` and `[3,2,0,1]`, add-one targets and fixed
   raw-unit probe rates `1,64,512`. Exact smoothed-target CE must be stationary;
   unsmoothed replay must match its generally nonzero analytic gradient.
   Compare the named approximate heads at that same point. Separately verify
   the exact duplicated-bias SGD identity and EF conservation on synthetic
   scalar cases; neither is a learned-quality result.

The native probes perform 768 new gradient-pair exposures over 384 probed
pairs and zero new count fits. Each reset inherits the same 131,072-pair fit;
summed reset histories are 786,432 inherited fit exposures plus 768 new ones,
over 131,072 distinct model-history pairs—not six newly executed fits.
Bind source/test/corpus/prior-state/native-build hashes and the actual
declaration; retain every stage and failure in exclusive new output paths.
Use one CPU Torch thread, a 1,800-second execution deadline and checked native
child cleanup. Tests may use synthetic fixtures, not execute this declared
scientific recipe before source freeze. No adaptive extension or repair is
authorized by a diagnostic outcome; review the evidence first.

#### Recommended follow-up after review

The owner approved the bounded diagnostic below on 2026-09-10. Precision,
objective and optimizer repairs remain outside this execution.

First separate delayed integer motion from a trajectory-level quality problem.
Use a short, declared sequence with persistent EF remainders and a matched
FP64 control at the same starting function, data order and gradient units.
Record true CE and LUT NLL, effective step sizes, remainder/weight motion,
route changes and context-disabled loss. Include fixed same-point probes at
later saved checkpoints: step-zero agreement cannot clear centering or
head approximations throughout training. Do not treat a count-fitted state
as an optimum of an unsmoothed or approximate training objective.

If that sequence isolates a precision/step-size problem, test one separately
declared repair, such as additional fractional weight bits or a controlled
step-size/preconditioning change. Keep fixed head mass and duplicated additive
offsets as distinct controls, not simultaneous changes: the former changes
effective rates/example weighting, while the latter changes parameterization.
Report range, state-size and arithmetic consequences before proposing an FPGA
implementation. Do not globally enlarge the learning rate merely because some
first updates round to zero. Only then return to broader development quality
and the conventional learned comparator above; final test remains untouched.

### Persistent trajectory diagnostic v1 (completed)

Completed on 2026-09-10; the actual pre-run declaration is byte-preserved in
the sealed run. All four trajectories and 120 later counterfactuals finish;
every native update and saved-gradient conservation/checkpoint replay passes.
The [results](EVIDENCE.md#persistent-trajectory-diagnostic-v1) do not support
simply increasing weight precision as the next repair. At step32 the unrounded
control is only 0.000528 CE better on the historical arm's three training
probes, but 0.008131 worse on the diverse arm. No own-window update or exact-CE
fresh-only affine probe increases CE. The full held-out quality gap remains
unexplained; these probes were used in the inherited fit, not held out.

Freeze the new helper/runner/test sources and this declaration before any
scientific execution. Preserve the complete learning-contract v1 record and
the original projection-comparison sources, states, order histories and gates.
No broad evaluation, final-test use, objective repair, cloud operation or FPGA
deployment is included. All computation is CPU-only, one Torch thread, with a
1,800-second execution deadline and owned native-child cleanup.

1. **Two matched pairs.** Start each historical/diverse arm from its exact
   seed-zero, step-zero count-fitted state, including zero EF remainders.
   Run native EFk18 and an unrounded-native-rule control for exactly 32
   consecutive old training windows, indices 0–31, without resets between
   updates. Both use dynamic first-tie routing, recomputed floor centering,
   the same shift-2 LUT head/row mass and raw integer legacy backward. The
   control applies `w <- w-G/2^18` to FP64 weights without weight truncation.
   This is **not exact-CE SGD** or a smooth floating-point model: it isolates
   optimizer weight realization while retaining the existing supplied rule.
   No third optimizer, rate grid, seed reroll or adaptive extension is allowed.
2. **Persistent state and feedback.** Check every native update's predictions,
   logit gradients and routes against the new fractional-capable reference;
   compare every resulting weight/remainder to independent EF arithmetic.
   Save every full raw parameter-gradient pair. Verify exactly that
   `z=w-r/2^18` obeys `z_after=z_before-G_native/2^18`, and
   `z_native-w_float=-sum(G_native-G_float)/2^18` across the whole trajectory.
   Report visible/shadow/floating movement, pending motion, per-region gradient
   discrepancies, row-mass ranges, and within/between-trajectory route changes.
   A shadow tracks the native model's own gradients evaluated at visible
   weights; it is not an independently optimized float model.
3. **Fixed checkpoint readouts.** At steps 0, 1, 8 and 32 save full native
   weights/remainders and full FP64 weights. Score visible EF, its reconstructed
   shadow and the unrounded control on fixed training windows 0/37/511, with
   temporal context on and off, recording true CE, normalized LUT NLL and
   accuracy. Require identical starting logits/routes on those windows and
   no checkpoint-evaluation mutation. Windows 37/511 are outside the new32
   gradient sequence but were used by the inherited count fit; **not holdout**.
4. **Later same-point ladder.** For both arms, inspect new native steps 8/32
   and separately the frozen old steps 256/1,024 on windows 0/37/511. The old
   states followed shuffled order and are **not continuations of the new32
   sequence**. Check raw logits/routes/head/full legacy gradients against the
   frozen observer. At each of 24 cases use the five declared head/centering
   directions at scale 1 only, preserving that anchor's actual remainders.
   Save full gradients for all 120 uncommitted counterfactual candidates.
   Score a fresh-only affine candidate `w-G/2^18`, the accumulated ideal
   `z-G/2^18`, rounded weights `w-trunc((r+G)/2^18)` with frozen affine routes,
   fixed routes/recomputed floor centering, and dynamic integer forward.
   Record visible and shadow baselines separately: releasing old remainder
   must not be attributed to the new gradient. The gradient is queried at
   visible `w`, not at `z`; no descent theorem for the shadow candidate follows.
   All candidate parameters are range-checked. Dynamic and fixed-integer
   activations reject clipping; ideal affine internal values are explicitly
   unconstrained, nondeployable mathematical counterfactuals.
5. **Retain and audit.** Reject clipping in the bounded forward paths above,
   nonfinite values, int32 overflow,
   stale tapes, source mutation, broken parity and conservation failures.
   Retain all partial/failure artifacts at exclusive new paths. Publish only
   after all four trajectories and 24 five-direction cases complete. Bind
   original learning-contract/projection records and seals, actual late-state
   bytes, corpus, native build, sources/tests and the actual declaration.
   The read-only check replays optimizer algebra from the saved full gradients
   and verifies all checkpoint payloads/hashes. It does not recompute model
   forward/backward or independently reproduce the scientific trajectories.

Accounting: 64 native plus 64 FP64 updates, 16,384 new gradient-pair exposures
over 4,096 distinct new-gradient pairs. The four shared fitted starts contribute
524,288 inherited fit-history exposures, giving 540,672 summed trajectory-history
exposures over 131,072 distinct historical pairs; zero new count fits.
The later ladder makes 15,360 gradient-pair evaluations, **not training updates**.
Old checkpoint history remains in its original record and was not reexecuted.
No quality threshold or beneficial outcome is promised. Review all results
before choosing a repair or authorizing another experiment.

#### Next decision after the persistent diagnostic

Do not treat deferred updates as lost gradients or assume finer weights will
improve quality. The observed divergence is exactly accounted for by querying
different gradients along the visible-integer and continuous trajectories.
Both improve their current training windows, without consistent improvements
across the fixed probe windows. Interference between windows and sensitivity
to update scaling/parameterization are hypotheses, not identified causes of
the historical validation gap.

The next useful controlled comparison is a separately declared change to
effective update scale **or** averaging across windows, not both at once.
Keep the current objective, initialization and persistent state fixed, match
and report data exposure, and score all fixed windows rather than only the
current update window. If using gradient accumulation, account for its changed
update count and normalization explicitly. Do not select a rate on the final
test or infer a repair from these three fitted training probes. A broader
development comparison and conventional learned baseline are still required
before paid FPGA scaling. The following separately declared screen tests one
update-scale intervention; it does not assume the intervention will help.

#### Paired update-scale screen v1 (declared 2026-09-10, before execution)

Primary question: does changing **only k18 to k20** (fourfold smaller raw-gradient
updates) improve the diverse-projection initializer? Historical initialization
is an interaction/replication control, never a fallback candidate. The earlier
sequential-order count-fitted k20 result was already worse than k18
(2.649575 versus 2.618411 NLL on development40–55) and failed context screening;
the new experiment is not a first, independently chosen test of that rate.

- Fix D512, the native LUT head/zshift2, integer forward/backward, fitted weights,
  zero starting remainders and the exact prior order permutations0/1/2. Train
  six new k20 cells for1,024 windows each. Import the six already-completed k18
  controls at steps0/256/1,024, preserving every state byte. No new count fit.
- Save complete states0/256/1,024, all new native prediction/head-gradient arrays
  and ordered update logs. Select **only final step1,024**; do not select a
  checkpoint or retry a seed. At updates1 and257 of every new cell, independently
  compute the full raw parameter gradient and compare every weight/remainder
  against the new library `fixed_point_ef_step` contract. These extra reference
  gradient evaluations duplicate already-charged training windows, not updates.
- Score both shared initial functions and all12 final endpoints on development
  windows88–151 (64 windows,8,192 positions). Independently compare all native
  predictions, head gradients and routes with the frozen host observer. Save
  compressed full native arrays and per-window NLL/context/mechanism metrics.
  Evaluation and checkpoint import must preserve full optimizer-state bytes.
  Keep the original **untruncated normalized-LUT NLL**; do not substitute r/R.
- Primary promotion screen requires diverse k20 mean paired NLL improvement
  at least0.01 nat and no order-seed regression. Preserve the old all-seed
  mechanism thresholds (zero head-VJP fraction<=.25; >=.75 of positions have
  at least16 head winners; nonzero gradients, no clipping/overflow, exact parity).
  Require context benefit>=.02, growth overinitial>=.02 and unigram advantage
  >=.05 eachseed. Use the stronger of alpha1 one-pass and exposure-matched replay
  bigrams: mean advantage>=.02 and **no seed worse** than the stronger baseline
  (stricter than the prior .02 per-seed allowance).
- If the primary screen fails, retain/report it and leave development152–863
  unopened. If it passes, separately freeze an inference-only confirmation of
  the same paired endpoints on that reserved cohort, requiring the same mean
  paired improvement and guards. No retraining or alternative winner selection.
  Even a confirmed result needs a conventional learned baseline before scaling.
- Freeze source/tests/protocol and bind all inputs before execution; retain any
  failure in an exclusive new directory, with a7,200-second global deadline.
  A read-only check validates source/artifact hashes, complete states/arrays,
  aggregation/baselines/gates and the12 saved full-gradient arithmetic steps.
  Its explicit `--recompute-evaluation` mode repeats all896 checkpoint windows
  using the independent host observer; neither mode replays every training step.

Accounting:6,144 new updates,786,432 new gradient-pair exposures;786,432 inherited
control-gradient exposures;1,572,864 summed inherited fit-history exposures;
3,145,728 summed histories over131,072 distinct training pairs. These are sums
over12 endpoint histories, not claims that the common fit was performed12times.
No final test, AWS, physical FPGA, throughput or statistical significance claim.
The canonical integer EF library primitive is integrated on arithmetic/parity
evidence regardless of whether a smaller learning update improves this workload.
It is experimental and distinct from the existing floating `EFTropicalSGD`.

**Completed outcome:** all twelve final endpoints and the independent replay
passed integrity/numerical checks. Diverse mean NLL changes from 2.671255 (k18)
to 2.668397 (k20), a 0.002858-nat improvement below the declared 0.01 threshold;
two order seeds regress. Its 0.016995-nat mean bigram advantage also misses the
0.02 requirement, although all diverse mechanism/context checks pass.
Historical mean NLL worsens by 0.045470 nat and all k20 context gates fail.
No recipe is promoted. Development152–863 and the final test remain unopened;
no FPGA scaling or new AWS resource is authorized by this result. See the
[full record and audit](EVIDENCE.md#paired-update-scale-screen-v1).

**Originally proposed next diagnostic, now completed below:** use only the already-inspected 32 training
windows to compare sequential k18 with one update per four-window block. Compute
all four full gradients at the block's unchanged weights, sum them, and apply EF
with divisor `2**18`—not `4 * 2**18`. This matches the first-order per-example
update scale and 4,096 gradient-pair exposures per arm, while changing 32 to 8
optimizer updates. Preserve the fitted start, persistent remainders, loss and
window order; check every accumulator prefix, full gradient/state transition,
shared-block CE/LUT change and existing training probe. A benefit would support
changing gradient-query/update timing, not uniquely identify interference,
prove held-out quality or authorize scaling. Freeze that new protocol separately
before execution; its separately declared execution and outcome follow below.

#### Four-window gradient accumulation v1 (declared 2026-09-10, before execution)

The approved next diagnostic uses the diverse fitted start and the same training
windows 0–31, in that fixed consecutive order. Compare sequential EF k18
(32 commits) with four-window accumulation (8 commits). At each batch's fixed
visible weights, compute four complete legacy raw LUT gradients, **sum**, and
apply one update with divisor `2**18`. Preserve remainders between commits;
do not average gradients, change the loss, reset carry, or tune the block size.
Both arms consume 4,096 gradient-pair exposures. The sequential arm is rerun in
the host reference and checked against every saved gradient and state hash of
the completed native persistent trajectory; it is not a new native run.

This changes both gradient-query timing and signed-quantization release timing.
Signed truncation does not make visible weights invariant to grouping: with
`Q=8`, zero initial weight/carry and gradients `(9,-2)`, sequential execution
ends at `(-1,-1)` whereas a single summed update ends at `(0,7)`. Both shadows
equal `-7/8`. Require exact shadow conservation, not identical visible states.
For each accumulation block, also apply its saved four gradients sequentially
from the same block start as an uncommitted arithmetic counterfactual. This
isolates grouping for that supplied gradient list, not for a separate trajectory.

Save all 64 full gradient queries, logits/head arrays/routes, complete weights
and remainders at all 40 commits, and every readout's logits. Match block
boundaries by windows consumed (0,4,...,32). Report shared four-window block
CE/LUT before and after each arm's updates; final mean CE/LUT on all 32 already
used training windows; and context-on/off readouts on the old training probes
0/37/511 at exposures 0,8,32. Derive batch gradient cancellation ratios from
the saved vectors. Check every int32 accumulation prefix, all forward/backward
range guards, public/frozen-optimizer parity, per-commit shadow conservation,
and the paired cumulative-gradient/shadow identity. No clipping or wrapping.

Freeze source/tests and input identities before execution; preserve a failed
run rather than retrying adaptively. Allow 1,800 seconds for this CPU run and
another bounded read-only audit. The audit independently recomputes every full
gradient with the frozen integer observer and every readout with the separate
integer-anchor forward. It verifies saved complete transitions and summaries;
it does not invoke native training or write a second trajectory.

Positive changes in both final training CE and LUT NLL are only exploratory
encouragement for studying this timing intervention. Mixed/nonpositive results
do not establish a repair. Report every block and probe regardless of direction;
do not treat windows as independent training seeds or claim significance.
No recipe promotion, new library default, reserved validation, final test, AWS
resource, physical FPGA execution, throughput claim or scaling authorization.
Accounting: 8,192 new gradient-pair exposures, 4,096 distinct new training pairs,
262,144 summed inherited fit-history exposures and 270,336 summed endpoint
histories over 131,072 distinct fitted pairs. The shared fit is not rerun;
independent replay and fixed-gradient counterfactuals are checks, not updates.

**Completed run outcome:** accumulation fails decisively on this training-only
comparison. Final CE is 5.376293 versus sequential 2.409066; LUT NLL is 5.138609
versus 2.408988, with batch4 worse on all 32 final windows. The first block-loss
ascent occurs on windows 4–7. All full-state arithmetic and sequential/native
control checks pass; independent recomputation also matches all 64 full gradients
and 292 score readouts.
No optimizer recipe is promoted and no reserved data or hardware scaling is
unlocked. See the [record and interpretation](EVIDENCE.md#four-window-gradient-accumulation-v1).
The uncommitted, carry-preserving step-fraction probe at the first failing block
was separately declared and completed below; it is not a trajectory extension.

#### First-failing-batch step fractions v1 (declared 2026-09-10, before execution)

After the completed accumulation failure and its independent audit, the user
approved continuing with a bounded local probe. Freeze `batch4/state_004.npz`,
the four complete gradients already saved for windows 4–7, and the original
state008 as the unit-step control. Do not train another trajectory or compute
new training gradients. Test exactly the uncommitted fresh-gradient fractions
`{0, 1/16, 1/4, 1}`; no adaptive fraction, checkpoint, window or model selection.

Preserve the inherited carry and its units. For `alpha=2**(-j)`, use
`Q'=2**(18+j)`, `A=2**j*r+sum(G)`, `q=trunc(A/Q')`, `w'=w-q` and `r'=A-q*Q'`.
Thus `w'-r'/Q' = w-r/2**18-alpha*sum(G)/2**18`. Check every raw gradient-sum
prefix and every scaled-carry-plus-prefix for int32 overflow, each int16
candidate and canonical output carry, and public/frozen update parity. The zero
fraction preserves the original complete state and k18; it validates the raw
gradient inputs without pretending to apply a nonzero update. Fraction one
must reproduce the full saved state008 weight/remainder arrays exactly.

For each fraction and each of the four windows, report five explicit readouts:
fresh-only affine `w-alpha*G/Q`; carry-preserving ideal affine
`z-alpha*G/Q`, where `z=w-r/Q`; rounded-candidate affine; rounded candidate
with fixed routes and integer floor centering; and dynamic integer candidate.
Affine routes and centering offsets are anchored at the visible state004 for
that window, not silently reselected. Also report visible and shadow dynamic
baselines. This gives 80 candidate readouts plus 8 baseline readouts. Affine
functions are diagnostic surrogates, not the derivative of the discontinuous
integer forward, and are not deployment candidates. All supplied parameters,
including fractional ideals, remain in the int16 value range; affine internal
activations are unconstrained extrapolations. Integer/floor models retain their
full would-clip guards.

Save full gradients, sum, all candidate weights/remainders with their k, and
all 88 logits/traces needed to reconstruct metrics. Bind the complete parent
run/record, actual input bytes and frozen source/tests. Retain any failed run
with exclusive publication and a 1,800-second deadline; separately bound the
read-only replay to 1,800 seconds. That replay recomputes the four original
gradients with the independent integer observer and all readouts; dynamic
integer outputs use a separate integer-anchor implementation. Affine/fractional
readouts reuse frozen diagnostic helpers and must not be described as having
an independently implemented forward oracle.

Report the full CE and normalized-LUT curves relative to each view's own
zero-fraction readout. In particular, the dynamic fractional-shadow baseline is
not the baseline for carry-affine replay, whose routes remain fixed at the
visible anchor. A smaller fraction improving the same block while
the unit update worsens it supports an oversized update at this anchor. It
does not show that all trajectory failures share this cause or establish an
optimizer repair. Differences across affine/floor/dynamic views can localize
effects, not uniquely identify routing, centering, carry release or capacity.
No tested fraction helping does not exclude every smaller fraction. There are
zero new training updates, fits or gradient-pair exposures; recomputing the
four already-charged gradients is 512 audit pair evaluations, not training.
No held-out/validation windows, native/cloud training, new optimizer default,
physical FPGA execution, recipe promotion or scaling authorization.

**Completed run outcome:** the full update worsens CE and LUT NLL on all four
windows in every diagnostic view, whereas 1/16 and 1/4 improve both losses in
all cases. Dynamic CE is 2.595076 at zero, 2.581830 at 1/16, 2.519917 at 1/4
and 3.261983 at one. Fresh-affine CE already worsens from 2.595076 to 3.474871
at the full step, so changing routes or integer rounding is not necessary for
that surrogate's local failure. This supports an oversized update at the
specific first-failing-block anchor; no fraction is committed or promoted and
no global repair is established. The ordinary check and all four-gradient/
88-forward recomputation checks pass. See the
[full curves and scope](EVIDENCE.md#first-failing-batch-step-fractions-v1).

#### Normalized accumulation order diagnostic v1 (declared 2026-09-11, before execution)

The user approved continuing after the local step-fraction probe. Test whether
four-window accumulation at a smaller, explicit scale remains useful across
multiple commits and training orders. This is a bounded **training-mechanism
diagnostic**, not a new development/held-out screen or a from-scratch result.
All nine host trajectories start from the identical diverse seed0 fitted
weights and zero carry. Keep D512, the integer/floor forward, legacy raw LUT
gradient, zshift2, corpus and training windows 0–31 unchanged.

| Arm | Gradients per commit | Fixed denominator | Commits per order |
|---|---:|---:|---:|
| `seq_k18` | 1 | `2**18` | 32 |
| `seq_k20` | 1 | `2**20` | 32 |
| `batch4_k20` | 4, queried at unchanged block-start weights | `2**20` | 8 |

For every commit use `A=r+sum(G)`, `q=trunc(A/2**k)`, `w'=w-q`,
`r'=A-q*2**k`. Sum raw integer gradients without intermediate division.
For batch4, this is equivalent in real units to mean-of-four gradients at
base k18, but **not** to integer-rounding that mean before EF. The denominator
is fixed for an entire trajectory; initialize its carry at zero and persist
the resulting canonical remainder. Do not repeatedly rescale carry using the
previous single-anchor fraction helper. The primary comparison is batch4/k20
versus sequential/k20, at matched nominal per-example scale. Comparison with
sequential/k18 changes both scale and timing. Grouping changes gradient-query
and signed-quantization release timing, not only gradient variance.

Use exactly three orders of the same 32 windows, shared across arms:

- `consecutive`: `[0,1,...,31]`.
- `shuffle_0`: Python `random.Random(0).shuffle(list(range(32)))`, namely
  `[3,7,29,20,19,5,0,10,14,26,17,21,27,23,2,22,4,25,6,18,11,30,9,12,15,16,8,1,13,28,31,24]`.
- `shuffle_1`: Python `random.Random(1).shuffle(list(range(32)))`, namely
  `[26,17,11,10,28,1,5,4,7,16,9,19,30,13,22,0,21,29,6,12,20,23,14,15,3,31,2,24,25,27,18,8]`.

These are order-sensitivity replicates with one shared initialization and data
set, not independent learning trials. Rerun all nine through the same host
implementation. Check every consecutive seq_k18 gradient and before/after
state hash against the frozen native persistent control, including its saved
full-state anchors. No new native training or count fit is performed.

Save all 288 full gradient-query arrays and tapes, all 216 post-commit complete
weight/remainder states and nine initial states. State names use consumed
exposures, not shuffled window IDs. Record query losses, gradient/carry norms
and ordered accumulator bounds. Check every raw-sum and carry-plus-sum prefix
for int32 overflow, every parameter for int16 overflow, all existing
forward/backward clipping guards, public/frozen EF parity and exact per-commit
and cumulative shadow conservation using each trajectory's own denominator.

Score the shared initial weights once on all 32 training windows and on old
probes 0/37/511 with context on/off (38 readouts). For each of nine trajectories,
score all 32 windows at the final state, and the three context-on/off probes at
8 and 32 consumed windows (44 readouts each): **434 scored readouts**, in
addition to the 288 gradient-query forwards. Retain duplicate probe/endpoint
readouts explicitly. Report every final window's true CE, untruncated normalized
LUT NLL and accuracy, plus context differences; no intermediate-checkpoint or
order selection. Probes 37/511 belong to the inherited count-fit training data,
not holdout. Do not add another before/after block-scoring grid adaptively.

The prespecified **mechanism-broadening gate** requires all of the following:

1. All nine cells complete and every arithmetic, provenance and replay check
   passes.
2. Batch4/k20 improves both final 32-window losses over the shared initial
   weights in every order.
3. Against sequential/k20, batch4 improves each loss by at least 0.01 nat on
   average across orders, with no order-level regression in either loss.
4. Its mean is no worse than sequential/k18 in either loss.

Use a fixed `1e-12`-nat comparison tolerance for inclusive loss thresholds;
improvement over initialization remains strict. Integer arithmetic and all
state/array comparisons are exact, with no numerical tolerance.

Report all outcomes even if this gate fails; do not select an alternate arm,
rate, order or checkpoint. Passing supports considering a separately declared
larger CPU test only. It does not promote a recipe, change an optimizer default,
unlock reserved validation, establish useful LM quality or authorize FPGA
scaling. A benefit only against seq_k20 is a grouping benefit under the smaller
scale, not a demonstrated practical repair against seq_k18. The old batch4/k18
failure remains historical context, not a complete factorial interaction study.

Before execution freeze new sources/tests and this declaration, and bind both
completed accumulation and step-fraction parent records, exact source/input
bytes and their recursive ancestry. Preserve all failures at exclusive paths;
allow at most 3,600 seconds for the single-threaded local run and separately
3,600 seconds for its read-only audit. Publish only a complete checked record.
The ordinary check reconstructs every update and metric from saved arrays;
`--recompute` independently checks all 288 full gradients/query outputs and
434 integer score readouts, without committing another training trajectory.
Require source/input/raw/seal/public bytes unchanged throughout each check.

Accounting: 36,864 new gradient-pair exposures over 4,096 distinct query pairs;
1,179,648 summed inherited fit-history exposures across nine endpoints;
1,216,512 summed endpoint histories over 131,072 distinct fitted pairs.
The common fit was performed once historically, not nine times in this study.
There are zero new fits, native updates, held-out readouts, cloud resources,
physical FPGA runs, throughput claims or statistical-significance claims.
Audit queries duplicate already-charged examples and are not training updates.

**Completed outcome:** all nine runs, ordinary completion checks and independent
replay of all 288 gradients and 434 readouts pass. Mean CE is 2.412051
for sequential k18, 2.432145 for sequential k20 and 2.434009 for batch4 k20.
Normalized batching improves over initialization in every order, but regresses
against sequential k20 in both shuffled orders and is worse than sequential
k18 in all three. The numerical mechanism gate therefore fails three clauses.
No recipe, order or rate is promoted. Fewer optimizer commits are not measured
hardware savings. See the [complete comparison](EVIDENCE.md#normalized-accumulation-order-diagnostic-v1).
The next control concerns relative per-token head-mass weighting at matched
nominal head scale, with common FP64/real-centering semantics, declared below.

### Relative head-weighting diagnostic v1 (declared 2026-09-11)

Test whether the truncated integer head's unequal per-token mass impairs
training, without confounding the comparison with different weight carriers,
backward rules or mean nominal head scale. This is a training-only mechanism
test, not an FPGA-ready optimizer or evidence of closing held-out LM quality.

Freeze two arms, `relative_mass` and `uniform_mass`, each in the same three
32-window orders as the normalized-accumulation study: consecutive, shuffle
seed 0 and shuffle seed 1. These are order-sensitivity checks on identical
examples, not independent initialization replications. Start all six cells
from the original diverse seed-0 fitted weights, in CPU FP64 with no remainder.
Use dynamic first-index routing and floor centering in forward, the same
selected-route real-centering VJP in backward, and `W_next = W - G / 2**18`.
There is no additional division by sequence length, integer rounding or EF.
The backward is a surrogate, not the derivative of the floor forward or LUT
NLL. Keep finite/int16-value-range guards for weights and selected activations.

Let `r = ROM_energy >> s`, `R = sum(r)`, `q = r/R`, `m = R/65536` and `y` be
one-hot targets. Calibrate one Python-FP64 scalar `c = mean(m)` from the
predecessor's 32 saved shared-initial training-logit arrays (4,096 positions),
before any new updates. This is disclosed training-only scale calibration,
not a new count fit or target-based selection. Freeze that c in every cell:

- Relative arm: `(c / mean_window(m)) * (r - R*y)`.
- Uniform arm: `c * 65536 * (q-y)`.

Detach all head quantities; do not differentiate through q, m or the scale.
Both arms have the same average nominal scalar head mass. Their parameter
gradient norms and actual updates need not match; record per-region L1/L2/Linf
gradient and realized-update norms. This isolates relative within-window row
weighting, not changes in average mass between windows. Neither arm is the
old native EF/legacy-centering recipe; comparisons against that recipe are
historical context, not an isolated test of this mechanism.

Perform 192 full gradient queries/updates and save 198 complete weights-only
states. Score 38 shared-initial readouts (32 training windows plus probes
0/37/511 with context on/off); in every cell score the 32 training windows at
the final state and the six probe readouts after 8 and 32 updates. Total:
302 scored readouts, plus 192 gradient-query forwards. No other checkpoints,
orders, rates, windows, held-out data or adaptive extension are included.

The mechanism gate requires all six cells healthy and a separate successful
independent replay; uniform must improve both true CE and original untruncated
normalized E/S LUT NLL over initialization in every order, improve both losses
by at least 0.01 nat on average against relative, and regress in neither loss
in any order. Inclusive comparisons use 1e-12 nat tolerance; improvement over
initialization is strict. Report every cell even on failure. No significance,
recipe promotion, default change, hardware saving or automatic larger study
follows from this gate. A null/negative result favors a separately declared
conventional learned comparison over another nearby-rate search.

Before execution freeze the four new helper/runner/test files and bind all
53 source files, the complete checked normalized parent, its completed CLI
audit receipt, inherited sources/inputs and recursive raw-tree ancestry.
Each public run/check performs the original parent's semantic check once;
an invocation-local copied validation object may avoid repeating that check,
but every later guard re-enumerates and hashes the complete recognized
ancestor closure. No persisted validation token or path/mtime cache is trusted.
Preserve exclusive raw paths, exact 1,092-payload-file membership and failures.
Allow 3,600 seconds for the single-threaded run and separately for its audit.

The ordinary check replays every saved update exactly. Independent recompute
uses separate full-matrix forward selection (same FP64 reduction order) for
all 494 forwards, a NumPy head implementation and autograd reverse on the
fixed-route affine surrogate for all 192 full gradients. Forward logits,
routes, offsets and saved state recurrences must match exactly; independent
FP64 head/reverse comparisons use fixed absolute 1e-9 and relative 1e-11
tolerances, while integer head arrays remain exact. Do not replace fractional
weights with an integer-only observer. Require complete source/input/raw/seal/
public bytes unchanged throughout checking.

Accounting: 24,576 new gradient-pair exposures over 4,096 distinct query pairs;
786,432 summed inherited fit-history exposures and 811,008 summed endpoint
histories over 131,072 distinct fitted pairs. The inherited fit occurred once,
not six times here. Zero new fits, native training, held-out readouts, cloud
resources or physical FPGA execution; audit replay adds no training updates.

**Completed outcome:** all six cells, ordinary completion checks and independent
replay of 192 gradients and 302 scored readouts pass. Uniform weighting improves
mean CE/LUT NLL by 0.016905/0.016593 nat, but regresses in consecutive order;
the no-order-regression clause and overall numerical gate fail. No recipe is
promoted. The next comparison below was separately declared before this audit
completed. See the [full result](EVIDENCE.md#relative-head-weighting-diagnostic-v1).

### Exact-CE fitted-start topology comparison v1 (declared 2026-09-11)

This is the next bounded training-only diagnostic, motivated by the user's
request to close the quality gap. Do not infer its result from the preceding
head-weighting comparison. It jointly removes native numeric/optimizer
restrictions to establish a useful learning reference, and separately tests
one function-preserving change to the fitted tropical model's connectivity.
Freeze the sources/tests before executing it, after the head-weighting study
and its independent audit have completed, regardless of that study's gate.

The initializer fixes vocabulary head v to channel A_v and each projection
row to one of the unused B/C/D channels, with all other candidates suppressed
by 8,192 raw units. Saved routing evidence supports persistent use of those
designated connections, not a mathematical impossibility of changing them.
The old explicit residual/projection graph has at most four channels per
head, sharing one class index. Real mean centering supplies additional dense
coupling, especially when different channels select different temporal taps;
do not call the full model class-independent or its total capacity four.

Compare three fixed arms, with one initialization each and no sweep:

1. Original diverse fitted tropical model, retaining its old connectivity.
2. Cross-class tropical connectivity, changing only the designated projection
   input for each row. Fix a seed-0 permutation sigma of 0..127 and set
   `p_l(v) = sigma^-1((sigma(v) + 2**l) mod 128)` for layer l=0..3.
   Replace A_v→B_v, B_v→C_v, C_v→D_v, D_v→B_v with the same bank transitions
   at class index p_l(v); designated weights stay zero and others −8192.
   At the supported fitted start, unused-bank ties preserve the initial
   function. The explicit selected/residual ancestry has 16 distinct class/
   channel ancestors per head, excluding the dense mean path, versus four
   before. This is a wiring certificate, not a performance/rank theorem;
   it shares newly learned unused features, not inherited anchor features.
3. A one-layer GRU with embedding and hidden dimensions 128 and a 128-way
   output correction, initialized with fixed seed 0. Add that correction to
   the same frozen fitted current-token log probabilities. Initialize the
   output correction's weights/bias to zero, so initial probabilities match;
   this intentionally gives zero GRU-body gradients on the first update.

All three inherit the original 131,072-pair fit history. The GRU has 131,968
trainable parameters; each tropical arm stores 1,196,032 parameters, many
initially inactive. This is not parameter-, compute- or hardware-cost matching.
Verify identical starting probabilities within absolute/relative 1e-12, not
identical raw logits or parameters. The frozen GRU prior is
`log_softmax(beta*(EMBED[token,A_v] + W_HEAD[v,A_v]))`, beta=ln(2)/64.

Use only existing training windows 0–3, reset state at each 128-token boundary,
and train each arm for exactly 128 full-batch updates on all four windows.
Optimize mean exact cross-entropy over all 512 positions. Tropical parameters
are dimensionless `u=beta*W`, with real mean centering, dynamic first-index
max/min routing and its selected derivative; use no floor, LUT, EF or rounding.
Positive homogeneity gives `logits(u)=beta*logits(W)` for this real-centering
model. Do not apply beta again to dimensionless logits or transplant the
native raw-gradient rate. Use eager CPU FP64 Adam, lr=0.001, betas=(0.9,0.999),
eps=1e-8, zero weight decay, no foreach/fused/compiled path or gradient clipping.
The prior eps=1e-4 result addressed a different float32 compiled-parity question.
Reject nonfinite parameters, gradients, moments or metrics; retain failures
without retry, rate selection or adaptive extension.

Score all four training windows initially with context on, then at update 128
with context on and off: 36 window readouts across three arms, in addition to
1,536 training-query windows. Context off means one causal tap for tropical
layers and resetting GRU hidden state at every position. The GRU can access
127 preceding positions, versus 90 for the four-layer K7 tropical model;
window boundaries alone do not equalize receptive fields.

Compute the unsmoothed empirical current-token optimum H(Y|X_current) on these
same 512 positions. Each arm's local fitting gate is final mean CE at least
0.02 nat below that floor (inclusive tolerance 1e-12), with all numerical
checks passing. Separately report whether mixed topology beats original by
0.02 nat, without promoting a topology by fiat. Report all endpoints, complete
query learning curves, accuracy, gradient/update statistics, context-off CE
and initial probability agreement; do not select an intermediate checkpoint.
Beating the floor proves use of information unavailable to a current-token-only
predictor on this set, possibly including position. It does not prove useful
linguistic context or held-out generalization. A GRU-only success establishes
a practical fitting gap under this budget, not an impossibility theorem for
tropical architectures. Comparing either tropical arm with old native runs
jointly changes precision, objective, centering and optimizer, not CE alone.

Save all 384 full-batch gradients and query outputs, all 387 complete parameter/
Adam states (including first and second moments and counters), all final
readouts and exact input/source identities. Ordinary checking reconstructs
every Adam update from saved gradients and compares complete saved states
exactly. Independent recompute reconstructs all 1,024 tropical window forwards
and VJPs with a separate real-centering selector and affine-autograd reverse,
comparing their 256 summed full-batch gradients; it also checks all 128 GRU
batch forwards/gradients using an explicit cell implementation
instead of the producer's built-in GRU. Recheck all 36 score readouts too.
Independent FP64 forwards/gradients use absolute 1e-9 and relative 1e-11
tolerances, integer routes exactly; the producer's saved-state replay remains
exact. These are independently implemented numerical checks, not independent
training replications. Bind the completed predecessor and audit receipt with
fresh semantic checking and unchanged complete ancestry bytes. Allow 3,600
seconds each for the single-threaded run and read-only audit, at exclusive
append-only output paths; publish only complete checked results. Retain exactly
1,203 payload files and bind 59 source/test files. Reuse the predecessor's full
semantic check body under the new run's timer, rather than invoking its public
wrapper that would install and then cancel a nested timer.

Accounting: 65,536 new gradient-pair exposures per model, 196,608 total, over
512 distinct query positions. Summed inherited fit histories are 393,216;
summed endpoint histories 589,824 over 131,072 distinct fitted pairs. The
empirical training entropy is a descriptive count statistic, not a new model
fit. There are no new fitting data, held-out readouts, from-scratch pretraining
claims, native/FPGA calls, cloud resources, default changes, statistical
significance claims or automatic hardware/quality promotion. A promising
training result must next survive a separately declared development comparison
and an implementable low-arithmetic numeric contract before FPGA scaling.

**Completed outcome:** all three trajectories and their separate independent
audit pass. Final CE is 1.532157 for original tropical connectivity, 1.426564
for cross-class connectivity and 0.172225 for the fitted-prior GRU. Every arm
beats the empirical current-token floor; cross-class mixing helps by 0.105593
nat, but leaves a 1.254339-nat GRU gap. Neither tropical curve has plateaued.
All 384 full-batch gradients, 387 complete Adam states and 36 scored readouts
are verified. See the [results and limitations](EVIDENCE.md#exact-ce-fitted-start-topology-comparison-v1).
The following rate-only diagnostic was declared before this audit completed.

### Exact-CE rate-only comparison v1 (declared 2026-09-11)

The completed producer of the preceding exact-CE comparison gives final
training CE 1.4265641780767353 for mixed tropical routing and
0.17222484800532445 for the fitted-prior GRU. Its independent audit must finish
successfully before this follow-up starts. The tropical curve is still falling
at its imposed 128-update endpoint; no fitting plateau or capacity ceiling has
been established. Saved-tape inspection finds changing temporal routes but
unchanged designated head/projection supports. That does not identify which
restriction causes the remaining gap. This follow-up changes **one learning
rate**, not the routing, optimizer family, training data or precision.

Run exactly one new mixed-topology arm from its original fitted initialization,
using Adam **lr=0.01** instead of 0.001. Keep all other settings of the preceding
study: dimensionless u=beta*W, real mean centering, exact mean CE, dynamic hard
first-index max/min selection, CPU FP64, betas=(0.9,0.999), eps=1e-8, no decay,
clipping, foreach, fused implementation or rounding. Use the same complete
training windows 0–3, reset state at window boundaries, and perform exactly
128 full-batch updates on their 512 positions. Start with empty Adam state,
not the preceding trained weights or moments. Verify exact equality of all
initial mixed parameters and identical initial probabilities against the
frozen predecessor. The old mixed lr=0.001 and prior-GRU endpoints are
historical controls from this very same data/update budget, not new replications
or a matched compute-cost comparison.

Primary diagnostic gate: final CE must close at least half of the existing
mixed-versus-GRU gap, namely
`CE_new <= (CE_old_mixed + CE_old_GRU)/2`, using the full-precision bound
historical values (approximately **0.79939451304103**). Separately report
near-GRU local fitting only if `CE_new <= CE_old_GRU + 0.05` (approximately
**0.22222484800532445**). Use inclusive tolerance 1e-12 and require all numerical
checks and the separate independent audit. Report both gates, the fraction of
the historical gap closed without clipping, and every final window's CE and
accuracy. Neither gate promotes an optimizer or establishes held-out quality.
The faster rate is a new training-data-informed diagnostic choice, not an
independently selected hyperparameter or an equal tuning-budget benchmark.

Retain every query learning-curve point and every complete state. Score the
four windows initially with context on and finally with context on and off,
for 12 score-window readouts. Ordinary checking must replay all 128 eager Adam
transitions exactly from saved gradients and compare all 129 parameter/moment/
counter states, including actual counter dtypes and complete parameter-group
options. The separate independent audit must reconstruct all 512 queried
tropical window forwards and selected VJPs, compare their 128 summed gradients,
and independently reproduce all 12 score-window readouts. Use the same FP64
tolerances (absolute 1e-9, relative 1e-11), exact integer route comparisons and
exact saved-gradient state replay as the predecessor. Record saved support
changes descriptively; unchanged support would qualify, not invalidate, a
successful fit. Do not infer a universal fixed-support bound from this run.

Freeze the new runner and tests with the 59 unchanged predecessor source/test
files (61 total). Retain exactly 403 payload files: 258 full state/checkpoint
metadata files, 128 queries, 12 scores, one cell summary and four root metadata
files, plus the separate inventory seal. Bind the completed exact-fit run, its actual audit receipt,
public record, initial state, corpus/split and complete ancestor closure.
Revalidate the predecessor semantically under one outer deadline, without
calling a nested timer-owning wrapper. Save this declaration before the first
new model readout. Use exclusive append-only raw/public paths, retain failures,
and publish only complete checked evidence. Allow 3,600 seconds separately
for the single-threaded run and its read-only independent audit. Nonfinite
values, provenance changes or a deadline failure stop the run; do not retry at
another rate, extend its training budget, select an intermediate checkpoint,
or alter a frozen predecessor to make the check pass.

Accounting: 65,536 new gradient-pair exposures over the same 512 distinct query
positions, plus one inherited 131,072-pair fitting history, for 196,608 exposures
in this new endpoint's history and 131,072 distinct fitted pairs. Historical
control costs remain historical and are reported separately, not called free
training. Replaying saved points adds no optimizer trajectory or new fitting
exposures. No new corpus fits, held-out readouts, native/FPGA calls, AWS resources
or library/default changes are authorized by this diagnostic. A positive result
would implicate the earlier rate/budget in this local fitting gap; a negative
result would reject this one faster rate, not prove architectural incapacity.
Development generalization and a low-arithmetic implementation remain separate
requirements before hardware scaling.

**Producer outcome:** final CE 0.124361 and 100% accuracy, versus historical
GRU 0.172225/97.27%, with identical initial mixed state and unchanged selected
head/projection supports. Both numerical thresholds pass. The first audit and
the single legacy recovery below reach their elapsed-time limits; the result
remained provisional at that handoff. **Numerical verification is subsequently
completed under the separately declared pinned-predecessor protocol below**;
no recipe is promoted. See the
[results and distinct audit receipts](EVIDENCE.md#exact-ce-rate-only-comparison-v1).

### Audit-only recovery (declared 2026-09-12)

The first rate-only independent audit was interrupted at its absolute elapsed-
time deadline after reporting successful replay through update 64. Its failed
receipt is retained separately; it is not a completed numerical audit. The
producer record, source files, parameters, gradients and thresholds remain
unchanged. Permit **one fresh read-only audit attempt**, with the exact same
CLI checks, all ancestor validation and all 128 gradient/state comparisons.
Use the same 3,600-second limit, a process-scoped keep-awake wrapper and an
absolute UTC-based watchdog for the owned process group. This is not a training
retry or an extension of the failed audit's deadline. Record actual start/end,
exit status and both attempts; do not promote a recipe unless the fresh full
audit succeeds within its own bound. No further retry is included here.

**Recovery outcome:** unsuccessful. The owned process-group watchdog interrupts
the recovery after 3,600.198591 seconds, during predecessor score-file validation
and before this attempt reaches the new-rate gradients. The wrapper exits 1,
its child exits −2, and no numerical discrepancy is reported. This is neither
a pass nor evidence of a numerical mismatch. Both failed attempts are retained;
no third retry, additional training or hardware launch follows automatically.
The completed producer and all source-bound evidence remain unchanged. Before
resuming, address reliable execution and the cost of repeated ancestor checking
under a separately reviewed audit protocol; do not waive checks or edit the
historical receipts to manufacture success.

### Pinned-predecessor rate audit v1 (declared 2026-09-13)

The user's continuation starts a separately reviewed **audit-only protocol**;
it is not another invocation under the exhausted recovery declaration above.
The two failed legacy receipts remain immutable. A fresh integrity-only check
of the original 61 source files, 403 rate payloads, ten ancestor trees and 163
external identities takes 5.12 seconds; it performs no model evaluations.
The replacement therefore separates byte verification from repeated numerical
reconstruction of already successfully audited predecessor trajectories.

Trust is narrowly anchored to the actual completed exact-fit audit, not a
caller-supplied `complete` flag or a freshly resealed collection. The new tool
hard-codes these SHA-256 values without a command-line override:

| Immutable record | SHA-256 |
| --- | --- |
| Rate public | `e393750a7191c9a64c1a4f3cb89150b6355ba61838f551ba7361e42c7ff1859b` |
| Rate manifest | `15e0e7dbf963ea62354101a77737cf040859ef80e9670f690918ddf486b8a10a` |
| Rate payload seal | `c00578a494d38a856539ece1dc7e9b917a830729a954d18fb9f82658672d277f` |
| Completed exact-fit audit | `8506cf77b68e2cb0a6e02aa43c441119f86aa226ebb46d7b1648444745bb92a0` |
| Exact-fit public | `6c36b6145626b610b3cc46645dfb69e27b16b3f6162531c8fab5e3a852c1c74e` |
| Exact-fit manifest | `eefa32acef07640554c23f3ab0b8ca6de91d61bafd3e37313658df3f529de12b` |
| Exact-fit payload seal | `35c18b1016ab485ef19756866e628580a95e47f57f8931bdeb03123bacf4debf` |

Before any rate-model reconstruction, freshly verify every actual source,
corpus, raw artifact, ancestor and external identity, including exact inventory
membership. Check the pinned successful audit's identities, numerical coverage,
timing and limitations. Reconstruct the entire rate binding from the pinned
parent manifest, canonical corpus/split, original mixed full state, four initial
score identities, historical controls and freshly verified ancestry closure;
require exact equality with the pinned rate manifest. Only then create the
private validation token accepted by the unchanged numerical checker. Reusing
the predecessor's completed semantic/numerical verification is justified by
this full immutability check; hashing alone is not numerical verification.

Independently recompute **all 512 rate window VJPs and their 128 batch-gradient
sums, all 12 score-window readouts, and all 129 complete Adam states**. Compare
independent gradients with saved gradients at the unchanged tolerances, then
apply the **saved** gradients for exact full-state Adam replay. Preserve every
parameter, moment, step counter and optimizer option. This invocation does not
numerically rerun predecessor trajectories; its receipt must say so explicitly.
Partial query progress is never a completed audit.

Use new files `manuscript/tools/audit_fpga_lm_exact_rate.py` and
`tests/test_fpga_lm_exact_rate_audit.py`; freeze them with the unchanged 61
producer source/test files (63 total) after independent review and synthetic
full-trajectory/adversarial tests. Do not modify the frozen producer, its
checker, original tests, raw payloads or public records. Save this declaration
before any real numerical audit. Permit one invocation with a **3,600-second
UTC and monotonic deadline**, covering input validation, every numerical check,
post-check immutability verification and success publication. Start process-
scoped sleep prevention with the invocation. Record actual program-generated
start/end timestamps and elapsed times; deadline expiry or any failed check
retains an exclusive failed-attempt record, never a success receipt. Use a new
exclusive attempt directory and a new exclusive public receipt path; never
overwrite either historical attempt or the producer evidence.

There are zero new fitting exposures, training trajectories, held-out readouts,
native/FPGA calls or paid resources. A complete audit would verify the reported
**local fitting** result only. Broader development quality, low-arithmetic
implementation fidelity and any hardware efficiency claim remain separate
gates. Any further invocation or segmented protocol requires a new declaration;
this declaration does not automatically authorize a retry or broader training.

**Pre-invocation freeze:** all 28 new audit tests pass, including a genuine
synthetic 128-update trajectory with 512 independent VJPs, 524 independent
forwards, every complete Adam state and an explicit Adam recurrence check.
Altered or resealed evidence, source drift, deadline failures and publication
collisions are rejected. The original rate/guard suites have 83 passes;
Ruff and whitespace checks pass. Freeze the following new source identities
with the original 61 unchanged files:

- Auditor SHA-256: `64f29f201cb857b7402d9b5926ca104bcc1a1e364c1d4c0afd542d0bcbf9d7b4`.
- Tests SHA-256: `8a68d31bf1e16daab854050ab3eee42b15f754fabd8d1307613c141580647034`.

Use exclusive private directory
`manuscript/build/fpga_lm_exact_rate_pinned_audit_2026-09-13/v1`
and public receipt
`manuscript/evidence/fpga_lm_exact_rate_pinned_audit_2026-09-13.json`.
The receipt reports verification completion before publication; retain the
program's terminal completion timestamp and both elapsed clocks separately
from the later human observation of process exit.

**Outcome:** successful. The invocation starts at 20:16:52 UTC and completes
at 20:27:52 UTC on September 13, exit 0, with terminal elapsed times
659.985885 UTC / 659.990614 monotonic seconds. All 512 independent window VJPs,
128 batch gradients, 12 scores and 129 complete states pass; all source/input/
raw/public bytes remain unchanged. Receipt SHA-256:
`56d4388eb440ee77e7416e7fe60592cdf594c7b3cbf066dbc03bc2ff079a940b`.
The receipt explicitly records pinned completed predecessor reuse, not another
numerical replay of historical trajectories. Both required local-fitting gates
now pass with their audit. This does not authorize a broader study or paid
launch; the two legacy failures remain preserved as failures.

### Guarded-forward full-trajectory parity v1 (declared 2026-09-13)

This implementation-only replay is conditional on successful completion of the
pinned-predecessor rate audit above. Before execution, hard-pin its actual
successful receipt and freeze the new verifier and tests together with their
complete source closure. Do not change either the original dense producer or
its audited trajectory, or the separately tested guarded-forward prototype.
The latter's sufficient inequality certifies a selected winner at each operation;
unsupported structures or uncertified operations must take its original dense
fallback. Equal bounds never certify a winner, and no route is forcibly fixed.

From the same original fitted mixed state and empty Adam state, recompute the
four query-window forwards and unchanged manual VJPs at **every one of the 128
updates**. Require byte-exact logits, route tapes, centering offsets and summed
gradients against the independently audited dense payloads. Apply the recomputed
gradients only after their exact agreement, then compare **all 129 complete
parameter/moment/counter states and optimizer options**. Preserve summation order,
full-dimensional centering and its reverse terms. Check all four initial, four
final context-on and four final context-off scores against the same saved dense
readouts: **524 guarded forward calls, 512 VJPs and 128 summed gradients** in all.
A mismatch fails the check; do not relax equality or reset a mismatching state.

This is implementation parity against an independently audited trajectory,
**not another independent-autograd oracle**. Record every operation's fast-path
or dense-fallback count, all step-level agreement results and complete coverage.
The guard may fall back without invalidating correctness; any performance
interpretation must report how often it did. Time the guarded computation and
saved-payload checking separately where practical, with actual CPU and elapsed
time and explicit timing boundaries. Without a directly matched dense timing
experiment, these measurements establish replay feasibility only, not a speedup,
energy saving or hardware advantage. Keep full parameters and full Adam states;
do not claim optimizer-state compression or arithmetic closure for FP64 Adam.

Use new verifier `manuscript/tools/verify_fpga_lm_guarded_rate.py` and new tests
`tests/test_fpga_lm_guarded_rate.py`. The source closure is the rate audit's 63
files plus the unchanged guarded helper/tests and new verifier/tests (67 total).
Require complete raw/source/input/ancestor immutability checks before and after
replay, strict successful-audit binding, synthetic full-trajectory and tamper
tests, and independent code review before the real invocation. Save this
declaration and the frozen identities before the first guarded model readout.
Use a new exclusive attempt directory and public receipt, retain failures and
do not duplicate the full historical checkpoints. Permit **one invocation with
a 3,600-second UTC/monotonic deadline**, covering validation, numerical work,
post-checks and publication, with process-scoped sleep prevention from launch.
There is no automatic retry or segmented continuation if it fails.

Replaying the same audited trajectory adds **zero new fitting exposures or
distinct optimizer trajectories**; it recomputes known operations for parity.
There are no new corpus positions, development/test readouts, rates, updates,
count fits, native/FPGA calls or paid resources. A successful result would make
the guarded implementation eligible for a separately frozen broader development
study, not establish generalization or promote the FP64 recipe to the library.

**Pre-invocation freeze:** the independent rate audit has completed successfully;
its receipt SHA-256
`56d4388eb440ee77e7416e7fe60592cdf594c7b3cbf066dbc03bc2ff079a940b`
is hard-pinned in the new verifier. All 35 new synthetic tests pass, including
the complete audited-dense-to-guarded 128-update replay, both dense-fallback
modes, actual use of recomputed gradients, signed-zero/state/option corruption,
audit-capsule integrity and publication failure cases. Independent code review,
Ruff and whitespace checks pass. Freeze all 67 source/test identities, with
the following new files:

- Verifier SHA-256: `24e99d7247e13444f707f76fe142130973b80623fb0c12fc17cbe1f295a7bdf4`.
- Tests SHA-256: `14b8c46e556a7e4280ca3bdbe2ad8503f71796209707a2e1144db271be23e05a`.

Use exclusive private directory
`manuscript/build/fpga_lm_guarded_rate_parity_2026-09-13/v1`
and public receipt
`manuscript/evidence/fpga_lm_guarded_rate_parity_2026-09-13.json`.
Do not launch until the current unrelated regression run has finished; the
recorded timings describe this local replay, not a controlled system benchmark.

**Outcome:** successful. After the 4,428-case regression invocation finishes,
the guarded replay starts at 20:33:28 UTC and completes at 20:35:22 UTC,
exit 0, in 114.021997 UTC / 114.022813 monotonic seconds. All 524 traces,
128 batch gradients and 129 complete optimizer states are byte-exact; all
512 VJPs are recomputed and their identical gradients applied. Every covered
projection/head operation certifies (268,288 / 67,072 respectively), with zero
dense-operation or whole-model fallbacks. Source/input/raw/audit bytes remain
unchanged. Receipt SHA-256:
`7a21df2543ece923fbdcd388ae326770db0aa0ad6623d1ff578c505ae4ae5a82`.
The 99.372646-second numeric replay and its timing buckets are feasibility
observations, not matched speedup measurements. No new training, data readouts
or hardware execution occurs; the broader comparison remains unrun.

### Next development comparison design (not yet frozen or run)

The audited rate-only experiment's 0.124361 CE / 100% accuracy closes the small
repeated-training-set gap to the fitted-prior GRU.
It does not answer the next question: does the recipe learn useful patterns
from varied text, or merely memorize those 512 positions? Prioritize that
question over another optimizer-family change or an FPGA launch.

The preceding guarded FP64 replay now verifies the complete existing dense
trajectory, including every gradient and Adam state. Its 114-second completion
establishes local replay feasibility, not a runtime guarantee on new windows
or a matched speedup. Keep every existing study dense and immutable. The new
study must retain exact fallback, full mean-gradient terms and optimizer state,
and its own bound independent numerical checks; it may not silently force the
old support or assume that every new-data guard will certify.

Proposed single development screen, to freeze with runtime/storage bounds and
audited sources before any new training or scoring:

- Mixed tropical Adam lr=0.01 versus fitted-prior GRU Adam lr=0.001. Reset both
  to their bound fitted initialization, not the endpoints fitted on 512 pairs.
  Preserve exact CE, real centering, tokenization and per-window resets.
- One pass over all 1,024 training windows, batches of four: 256 updates and
  131,072 new gradient-pair exposures per model. Use three shared training-order
  seeds 0/1/2, with model initialization seed fixed; these are order replicates,
  not independent corpora or architecture-initialization replications.
- Six cells and 1,536 total updates; no rate sweep, extra epoch, early-checkpoint
  selection or result-dependent extension. Establish feasible run/audit/storage
  bounds before freezing this replication count and before reading new scores.
  Execute paired orders sequentially for resource control, but complete/report
  all six regardless of early losses unless a predeclared numerical, integrity,
  deadline or storage failure stops the experiment. Preserve the same 512-position
  mean-CE normalization; expanding the corpus is not a reason to rescale either rate.
- Score only already-consumed development windows **88–151**, initially and
  finally, with final context-disabled readouts too. These 8,192 positions are
  development data, not a fresh confirmation set. Keep **152–863 and the final
  test untouched**. Retain complete per-window paired metrics and report every
  order; do not treat adjacent text positions as iid samples.
- Report the fitted initial prior and fixed exact alpha=1 bigram, plus the
  repeated-count alpha=1 comparator (equivalent to one-pass alpha=0.5), without
  a new fit on development data. Require tropical improvement over its initial
  prior in every order, mean CE at least 0.02 nat below the stronger fixed
  bigram, and CE no more than 0.05 nat above GRU in every paired order. Context
  penalties support a mechanism interpretation but do not replace these loss
  comparisons. These are proposed development gates, not a hardware gate.

Before freezing, specify whether each architecture's identical initial
development evaluation is shared across its three order replicates. Sharing
would require 128 initial plus 768 final context-on/off window readouts, **896
total**, without dropping any paired endpoint. Keep full gradients and complete
Adam states with independent numerical recomputation, not just a guarded manual
replay. Tropical parameter/moment/gradient tensors alone would occupy about
**29.5 GB uncompressed across three cells**, before traces, GRU or metadata;
compressed storage may be much smaller but requires an explicit disk cap and
free-space checks. Freeze per-cell and aggregate producer/audit deadlines as
well. The verified 114-second old-trajectory replay does not bound new-data
fallback or full independent-autograd cost.

Each endpoint inherits 131,072 fitted-pair exposures and adds 131,072 gradient
exposures: 262,144 history exposures over 131,072 distinct pairs. Across six
models this is 786,432 new gradient exposures and 786,432 summed inherited
fitting histories, with zero new count fits. The inherited fit was performed
once, not six times during this screen. Report historical control/tuning costs,
the different receptive fields and parameterizations; this is not equal tuning,
compute, parameter or hardware-cost matching. A failed one-pass recipe would
not prove architectural incapacity or convergence. Online training losses visit
different batches, so their progression alone cannot establish overfitting.

Only a successful, audited broader result should motivate a low-arithmetic
implementation comparison and a locked reserved-cohort confirmation. Neither
the local fit nor this proposed screen authorizes an automatic paid launch,
establishes from-scratch LLM pretraining or demonstrates cheaper learning.

### Varied-text exact-CE development comparison v1 (declared 2026-09-13)

The user's continuation now advances the preceding design into a fixed local
development experiment. Both prerequisites are complete: the independent rate
audit and byte-exact guarded trajectory replay. This declaration precedes all
new model training and development readouts. It authorizes no AWS, FPGA, native
training, reserved-cohort evaluation or manuscript-level quality claim.

Run exactly six cells in paired order: mixed tropical then fitted-prior GRU for
training-order seeds 0, 1 and 2. Reset each cell to its original bound fitted
initialization (model seed 0) and empty Adam state, never a trained endpoint.
Require exact initial parameter bytes against the corresponding original
exact-fit checkpoint. Preserve the frozen GRU prior and architecture. Use
tropical Adam **lr=0.01** and GRU Adam **lr=0.001**, CPU FP64 and one Torch
thread, betas=(0.9,0.999), eps=1e-8, no decay, clipping, foreach, fused execution
or rounding. Retain dimensionless tropical units, real full-width centering,
dynamic first-index max/min selection and per-window state resets. Tropical
forward uses the verified guarded helper with its unchanged exact dense fallback;
all fallbacks are allowed and counted, not bypassed or treated as failures.

Each order is one deterministic permutation of training windows **0–1023**.
Use the same permutation in each paired architecture, consecutive batches of
four complete 128-position windows, for **256 updates / 131,072 gradient-pair
exposures per cell**. Every gradient is the selected derivative of exact mean
CE over all **512 positions** in its batch. Preserve original summation order.
Do not rescale learning rates when expanding the corpus, repeat a training
window, add epochs, clip gradients, select a checkpoint or adapt a hyperparameter
from the newly observed losses. Complete and report all six cells regardless
of early quality; only a prespecified numeric/integrity/resource failure may
stop the experiment, in which case it has no complete success record.

Read development windows **88–151 only** (8,192 positions already consumed by
earlier development work). Score context-on once initially for each architecture
and reuse those exact 64-window records across its three identical initial
states. Verify the two architectures' initial probabilities at the original
1e-12 absolute/relative tolerance. Score all 64 windows context-on and context-off
at each cell's final endpoint: **128 shared initial + 768 final = 896 model-window
readouts**. Keep validation **152–863 and the entire final test untouched**.
The records must explicitly say development was evaluated, rather than falsely
claiming no corpus held out from these training updates was scored.

Two fixed bigram comparators add **128 baseline-window readouts** on the same
development slice. Reuse the original alpha-one count table from the completed
capacity experiment. Its entries already include one pseudocount, so the
repeated-count alpha-one table is **2*C−1**, not 2*C; this equals one-pass
alpha=0.5 probabilities. Verify the inherited counts against the same original
training positions by integer reconstruction only. This does not refit a model,
change the table or access development targets for fitting. Retain both baselines
and use the lower mean CE as the fixed stronger baseline for the gate.

Compute all gates only from complete final endpoints: tropical CE is strictly
lower than its own initial CE in **every order** (equality fails); tropical
mean CE is at least **0.02 nat below the stronger bigram**; and tropical CE is
no more than **0.05 nat above its paired GRU in every order**. The two numerical
margin bounds use inclusive tolerance 1e-12; the strict initial-improvement
test does not turn equality into a pass. Report every paired
window's loss/accuracy and context penalty, all order means and spread. Do not
treat adjacent tokens as iid or infer significance from three order replicates.
Context ablation supports interpretation but cannot replace these loss gates.
This tests two selected one-pass recipes, not convergence, equal tuning or
architecture limits; the tropical recipe had an additional rate diagnostic.
Parameter counts, effective structure and receptive fields remain unmatched.

Save every complete query/gradient and every complete parameter/first-moment/
second-moment/step-counter state and optimizer option: **1,536 batch gradients,
1,536 updates and 1,542 full states** across the six cells. The ordinary checker
must replay all saved-gradient Adam transitions exactly. A separate read-only
audit must independently reconstruct every new tropical query using full-matrix
selection and fixed-route autograd, compare every summed gradient, and verify
every GRU query with the explicit-cell implementation. Independently recompute
all 896 model readouts and both exact-count baselines. The tropical audit covers
**3,072 window VJPs / 768 batch-gradient sums**; the GRU audit covers **768 batch
gradients**. Use unchanged FP64 tolerances (atol=1e-9, rtol=1e-11) and exact
integer routes; apply the **saved gradients** for exact full-state audit replay.
Guarded/manual parity is not substituted for this independent audit.
Supplement the independent checks with guarded forward replay at all 3,520
tropical query/score windows to verify the reported shortcut/fallback counters
and exact implementation traces too. These supplemental calls are reported
separately, not counted as independent-autograd checks or new learning. The
independent GRU audit has 768 four-window query calls and 448 single-window score
calls; combined independent model coverage is 7,040 forward-window equivalents.

Bind the successful guarded receipt
`7a21df2543ece923fbdcd388ae326770db0aa0ad6623d1ff578c505ae4ae5a82`
and its actual four-file capsule, all 67 unchanged sources and the entire
original rate/source/input/ancestor closure. Reuse those pinned completed
numerical audits only after full fresh byte verification; do not recursively
rerun their numerical trajectories. Add the capacity count provenance explicitly,
because its raw tree is not a member of that old ten-tree closure. Hard pins:

| Capacity record | SHA-256 |
| --- | --- |
| `counts.json` | `d61f20b3108fe6443a388f27ac6469859219e1965126afc4c6d43d6bba605244` |
| `manifest.json` | `8268966aa461bd5a7ede14b8363c262337034f2f0ceb56c75d872d8ee383a318` |
| `artifact_manifest.json` | `eb6605c83590eeb4019ade6164a2d79a6051004a00b33d9c427a0b80fe3f195b` |
| Public capacity record | `027b012522fafce2309a4bfb5d548ebd1f5e1dd0921f3628dcc3b501f62e9dc9` |

The capacity raw root is `manuscript/build/fpga_lm_bigram_capacity_2026-09-09/v1`;
its public record is `manuscript/evidence/fpga_lm_bigram_capacity_2026-09-09.json`.
Verify its complete sealed inventory and source/dataset binding without
recomputing historical capacity validation scores or native executions.

Use new runner `manuscript/tools/compare_fpga_lm_development.py` and tests
`tests/test_fpga_lm_development.py`, frozen with the unchanged 67-file closure
(**69 source/test files total**). Require synthetic six-cell producer/audit
tests, full-Adam recurrence and tamper/resource/publication checks, plus independent
code review before the real invocation. Save this declaration and exact source
identities before the first new model readout. Use exclusive append-only raw,
public and separate audit paths. Retain partial files and a failure record on
error; never publish incomplete evidence, overwrite an old attempt or delete
checkpoints to fit the budget.

Allow **10,800 seconds globally for the single producer invocation** and
**10,800 seconds separately for its single independent audit**, with a
**3,600-second per-cell bound** inside each outer limit. Both UTC and monotonic
clocks must remain in bounds, including input validation, ordinary checking,
final byte verification and publication. Use one outer timer owner with scoped
cell deadlines, not nested wrappers that cancel one another's timers. Start
process-scoped sleep prevention with each invocation. A timeout ends that
invocation; no automatic retry, audit segmentation or adaptive extension is
authorized by this declaration.

Limit the new experiment's retained raw/public/audit files to **12 GiB** and
keep at least **4 GiB free**. Check capacity/free space before writes, allowing
conservatively for the next complete uncompressed tensor payload and metadata;
also check actual cumulative sizes. A storage failure retains prior artifacts
and stops, rather than discarding gradients or moments. The read-only preflight
finds about 1.85 TiB free on the workspace volume, well above these limits.
The earlier 114-second replay is feasibility evidence only; new-data fallback,
compression and independent-autograd work may cost more. Timings and guard
counts are descriptive, not matched CPU/GPU/FPGA efficiency measurements.

Accounting remains explicit: each endpoint inherits 131,072 fitted-pair
exposures and adds 131,072 gradient exposures, for **262,144 history exposures
over 131,072 distinct pairs**. Across six endpoints there are **786,432 new
gradient exposures**, 786,432 summed inherited fitting histories and 1,572,864
summed endpoint-history exposures. The original count fit happened once;
integer reconstruction and numerical auditing add computation, not newly
selected training data or distinct fitted trajectories. Reserved-cohort
confirmation, a hardware-compatible optimizer and useful-cost benchmarking
remain separately frozen next steps even if all development gates pass.

#### Execution freeze (2026-09-14 UTC, before new readouts)

Independent source review is GO. The complete new test module passes all
110 cases (root invocation: 15.84 seconds), including a genuine synthetic
six-cell producer/check/separate audit and a 256-update full-Adam recurrence.
Ruff and `git diff --check` pass. Freeze these two additions and all 67 inherited
sources without further edits during production or numerical audit:

| Added source | SHA-256 |
| --- | --- |
| `manuscript/tools/compare_fpga_lm_development.py` | `ae324d5ec86c27c25f6115ae5ee1a6899ad88421ccf8a325f3fb2e111af5cfea` |
| `tests/test_fpga_lm_development.py` | `c5373326f8a28f3360a512812a6959c5b21660622e594badc5551bfe82654b84` |

Use the following single producer invocation from the repository root. The
September 14 path records the actual UTC launch date; the scientific protocol
above was declared September 13. All output paths must be absent beforehand.

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/compare_fpga_lm_development.py --run \
  --output manuscript/build/fpga_lm_development_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_development_2026-09-14.json \
  --prior-run manuscript/build/fpga_lm_exact_rate_2026-09-11/v1 \
  --prior-record manuscript/evidence/fpga_lm_exact_rate_2026-09-11.json \
  --guard-receipt manuscript/evidence/fpga_lm_guarded_rate_parity_2026-09-13.json \
  --corpus data/tinyshakespeare.txt
```

Only after successful complete producer publication, run this separate full
numerical audit once, regardless of whether its development quality gates pass:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/compare_fpga_lm_development.py --audit \
  --output manuscript/build/fpga_lm_development_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_development_2026-09-14.json \
  --audit-output manuscript/build/fpga_lm_development_audit_2026-09-14/v1 \
  --audit-record manuscript/evidence/fpga_lm_development_audit_2026-09-14.json
```

The 896 model and 128 baseline readouts denote producer score membership;
ordinary verification rederives losses from saved scores and reconstructs the
same baseline entries without new model forward calls. The separate audit
recomputes these same members, not another statistical cohort. This additional
verification computation is not additional training or independent data.
Owned public receipts failing post-publication checks are moved recoverably to
`rejected_public.json`, with exact rejected-byte identity in `failure.json`;
a different-inode foreign replacement is left untouched.

### Frozen-endpoint fitting diagnostic v1 (declared 2026-09-14, not yet admitted)

The completed six-cell producer improves beyond the fitted bigram prior but
misses every paired GRU margin. Its separate numerical audit is still running.
While that audit proceeds, prepare a forward-only diagnostic to distinguish
fitting from train/development differences. **No real diagnostic readout is
admitted until the existing full numerical audit completes successfully and
its actual public/private capsule is hard-pinned.** A partial or failed audit
cannot satisfy this requirement. This is a new declaration, not an extension
or modification of the six-cell trajectories; their 69 sources remain frozen.

Select exactly **64 training windows `range(0, 1024, 16)`**, namely
0, 16, ..., 1008, from the same corpus and original training split. This
deterministic evenly spaced cohort spans the fitted/trained window range;
it is neither random sampling nor the full training corpus. Do not choose
windows using observed losses. It has 8,192 positions, matching the size but
not the distribution of the existing development cohort. Reuse development
windows 88–151's already audited saved metrics; **no new development or final
test model evaluation** is allowed. Reserved validation 152–863 stays untouched.

Use the two original shared fitted initial models and the six complete final
checkpoints (step 256), never an intermediate or best-seed checkpoint. Extract
exact parameter bytes from their complete audited NPZ states with safe
non-pickle loading; verify full state metadata and preserve the external GRU
prior. Do not instantiate an optimizer, compute gradients, alter weights,
refit counts, train new trajectories or modify predecessor artifacts. Use
the same CPU FP64, one-thread, context-on forward semantics. Only the training
cohort receives new model calls: **2 shared initials × 64 + 6 endpoints × 64 =
512 model-window scores**, comprising **256 tropical and 256 GRU**. Initial
probabilities must agree between architectures at atol=rtol=1e-12. Context-off
training scoring is outside this diagnostic; the earlier development ablation
is not a substitute for it.

Evaluate the two fixed inherited count tables `C` and `2*C-1` on the same
training cohort, adding **128 baseline-window entries**. Reuse the completed
experiment's verified alpha-one count provenance; no new model/count fit or
corpus recount is required. Save every new per-window loss, accuracy, logits,
full tropical trace and guard counters, source/runtime identity and input
hash. Verify both shared initial and each final state against its bound
checkpoint before and after scoring. Report all six cells in the same paired
seed-major order, without model selection or a success/promotion gate.

For each architecture/order, report initial and final training-cohort CE and
the reused initial/final development CE; their improvements; final
development-minus-training CE; and its change relative to initialization:
`(D_final - T_final) - (D_initial - T_initial) = train_improvement - dev_improvement`.
Also report paired tropical-minus-GRU differences on both cohorts, all
per-window losses/accuracies, and descriptive order means/spread. Absolute
train/development differences already include inherited prior fitting and
distribution differences. A gap on this deterministic training sample does
not establish full-training loss, convergence, overfitting, iid significance
or an architectural capacity limit. No reserved-cohort or hardware promotion
follows from this diagnostic alone.

Keep a separate forward-only numerical audit: independently recompute all
**256 tropical scores** with full-matrix selection and all **256 GRU scores**
with the explicit-cell implementation, comparing FP64 values at atol=1e-9,
rtol=1e-11 and integer routes exactly. Add **256 supplemental guarded forwards**
to verify exact saved traces and semantic shortcut/fallback counts; do not
label these supplemental calls independent dense checks. Recompute the same
128 baseline entries. No VJP, optimizer replay, historical numerical rerun
or new development readout belongs in either diagnostic invocation.

Implement in new `manuscript/tools/diagnose_fpga_lm_endpoint_fit.py` with new
`tests/test_fpga_lm_endpoint_fit.py`: **71 source/test files** including the
unchanged 69-file ancestor closure. Bind the actual completed six-cell producer,
all its 5,529 payloads, its initial/final full states and inherited capacity/
guard/input closure, plus the future successful audit receipt and exact private
capsule. Incomplete/forged audits, changed bytes, additional/missing payloads,
mutated sources or an absent hard pin must fail closed. Test with synthetic
geometry only until admission, including independent scores, zero training/
gradient/optimizer calls, data fences, provenance and failure publication.
Require independent review and exact source hashes before real readouts.

Allow **1,800 seconds each** for one producer and one separate audit, counting
validation, all checking and publication under UTC and monotonic deadlines.
Prevent sleep from launch; no retry, segmentation or adaptive deadline. Limit
new diagnostic raw/public/audit retained files to **256 MiB**, excluding
read-only ancestors, with at least **4 GiB free** and conservative per-write
bounds plus actual cumulative accounting. Existing score payload sizes suggest
this is feasible (tropical context-on mean about 305 kB, GRU about 63 kB), not
a guarantee about new scores. Use exclusive append-only paths; retain partial
files and failure evidence; retract only owned rejected public receipts
recoverably. No native, AWS, FPGA or paid work is authorized by this section.

The successful-audit hard pin, final source hashes and exclusive commands/paths
remain **pending admission**. Implementing and testing this new scorer does
not admit its real-corpus execution while the predecessor audit is incomplete.

Pre-admission implementation review is GO at candidate source
`fd3170fea13f198924d063386b69301bf50263818ec2b881cc7bc0d3c6082312`,
with test file `3358cb23083964bc46c10561a020012cf7723e2871c38d2ccdeb54ee73398490`.
All 78 synthetic tests pass (tester invocation: 12.00 seconds), including
512 genuine tiny-model producer forwards, 256 dense tropical and 256 expanded-GRU
audit forwards, 256 supplemental guarded checks, and zero ordinary-check
forwards. Gradient, backward, optimizer construction, count fitting and old
trajectory APIs are forbidden in that fixture. Source review, Ruff and diff
checks pass. These are **offline candidate identities, not a real-execution
freeze**: the absent admission pin deliberately prevents real scoring. Insert
only the actual successful audit receipt hash after its process exits 0, then
rerun tests, review the exact change and record final source identities/commands.

#### Admission and execution freeze (2026-09-14, before endpoint readouts)

The predecessor's complete numerical audit exits 0 at **01:36:16 UTC**, after
4229.073073 UTC / 4229.049091 monotonic seconds. Its successful public receipt
SHA-256 is `500147cf9f03d8ac4b91362c74961ab8bd14aae438da2d6676a0f8a9b9a85e0a`;
the actual three-file private capsule, producer-public/seal identities and all
69 inherited sources pass independent read-only admission review. The only
change from the reviewed offline candidate is replacing the absent pin with
this actual successful receipt hash. All **78 tests pass again in 11.43 seconds**,
Ruff passes, and real input-only validation binds all 5,529 producer payloads
and 71 sources with **zero new model readouts**. Admission is now satisfied;
the scientific scope, sample, resource limits and zero-update rule above do
not change.

| Final added source | SHA-256 |
| --- | --- |
| `manuscript/tools/diagnose_fpga_lm_endpoint_fit.py` | `7798737aff5e9fbf6b65cbec4f07b64a28cfb03a21d256ee877b24e5cb0afecc` |
| `tests/test_fpga_lm_endpoint_fit.py` | `3358cb23083964bc46c10561a020012cf7723e2871c38d2ccdeb54ee73398490` |

Freeze all 71 sources until both invocations finish. Use these exclusive,
previously absent paths for the single producer:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_endpoint_fit.py --run \
  --output manuscript/build/fpga_lm_endpoint_fit_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_endpoint_fit_2026-09-14.json \
  --prior-run manuscript/build/fpga_lm_development_2026-09-14/v1 \
  --prior-record manuscript/evidence/fpga_lm_development_2026-09-14.json \
  --prior-audit manuscript/evidence/fpga_lm_development_audit_2026-09-14.json \
  --corpus data/tinyshakespeare.txt
```

After that producer exits successfully, invoke the separate forward audit
once; no results-dependent change or retry is permitted:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_endpoint_fit.py --audit \
  --output manuscript/build/fpga_lm_endpoint_fit_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_endpoint_fit_2026-09-14.json \
  --audit-output manuscript/build/fpga_lm_endpoint_fit_audit_2026-09-14/v1 \
  --audit-record manuscript/evidence/fpga_lm_endpoint_fit_audit_2026-09-14.json
```

#### Completed outcome and next decision (2026-09-14)

Both admitted endpoint invocations complete with exit 0. The producer runs
01:39:13–01:41:06 UTC (113.324741 UTC seconds); its independent forward audit
runs 01:41:44–01:44:09 UTC (144.480646 UTC seconds). Producer public SHA-256:
`6bb2ae3f379049378475196506197942c8fefb1f3aab34f3981d4c5d9bae3e3e`;
successful audit SHA-256:
`8a10febeb610962a6b6e86a8aacc5e94f934d7fc2730a9f23e7835d763291fbc`.
All 512 forward scores, 128 fixed baseline entries, zero-update restrictions,
71 source identities and predecessor artifacts verify. See the full
[results and limits](EVIDENCE.md#frozen-endpoint-fitting-diagnostic-2026-09-14).

Mean tropical-minus-GRU CE is **0.219129 on the fixed training sample** and
**0.248544 on reused development**, with only 0.029415 nat additional
development-minus-training separation. Every sampled window has a deficit in
every order, without implying independent repeated samples. This is an achieved
fitting deficit at the current budget; it does not identify slower convergence,
conditioning or insufficient representation.

**Proposed, not admitted:** continue the two order-0 endpoints for one fixed
additional pass, preserving complete Adam state, rates, graph, batch size,
training order and exact guarded/fallback semantics. Use the same predeclared
sampled-training and already-used development cohorts, with no rate selection,
early checkpoint selection or adaptive extension. Order 0 would be a diagnostic
pilot, not a best-seed/all-order success claim; confirmation across orders would
still be required. First test simple update-budget sensitivity before changing
bias coordinates or connectivity. A positive result demonstrates budget
sensitivity within this graph; a negative result does not prove a capacity
limit. A new protocol, exact state-restoration tests, resource bounds and source
freeze are required before execution. Today's admitted work is complete; no
new optimization trajectory or hardware tranche is started.

### Paired order-0 exact-Adam continuation v1 (declared 2026-09-14)

**Status: protocol declared; implementation, offline tests and independent
review must pass before the source freeze and numerical execution.** The user's
2026-09-14 continuation authorizes the previously proposed local control, not
FPGA scaling or a new paid tranche.

**Question.** Does one additional training pass narrow the achieved-fitting
deficit within the unchanged graph and optimizer recipe? The preceding audited
order-0 endpoints have sampled-training CE 2.1659766247861314 (tropical) and
1.9480478068284957 (GRU), and development CE 2.6083136116898573 and
2.3711543874523446. They are fixed inputs, not endpoints chosen by this run.
Order 0 is a diagnostic pilot selected in the preceding plan; it is not a new
seed sweep, best-run selection, all-order confirmation or independent sample.

**Immutable admission inputs.** Bind the complete prior development run,
endpoint diagnostic, their full artifact inventories and successful audit
capsules, and the unchanged 71-source ancestry. The development public receipt
is `2ec2fdef77c5e99f14e0402bafb3f0f1219b4c4d9cb084be737a1c035db65149`
and its successful independent audit is
`500147cf9f03d8ac4b91362c74961ab8bd14aae438da2d6676a0f8a9b9a85e0a`.
The endpoint public receipt is
`6bb2ae3f379049378475196506197942c8fefb1f3aab34f3981d4c5d9bae3e3e`
and its successful independent audit is
`8a10febeb610962a6b6e86a8aacc5e94f934d7fc2730a9f23e7835d763291fbc`.
These completed historical numerical audits are reused after full byte-binding;
their old optimization trajectories are not recursively rerun.

**State and execution.** In fixed order `mixed`, then `prior_gru`, resume the
actual `*_seed_0/state_000256.npz` full states, not the original fitted
initializations and not reconstructed empty-Adam states. Restore every named
parameter, first/second moment, scalar counter including its dtype, parameter
order, and optimizer group option. Re-export and compare the complete restored
state byte-for-byte before any new gradient. Counters start at 256 and finish
at 512 for every parameter. Verify this restoration and its next update against
an uninterrupted toy Adam trajectory before admission.

Keep CPU FP64, one Torch thread, the same software runtime, dimensionless
`u = beta * W`, full-width real mean centering, dynamic first-index max/min
ties, guarded forward with exact dense fallback, and the unchanged manual
selected-route VJP. Keep the fitted frozen GRU prior, embedding/hidden width
128, one recurrent layer, and all existing architecture choices. Tropical
Adam LR remains 0.01 and GRU LR 0.001; betas (0.9, 0.999), epsilon 1e-8,
zero weight decay, no clipping, foreach, fused execution or rounding. This is
the existing selected recipe, not equal hyperparameter tuning or matched
architecture/compute.

Perform exactly 256 additional updates per arm, each the mean exact CE over
four windows of 128 positions. Repeat precisely the preceding order-0
permutation: `random.Random(0).shuffle(list(range(1024)))`. No new permutation,
rate schedule, reset, early stopping, intermediate checkpoint selection,
results-dependent extension or architecture change. Save each complete batch
query, its actual gradient and tropical routes/guard counters or GRU logits,
plus all complete states 256 through 512. There are **512 new batch gradients,
512 new Adam commits and 514 saved complete states**, including the two
imported start states. Query filenames use additional-pass indices 0–255;
their metadata also records absolute before/after optimizer steps 256–512.
Each arm sees 131,072 new supervised pairs; 262,144 in
total across both arms, on the same 131,072 distinct pairs. Per-arm endpoint
history is 131,072 inherited fitted-prior pairs plus 262,144 cumulative
gradient exposures, not 393,216 distinct training pairs. No new count fit.

**Readouts and route diagnostics.** Only final step 512 is newly scored,
context on, on the same deterministic training subset
`range(0, 1024, 16)` and reused development windows 88–151: 64 windows / 8,192
positions per cohort. There are **256 new model-window readouts** across both
arms and cohorts. Reuse the bound step-256 per-window metrics and traces;
do not recompute old endpoint forwards or baseline fits. Do not score reserved
validation 152–863 or any test window. There is no fresh confirmation cohort,
full-training-loss estimate, context-off claim or quality-promotion gate.

For each arm, report both cohort CE/accuracy before and after, improvements,
and development-minus-sampled-training separation. Report tropical-minus-GRU
CE before and after and the change in that paired deficit for each cohort;
a negative change means it narrowed. Report paired window differences rather
than silently treating repeated windows as independent samples. From the
actual saved tropical traces on the *same* two cohorts, count changed and
total `AD_HEAD`, `PROJ_IDX` and `TAP_IDX` entries from 256 to 512. Separately
check/report inactive projection/head sentinel parameters and moment support
through the saved trajectory, and count observed head/projection selections
departing from the resumed single-active-entry support over all retained new
training queries and final scores. Temporal-tap changes are legitimate
adaptation, not projection-support failures. Do not freeze routes, suppress
fallback or condition admission/success on structural stability. Endpoint route agreement
does not imply that routes were constant between endpoints.

**Verification.** A separate bounded audit independently derives every new
tropical window VJP and full batch gradient with the dense selector/autograd
reference, and every GRU batch gradient through expanded recurrent cells.
Independently compare gradients, then use the **saved** gradients for exact
full-Adam state replay, checking every intermediate parameter, moment, counter
and group. Audit every new score and all descriptive/route/support summaries;
supplement dense checks with exact guarded trace/counter replay. Expected
independent coverage is 1,024 tropical window VJPs, 256 tropical batch Gs,
256 GRU batch Gs, 512 commits, 514 states, 128 tropical and 128 GRU final
score windows. There are 1,152 independent tropical forwards and 1,152 GRU
window equivalents, plus 1,152 supplemental guarded forwards. Integer routes
and saved-gradient state replay are exact; independent floating numerical
comparisons retain the existing absolute 1e-9 / relative 1e-11 tolerance.

**Operational bounds and freeze.** New runner
`manuscript/tools/continue_fpga_lm_development.py` and tests
`tests/test_fpga_lm_continuation.py` extend the source closure to 73; no frozen
source or prior artifact is edited. Record the final hashes, offline tests,
review decisions and exact commands below before any corpus/model gradient
or new score. Each producer and separate audit invocation has a 10,800-second
UTC/monotonic deadline; each arm has a 3,600-second bound. Aggregate retained
new raw/public/audit output is capped at 8 GiB with at least 4 GiB free-space
reserve, including publication overhead. Use exclusive new output paths,
complete sealed inventories, exact source/ancestry checks before and after,
and preserve any failed private attempt. Never overwrite old evidence or
publish a partial run as successful. These are local verification timings,
not optimized training efficiency or accelerator measurements.

**Interpretation boundary.** A shrinking gap establishes paired-gap sensitivity
to this additional budget under the selected unchanged recipe, but could also
occur if the GRU worsens. Claim improved tropical fitting only when its own
sampled-training CE decreases; report both arms' absolute changes separately.
A persistent or larger gap does not prove convergence or a representational
limit. Even a positive
order-0 result requires other-order confirmation before broader conclusions.
Bias-coordinate and connectivity changes remain separate future controls,
with new declarations and parity tests; no next experiment or FPGA tranche
is automatically admitted by this pilot.

#### Offline review and source freeze (2026-09-14)

All 99 new synthetic tests pass independently (root 4.55 s; test author 4.66 s;
runner author 4.54 s). They include real tiny two-arm producer/readback/audit
execution, exact nonzero full-state restoration and uninterrupted next-update
parity for both counter dtypes, independent scalar Adam recurrence,
saved-gradient versus near-independent-gradient separation, complete cohort
membership, observed support changes, and provenance/lifecycle fault tests.
The second-load checkpoint substitution test led to an additional pre-mutation
full-tensor digest check. Inherited runtime equality is enforced explicitly.
These are offline synthetic fixtures, not new scientific corpus readouts.

Independent mathematical/provenance/lifecycle review and root review pass.
Both authors have frozen their files with no further edits planned:

| Source | SHA-256 |
| --- | --- |
| `manuscript/tools/continue_fpga_lm_development.py` | `0d54ee17a70b63bc8de21f932dd83c5ee08468d0d27158ca24472aa3b12574ae` |
| `tests/test_fpga_lm_continuation.py` | `44e4e583dd461767f3de6c97dacae700d3812793e20ef3052c6a26f8756fcedf` |

The existing standard suite, excluding only the new test file while it was
under construction, passes **4,651 tests, 14 skipped, 10 deselected** in
380.57 seconds. The expected fake-unavailable-backend warning is unchanged.
The 81 documentation/evidence tests pass in 6.82 seconds; strict documentation
build, Ruff on the two new files, and `git diff --check` pass. Existing source
and historical evidence are not edited by this continuation work.

**Admission:** the actual input-only binding check has now exited successfully
with numerical entry points blocked: 73 sources, all 517 endpoint payloads,
all 5,529 predecessor development payloads, the complete audited ancestry, and
the original 1,115,394 corpus bytes verify. Runtime is unchanged Python 3.11.5,
Torch 2.10.0, NumPy 1.26.4, CPU with one Torch thread. No new model call was
made. The following producer is admitted **once**, with the fixed source above:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/continue_fpga_lm_development.py --run \
  --output manuscript/build/fpga_lm_continuation_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_continuation_2026-09-14.json \
  --prior-run manuscript/build/fpga_lm_endpoint_fit_2026-09-14/v1 \
  --prior-record manuscript/evidence/fpga_lm_endpoint_fit_2026-09-14.json \
  --prior-audit manuscript/evidence/fpga_lm_endpoint_fit_audit_2026-09-14.json \
  --corpus data/tinyshakespeare.txt
```

After that producer exits successfully, run the separate independent audit
**once**, with no results-dependent code or protocol changes:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/continue_fpga_lm_development.py --audit \
  --output manuscript/build/fpga_lm_continuation_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_continuation_2026-09-14.json \
  --audit-output manuscript/build/fpga_lm_continuation_audit_2026-09-14/v1 \
  --audit-record manuscript/evidence/fpga_lm_continuation_audit_2026-09-14.json
```

### Frozen-core-bias conditioning control v1 (declared 2026-09-14)

**Status: offline preparation only; not admitted.** The paired continuation
producer has completed, but its separate numerical audit is still running.
No next scientific trajectory is admitted until that audit succeeds, its
actual receipt is pinned, and this new protocol's tests/reviews/source freeze
are complete. This is one local conditioning control within the quality-gap
investigation, not a parameter sweep or a hardware tranche.

**Question and reason.** The saved continuation endpoints show absolute
improvement in both models but a larger paired fitting deficit, pending the
running audit. Head/projection selections remain on the original static
channel support, while temporal selections change. A source-derived hypothesis
is that redundant core-bias update coordinates worsen optimizer conditioning.
That is not an established cause of the gap. Test it before changing channel
connectivity, by freezing all `MP` and `MB` parameters at their **original
fitted initial values** and learning `EMBED`, `MH`, and `W_HEAD` as before.

**Conditional mathematical justification.** Let each layer's projection
selector `pi_l(i)` and head selector `sigma(v)` be static across *time and all
inputs under comparison*, shared by the compared models. This is stronger
than equality of two route tapes that may themselves vary with time/input.
With identical embeddings/temporal weights, real full-width centering and
residual additions, changing only selected `MP`/`MB` values induces a constant
channel offset with recurrence

```text
d_0 = 0
d_(l+1,i) = d_(l,i) + d_(l,pi_l(i)) - mean(d_l)
            + Delta_MP_(l,i,pi_l(i)) + Delta_MB_(l,i)
Delta_logit_v = d_(N,sigma(v))
```

All valid temporal candidates receive the same channel offset, including at
shortened causal prefixes. Thus a static active-head bias can represent the
same final class offset, **provided the static projection/head selections
also remain valid after compensation**. This is a real-arithmetic conditional
statement, not a globally equivalent model class or byte-exact FP64 rewrite.
Finite sentinels, selector changes, or rounding require actual checks. Offline
synthetic tests must exercise the recurrence and head compensation, and a
counterexample where unchanged but time-varying projection tapes make the
core-bias effect input-dependent. A fitting gain cannot be attributed solely
to redundant biases if the required static-support conditions fail.

**Single fixed trajectory.** Run only `mixed_core_bias_frozen`, from the exact
original fitted mixed model-seed-0 state at step 0 and **empty Adam**, not either
trained continuation endpoint. Bind the original full initial-state bytes and
the completed earlier mixed/GRU controls through the audited ancestry. Keep
the original initial function exactly unchanged; reuse its existing initial
scores instead of adding new initial forwards. The second-pass result is an
evidence/admission dependency, not inherited model training exposure.

Use the unchanged one-pass recipe: CPU FP64, one Torch thread, same runtime;
dimensionless `u = beta * W`, exact CE averaged over four 128-position windows,
real full-width means, dynamic first-index max/min, guarded forward with exact
dense fallback and unchanged input/parameter VJPs. Use exactly 256 updates and
the original `random.Random(0)` permutation of training windows 0–1023. Adam
LR 0.01, betas (0.9, 0.999), epsilon 1e-8, zero decay, no clipping, rounding,
foreach, fused execution, schedules, early stopping or checkpoint selection.
No new GRU run, alternate rate, alternate freeze pattern or extra pass.

**Explicit constrained update and evidence.** Compute and retain each complete
raw gradient with the unchanged model VJP. Separately save a complete *applied*
gradient: exactly zero tensors for `MP`/`MB`, byte-identical raw gradients for
all other regions. Apply the **saved applied gradient**, never the unmasked raw
gradient, to full eager Adam. With empty initial Adam and zero decay, the
frozen regions' weights remain byte-identical to their original values and
their first/second moments remain zero; check this at every saved state.
Their scalar counters still advance with the full optimizer for bookkeeping.
This mask is an intentional constrained optimization intervention, not a new
derivative formula or an unexplained discrepancy in gradient auditing.

Retain all 256 raw batch queries, 256 separate applied-gradient archives and
257 complete full-Adam states (0–256). State files use the explicitly inherited
development-state format; the enclosing new manifest declares this control's
distinct recipe. Each raw tropical query has four full window traces and
guard counters. All parameters, including inactive entries, retain full saved
arrays/moments. This one new arm has **131,072 new supervised gradient pair
exposures**, plus 131,072 inherited fitted-prior pairs: 262,144 total model-
history pair exposures on 131,072 distinct pairs. The original one-pass
controls have the same exposure accounting. No new count fit or baseline
evaluation is performed.

**Fixed comparisons.** Score only the final step 256 with context on, on the
same 64 sampled training windows `range(0, 1024, 16)` and 64 development
windows 88–151, totaling **128 new model-window scores**. Reserved validation
152–863 and all test data remain untouched. Reuse the audited original
mixed/GRU order-0 step-256 per-window metrics and fixed baseline references;
do not rerun those models. Report each cohort's CE/accuracy, improvement from
the identical original initialization, new-minus-original-mixed CE, residual
new-minus-GRU CE and paired window differences. This is a one-order diagnostic,
not independent repeated samples, whole-training loss, convergence evidence,
equal compute/hyperparameter tuning or fresh held-out confirmation.

Report frozen-region invariants at all states, inactive sentinel/moment
changes in the other regions, and head/projection departures from the original
static support over every new query and score. Separately compare temporal,
projection and head selections between the *two recipes' final states* on the
same cohorts; do not call that a within-trajectory before/after comparison.
Allow actual selector departures and dense fallback; they qualify the
interpretation rather than causing a quality-dependent stop. The declared
parameter-freezing invariant itself must hold exactly.

**Independent verification and bounds.** Audit all 1,024 raw window VJPs,
256 full raw batch gradients, 256 applied-gradient mask identities, 256
saved-applied-gradient Adam commits, 257 states and all 128 final scores.
Independent raw VJPs use the dense selector/autograd reference; compare them
to saved raw gradients at the existing absolute 1e-9 / relative 1e-11
tolerances, then check the applied mask exactly and use saved applied values
for byte-exact state replay. There are 1,152 independent dense forwards and
1,152 supplemental guarded forwards. Recompute all metrics, mask/freeze
invariants, support summaries and comparisons. No old numerical trajectory
is recursively replayed and no unreported model readout is added.

New sources `manuscript/tools/diagnose_fpga_lm_core_bias.py` and
`tests/test_fpga_lm_core_bias.py` extend the unchanged 73-source closure to 75.
The current continuation producer is pinned at
`07f18f23ce6cde30da7a6816eed661c4c901c4faeb96aa8f411149cebe69b813`;
its independent audit pin remains absent until actual successful completion.
Each producer and separate audit has a 10,800-second UTC/monotonic deadline
and a 3,600-second model-cell limit. Combined new raw/public/audit outputs
are capped at 8 GiB with 4 GiB free reserve, exclusive paths, sealed exact
inventories, before/after source/ancestry checks and retained failed attempts.
No old source/evidence may be edited or success relabeled. Final source/test
hashes, admission proof and exact commands must be recorded before any new
scientific gradient or score.

**Decision boundary.** Improvement over the original mixed control would show
this constrained parameterization/optimizer change matters at fixed budget;
it would not by itself establish the unique cause, all-order confirmation or
useful FPGA learning. A null or worse result would not prove a capacity limit.
No library default, manuscript quality claim or FPGA scaling decision follows
automatically. Connectivity changes remain a separate potential next control.

#### Offline conditioning-control review (2026-09-14)

The 66 new synthetic tests pass (test author 1.41 s; runner author 1.34 s).
They include actual independent dense forward checks of the static-selector
offset recurrence/head compensation and its unchanged-but-time-varying-route
counterexample; exact raw/applied mask separation; full frozen parameter and
moment invariants across 256 synthetic Adam steps; genuine tiny producer,
ordinary replay and independent gradient/state/score audit; and admission,
source, cohort and failure-safe publication checks. No actual corpus/model
experiment is performed by these fixtures. Mathematical, full source and
lifecycle reviews pass, as do Ruff and `git diff --check`.

The offline source candidate, with the successful audit pin still `None`, is
`0d1bb32b69468d509f6359f11bb6844f2f22cfaad2d0ff54a0236b30b6409d13`.
Frozen test source SHA-256 is
`7de5c2200b74f56b06b775c76ec3e2e29570032b370e8e5e717a0614f9a060e9`.
Only insertion of the actual successful predecessor audit pin is planned,
followed by another 66-test run, exact source hash and final admission record.
No scientific execution is admitted by this offline review alone.

#### Conditioning-control admission and final freeze (2026-09-14)

The predecessor audit exited 0 at 11:48:45 UTC. Its actual successful receipt
`ebc8bf0b4b4b3b723120ba94c0927818d70159e0b06fd24bc86ea4d63666b25c`
is now pinned. Independent review verifies that this one literal is the only
change from the offline candidate. All 66 tests pass again (root 1.26 s).
Final runner SHA-256:
`81189fa247b27428a386ac423882c0a9036c575e425328166bb76857c77c9292`.
Test SHA remains `7de5c2200b74f56b06b775c76ec3e2e29570032b370e8e5e717a0614f9a060e9`.

Actual input-only binding exits successfully with numerical entry points
blocked: 75 sources, 1,802 continuation payloads, 517 endpoint payloads,
5,529 development payloads, full audited ancestry and 1,115,394 corpus bytes.
Runtime is unchanged CPU FP64, Python 3.11.5, Torch 2.10.0, NumPy 1.26.4,
one Torch thread. No new model call was made. The following producer is
admitted once; its separate audit follows only after successful producer exit:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_core_bias.py --run \
  --output manuscript/build/fpga_lm_core_bias_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_core_bias_2026-09-14.json \
  --prior-run manuscript/build/fpga_lm_continuation_2026-09-14/v1 \
  --prior-record manuscript/evidence/fpga_lm_continuation_2026-09-14.json \
  --prior-audit manuscript/evidence/fpga_lm_continuation_audit_2026-09-14.json \
  --corpus data/tinyshakespeare.txt

/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_core_bias.py --audit \
  --output manuscript/build/fpga_lm_core_bias_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_core_bias_2026-09-14.json \
  --audit-output manuscript/build/fpga_lm_core_bias_audit_2026-09-14/v1 \
  --audit-record manuscript/evidence/fpga_lm_core_bias_audit_2026-09-14.json
```

#### Completed outcome and next decision (2026-09-14)

Both declared producers and their independent numerical audits complete with
exit 0. The paired extra pass improves absolute CE but widens the tropical
minus GRU gap to 0.3019 nat on sampled training and 0.3069 on development.
The original-start frozen-core control improves CE by only 0.0125 / 0.0099
nat against its first-pass mixed reference; gaps to the first-pass GRU remain
0.2054 / 0.2272. These are separate, single-order comparisons at different
training budgets, not competing endpoints for selecting a winner. Full results,
scope and provenance are in [EVIDENCE.md](EVIDENCE.md#paired-order-0-adam-continuation-2026-09-14)
and the [core-bias audit section](EVIDENCE.md#frozen-core-bias-control-2026-09-14).
No recipe is promoted. The evidence motivates the proposed routing question
below, not another automatic budget increase or FPGA scaling.

### Proposed next question: two-candidate channel routing (not admitted)

The current runs show temporal-route adaptation without observed projection
or head channel changes. The next proposed diagnostic asks whether permitting
an actual input-dependent **channel choice**, at the same update budget, helps
the remaining fitting deficit. This is a hypothesis, not a diagnosed capacity
limit. No implementation, training or new model evaluation is admitted here.

The candidate design starts from the original fitted state with frozen
projection weights and core biases, as in the conditioning control. Each
projection row keeps its primary input at weight 0 and receives exactly one
distinct alternate at dimensionless weight -0.01; all other entries retain
their original sentinel. Embeddings, temporal weights and the head train with
the same fixed order-0, 256-update recipe. There is no map, margin or rate sweep.

Use the actual bank/class mapping from `fpga_lm_diverse_projection.mapping`
and seeded permutation from `fpga_lm_projection_topology.mapping`. If the
primary input is `j = Y[c]` in bank `Y` (B, C or D), its alternate is
`Y[inverse_sigma[(sigma[c] + 1) % 128]]`. These banks are **not** physical
128-coordinate blocks; a physical-block shortcut can overwrite the primary.
Source and original-tensor inspection supports an initial-function-preserving
construction: all unused-bank embeddings, temporal weights and core biases
start at zero, and shared centering, temporal reduction and residual additions
preserve equality of unused-bank activations through every initial layer. The alternate
therefore begins strictly below the unchanged primary. This structural argument
does not replace exact initial logits/route parity tests before a future run.

Before admission, separately declare the full protocol, bound its actual
ancestry and sources, and test initial parity, raw/applied gradients, state
replay and independent verification. The current one-active-entry fast-path
certificate does not apply: use the unchanged dense fallback/reference, with
no hardware or speed claim. Record whether and where alternates actually win;
if none activate, a null quality result does not test the intended mechanism.
Any gain would be a fixed-budget result for this recipe, not proof of a global
capacity cause. Reuse only the existing diagnostic cohorts; reserved validation,
test data, FPGA scaling and optimizer-default changes remain separate decisions.

### Two-candidate channel-routing control v1 (declared 2026-09-14)

The user has requested continuation. This section develops the preceding
proposal into one fixed, original-start diagnostic. Implementation and synthetic
tests are authorized; real execution remains pending the explicit source freeze,
input-only admission check and commands below. Historical declarations and
evidence remain unchanged. There is no adaptive sweep or automatic second run.

**Question.** Does allowing a second competitive projection input improve
fixed-budget fitting over the audited frozen-core control? The intervention is
an initialization change within the existing full-dimensional max-plus graph,
not a hard restriction to two columns or a new fast path. Observed channel
competition must be measured separately from quality. Neither a gain nor a null
result identifies a global capacity limit or establishes useful LLM pretraining.

**Immutable inputs.** Bind the complete original-start core-bias producer,
its successful independent audit and exact private three-file audit capsule,
all 75 inherited source/test identities, and the full recursively checked
continuation, endpoint and development ancestry. Required predecessor pins:

| Input | SHA-256 |
| --- | --- |
| Core producer public | `b5d16c31d3bba43ce8ad7750ae62ff8b0fc278f8720e8f84b94008c90f13d23c` |
| Core producer seal | `6ca008ad1681d2a0db14509ec47889fb4bbc4960c6637f3565a06db5185eb6ee` |
| Successful core audit public | `c9185adcdef0db345fbcbd27d7372c9590a69f7979d688faeaa327196f2a7fcf` |
| Core audit manifest | `0a1f2507d6d95616ccab735d0457dc76b97662fe9e8bef16a64f1d04bcd6d4dd` |

The same 1,115,394-byte ASCII corpus and partition/window definitions apply.
The new source closure adds only `diagnose_fpga_lm_channel_choice.py` and
`tests/test_fpga_lm_channel_choice.py`, for 77 sources. No frozen dependency,
old checkpoint, evidence record, corpus or prior optimizer state is modified.

**Single intervention and initialization gate.** Start from the actual original
fitted mixed state 0 with genuinely empty Adam, never a trained endpoint.
Validate its exact D512/V128/N4/K7 structure: unused-bank embeddings and all
temporal weights/core biases are zero; each projection row has the canonical
seeded primary at zero and all other entries at the original sentinel; the
head retains its exact fitted anchors. Preserve all non-projection bytes.
For primary `j = Y[c]`, with `Y` its actual B/C/D bank, add the distinct input
`Y[inverse_sigma[(sigma[c] + 1) % 128]]` at dimensionless weight **-0.01**.
Keep primary zero and every other sentinel unchanged. Use actual bank lookups,
not contiguous physical blocks. The full MP/MB tensors remain frozen thereafter.

Before the first update, save **128 new candidate initial traces**, on training
windows `range(0, 1024, 16)` and development windows 88–151. Require byte-exact
logits, centering offsets and all head/projection/tap selectors against the
fully bound, independently audited **original initial** traces. These are
initial-function checks, not new comparator runs or a model-selection screen.
Guard counters intentionally differ: the candidate must use the existing
whole-model dense fallback, whereas the old initial model could certify its
one-entry fast path. Compare candidate guard metadata to its actual execution,
not to historical guard counters. No new original-model forward is required.

Also compare the already scheduled first training query against the saved
core-control query 0 before committing an update: identical ordered inputs,
targets, logits, full traces and all five full raw-gradient tensors. Same
logits/tapes and the unchanged batch divisor imply the same selected reverse
rule; this must be checked, not inferred from similar initial CE. This adds no
gradient query or training exposure. Any initialization/parity failure stops
and preserves a failure record; do not silently change the map or margin.

**Fixed learning recipe.** One arm, one original order-0 pass over 1,024 windows,
256 updates of four 128-position windows, exact mean CE in dimensionless units,
CPU FP64 and one Torch thread. Use unchanged eager Adam, LR 0.01, betas
0.9/0.999, epsilon 1e-8, no decay, clipping, schedule, rounding, foreach or fused
implementation. Keep every raw gradient. Separately archive applied gradients
with only MP/MB replaced by positive zeros; EMBED/MH/W_HEAD are byte-identical
to raw values. Apply the saved masked values, retain all full parameters,
moments, counters/dtypes and groups, and verify MP/MB weights stay exactly at
their transformed initial values with zero moments and advancing counters.

The unchanged full-D dynamic forward, temporal max/min, real mean centering,
first-index tie policy and manual VJP remain in force. The current guarded
wrapper must take its ordinary whole-model dense fallback; do not introduce
a two-column selector or reuse historical routes to choose new winners.

**Readouts and controls.** Save 64 final training and 64 final development
context-on traces on the same fixed cohorts. Reuse the audited original
first-pass mixed, frozen-core mixed and fitted-prior GRU references, plus their
initial/baseline records; do not retrain or newly evaluate a comparator or fit
a baseline. Compare per-window CE/accuracy and their descriptive means without
best-window/seed selection. No context-off, reserved validation 152–863 or final
test evaluations are included. Correlated reused development windows are not
fresh held-out confirmation or independent trials.

**Routing measurements.** Partition every valid projection selection into
primary, alternate or other-sentinel, with exhaustive per-layer and total
counts. Reject malformed/out-of-range tapes, but report legitimate other
winners without censoring or declaring a quality failure. Record head support
and temporal-route changes separately. At each fixed pre-update checkpoint,
count rows choosing **both primary and alternate** across its four query
windows; do the same across each final 64-window cohort. Aggregating alternate
winners across changing weights alone does not establish input dependence.
These are unweighted saved-tape statistics, not gradient mass or causal
context ablations. If alternates never win, a null quality result does not
exercise the proposed mechanism. Activation itself is not a success gate.

**Accounting and verification.** Retain 256 raw queries, 256 applied-gradient
archives, 257 complete states, 128 initial parity traces and 128 final traces.
There are 1,024 training-window forwards and 256 score/parity forwards, or
**1,280 new model-window forwards**, with no new original-model readout. New
gradient-pair exposures are 131,072; inherited fitted-prior exposures are
131,072, for 262,144 history exposures on 131,072 distinct pairs. There is no
inherited gradient pass: recent experiments are proof dependencies only.
Ordinary readback uses saved gradients/traces. A separate independent audit
reconstructs all **1,024 window VJPs, 256 raw batch gradients and 1,280 dense
forwards**, verifies all masks, initial parity, states and scores, and applies
the **saved applied gradients** for byte-exact Adam replay. Another 1,280
supplemental guarded-wrapper forwards check the actual dense-fallback path.
Use inherited floating tolerances (absolute 1e-9, relative 1e-11) only for the
independent implementation comparison; selectors, parity, masks and state
replay remain exact. No historical numerical trajectory is rerun.

**Lifecycle and bounds.** Each producer/audit invocation has an absolute
10,800-second limit and a 3,600-second numerical-cell limit, enforced by the
inherited deadline mechanism. Cap the combined new raw/public/audit outputs at
8 GiB and retain 4 GiB free space. Require exclusive new output paths, complete
source/input/payload manifests and exact inventory before sealing/publication,
immutable saved declarations, private/public audit-byte agreement and successful
postchecks. Preserve failed attempts; never overwrite or relabel them as
success. Freeze the exact inventory, final source hashes and commands before
real execution. A producer completion does not substitute for its independent
audit. Timings cover verification overhead and are not efficiency benchmarks.

There is no quality-promotion gate, optimizer-default change, native or physical
FPGA execution, paid resource launch or hardware advantage claim in this study.
The next decision follows the audited result; no further experiment is queued.

#### Channel-control implementation and offline checks

The planned exact producer inventory is **1,288 payloads**: the core control's
1,158-file layout, 128 additional initial trace archives, `transform.json` and
`initial_parity.json`. The seal is separate from its payload inventory. The
successful independent audit has exactly its private manifest, saved declaration
and verification receipt; its public receipt must be byte-identical to the
private verification. These counts are to be tested before final admission.

The inherited regression suite passes **4,816 tests, 14 skipped and 10
deselected in 390.38 seconds**, with the expected unavailable-test-backend
fallback warning. The command excludes only the new test file while it is being
written: `python -m pytest -q --ignore=tests/test_fpga_lm_channel_choice.py`.
All 75 inherited source hashes and all four predecessor identity pins above
also match. This is an offline checkpoint, not admission or a numerical result
for the new control. Its targeted tests, final source freeze and actual
input-only binding still precede real execution.

The complete new synthetic suite now passes **79 tests** (root 2.13 s;
independent tester 2.19 s), including genuine tiny producer/readback/independent
replay, initial trace/raw-gradient parity, actual alternate activation,
same-checkpoint routing counts, noncausal-tape rejection, frozen full Adam
states, corrupt/resealed artifacts and fail-closed publication. Ruff and
`git diff --check` pass. The offline runner candidate, with admission pin still
`None`, is `809eeb55fbb7e87fbc2478641e966a83a6b8c375da8f137d775509a19a80759d`.
Frozen test SHA-256 is
`0b2b313b47ec3391dc79cc6b3d7ab2f6de08cf6d2c2bd3f9879c7bde6ff7bda1`.
Only insertion of the actual successful predecessor-audit pin is planned,
followed by a repeat of these tests and the explicit input-only admission.
This offline checkpoint has made no real model readouts or training updates.

#### Channel-control final freeze and admission (2026-09-14)

Root and independent whole-source reviews pass. The actual successful core
audit receipt is now pinned; a separate file-only review confirms this one
literal is the **only** change from the offline candidate. All 79 tests pass
again after pinning (root 2.23 s). Final runner SHA-256:
`f2c636f088114bc3027d82d75d79eb1db4e0e645a02f055f835e1288717813a1`.
The frozen test SHA remains
`0b2b313b47ec3391dc79cc6b3d7ab2f6de08cf6d2c2bd3f9879c7bde6ff7bda1`.

Actual input-only binding completes successfully with numerical/optimizer and
subprocess entry points blocked: **77 sources, 1,158 core payloads, 128 original
initial trace inputs**, full audited ancestry and **1,115,394 corpus bytes**.
There are zero blocked-entry-point attempts, model readouts or optimizer steps.
The expected new payload inventory is 1,288. Runtime remains CPU FP64,
Python 3.11.5, Torch 2.10.0, NumPy 1.26.4 and one Torch thread.

The following producer is admitted **once**. Only successful producer exit
permits the separate independent audit below. Existing output paths must not
be overwritten; any failed attempt remains a failure with its retained files.

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_channel_choice.py --run \
  --output manuscript/build/fpga_lm_channel_choice_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_channel_choice_2026-09-14.json \
  --prior-run manuscript/build/fpga_lm_core_bias_2026-09-14/v1 \
  --prior-record manuscript/evidence/fpga_lm_core_bias_2026-09-14.json \
  --prior-audit manuscript/evidence/fpga_lm_core_bias_audit_2026-09-14.json \
  --corpus data/tinyshakespeare.txt

/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_channel_choice.py --audit \
  --output manuscript/build/fpga_lm_channel_choice_2026-09-14/v1 \
  --record manuscript/evidence/fpga_lm_channel_choice_2026-09-14.json \
  --audit-output manuscript/build/fpga_lm_channel_choice_audit_2026-09-14/v1 \
  --audit-record manuscript/evidence/fpga_lm_channel_choice_audit_2026-09-14.json
```

#### Channel-control completed outcome and next question (2026-09-14)

The admitted producer and its separate independent numerical audit both
complete with exit 0. The [channel-control evidence](EVIDENCE.md#two-candidate-channel-routing-control-2026-09-14)
records the sealed outputs, complete verification counts and timings. Initial
function and first-gradient parity pass; both choices become active at
**2,031 / 2,048** development layer/row pairs within the final checkpoint.
Nevertheless sampled-training / development CE is **2.186878486 / 2.608303389**,
worse than frozen core by **0.033448323 / 0.009903184 nat**, with the GRU still
at **1.948047807 / 2.371154387**. This is a negative result for the specific
fixed-offset intervention, not a capacity proof or diagnosis of the remaining
gap. The reused development cohort and single training order remain exploratory.

**Proposed only, not admitted or run:** does learning each candidate pair's
score contrast improve this fixed-budget recipe? One candidate parameterization
is `primary = m + d`, `alternate = m - d`, with fixed `m = -0.005` and initial
`d = 0.005`, retaining the present dimensionless `0 / -0.01` start. Its selected
derivative is `g_d = g_primary - g_alternate`. Keep `MB` and all other `MP`
entries frozen, with the existing `EMBED`/`MH`/`W_HEAD` recipe, original
initialization, order and budget. Contrast-coordinate Adam units, state,
independent replay, tests and bounds must be declared and reviewed before any
new execution. This paragraph is not an executable experiment declaration.

Fixing the pair's mean weight removes an independently trainable common-mode
parameter direction; it does **not** fix the output mean or eliminate bias
effects. Even at unchanged routes, contrast changes output values and subsequent
gradients. Thus a gain would not isolate a pure routing effect or prove that
fixed thresholds caused the deficit. The present fixed-offset intervention is
not an established strict capacity superset of the original single-route
family, and trainable features already provide indirect control of relative
candidate values. No additional scientific run, optimizer-default change or
FPGA launch is queued.

### Three-arm learned-routing contrast block v1 (declared 2026-09-15 UTC)

The user now authorizes agents to conduct the next few experiments and then
review progress. This supersedes the preceding no-next-run decision only for
the three controls below. Implementation and synthetic testing are authorized;
real execution still requires final source freeze, independent review,
input-only binding and the admission commands recorded below. No adaptive
fourth arm, automatic extension, FPGA execution or paid resource is included.

**Question and fixed contrasts.** The audited fixed-offset two-choice model
activates channel competition but reaches sampled-training / development CE
2.186878486 / 2.608303389, worse than frozen core at
2.153430163 / 2.598400205 and the GRU at 1.948047807 / 2.371154387.
Test adjustable pair scores, their step scale, and the pair-common constraint:

| Arm / CLI identifier | Contrast `d` Adam LR | Pair-common coordinate `m` |
| --- | ---: | --- |
| A / `contrast` | 0.01 | Fixed |
| B / `contrast_slow` | 0.001 | Fixed |
| C / `contrast_mean` | 0.01 | Learned with Adam LR 0.01 |

All other trainable coordinates (`EMBED`, `MH`, `W_HEAD`) retain LR 0.01.
Compare A with the audited fixed-offset model, B with A, and C with A.
Report every arm against original mixed, fixed-offset, frozen-core and GRU
references; do not present only the lowest development loss.

**Immutable inputs.** Bind the completed two-choice producer, its successful
independent audit and exact private three-file capsule, all 77 inherited
source/test identities, all 1,288 producer payloads (911,591,227 bytes), and
the full recursively checked core/continuation/endpoint/development ancestry.
The predecessor pins are:

| Input | SHA-256 |
| --- | --- |
| Channel-choice producer public | `f8e7f3ddc66fff4f3d1d41ff1228eca030da8c183f437d38cf4181befa2ecbb8` |
| Channel-choice producer seal | `e17ed897707852ce5124f60a7726529cff7a9760567e5941ce797232882e3767` |
| Successful channel-choice audit public | `96e824375caf67522059a6b2a8864ff58b1c2ba98f668e62f90b6680d53d8e72` |
| Channel-choice audit manifest | `6e9b8e56f5d5d2ee41ce41bff3e63855ad95a3698901d42860f484f30022c2ec` |

The new source closure adds only `diagnose_fpga_lm_routing_contrasts.py` and
`tests/test_fpga_lm_routing_contrasts.py`, for 79 identities. All previous
sources, evidence, states and corpus bytes remain immutable. Use the unchanged
1,115,394-byte ASCII corpus, partitions and 129-byte-stride windows.

**Parameterization and actual initial parity.** Start each arm independently
from the original fitted mixed state 0, not a trained endpoint, and genuinely
empty Adam. Retain the audited two-choice map in the actual interleaved feature
banks, with full D512/V128/N4/K7 geometry. At each of 2,048 layer/row pairs,
materialize the primary and alternate weights as `m + d` and `m - d` in that
literal FP64 operation order, initially `m = -0.005`, `d = 0.005`. This must
reproduce the audited physical `0 / -0.01` initialization byte-for-byte.
All other `MP` entries and all `MB` entries stay at their original fixed bytes.
Do not simultaneously optimize physical candidate weights and their coordinates.

Each arm separately saves and verifies 128 new initial traces (64 sampled
training and 64 development), with exact logits, centering and all selectors
against the bound original initial traces. Its scheduled first batch must
match the archived first query's ordered inputs, targets, complete traces and
all five raw gradients before any update. Initial guard counters are checked
against actual execution, not equated to the historical one-support guard.
These checks use no additional gradient query or original-model forward.
Any failure stops the arm without changing its map, margin or recipe.

**Selected derivatives and optimizer semantics.** Preserve every full raw
five-region gradient. Pull it back to the named optimizer coordinates with
`g_d = G_primary - G_alternate` and, for C,
`g_m = G_primary + G_alternate`; there is no factor of one half. The three
feature/head gradients remain unchanged. In A/B the fixed `m` is an explicit
serialized buffer omitted from Adam, not a parameter with fabricated moments.
In C it is an ordinary optimized coordinate. All coordinates use eager Adam
betas 0.9/0.999 and epsilon 1e-8, no decay, clipping, schedule, rounding,
foreach or fused implementation. Separate declared parameter groups set only
the rates above. Store all coordinate parameters, moments, scalar counter
dtypes, groups/options and their named association, as well as the complete
materialized five-region model at every checkpoint. Saved applied coordinate
gradients, not newly rounded substitutes, drive exact state replay.

The declared optimizer has 147,456 scalar coordinates in A/B and 149,504 in C;
the materialized full model still has 1,196,032 entries. These are declared
coordinate counts, not measured nonzero-gradient counts or a hardware-memory
claim, and are not exactly parameter-matched to the 131,968-parameter GRU.

Fixing `m` fixes a parameter direction, not mean outputs or functional biases.
Changing `d` shifts the winning value even when the winner stays unchanged.
Nor does FP64 algebra guarantee `(m+d)+(m-d) == 2*m` bitwise: check the stored
buffer and canonical materialization, not that invalid floating-point identity.
Initially the alternate loses, so `G_alternate = 0`. In C the two equal-rate
coordinates initially receive equal gradients; in real arithmetic the primary
moves twice a single coordinate step while alternate changes cancel. C versus
A therefore tests the additional coordinate under its declared optimizer,
not capacity at matched physical-edge update magnitude. A nominal 0.01 change
in `d` changes the pair gap by 0.02, versus 0.002 for the smaller-rate arm;
actual Adam updates must be measured from the saved states.

**Matched budget and execution.** Each arm uses the exact original order-0
shuffle of training windows 0–1023, 256 updates, batch size four, sequence
length 128, exact mean CE, CPU FP64 and one Torch thread. Every arm starts
fresh; no arm inherits another arm's state. Retain the existing full-D dynamic
selector, alternating temporal max/min, real centering, residuals and first-index
tie rules. Record actual guarded-fast/fallback counters and verify them in
the independent audit; do not force a two-column forward or assume a later
guard certificate cannot change. Initial dense fallback is checked explicitly.
No new fast path or native/FPGA execution is part of this block.

**Readouts, decisions and limitations.** Score the same 64 training windows
`range(0,1024,16)` and development windows 88–151, context on, at each final
checkpoint. Reuse audited comparator scores and traces; do not retrain a GRU,
refit a prior, evaluate context-off, or access reserved validation 152–863 or
the final test. Record complete per-window CE/accuracy and descriptive means.
Report exhaustive primary/alternate/other route counts, both-choice rows
within each fixed query/final checkpoint, and head departures; valid off-pair
winners are observations, not censored failures. Counts are not gradient mass
or evidence of within-trajectory instability on identical inputs.

For review, distinguish recovering the failed fixed-offset baseline from
improving the strongest frozen-core recipe. Predeclare **material progress**
as at least 0.01 nat lower CE than frozen core on both sampled training and
development; this is a descriptive replication trigger, not statistical
significance or a promotion gate. Also report each remaining GRU gap directly.
Mixed or smaller effects and failures remain in the report. Interpret results
only after the corresponding independent audit; the block review waits for
all three terminal outcomes. Single-order, repeatedly reused development
scores are exploratory and cannot establish useful LLM pretraining, global
capacity, convergence or hardware advantage. Any later replication, untouched
cohort access or scaling is a separate decision after this review.

**Accounting and independent verification.** Per arm retain 256 full raw
queries, 256 coordinate-gradient archives, 257 complete coordinate/materialized
states, 128 initial parity traces and 128 final traces. The planned layout has
1,288 payloads per arm, to be confirmed by synthetic exact-inventory tests
before final admission. Each arm makes 1,024 query-window forwards and 256
initial/final score forwards, or 1,280 producer forward windows; new gradient
exposures are 131,072, inherited fitted-prior exposures 131,072, and inherited
gradient passes zero. Across the block: 768 updates, 771 states, 3,840 producer
forward windows and 393,216 new gradient exposures. Summed prior-plus-gradient
histories are 786,432 exposures, but all three use the same 131,072 distinct
training pairs; they are not independent datasets or seeds.

For each arm separately, an independent dense forward and selected-route
autograd reverse reconstruct all 1,024 window VJPs and 256 batch gradients.
Check coordinate pullbacks independently through the materialization map,
all initial/final scores, and another 1,280 supplemental guarded forwards.
Use inherited tolerances (absolute 1e-9, relative 1e-11) only for independent
floating comparisons; selectors, initial parity, applied-coordinate gradients
and full saved-state replay remain exact. Do not rerun any old numerical
trajectory. Require immutable source/input/raw/seal/public bytes and an exact
three-file private audit capsule with byte-identical public verification.

**Parallelism and lifecycle.** Agents may supervise the three separate
single-thread CPU arms concurrently only after joint source freeze and root
admission. Each arm has exclusive producer/public/audit paths and one producer
followed by one independent audit after successful producer exit. Per invocation
retain the 10,800-second absolute deadline and 3,600-second numerical-cell
deadline. Charge each arm's producer, public and audit outputs together against
8 GiB, for 24 GiB across the block. Require at least 28 GiB free before concurrent
launch to reserve those maxima plus a 4-GiB free-space margin; retain per-write
storage checks. Preserve failures and partial artifacts, never overwrite or
relabel an attempt, and do not retry scientifically without a reviewed new
declaration. Timings under concurrency and verification overhead are not
optimizer, architecture or hardware efficiency benchmarks.

#### Three-arm final freeze and admission (2026-09-15 UTC)

Independent whole-source and mathematical/lifecycle reviews pass. The new
synthetic suite has **92 passes** (tester 4.56 s), including complete tiny
producer/readback/independent audits for every arm, three 256-step coordinate
Adam checks, coordinate autograd/finite differences, full-state corruption,
source/input drift, pre-first-commit parity failure and publication safeguards.
All 77 inherited source identities remain unchanged. Ruff, CLI help and
`git diff --check` pass.

Offline runner SHA-256 was
`2601ec3e019f196a2038148a2ac3da092c29dfa3426e3c46a610532df900305d`.
Root inserted only the actual successful predecessor-audit receipt pin.
Independent file review confirms that replacing this literal with `None`
reproduces the exact offline hash. Final runner SHA-256 is
`f9626fb394fed425581444c9b30fbf9c31debe034ec0c5d70e467b0857a3f484`;
frozen test SHA-256 is
`0b10c14cc0dde6add5256574bd3cbe3ca8976f4830718424c6eee1bdcb175237`.
The root's post-pin suite passes **92 tests in 4.60 seconds**.

Actual input-only binding passes separately for all three arm identifiers:
79 source identities, 1,288 predecessor payloads, 128 bound original initial
traces and 1,115,394 corpus bytes each. Numerical/model/optimizer and subprocess
entry points were blocked; there were **zero attempted blocked calls**, new
model readouts or optimizer steps. Runtime is Python 3.11.5, Torch 2.10.0,
NumPy 1.26.4, CPU FP64 and one Torch thread. Every new producer and audit
expects 1,288 producer payloads. Preflight confirms all twelve output targets
are absent and free space exceeds the 28-GiB aggregate reservation. The host
reports 16 logical CPUs; three single-thread arms may run concurrently.

**Admission:** one producer and, only after its actual successful exit, one
independent audit for each of `contrast`, `contrast_slow`, `contrast_mean`.
Each agent owns one arm's exact output paths. No source/test edits, extra arms,
scientific retries or new cohort access are admitted during this block.
The full regression suite may complete concurrently; targeted numerical and
lifecycle tests above already pass. Hold this declaration unchanged until all
three producer/audit attempts have reached terminal outcomes.

For each listed identifier, use the following commands with `routing_arm` set
to that identifier exactly once. Do not run the audit unless the producer has
actually exited successfully; a printed intermediate metric is not completion.

```sh
routing_arm=contrast
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_routing_contrasts.py --run \
  --arm "$routing_arm" \
  --output "manuscript/build/fpga_lm_routing_contrasts_2026-09-15/$routing_arm/v1" \
  --record "manuscript/evidence/fpga_lm_routing_contrasts_${routing_arm}_2026-09-15.json" \
  --prior-run manuscript/build/fpga_lm_channel_choice_2026-09-14/v1 \
  --prior-record manuscript/evidence/fpga_lm_channel_choice_2026-09-14.json \
  --prior-audit manuscript/evidence/fpga_lm_channel_choice_audit_2026-09-14.json \
  --corpus data/tinyshakespeare.txt

/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_routing_contrasts.py --audit \
  --arm "$routing_arm" \
  --output "manuscript/build/fpga_lm_routing_contrasts_2026-09-15/$routing_arm/v1" \
  --record "manuscript/evidence/fpga_lm_routing_contrasts_${routing_arm}_2026-09-15.json" \
  --audit-output "manuscript/build/fpga_lm_routing_contrasts_audit_2026-09-15/$routing_arm/v1" \
  --audit-record "manuscript/evidence/fpga_lm_routing_contrasts_audit_${routing_arm}_2026-09-15.json"
```

#### Three-arm completed review (2026-09-15 UTC)

All three producers and all three separate numerical audits reached actual
exit 0. A subsequent file-only review verifies all 79 unchanged source files,
3,864 payloads (3,041,772,186 bytes), exact three-file audit capsules, and their
public/manifest/seal/declaration links. These terminal outcomes release the
previous declaration freeze; its saved copies remain immutable.

None of the three arms improves on frozen-core on either scored cohort.
Final sampled-training/development CE is 2.197280004 / 2.614672326 (contrast),
2.185692063 / 2.620297871 (slower contrast), and
2.197037636 / 2.616082766 (contrast plus mean), versus frozen-core
2.153430163 / 2.598400205 and the matched first-pass GRU
1.948047807 / 2.371154387. The slower contrast trades better sampled fitting
against worse development loss relative to the faster contrast; the added
mean coordinate does not recover the deficit. All material-progress flags
are false. This is a single-order, reused-development negative result, not
a global representation limit. See [completed evidence](EVIDENCE.md#three-arm-learned-routing-contrast-block-2026-09-15).

### Frozen-feature residual readout oracle v1 (declared 2026-09-15 UTC)

The user has explicitly requested the next diagnostic. This declaration
supersedes the preceding block's prohibition on another experiment only for
the two readout controls below. No further routing sweep, backbone training,
new evaluation cohort, paid resource, FPGA/native execution or default change
is authorized by this block.

**Question:** can a richer output correction exploit information already
available at the audited frozen-core endpoint? Compare a bias-only correction
with a nested affine correction, so a gain from class recalibration is not
mistaken for an improvement from the feature-dependent part of the readout.
This is a CPU diagnostic oracle, not a proposed multiply-heavy FPGA head or a
parameter-/optimization-budget-matched architecture comparison.

Freeze every original parameter at the audited core-bias order-0 checkpoint
256, retaining the original full-D512/V128/N4/K7 forward, exact CE units,
real centering, causal context and first-index ties. The unchanged core
producer/audit/corpus binders validate the actual source and complete input
ancestry. Do not initialize from the newer, worse routing endpoints.

Cache the final head-input vector `h` and actual original logits `z0` for
exactly 1,024 existing training windows (indices 0–1023) and 64 development
windows (88–151), each 128 prediction positions. Sampled-training windows
`range(0,1024,16)` reuse this cache. Store all inputs, targets, full selector
tapes, centering offsets and actual guard counters alongside every feature
vector. The production path uses the unchanged guarded forward and exact
tape reconstruction; no backbone gradients or optimizer updates occur.
Require byte-exact frozen-core traces/logits on all 128 previously scored
endpoint windows before fitting. No static head-support assumption is needed.

For each control, start every new coordinate at zero:

- **bias:** `z = z0 + b`, 128 new coordinates;
- **affine:** `z = z0 + h @ W.T + b`, 65,664 new coordinates.

The correction initially reproduces actual `z0` byte-for-byte. Features and
base logits are not rescaled, centered or multiplied by another beta.
Both controls minimize the same full-training mean exact cross-entropy plus
`0.001/2 * ||theta||_2^2`; all biases and, where present, all weights are
regularized. The bias-only family is nested in the affine family. Neither
fitting callback receives development arrays. Lambda is fixed, not selected
from this development set.

Use CPU FP64, one Torch and one BLAS thread, chunk size 2,048 positions,
SciPy L-BFGS-B with analytic full-data gradients, no bounds, `maxcor=10`,
`maxls=20`, `ftol=0`, and
`gtol=sqrt(2*lambda*1e-5 / parameter_count)`. Enforce a maximum of 200 accepted
iterations, 400 attempted full objective/gradient evaluations, and 3,600
seconds per solve.
The per-solve time limit is checked before/after each data chunk and accepted
callback; an in-flight BLAS operation is covered by the global invocation
watchdog rather than a separate preemptive per-solve timer.
Check the actual full-gradient Euclidean norm separately: the numerical target
is `||g||_2^2/(2*lambda) <= 1e-5`.
Stop at that target or the fixed cap; return only the last fully evaluated
accepted point, including the zero start if no later point is accepted.
Do not restart or select an unaccepted line-search trial. A cap/early solver
stop above target is an inconclusive optimization outcome, not evidence of
insufficient representation.

In exact real arithmetic the lambda-strongly-convex objective obeys
`F(theta)-F* <= ||grad F(theta)||_2^2/(2*lambda)`. Without a justified bound
on floating-point gradient error this is only a **numerical suboptimality
estimate**, not formal certification. Report raw CE, penalty, regularized
objective, full gradient norm, estimate, SciPy termination and actual call/
iteration counts separately. The installed SciPy Python/compiled optimizer
files, package versions, BLAS configuration and complete 82-file project
source closure must be identified and remain unchanged during execution.

Save initial/final coefficient and full-gradient arrays and every actual
objective trial's scalar/hash provenance plus accepted-iteration references.
Ordinary readback checks the immutable cache, endpoint gradients/objectives,
solver receipt consistency and all final scores; it does not refit a model.
No exact L-BFGS trajectory replay or independent verification of every trial
is claimed. An independent invocation must reconstruct all 1,088 feature
windows through fresh full-matrix dynamic selection, compare all feature
columns as well as logits/routes, and verify the same number of supplemental
guarded forwards and their actual counters. It independently checks the four
initial/final full-training readout gradients through Torch autograd and all
256 initial plus 256 final fixed-cohort readout windows across the two arms.
Floating comparisons use `atol=1e-9, rtol=1e-11`; discrete selectors/data and
initial parity are exact. Saved coefficients, not recomputed optimization,
determine the audited endpoints.

Report BOTH controls versus unchanged frozen-core, original mixed and GRU
references, and affine minus bias, with paired per-window CE/accuracy and
means. A decrease of at least 0.01 nat on both sampled-training and development
cohorts is a descriptive progress flag only, contingent on numerical completion
and independent verification. This repeatedly used development slice is not
fresh confirmation. A positive affine result beyond bias supports usefulness
of feature-dependent readout information; it neither proves a unique causal
bottleneck nor establishes contextual information, a globally adequate backbone,
useful LLM pretraining or hardware efficiency. A negative numerically optimized
result concerns only this endpoint and regularized affine-correction problem,
not all linear/nonlinear readouts or the architecture's capacity.

Count each solver's attempted/completed full-gradient calls separately from
accepted iterations: each complete call processes the same 131,072 training
pairs. There are zero new distinct training pairs and zero new backbone
gradients. Each arm additionally has four producer endpoint-verification
full-gradient evaluations (initial/final after fitting and during readback)
and two independent-audit evaluations, not extra learning. This readout
optimization budget is explicitly not matched to the earlier GRU training.
The backbone inherits its original fitted-prior and one gradient-training
pass; proof dependencies are not additional inherited training passes.

Admit exactly one producer containing both controls and, after its actual
exit 0, one independent endpoint audit. Each invocation has a 10,800-second
global watchdog; all new private/public outputs together are capped at 8 GiB
with 4 GiB free reserve. Preserve failures, never overwrite/relabel an attempt,
and freeze all sources plus this declaration until both invocations terminate.
Initial output paths must be absent:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_readout_oracle.py --run \
  --output manuscript/build/fpga_lm_readout_oracle_2026-09-15/v1 \
  --record manuscript/evidence/fpga_lm_readout_oracle_2026-09-15.json \
  --prior-run manuscript/build/fpga_lm_core_bias_2026-09-14/v1 \
  --prior-record manuscript/evidence/fpga_lm_core_bias_2026-09-14.json \
  --prior-audit manuscript/evidence/fpga_lm_core_bias_audit_2026-09-14.json \
  --corpus data/tinyshakespeare.txt

/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_readout_oracle.py --audit \
  --output manuscript/build/fpga_lm_readout_oracle_2026-09-15/v1 \
  --record manuscript/evidence/fpga_lm_readout_oracle_2026-09-15.json \
  --audit-output manuscript/build/fpga_lm_readout_oracle_audit_2026-09-15/v1 \
  --audit-record manuscript/evidence/fpga_lm_readout_oracle_audit_2026-09-15.json
```

Numerical execution remains conditional on completed synthetic tests, review,
input-only binding, exact source/dependency identities and output/disk admission.

#### Readout-oracle final admission (2026-09-15 UTC)

Source/math/lifecycle review passes. The final focused suite has **87 passes
in 1.37 seconds**; Ruff and diff checks pass. The input-only invocation exits 0
after verifying all **82 source identities**, the core predecessor's **1,158
payloads**, the **1,115,394-byte corpus**, and the actual complete core checkpoint
256. Instrumented model/gradient/optimizer/subprocess entry points record zero
attempts. All four declared output paths are absent. Exactly **1,097** new
producer payloads are expected. Admission observes **1,985,335,720 KiB free**,
above the 8 GiB combined cap plus 4 GiB reserve. The latest full regression is
running separately; the preceding unchanged-source full suite passed 4,987
tests, and the new 87-case focused suite is complete.

| Frozen project input | SHA-256 |
| --- | --- |
| Readout helper | `c36800534a8e55bc8d572630218238ec7fd1d99c1ded186b5a493b258632ccd7` |
| Readout runner | `ebf25237e156fc4688a6153ee5ce763522f98aaef5980ee997ff8c5a7f7cac8c` |
| Readout tests | `367d1189692f8e1ecb8edb01cc87243a51922e19bb50a43b7cc13484ebac4253` |
| Frozen-core state256 NPZ (978,077 bytes) | `960e4831a590a314d9db76fc6655c3c4f269433a3a3b822f04c94de22a7b649b` |

All old scientific files remain byte-preserved. Runtime is Python 3.11.5,
Torch 2.10.0, NumPy 1.26.4, SciPy 1.17.1 and threadpoolctl 3.6.0, CPU FP64 with
one Torch/BLAS thread; the manifest additionally pins installed optimizer files
and actual loaded BLAS libraries/configuration. Admit exactly the single
producer and, after actual exit 0, its single independent endpoint audit using
the commands above. Keep all 82 sources and this final saved declaration
unchanged until both attempts terminate. No adaptive rerun is admitted.

#### Readout-oracle completed review (2026-09-15 UTC)

Both admitted invocations reach actual exit 0: producer at **14:59:39 UTC**
and independent audit at **15:07:51 UTC**. Final file-only verification passes
all 82 unchanged sources, 1,097 payloads (941,551,289 bytes), the exact
three-file audit capsule, original trace/solver endpoint bindings, runtime
pins and the common saved declaration
`b083c16d7885fe33a27e51ca49e18c3f17351cf003fa227a183dbcaefd2c45ab`.
All new retained private/public evidence totals 973,963,972 bytes. This releases
the live declaration freeze; the saved declarations and all scientific inputs,
code and evidence remain immutable. This paragraph is a post-run review,
not a change to the protocol that produced the evidence.

| Readout | Sampled-training CE | Development CE | Optimization outcome |
| --- | ---: | ---: | --- |
| Original frozen-core head | 2.153430163 | 2.598400205 | Historical reference |
| Bias correction | 2.149490148 | 2.590784861 | Numerical target met, 15 iterations / 18 full evaluations |
| Affine correction | 2.061243836 | 2.573624209 | 200-iteration cap / 212 full evaluations; inconclusive |
| Prior-plus-GRU | 1.948047807 | 2.371154387 | Historical reference |

The richer correction reduces CE beyond bias by 0.088246313 / 0.017160652 nat,
and all endpoint gradients/features/scores independently verify. But its
gradient-based numerical gap estimate is 166.893956966, not <=1e-5; that is
neither its actual remaining gap nor proof of how much improvement remains.
The accepted objective is still decreasing at the cap. Affine clears both
raw 0.01-nat improvement clauses but lacks numerical completion; bias has
numerical completion but clears neither quality clause. Neither satisfies all
completed-progress prerequisites. No recipe is promoted.

The quality gap is not closed. The earlier extra-pass mixed result is also
marginally better than this affine endpoint, with a different history/budget;
this is not a new overall best. A positive residual-readout gain does not
isolate contextual information or a unique bottleneck, since the larger
family can also change effective calibration/regularization costs. An
unknown affine optimum cannot establish a backbone capacity limit. See the
[full audited readout evidence](EVIDENCE.md#frozen-feature-residual-readout-oracle-2026-09-15)
for all comparisons, solver losses versus sampled scores, histories, timing,
exposure counts, hashes and limitations. The full regression has 5,074 passes,
14 skips and 10 deselected, plus one expected warning.

**Next question for review, not admitted or run:** use a fixed training-only
feature-Gram preconditioner to optimize the *same* affine objective within a
new, separately declared solver budget. An invertible coordinate change must
preserve the original logits, zero start and original-coordinate ridge penalty;
ordinary feature normalization with a new isotropic penalty would change the
question. Independently check the transform and gradient pullback, and judge
the numerical target in original coordinates. Reuse the frozen cache and bias
control, without new backbone training or cohort access. The concrete algebra
and its caveats are in the evidence note. No extra solve, changed lambda,
hardware scaling or paid execution follows automatically from this proposal.

### Same-objective Gram-preconditioned readout v1 (declared 2026-09-15 UTC)

The user's subsequent request to keep going authorizes this next bounded
conditioning experiment. It does not retrospectively change the completed
readout-oracle protocol above. Preserve all 82 historical scientific sources,
raw evidence and saved declarations. Add a new helper, runner, independent
receipt validator and tests (86-source closure); do not patch the old solver.

**Question:** can fixed training-feature conditioning improve the optimization
of exactly the previous affine correction problem within the same iteration/
full-gradient caps? The previous affine endpoint hit 200 iterations, so its
near-optimality is unknown. Its large gradient-based estimate is not an actual
remaining loss gap. No backbone capacity conclusion follows from it.

Reuse only the independently audited cache from
`fpga_lm_readout_oracle_2026-09-15/v1`: train windows 0–1023, sampled-training
windows `range(0,1024,16)`, development windows 88–151, 128 positions each.
The frozen-core state256, D512/V128/N4/K7 geometry, actual context-ON `h`/`z0`,
data, targets and all previous controls are unchanged. The binder checks the
actual pinned producer publication, all 1,097 sealed payloads, successful
independent audit and exact three-file audit capsule, 82 historical sources,
saved historical declaration and canonical core/checkpoint/corpus ancestry.
The current PLAN is intentionally different from the historical saved PLAN;
never rewrite an old declaration to satisfy a new run. Load cache arrays and
check their geometry, hashes and token/target alignment without re-executing
the backbone. No new distinct training pairs, backbone forwards/gradients/
updates, reserved development or final-test access are allowed.

Let `X=[h,1]`, `A=[W,b]`, and fix `lambda=0.001`. On all and only training
features form `M=lambda*I + X.T@X/(2*N)`. For lower Cholesky `M=L@L.T`, set
`S=L^(-T)`, and optimize `U` with physical coefficients `A=U@S.T`.
The gradient pullback is `G_U=G_A@S`. The physical archive packs `W.ravel()`
followed by `b`; this differs from flattening augmented `A`. All new coordinates
start at positive zero, not at the previous capped endpoint.

Every objective evaluation first materializes physical `W,b`, then uses the
unchanged `z0+h@W.T+b` operation order and original objective
`mean CE + lambda/2*(||W||_F^2+||b||_2^2)`. No feature centering, rescaled penalty,
new beta, new lambda, warm start, damping search or parameter sweep is admitted.
In particular `lambda/2*||U||^2` would be a different objective and is prohibited.
In exact arithmetic softmax covariance has spectral norm at most 1/2, giving
the block curvature upper bound `I_vocab tensor M`, and `S.T@M@S=I`.
This does not bound the lower curvature close to one or guarantee convergence.

Use CPU FP64 with one Torch/BLAS thread, chunks of 2,048 positions, SciPy
L-BFGS-B with `maxcor=10,maxls=20,ftol=0,gtol=0`, 200 accepted iterations,
400 attempted full objective/gradient evaluations and 3,600 seconds per solve.
Zero `gtol` disables premature success from a transformed-coordinate gradient
threshold; a custom check stops on the ORIGINAL physical full-gradient estimate
`||G_A||_F^2/(2*lambda) <= 1e-5`. This remains a numerical estimate, not a formal
roundoff-certified certificate. Return only the last fully evaluated accepted
point. Count attempted/completed calls separately, retain scalar/hash history
and accepted references, and label caps/other stops above the target inconclusive.
No automatic extension or restart follows a cap. The per-solve deadline is
checkpoint-enforced around chunks and callbacks; the global watchdog covers
in-flight native operations. The Gram construction/factorization cost is outside
the solve timer and reported separately; the global bound includes everything.
Count four producer training-Gram passes: construction, pre-solve validation,
post-fit saved-output validation and ordinary readback. The independent audit
adds one Torch training-Gram pass. Only the construction chooses a factor;
the remaining passes verify the fixed factor. Each producer initial/final
endpoint is checked twice (four full-gradient queries total), and each is
independently checked once (two audit queries), separate from solver work.
Pin the installed triangular-solve Python and LAPACK files in addition to
the inherited optimizer/runtime identities.

Retain `M,L,S`, their construction receipt, positive-zero and final `U`, physical
coefficients, original and transformed gradients, solver receipt, all endpoint
objectives/CE/penalties and paired fixed-window scores. Ordinary readback verifies
saved arrays and scores without refitting. A separate independent invocation
recomputes the training Gram with Torch, checks the saved factor against that
matrix, and verifies physical materialization, gradient pullback and initial/
final original objective/gradient through Torch autograd. Independent scores
use Torch rather than NumPy matrix multiplication. Direct equality of two
ill-conditioned inverse factors is not required: compare Gram matrices at
`atol=1e-9,rtol=1e-11`; require finite symmetric matrices, lower-triangular
positive-diagonal Cholesky factor, normalized factor/inverse backward residuals
at most `64*(D+1)*eps`, and absolute inverse residual at most `1e-6`.
Report whitening/conditioning diagnostics without claiming exact FP64 identity.
Endpoint comparisons use `atol=1e-9,rtol=1e-11`; data/discrete fields and
positive-zero initialization are exact. Do not claim independent replay of
unretained optimizer trial arrays or a new reconstruction of inherited features.

Reuse the completed bias-only and unpreconditioned affine controls, original
frozen-core head, original mixed and GRU references without refitting. Report
regularized full-training objective separately from raw sampled-training and
development CE/accuracy. Compare new-minus-old-affine, bias, frozen core and
GRU by paired windows. Same iteration/call ceilings do not imply matched actual
calls or wall time; count Gram and verification work separately. Improvement
on the repeatedly used development slice is exploratory, not fresh confirmation.
Even numerical convergence would establish only this fixed regularized affine
problem, not a unique backbone bottleneck. The multiply-heavy affine correction
is a diagnostic, not an FPGA-efficient deployment head or useful LLM pretraining.
No recipe promotion, hardware scaling or paid execution is part of this run.

Admit one producer and, after actual exit 0, one separate independent audit.
Each has a 10,800-second global watchdog. New private/public outputs together
have an 8 GiB cap and 4 GiB free reserve. Preserve failures, do not overwrite
or relabel attempts, and freeze all 86 sources plus the final declaration
through both actual terminal exits. Eight flat producer payloads plus their
seal are expected. The new audit retains exactly three private files. Commands:

```sh
/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_readout_preconditioned.py --run \
  --output manuscript/build/fpga_lm_readout_preconditioned_2026-09-15/v1 \
  --record manuscript/evidence/fpga_lm_readout_preconditioned_2026-09-15.json \
  --prior-run manuscript/build/fpga_lm_readout_oracle_2026-09-15/v1 \
  --prior-record manuscript/evidence/fpga_lm_readout_oracle_2026-09-15.json \
  --prior-audit manuscript/evidence/fpga_lm_readout_oracle_audit_2026-09-15.json \
  --corpus data/tinyshakespeare.txt

/usr/bin/caffeinate -is python -u manuscript/tools/diagnose_fpga_lm_readout_preconditioned.py --audit \
  --output manuscript/build/fpga_lm_readout_preconditioned_2026-09-15/v1 \
  --record manuscript/evidence/fpga_lm_readout_preconditioned_2026-09-15.json \
  --audit-output manuscript/build/fpga_lm_readout_preconditioned_audit_2026-09-15/v1 \
  --audit-record manuscript/evidence/fpga_lm_readout_preconditioned_audit_2026-09-15.json
```

Numerical execution remains conditional on synthetic tests, independent review,
input-only binding with numerical entry points blocked, absent output paths,
exact final source/runtime identities and disk admission.

**Completed outcome (2026-09-15):** producer and independent audit both exit 0.
The 200-iteration cap leaves the numerical target unmet; improved training
objective does not improve development CE. See the
[final evidence and receipts](EVIDENCE.md#same-objective-gram-preconditioned-readout-2026-09-15).
The historical declaration saved with the run is unchanged; this completion note
was added after both successful exits. No further run is declared here.

### Bounded hardware verification

Start with one `f2.6xlarge` and a 30-minute absolute deadline for verification, reserving
a **$10 pilot envelope** for compute, disk, IP and artifact overhead. The verified
2026-09-09 Linux On-Demand rate is **$1.98/hour** in N. Virginia and Oregon
([official AWS pricing feed](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json));
recheck rates, AMI software charges and actual disk sizing before launch.
Release the next training/build tranche only after correctness and resource
reconciliation. Keep at least $100 unallocated contingency below the $1,000
authority; the operational reservation ceiling is $900, not a guarantee that
billing stops automatically at that amount.

Require explicit account/profile/region, verified AMI/AGFI, existing subnet,
security group and instance profile, encrypted delete-on-termination storage,
IMDSv2, no SSH ingress, an allowlisted hashed source bundle and a private
run-specific artifact prefix. Reserve worst-case costs before launch; permit
one owned instance at a time and bounded retries. Use guest and external
controller deadlines, terminate only exact owned resources, and keep state and
reservations until termination is confirmed. Billing alerts are delayed, not
a hard cap. Include EBS, S3, requests, data transfer and lingering resources in
reconciliation; do not delete the historical research archive to cut costs.

The historical launchers do not meet these controls. Their `DRY_RUN` can upload
data, some watchdog/build checks fail open, and termination state can be lost.
The new workflow must fail closed until its launch/supervision integration is
validated. The separate `fpga_lm_control.py`, `fpga_lm_watchdog.py`,
`fpga_lm_guest.py` and explicit `fpga_lm_cloud.py` adapter implement the pilot
lifecycle. A Standard Step Functions execution must independently pass its
DescribeInstances startup check and enter its deadline wait before EC2 launch;
the guest then waits for an exact-instance, source-bound S3 start permit.
Startup health validates discovery and configuration, not the destructive
termination path. Full CSIM/PCI checkpoint comparison and observed EC2
termination are required to close the live integration test.

The live preparation order is: create and read back the scoped profile, freeze
the configuration, run resource/budget preflight against the **empty** new
artifact prefix, stage the exact inputs conditionally, then run EC2's permission
dry run. Only the reviewed controller may subsequently reserve, arm, verify and
launch. A prefix staged before that preflight is preserved and abandoned, not
overwritten or accepted by bypassing the check. Use the same cumulative ledger
for every compute attempt.

An interrupted preparation that already **passed empty-prefix preflight before
staging**, but never reserved, deployed a watchdog, issued a permit or launched,
may use the explicit `preflight-staged` resumption check. It requires exactly
the three approved source/SDK/NumPy objects and verifies actual bounded byte
readbacks, not ETags. Separately prove exact-token instance and deterministic
watchdog absence. Preserve the original configuration; only its deadline and
regenerated guest script may differ in new exclusive configuration files. The
controller must then pass its ordinary reservation, absence and arming gates.
This is not permission to extend the lifetime of an attempted run or to reuse
a historical ledger identity. Pilot G follows this path after explicit source
upload approval; its previously blocked preparation record remains historical.

Live integration exposed paginated EC2 responses with an empty first page.
The controller now follows at most ten pages within 60 seconds; the watchdog
also completes bounded ten-page scans before deciding absence or acting on a
single ownership-validated candidate. Repeated tokens, ambiguous candidates
and exhausted bounds do not establish absence. Watchdog cleanup pagination
failures retry only within its existing absolute cleanup deadline. Initial
role-assumption failures have bounded retries, without widening IAM trust;
the controller's 120-second arming window still blocks a slow/failed startup.

On 2026-09-09 AWS SSO was refreshed and read-only resource preflight passed.
The reproduction AMI declares 120 GiB root plus 10 GiB secondary EBS storage;
both mappings are explicitly encrypted and delete-on-termination. Its private
Marketplace snapshots cannot be independently described by this account, and
the AMI is deprecated; availability/compatibility must not be inferred beyond
the observed AMI response. The selected existing subnet has an internet
gateway but no NAT or VPC endpoints. This pilot therefore explicitly permits
one temporary public IPv4 with zero ingress and HTTPS/DNS-only egress, without
provisioning a NAT. A new run-only instance role can read frozen inputs and
write only its own results; it cannot authorize its own start or launch EC2.
The source bundle and public-domain development corpus are privately staged;
the AWS SDK userspace source is pinned and archived with its license.
The pre-existing historical F2 sentinel is excluded from the new controller
and budget ledger and remains untouched pending the owner's direction.

## The result we want

> At a useful level of task performance, TikiTorch makes learning cheaper or
> accessible on smaller hardware.

Prioritize FPGA language-model pretraining; ECG and software-only studies are
secondary opportunities. Compact route records are the
mechanism, not the practical result: saved-state bytes alone establish neither
lower peak memory, shorter time to quality, better quality, lower energy nor
lower cost. The existing source-bound three-cell CPU protocol in
`manuscript/tools/measure_retained_state.py` answers a narrow representation
question, not the remaining systems or modeling questions.

A separate hybrid software recipe may combine a route-trained tropical body with ordinary
PyTorch input/output layers, losses and optimizers. Charge these boundaries in
all resource measurements. That hybrid result is valid without a multiplier-free
claim, but it must not be presented as the fully on-fabric training result.

Keep these questions distinct:

| Question | Comparison or check | Evidence needed |
|---|---|---|
| Representation | Declared selected reverse and retained fields | Actual storage and scaling, including auxiliary state |
| Same-model systems utility | Eager/compiled autograd, strong checkpoint policies, unpacked/packed routes | Equivalent outputs, selected gradients and repeated updates; complete-step time, measured peak memory and tested capacity |
| Modeling utility | Tuned tropical/hybrid and conventional/low-arithmetic architectures | Held-out quality versus total time, memory and cost under controlled data and tuning budgets |
| Closure | Exact forward/loss/reverse/update recipe | No-waiver executed arithmetic audit, separate from quality and physical synthesis |

The primary practical outcome is **time to a predeclared quality target under
a fixed device-memory budget**, with quality/cost frontiers alongside it.
Select the quality target, acceptable difference, metric aggregation and
hardware budget from training/validation pilots before final test evaluation.
Report failures to reach the target, retaining their consumed budgets.

For the secondary software track, an aspirational target is about **2× lower measured peak training memory than
compiled autograd**, with competitive quality and better time to quality than
the best compatible checkpoint policy at the constrained budget. This is a
research target, not a forecast, required positive outcome or current result.
A smaller robust capacity/cost advantage may still be worthwhile. An unrelated
GPU model, failed baseline or synthetic successor task cannot establish it.

## Comparison protocol

### Freeze before final runs

Record the clean candidate Git revision and source archive/hash; environment,
hardware and driver/compiler identities; dataset license, byte hashes and
train/validation/test identities; tokenizer/preprocessing; loss, optimizer,
precision, seeds, shapes, update counts, numerical tolerances and timeout policy.
Archive exact commands and deterministic reducers. Distinguish pilot, tuning,
validation-selection and final-test runs in the manifest. Fill in the exact
dataset configuration, hardware/budget, owner and candidate identity in a
task-specific approved protocol revision before claim-bearing runs.

Use seeds **0, 1 and 2** for the initial multi-seed study and retain every
outcome. Additional seeds require an appended protocol version, not selection
of favorable seeds. Fix tolerances before performance rankings: failed route
identity or nonfinite values always fail equivalence, and optimizer/dtype
changes define new recipes. Preserve OOMs, timeouts, unsupported compiler cells,
instability and negative findings; report maximum *tested* capacity only.

### Same-model systems comparison

Start from `scripts/route_certificates/training_benchmark.py` on one existing
authorized CUDA runner, piloting the smallest shape's compile/resource costs
before a full matrix. No paid provisioning is part of this protocol.

The proposed development ladder is `(D,L,T) = (64,2,128), (128,4,512),
(256,4,2048)`, batches 1, 4 and 16, fixed task vocabulary and all declared seeds.
This is not the final ECG or pretraining configuration. Use justified task
resolutions/lengths and capacities; do not silently truncate ECG or shrink an
LM vocabulary to fit the ladder. Adjust it only in a documented pilot revision.

Compare eager autograd, native compiled autograd, eager and compiled
checkpointing (per-layer, grouped and selective where available), native compiled unpacked routes and
native compiled packed routes. Match model, initialization, input order, loss,
optimizer, precision, effective batch and update count. Fix warmups/repetitions
after the pilot and retain per-repetition values. Include packing, decoding and
optimizer time; separate compile/startup from steady-state work.

**Implementation versus evidence:** the comparator now includes both eager and
compiled checkpoint candidates and the current `amax`/`amin` save targets.
Its separate observations verify policy coverage, not physical retention or
recomputation savings. Validate the intended device/shape and measure those
effects before final rankings, preserving unsupported cells; a policy name
does not prove coverage or a systems advantage.

Compare the **measured final state** after all updates against eager, not only
a fresh one-step run. Also retain intermediate losses/gradients/parameters to
identify first divergence, and test repeated updates on fresh batches alongside
fixed-batch timings. Resolve or explicitly limit the current native compiled
AdamW discrepancy; a momentum-SGD study is a distinct configuration, not its fix.

Report complete-step latency median/dispersion, allocated and reserved device
peaks, maximum tested validated workload, route bytes, transient workspace and
compile time separately. Select the best successful checkpoint policy within
the declared allocator budget and show the whole Pareto frontier. Report
memory traffic only if measured, with the profiler boundary; neither arithmetic
counts nor allocator metrics establish energy.

Give all relevant arms compatible gradient accumulation, chunking, offloading,
mixed precision and optimized loss implementations. Report host/offloaded
memory too: moving storage is a trade-off, not elimination. Charge preprocessing,
tuning, failed runs and all soft-to-hard phases to their declared budgets.
End-to-end time includes compile/startup; a final hard phase may be reported
separately only as a labeled component, never as the whole pretraining cost.

### Useful task quality and selection

The final practical claim needs a legally usable **non-synthetic task** with
fixed held-out data. TinyStories and synthetic successor-token tests are
diagnostic, not fulfillment of this gate. A corpus needs distinct training,
validation and test partitions; the convenience runner's 85/15 split is not
a complete model-selection protocol. Freeze preprocessing/tokenization and
remove known overlap/leakage. Use validation for hyperparameters, stopping and
checkpoint selection; evaluate the untouched test split once per selected
seed/configuration, confirming the predeclared quality region.

Compare the same hard model across differentiation implementations to isolate
systems effects, and separately tuned architectures at controlled budgets to
assess modeling utility. Include a conventional dense sequence baseline and a
directly relevant morphological/low-arithmetic baseline selected from the
[literature review](LITERATURE.md), plus no-update and simple baseline controls.
Match/report parameters, examples/tokens, update and tuning budgets. Equal
parameter counts alone do not establish matched quality; unrelated published
throughput or scores are not controlled denominators.

Report each seed and uncertainty across seeds; correlated tokens or timing
repetitions are not independent training runs. If the hard model does not learn
well, preserve that result and investigate before scaling hardware claims.

### Ablations that explain the result

1. Packed versus unpacked selected state, charging decoding and workspace.
2. Several checkpoint policies, including a hybrid route/checkpoint path when
   uncovered boundaries dominate.
3. Vocabulary/top-k head size: `B*T*V*k*ceil(log2 D)` can dominate storage.
4. Depth, active-weight coverage, gradient density and route churn: a selected
   derivative does not guarantee adequate optimization coverage.
5. Hard-only versus soft-to-hard training, with each phase's separate audit/tape
   applicability; the soft phase is not activation-free or automatically closed.
6. Loss/optimizer choice and closure status, charging dense or smooth boundaries
   even when the hard body has a compact certificate.

Freeze primary hypotheses and budget first; label secondary analyses exploratory.
No outcome is assumed positive.

## Track A: useful ECG classification

### Task and inexpensive first gate

Use [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), five-label
diagnostic-superclass prediction, and recommended patient-separated folds
(1–8 train, 9 validation, 10 test). Freeze the annotation-to-label map, eligible
record IDs and hashes, and confirm the release's data terms. Never randomly
split windows from one patient across partitions.

Begin at 100 Hz; advance to 500 Hz only for a useful reason established by a
validation-only pilot. Fix preprocessing, lead order, missing-value handling
and normalization from training data only. This retrospective benchmark does
not establish clinical readiness or safety.

Use five independent logits with a multilabel loss such as `BCEWithLogitsLoss`,
not the library's default single-label cross-entropy. Select class weighting
and thresholds on training/validation data only. Proposed primary metric:
macro-AUROC, with macro-AUPRC and per-class metrics alongside it. Freeze exact
reductions and missing-label handling before final runs.

### Required implementation and decision gate

1. Separate the route-trained body from input, pooling and output boundaries.
   The current exact-class MorphSSM route adapter supports a narrower LM, not
   this ECG classifier.
2. Specify pooled-classification reverse semantics and test body/boundary
   parameter gradients against autograd, including repeated optimizer updates.
3. Compare continuous input projection with a declared quantized alternative.
   The default tropical input projection uses 32 bins over `[0, 1]`, without a
   straight-through input derivative. Raw signed ECG cannot simply use that
   default without defined scaling, clipping and quantization effects.
4. Report the actual receptive field and aggregation rule. Longer tensors
   alone do not establish longer represented temporal dependencies.

Tune conventional ECG networks, beginning with the ResNet/InceptionTime families
in the [PTB-XL benchmark study](https://arxiv.org/abs/2004.13701), alongside the
shared protocol's controls. Use the same release and split; published results
from different pipelines remain context.

First establish competitive learning in a small CPU or approved existing-device
pilot, then attempt the memory-constrained study. Smaller conventional networks,
100-Hz inputs or checkpointing that reach equal quality more cheaply count
against the claim. Do not manufacture an advantage through unnecessary length,
resolution or oversized baselines.

Deliverable: frozen multi-seed quality/time/memory evidence of more accessible
ECG learning, or a documented negative result and stop/revision decision.

## Track B: language-model pretraining

### Development pilot, then natural text

Pretrain from scratch; a final hard fine-tuning phase is not the whole result.
The proposed ladder is:

1. **10–30M parameters on
   [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** to
   investigate optimization, route scaling and head cost. This development
   milestone establishes neither general LLM competence nor natural-text success.
2. After quality/systems gates pass, **100–150M parameters on a frozen
   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
   subset**. Select the exact configuration/revision, tokenizer, deduplication/
   contamination controls and separate train/validation/test partitions first.
   Set token/tuning budgets from pilot throughput on explicitly approved hardware.

These sizes are proposed scopes, not current capabilities or approved allocations.
Freeze parameter counting, sequence lengths, tokens, vocabulary, optimizer/schedule
and effective batch. Report held-out token cross-entropy/perplexity with the same
tokenizer; cross-tokenizer comparisons need a separately specified comparable
metric. A prespecified small downstream suite belongs only at the later gate,
with its selection and cost separately recorded.

### Solve the vocabulary-head bottleneck

The current tropical top-k head retains
`B * T * V * k * ceil(log2(D))` bits before framing/alignment. At
`B=1, D=512, T=2048, V=32768, k=8`, its head alone needs **576 MiB of logical
route payload**. This shape calculation is not measured peak memory and shows
why small-vocabulary evidence is insufficient.

- **Practical first:** route-trained body, conventional linear readout and
  memory-efficient cross-entropy. Retaining the final hidden-state boundary
  may beat vocabulary-wide routes. Use an exact loss or an explicitly declared
  numerical approximation. Disable gradient filtering for exact-comparison arms,
  or label and independently validate the changed recipe. Give dense baselines
  the same loss. [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009) is a design
  reference, not an already integrated dependency.
- **Fully tropical path:** bounded/tiled readout with selective recomputation
  or another validated head representation. Charge boundaries, recomputation
  and workspace; tiling alone need not remove retained reverse information.

The hybrid adapter and scalable head remain prerequisites. Do not relax
route-adapter guards to imply unsupported coverage. `MorphSSM(head_type="linear")`
already works with ordinary autograd; the missing piece is a route-trained
hidden-state boundary, not a new dense-head model implementation.

The smallest next library extension is `route_hidden(model, ids, packed=...)`
after the final tropical normalization, followed by the ordinary linear head
and loss. Factor the existing backbone forward/VJP, keep current top-k tapes
compatible, and fingerprint hidden-boundary tapes distinctly so they cannot be
replayed as logit tapes. Initially retain the untied-parameter and exact-backbone
guards. Check packed/unpacked FP64 forward and all-parameter gradients, arbitrary
hidden cotangents, ties, dropout, multi-step updates, first-order-only behavior,
saved-tensor hooks, causal-prefix invariance and native compilation. Charge the
floating hidden boundary (`B*T*D` elements), dense head and full loss state in
every comparison; this is not a whole-model activation-free training claim.

Also measure effective context, not merely tensor length. With causal kernel
size three and dilations 1,2,4,..., the current one/two/four-layer benchmark
models have receptive fields of 3/7/31 tokens. Add a genuinely context-dependent
learning control before scaling to text. The successor-token task needs no
context, and the optional character-corpus runner has only an 85/15 split and
one sampled held-out batch; neither substitutes for tuned baselines and a
frozen train/validation/test protocol.

### Baselines and decision gate

Use same-model systems arms, and separately a tuned conventional decoder-only
transformer and relevant low-arithmetic baseline from the literature review.
Include smaller dense models for smaller-hardware claims. Control token/tuning
budgets and report quality/cost frontiers, not merely equal parameter counts.

Stop scaling if quality is uncompetitive, head memory dominates or checkpointing
removes the useful advantage. Preserve all hard-only/soft-to-hard phase costs
and their distinct closure status. Historical small-language results do not
establish matched-budget LLM pretraining success.

Deliverable: reproducible from-scratch time-to-quality/capacity evidence at a
stated scale, without presenting the pilot as a broadly capable or billion-
parameter LLM.

## Primary hardware track: FPGA language-model pretraining

This is now the primary experimental track, not a capstone deferred until a
GPU advantage. First prove fresh-window learning and evaluation on the existing
small image. Then profile the vocabulary head, forward comparisons, reverse
scatter, update scans, data transport and checkpoint costs before redesign.

Target a useful recipe whose **forward, pooling/head, loss, backward and
parameter updates all execute on FPGA**, without a GPU performing learning.
A host may orchestrate I/O/evaluation, with explicit cost; offloading embedding,
head, loss, gradient formation or updates to the host does not satisfy this
complete-learning boundary. Existing D512 hardware is a starting point, not
evidence that useful pretraining or an LLM-scale advantage is already achieved.

The existing D512 design uses roughly 7.69 MiB of its 8 MiB staging window and
does not use the F2's external HBM/DDR for model state. More instance hours alone
cannot scale it. Priorities after the quality gate are narrower/packed state,
checked bulk transfers, tiled forward/gradient/update operations and an
HBM/DDR-backed memory architecture. Reusing four independent small kernels is
not data-parallel training of one larger model; synchronized gradient/state
semantics need a separate implementation and correctness proof.

For the first external-memory redesign, choose tile/bank sizes from measured
traffic rather than parameter count alone. Preserve global first-index tie
selection across tiles, retain enough route state for the chosen reverse,
and test bank-collision accumulation and overflow/saturation explicitly. Reuse
weight tiles across positions where the causal schedule permits it; otherwise
the forward comparison scan can become bandwidth-bound even without MACs.
Separate token staging, fabric execution, route/gradient traffic, update scans
and full checkpoint I/O in the profile. Do not count narrower storage or a
larger memory capacity as higher training throughput without measurements.
Keep the existing image/reference intact while validating a new layout and
numeric contract in C-simulation before paid synthesis.

Verify these independently:

- **No data multiplication:** no-waiver executed software audit plus applicable
  synthesis/operator evidence for the frozen numeric implementation.
- **No DSPs in that learning implementation:** verified zero DSP utilization in
  its synthesized/routed hierarchy, separately disclosing shell/out-of-boundary
  resources. A multiplier-free design can use DSP adders; zero DSP use does not
  exclude multiplication implemented in logic.
- **No GPU learning:** a complete execution/host-boundary record, not simply
  absence of a GPU library import in a kernel.

Specify quantization, normalization, loss and optimizer anew and reproduce
quality in software first. Quantized training is not the floating-point route
theorem. Match the exact recipe, model, data order, update and state transition
between reference and hardware. Require full-state one-step/repeated-step
known-answer tests, disjoint training examples, independent held-out evaluation,
timing-closed reports and exact source/image/toolchain identities.

Historical appendix feasibility needs no new speed/energy claim. Expanding
those claims requires resolving frozen MNIST evaluator/KAT differences and
D512 resident-window limits, with archived HLS/routed reports and bitstream
identities. Measure full-card idle/active power under synchronized matched
workloads with uncertainty. Show host staging/readback, setup and build costs;
infer neither energy nor cost from operator counts. Toolchain, board access,
build costs and artifact rights require explicit approval.

## Sequence and required artifacts

1. **Now:** validate the new streaming driver, exact layout/transport checks,
   held-out evaluator, artifact/provenance handling and budgeted cloud controls.
2. **Existing-image FPGA pilot:** verify current AWS access and exact inputs, run
   a short full-state KAT and then bounded fresh-window training if it passes.
3. **Quality and systems gate:** establish held-out learning with simple controls
   and report host-inclusive time/resources; preserve negative results.
4. **Scale the hardware architecture:** prioritize profiled bottlenecks and
   external-memory tiling, synthesize only validated changes, and charge failed
   builds to the same cumulative budget.
5. **Larger pretraining and publication:** move to the frozen natural-text/tokenizer
   protocol only after small-model quality/correctness gates pass; ask before
   exceeding $1,000 and revisit manuscript claims using measured outcomes.

Store research evidence beneath [`manuscript/results/`](../results/README.md),
separate from the reusable library. Use append-only run IDs, raw rows and incident
records; keep historical logs/manifests immutable and labeled historical. Retain
source/environment/data identities, all comparator configurations and unsupported
cases, deterministic plots/tables with pinned input hashes, exact evidence/claim
mapping and an independent reproduction report.

Large datasets, checkpoints and board images belong in approved external storage
with durable links/checksums, not casual commits of binaries or credentials.
Follow [evidence preservation](EVIDENCE.md) and the
[release provenance gate](../../docs/release_provenance.json).

The bounded FPGA research and AWS spending described at the top are authorized;
this plan does not expand that authority to unrelated provisioning, dataset
redistribution, publication, visibility changes, history rewriting or submission.
Authorship, supervision and venue decisions remain human decisions;
release/submission requires separate authorization after scientific and rights review.
