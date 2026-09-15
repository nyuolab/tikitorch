# TikiTorch evidence and artifact policy

Scientific review: 2026-09-08; layout remeasurement and optimizer diagnosis: 2026-09-09.
This ledger records the evidence available for the combined manuscript,
including a separately recorded current-code experiment;
it is not publication clearance.
Raw artifacts outrank prose. Existing hardware and route records are preserved
unchanged; the current Python implementation must not inherit their build or
performance claims.

## Claim-to-artifact map

| Proposed claim | Evidence and configuration | Status and permissible use |
| --- | --- | --- |
| PyTorch-shaped infrastructure for tropical/hard-routed learning | `tikitorch/{functional,nn,models,optim,autodiff,audit,backend}`, `tests/test_import.py`, `docs/guide/route-training.md` | Implementation contribution. Familiar interfaces do not imply complete PyTorch API coverage or general hardware compilation. Main text. |
| Selected routes can determine both input and additive-parameter VJPs | Explicit primitive rules and graph composition; `tikitorch/autodiff/{primitives,morphssm_tape}.py`; focused route/gradient/tie tests | Conditional mathematical result plus tested implementations, not a novel gradient. Finite graph, declared derivative selection, complete auxiliary fields, supplied cotangent, and available static state are essential. Main text and proof appendix. |
| The new training path retains device-native packed routes | `tikitorch/autodiff/device_tape.py`, `morphssm_tape.py`; `tests/test_device_tape.py`, `tests/test_route_training.py` | Current implementation, not the implementation that produced the historical retained-state table. Byte-aligned `uint8` fields have up to seven padding bits per field; portable export is a separate CPU representation. Main implementation section. |
| Closure can be checked for an executed recipe | `tikitorch/audit.py`, `tests/test_audit.py`, `tests/test_closure_guard.py`, `tests/test_dyadic_backward_closure.py` | Current `strict-eager-v3` software semantics classify unknown operations as failures. An executed-path software pass is not a physical no-multiplier or zero-DSP proof. Main text. |
| Source-snapshotted device-route payload is smaller than a specified checkpoint boundary set | `manuscript/evidence/retained_state_layout_2026-09-09.json`, `manuscript/tools/measure_retained_state.py`, `manuscript/source/generated/current_retained_state.tex` | CPU regression remeasurement after relocating the helper, with its original sources now preserved in the pinned CPU snapshot; exact logits/canonical routes and selected-VJP checks. All earlier records remain unchanged. Main table. Comparator is analytical boundary storage, not measured peak or an optimal checkpoint policy. |
| Historical portable representation is smaller than a specified checkpoint-retained set | Run `529c45f4-7274-4999-b35c-c59d1db917e6`, detailed below | Historical imported representation evidence. Three batch-one FP32 CPU configurations, not a current GPU/full-step/optimized-checkpoint result. Appendix only. |
| Five-arm full-step validation exists | `scripts/route_certificates/training_benchmark.py`; `docs/codebase_audit_2026-09-08.md`; local run `14f00d16-e3ea-4622-9c80-c03d30bb259f` | Local CPU protocol smoke reported by the code audit, not a checked-in comparative research dataset. Width 16, one layer, vocabulary 11, batch 2, sequence 12, seed 0, momentum SGD, five total updates. Do not publish latency or capacity superiority from this run. Implementation/validation scope only. |
| D512 fused training executes with a matching archived observable record | `manuscript/results/fpga_silicon/d512_verify/{silicon_fused,golden_fused}.jsonl`, `BITEXACT`; AGFI `agfi-003a8a2f2b1ba7df7` | Historical silicon feasibility. The 30-update records agree on resident accuracy, weight CRC and 16 predictions. Not full-array, every-signal, or 2,000-step trajectory equivalence. Appendix; at most a bounded main-text feasibility sentence. |
| D512 maintains a live fused update loop for 2,000 updates | `manuscript/results/fpga_silicon/d512_verify/train_2000.err` and `train_2000.jsonl`; `tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp`; launcher `scripts/run_morph_lm_d512_verify_f2.sh` | Four layers, D512, vocabulary 128, seven taps, one resident 128-position window. 256,000 processed positions in 418.3963 s; log reports 612 token/s. No held-out language result, distinct-token count, final-route certification or accelerator comparison. Appendix. |
| Current fixed-point LM trains on fresh windows with frozen held-out checks | `scripts/fpga_lm_pretrain.py`; `manuscript/evidence/fpga_lm_csim_2026-09-09/` | Four local C-simulation pilots, including D512 at 1,024 updates. Independent integer predictions/LUT-gradient checks pass; current-code training and diagnostic records only. These CSIM records do not establish silicon execution, contextual-learning advantage or an LLM-scale result. See below. |
| Current fixed-point LM reproduces full C-simulation checkpoints on physical FPGA | [Pilot G](../evidence/fpga_lm_pci_2026-09-09/pilot_g.json) | D512, four layers, 64 fresh-window updates and 8,192 training token exposures. Full weight payloads match byte-for-byte at steps 0, 32 and 64. Hardware correctness only; no demonstrated contextual-learning, GPU, energy or cost-to-quality advantage. See the paired pilot below. |
| Head initialization and LUT scale repair the small model's context-use gate | [Twelve-cell head screen](../evidence/fpga_lm_head_screen_2026-09-09.json) | Two anchored-head recipes pass gradient-flow/context gates across three seeds in C-simulation. The finer-LUT mean NLL is 2.7213, better than matched unigram but worse than matched bigram. Development trainability result, not a new FPGA, useful-LLM, significance or efficiency result. |
| Longer training improves the coarse recipe but exposes a fine-scale accuracy/loss reversal | [Six-cell budget extension](../evidence/fpga_lm_budget_2026-09-09.json) | At 1,024 fresh windows, coarse mean NLL is 2.7366 and fine mean NLL is 2.9006 versus matched bigram 2.5933 on a new development cohort. Both retain context use; neither passes the paid-learning gate. CPU/C-simulation only. |
| Fabric-produced weights generalize on downsampled MNIST | `manuscript/results/predictive_coding/onfabric_train/{train,crosscheck}.txt`, dumped `trainedW0.i32`, `trainedW1.i32`; recipe report `manuscript/results/predictive_coding/ONFABRIC_TRAINING.md` | Historical small-task feasibility: 256→256→10 dyadic MLP, 16×16 inputs, batch 32, five epochs, 60,000 training/10,000 test examples. Final driver test accuracy 74.75%; independent exported-weight evaluation 74.13%. Unreconciled pipelines, not parity. Appendix. |

The theorem's operator inventory is broader than the validated production
bridge. The present `route_logits` path explicitly rejects, among other cases,
parameter aliases, hooks/forward overrides, non-reference custom backends,
non-unit residual scaling, recurrent scans, dense heads, soft routing and
higher-order differentiation. A theorem about composition does not establish
that all such graph features have been implemented in that bridge.

## Fresh-window FPGA LM development (2026-09-09)

The new [runner](../../scripts/fpga_lm_pretrain.py) has explicit CSIM/PCI
backends, a compiled layout probe, checked PCI transport, disjoint ASCII128
windows, full checkpoints and a paired **full checkpoint-byte** comparator.
CPU fault-injection tests cover PCI errors; they do not certify a physical
board. Historical 30-update checksum/sample checks are not the new streaming
correctness gate.

Four append-only runs under
[`fpga_lm_csim_2026-09-09/`](../evidence/fpga_lm_csim_2026-09-09/) use the
same source-bound integer kernel and Tiny Shakespeare corpus identified in
the [plan](PLAN.md). All run on the host CPU, not an FPGA. Each record contains
source/library/compiler identities, frozen corpus splits, commands/timeouts,
per-window metrics, full checkpoint digests and independent frozen prediction/
loss-gradient checks. Large checkpoint payloads remain in ignored local build
storage, explicitly marked **nonportable**; the JSON alone is not a downloadable
checkpoint bundle. Rebuild/rerun at fresh paths or archive the payloads in
approved private storage before an independent checkpoint comparison.

| Local run | Fresh updates / training exposures | Validation windows | Initial → final LUT-normalized NLL |
|---|---:|---:|---:|
| `d16` | 8 / 1,024 | 2 | 4.8520 → 4.8219 |
| `d512` | 2 / 256 | 1 | 4.8520 → 4.8520 |
| `d512_64` | 64 / 8,192 | 2 | 4.8520 → 4.6302 |
| `d512_1024` | 1,024 / 131,072 | 8 | 4.8520 → 3.4087 |

This is **LUT energies normalized before gradient truncation**, not ordinary
softmax perplexity or the exact derivative of that diagnostic NLL. The numeric
recipe uses thresholded integer SignSGD (`lr=1`, `threshold=2048`, `zshift=4`,
seed 0), not the CPU AdamW mitigation. The final test partition is untouched.
No saturation or rounded-zero target mass was observed in these validation
windows. Their CPU timings are incidental, host-inclusive scoped measurements,
not a controlled systems comparison or projected FPGA throughput.

The longer D512 run improves validation NLL, but has **not demonstrated useful
context learning**. Its final accuracy is 14.94%; disabling all non-current
temporal taps leaves final accuracy and NLL unchanged on the eight evaluated
windows. Train-fitted unigram and bigram NLL are 3.3411 and 2.4349, respectively.
Those controls fit the **entire** training partition, whereas the model saw
only the first 1,024 windows: this is a sanity check, not a matched-token-budget
ranking. Do not infer that learning can never work, or that loss improvement
alone demonstrates context use. Keep a short silicon correctness pilot distinct
from substantial pretraining scale-up; diagnose context/input sensitivity and
the quantized head/update recipe before the latter.

### Checked input-sensitivity and head-routing diagnosis

The selected
[checked diagnostic](../evidence/fpga_lm_csim_2026-09-09/d512_1024/context_diagnostic_checked.json)
(SHA256 `79f685ddb920b84a9027e98d01fa34181636f0c0929f6e0d7346f931e4443a02`)
uses the final D512 checkpoint and the same eight validation windows, without
training. Its [reproducer](../tools/diagnose_fpga_lm_context.py) leaves the frozen
training runner/kernel unchanged. The earlier `context_diagnostic.json` is
preserved; the selected version adds full loss-gradient validation of the
independent head observer and stricter input/artifact checks.

- Removing temporal taps, replacing all inputs with spaces, or reversing the
  input tokens leaves all 1,024 predictions and 131,072 rounded loss-gradient
  entries unchanged per variant, with targets fixed. These counterfactuals
  test sensitivity; they are not alternative language-quality benchmarks.
- On **949/1,024 positions (92.68%)**, all 128 vocabulary rows select one shared
  hidden feature. The balanced integer loss gradient sums to zero, so its
  scatter gives **exactly zero head-input VJP at that position**. The remaining
  75 positions have nonzero head-input VJP; future causal positions may still
  send gradient to earlier states. This is not evidence that all backbone
  gradients vanish or that the head never learned during the trajectory.
- Equal rounded outputs do not prove equal raw logits or universal input
  independence. The evidence identifies a specific failure mode at this
  checkpoint, not the unique cause of all modeling limitations.

This diagnosis motivated the separately declared head-route diversity,
initialization and logit/LUT-scale screen [below](#head-routing-repair-screen-v1),
with matched-data-budget simple controls. An epsilon change cannot recover an
exactly cancelled gradient. Preserve
this negative result and its frozen source; do not call a changed recipe the
same experiment or extrapolate its CPU timings to hardware.

```bash
python manuscript/tools/diagnose_fpga_lm_context.py \
  --run manuscript/evidence/fpga_lm_csim_2026-09-09/d512_1024 \
  --output manuscript/build/context-diagnostic-reproduction.json
```

This requires the recorded corpus and checkpoint payloads; after a fresh clone,
first reproduce the training run at a new path and pass that directory instead.

### Live cloud integration: failures are evidence, not training results

The append-only [pilot integration records](../evidence/fpga_lm_pci_2026-09-09/)
separate cloud/bootstrap validation from scientific results. The first three
attempts on 2026-09-09 failed before a paired current-code silicon result:
AWS CLI S3 permit-readback argument handling (A), missing system NumPy (B),
and a historical Ubuntu 20.04/Python 3.8 runtime incompatible with the initially
prepared wheels (C). All three launched instances were independently confirmed
terminated, with their disposable disks and network interfaces absent.
The subsequent watchdog startup attempt (D) failed before any EC2 launch.
Preparation E was abandoned after an out-of-order input upload correctly failed
the empty-prefix preflight. Pilot F passed pinned runtime/SDK bootstrap, then
failed before native compilation because Python 3.8 supplied a relative
`__file__`. Its instance, both exact disposable disks and pilot-group interfaces
are confirmed gone. Absolute runner invocations fix that issue without changing
the scientific sources; the regression also builds a real D16 CSIM library.

Preparation G initially paused for explicit source-upload approval. The owner
subsequently approved it; the expired, never-executed configuration was preserved
and a fresh configuration passed staged-input/resource checks before normal
controller arming and launch. The successful paired result is described below.
Six reservations now total **$60 retained**, not an invoice. The historical
sentinel is still running and untouched outside this ledger.
All five newly launched pilot instances are confirmed terminated. After G's
termination, both exact disposable disks, all pilot-group network interfaces
and active new-project instances were absent. Its controller correctly left
cleanup unconfirmed after the short wait; explicit recovery recorded the later
termination before reconciliation. Successful training was not mistaken for
completed cloud cleanup.

The corrected runtime recipe pins a licensed offline CPython 3.8/NumPy 1.24.4
wheel for this deprecated development AMI. It does not expand the library's
Python support. Controller discovery must process continuation pages before
claiming absence, and watchdog startup must succeed before launch; neither
an empty first page nor an attempted API call establishes the required state.
See the [EC2 discovery contract](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_DescribeInstances.html).

Prior reservations remain counted after cleanup; actual AWS invoicing is pending.
Watchdogs for A/B/C/F independently completed after observing already terminated
instances. That validates their deployed discovery/terminal path, **not an actual
watchdog-issued termination**. The first full-checkpoint CSIM/PCI gate now passes;
useful contextual learning and an end-to-end cost advantage remain separate,
still-open gates.

### Current-code FPGA checkpoint-equivalence pilot

[Pilot G](../evidence/fpga_lm_pci_2026-09-09/pilot_g.json) executed the frozen
D512, four-layer, 128-symbol character model on a physical F2 FPGA using
`agfi-003a8a2f2b1ba7df7`, with the expected loaded shell verified before and
after execution. The 1,196,032-parameter fixed-point model completed **64 fresh
128-token updates (8,192 training exposures)**. Its complete 4,784,128-byte
weight payloads match newly compiled C-simulation **byte-for-byte at updates
0, 32 and 64**. The final payload also matches the actual earlier local
`d512_64` checkpoint, SHA256
`453cf85d733f1e495afd76cf2269abf2a35429808a3e62511265c93d2c82f331`.
This establishes those checkpoint comparisons, not every intermediate training
signal or a general PyTorch-to-FPGA export equivalence.

Both backends recorded three frozen-weight evaluations on two validation windows
each, checking 768 native predictions and 98,304 rounded loss-gradient entries
against the independent integer reference per backend. The source-bound checks
passed. Raw native prediction/gradient arrays were not retained, so the later
artifact audit rechecked coverage, checkpoint bytes and recomputed metrics,
not a second direct comparison of those raw arrays. All 35 manifest-covered
artifact files, source/corpus identities and paired protocols were independently
verified. This audit is not a second cold-start execution by an outside group.

The recorded PCI update loop, including its I/O, took **14.0365 s**; its runner
took **31.3252 s** (583.6 and 261.5 training tokens/s respectively). The paired
serial C++ simulation took 35.7482 s and 42.6893 s. These are descriptive timings
from one correctness run, **not an optimized CPU/GPU comparison**. Runner timing
excludes provisioning, boot, dependency downloads/build, AFI loading, initial
corpus hashing and final artifact transfer; it is not cloud end-to-end cost.
No new DSP utilization, power or cost-to-quality claim follows from this run.

Validation LUT-normalized NLL improves **4.8520 → 4.6302**, with final accuracy
16.80%, on only 256 validation tokens. Removing temporal context produces the
same final NLL and accuracy. The final test partition remains untouched.
The useful-context-learning gate remains unmet for this frozen hardware recipe.
The subsequent CPU head-repair screen and budget extension below pass that gate
for changed initialization/LUT recipes, but not the matched-bigram gate. Longer
FPGA training remains gated; the next diagnostic examines confidence at fixed
checkpoints without changing training or the original evaluation scores.

The private S3 bundle and SHA256 are in the pilot record; large binary checkpoints
are not added to the library repository. The bundle contains portable relative
checkpoint names, source/build/environment facts, metrics and complete payloads.
After checksum-verified safe extraction, rerun the existing local comparator:

```bash
python scripts/fpga_lm_pretrain.py --compare-runs \
  manuscript/build/fpga_lm_pci_2026-09-09/pilot-g/csim \
  manuscript/build/fpga_lm_pci_2026-09-09/pilot-g/pci
```

Fresh clones need authorized access to the recorded private artifacts, or a new
execution with the documented inputs and hardware; the evidence JSON alone does
not contain the checkpoint payloads.

### Head-routing repair screen v1

The [source-bound screen record](../evidence/fpga_lm_head_screen_2026-09-09.json)
covers **four fixed arms × three paired seeds**, all completed without a failed
cell. Every run used the same first **256 fresh training windows (32,768
supervised pairs)**, the D512/four-layer/128-symbol fixed-point model, integer
SignSGD `lr=1`, threshold2048 and balanced LUT loss. Initial non-head weights
match byte-for-byte across paired arms. The two anchored arms also share their
entire initial weight payload; only the LUT index shift differs.

Initialization and update256 were evaluated on validation windows **8–23**:
2,048 distinct held-out positions, reused across checkpoints and models.
Unigram/bigram controls use exactly the same 32,768 supervised pairs, with
`alpha=1`, no extra training bytes and no transitions invented across windows.
Their NLLs are **3.225409** and **2.621001**. The final test split is untouched.

| Recipe | Mean final LUT NLL | Final no-past minus intact NLL, seed range | Mechanism gate | Context gate |
| --- | ---: | ---: | --- | --- |
| Original head, shift4 | 4.060364 | 0.0000–0.0058 | Fail | Fail |
| Wider head, shift4 | 3.817365 | 0.0253–0.0338 | Pass | Fail |
| Distinct anchors, shift4 | 2.979152 | 0.0347–0.0433 | Pass | Pass |
| Same anchors, shift0 | 2.721279 | 0.1840–0.3277 | Pass | Pass |

Each pass requires **every seed** to meet the prespecified conditions; the
three-seed means are descriptive, not significance tests. The wider head
preserves gradients and yields some context sensitivity, but fails the unigram
quality condition. Both anchored recipes beat unigram by the required margin,
and their intact-context advantage grows from initialization in every seed.
The best recipe's per-seed final NLLs are **2.741204, 2.708303 and 2.714329**.
It still trails matched bigram by **0.100277 nats/token in the seed mean**.
**No arm passes the larger-paid-learning benchmark gate.**

The intervention changes the optimization behavior, not the native kernel.
For a balanced loss gradient, routing all classes to one feature cancels the
head-input cotangent exactly. All anchored runs instead retain **128 distinct
head winners at every evaluated position**, with no zero head-input VJP on
positions having nonzero logit gradients. No forward int16 clipping was observed
in their validation checks. The finer-LUT finals have **3, 2 and 0** rounded-zero
target masses out of 2,048 positions; these are reported separately, not silently
treated as exact softmax probabilities. This loss metric is the normalized LUT
distribution before gradient truncation, not conventional softmax perplexity.
Validation checks do not instrument every native training arithmetic operation.

The observer follows saved routes backward through temporal accumulation,
arithmetic-shift centering and residual skips. Projection-branch cancellation
remains uneven across layers and seeds; it does **not** imply zero whole-layer
or parameter gradients, nor does it consistently track final quality. The
independent D16 native regression matches every saved route, prediction, loss
gradient and parameter-gradient word across six head/scale configurations.
The observer reproduces the kernel's straight-through clipping surrogate;
it is not a derivative of integer rounding or clipping.

The full run retains **24 complete checkpoints** and raw native predictions and
rounded loss gradients, including all training outputs. Its artifact manifest
covers **1,338 files / 358,762,475 bytes**. The offline audit rehashes every file
before and after recomputation and checks the actual compiled binary's
source/layout identity. From the checkpoints it recomputes **49,152 validation
predictions and 6,291,456 loss-gradient entries**, all equal to saved native
payloads, together with context probes and per-layer observations. These are
repeated evaluations, not that many distinct validation tokens. Saved-route
checks retain hashes, not raw route arrays. Training arrays are hash/coverage
verified, not independently replayed through all 256 updates. This audit shares
the host observer with the screen and is not an outside-group reproduction.

All eight scientific sources from Pilot G remain unchanged. The improved recipe
uses host-staged initialization/configuration compatible with the existing ABI,
but **has not yet been run on physical FPGA**. No new AWS instance, accelerator
speed, energy, DSP-utilization or cost-to-quality result is asserted. Binary
artifacts remain local and ignored; the checked-in JSON alone is not the full
reproduction bundle. Commands are in the [handoff](../README.md#reproduce-the-local-head-routing-screen).

The subsequent [1,024-window CPU budget extension](PLAN.md#cpu-learning-budget-extension-v1-completed)
of both anchored recipes is now complete, with its separate cohort and
matched-budget baselines recorded below. It does not replace the v1 screen.

### CPU learning-budget extension v1

The [audited six-cell extension](../evidence/fpga_lm_budget_2026-09-09.json)
restarts both anchored recipes from initialization for seeds **0,1,2**, with
**1,024 fresh windows / 131,072 supervised pairs per cell**. All six cells reuse
those same training pairs; this is not six times as much distinct training data.
Model geometry, native sources, integer SignSGD and initialization are unchanged.
Evaluations at **0,256,1,024** use validation windows **24–39**: 2,048 distinct
positions, reused across checkpoints and cells. The final test remains untouched.

Controls are fitted separately to the model's exact supervised pairs at each
trained checkpoint: 32,768 pairs at update256, 131,072 at update1,024, vocabulary
128 and `alpha=1`, without invented cross-window pairs. Their unigram NLL is
**3.312369 → 3.308974** and bigram NLL **2.666332 → 2.593301**. Initialization is
a no-update control, with no fitted-baseline comparison. The following curves
are on the **same new cohort**; absolute comparisons against v1's windows8–23
would not be paired observations.

| Trained recipe | Mean NLL, update256 → 1,024 | Mean accuracy, update256 → 1,024 | Final no-past minus intact NLL, seed range |
| --- | ---: | ---: | ---: |
| Anchors, shift4 | 3.063506 → 2.736624 | 20.26% → 27.67% | 0.0964–0.1336 |
| Same initial anchors, shift0 | 2.804498 → 2.900561 | 27.41% → 29.49% | 0.4832–0.4997 |

Both recipes pass the fixed mechanism/context gates in every seed, but
**neither passes the matched-bigram or larger-paid-learning gate**. Their final
mean NLL deficits to bigram are **0.143323** and **0.307260** nats/token.
Both beat bigram's **25.73%** accuracy in every seed, but accuracy is not the
prespecified language-quality gate. Means and seed ranges are descriptive,
not statistical-significance results or useful-LLM evidence.

For every fine-scale seed, NLL worsens even as accuracy improves. Rounded-zero
target masses rise from **7,3,0** to **34,24,30 / 2,048** positions. This refers
to the mass **before** target-gradient balancing: the normalized LUT diagnostic
probability remains positive and the target gradient need not vanish. There is
no observed validation int16 clipping or head-input gradient cancellation.
Projection-branch cancellation often falls while NLL gets worse; it does not
establish the remaining causal bottleneck. A larger no-context loss difference
also need not mean better intact-model quality.

The audit verifies **990 artifact files / 527,702,599 bytes**, all **18 complete
checkpoints**, and six actual update256 payloads byte-equal to the corresponding
v1 checkpoints despite the changed frozen validation cohort. It recomputes
**36,864 saved native predictions and 4,718,592 rounded loss-gradient entries**,
plus context controls and **155,713,536 route-digest entries**, with agreement.
These totals cover 288 repeated checkpoint/window evaluations, not additional
distinct held-out data. Full artifact/source hashes are checked before and
after; both baseline budgets, paired initialization and final gates are
recomputed. Training arrays are hash/coverage checked, not independently
replayed through every update; route arrays are represented by saved digests.
The audit shares the frozen host observer and is not an outside reproduction.

No AWS operation or new FPGA execution occurred. The original kernel and
physical-pilot scientific sources remain unchanged; improved initialization
and LUT settings still require physical checkpoint verification. Binary payloads
stay ignored locally; the JSON does not contain a portable checkpoint bundle.
Reproduction commands are in the [handoff](../README.md#reproduce-the-local-head-routing-screen).

The accuracy/loss reversal motivated the separately declared
[post-hoc confidence diagnostic](PLAN.md#post-hoc-fixed-checkpoint-confidence-diagnostic),
now completed below. Rescaling on the same development data cannot establish
independent generalization or a repaired training recipe.

### Fixed-checkpoint confidence diagnostic

The [complete diagnostic record](../evidence/fpga_lm_calibration_2026-09-09.json)
retains all **60** combinations: both trained recipes, three seeds, checkpoints
256/1,024, and evaluation-only LUT shifts **0,1,2,3,4**. All use the same
development windows24–39 and immutable weights. No native library, training,
FPGA, AWS or final-test evaluation is executed. Changing the evaluation shift
changes the normalized LUT distribution, not logits, routes or raw-logit argmax
accuracy. The table reports every grid shift, averaged across the three seeds:

| Trained shift / checkpoint | Eval shift0 | Eval shift1 | Eval shift2 | Eval shift3 | Eval shift4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 / 256 | 7.951358 | 6.618644 | 4.748973 | 3.377541 | 3.063506 |
| 4 / 1,024 | 7.229253 | 6.290984 | 4.760074 | 3.247369 | 2.736624 |
| 0 / 256 | 2.804498 | 3.062071 | 3.742249 | 4.254216 | 4.546504 |
| 0 / 1,024 | 2.900561 | 2.906081 | 3.457216 | 4.024788 | 4.409251 |

**No fixed rescaling in this grid repairs the matched-bigram gap.** Fine-model
final shift1 improves NLL slightly for seeds0/2 but worsens it for seed1, leaving
the mean slightly worse. Its mean maximum probability drops **0.3871 → 0.1650**
with unchanged mean accuracy **0.2949**: reducing severe wrong predictions also
weakens confidence on correct predictions. This does not rule out every finer
calibration method, but it is not a demonstrated calibration fix or a new
training result. Historical gates remain unchanged and closed.

The fine recipe's original-shift severe-error accounting is informative.
Across the same 2,048 validation positions repeated for three seeds, rounded-zero
target events rise **10 → 88 / 6,144 seed-position observations**. Their
contribution to mean NLL rises **0.018886 → 0.167318**, while the complementary
group's contribution falls **2.785612 → 2.733243**. The groups change membership;
this is aggregate loss accounting, not a causal treatment effect or 6,144
distinct validation tokens. Original-shift p99 token loss rises from
**8.26–8.97** to **10.93–11.46** nats across seeds. Actual target-index clipping
(`>255`, distinct from the boundary `==255`) rises from zero to **61** events.
The LUT's rounded energy floor can begin below the last index; clipping counts
are not a complete count of minimum-energy events. Named tail groups overlap,
so their loss contributions must not be added as disjoint categories.

Before evaluating alternative shifts, each of the **192** original-shift
checkpoint/window results matches both the frozen integer reference and actual
saved native payloads: **24,576 predictions and 3,145,728 gradient entries**.
The completed extension audit, full artifact/checkpoint hashes, executable
dependencies, original v1 linkage and corpus are checked before and after.
Per-window logit/prediction hashes are invariant across evaluation shifts.
The record retains per-window scores, pooled token quantiles and subgroup
contributions with both subgroup and all-token denominators. The calculation
is independently code-reviewed and scalar-oracle tested, not independently
reproduced by an outside group. Commands are in the
[handoff](../README.md#reproduce-the-local-head-routing-screen).

This narrows the next experiment to a **training intervention**, with fixed
intermediate LUT scales evaluated prospectively before changing projection
geometry. Post-hoc confidence checks cannot predict those training trajectories.

### Intermediate training scales v1

The [declared study](PLAN.md#intermediate-scale-training-control-completed)
completed all 12 CPU/C-simulation cells: six new shift1/2 runs and six immutable
shift0/4 checkpoint controls, three seeds each. All evaluate validation40–55,
at updates0/256/1024 and their original training scale. These absolute losses
must not be compared directly with the earlier cohorts' scores.

| Training scale | Mean NLL at256 | Mean NLL at1,024 | Final deficit to matched bigram |
| --- | ---: | ---: | ---: |
| Shift0, historical weights | 2.901671 | 3.105856 | 0.445703 |
| Shift1, new training | 2.749688 | 2.862251 | 0.202098 |
| Shift2, new training | 2.805720 | 2.783547 | 0.123394 |
| Shift4, historical weights | 3.142680 | 2.827947 | 0.167794 |

The matched bigram uses 32,768 pairs at 256 (**2.726570**) and 131,072 at 1,024
(**2.660153**). Every recipe passes the unchanged mechanism/context gates;
**none passes bigram or larger-paid-learning gates**. Shift2 is more stable
than shift1 in this bounded budget, not a successful quality-gap closure.
Historical controls perform zero new training updates; their checkpoint
training history must not be counted as new execution.

The [completed deep audit](../evidence/fpga_lm_scales_2026-09-09.json) verifies
2,028 files totaling 648,166,362 bytes, 36 full checkpoints, 18 exact historical
checkpoint matches and six exact new-initialization matches. It recomputes 576
checkpoint-validation windows: 73,728 native predictions, 9,437,184 logit-gradient
entries and 311,427,072 route-digest entries. There are 6,144 new updates and
786,432 new training-pair exposures across the six runs, but 131,072 distinct
supervised pairs. Training arrays are coverage/hash checked, not every training
update independently replayed; the checkpoint observer is shared host code,
not an outside scientific reproduction.

Documentation erratum in the frozen scale helper: fresh control-checkpoint
processes provide conservative isolation, but are **not required to avoid stale
resident weights** in mode3. The kernel reloads supplied weights and clears
gradient arrays on every mode3 call. The recorded control protocol remains
valid; its executable source is preserved for reproducibility.

### Bigram representability control

The [completed record](../evidence/fpga_lm_bigram_capacity_2026-09-09.json)
constructs the same alpha1 bigram distribution inside the frozen D512 integer
architecture. Four fixed shifts0/1/2/4 give identical NLL **2.660464**, versus
floating bigram **2.660153**: quantization costs only **0.0003113 nats/token**
on this cohort. There are 22 top1 prediction changes caused by quantized ties;
accuracy is 26.0254% versus 26.0742%. No clipping is observed. The largest
constructed LUT index is 189, below the 255 clamp.

All four full checkpoints preserve their exact host-fitted weight bytes through
native `lr=0` evaluation. The checked 291-file, 60,200,629-byte raw tree covers 64
checkpoint-validation windows, 8,192 saved predictions, 1,048,576 gradient entries
and 1,048,576 head-route entries over 2,048 distinct positions. Two frozen host
forwards and direct embedding logits agree with the native arrays; context
removal preserves the constructed distribution. This is a **host count-fitting
and representability diagnostic**, not gradient learning, contextual improvement,
FPGA execution or a passing quality gate. It makes a basic bigram representation
limit unlikely; it does not prove which optimization repair will succeed.

The follow-up [integer optimizer protocol](PLAN.md#integer-optimizer-repair-v1-completed)
retains this checkpoint as an explicitly charged warm-start control. Existing
native/physical evidence stays frozen; the proposed magnitude-aware integer
update needs a new optimizer/state interface and FPGA image.

### Fixed-gradient optimizer diagnosis

The [training-only diagnostic](../evidence/fpga_lm_sign_bias_2026-09-09.json)
keeps the constructed checkpoint fixed while accounting for all 1,024 training
windows. Among 976 anchored embedding coordinates with nonzero accumulated
threshold2048 SignSGD drift, **798 oppose the aggregate fixed surrogate
gradient**; their cosine is **−0.154087**. Shared head coordinates have 18/128
opposed signs and cosine +0.332238; each layer's bias repeats that head pattern.
This is a measured finite-sample distortion at one fitted checkpoint, not proof
that it fully explains the old models' evolving trajectories.

Three declared training windows match all five complete parameter-gradient
regions against the frozen observer, 1,196,032 entries per window. The fixed
`k=16,18,20` remainder calculations all satisfy the exact cumulative-gradient
identity and bounded remainders. At k18/20 the largest virtual whole step per
window is 1/1 for embedding coordinates and 3/1 for shared head/bias coordinates.
These virtual updates **never change the weights used for the next gradient**.
The record therefore supports testing magnitude-aware updates, not claiming a
trained optimizer improvement. No native execution, final test or AWS operation
occurs in this diagnostic; its 33 tests and 65 capacity tests pass.

### Integer error-feedback learning v1

All four CPU cells in the [completed record](../evidence/fpga_lm_efsgd_2026-09-09.json)
finish without clipping or sanitizer failures. The new optimizer accumulates
`a=r+G`, applies `trunc_toward_zero(a/2^k)` to the integer weight and retains
the unused gradient remainder. The forward, balanced LUT loss and selected-route
backward remain unchanged; the deployed FPGA image does **not** implement this
optimizer or its persistent state interface.

| Initialization / denominator | NLL at0 | NLL at256 | NLL at1,024 | Final context benefit |
| --- | ---: | ---: | ---: | ---: |
| Original anchors, k18 | 4.850500 | 3.250834 | 2.968443 | 0.071953 |
| Original anchors, k20 | 4.850500 | 3.489289 | 3.240623 | 0.056053 |
| Count fitted, k18 | 2.660464 | 2.648488 | **2.618411** | **0.053022** |
| Count fitted, k20 | 2.660464 | 2.653755 | 2.649575 | 0.007685 |

The useful exploratory candidate is **count-fitted k18**: it beats the stronger
one-pass bigram **2.660153** by **0.041742 nats/token**. The actual-exposure
replay-count baseline is slightly worse (**2.665067**), so it is not used to
inflate the claimed improvement. Removing temporal context from the candidate
worsens NLL to about **2.671433**; the gain is not just retaining the fitted
bigram. Final accuracy is 28.32%. Each fitted model uses 131,072 host-fit pairs
then 131,072 replayed gradient pairs: **262,144 exposures, 131,072 distinct pairs**.
At 256 its baseline correctly fits all 1,024 original windows plus prefix 256,
not two copies of that prefix. Neither original-initialization run beats bigram.

The checked raw tree contains 1,217 files / 819,968,901 bytes, including 12 full
optimizer states of 9,569,360 bytes each and 192 checkpoint-validation windows.
Saved native predictions/gradients/routes are recomputed from checkpoints;
all training-array coverage/hashes and four-cell membership are checked.
The runs perform 4,096 updates / 524,288 gradient-pair exposures and 262,144
host-fitting pair exposures in total, over 131,072 distinct pairs. Full state
includes weights, all remainders, step count, loss/configuration and source
identity, and remains byte-identical through forward-only evaluation.

**What the optimizer math buys.** With constant `Q=2^k`, successful updates
satisfy `Q*(w_0-w_T) = sum_t G_t + r_0-r_T` coordinatewise and `|r_T|<Q`.
Starting with zero remainder, cumulative integer weight displacement differs
from `sum_t G_t/Q` by **less than one weight unit**. Equivalently the shadow
value `x_t=w_t-r_t/Q` obeys `x_t=x_(t-1)-G_t/Q` exactly. This holds for the
actual evolving gradient sequence, not only the fixed-checkpoint diagnostic.
It preserves sub-unit contributions that independent rounding or thresholded
sign updates can discard. It does **not** equate the routed quantized model
with a separately evolving real-valued SGD model, prove convergence, or make
the balanced surrogate the exact derivative of the reported normalized NLL.
The global bound assumes zero initial remainder; a resumed subinterval has
two endpoint remainders and displacement discrepancy less than two units.
Overflow or clipping would invalidate the identity, so execution rejects both.

**Causal limitation.** This is evidence for the complete **fitted-start plus
error-feedback recipe**, not universal superiority of error feedback. From the
same original anchor weights, shift 2, seed 0, data order and cohort, the retained
SignSGD control reaches NLL **2.771690**, better than original-anchor k18
error feedback (**2.968443**). The exploratory study lacks the corresponding
SignSGD run from the exact count-fitted start. That matched control is needed
before attributing the fitted recipe's gain to the optimizer or assuming that
the deployed hardware optimizer must be replaced.

This is one seed, reused development validation40–55, a LUT-normalized loss,
and an explicitly **host-fitted start**. It is not from-scratch LLM pretraining,
a statistically established advantage, FPGA execution, a speed result or a
passing three-seed/paid gate. Not every training update is independently
replayed. The follow-up [unchanged-recipe order confirmation](PLAN.md#shuffled-order-optimizer-confirmation-v1-completed)
is reported below; no geometry tuning was performed on this cohort.

### Error-feedback training-order confirmation v1

The [completed three-cell record](../evidence/fpga_lm_efsgd_confirmation_2026-09-09.json)
keeps the count-fitted k18 initialization, arithmetic and 1,024-update budget
unchanged. Seeds select three shuffled orders of the same training windows;
validation moves to development windows **56–71**, 2,048 positions. All initial
full states are byte-identical to the prior fitted state, including zero
remainders. No model count fitting is newly performed.

| Training-order seed | Final NLL | Advantage over stronger bigram | Context-removal penalty | Context gate |
| --- | ---: | ---: | ---: | --- |
| 0 | 2.576359 | 0.068112 | 0.101865 | Pass |
| 1 | 2.620202 | 0.024268 | 0.015069 | **Fail** |
| 2 | 2.588656 | 0.055814 | 0.094489 | Pass |

Initial NLL is **2.644726** with zero context benefit. The conservative bigram
comparator is the one-pass **2.644470**, not the weaker actual-exposure replay
control **2.648359**. Every order beats that stronger baseline, with mean
advantage **0.049398 nats/token**. All mechanism checks pass with no observed
clipping, head cancellation or native/observer discrepancy. But order seed 1
misses the unchanged **0.02** context-benefit/growth threshold. The all-run
context and combined scaling gates are **false**; the mean and per-seed bigram
gate is true. Positive loss results do not override the failed context gate.

The audit covers **914 files / 615,891,773 bytes**, nine complete 9,569,360-byte
optimizer states and 144 checkpoint-validation windows: 18,432 native
predictions and 2,359,296 logit-gradient entries, plus all saved checkpoint
routes. Every actual shuffled training index, training-array hash/coverage,
full-state evaluation/import invariance and historical initialization binding
is checked. Not every training update is independently replayed.

There are **3,072 new updates / 393,216 new gradient-pair exposures** and zero
new model fits. Summed per-model inherited fitting histories contribute
393,216 exposures, for 786,432 total model-history exposures over **131,072
distinct pairs**; the shared old fit was not executed three more times.
These are order replicates, not independent data samples or weight seeds.
This is evidence that the fitted pipeline can beat bigram on a new development
slice, not reliable contextual learning in every run, statistical significance,
from-scratch pretraining, physical FPGA execution or a cost advantage.

The matched [fitted-start SignSGD control](PLAN.md#matched-warm-start-signsgd-control-v1-completed)
is reported below.
The improved recipe has not passed physical checkpoint equivalence, and no AWS
operation or paid scaling followed this confirmation.

### Matched fitted-start optimizer control v1

The [completed, audited control](../evidence/fpga_lm_warm_signsgd_2026-09-09.json)
uses the exact original fitted weights, sequential training windows 0–1023,
development 40–55, zshift 2 and the existing native SignSGD (`lr=1`, threshold
2048). The matched historical EF k18 result is retained unchanged.

| Optimizer | Initial NLL | NLL at 256 | NLL at 1,024 | Final context benefit |
| --- | ---: | ---: | ---: | ---: |
| Frozen SignSGD | 2.660464 | 2.741048 | 2.824754 | 0.124364 |
| Historical EF k18 | 2.660464 | 2.648488 | 2.618411 | 0.053022 |

SignSGD is **0.206343 nats/token worse** than EF and **0.164601 worse** than the
stronger bigram baseline. Its greater context-removal penalty does not make
its overall learning quality better. This is a matched comparison of the two
fixed optimizer packages, including magnitude, scaling and deadband—not an
isolated test of the remainder mechanism or a general claim about SignSGD.

Every one of its 1,024 updates is preflighted with the full host parameter
gradient and int16 update oracle. Actual native predictions, logit gradients,
routes and **every post-update weight** match; native full parameter-gradient
arrays themselves are not exported. There is no clipping or guard failure.
The worker finishes in 654.03 seconds, below its 1,200-second limit; this
includes host checking and is not a hardware or performance measurement.

The audit verifies **307 files / 191,421,441 bytes**, three complete
4,784,128-byte weight checkpoints and 48 checkpoint-validation windows: 6,144
native predictions and 786,432 logit-gradient entries, plus checkpoint routes.
It checks the complete sequential exposure order, training-array hashes,
runtime guard receipts and full-weight hash chain. Posthoc verification does
not rerun the entire training trajectory or reconstruct raw training routes
from the retained route digests.

There are 131,072 new gradient-pair exposures, 131,072 inherited fitting
exposures, zero new fitting, and 262,144 model-history exposures over 131,072
distinct pairs. This one development control evaluates no three-seed quality
gate, final test, physical FPGA result, cost advantage or paid authorization.

### Localizing the order-sensitive context failure

The completed confirmation's saved per-window diagnostics show the following
at its final checkpoint for projection layer 1 (zero-based):

| Order seed | Retained projection gradient L1 | Cancelled projection VJP positions / 2,048 | Temporal-carry positions / 2,048 |
| --- | ---: | ---: | ---: |
| 0 | 40.75% | 32 | 1,680 |
| 1 | 3.93% | 759 | 558 |
| 2 | 29.87% | 91 | 1,491 |

The L1 ratios use recorded incoming and scattered gradient mass, not counts
of selected routes. All heads retain 128 distinct winners and zero cancelled
head-input VJPs. Nonzero optimizer remainders remain in every region; these
observations do not suggest a global remainder reset. They localize a
trajectory difference, not its cause: cancellation can both influence and
result from the learned weights. The top layer also has substantial
cancellation in all three runs, so sparse routing alone does not explain the
failed seed.

This motivates the [forward-preserving projection comparison](PLAN.md#forward-preserving-projection-diversity-v1-completed),
which changes initial gradient routing while retaining the fitted forward
function. The initializer's mathematical guarantee is not a learning result.

**Why forward equality and different gradients are compatible.** At the fitted
point, write each residual row as `R_l(m)=E[token_m]+s_l(m)*1`. Mean-centering
removes the integer common shift exactly. All unused channels then equal
`u(token)=-floor(sum(E[token])/512)`; anchored channels are no larger.
Either temporal max or min preserves that ordering and gives a common unused
value `C_l(m)`. The old all-zero projection outputs this value in every row.
The new designated-unused projection does too: its selected edge has weight 0,
and every competing edge is at most `C_l(m)-8192`. Thus all residual activations
and raw head logits match, not just normalized probabilities. With anchor
magnitudes at most 756, `u,C_l` lie in `[0,189]`, residuals and winning logits
in `[-756,756]`, and the head margin is at least 7,436. No initial clipping is
needed to obtain the equality.

In backward, equal-valued channels are still **distinct learnable coordinates**.
The balanced head cotangent starts on group A with sum zero. The designated
`A→B→C→D→B` mapping is injective on the first three incoming supports
`A`, `A∪B`, `A∪B∪C`. Unused-channel temporal winners are identical at this
point, so each zero-sum group moves together and integer centering introduces
no remainder. Projection scatter therefore preserves L1 in layers 3, 2 and 1
at initialization. The bottom can merge A and D, and a constant-token window
selects tap 0: neither global injectivity nor nonzero past-time carry follows.
Window-summed parameter gradients can still cancel. Once training separates
the channel values or routes, these initialization-only guarantees need not
persist. The experiment below tests learning, not merely this invariant.

### Forward-preserving projection comparison v1

All six cells in the [new comparison](../evidence/fpga_lm_diverse_projection_2026-09-09.json)
complete: three new MP-initialized training runs and three unchanged historical
checkpoint controls, evaluated on the same new development windows **72–87**.
The initial NLL is exactly **2.635290** in both arms. Initial native predictions,
logit gradients and head/tap routes match byte-for-byte; MP routes change to
the declared mapping. Other initial weights and optimizer state are unchanged.

| Order seed | Historical final NLL | New final NLL | New minus historical NLL | Historical context benefit | New context benefit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.559821 | 2.589094 | +0.029274 | 0.106473 | 0.143325 |
| 1 | 2.605466 | 2.608495 | +0.003029 | **0.013775** | 0.157768 |
| 2 | 2.572647 | 2.636362 | +0.063715 | 0.092345 | 0.099388 |
| Mean | **2.579311** | 2.611317 | **+0.032006** | 0.070865 | 0.133494 |

The stronger bigram comparator is **2.635265** (actual-exposure replay; the
one-pass control is 2.635503). The new arm's mean advantage is **0.023948
nats/token**. Two new runs beat bigram; seed 2 trails it by **0.001097**, which
is within the unchanged per-seed 0.02 loss allowance. All three new runs pass
the fixed 0.02 context-benefit/growth threshold. Consequently the new arm
passes the declared mechanism/context/bigram screen. The historical arm has
the better mean bigram advantage (**0.055954**) but again fails the context
gate in seed 1. The prior failed confirmation is retained unchanged.

**This is a trade-off, not predictive-quality superiority.** The new arm has
worse NLL than its matched historical control in every order; seed 2 also
slightly worsens its own initial NLL. It strengthens the context-removal
signal across these three orders, but preserving initial projection-gradient
L1 is not a guarantee of better learning. A context-removal penalty describes
this model under an inference intervention, not its advantage over a trained
context-free model or evidence of long-range language understanding.

The publication audit checks **1,824 files / 1,028,553,505 bytes**, including
18 complete 9,569,360-byte optimizer states and 288 checkpoint-validation
windows: 36,864 native predictions, 4,718,592 logit-gradient entries and all
saved checkpoint routes. Checkpoint outputs are independently recomputed by
the host observer; this is not a second independent training trajectory.
A separate integrity review verifies all actual hashes/state receipts, nine
historical byte matches, paired initial arrays, count baselines, exposure
accounting, aggregates and unchanged gates. It does not repeat that numerical
forward replay or rerun training.

No clipping, rounded-zero target mass, head-gradient cancellation or native
checkpoint/observer disagreement is observed. All full states retain weights,
remainders, step, configuration and loss identity through evaluation/import.
Historical controls perform **zero new training**; the three new runs perform
3,072 updates / 393,216 new gradient-pair exposures. Summed across six model
histories, inherited gradient exposures contribute 393,216 and inherited fit
exposures 786,432, for **1,572,864 model-history exposures over 131,072 distinct
pairs**. There are no new model count fits; shared old fitting/training was
not executed again. Training arrays are hash/coverage checked, not a full
independent replay of every new update.

These are three order replicates sharing one training set, fitted function and
2,048 validation positions—not independent datasets or 6,144 independent
held-out positions. The candidate still starts from a supervised host count
fit, not from-scratch pretraining. Passing the numerical screen authorizes no
paid run and establishes no physical FPGA execution, significance, useful-task
advantage, speed or cost saving. Before selecting either arm for hardware,
freeze the final checkpoints and compare broader development quality and a
conventional learned baseline; keep final test untouched.

### Learning-contract diagnostic v1

The [2026-09-10 diagnostic record](../evidence/fpga_lm_learning_contract_2026-09-10.json)
completes the [predeclared CPU-only protocol](PLAN.md#learning-contract-diagnostic-v1-completed).
This is a learning/numerical diagnostic, not a repaired LM, a new quality
benchmark or an FPGA gate. All prior sources, checkpoints and failed quality
gates remain unchanged. No held-out corpus windows, final test or AWS resources
are used. The independent reference, numerical helpers and runner have
**237 passing synthetic tests**; those tests do not execute this scientific
recipe. The actual run follows their source freeze.

**Contextual FP64 learning works in the tested setting.** The D16/V4 teacher
uses past-token information: disabling context gives KL 0.01287247, and the
largest probability difference between positions sharing a current token is
0.09724146. Its nontrivial route margins are positive. Three cells receive
64 full-batch updates each, using four training and four separate synthetic
evaluation windows with exact soft targets:

| Student / routing | Initial → final training KL | Initial → final synthetic-evaluation KL |
| --- | ---: | ---: |
| Nearby / fixed routes | 0.00938362 → 0.00002893 | 0.01266138 → 0.00005289 |
| Nearby / dynamic routes | 0.00938362 → 0.00002893 | 0.01266138 → 0.00005289 |
| Structured cold / dynamic routes | 0.17161805 → 0.01093638 | 0.10997660 → 0.01448539 |

The nearby runs have identical full trajectories and final weight files.
Their final dynamic head/projection/tap routes also match the initial routes
on both window sets: the fixed-affine solution transfers to the actual model
on these inputs. The cold control changes 407 temporal routes across the two
sets; head and projection routes stay fixed. All three observed CE trajectories
are monotone. Only the fixed-affine run has a globally applicable Hessian
step-size bound; the dynamic runs use an initial-cell bound. Manual/autodiff
VJP disagreement is at most 2.67e-15, with directional error at most 1.32e-10.
The cold control still inherits diagonal projections and an anchored head;
it is not arbitrary random initialization. No final student context-ablation
score was collected. This is synthetic learnability, not corpus generalization.

**The six actual D512 updates match the independent reference.** Each arm
resets its source-bound step-zero state before probing old training windows
0, 37 and 511. Raw logits, predictions, routes, the full legacy parameter
gradient, and every resulting native weight/remainder agree. These are six
independent updates, not a six-step training trajectory.

At those same points the counterfactual ladder separates the loss-head
direction, integer centering, weight rounding and route changes. The table
shows mean true-CE improvement on the three probed training windows at the
declared scale 1 (positive is lower loss):

| Initialization / direction | Ideal affine update | Rounded weights, affine routes | Full integer dynamic forward |
| --- | ---: | ---: | ---: |
| Historical / exact normalized CE | 0.019382 | 0.003765 | 0.003740 |
| Historical / native legacy | 0.014321 | 0 | 0 |
| Diverse / exact normalized CE | 0.134095 | 0.053836 | 0.052571 |
| Diverse / native legacy | 0.113861 | 0.031400 | 0.031004 |

Only legacy scale 1 is an actual native EFk18 step. Other directions are
numerical counterfactuals, not implemented deployable optimizers. All three
historical native steps change zero integer weights, but their nonzero
gradients are preserved in the checked EF remainders. The diverse native
steps change 53, 32 and 60 weights. Thus quantization delays or reduces
immediate model movement; it does **not** discard these updates or prove that
EF cannot learn over time. At scale 1, diverse native mean LUT-NLL improvement
similarly shrinks from 0.113469 ideal-affine to 0.032355 full-integer.

Fixed-mass exact softmax and normalized LUT directions produce identical
rounded weights in all six scale-1 probes, with ideal-affine CE differing by
at most 0.0000151 nats. Additional LUT-mass truncation is also small here.
The native row mass varies from 33,107 to 64,697 versus fixed 65,536; changing
that mass changes effective update sizes and example weighting, not just
probability approximation. Its improvement here is not isolated from a
learning-rate change. With the same native head, legacy versus real-mean
centering backward yields identical candidate scores throughout this probe
set. Recomputing integer centering with fixed routes changes CE only at
floating-point roundoff level. Dynamic-route effects are much smaller than
weight rounding here, but these tied initial states do not characterize later
training, and route-count changes alone are not a loss explanation.

**Stationarity and parameterization controls narrow, but do not settle, the
cause.** Exact CE on matching add-one soft targets is stationary to roughly
1e-18. The normalized LUT introduces expected-gradient L1 residuals of
0.0001044 and 0.0001269 in the two declared four-class rows, but the largest
matched-target KL after any declared probe is only 2.78e-8. Exact unsmoothed
training already produces larger nonzero gradients at that same smoothed fit.
Consequently count-fit erosion can be objective mismatch even with correct
arithmetic; these synthetic rows do not measure its contribution to corpus
quality. The duplicated-bias quadratic verifies that updating two identical
additive coordinates at rate eta is equivalent to one at twice eta. It does
not demonstrate that tying biases improves this model. EF conservation passes
the declared scalar controls and all full native-state comparisons.

The run retains **32 files / 58,214,083 payload bytes**, including six complete
optimizer states, three toy trajectories/final weight sets, all 90 numerical
candidates, hashes of sources/tests/prior inputs, and the actual pre-run
declaration. Publication and read-only verification pass; integrity checking
does not repeat the learning trajectories. Accounting is 192 new FP64 updates
and 9,216 soft-target position exposures over 48 synthetic training positions,
plus six native updates / 768 gradient-pair exposures over 384 corpus pairs.
There are zero new count fits. Summing six reset histories adds 786,432
inherited fit exposures, not six newly executed fits; those histories share
131,072 distinct corpus pairs. No statistical significance, from-scratch LM
quality, hardware efficiency or cost claim follows.

**Review conclusion:** local routed learning and the inspected native update
contract are sound. The largest immediate discrepancy is update resolution,
with row-dependent step mass and parameterization as additional controls.
The full validation-quality gap is still open. The next proposed step is a
short matched persistent-EF/FP64 trajectory plus later-checkpoint probes,
before choosing any precision, objective or optimizer repair. See the
[follow-up](PLAN.md#recommended-follow-up-after-review).

### Persistent trajectory diagnostic v1

The [checked 2026-09-10 record](../evidence/fpga_lm_persistent_trajectory_2026-09-10.json)
completes the [declared follow-up](PLAN.md#persistent-trajectory-diagnostic-v1-completed):
two initializations, each with 32 consecutive-window native EFk18 updates and
32 unrounded-native-rule updates. The latter uses continuous FP64 weights but
keeps the same dynamic routing, floor forward, LUT head/row mass and raw legacy
backward. It is **not smooth exact-CE SGD**. This comparison changes weight
realization, not the objective or supplied gradient rule at a shared state.

All 64 native updates agree with the independent reference in predictions,
routes, raw head gradients and every resulting parameter/remainder word.
Every full parameter-gradient pair is retained. With `Q=2^18`, the compensated
state `z=w-r/Q` obeys `z_after=z_before-G_native/Q` exactly, and the difference
from the autonomous floating control is exactly
`z_native-w_float=-sum(G_native-G_float)/Q`. Saved-gradient arithmetic replay
also matches all eight complete native and eight FP64 checkpoints. No clipping,
overflow, conservation failure or state-reset repair is needed.

**Precision alone is not a demonstrated repair.** The table reports mean true
CE on the three fixed **training** windows 0/37/511. All were used by the
inherited fit; windows37/511 are only unseen by the new32 gradient passes.

| Initialization | Initial CE | Step32 visible EF | Step32 compensated EF | Step32 unrounded control |
| --- | ---: | ---: | ---: | ---: |
| Historical | 2.397663 | 2.383522 | 2.382169 | 2.382994 |
| Diverse | 2.397663 | 2.390275 | 2.408190 | 2.398406 |

The unrounded control is just **0.000528** better than visible EF historically,
but **0.008131 worse** with diverse initialization. The latter difference is
not consistent across windows: the control wins on0/511 and loses window37
by 0.031297 CE. These three windows cannot support a general quality ranking.
Normalized LUT-NLL means agree in direction: historical visible/control
2.384132/2.383522; diverse 2.390569/2.399026.

Historical EF has zero visible motion only at step1, then starts moving at
step2. Its step1 compensated state exactly equals the floating control.
Thereafter the trajectories query different gradients. At step32 their
compensated-versus-floating maximum coordinate discrepancies are 5.218811
and 6.764042, while each visible-versus-compensated discrepancy remains below
one raw weight unit. This is accumulated feedback from different gradient
queries, not loss of the stored fractional updates. Scoring the compensated
state is an inference readout, not a third independently trained model; it is
particularly worse than visible EF on the diverse probes.

Every floating update lowers its own current-window CE; native updates do too,
except the first historical step which leaves it unchanged. Nevertheless,
diverse fixed-probe CE worsens from step8 to32 for all three readouts. This
argues against a pervasive immediate-descent or wrong-sign implementation bug
in the tested sequence. It is consistent with cross-window interference and
update sensitivity, but identifies neither as the cause of held-out failure.
The step32 context-disabled-minus-enabled CE penalties are historical
0.002888 visible versus -0.000248 floating, and diverse 0.025154 versus
0.013856. More fractional movement is not uniformly better context learning.

**The later numerical ladder also completes.** All 24 cases cover new native
steps8/32 and separately frozen shuffled-history steps256/1,024, for both arms
and all three probe windows. Their independent raw logits/routes/head/full
legacy gradients agree. Every candidate inherits its anchor's actual
remainders and remains uncommitted. The 120 candidates preserve fresh-only
versus accumulated-ideal effects separately, with visible and compensated
baselines; gradients are queried at visible weights, not compensated weights.
Both exact-CE and legacy fresh-only affine candidates lower CE in **24/24**
cases at the declared scale1. These probes therefore do not demonstrate a
finite-step ascent failure, even though directions and update sizes differ.
The older states are not continuations of the new consecutive-window runs.

Head-mass effects remain larger than centering effects in this sample. Across
the 24 cases, switching normalized truncated mass to native variable mass
changes accumulated-ideal CE by at most 0.039275 and dynamic candidate CE by
0.036324. Changing only backward centering reaches 0.0000705 and 0.0001867,
respectively. For exact-CE candidates, the largest weight-rounding effect is
0.076430, versus 0.001105 for forward-floor recomputation and 0.005520 for route
reselection. These are maximum absolute paired differences, not additive
causal shares or evidence that a fixed-mass training recipe improves quality.

A **post-hoc, descriptive analysis of saved gradient arrays** finds opposing
directions in 12/24 exact-softmax/real-centering surrogate-gradient window pairs
and 14/24 legacy pairs. No new model forward, gradient query or training is used.
However, conflict is stronger/more persistent in the historical arm; all three
diverse step1,024 pairs align positively. This confirms local interference can
occur but does not explain the diverse quality penalty. The three training
windows and repeated anchors are correlated, not independent samples or a new
statistical gate; surrogate alignment does not guarantee a finite routed step.

The sealed run contains **214 files / 91,646,300 payload bytes**, bound to
34 source/test hashes and 17 exact input files. There are 249 new passing
synthetic reference/arithmetic/provenance/lifecycle tests; they do not execute
this scientific recipe. Publication and a separate read-only check pass.
The checker replays optimizer algebra from saved full gradients, not model
forward/backward or an independent scientific training trajectory.
An independent NumPy audit also reconstructs all 64 native/floating update
pairs from the saved gradients and all 120 counterfactual rounded candidates,
including persistent remainders and exact checkpoint cross-links. This is
independent arithmetic verification, not a second model-training run. The full
regression suite passes 3,345 tests, with 14 skips and 10 slow/native deselections;
lint, strict documentation, artifact layout and review-PDF freshness checks pass.

Accounting is 64 native plus 64 floating updates, 16,384 new gradient-pair
exposures over 4,096 distinct new-gradient pairs. Four shared fitted starts
contribute 524,288 inherited fit-history exposures: 540,672 summed trajectory
history exposures over 131,072 distinct historical pairs, with zero new fits.
The later ladder's 15,360 gradient-pair evaluations are not training updates.
No held-out corpus, final test, cloud resources or physical FPGA are used;
no significance, useful-LM quality, efficiency or cost advantage follows.

The separately declared update-scale screen below follows this diagnostic.
Broader development quality and a conventional learned baseline remain required
before hardware scaling; intact error feedback alone is not a quality result.

### Paired update-scale screen v1

The [completed record](../evidence/fpga_lm_update_scale_2026-09-10.json) and
[independent audit receipt](../evidence/fpga_lm_update_scale_audit_2026-09-10.json)
follow the [fixed protocol](PLAN.md#paired-update-scale-screen-v1-declared-2026-09-10-before-execution), which
compares k18 with k20 (fourfold smaller raw-gradient updates), holding the loss,
integer forward/backward, fitted starts and three training-order permutations
fixed. Historical initialization is an interaction control, not a fallback
candidate. Its earlier sequential-order k20 failure was acknowledged before
execution. The diverse initialization is the primary comparison.

Six new k20 cells complete 1,024 updates each. Six verified k18 histories are
imported unchanged, not retrained. All twelve final endpoints and both shared
initial functions are evaluated on development windows 88–151 (8,192 positions).
Step 256 is retained for diagnostics, not used to select a checkpoint. The
criterion is the original untruncated normalized-LUT NLL, not the rounded head
gradient distribution. Per-window metrics and complete native arrays are kept.

| Initialization | Order seed | k18 NLL | k20 NLL | k20 minus k18 | Context benefit, k18 / k20 |
|---|---:|---:|---:|---:|---:|
| Historical | 0 | 2.612077 | 2.677393 | +0.065316 | 0.113242 / 0.005130 |
| Historical | 1 | 2.658839 | 2.677509 | +0.018670 | 0.023404 / 0.006584 |
| Historical | 2 | 2.625131 | 2.677556 | +0.052425 | 0.099853 / 0.005972 |
| Diverse | 0 | 2.648941 | 2.667467 | +0.018526 | 0.138120 / 0.029420 |
| Diverse | 1 | 2.668318 | 2.669849 | +0.001530 | 0.127943 / 0.023420 |
| Diverse | 2 | 2.696506 | 2.667875 | −0.028631 | 0.088598 / 0.025740 |

Lower NLL is better; context benefit is context-disabled minus ordinary NLL.
Historical mean NLL worsens from **2.632016 to 2.677486**. Diverse mean NLL
improves from **2.671255 to 2.668397**, only **0.002858 nat**, below the declared
0.01 requirement; two of three order seeds regress. The diverse k20 results
have a much narrower observed range across these orders, but less context
benefit. This is a descriptive tradeoff, not statistical significance or a
successful repair. The stronger bigram NLL is **2.685392** (one pass; replay is
2.689263). Diverse k20 beats it in all three orders, but its mean advantage is
only **0.016995 nat**, below the required 0.02. All three diverse mechanism and
context gates pass; historical k20 fails context in every order. The primary
promotion screen fails on paired improvement, consistency and bigram margin.
Development 152–863 and
the final test remain unopened; no rate, checkpoint or fallback was selected.

The research runner uses the new public experimental
`tikitorch.optim.fixed_point_ef_step` at twelve full-gradient parity anchors.
It checks every weight/remainder against the native update, including live
remainders. The library implementation is a functional CPU integer reference,
not the existing floating-residual `EFTropicalSGD` and not a PyTorch AdamW
replacement. No optimizer defaults change. Its exact conservation guarantee
does not establish descent or equality to a separately evolving floating model.

The runner and its 132 synthetic tests were frozen before execution, together
with the 83 integer-primitive tests (31 total bound source/test files; 25 input
files). The separate incremental auditor and its 19 mocked tests were frozen
before readout replay. The ordinary check validates complete states/array
coverage, baselines, aggregation, ancestry and saved optimizer arithmetic;
the independent auditor repeats the declared 896 checkpoint/window readouts
and twelve raw-gradient anchors. Neither independently replays every training
step. Raw runs and audit receipts live separately under
`manuscript/build/fpga_lm_update_scale_2026-09-10/{v1,audit-v1}/`.

Both checks completed successfully. The sealed scientific run contains
**1,960 files / 1,060,465,323 bytes**. Public record SHA256:
`0213e78e1c6574a1534e04818c9818618524f72b30aa4f1f65ce60a6550722a9`;
independent audit receipt SHA256:
`1d755c146b4a27669b485c7baf9f5f45c20117ff24008533fdfdd60ed3a2caa5`.
The incremental replay checks 114,688 prediction values, 14,680,064 head-gradient
words and all route arrays. Its twelve gradient anchors cover 14,352,384
parameter coordinates, each checked for both updated weight and remainder.

Accounting: **6,144 new updates**, 786,432 new gradient-pair exposures and
786,432 inherited control-gradient exposures. Summed inherited initialization
fit histories contribute 1,572,864 exposures: 3,145,728 summed endpoint-history
exposures over 131,072 distinct training pairs. The common fit was not performed
twelve times, and no initialization was refitted. One-pass/replay count baselines
are fitted separately for comparison; their own exposure counts are recorded.
Repeated independent gradient/readout checks are evaluations, not updates.
No cloud resource, physical FPGA, new benchmark timing or useful-LLM claim.

**Originally proposed next diagnostic, now completed below:** compare sequential k18 with four-window
gradient accumulation over the already-used 32 training windows. Sum the four
gradients at unchanged block weights and divide by the same `2**18`, preserving
the first-order per-example scale instead of accidentally applying another
fourfold learning-rate reduction. Keep initial weights, data, loss and persistent
remainders fixed; explicitly report 32 versus 8 optimizer updates and 4,096
gradient-pair exposures per arm. Check all accumulator prefixes and compare
shared-block CE/LUT changes plus the old training probes. Improvement would
support a change to gradient-query/update timing, not uniquely identify
interference or duplicated MP/MB gradients as the cause, establish held-out
quality, or authorize hardware scaling.

### Four-window gradient accumulation v1

The [completed host run](../evidence/fpga_lm_gradient_accumulation_2026-09-10.json)
follows the [predeclared protocol](PLAN.md#four-window-gradient-accumulation-v1-declared-2026-09-10-before-execution):
same diverse fitted start, training windows 0–31, legacy raw LUT gradients and
k18. Sequential execution makes 32 commits; batch4 sums four gradients at fixed
block weights and makes 8 commits. Both use 4,096 gradient-pair exposures. No
gradient averaging or rate change is hidden in the comparison.

The result is **negative**, not a candidate optimizer repair. Both initial
training means are CE **2.446595**, normalized-LUT NLL **2.446619**.

| Final readout on the same 32 training windows | Sequential | Batch4 |
|---|---:|---:|
| True CE | 2.409066 | 5.376293 |
| Normalized-LUT NLL | 2.408988 | 5.138609 |
| Accuracy | 0.283691 | 0.186523 |

Batch4 is worse on **all 32 endpoint windows** under both losses. These are
correlated, previously used training windows, not independent seeds or held-out
quality evidence. Sequential updates improve both losses on all eight shared
blocks. Batch4 raises CE on four blocks and LUT NLL on five; the first ascent is
the second block (windows 4–7): CE **2.595076 → 3.261983**, LUT NLL
**2.594828 → 3.256847**. The third block then raises CE by another 2.150382.
The normalized-LUT metric can substantially understate the continuous CE loss
on this deteriorated trajectory; it retains its declared finite-ROM semantics.

The batch shared-query ratio `||sum(G)|| / sum(||G||)` rises from 0.6101 to
0.8606 to 0.9723 and remains between 0.9124 and 0.9667 thereafter: those vectors
largely reinforce rather than cancel. Sequential ratios use different query
weights and are not a controlled causal comparison. At exposure 32, removing
context improves batch4 CE on all three old probes (0/37/511); mean context
benefit is **−0.323726 nat**. Context-disabled CE is still **4.912531**, so the
damage is not confined to temporal-context use.

All 32 sequential full gradients and before/after weight/remainder hashes match
the frozen native predecessor. Every new host commit agrees with the public
primitive and frozen arithmetic oracle; accumulated and paired shadow identities
hold exactly. Every raw-sum and carry-plus-sum prefix fits int32; the largest
absolute carry-plus-gradient prefix is **23,458,529**. No weight/activation
clipping or backward int32 overflow is detected. This is a learning-dynamics
failure in the tested recipe, not an observed arithmetic overflow.

The eight uncommitted fixed-gradient regrouping controls preserve the same
shadow, but change 52–126 visible coordinates by one integer unit per block.
That does not bound subsequent nonlinear amplification. The experiment changes
both gradient-query and signed-quantization release timing; it does not uniquely
identify interference, routing, centering or quantization as the cause.

Provenance: **42 bound source/test files, 78 bound input files, 402 raw payload
files / 19,679,786 bytes**, plus the separate artifact seal. All 64 full-gradient
queries, 42 complete states (two initial plus 40 committed), 292 score readouts,
the declaration and summaries are retained under
`manuscript/build/fpga_lm_gradient_accumulation_2026-09-10/v1/`.
Public record SHA256:
`bf471012a60710dca536912700b525a2682e43eb6d35caa05eb1360344291c00`.
The ordinary integrity/arithmetic/metric replay passes. The completed
[independent CLI audit](../evidence/fpga_lm_gradient_accumulation_audit_2026-09-10.json)
also matches all **64 full gradients and 292 readouts**, including every raw
query logit, head array and route tape. Full sources, inputs and artifact/receipt
bytes remain unchanged through replay. The new helpers and runner have
**214 passing synthetic tests**, including genuine tiny-model independent replay.

Accounting: 8,192 new gradient-pair exposures over 4,096 distinct query pairs;
262,144 summed inherited fit-history exposures; 270,336 summed endpoint histories
over 131,072 distinct fitted pairs. The shared fit is not rerun. There are no new
native training calls, cloud resources, reserved-validation/final-test readouts,
library-default changes or scaling authorization. Audit and uncommitted
arithmetic checks are not additional training updates.

**Originally proposed next diagnostic, now completed below:** freeze the first failing batch
anchor (state004), its four saved gradients and windows 4–7. Compare uncommitted
fresh-gradient fractions `{0, 1/16, 1/4, 1}`. Preserve live carry: for fraction
`2**(-j)`, use `Q'=2**(18+j)` and `A=2**j*r+sum(G)`, not unchanged `r` with a
different shift. Fraction one must reproduce state008 exactly; fraction zero
must leave the full original state unchanged. Separate fresh-only affine,
carry-preserving ideal, fixed-integer and dynamic-integer readouts. Smaller
fractions helping would support an oversized update at this anchor; no tested
fraction helping would not exclude every smaller step or establish bad model
capacity. This is not permission to choose a rate into a trajectory or scale.

### First-failing-batch step fractions v1

The [completed record](../evidence/fpga_lm_batch_step_2026-09-10.json) follows the
[separately frozen protocol](PLAN.md#first-failing-batch-step-fractions-v1-declared-2026-09-10-before-execution).
All four uncommitted fractions start from the same batch4 state004 and the same
four saved gradients for training windows 4–7. There are no new training
updates, fits or training-gradient queries. Live carry is preserved by rescaling
its integer representation with the denominator, not discarded or reinterpreted.
Fraction zero exactly preserves the source state; fraction one exactly matches
the original state008 weights and remainders.

The result supports an **oversized update at this particular anchor**. The
failure already occurs in the unrounded, fresh-only affine surrogate with
routes and centering offsets fixed at the visible starting state:

| Fresh-gradient fraction | Fresh affine CE | Carry-affine CE | Rounded affine CE | Fixed-integer CE | Dynamic-integer CE |
|---|---:|---:|---:|---:|---:|
| 0 | 2.595076 | 2.680917 | 2.595076 | 2.595076 | 2.595076 |
| 1/16 | 2.534414 | 2.600382 | 2.581829 | 2.581767 | 2.581830 |
| 1/4 | 2.507643 | 2.521971 | 2.519737 | 2.519721 | 2.519917 |
| 1 | 3.474871 | 3.344291 | 3.267050 | 3.267088 | 3.261983 |

Compare each column with its **own zero-fraction value**. The carry-affine zero
is not the dynamic-shadow baseline (CE 2.681315) or the visible baseline
(2.595076). Both reduced fractions improve **CE and normalized-LUT NLL on every
one of the four windows in all five formulations**; the full fraction worsens
both losses in every case. Dynamic LUT NLL is respectively **2.594828, 2.581528,
2.519936, 3.256847**. Dynamic CE gains are 0.013246 and 0.075158 nat for the two
reduced fractions; the unit step raises CE by 0.666908 nat.

The fresh-affine unit step raises CE by **0.879795 nat**, before parameter
rounding, floor recomputation or route reselection is introduced. Thus those
effects are not necessary for this surrogate's failure. In the idealized
fresh-affine model, logits are affine in the step fraction and exact CE is
convex. Improvement at 1/16 therefore implies a negative initial slope, while
the unit step overshoots a beneficial portion of that surrogate direction.
This mathematical interpretation is not a derivative claim for the actual
discontinuous integer model. At the unit fraction,
floor recomputation changes rounded-affine CE by only +0.000037 nat, and dynamic
route reselection changes fixed-integer CE by −0.005104 nat. These are local
paired comparisons, not universal bounds on rounding/routing effects or proof
that either never matters. Smaller fractions helping does not establish a
globally appropriate rate, that the raw legacy vector is an exact derivative,
or a general cause of the broader language-model quality gap.

All candidate arithmetic, original-state/unit-step controls, parameter bounds,
and declared integer forward guards pass. Affine internal activations remain
unconstrained, explicitly nondeployable extrapolations. All candidates are
retained, **none selected or committed**. In particular, this result does not
promote k20 or undo the failed three-order update-scale screen.

Provenance: **46 bound source/test files, 11 direct bound input files**, plus
complete checked parent ancestry; **107 payload files / 11,118,102 bytes**, plus
the artifact seal. Full states/ideals, four original query archives, gradient
sum, 88 logit/route/offset traces and summaries are retained under
`manuscript/build/fpga_lm_batch_step_2026-09-10/v1/`.
Public record SHA256:
`d65a4c9a35a659ad33d7543920f95a144348822a7e59528c95592c2d780cb404`.
The ordinary saved-point/state/metric check passes. The completed bounded
[recomputation audit](../evidence/fpga_lm_batch_step_audit_2026-09-10.json) also
matches all four original full gradients and all 88 forward traces/readouts,
with unchanged source, input, raw, seal and public bytes.
Dynamic integer readouts and gradients have distinct frozen implementations;
affine/fixed-integer/fractional-shadow replay reuses its defining helpers.
The new helper and runner add **175 passing synthetic tests**.

**Originally proposed next diagnostic, now tested below:** assess a separately declared, properly
normalized accumulated-update recipe across multiple blocks and training orders,
with explicit gradient units, mean-versus-sum convention and persistent-state
accounting. A quarter fresh-gradient contribution corresponds to averaging four
raw gradients in real units, but carry handling and integer rounding must still
be specified exactly. This was a deliberately matched-scale diagnostic, not
evidence that gradient accumulation generally fails. Any new recipe must clear
its own training/development screen before reserved confirmation, conventional
learned-baseline comparison or FPGA scaling. No automatic rate selection,
trajectory extension, new data or cloud resource is authorized by this result.

### Normalized accumulation order diagnostic v1

The [record](../evidence/fpga_lm_normalized_accumulation_2026-09-11.json) follows
the [2026-09-11 declaration](PLAN.md#normalized-accumulation-order-diagnostic-v1-declared-2026-09-11-before-execution):
three arms and three fixed permutations of training windows 0–31, with identical
diverse fitted starting weights and zero carry. Every trajectory retains its
own fixed denominator. Batch4 sums four raw gradients at unchanged block-start
weights and divides by `2**20`; it does not integer-round a mean before EF or
repeatedly rescale carry. Sequential k20 is the matched per-example-scale
control; sequential k18 is the original-scale reference.

All nine trajectories, ordinary completion checks and the separate independent
gradient/readout replay pass. The shared
initial mean is CE **2.446595**, normalized LUT NLL **2.446619**, accuracy
**27.25%**. Every endpoint improves both losses over this initial state:

| Arm | Order | Final CE | Final LUT NLL | Accuracy |
|---|---|---:|---:|---:|
| Sequential k18 | Consecutive | 2.409066 | 2.408988 | 28.37% |
| Sequential k18 | Shuffle 0 | 2.415321 | 2.415338 | 28.30% |
| Sequential k18 | Shuffle 1 | 2.411765 | 2.411780 | 27.91% |
| Sequential k20 | Consecutive | 2.434074 | 2.433924 | 27.22% |
| Sequential k20 | Shuffle 0 | 2.430641 | 2.430232 | 27.29% |
| Sequential k20 | Shuffle 1 | 2.431718 | 2.431752 | 27.56% |
| Batch4 k20 | Consecutive | 2.433578 | 2.433330 | 27.25% |
| Batch4 k20 | Shuffle 0 | 2.435819 | 2.435464 | 27.47% |
| Batch4 k20 | Shuffle 1 | 2.432630 | 2.432347 | 27.44% |

Mean CE is respectively **2.412051 / 2.432145 / 2.434009**, and mean LUT NLL
**2.412035 / 2.431970 / 2.433714**. Normalized batching avoids the earlier
consecutive-order batch4/k18 explosion (CE 5.376293), but **does not provide a
quality advantage**. Against sequential k20 it is worse by mean **0.001864 CE /
0.001744 LUT**; only consecutive order improves slightly, with both shuffled
orders regressing under both losses. It also trails sequential k18 by mean
**0.021958 CE / 0.021678 LUT**, and is worse in each paired order.

The declared numerical mechanism gate fails three clauses: the required
0.01-nat mean benefit, no order-level regression, and no worse mean than k18.
No arm, rate or order is selected. A normalization that removes a short-run
instability is not a demonstrated solution to the broader LM quality gap.
These three orders share one initialization and data set; they are not
independent data/initialization trials or evidence of generalization.
Eight commits instead of 32 is an operation-count difference, not measured
speed or energy savings. A possible hardware benefit from fewer update scans
remains unmeasured; this experiment tests a learning intervention, not its
quality/cost frontier.

The old context probes 0/37/511 also remain **training data**. Final mean
context-disabled-minus-enabled CE is **0.025577 / 0.003239 / 0.003491** for the
three arms. Greater context sensitivity does not itself mean better prediction:
absolute context-on probe CE is **2.392156 / 2.386531 / 2.388195**, versus initial
**2.397663**. Those three probes rank the sequential rates differently from the
full 32-window endpoint readout. Block means describe the actual gradient-query
points; sequential queries have different weights, so these are not fixed-point
before/after block losses or proof of monotonic descent.

All recorded forward/backward range guards, ordered int32 prefix checks,
public/frozen EF comparisons and exact per-commit/cumulative shadow identities
pass. Maximum absolute raw-gradient prefix is **3,647,273**; maximum absolute
carry-plus-prefix is **3,391,461**. The consecutive seq_k18 rerun matches every
saved native gradient and before/after state hash, including the full saved
state anchors. This does not constitute a new native or physical FPGA run.

Provenance: **49 source/test files, 10 direct input files**, plus complete
checked predecessor ancestry; **1,401 payload files / 75,624,718 bytes**, plus
the artifact seal. All 288 full gradient-query archives, 225 complete states
with denominator/counter receipts, 216 transition receipts and 434 score
readouts are retained under
`manuscript/build/fpga_lm_normalized_accumulation_2026-09-11/v1/`.
Public record SHA256:
`09767892fa12fb9a439c0778c0920b37b0ab6a0cca4c37800243d202115ed263`.
The new runner adds **125 passing synthetic checks**. Their genuine tiny-model
replay includes every query/readout and a separate direct-int64 reconstruction
of all 216 committed states; this does not substitute for the corpus audit.
The completed [corpus audit](../evidence/fpga_lm_normalized_accumulation_audit_2026-09-11.json)
separately matches all **288 full gradients and 434 integer score readouts**,
including every query logit, head array and route tape. It replays all 216
commits and verifies all 225 saved complete states. Bound source/input/raw/seal/
public bytes are unchanged throughout. This is an independently implemented
numerical replay at saved points, not an independent-data learning replication.

Accounting: **36,864** new gradient-pair exposures over **4,096** distinct
query pairs; **1,179,648** summed inherited fit-history exposures and
**1,216,512** summed endpoint histories over **131,072** distinct fitted pairs.
The common historical fit is not rerun. There are no new fits, native updates,
reserved-validation/final-test readouts, cloud resources or optimizer defaults.

**Follow-up, now executed:** isolate the relative
per-token weighting introduced by the LUT head's variable mass, with matched
nominal head scale. The current raw head is `g_t = R_t * (q_t - y_t)`, where
`q_t` is its normalized truncated distribution, `y_t` the one-hot target,
and `R_t` varies across tokens.
Use the same forward, real-centering backward and FP64
update carrier in both diagnostic arms, so the paired intervention changes
row weighting rather than simultaneously changing precision or centering.
Equal average head mass does not imply identical full-gradient norms; report
those separately. This would be a diagnostic surrogate comparison, not an
integer/FPGA-ready recipe. A positive effect would still need an integer
implementation and independent quality confirmation; a null result would
favor a conventional learned comparison over further nearby-rate searches.
No scaling is triggered automatically by this record. Results follow below.

### Relative head-weighting diagnostic v1

The [six-cell record](../evidence/fpga_lm_head_weighting_2026-09-11.json) follows
the [frozen declaration](PLAN.md#relative-head-weighting-diagnostic-v1-declared-2026-09-11).
Both arms use the same fitted start, FP64 weight carrier, floor forward,
real-centering selected reverse and `W-G/2**18` update. The fixed scalar
`c=0.72490069642663` comes from the predecessor's 32 saved initial training
logit arrays. Relative weighting uses `c*65536*(m/mean_window(m))*(q-y)`;
uniform uses `c*65536*(q-y)`. Equal average nominal head mass is not equal
parameter-gradient norm; all per-region gradient/update norms are retained.

| Arm | Order | Final true CE | Final LUT NLL | Accuracy |
|---|---|---:|---:|---:|
| Relative | Consecutive | 2.419383 | 2.418831 | 28.27% |
| Relative | Shuffle 0 | 2.447907 | 2.447528 | 26.00% |
| Relative | Shuffle 1 | 2.467305 | 2.467108 | 24.83% |
| Uniform | Consecutive | 2.421750 | 2.420945 | 27.81% |
| Uniform | Shuffle 0 | 2.417636 | 2.417780 | 27.73% |
| Uniform | Shuffle 1 | 2.444494 | 2.444964 | 26.98% |

Uniform weighting improves mean CE by **0.016905 nat** and LUT NLL by
**0.016593 nat**, and improves both losses over initialization in every order.
However it regresses in consecutive order by **0.002367 CE / 0.002114 LUT**.
The predeclared no-order-regression clause fails, so the numerical mechanism
gate is **false**. Mean CE is 2.444865 for relative and 2.427960 for uniform,
versus initial 2.446595. Relative row weights span approximately 0.638–1.524;
uniform row weights are exactly one. This supports sensitivity to example
weighting, not a consistent repair of the broader quality gap. Neither arm is
the old native EF recipe, and there are no new held-out results.

All six trajectories, ordinary completion checks and the separate
[independent full-gradient/readout audit](../evidence/fpga_lm_head_weighting_audit_2026-09-11.json)
pass. That audit independently reproduces all 192 full gradients and 302 scored
readouts (494 forwards total), and verifies all saved states/updates without
changing source/input/raw/seal/public bytes. It is numerical replay at saved
points, not independent-data learning replication. Raw evidence retains
**1,092 payload files / 137,368,814 bytes**, 192 full gradient queries/updates,
198 weights-only states and 302 scored readouts, with 53 source/test files
and the complete checked ancestor closure. Public record SHA256:
`23df8863e625a56d20b2ca3187fad992b46519b82021819adfbcc6e4571adca8`.
The new implementation has 112 passing synthetic checks, including independent
tiny-model replay; these are not a substitute for the corpus audit.

Accounting: 24,576 new gradient-pair exposures over 4,096 distinct queried
pairs; 786,432 summed inherited fit histories and 811,008 summed endpoint
histories over 131,072 distinct fitted pairs. There are zero new fits, native
training calls, held-out readouts, AWS resources or optimizer-default changes.

**Following diagnostic:** ordinary exact-CE/Adam fitting under dimensionless
real-centering semantics, comparing the existing connectivity, a
function-preserving cross-class connectivity change, and a small GRU with
the same frozen fitted prior. The target is to beat the empirical training
current-token entropy floor, not merely reduce loss from a warm start.
This is a separately declared local fitting diagnostic, not proof of
generalization or permission to scale an uncompetitive model on FPGA.

#### Fitted-connectivity bottleneck hypothesis

A read-only inspection of existing route arrays found no departure from the
initializer's designated head/projection connections. In the older diverse
study's three orders at checkpoints 0/256/1024 over its 16 existing validation
windows, deviations are **0/2,359,296 head entries** and
**0/37,748,736 projection entries**. Those are checkpoint observations, not
every step of the 1,024-update trajectories. Across all 288 gradient-query
points of the normalized-accumulation study, deviations are
**0/4,718,592 head entries** and **0/75,497,472 projection entries**.
This inspected saved arrays only; it made no new corpus readout or update.

The source initializer assigns head v to A_v and repeats A_v→B_v, B_v→C_v,
C_v→D_v, D_v→B_v in every projection layer; other candidates start at −8192.
Conditional on those selected supports, at most **84,096 parameter
coordinates** can receive gradients, despite **1,196,032 stored parameters**.
This is a possible-support count, not the measured nonzero-gradient count or
effective rank. Each head's explicit residual/projection ancestry reaches four
same-class-index channels. The model is not completely class-independent:
mean centering provides dense coupling, and temporal selection can turn shared
centering terms into channel-dependent effects. Its four K7 layers have a
maximum 91-token receptive field, not the three-token field of another Python
model used elsewhere in the repository.

The new topology primitive uses layer-specific conjugated cyclic permutations
to make those explicit paths reach 16 distinct class/channel ancestors while
preserving the supported fitted initial function. It has **32 passing
synthetic tests**, including full D512 initial forward checks and all 128
ancestor certificates. The certificate excludes the mean path, does not inject
other classes' fitted A-bank features into the unused banks, and is not a
capacity or learning theorem. It makes a concrete wiring restriction testable;
the completed producer below tests its local fitting effect, not generalization.

### Exact-CE fitted-start topology comparison v1

The [three-arm producer record](../evidence/fpga_lm_exact_fit_2026-09-11.json)
completes the [declared training-only comparison](PLAN.md#exact-ce-fitted-start-topology-comparison-v1-declared-2026-09-11).
Ordinary completion checks and the separate
[independent audit](../evidence/fpga_lm_exact_fit_audit_2026-09-11.json) pass.
All three arms begin at CE **2.479519** and accuracy **26.95%**
on the same 512 training positions. Their largest initial probability
disagreement is **3.33e-16**. Tropical parameters are u=beta*W with real mean
centering, exact CE and eager FP64 Adam at lr=0.001; these are not the native
floor/LUT/integer optimizer semantics.

| Model | Final CE | Accuracy | Context-off CE | Off minus on CE |
|---|---:|---:|---:|---:|
| Original tropical connectivity | 1.532157 | 57.81% | 2.383349 | 0.851192 |
| Cross-class tropical connectivity | 1.426564 | 65.04% | 2.312636 | 0.886072 |
| GRU with the same frozen fitted prior | 0.172225 | 97.27% | 2.307839 | 2.135614 |

All endpoints follow **128 full-batch updates** on training windows 0–3.
The empirical current-token-only optimum on these positions is
**H(Y|current token)=1.775483**; every model beats its declared floor-minus-0.02
threshold. Cross-class connectivity improves CE over original connectivity by
**0.105593 nat**, also exceeding its declared 0.02 threshold. The separate
audit passes for all numerical gates. The original topology succeeds
too, so cross-class connectivity is not necessary for this local fitting gate.

The main unresolved result is the **1.254339-nat gap between mixed tropical
routing and GRU**. This is not a successful quality-matched language model.
The tropical curve has not plateaued: its last 32 updates improve CE by about
0.158 nat, and the final pre-update query loss is 1.431196 before the endpoint
1.426564. One rate and a fixed budget do not establish a capacity ceiling.
An exploratory saved-state calculation compares `dot(G_t, u_(t+1)-u_t)`
with the actual next-query/final CE change, without new model evaluations.
All 128 updates have negative predicted and actual changes in each tropical
arm. Mixed's summed predicted/actual changes are −1.105251/−1.052954
(95.27% realized); its last 32 give −0.170865/−0.158128 (92.55%). This supports
reliable local descent on this trajectory, not a global derivative theorem,
an optimal learning rate or a prediction that tenfold-larger steps must work.
Conversely, beating the current-token entropy proves use of additional
information on these positions, possibly position itself, not useful linguistic
generalization. All results are fitted-start, repeated-training-set results.

Saved-only inspection of every tropical query tape finds **zero head changes
in 8,388,608 entries and zero projection changes in 134,217,728 entries per
arm**, relative to that arm's initialization. Final context-on readouts also
retain those supports. All 1,046,528 off-support projection weights and 65,408
off-support head weights remain at **−88.722839 dimensionless units**
(beta times −8192), with zero optimizer moments. Temporal routing is active:
in the final mixed readouts the four layers select past taps about
86.18%, 87.89%, 83.68% and 75.33% of the time, and every channel/layer uses
all seven taps somewhere on these four windows. Thus the model has adapting
temporal routes inside restricted head/projection connectivity, not a wholly
frozen or context-free computation. These are observations of saved arrays,
not new corpus evaluations or a proof that the restrictions cause the gap.

The models are not parameter- or compute-matched. The tropical arms store
1,196,032 parameters each, with at most 84,096 eligible gradient coordinates
conditional on the retained supports; the GRU has 131,968 trainable parameters.
The tropical receptive field is 91 positions versus the GRU's 128-position
window. Compared with old native studies, the recipe also changes precision,
loss, centering, optimizer and repeated-data budget together. It does not
identify Adam alone as the repair, establish native fidelity or measure speed.

Raw evidence retains **1,203 payload files / 935,065,134 bytes**, binding 59
source/test files and the complete ancestor closure. It contains 384 full-batch
gradients/updates, 387 complete parameter/Adam states and 36 score-window
readouts. Independent checking reconstructs 1,024 tropical window VJPs,
compares their 256 summed gradients and all 128 explicit-cell GRU gradients,
and rechecks every score, state and update without changing evidence bytes.
The bounded CLI audit's recorded start-to-completion span is 1,470 seconds, with 1,048 tropical window
forwards and 131 GRU batch forwards (1,572 total window equivalents).
Public SHA256:
`6c36b6145626b610b3cc46645dfb69e27b16b3f6162531c8fab5e3a852c1c74e`.
The new topology/helper/runner have 86 passing synthetic tests, including
independent replay and all Adam recurrences; these do not replace this audit.

Accounting: 196,608 new gradient-pair exposures across three arms (65,536 each)
on 512 distinct positions, plus 393,216 summed inherited fit histories, gives
589,824 summed endpoint-history exposures over 131,072 distinct fitted pairs.
There are no new count fits, held-out readouts, native calls, AWS resources,
library/default changes or FPGA/scaling promotion.

**Next declared comparison:** one mixed-topology run at Adam lr=0.01, changing
only the learning rate. The [rate-only declaration](PLAN.md#exact-ce-rate-only-comparison-v1-declared-2026-09-11)
fixes the same start and 128-update budget, historical controls, a half-gap
closure threshold and a separate GRU-plus-0.05 local-fitting threshold.
It starts after the completed independent audit, at 23:17:16 UTC on 2026-09-11.
There is no rate sweep or automatic hardware transition, regardless of the
result. The new runner has 41 passing synthetic tests; the combined head-
weighting, exact-fit and rate-control suite has 239 passes. Its outcome follows.

### Exact-CE rate-only comparison v1

The [completed producer](../evidence/fpga_lm_exact_rate_2026-09-11.json) changes
only the mixed tropical model's eager Adam learning rate, from **0.001 to 0.01**.
It passes ordinary completion checks and now passes **independent numerical
verification under the separately declared pinned-predecessor protocol**. The
two earlier deadline-interrupted audits remain failed attempts, not successes;
their unchanged receipts and the new completed audit are distinguished below.
All original mixed parameters, empty initial Adam state, initial logits and
selected routes match the predecessor exactly. The model starts from the same
count-fitted initialization, not from its trained checkpoint. Data, loss units,
real centering, optimizer options and the 128 full-batch update budget are
unchanged. The guarded-forward prototype is not used.

| Model and rate | Final training CE | Accuracy | New training in this follow-up |
|---|---:|---:|---:|
| Mixed tropical, Adam 0.001 | 1.426564 | 65.04% | None: frozen historical control |
| Mixed tropical, Adam 0.01 | **0.124361** | **100.00%** | 128 full-batch updates |
| Fitted-prior GRU, Adam 0.001 | 0.172225 | 97.27% | None: frozen historical control |

All three use the same 512 training positions and initial probabilities
(initial CE 2.479519). The new endpoint improves CE by **1.302203 nat** over
the old tropical rate and is **0.047864 nat below the GRU**. Both declared
local-fitting thresholds pass: half-gap closure requires CE ≤0.799395,
and near-GRU
local fitting requires CE ≤0.222225. The unclipped fraction of the historical
gap closed is 1.038159; a value above one means the new endpoint is lower than
the historical GRU endpoint. These are final-endpoint gates, not selections of
a favorable intermediate checkpoint; their required numerical audit now passes.
All four windows have 100% accuracy, with mean CE between 0.122764 and 0.128080.

The independently verified result identifies the original rate as a major
limitation at this fixed update budget on **this local fitting task**.
It does not establish a universal best rate, a superior
architecture or useful language-model generalization. The faster rate was
chosen after inspecting the predecessor; the models do not have matched
hyperparameter-search, compute or receptive-field budgets. The GRU was not
retuned in this follow-up. Repeated fitting of 512 positions can memorize
content or position; it is not a held-out result or from-scratch pretraining.

Head and projection supports remain unchanged through every query: zero
deviations in **8,388,608 head entries / 134,217,728 projection entries**.
All final context-on readouts retain those supports too. The audited trajectory
therefore fits these positions without unlocking off-support connections.
Removing temporal
context gives CE **3.730651**
and accuracy **22.07%**, a 3.606290-nat penalty. This producer readout shows dependence on
the available sequence information, not linguistic transfer; the severe
context-off regression is not itself a useful-quality improvement.

Raw evidence retains **403 payload files / 304,332,071 bytes**, all 128 full
gradients/updates, 129 complete parameter/Adam states and 12 score windows.
It binds 61 source/test files and 10 ancestor trees / 163 external identities.
The saved pre-readout declaration SHA256 is
`8517e4726961c11d579e7bfeb1338e3eae0732c6ad1d87a5cc006ac90ffad5c3`;
public record SHA256 is
`e393750a7191c9a64c1a4f3cb89150b6355ba61838f551ba7361e42c7ff1859b`.
The completed separate audit reconstructs 512 window VJPs, compares their 128
summed gradients, rechecks all 12 scores (524 independent forwards total) and
replays every complete saved Adam state exactly, without changing evidence bytes.

Both incomplete attempts are retained separately from the unchanged producer:

- [Attempt 1](../evidence/fpga_lm_exact_rate_audit_attempt1_2026-09-12.json)
  started at 23:41:16 UTC on 2026-09-11, with an absolute deadline of 00:41:16
  UTC on 2026-09-12. The owned watchdog interrupted it; exit 1 was observed at
  00:43:33 UTC. Its last reported completed update block was 64, not 128.
- The single declared [audit-only recovery](PLAN.md#audit-only-recovery-declared-2026-09-12),
  [attempt 2](../evidence/fpga_lm_exact_rate_audit_attempt2_2026-09-12.json),
  started at 01:24:06 UTC and ended at 02:24:07 UTC on 2026-09-12 after
  3,600.20 seconds. It used the same full checks and 3,600-second bound, a
  process-scoped `caffeinate -is` wrapper and an absolute UTC watchdog for the
  owned process group. The child exited −2 and wrapper exited 1 after
  `KeyboardInterrupt` while loading predecessor head-weighting score files,
  before the new rate-control gradient replay.

Neither attempt skipped ancestor validation, changed training or reported a
numerical mismatch. Neither completed all numerical checks. The producer
public record, manifest and inventory seal remain unchanged; failures are not
rewritten as successful audit receipts. Those declarations included no further
retry. The separately reviewed protocol below resolves numerical verification
without changing either failed attempt or the underlying producer.

#### Completed pinned-predecessor audit (2026-09-13)

The new [successful receipt](../evidence/fpga_lm_exact_rate_pinned_audit_2026-09-13.json)
records the declared distinction: the rate trajectory is independently
numerically reconstructed, while the already completed exact-fit predecessor
audit is reused after fresh verification of its exact trusted bytes. All
403 rate payloads, 61 original source/test files, ten ancestor trees and 163
external identities are unchanged. The two new auditor/test files bring this
invocation's source closure to 63. All 512 window VJPs, 128 batch-gradient sums,
12 score windows and 129 complete Adam states pass. The independent gradients
are compared at the unchanged tolerances; **saved gradients** drive exact Adam
state replay. Predecessor numerical trajectories are not rerun in this invocation.

The program starts at **20:16:52 UTC** and completes publication/postchecks at
**20:27:52 UTC**, with exit 0 and elapsed **659.985885 UTC seconds /
659.990614 monotonic seconds**, within its 3,600-second bound. The receipt's
separately labeled verification timestamp is 20:27:49 UTC, before publication;
the terminal completion timestamp is program-generated, not the later observation
of exit. Process-scoped sleep prevention is present from launch. Receipt SHA-256:
`56d4388eb440ee77e7416e7fe60592cdf594c7b3cbf066dbc03bc2ff079a940b`.
The saved pre-invocation declaration SHA-256 is
`4c8dc4ec73e162656d51d633d3dfb08003616d040b08007ad4ff681f1456277a`.
This is local numerical verification, not an external laboratory replication.
Both **local-fitting gates now pass with their required audit**; no broader
quality, library recipe or hardware claim is promoted.

Accounting: 65,536 new gradient-pair exposures on 512 distinct positions, plus
the inherited 131,072-pair count fit, gives 196,608 exposures in this new
endpoint's history over 131,072 distinct fitted pairs. Historical controls
retain their original reported costs; they are not free training or new
replications. There are no new count fits, held-out readouts, native/FPGA calls,
cloud resources, library-default changes or automatic recipe promotion.

The local numerical audit requirement is now resolved, and the separately
declared [guarded-forward replay](#guarded-full-trajectory-parity-2026-09-13)
also passes complete trajectory parity. The next scientific hurdle remains **generalization over a
broader training run against a comparably specified learned baseline**.
An implementable lower-
arithmetic numeric/optimizer contract and new hardware parity would follow;
neither can be inferred from this FP64 local fit.

## Main table: preserved CPU sources

The selected historical-source record fixes batch one, vocabulary 38, top-k head k=8, kernel seven,
no dropout, hard routing, FP32 and the three `(D,L,T)` cells below. Parameter,
input and cotangent seeds are 0, 1 and 2. The cotangent has unit L2 norm.
Source hashes bind every library Python file plus the measurement script;
the manifest records the dirty worktree honestly, not as a clean release.

| D/L/T | Device payload B | Portable B | Analytical boundary B | Boundary / payload | Maximum gradient error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/2/96 | 36,084 | 38,298 | 73,728 | 2.04× | 7.30e-7 |
| 128/4/96 | 87,588 | 91,342 | 245,760 | 2.81× | 2.92e-6 |
| 256/4/192 | 329,928 | 333,715 | 983,040 | 2.98× | 1.49e-6 |

All cells pass exact logits, integer route consistency on actual eager inputs,
packed/unpacked/portable field checks, and external-cotangent/public-autograd
parameter VJPs at predeclared atol=rtol=1e-5. The route oracle shares canonical
primitives; it is not an independently implemented algorithm. The analytic
comparator is `(L+1)*B*T*D*4`, excluding common cotangent, loss, parameters,
optimizer, metadata and transient workspaces. No inference about measured peak,
traffic, latency or statistical task generalization follows from these ratios.

Run `python -m manuscript.tools.historical_cpu_evidence --check` to verify the
selected record/table against their pinned original sources. The new integer
EF library module makes the original direct current-source `--check` stale,
as intended. A fresh measurement must use new paths and be explicitly selected,
not overwrite the old record or change its source hashes.

The [CPU source snapshot](source_snapshots/cpu_2026-09-10/manifest.json) preserves
71 exact files, including the 65 then-current library Python files and original
strict checkers. It binds five selected CPU records and two existing generated
tables. Its manifest SHA256 is
`85dbc27660bb51eb09f1e17fdea0240b9d8e5efc97b1e41934f972d3dd42aba4`.
The wrapper verifies pinned archive membership/content before materializing a
private temporary source tree and invoking only the original check routines.
No CPU measurement is rerun, no source check is waived, and later library code
is not assigned historical results. PDF verification now explicitly uses this
historical mode. The original/compat AdamW records have older helper versions
and are not newly claimed to be covered by this selected snapshot.

The selected [layout remeasurement](../evidence/retained_state_layout_2026-09-09.json)
reproduces the same numerical results after moving the source-bound helper to
`manuscript/tools/`. The original, handoff and cleanup records remain unchanged
in `evidence/`; they identify earlier source snapshots, not competing current
tables. To reproduce without overwriting either evidence or manuscript tables:

```bash
python manuscript/tools/measure_retained_state.py --run \
  --output manuscript/build/reproduction.json \
  --table manuscript/build/reproduction.tex
```

Use new output names for each run. A maintainer must explicitly select and
review new evidence before regenerating the main table and review PDF.

## AdamW numerical diagnostic

The [selected diagnostic record](../evidence/adamw_diagnostic_benchmarks_2026-09-09.json) and
[reproducer](../../scripts/route_certificates/optimizer_diagnostic.py) isolate the
previous update-parity failure. This is an untimed local CPU diagnostic, not a
benchmark or a task-quality result. It uses PyTorch 2.10 native Inductor, FP32,
width 16, one layer, vocabulary 11, batch two, sequence 12 and learning rate
`1e-3`. **Only model/reverse paths are compiled; stock AdamW stays eager.**

- On identical initial embeddings, the first differing primitive is the mean
  reduction in normalization (maximum `1.49e-8`). Initial routes agree. The
  same-logits loss cotangent agrees exactly; the same-tape/same-cotangent VJP
  differs by at most `1.49e-8`. Equal gradients produce identical AdamW updates.
- A head feature selected by every class or none in each row has zero selected
  bias derivative by cross-entropy's common-shift invariance. Floating-point
  cancellation leaves tiny signed residuals. This is a structural control,
  not permission to threshold small gradients.
- The first adaptive update is `-lr*g/(abs(g)+eps)`, with sensitivity at most
  `lr/eps`. Gradients `-2.586e-8` versus `1.703e-8` can therefore produce about
  `1.35e-3` parameter disagreement at the default epsilon. Subsequent route
  changes can separate independently evolving trajectories further.

The exploratory sweep retains **20 fresh-batch updates × seeds 0/1/2 × ordinary
compiled autograd, unpacked and packed replay**, with unchanged `atol=rtol=1e-4`
loss/gradient/parameter checks:

| AdamW epsilon | Bounded trajectory outcome |
|---|---|
| `1e-8` (unchanged default) | Fails every seed |
| `1e-6` | Fails seeds 0 and 1; a shorter passing test was insufficient |
| `1e-5` | Passes every seed and compiled path |
| `1e-4` | Passes every seed and compiled path; maximum parameter difference `1.37e-6`, gradient difference `5.36e-7` |

Per-step identical-state controls separate kernel agreement from independent
trajectory sensitivity; they do not reset the real measured updates. The
explicit `--adamw-eps 1e-4` recipe is a **bounded CPU mitigation**, not a new
optimizer, proof that default parity is fixed, or an optimal learning-rate/epsilon
choice for a real task. All comparison arms must use the same setting. No
tolerance was loosened, and no library derivative or PyTorch default was changed.
GPU, task-scale models and task-quality validation remain open. The next phase
is target-device matched benchmarks, then reconsidering LM pretraining after
its head boundary is ready; those studies have not been run here.

```bash
# Standard-library check against pinned historical sources; no experiment:
python -m manuscript.tools.historical_cpu_evidence --check
# Rerun with a new output path; never overwrite the checked record:
python -m scripts.route_certificates.optimizer_diagnostic \
  --steps 20 --eps 1e-8 1e-6 1e-5 1e-4 \
  --output manuscript/build/adamw-reproduction.json
```

The record binds the diagnostic, benchmark and library sources. Its numerical
failures are valid negative evidence, not integrity-check failures. The review
PDF freshness check also verifies this record and its inputs.
The [initial diagnostic record](../evidence/adamw_diagnostic_2026-09-09.json)
and [compatibility refresh](../evidence/adamw_diagnostic_compat_2026-09-09.json)
are preserved unchanged. The selected run repeats the same numerical protocol
after extending the diagnostic's shape CLI and the shared benchmark controls;
it does not change optimizer arithmetic. Historical records bind their original
source versions; the current-source check deliberately rejects a stale binding.

### Wider local follow-up

The [wider diagnostic](../evidence/adamw_diagnostic_wider_2026-09-09.json) keeps
the same CPU/FP32/Inductor optimizer protocol and unchanged tolerances, with
`D=32`, two layers, sequence 24, batch two and vocabulary 11. Explicit
`eps=1e-4` passes 20 fresh-batch updates for all three seeds and all three
compiled paths. Maximum parameter difference is `6.56e-6`, gradient difference
`2.02e-7`, and identical-state gradient difference `1.96e-7`; recorded route
differences are zero. This extends the local check, not the task-scale or
accelerator claim. The PDF freshness check binds both selected diagnostics.

```bash
python -m scripts.route_certificates.optimizer_diagnostic \
  --width 32 --layers 2 --sequence 24 --steps 20 --eps 1e-4 \
  --output manuscript/build/adamw-wider-reproduction.json
```

## Current CPU comparator pilot

The [ten-case pilot manifest](../results/route_certificates/runs/device_route_training/172834f0-c1cf-42c1-ab87-ec0cdf6f0da9/manifest.json)
records the six-arm v2 protocol: eager/compiled autograd, eager/compiled
checkpointing with layer/grouped/selective candidates, and compiled
unpacked/packed replay. This is one FP32 CPU seed at width 16, one layer,
batch two, sequence 12 and vocabulary 11. All cases share eager AdamW
(`lr=1e-3`, `eps=1e-4`, betas `0.9/0.999`, decay `0.01`). Each fresh worker
performs one startup update, two warmups and five timed updates on the same
batch, followed by a comparison of its measured final state with eager.
All ten cases pass; executable sources remain unchanged throughout the run.

The fastest observed baseline is compiled selective checkpointing at
`0.349 ms/update`; packed replay measures `0.375 ms` and unpacked replay
`0.265 ms`. **There is no demonstrated packed-replay speed advantage in this
pilot.** Five fixed-order repetitions are not a multi-seed systems study.
Compiler disk caches were not cleared, so first-call time is not cold compile
time. CPU RSS is not peak tensor memory. The separate eager successor-task
learning control passes but needs no context and does not measure LM quality.
The model's actual receptive field is three tokens, independently of the
12-token timing shape.

The [stdlib reducer](../tools/summarize_training_pilot.py) pins the raw manifest,
checks its source inputs and protocol, and generates the appendix table. It
also verifies the separate negative case below. Run from the repository root:

```bash
python -S manuscript/tools/historical_cpu_evidence.py --check
# A new append-only timing pilot, not an overwrite or a final study:
python -m scripts.route_certificates.training_benchmark --device cpu \
  --adamw-eps 1e-4 --warmup 2 --reps 5 --timeout 60 --learning-steps 50
```

### Native selective-checkpoint limitation

Selective policy calls in the successful one-layer pilot encounter
`amax`/`amin` reductions. Eager observations are runtime policy calls;
compiled observations are compiler-trace policy calls, not kernel counts or
measured retention/recomputation savings.

A separate [two-layer failure record](../results/route_certificates/runs/compiled_selective_followup/907579f2-5b1c-42a5-93b0-0d1867e8577e/manifest.json)
and [worker log](../results/route_certificates/runs/compiled_selective_followup/907579f2-5b1c-42a5-93b0-0d1867e8577e/worker.log)
preserve a native CPU/Inductor `SIGSEGV` (`returncode=-11`), before the worker
produces a measurement row. That diagnostic uses width 16, two layers, sequence
12, batch two, vocabulary 11 and momentum SGD (`lr=1e-3`), not the AdamW pilot
recipe. A fresh-process native test also locates the failure in parity, before
coverage instrumentation. The compiler failure's underlying cause is unresolved;
it is neither a mathematical VJP diagnosis nor proof that every selective
configuration fails. Optional native-policy tests now isolate it and visibly
skip unavailable/crashing cases; normally completed parity failures still fail
tests. The benchmark preserves failed workers and excludes them from winners.

## Historical appendix table

Canonical inputs are
`manuscript/results/route_certificates/actual_morphssm/exp1_saved_state/529c45f4-7274-4999-b35c-c59d1db917e6/{saved_state.jsonl,manifest.json}`.

The archived manifest identifies:

- Timestamp: `2026-08-19T10:39:26Z`.
- Command: `python -m ontogenic_learner.ontogenic_v2.exp1_saved_state`.
- External source commit: `7b29124b63c30abadb0288b095db0cc85c5400f7`, with
  nonempty dirty-patch digest `24b605604f6bdee19f0dc4d9909fc6189c3b9d21991df767b5aa6c8284711f91`.
- TikiTorch commit: `ee3c775281b6046781144b0768a050ac26dfbc19`, with the empty
  dirty-patch digest. This is not the current release candidate.
- CPU, macOS arm64, Python 3.11.5, PyTorch 2.10.0, NumPy 1.26.4; batch 1,
  vocabulary 38; `(d,L,T)` of `(64,2,96)`, `(128,4,96)`, `(256,4,192)`.
- No input digest is supplied (`inputs: []`). The dirty-patch digest identifies
  a patch but is not itself a reconstruction of its contents. Clean native
  reproduction remains required for a stronger reproducibility claim.

`manuscript/tools/generate_tables.py` checks pinned SHA256 identities for every selected raw
input, rejects duplicate/missing/unexpected selected cells, checks the middle
cell against the manifest, derives boundaries, and ignores the legacy
`ratio_*` fields. Run:

```bash
python manuscript/tools/generate_tables.py --check
python -m pytest -q tests/test_manuscript_tables.py
```

The exact source hashes, source revisions, environments, accounting equations
and selected numerical rows are in
[`source/generated/table_provenance.json`](../source/generated/table_provenance.json). Generated
LaTeX files contain only `tabular` environments; captions remain responsible
for the boundary qualifications.

| d/L/T | Payload B | Portable B | Checkpoint core B | Supplied-cotangent ratio | Loss-inclusive ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64/2/96 | 36,084 | 37,431 | 73,728 | 1.97× | 1.69× |
| 128/4/96 | 87,588 | 89,799 | 245,760 | 2.74× | 2.48× |
| 256/4/192 | 329,928 | 332,173 | 983,040 | 2.96× | 2.79× |

These ratios compare a portable serialized certificate, including its
directory, with the *declared checkpoint-retained set*. The archived checkpoint
quantity is a shape-derived layer/embedding boundary formula, not an executed
best-policy checkpoint allocator measurement. The checkpoint core is
`checkpoint_retained - boundary_logits`. The loss-inclusive comparison adds
FP32 logits and int64 targets to both sides. Parameters, optimizer state,
common supplied cotangent, transient workspace, allocator overhead, and
checkpoint recomputation are excluded. Do not infer a peak-memory reduction
from these values.

The newer benchmark measures actual checkpoint policies (layer, grouped and,
when available, selective) plus eager, compiled-autograd, unpacked-route and
packed-route execution. It must generate its own source-bound records. It
cannot reuse this table's byte counts, legacy Python timing or historical tie
contract as measurements of current device training.

## Hardware evidence qualifications

1. **D512 match scope.** The matching JSON files have one 30-update observable
   record, not a complete weight snapshot. A CRC is a collision-prone summary;
   equality plus sampled predictions is not proof of full-state identity.
   The longer invocation records do not supply corresponding golden trajectories.
   Archived HLS/re-synthesis diagnostics in `manuscript/results/a2_power/` are not a substitute
   for the exact final D512 routed-build manifest and reports.
2. **MNIST evaluation boundary.** The driver performs host evaluation of
   fabric-trained weights. Epoch wall times include weight readback and test
   evaluation; they are not isolated training throughput. The archived optional
   full-training one-step verifier was not run (`VERIFY=` in `pctrain.log`).
   The archive has no paired frozen evaluation tensors or complete matched
   training trajectory.
3. **MNIST recipe compensation.** The historical kernel's normalization shift
   differed by one bit from the intended recipe. Targets were doubled from 8
   to 16 without rebuilding the AFI. This is a disclosed recipe change, not an
   independently verified clean rebuild. The corrected fabric run first reports
   test accuracy at epoch 1; an initialization-to-chance trajectory was not logged.
4. **MNIST reporting discrepancy.** The narrative report describes the 0.896
   recipe comparison as eight epochs/batch 128, while `recipe_diag.txt` reports
   0.8961 after two epochs with batch 32. Neither is a controlled denominator for
   the fabric run. Omit that comparison from the manuscript. The final-epoch
   74.75% result is not the best epoch (epoch 4 is 74.79%); label selection explicitly.
5. **DSP semantics.** The MNIST report describes one DSP bound as an adder and
   zero DSP multipliers. A data-multiplier-free recipe and zero physical DSP
   blocks are different claims. Do not assign zero physical DSPs to current
   software or to a D512 build lacking its exact final report.
6. **D128 scaling.** `manuscript/results/lm_shakespeare/fig_hallmark_lm.md` reports 14,236,
   26,522 and 53,001 token/s (1.86× and 3.72× relative to the single instance).
   WNS is −0.020/−0.097/−0.173 ns. These are not timing-closed measurements or an
   iso-model GPU comparison. Omit the ladder from the main paper; appendix
   discussion of the failed timing gate is sufficient.
7. **Power.** Core-rail logs and WNS-derived projections do not measure
   synchronized full-card energy. No present artifact supports a board-to-board
   energy win. Omit numerical energy-superiority claims and projected N=8 results.

## Historical language-quality correction

The selected endpoints in `manuscript/results/lm_shakespeare/strictH_bpe_anneal_s9decay_6k.jsonl`
(final row: 2.7929592359328628 bpb; 1,004,032 parameters; step 5999) and
`manuscript/results/lm_shakespeare/bpeA_dense.jsonl` (2.2731051237221513 bpb; 770,816
parameters; step 1499) round to the older hallmark figure's 2.793/2.273 rows.
`scripts/make_hallmark_figure.py` selects those tags and uses 24×96 tokens per
update, giving nominal endpoint budgets of 13.824M/3.456M (rounded 13.8M/3.46M).
They are **not parameter-matched** as the old summary states. The MorphSSM
record begins at tau=0.5 and ends at zero; it is an annealed, not fully hard,
training trajectory. The report's ROM/LZC audit waivers do not certify it under
current no-waiver semantics. These facts are disclosed as a negative historical
quality observation, not a matched-compute closure penalty or a current-code
systems measurement. Preserve the raw files and distinguish this correction
from their historical prose. The 2.793-bpb checkpoint was not trained on F2.

## Placement and outstanding gates

**Main:** TikiTorch's programmable hard-operator/differentiation/audit interface,
the exact selected-VJP boundary, counterexamples, current implementation,
source-snapshotted retained-state table with provenance, and the protocol required to
test practical utility. Keep the distinctive claim about useful derivative
state and explicit execution contracts, not generic tropical arithmetic novelty.

**Appendix:** proof details, complete operator/auxiliary-field inventory,
historical portable-table accounting, replay failure cases, hardware feasibility records,
normalization/target caveat and failed timing/comparison gates.

**Omit as positive results:** unmatched accelerator speedups, energy projections,
full-state bit-exact language, broad activation-free training, extrapolated GPU
capacity, obsolete test counts as current validation, and legacy comparisons
that mix portable tape state with current device packing or optimizer/loss state.

The original default-AdamW trajectory failure remains valid; the numerical
diagnostic above explains it and records a separately configured bounded CPU
mitigation. Momentum-SGD passes are not AdamW equivalence. CUDA validation and real-task multi-seed matched
experiments are unrun. A clean source-bound submission artifact, frozen
supported software stack and named author/affiliation/venue decisions also
remain necessary. Physical hardware work is required only for claims actually
retained; a focused software paper need not make unmeasured energy or FPGA
superiority claims to justify its contribution.

Distribution clearance is an independent gate:
`docs/release_provenance.json` still records four unresolved AWS-derived RTL
records and retained-history review. Neither manuscript improvement, a package
build, nor passing tests resolves those rights/authority questions.

## Artifact preservation and layout

The research archive moved from root `results/` to `manuscript/results/` on
2026-09-08. Reusable Python code stays in `tikitorch/`; active research runners
stay in `scripts/`. There is no root compatibility symlink. Use a Git source
checkout for research: neither the paper nor its archive ships in the library's
wheel or source distribution.

### What was preserved

The move was checked with SHA-256 before and after: all 686 locally present
files, including ignored files, were byte-identical immediately after relocation.
The durable [relocation inventory](../evidence/artifact_relocation.json) covers the 544
Git-tracked artifacts from commit `197b2644441877a9662d5a33b84b118bef169bc5`:
538 remain byte-identical and six Markdown files have only relative hyperlink
adjustments. No numerical results, logs, embedded commands, JSON records,
checksums or cloud object keys were rewritten. Ignored local files are not
promoted into the curated inventory or distribution archives.

From the repository root, check this boundary without Git, PyTorch or network access:

```bash
python scripts/check_artifact_layout.py
```

This verifies frozen paths, hashes and executable modes and rejects recreation
of root `results/`, including a broken symlink. Future artifacts may be added;
do not overwrite frozen evidence to make a check pass. Keep corrections in new
records with an explanation and source identity. Six navigation exceptions are
recorded explicitly, not a general permission to change historical Markdown.

#### Paper-directory consolidation amendment

Later on 2026-09-08, root `papers/` was consolidated into this workspace;
see the [archive map and source preservation record](../archive/README.md).
The sole results-archive amendment is the hyperlink in
`results/route_certificates/README.md`: its target changed from
`../../../papers/route_certificates/README.md` to
`../../archive/route_certificates/README.md`. Its scientific text is unchanged.

This is an explicit amendment to one of the same six navigation exceptions,
not permission to regenerate evidence hashes. The inventory-recording path
reconstructs only the exact permitted transformation from the original Git blob; the
inventory retains the original source commit and digest and updates only that
file's current digest. The pre-amendment inventory is preserved at commit
`1b4b00308a53919dee6afbc8fdca86011659d348`.
Ordinary read-only verification still needs no Git or network access.

- Previous SHA-256: `1c5ae9a7dbbfd7323235bc99b49b64da6a8e211023f58c5fba2f05d99c7a3e3e`
- Current SHA-256: `9809baad63f34c2a542d17b0d58efc634fc1d9755d336f1b6566ae389c0dcb54`

All other frozen results files and all numerical records remain unchanged.

### Reading and reproducing historical records

Paths inside historical records retain the original `results/...` spelling.
For file lookup, translate that prefix to `manuscript/results/...`. Active
documentation, readers and runner defaults use the new location. S3 bucket
links and object keys are unchanged: local relocation does not move cloud data.

Some checksum lists use `results/...` paths; those can be resolved with
`manuscript/` as the working directory. That does **not** make arbitrary archived
shell commands runnable from there: scripts and hardware sources remain at
their existing repository locations. Treat old commands as revision-bound
records; review current runner instructions and use a new output directory.
Do not run cloud, graphics-generation or hardware scripts merely to inspect
the archive; some require credentials, paid services or external toolchains.

Relocation is not reproduction or publication clearance. Historical hardware
reports do not certify current Python code. Follow the evidence boundaries above
and the [release gates](../../docs/release_validation.md); third-party source and
retained-history review remain open.

The 2026-09-09 sharing cleanup consolidated the handoff and readiness checklist
into [the manuscript README](../README.md), combined the plan/protocol in
[PLAN.md](PLAN.md), and moved build sources/helpers into `source/` and `tools/`.
The relocation inventory moved to `evidence/artifact_relocation.json` unchanged.
Historical evidence and archived scientific sources were not rewritten.

## Fixed-configuration int32 gradient-range lemma (2026-09-09)

This is an arithmetic certificate for the source identities in the
[CPU error-feedback record](../evidence/fpga_lm_efsgd_2026-09-09.json), not a
new hardware measurement. It covers the existing
[balanced LUT gradient](../../tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp#L427-L456)
and [selected-route backward](../../tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp#L476-L575)
under the following fixed contract: four layers, width 512, vocabulary 128,
one 128-position window per update, batch 1, loss shift 2 and error-feedback
denominator `2^18`. Weights and stored activations have the declared signed-int16
representation; the validated ROM has entries in `[1,65536]`. Routes come from
the causal forward pass; backward scratch is zeroed and positions are processed
in descending order. Each parameter accumulator starts with a canonical
remainder `|r| <= 262143`, and the remainder is reduced after **every** window.
Signed right shift implements arithmetic/floor division. Parameter sharing and
array separation are exactly those of this source, not a generalized graph.

**Lemma.** In unbounded-integer execution of that contract, every backward
int32 intermediate and every parameter-accumulation prefix, including its
starting remainder, has absolute value at most **1,361,832,959**. Consequently
these operations cannot overflow signed int32 under the stated contract.

**Proof.** For one position, let `p_v = floor(e_v / 2^sh)` and `A = sum_v p_v`.
The implemented normalization gives `A <= floor(S / 2^sh) <= 65536`. The
balanced target subtraction therefore gives `||g_logits||_1 = 2(A-p_target)
<= 131072`; over the window, the head's incoming mass is at most
`B_4 = 128 * 131072 = 16,777,216`. Selected head, projection and temporal
scatters do not increase global L1 mass. Causality makes descending
time/layer order topological, so each complete temporal-adjoint row is centered
once, not repeatedly along a same-layer time recurrence.

For a width-512 integer row `u`, put `s = sum(u)` and
`C(u) = u - floor(s/512) * 1`. Since
`512 * |floor(s/512)| <= |s| + 511`,
`||C(u)||_1 <= 2||u||_1 + 511`. If `B_l` bounds the complete residual-node
adjoints at depth `l`, summed over all positions, the residual skip and centered
branch consequently obey

```text
B_(l-1) <= 3 B_l + 128 * 511 = 3 B_l + 65,408
B_0     <= 81 * 16,777,216 + 40 * 65,408 = 1,361,570,816
|a|     <= B_0 + 262,143 = 1,361,832,959 < 2,147,483,647.
```

This also bounds prefixes: each scatter/reduction prefix is dominated by the
absolute mass of its contributing sources, without relying on cancellation.
Parameter gradients are separate additive sinks, never inputs to `gR`,
`grad_MC` or the temporal-adjoint stream, so their remainders are added once
per sink rather than amplified through the graph. The largest pre-centering
bound is `B_1 = 453,835,136`; hence the subtraction itself satisfies
`|gbs[c] - mean| <= B_1 + ceil(B_1/512) = 454,721,533`. The subsequent residual
addition and the shared embedding-gradient prefixes are covered by `B_0`.
Thus intermediate reductions, not only final gradients, fit int32.

**Limits.** Forward arithmetic also fits int32: the largest width-512 int16
sum has magnitude 16,777,216; pairwise candidates have magnitude at most 65,536,
projection-plus-bias at most 98,304, and the LUT sum at most 8,388,608. These are
**not** int16 no-clipping guarantees. The native adapter still uses
[int64 forward/update preflight](../tools/fpga_lm_efsgd_v1.cpp#L76-L143),
rejects clipping, and runs with nonrecovering undefined-behavior sanitization.
The lemma supports considering existing int32 on-chip gradient/remainder
storage in a separately implemented FPGA optimizer; it does not remove those
checks, implement the new state ABI, or establish HLS scheduling correctness,
synthesis, timing, resource use or physical checkpoint equivalence. Larger
batches, different losses, uncanonical state or accumulation across windows
need a new argument. At five layers this recurrence gives 4,084,777,856 before
the remainder: the certificate becomes inconclusive, not proof that overflow
must occur. No frozen source, experiment or evidence record was changed to
obtain this bound.

<a id="conditional-sparse-execution-opportunity-not-implemented-or-measured"></a>

## Conditional sparse execution: prototype, no measured savings

The forward-preserving initializer offers a narrower hardware hypothesis than
"hard routing makes forward sparse." Suppose a projection row retains one
explicit designated weight `w_j`, with every other weight exactly `-8192` and
its EF remainder zero. For the actual convolution vector `c`, compute
`M=max_i c_i` once. The strict guard

```text
c[phi(j)] + w_j > M - 8192
```

certifies that the designated edge is the unique exact winner. Checking every
row costs O(D), rather than evaluating D squared candidates. An analogous
head guard `R[A_v]+w_v > max_i R[i]-8192` costs O(D+V). This is a sufficient
condition, not a necessary one: equality or a failed bound needs exact fallback
with the original tie rule. Apply guards before bias/clipping and preserve all
original range checks. The general dense forward still scans its candidates.

If these guards hold at **every position of every training window**, MP/head
parameter-gradient writes touch only designated coordinates. Off-edge gradients and remainders
then stay zero, so coordinatewise EF without weight decay leaves off-edge
weights at -8192 by induction. Under that conditional invariant, embedding,
temporal, bias, projection and head state contains respectively
65,536 + 14,336 + 2,048 + 2,048 + 128 = **84,096 explicit weight/remainder
pairs**, versus 1,196,032 dense coordinates. This excludes activations, routes,
backward scratch, LUT/configuration and fallback storage; it is not a claim
that all dense parameters remain independently active during learning.

Saved checkpoint margins cannot establish that whole-trajectory invariant.
A separate, declared instrumented replay must compare complete dense states
and measure guard failures. If a guard fails, exact fallback must preserve or
replay the window from its pre-update full state. A newly selected off-edge
coordinate needs an explicit remainder **even before its weight changes**;
discarding that remainder or silently forcing the designated route changes
the optimizer. Test ties, exceptions, resume and dense-state reconstruction.

The next implementation milestone is therefore a fixed-configuration,
stateful EF C-simulation interface and guarded-path prototype, retaining the
frozen dense CPU oracle. The existing physical interface reloads weights and
clears gradient state; it cannot persist/export these remainders. Its 8 MiB
staging aperture also cannot hold the 9,569,360-byte dense numerical state in
one transfer. A new versioned, atomic paged state protocol must reject missing,
reordered or incompatible pages and compare weights, remainders, step, loss
and configuration after multiple distinct windows and import/resume.

Only exact state/guard/fallback validation should inform the choice of dense
or compact HLS architecture. Synthesis, timing, new physical-image identity and
full-state FPGA/reference equivalence remain separate gates. No guard lifetime,
memory saving, speed, energy or cost advantage has been measured here. The
same exact optimization must also be considered for fair CPU/GPU comparators.

### Separate FP64 guarded-forward prototype

The experimental [guarded forward](../tools/fpga_lm_guarded_forward.py) now
implements a narrower exact-CE CPU path, **not the EF or FPGA implementation
above**. No original dense training producer imports or uses it; the separate
implementation-parity replay below now does. It accepts complete FP64 weights,
verifies exactly one nonconstant entry per MP/head row with off-weight
`C=beta*(-8192)`, and checks the strict bound using the current selected weight
and actual temporal/hidden vector. Unsupported structure falls back to the
original complete forward; a failed or equal bound falls back to the original
dense selector for that operation. It never forces a route to remain fixed.

The prototype preserves the full-width real mean, explicit centering/addition
order, temporal max/min selection and first-index ties. Its outputs can feed
the unchanged dense manual reverse and dense Adam. Synthetic tests compare
logits/routes/offsets exactly, check independent reverse derivatives and cover
ties, signed zeros, changed off-weights, unsupported states and D512 geometry.
Guard bounds are not runner-up margins; the fast path returns no margin array.
The whole-model fallback may retain genuine margins from the original selector.

For Adam, constant off-support state is conditional on **both** zero gradients
at every query and initially zero first/second moments, with no weight decay
or external mutation. Keep original per-region step counters even on zero-
gradient steps. Weight structure alone does not prove zero moment state.
Real centering can propagate gradients into every channel, so this argument
does not justify restricting embedding or temporal-gradient coordinates.
The prototype deliberately retains full dense parameters, reverse arrays,
optimizer states and checkpoints.

This is preparation for larger CPU quality tests, not a measured acceleration,
memory saving or deployment. The separately bound full-trajectory check below
now verifies equivalence on the existing rate-only trajectory. The original
rate-only producer remains entirely dense; new experiments must retain guards,
exact fallback, source binding and their own declared numerical checks.

### Guarded full-trajectory parity (2026-09-13)

The [completed parity receipt](../evidence/fpga_lm_guarded_rate_parity_2026-09-13.json)
verifies the guarded implementation against the original dense trajectory,
after pinning its successful independent numerical audit. Every one of the
**524 forward traces, 128 full-batch gradients and 129 complete Adam states**
matches byte-for-byte, including tensor dtypes, signed zeros, moments, step
counters and optimizer options. All 512 manual window VJPs are recomputed in
the original summation order. The verifier applies those **recomputed identical
gradients**, not a substituted saved gradient, before checking each full state.
All twelve original score-window readouts match too. This is implementation
parity against an independently audited trajectory, not another independent-
autograd oracle or new training/generalization result.

Every covered projection/head operation is certified: **268,288 projection
operations and 67,072 head operations** use the guarded path, with **zero**
dense-operation or whole-model fallbacks. Temporal hard selections remain
dynamic and retain their original first-index tie rules. The guards establish
the shortcut at each actual operation; unchanged saved supports alone would
not have proved it. This result covers these saved training/readout points,
not a guarantee that future data or trajectories will avoid fallback. Full
parameter/gradient/Adam arrays are retained, and structural validation still
inspects the full weight matrices. No optimizer-state compression is measured.

The invocation starts at **20:33:28 UTC** and completes publication/postchecks
at **20:35:22 UTC**, exit 0, in **114.021997 UTC / 114.022813 monotonic seconds**,
within its 3,600-second deadline. The numeric parity replay itself takes
99.372646 elapsed / 98.353598 CPU seconds. Selected non-exhaustive timing scopes:

| Replay scope | Calls | Elapsed seconds |
| --- | ---: | ---: |
| Guarded forward | 524 | 52.786 |
| Manual VJP and batch accumulation | 512 | 25.911 |
| Adam update | 128 | 0.627 |
| Saved-payload loading | 270 | 6.682 |
| State/metadata validation, including some I/O | 129 | 5.731 |

These are feasibility timings for this local CPU replay, not a controlled
performance benchmark. The buckets omit some comparisons, hashing and
orchestration. **Do not divide the independent audit's 660 seconds by this
replay's 114 seconds to claim a speedup**: the former independently rederives
gradients with full-matrix forward/autograd, whereas this replay uses the
guarded forward and shared manual reverse. No matched dense, GPU or FPGA timing,
peak-memory, energy or cost comparison was performed.

All 67 source/test files, original rate payloads, ancestor closure and successful
audit capsule remain unchanged. The new private record contains four files
(manifest, saved declaration, verification facts and complete parity results),
without copying the full historical checkpoints. Receipt SHA-256:
`7a21df2543ece923fbdcd388ae326770db0aa0ad6623d1ff578c505ae4ae5a82`.
Saved declaration SHA-256:
`1edfd03ae5788acee8e75765d8994f970e3bd4ca01ce0ff0cf1ad5733bb818d0`.
There are zero new fitting exposures, distinct optimizer trajectories,
development/test readouts, native/FPGA calls or paid resources. This passes
the implementation-parity prerequisite for the separately declared broader
development comparison below.

### Varied-text exact-CE development comparison (2026-09-14)

The [six-cell producer record](../evidence/fpga_lm_development_2026-09-14.json)
is complete, including ordinary exact saved-gradient replay of every Adam
transition. The [separate independent numerical audit](../evidence/fpga_lm_development_audit_2026-09-14.json)
**also passes**, including all source/input/publication postchecks. The
[protocol and execution freeze](PLAN.md#varied-text-exact-ce-development-comparison-v1-declared-2026-09-13)
precede every new readout; no recipe was adapted during this comparison.

Both architectures start from their original fitted initialization and empty
Adam state, not the earlier tiny-batch trained endpoints. Initial development
probabilities agree to a maximum absolute error of 5.56e-16; both start at
CE **2.6854248377** and accuracy **25.5493%**. Each of the three paired order-only
replicates makes one pass over the same 1,024 training windows, with 256
four-window updates (128 positions/window). Tropical Adam lr=0.01 and fitted-prior
GRU Adam lr=0.001 remain fixed, CPU FP64, one Torch thread, exact mean CE,
no decay or clipping. Evaluation is only the previously consumed development
windows **88–151**; reserved validation **152–863 and all final test remain
unevaluated**.

| Order seed | Tropical final CE | Prior-GRU final CE | Tropical minus GRU | Tropical context-off CE penalty |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2.608314 | 2.371154 | 0.237159 | 0.255914 |
| 1 | 2.628976 | 2.372431 | 0.256545 | 0.254427 |
| 2 | 2.625862 | 2.373935 | 0.251928 | 0.245728 |
| Mean | **2.621051** | **2.372507** | **0.248544** | **0.252023** |

The inherited alpha-one bigram has CE **2.6853922339**; the fixed repeated-count
alpha-one comparator (`2*C-1`, equivalent to one-pass alpha=0.5) has CE
**2.6892633687**. Tropical improves its own initial loss in every order and
beats the stronger bigram in mean by **0.0643415567 nat**, passing the declared
0.02-nat margin. But **all three paired GRU margins fail** (each allowed at most
0.05 nat), so the overall development gate is **false**. There is no selected
best-seed success or promotion to FPGA scaling.

Tropical order sample SD is 0.011140 nat (range 0.020663); GRU SD is 0.001392
(range 0.002780). Mean accuracies are 30.5013% and 36.1694%, respectively.
Both have positive context penalties; the GRU mean is 0.382129 nat. These
are descriptive order summaries on the **same 8,192 development positions**,
not three independent datasets, token-iid significance tests, or a final-test
result. This shows improvement beyond the inherited prior for this selected
one-pass recipe, with an unresolved comparator gap; it does not establish
convergence or an architectural capacity limit. The architectures, receptive
fields, parameterization and prior tuning histories are not matched.

The producer starts **2026-09-14 00:02:41 UTC**, completes all publication and
postchecks at **00:25:23 UTC**, and exits 0 after **1362.503917 UTC /
1362.495043 monotonic seconds**. It retains **5,529 sealed payload files /
4,985,424,739 payload bytes**, plus the inventory seal and public record.
All **1,536 batch gradients, 1,536 updates, 1,542 complete Adam states,
896 model score readouts and 128 baseline score entries** are accounted for.
Ordinary checking recomputes losses from saved logits and the same baseline
entries, with no new model forward calls; independent checking is separate.

A separate read-only file/metadata audit rehashed all 5,529 payloads and 69
sources and independently checked orders, original-state identities, data
boundaries, exposure counts and gate logic. All recorded guard rows also match
their saved NPZ arrays: across **3,520 tropical forwards**, they report
**1,802,240 projection / 450,560 head fast operations**, with zero dense-operation
or whole-model fallback. The completed numerical audit also rederives these
counters from actual guarded forwards and checks every corresponding independent
dense trace. They do not measure hardware utilization or speedup.

Across the six trajectories there are **786,432 new gradient-pair exposures**
over 131,072 distinct training pairs. Each endpoint also inherits the original
131,072 fitted-pair history; summed endpoint-history exposures are 1,572,864.
The original count fit occurred once; count reconstruction and auditing are
verification computation, not new model fitting. No native training, FPGA
execution, paid resource, reserved-cohort evaluation or public optimizer-default
change occurred. Local timing is not a matched accelerator or cost-to-quality
benchmark.

Producer public SHA-256:
`2ec2fdef77c5e99f14e0402bafb3f0f1219b4c4d9cb084be737a1c035db65149`.
Producer inventory-seal SHA-256:
`9d7ddafbe4b77f6e795e92216dcd943f6067990bd7f93b44839e509c5c809f84`.
Saved producer declaration SHA-256:
`b748b884d351dbac0f9c2a6d0dc96f2e9954ce1cb3285c25e463aa2038c0cc18`.
The new runner and test file, plus all 67 inherited sources, remain frozen
through audit. Its 110 new synthetic tests pass; 81 targeted manuscript/evidence
tests and the strict documentation build also pass. Raw tensors are in ignored
local build storage, not a portable downloadable benchmark bundle.

The subsequent standard repository regression invocation (`python -m pytest -q`)
passes **4,651 tests, with 14 skipped and 10 deselected**, in **438.31 seconds**.
This includes the 110 development-runner tests and 78 synthetic tests of the
forward-only follow-up candidate; the repository defaults exclude `slow` and
`kernel` tests. Its one warning is the intentional unavailable-backend fallback
test. This is software regression coverage, not additional scientific training,
an independent model-quality confirmation or hardware execution.

The independent audit starts **2026-09-14 00:25:47 UTC** and exits 0 after final
publication checks at **01:36:16 UTC**, in **4229.073073 UTC / 4229.049091
monotonic seconds** (70 minutes 29 seconds). Its recorded numerical verification
finishes earlier at 01:36:10 UTC (4222.464867 UTC seconds); that is distinct from
the terminal publication/postcheck completion. It independently verifies all
**3,072 tropical window VJPs, 768 tropical batch-gradient sums, 768 explicit-GRU
batch gradients, 896 model score readouts and 128 baseline entries**. There
are 3,520 independent tropical forward windows, 768 four-window GRU queries
and 448 single-window GRU scores: **7,040 independent model-window equivalents**.
Another 3,520 guarded forwards verify implementation traces/counters separately.
Saved gradients drive the exact replay of all 1,542 full Adam states after
independent gradient comparisons. No new learning trajectory or data cohort
is introduced by this audit.

All 69 source files, 5,529 producer payloads, original receipts, capacity/input
provenance and ancestor artifacts remain unchanged through the audit's final
checks. Successful audit public SHA-256:
`500147cf9f03d8ac4b91362c74961ab8bd14aae438da2d6676a0f8a9b9a85e0a`;
private audit-manifest SHA-256:
`b3744eb06cd95a465c45bb2b89ede97ba84e8f90900085475e413396f73932e5`.
The audit confirms the reported **failed** development quality gate; numerical
correctness is not quality parity, FPGA readiness or an efficiency result.

The separately declared frozen-endpoint diagnostic below now compares sampled
training and development losses without updates. Changing-batch online losses
cannot substitute for endpoint training scores. Keep fitting, generalization
and possible connectivity limits distinct.

#### Source-derived conditioning hypothesis, not a measured cause

The fixed mixed initialization has one eligible projection entry per row and
one eligible head entry per class while the guard remains valid. Those
projection entries select channels; their trainable values are **additive
offsets, not mixing coefficients**. In exact real arithmetic, a time-independent
channel offset `b` becomes `b - mean(b)` under full-width centering. Every valid
temporal tap of that channel receives the same offset, so its max/min adds the
offset without changing the temporal winner. With fixed projection/head
winners, induction through the residual layers shows that selected MP, MB and
head-offset changes collectively affect only a static class-bias vector.
In particular, the selected MP and corresponding MB derivatives are aliased.
This is a conditional algebraic property, not a byte-exact FP64
reparameterization claim; changing projection winners leaves its assumptions.

The model is nevertheless contextual: learned token embeddings and per-tap MH
weights change temporal features and winners; residual paths and full-width
means remain. Different channels can select those means at different past
times. The topology's 16-channel selected ancestry excludes centering's dense
dependencies, and is neither the full dependency graph nor a capacity bound.
The tropical maximum receptive field is 91 positions versus the GRU's 128 in
these windows. See [topology construction](../tools/fpga_lm_projection_topology.py),
[guarded forward](../tools/fpga_lm_guarded_forward.py) and the
[reference reverse](../tools/fpga_lm_float_reference.py).

This motivates distinguishing poor fitting, generalization and optimizer
conditioning before widening the graph. Freezing or consolidating aliased
biases would be a conditioning control, not an expressivity test. A later
second eligible projection edge could retain hard max/add arithmetic while
testing richer channel selection, but would require a new fixed initialization,
protocol and parity check; the current one-entry shortcut would no longer
certify it. None of these controls has been run or established as a fix.

### Frozen-endpoint fitting diagnostic (2026-09-14)

The [forward-only producer](../evidence/fpga_lm_endpoint_fit_2026-09-14.json)
and its [independent forward audit](../evidence/fpga_lm_endpoint_fit_audit_2026-09-14.json)
both pass. After the complete predecessor audit succeeded, this diagnostic
scored exactly 64 training windows **0, 16, ..., 1008**, fixed in advance,
using the two original initial models and all six step-256 endpoints.
It reuses the previous development scores; there are **no new development
evaluations, gradients, optimizer steps, count fits or changed checkpoints**.
The training cohort is a deterministic spanning sample of 8,192 positions,
not a random sample or full-training loss estimate.

| Cohort | Shared initial CE | Tropical final mean CE | Prior-GRU final mean CE | Tropical minus GRU |
| --- | ---: | ---: | ---: | ---: |
| Sampled training | 2.399079 | 2.166075 | 1.946946 | **0.219129** |
| Reused development | 2.685425 | 2.621051 | 2.372507 | **0.248544** |

Training CE for orders 0/1/2 is **2.165977 / 2.169387 / 2.162860** for
tropical and **1.948048 / 1.945010 / 1.947779** for GRU. Every order has a
sampled-training gap (0.217929 / 0.224376 / 0.215081 nat). Tropical's training
range is 0.006527 nat; GRU's is 0.003037. Mean sampled-training accuracies are
37.1663% and 43.5140%, respectively, versus shared initial 29.3579%.
Initial model probabilities agree within 6.11e-16 maximum absolute error.
The fixed alpha-one and repeated-count baseline CEs on this sample are
2.398943 and 2.378790; neither baseline is refitted.

**Most of the observed comparator gap is already present on sampled training
examples.** The 0.248544-nat development difference decomposes arithmetically
into the 0.219129-nat training difference plus **0.029415 nat** of additional
development-minus-training separation. This is a descriptive decomposition,
not an attribution of percentages to causal mechanisms. Both models inherit
the same initial development-minus-training difference, 0.286346 nat. It grows
by 0.168630 for tropical and 0.139215 for GRU. Tropical improves sampled-training
CE by 0.233004 and development CE by 0.064374; GRU improves them by 0.452133
and 0.312918. Thus this recipe has a fitting deficit at the declared budget,
not merely a development-only failure. These endpoints do **not** distinguish
slower optimization, conditioning and representation limits, prove convergence,
or support a whole-training/overfitting/capacity claim.

The gap is broad across the scored windows: tropical has higher CE on all
64 sampled training windows and all 64 development windows in each of the
three orders. These are repeated evaluations of the same cohorts, **not 192
independent examples** per cohort, and no significance test is implied.

The producer retains **517 sealed payloads / 83,154,973 bytes**, covering
**512 model-window scores and 128 fixed baseline entries**. The separate audit
recomputes all **256 dense tropical and 256 expanded-GRU forwards**, plus
256 supplemental guarded forwards for exact implementation traces/counters.
All 131,072 projection and 32,768 head operations in those 256 tropical scores
are certified fast, with no dense-operation or whole-model fallback. Combined
new producer/public/audit storage is **83,900,143 bytes**, below 256 MiB.
Ordinary readback introduces no model forwards. There are no new learned
trajectories or gradient-pair exposures; all development/optimizer/FPGA/paid-
execution flags remain false or zero as appropriate. Reserved validation
152–863 and the final test remain unevaluated.

Producer execution: **01:39:13–01:41:06 UTC**, exit 0, **113.324741 UTC /
113.324032 monotonic seconds**. Independent audit: **01:41:44–01:44:09 UTC**,
exit 0, **144.480646 UTC / 144.479744 monotonic seconds**. Its numerical
verification ends at 01:44:02 (138.001543 UTC seconds), before final publication
postchecks. Both are within their 1,800-second bounds. These timings compare
different verification workloads, not optimized model-training efficiency.

| Identity | SHA-256 |
| --- | --- |
| Producer public | `6bb2ae3f379049378475196506197942c8fefb1f3aab34f3981d4c5d9bae3e3e` |
| Producer seal | `e2375927ff42bc9f612073ef865efa40038accaa4d5e7e3ea6c311318568924b` |
| Saved producer declaration | `461144ecce7cd20b76f7eb3d7c41a329e81bb6c3239e69561b351d1a4caec62f` |
| Successful audit public | `8a10febeb610962a6b6e86a8aacc5e94f934d7fc2730a9f23e7835d763291fbc` |
| Audit manifest | `552e3f03eaab5cbb304acb44bb12d1cdebed5dfaedc4a5e7e4c6ceea37e07cf8` |

All 71 source/test identities and original experiment artifacts remain frozen
through the audit. The sole change from the reviewed offline candidate was
inserting the actual completed predecessor audit pin; all 78 tests passed
again afterward. The 4,651-test repository regression run preceded that one-
constant admission change. No public library optimizer default is promoted.

The next proposed control is **one fixed additional pass for the two order-0
endpoints**, preserving actual Adam moments/counters, learning rates, batching,
training order, graph and guard/fallback behavior. This is a diagnostic pilot,
not best-seed selection or an all-order quality claim. A reduced paired fitting
deficit would establish budget sensitivity within the current graph; a null
result would still not prove a capacity limit. Bias reparameterization changes
coordinate/effective-step geometry, so it should follow this simpler budget
control. This proposal has **not** been run or admitted here; it requires a
separate complete protocol and source/state restoration tests. No more tuning,
reserved-cohort scoring or FPGA scaling follows automatically.

### Paired order-0 Adam continuation (2026-09-14)

The [declared second-pass control](PLAN.md#paired-order-0-exact-adam-continuation-v1-declared-2026-09-14)
and its independent numerical audit both complete successfully. The fixed
additional pass improves both models, but **widens the tropical-minus-GRU
deficit** on the sampled training and reused development cohorts.

| Cohort / model | Before: step 256 CE | After: step 512 CE | CE improvement |
| --- | ---: | ---: | ---: |
| Sampled training / tropical | 2.165976625 | 2.060771004 | 0.105205621 |
| Sampled training / GRU | 1.948047807 | 1.758909062 | 0.189138745 |
| Reused development / tropical | 2.608313612 | 2.573144425 | 0.035169187 |
| Reused development / GRU | 2.371154387 | 2.266224492 | 0.104929896 |

The paired training deficit increases **0.217928818 → 0.301861943**
(+0.083933125 nat); development increases **0.237159224 → 0.306919933**
(+0.069760709 nat). Final training accuracy is 0.4033203125 tropical versus
0.4820556641 GRU; development accuracy is 0.3210449219 versus 0.390625.
Development-minus-sampled-training CE grows from 0.442336987 to 0.512373421
for tropical and 0.423106581 to 0.507315430 for GRU. Both improve on all 64
sampled training windows; tropical improves on 52/64 development windows,
GRU on 63/64. Tropical still trails GRU on every window in both cohorts.
The paired deficit widens on 63/64 training and 61/64 development windows.
These are descriptive correlated comparisons, not independent trials.

#### What changed, and what did not

Both arms resume their actual order-0 step-256 full Adam state and finish at
512, preserving named parameters, first/second moments, scalar counters and
counter dtypes, parameter ordering, learning rates and optimizer options.
Initial imported archives are byte-identical to their historical sources.
There is no empty-Adam reset, schedule, rate change or architecture change.
The second pass repeats exactly the original order-0 permutation of 1,024
windows, four 128-position windows per update. Each arm adds 131,072 gradient
pair exposures to one inherited gradient pass and one inherited fitted-prior
pass: 393,216 model-history exposures on 131,072 distinct pairs. No count fit,
new baseline evaluation, reserved validation or test evaluation is added.

Head and projection channel support stays unchanged on all retained training
queries and final readouts: **0 departures / 18,874,368 head entries** and
**0 / 301,989,888 projection entries**, with no unclassified entries. All 257
tropical full-state checks preserve the 1,046,528 inactive projection weights
and 65,408 inactive head weights at their sentinels with zero first/second
moments; saved off-support raw gradients are zero too. The 2,048 projection
rows and 128 head rows retain their original single-active-entry support.

On the same fixed cohorts, head and projection endpoint selections change
zero times, while temporal taps change **5,439,101 / 16,777,216** sampled
training entries (32.4196%) and **5,412,909 / 16,777,216** development entries
(32.2635%). These are unweighted route counts, not gradient mass or a
causal context ablation. The model continues adapting; its absolute loss
improves. Endpoint agreement alone would not establish interval stability,
which is why all retained queries and states are checked separately.

The 1,152 observed tropical forwards use **589,824 projection fast paths and
147,456 head fast paths**, with zero dense or whole-model fallback. This is
guard/path evidence on these inputs, not a speed, energy, memory or FPGA
advantage. Full parameter and optimizer arrays are still retained.

#### Verification and provenance

Producer public: [continuation record](../evidence/fpga_lm_continuation_2026-09-14.json).
Independent numerical audit:
[completed receipt](../evidence/fpga_lm_continuation_audit_2026-09-14.json).
Raw roots are `manuscript/build/fpga_lm_continuation_2026-09-14/v1` and
`manuscript/build/fpga_lm_continuation_audit_2026-09-14/v1`.

The producer seals **1,802 payloads / 1,700,487,164 bytes**. Ordinary readback
checks all **512 saved batch gradients, 512 Adam commits, 514 complete states
and 256 final scores**. The separate audit independently reconstructs 1,024
tropical window VJPs, 256 tropical and 256 expanded-GRU batch gradients, all
128 tropical and 128 GRU endpoint scores, then applies the **saved** gradients
for byte-exact full-state replay. It covers 1,152 dense tropical forwards and
1,152 GRU window equivalents; an additional 1,152 guarded forwards verify
exact traces and counters. Independent floating comparisons retain absolute
1e-9 / relative 1e-11 tolerance; integer selectors and state replay are exact.

Producer: **11:19:50–11:27:34 UTC**, exit 0, 463.715350 UTC / 463.712855
monotonic seconds. Independent audit: **11:28:08–11:48:45 UTC**, exit 0,
1,237.395886 UTC / 1,237.411654 monotonic seconds. Its numerical verification
finishes at 11:48:37, before publication postchecks. Both remain within the
declared bounds. These timings are verification workloads, not optimized
training-efficiency measurements.

| Identity | SHA-256 |
| --- | --- |
| Producer public | `07f18f23ce6cde30da7a6816eed661c4c901c4faeb96aa8f411149cebe69b813` |
| Producer manifest | `a0d0e45b4ed0f98348338fd5e425da8e305ff04844338c1ddbb0151d39f8410b` |
| Producer seal | `baf1271be9668520f442719efcccee300c7767571eadbea7c680792de54d1de4` |
| Saved producer declaration | `6dd20dabe476d5ae440adfdb179c381d46c3b5b5439d4812f0ceb329b9f718d3` |
| Successful audit public | `ebc8bf0b4b4b3b723120ba94c0927818d70159e0b06fd24bc86ea4d63666b25c` |
| Audit manifest | `e6d0cb55352dcef3d24065245510c28dcec08e709c735b27e01a001696cc0b4a` |

All 73 source/test identities and the fully bound historical ancestry remain
unchanged through the audit. An independent file-only reviewer separately
rehashes every producer payload/source and checks all state/query/score tensor
identities, masks/support, counters, memberships, comparisons and prior
receipts without model or optimizer calls. The new runner has 99 passing
synthetic tests; the existing suite has 4,651 passes (14 skips, 10 deselected).

**Conclusion:** one extra pass helps absolute tropical quality but does not
close the achieved-fitting gap under this recipe. It does not establish
convergence or a capacity limit. The next separately declared diagnostic is
the [frozen-core-bias conditioning control](PLAN.md#frozen-core-bias-conditioning-control-v1-declared-2026-09-14),
not another automatic budget extension or a connectivity change. Its
conditional bias-equivalence rationale is not evidence that bias duplication
caused this result. No public optimizer default, manuscript quality claim or
hardware scaling is promoted by this experiment.

### Frozen-core-bias control (2026-09-14)

The [declared conditioning control](PLAN.md#frozen-core-bias-conditioning-control-v1-declared-2026-09-14)
and its separate independent numerical audit both complete successfully.
Freezing the core gives a modest CE gain, but leaves a substantial GRU deficit;
this does not establish the cause of the gap or promote a recipe.

This single `mixed_core_bias_frozen` arm starts from the exact original fitted
mixed state 0 with empty Adam, not either continuation endpoint. It repeats
the original order-0 permutation of 1,024 training windows, with 256 updates
of four 128-position windows, mean exact CE, FP64 CPU arithmetic and Adam
LR 0.01. The graph, temporal operators, centering, guard/fallback behavior and
raw VJP are unchanged. Only the **applied** `MP` and `MB` gradients are zeroed;
`EMBED`, `MH` and `W_HEAD` retain their raw gradients. No GRU is newly trained.

| Model / fixed reference | Sampled-training CE | Development CE | Training accuracy | Development accuracy |
| --- | ---: | ---: | ---: | ---: |
| Shared original initialization | 2.399078871 | 2.685424838 | 0.293579102 | 0.255493164 |
| Original mixed, first pass | 2.165976625 | 2.608313612 | 0.371582031 | 0.310424805 |
| Frozen-core mixed, first pass | 2.153430163 | 2.598400205 | 0.375732422 | 0.309570312 |
| Original fitted-prior GRU, first pass | 1.948047807 | 2.371154387 | 0.432128906 | 0.362182617 |

Freezing the two regions reduces CE versus the original mixed control by
**0.012546461 nat on sampled training and 0.009913406 on development**.
The remaining new-minus-GRU deficits are **0.205382357 and 0.227245818 nat**;
the GRU has lower CE on every scored window in both cohorts. The new arm
beats the original mixed CE on 44/64 training and 38/64 development windows.
It improves over the original initialization by 0.245648707 training and
0.087024632 development nat, on 64/64 and 55/64 windows respectively.
Development accuracy is slightly lower than the original mixed control,
despite lower mean CE. These are paired descriptive comparisons, not
independent trials, a whole-training-loss estimate or convergence evidence.

Only final context-on scores are newly computed: 64 deterministic training
windows `range(0, 1024, 16)` and the same 64 development windows 88–151.
Initial scores, both original first-pass controls and baseline references are
reused from audited evidence. Reserved validation 152–863 and the final test
remain untouched. The new arm adds 131,072 gradient-pair exposures to 131,072
inherited fitted-prior pairs: **262,144 model-history exposures on 131,072
distinct pairs**. The continuation is an admission/proof dependency, not
inherited gradient history. There is no new count fit or baseline evaluation.

All 256 full raw gradients and 256 separate applied-gradient archives are
retained. This is not a no-op mask: a separate file-only integrity review finds
nonzero raw `MP` and `MB` gradients in every batch, with 425,374 nonzero entries
per region summed across the 256 batches. Replay checks the mask exactly and
applies saved values to Adam. Across all **257 states**, `MP`/`MB` weights
stay byte-identical to initialization; their moments are initially absent,
then exact zero, while every scalar counter advances from 0 to 256. Full
parameter/moment/counter arrays remain stored in the inherited state format.

Observed head/projection support has **0 / 18,874,368 head departures** and
**0 / 301,989,888 projection departures** over all new queries and scores,
with no unclassified entries. Inactive head/projection sentinels and moments
remain unchanged. The two recipes' final head/projection tapes agree on the
matched cohorts, but temporal taps differ at **7,661,569 / 16,777,216** training
entries (45.6665%) and **7,599,519 / 16,777,216** development entries (45.2967%).
These are **between-recipe final-state** comparisons, not within-trajectory
stability or gradient-mass measurements. All 1,152 observed guarded forwards
use 589,824 projection and 147,456 head fast operations, with no fallback;
this is not a speed, energy, storage or hardware advantage claim.

The bias-compensation argument assumes identical embeddings/temporal weights
and shared projection/head selectors static across time and all compared
inputs, remaining valid after compensation; equal time-varying tapes do not
suffice. This is conditional real arithmetic, not global model, FP64 byte
or Adam equivalence. The gain does **not** establish redundant biases as the
gap's cause, prove a capacity limit, or justify hardware scaling.

Producer: [sealed evidence](../evidence/fpga_lm_core_bias_2026-09-14.json);
audit: [completed receipt](../evidence/fpga_lm_core_bias_audit_2026-09-14.json).
Raw roots: `manuscript/build/fpga_lm_core_bias_2026-09-14/v1` and
`manuscript/build/fpga_lm_core_bias_audit_2026-09-14/v1`.
The producer retains **1,158 payloads / 746,590,352 bytes**, including 256 raw
queries, 256 applied archives, 257 states and 128 final scores. Execution is
**12:10:03–12:17:41 UTC** (manifest start), exit 0, **457.604624 UTC /
457.607715 monotonic seconds**. The independent audit reconstructs **1,024
raw window VJPs and 256 batch gradients**, verifies all **256 masks/commits,
257 states and 128 scores**, and covers **1,152 dense plus 1,152 supplemental
guarded forwards**. It applies saved masked gradients for exact state replay.
Audit execution: **12:18:17–12:39:00 UTC**, exit 0, **1,243.612299 UTC /
1,134.863737 monotonic seconds**. Numerical verification ends at 12:38:53,
after 1,235.967719 UTC / 1,127.219065 monotonic seconds, before publication
postchecks. Both observed clocks are reported; no efficiency inference follows.
Separate file-only integrity review also passes: all 75 sources, 1,158 payloads
and the exact three-file audit capsule are rechecked, without numerical replay.

| Identity | SHA-256 |
| --- | --- |
| Producer public | `b5d16c31d3bba43ce8ad7750ae62ff8b0fc278f8720e8f84b94008c90f13d23c` |
| Producer manifest | `22fe0453d208fbdc932589a6fd86ab34529e91e14bc10b5d38044e8b2ab145fa` |
| Producer seal | `6ca008ad1681d2a0db14509ec47889fb4bbc4960c6637f3565a06db5185eb6ee` |
| Saved producer declaration | `e8ccce0152a07f28b4862a6fcd735ae7d24c57a64b34e6e86f9264f37ec611b6` |
| Successful audit public | `c9185adcdef0db345fbcbd27d7372c9590a69f7979d688faeaa327196f2a7fcf` |
| Audit manifest | `0a1f2507d6d95616ccab735d0457dc76b97662fe9e8bef16a64f1d04bcd6d4dd` |

All **75 source/test identities** and bound input/raw/seal/public bytes remain
unchanged through audit. The control's **66 synthetic tests** cover conditional
bias compensation/counterexample, raw/applied gradients, exact Adam replay and
publication failures. The combined regression has **4,816 passes, 14 skips and
10 deselected in 394.27 seconds**. **Conclusion:** freezing improves CE slightly
but does not close the observed comparator gap. No optimizer default is changed.

### Two-candidate channel-routing control (2026-09-14)

The [declared routing control](PLAN.md#two-candidate-channel-routing-control-v1-declared-2026-09-14)
producer completed with exit 0, from the original fitted mixed state 0 with
empty Adam. Its sealed result is worse than the frozen-core control despite
substantial input-dependent alternate selection. The separate independent
numerical audit also **completed successfully, exit 0**, at **16:15:09 UTC**.
It verifies the complete new trajectory and readouts without changing the
fixed-budget negative result or promoting a recipe.

Each projection row retains its original primary at zero and gains one
distinct seeded-successor input in the same actual feature bank at
dimensionless weight -0.01. Other projection entries retain their sentinels.
The entire `MP`/`MB` tensors are frozen using zero applied gradients;
`EMBED`, `MH` and `W_HEAD` train with the unchanged order-0, 256-update,
four-window exact-CE Adam-0.01 recipe. The full-dimensional selector remains
unchanged, using the existing whole-model dense fallback, not a two-column
selector or a new fast path.

The closed `initial_parity.json` records **128 byte-exact initial trace
comparisons** between the candidate and original, on the fixed 64 sampled
training windows and 64 development windows. The scheduled first batch
matches the archived core-control inputs, targets, complete traces and all five raw gradients
byte-for-byte, **before any Adam commit**, with no additional gradient query.
Historical and candidate guard counters are intentionally not equated.
The independent audit rechecks both initialization controls. Separate producer
file/array review also confirms that the first updated `EMBED`/`MH`/`W_HEAD`
parameters, moments and counters are byte-identical to the core control's
state 1. The intentional projection initialization difference is not hidden
by that restricted state comparison.

| Model / fixed reference | Sampled-training CE | Development CE | Training accuracy | Development accuracy |
| --- | ---: | ---: | ---: | ---: |
| Shared original initialization | 2.399078871 | 2.685424838 | 0.293579102 | 0.255493164 |
| Original mixed, first pass | 2.165976625 | 2.608313612 | 0.371582031 | 0.310424805 |
| Frozen-core mixed, first pass | 2.153430163 | 2.598400205 | 0.375732422 | 0.309570312 |
| Two-candidate mixed, first pass | 2.186878486 | 2.608303389 | 0.363769531 | 0.308715820 |
| Original fitted-prior GRU, first pass | 1.948047807 | 2.371154387 | 0.432128906 | 0.362182617 |

The candidate's CE is **0.033448323 nat worse on sampled training and
0.009903184 worse on development** than the frozen-core control, with lower
accuracy on both cohorts. It has lower CE on only 11/64 and 25/64 matched
windows respectively. Versus the original mixed model, training CE is worse
by 0.020901861 nat; development CE differs by just -0.000010222 nat.
The remaining new-minus-GRU deficits are **0.238830679 / 0.237149002 nat**,
with the GRU better on every scored window in both cohorts. The candidate
still improves over its original initialization by 0.212200385 / 0.077121448
nat. These paired comparisons are descriptive, not whole-training losses,
independent trials or convergence evidence.

The intended competition is actually exercised in the producer's saved
tapes. At the final fixed checkpoint, alternates win **8,373,347 / 16,777,216**
sampled-training selections and **8,385,576 / 16,777,216** development
selections, approximately half in each cohort. Both primary and alternate
are selected across the same cohort at **2,026 / 2,048** training layer/row
pairs and **2,031 / 2,048** development layer/row pairs. Each observation
uses one fixed checkpoint, not merely switches accumulated across training.
The first alternate wins and both-choice rows occur at query step 1, the
second batch, after the common first update.
The 256 queries contain 132,947,704 alternate selections out of 268,435,456,
with both choices at 501,880 / 524,288 **row-checkpoints**. There are no
other-sentinel selections in those queries or final cohorts; final head
departures are 0 / 1,048,576 per cohort. These unweighted counts do not measure
gradient mass or establish why quality worsens.

The producer retains **1,288 payloads / 911,591,227 bytes**, including 256 raw
queries, 256 applied-gradient archives, 257 full Adam states and 256
initial/final score windows. Ordinary replay checks the complete masked state
chain: transformed `MP`/`MB` stay byte-identical, their moments are initially
absent and then zero, and counters advance. All **1,280** producer forward
windows use the existing whole-model dense fallback. The completed audit
independently reconstructs **1,024 window VJPs, 256 raw batch gradients and
1,280 dense forwards**, and verifies **256 masks/commits, 257 complete states,
256 score windows, 128 initial parity checks and one first-batch check**.
Another **1,280 supplemental guarded forwards** reproduce the recorded
full-dimensional fallback. Saved masked gradients drive byte-exact state
replay; all **77 source identities and input/raw/seal/public bytes** remain
unchanged.

Original mixed, frozen-core and GRU controls are reused; no new comparator
training, baseline fit, reserved validation or final-test evaluation is
included. Only the fixed 64 sampled training windows and reused development
windows 88–151 are scored. The history is 131,072 fitted-prior plus 131,072
new gradient-pair exposures on 131,072 distinct training pairs; previous
experiments are proof dependencies, not inherited gradient passes.

Producer: [sealed evidence](../evidence/fpga_lm_channel_choice_2026-09-14.json);
audit: [completed receipt](../evidence/fpga_lm_channel_choice_audit_2026-09-14.json).
Raw roots: `manuscript/build/fpga_lm_channel_choice_2026-09-14/v1` and
`manuscript/build/fpga_lm_channel_choice_audit_2026-09-14/v1`.
Execution is **15:30:36–15:45:56 UTC**, exit 0, **919.471075 UTC /
919.467935 monotonic seconds**. Audit execution is **15:46:28–16:15:09 UTC**,
exit 0, **1,720.145557 UTC / 1,720.138297 monotonic seconds**. Inner numerical
verification completes at **16:14:54 UTC**, after **1,705.184203 UTC /
1,705.177002 monotonic seconds**, before final publication/postchecks.
Both observed clocks are reported; these are invocation timings, not benchmarks.
Final file-only review also passes: all 77 sources and 1,288 producer payloads
are rehashed unchanged, the exact three-file audit capsule totals 225,295
bytes, and its verification is byte-identical to the 11,446-byte public receipt;
the bound core capsule and all receipt/manifest/declaration pins also match,
without further numerical replay.
Admission verified **77 source identities**, the audited predecessor's
**1,158 payloads**, **128 original initial traces** and the unchanged
**1,115,394-byte corpus**, with numerical and subprocess entry points blocked.
The final synthetic suite has **79 passes in 2.23 seconds**. The combined
regression has **4,895 passes, 14 skips, 10 deselected and one expected warning
in 406.53 seconds**.

| Frozen identity | SHA-256 |
| --- | --- |
| Runner | `f2c636f088114bc3027d82d75d79eb1db4e0e645a02f055f835e1288717813a1` |
| Tests | `0b2b313b47ec3391dc79cc6b3d7ab2f6de08cf6d2c2bd3f9879c7bde6ff7bda1` |
| Saved declaration | `65bffcd80803d1ec614f2a05fc21188f12de0365ff461b10a8bc5caecb2dba05` |
| Producer public | `f8e7f3ddc66fff4f3d1d41ff1228eca030da8c183f437d38cf4181befa2ecbb8` |
| Producer manifest | `aa0ec3f6e007965e467edc61b34440337bb7355deda264478c94f5cd5006ff5b` |
| Producer seal | `e17ed897707852ce5124f60a7726529cff7a9760567e5941ce797232882e3767` |
| Successful audit public | `96e824375caf67522059a6b2a8864ff58b1c2ba98f668e62f90b6680d53d8e72` |
| Audit manifest | `6e9b8e56f5d5d2ee41ce41bff3e63855ad95a3698901d42860f484f30022c2ec` |

**Conclusion:** this particular active two-candidate routing
control does not improve fixed-budget fitting over the frozen-core control.
It does not establish a global capacity limit or the gap's cause. No quality
promotion, efficiency or hardware claim is made.

### Three-arm learned-routing contrast block (2026-09-15)

All three producers and their separate independent numerical audits in the
[declared block](PLAN.md#three-arm-learned-routing-contrast-block-v1-declared-2026-09-15-utc)
completed with actual exit 0. **None meets the declared material-progress
criterion:** all three have higher mean sampled-training and development CE
than the frozen-core control. This is a negative result for these three fixed
recipes, not a capacity bound or a diagnosis of the remaining quality gap.

Each arm starts independently from the same original fitted state, with empty
coordinate Adam and physical projection pairs `primary=m+d`, `alternate=m-d`,
initially `m=-0.005`, `d=0.005`. A (`contrast`) trains `d` at LR 0.01 with `m`
fixed; B (`contrast_slow`) changes only that rate to 0.001; C (`contrast_mean`)
trains both `d` and `m` at LR 0.01. `EMBED`, `MH` and `W_HEAD` retain LR 0.01;
`MB` and noncandidate projection entries remain frozen. All use the same
order-0 shuffle, 256 four-window updates, exact mean CE and CPU FP64. Fixed
`m` is a serialized buffer omitted from Adam, not a fabricated optimizer state.

| Model / fixed reference | Sampled-training CE | Development CE | Training accuracy | Development accuracy |
| --- | ---: | ---: | ---: | ---: |
| Shared original initialization | 2.399078871 | 2.685424838 | 0.293579102 | 0.255493164 |
| Original mixed, first pass | 2.165976625 | 2.608313612 | 0.371582031 | 0.310424805 |
| Frozen-core mixed, first pass | 2.153430163 | 2.598400205 | 0.375732422 | 0.309570312 |
| Fixed-offset two-candidate mixed | 2.186878486 | 2.608303389 | 0.363769531 | 0.308715820 |
| A: contrast LR 0.01 | 2.197280004 | 2.614672326 | 0.366699219 | 0.303344727 |
| B: contrast LR 0.001 | 2.185692063 | 2.620297871 | 0.366821289 | 0.301879883 |
| C: contrast and mean LR 0.01 | 2.197037636 | 2.616082766 | 0.363037109 | 0.308471680 |
| Original fitted-prior GRU, first pass | 1.948047807 | 2.371154387 | 0.432128906 | 0.362182617 |

The prespecified paired CE differences (first recipe minus its comparator) are:

| Comparison | Sampled training | Development |
| --- | ---: | ---: |
| A minus fixed-offset control | +0.010401517 | +0.006368937 |
| B minus A | -0.011587941 | +0.005625544 |
| C minus A | -0.000242368 | +0.001410439 |

The slower contrast rate improves sampled fitting relative to A, but worsens
development CE. Adding the common coordinate gives an almost unchanged mean
training CE and a slightly worse development CE than A. Neither comparison
recovers the frozen-core result. New-minus-frozen-core CE deficits are
**0.043849840 / 0.016272121** for A, **0.032261899 / 0.021897665** for B, and
**0.043607473 / 0.017682560** for C, respectively on sampled training and
development. Remaining GRU gaps are **0.249232197 / 0.243517939**,
**0.237644256 / 0.249143483**, and **0.248989829 / 0.244928378**. The GRU has
lower CE on every one of the 64 windows in each cohort for every arm.

All three `material_progress` flags are false. The declared threshold was at
least 0.01 nat improvement over frozen core on **both** cohorts, not merely
recovery of the failed fixed-offset baseline. The training cohort remains the
fixed sample `range(0,1024,16)`, not a full-training loss. Development windows
88–151 are heavily reused, and all arms use one training order; windows and
arms are not independent replications. No reserved validation, final test,
context-off readout or new comparator training is included.

Channel competition remains active at the final fixed checkpoints:

| Arm | Training both-choice rows / 2,048 | Development both-choice rows / 2,048 | Alternate share, training / development |
| --- | ---: | ---: | ---: |
| A | 1,973 | 1,971 | 49.9368% / 49.9026% |
| B | 2,029 | 2,037 | 49.5026% / 49.5333% |
| C | 1,939 | 1,925 | 50.3817% / 50.3285% |

There are no other-sentinel winners or head departures in these final cohorts.
Each both-choice count pools positions at one checkpoint, not changing weights
across training. These counts do not measure gradient mass, temporal instability
on identical inputs, or the cause of the CE differences. All 1,280 forward
windows per arm actually use whole-model dense fallback: 655,360 projection
and 163,840 head dense operations, with zero fast operations. A changed valid
certificate or off-pair winner would have been reported, not censored.

The saved first-update statistics verify the parameterization caveat. A's
maximum primary and alternate movements are each approximately 0.01, with
pair-gap movement approximately 0.02; B's are approximately 0.001 and 0.002.
C's maximum primary movement is approximately 0.02, while its alternate moves
only **1.734723475976807e-18** from FP64 rounding, not exactly zero. Its `m` and
`d` first-step statistics agree, and all three arms have matching first-step
feature/head statistics. Thus C versus A is not matched physical-edge update
magnitude. Fixing the pair-common weight coordinate also does not fix output
bias: changing the contrast changes winning values even at unchanged routes.
Neither a gain nor this negative result would isolate a pure routing effect.

Every arm verifies 128 initial traces against archived original-function
traces and the first raw-gradient query before its first commit. Each retains
256 full raw queries, 256 applied-coordinate gradient archives, 257 complete
coordinate/materialized states and 256 initial/final score windows. Each
independent audit reconstructs 1,024 physical window VJPs, 256 batch gradients,
the coordinate-map autograd pullbacks, all 1,280 dense forwards and another
1,280 supplemental guarded forwards. Saved coordinate gradients drive exact
Adam/state replay; independently recomputed gradients are compared under the
declared floating tolerances, not substituted to create another trajectory.
All source, input, raw, seal and public-byte checks pass.

Across the block this is **768 updates, 771 full states, 3,840 producer forward
windows and 393,216 new gradient-pair exposures**. Summed inherited fitting
plus new gradient histories are 786,432 exposures on the same 131,072 distinct
training pairs per model; no new count fitting occurred. The three seals cover
**3,864 payloads / 3,041,772,186 bytes**: A 1,008,360,727 bytes, B 1,039,114,943,
and C 994,296,516. Separate file-only review verifies all 79 source identities
and the three exact private audit capsules, each with byte-identical public
verification. The block's synthetic suite passes **92 tests**; its completed
combined regression has **4,987 passes, 14 skips, 10 deselected and one expected
backend-fallback warning in 416.29 seconds**.

All times below are actual successful invocation completions, not the earlier
verification timestamps embedded in audit receipts. Elapsed values include
validation/publication overhead; concurrent single-thread runs are not a
matched efficiency benchmark.

| Arm / invocation | Start → completion UTC (2026-09-15) | Elapsed UTC / monotonic seconds |
| --- | --- | ---: |
| A producer | 00:40:07 → 00:57:29 | 1042.300298 / 1042.263914 |
| A audit | 00:57:59 → 01:34:23 | 2184.508188 / 2184.514004 |
| B producer | 00:40:37 → 00:58:01 | 1044.067883 / 1044.031051 |
| B audit | 00:58:29 → 01:34:23 | 2154.045115 / 2154.051326 |
| C producer | 00:41:01 → 00:58:20 | 1039.286801 / 1039.249695 |
| C audit | 00:59:31 → 01:35:18 | 2146.423924 / 2146.430907 |

Sources: A [producer](../evidence/fpga_lm_routing_contrasts_contrast_2026-09-15.json)
and [audit](../evidence/fpga_lm_routing_contrasts_audit_contrast_2026-09-15.json);
B [producer](../evidence/fpga_lm_routing_contrasts_contrast_slow_2026-09-15.json)
and [audit](../evidence/fpga_lm_routing_contrasts_audit_contrast_slow_2026-09-15.json);
C [producer](../evidence/fpga_lm_routing_contrasts_contrast_mean_2026-09-15.json)
and [audit](../evidence/fpga_lm_routing_contrasts_audit_contrast_mean_2026-09-15.json).
Raw roots are `manuscript/build/fpga_lm_routing_contrasts_2026-09-15/<arm>/v1`;
private audits use `fpga_lm_routing_contrasts_audit_2026-09-15/<arm>/v1`.

| Frozen identity | SHA-256 |
| --- | --- |
| Runner | `f9626fb394fed425581444c9b30fbf9c31debe034ec0c5d70e467b0857a3f484` |
| Tests | `0b10c14cc0dde6add5256574bd3cbe3ca8976f4830718424c6eee1bdcb175237` |
| Shared saved declaration | `ea5e3bbce1b04e9db67787897f8052fc3b60b81324fa2dd3e648f02ea38342d8` |
| A producer public | `00e5cdf88d009df494ceb7f530a80eefcd5ef66fce120ebdb1feab7e9c19cc16` |
| A producer manifest | `d245fbc9b15ea051b29e3b2f203358c79381e21fc709a2f6f23304d24fcfaa7d` |
| A producer seal | `ced85ecaea9cef06d281c5cde8c821f8141f4f4b95a7c61b5fd4d569d2b02425` |
| A audit public | `a01a39b9ff365a2b9c83e8ad57ba8690e92c25c2d33c8c81a063dd91455a888d` |
| A audit manifest | `a1df0effda6c9bcc6b0296d776468e2e53c31b9014cf1bf4597903e400ba9951` |
| B producer public | `0530d64442ef49fa8c03f088b7ae077ab89864bbe89b8f8c92252dd192280f2d` |
| B producer manifest | `e016d08bb781a610cdcad1d8f6429cf1a336c0dd60f3d2a7867a8919c221c5f4` |
| B producer seal | `4fb7620d1828d1e89040f71ea2d56936e6977b40fceb0af759f665768aa6a7ae` |
| B audit public | `ae7b921ff339fef8e01f4c0138c5cad09833a22ea9156fd0626c4635a770eb64` |
| B audit manifest | `bd02b17d5baafebba3620ad6f1cef7ee9c86c8bedd18e645d73ff19447da439d` |
| C producer public | `902de8a204d199d1770c2c3856b85adf3405fdf2be6cf09d4833a4edaf50d5ac` |
| C producer manifest | `9455a4dd4c9ae636a1897fe5e60b0c3c7fd00cac6f810b2a314b0639e3777c5e` |
| C producer seal | `0526cae6f3cbfa92338e0da4dde077f45de20285e85a261745aa8523411bf9e2` |
| C audit public | `b476e1477e102d7a94da703e3a893862292eb9e48de09528f28f230574aca4a7` |
| C audit manifest | `97a367eef1548ad1544f41860b7d0297f3228e1dbc748ebda895ea8cbb78db8e` |

The strongest tested frozen-core and GRU references therefore remain unbeaten
by this block. The result does not justify more adaptive offset/rate trials,
nor identify a global representation limit. No optimizer default, physical
FPGA execution, paid run or hardware advantage is promoted by these outcomes.

### Frozen-feature residual readout oracle (2026-09-15)

The [declared follow-up](PLAN.md#frozen-feature-residual-readout-oracle-v1-declared-2026-09-15-utc)
and its separate independent endpoint audit both reach **actual exit 0**.
A residual affine readout improves the fixed frozen-core endpoint beyond the
bias-only control on both scored cohorts. However, the affine solve reaches
its iteration cap without meeting the numerical optimization target: this is
an audited achievable improvement, **not a near-optimal readout or capacity
bound**.

Every backbone parameter remains at the audited frozen-core order-0 checkpoint
256. Both corrections start at zero and use the same 1,024 existing training
windows, 131,072 prediction positions, actual baseline logits and unscaled
512-dimensional head-input features. Bias has 128 new coordinates; affine
has 65,664, including its bias. Both use full-training mean exact CE plus
`0.001/2 * ||theta||^2`, with every new coordinate regularized. No backbone
gradient, optimizer update, count fitting, new cohort or context-OFF query
is performed.

The following training scores are the **64 sampled training windows**, not
the full-training solver objective. Development remains the repeatedly used
64 windows 88–151; reserved development and final test remain untouched.

| Model / correction | Sampled-training CE | Development CE | Training accuracy | Development accuracy |
| --- | ---: | ---: | ---: | ---: |
| Frozen core, no correction | 2.153430163 | 2.598400205 | 37.573242% | 30.957031% |
| Bias-only correction | 2.149490148 | 2.590784861 | 37.548828% | 31.066895% |
| Affine correction, capped solve | 2.061243836 | 2.573624209 | 39.428711% | 31.823730% |
| Historical prior-plus-GRU | 1.948047807 | 2.371154387 | 43.212891% | 36.218262% |

Relative to frozen core, bias reduces CE by **0.003940015 / 0.007615344 nat**
on sampled training/development, improving 43/64 and 52/64 windows. Affine
reduces it by **0.092186328 / 0.024775996 nat**, improving 64/64 and 46/64.
Affine minus bias is **−0.088246313 / −0.017160652 nat**, with lower CE on
64/64 and 44/64 windows; accuracy rises by 1.879883 / 0.756836 percentage
points. Relative to the original one-pass mixed model, bias gains
0.016486476 / 0.017528751 nat and affine gains 0.104732789 / 0.034689402.

The GRU still leads bias by 0.201442342 / 0.219630474 nat and affine by
**0.113196029 / 0.202469822 nat**. Affine has lower CE than GRU on only 3/64
sampled-training windows and none of the 64 development windows. Nor is this
a new overall best tropical-related endpoint: the earlier
[extra-pass mixed continuation](../evidence/fpga_lm_continuation_2026-09-14.json)
has CE 2.060771004 / 2.573144425, marginally lower than this affine endpoint
by 0.000472831 / 0.000479784 nat. These are descriptive comparisons with
different optimization histories and budgets, not a matched superiority test.

Full-training initial CE is **2.179286072** for both controls. Their final
full-training optimization records are distinct from the sampled scores:

| Control | Full-training CE | Ridge penalty | Regularized objective | Accepted iterations / complete FG calls | Numerical gap estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bias | 2.174418880 | 0.000486324 | 2.174905203 | 15 / 18 | 0.000007805573 |
| Affine | 2.086664560 | 0.002417405 | 2.089081964 | 200 / 212 | 166.893956966 |

Bias meets `||g||^2/(2*lambda) <= 1e-5`; affine stops at the predeclared
200-iteration cap, with full-gradient squared norm 0.333787914. The large
affine estimate is **not the actual remaining objective gap** and is not a
lower bound on possible further improvement. It is a loose, gradient-derived
numerical upper-bound estimate whose stopping criterion is unmet. Without a
justified floating-point error bound, even the small bias estimate is not a
formal certificate. The affine accepted objective still decreases by
0.002392962 between iterations 190 and 200 while gradient norms fluctuate;
this does not establish a unique conditioning problem or predict convergence.

Affine meets the two raw 0.01-nat improvement clauses but lacks numerical
completion. Bias has numerical completion but does not meet either 0.01-nat
clause. Thus neither satisfies all prerequisites for the declared completed
progress flag; neither is promoted. A finite capped solve is a retained,
inconclusive optimization outcome, not an execution failure.

The positive comparison establishes that this richer correction family has
useful achievable behavior at the frozen endpoint beyond the tested bias
control. It does not establish specifically contextual information or a unique
readout bottleneck: correlated or nearly constant feature directions can also
implement recalibration with a different effective ridge cost. The bias family
is nested in the affine family for the regularized objective; this does not
imply an ordering of raw CE at separately regularized optima. Remaining gaps
cannot be assigned to backbone capacity while the affine optimum is unknown.
No useful-LLM-pretraining, hardware-efficiency or FPGA-head claim follows.

The producer caches **1,088** unique backbone windows: all 1,024 training
windows and 64 development windows. Sampled-training scores reuse that cache.
All 128 previously scored endpoint traces/logits match the audited core
byte-for-byte before fitting. The independent audit reconstructs all 1,088
full-width feature arrays through fresh dense dynamic selection, checks another
1,088 supplemental guarded forwards and their actual counters, and verifies
four initial/final full-training gradients by readout-only Torch autograd.
It checks **256 initial and 256 final readout score windows** across both arms.
Discrete data/routes and initial parity are exact; independent floating
recomputations use the declared `atol=1e-9, rtol=1e-11`.

The 18 bias and 212 affine solver calls are all complete: **2,359,296 and
27,787,264 labeled-position evaluations**, respectively, on the same 131,072
distinct training pairs. Each arm also has four producer endpoint-verification
full-gradient evaluations and two independent-audit evaluations; these are
verification, not extra fitting. Initial/final coefficients and full gradients
are retained with scalar/hash trial histories and accepted-iteration references.
The audit verifies endpoints, not every line-search trial or an exact L-BFGS
trajectory. The readout budget and parameterization are not matched to GRU.

Input-only admission verifies 82 source identities, 1,158 predecessor payloads,
the 1,115,394-byte corpus and the actual frozen checkpoint, with zero attempts
at instrumented numerical/subprocess entry points. The final producer seal
covers **1,097 payloads / 941,551,289 bytes**. The audit's exact three-file
private capsule is 255,775 bytes and its public receipt is 12,757 bytes.
Invocation checks verify unchanged source, input, raw, seal and public bytes.
Separate stdlib-only postcheck confirms all 82 sources, all sealed payloads,
SciPy implementation pins, the 1,088 cache identities/guard rows, 128 archived
trace parities, endpoint-to-solver-history hashes, and exact private/public
audit bytes, without any model, objective, gradient or optimizer calls.
The new suite passes **87 synthetic tests**; the completed combined regression
passes **5,074 tests**, with 14 skips, 10 deselected and one expected
backend-fallback warning in 413.61 seconds. The 81 documentation/evidence
checks also pass.

| Invocation | Start → actual completion UTC (2026-09-15) | Elapsed UTC / monotonic seconds |
| --- | --- | ---: |
| Producer | 14:50:13 → 14:59:39 | 565.431355 / 565.417970 |
| Independent audit | 15:00:14 → 15:07:51 | 457.717968 / 457.712660 |

The audit receipt's earlier verification timestamp is 15:07:36 UTC, with
442.621397 / 442.616264 elapsed seconds; the table reports actual successful
invocation completion including publication/cleanup. These timings are not
a matched model-efficiency comparison.

Sources: [producer record](../evidence/fpga_lm_readout_oracle_2026-09-15.json)
and [independent audit](../evidence/fpga_lm_readout_oracle_audit_2026-09-15.json).
Raw output is `manuscript/build/fpga_lm_readout_oracle_2026-09-15/v1`;
the private audit is `manuscript/build/fpga_lm_readout_oracle_audit_2026-09-15/v1`.

| Frozen identity | SHA-256 |
| --- | --- |
| Runner | `ebf25237e156fc4688a6153ee5ce763522f98aaef5980ee997ff8c5a7f7cac8c` |
| Numerical helper | `c36800534a8e55bc8d572630218238ec7fd1d99c1ded186b5a493b258632ccd7` |
| Tests | `367d1189692f8e1ecb8edb01cc87243a51922e19bb50a43b7cc13484ebac4253` |
| Saved declaration | `b083c16d7885fe33a27e51ca49e18c3f17351cf003fa227a183dbcaefd2c45ab` |
| Producer public | `d893d1e5f6be61e37eb747f6c1be2d110145f56019047ac51c147092da7db3de` |
| Producer manifest | `a92d69402255fe99d57f2ab9dbf7b1b907f37e3b431be7dafee80607ac72b7bd` |
| Producer seal | `7d93056f9b6558b96c76ad1a9e8498675368642b6d1deda51a635fdb508836f1` |
| Audit public | `09c9390c2a051163ada61dd3760310c1a8462389362723aa744159242e86221a` |
| Audit manifest | `a1642db369a963b3ace60f40c40ec152f54eb5e71773d6741def42e008fb7466` |

**Proposed next question, not admitted or run:** can a fixed invertible
coordinate preconditioner solve this same affine objective more effectively
under a separately frozen, matched finite solver budget? For example, with
augmented features `X=[h,1]` and physical coefficients `A=[W,b]`, form the
training-only softmax-curvature upper bound `M=lambda*I+X.T@X/(2*N)` and choose
`S` with `S.T@M@S=I`. Optimize `U` with `A=U@S.T`, preserving the original
logits and penalty `lambda/2*||U@S.T||^2`, not imposing a new isotropic penalty
on transformed coordinates. Check the pullback `G_U=G_A@S`, and retain the
stopping estimate from the **original-coordinate** full gradient. This is a
change of optimization coordinates in real algebra, requiring independent
numerical checks, not a different feature normalization, objective or model.
Another cap would remain optimization-inconclusive. No such preconditioner
has been evaluated here, and no further solve, new cohort, backbone update,
paid resource or hardware execution is admitted by this proposed question.


## Same-objective Gram-preconditioned readout (2026-09-15)

**Producer and separate independent audit complete successfully.** The single declared CPU FP64
solve reuses the previously audited frozen-feature cache and changes only
optimization coordinates. All 1,024 training windows, the 64 sampled-training
windows and development windows 88–151 remain fixed. The affine correction
starts at positive zero and retains the original physical ridge penalty
`lambda=0.001`. No backbone forward, gradient or update, new training pair,
reserved-development/final-test evaluation, paid resource or FPGA execution
occurs. The historical sources, controls and saved declarations remain intact.

| Fixed readout/control | Sampled-training CE | Development CE |
| --- | ---: | ---: |
| Frozen-core head | 2.153430 | 2.598400 |
| Bias-only correction | 2.149490 | 2.590785 |
| Earlier capped affine correction | 2.061244 | 2.573624 |
| Gram-preconditioned affine correction | 1.978804 | 2.603929 |
| First-pass GRU reference | 1.948048 | 2.371154 |

Conditioning reduces the original full-training regularized objective from the
previous affine endpoint by **0.05855947**, to **2.03052249** (full-training CE
**2.00209854**, ridge penalty **0.02842396**). These full-training statistics
are distinct from the sampled-training scores in the table. New-minus-old-affine
CE is **−0.08243983** on sampled training and **+0.03030490** on development.
Development CE also worsens by **0.00552890** versus the frozen-core head.
Development accuracy rises to **31.9214%**, versus **31.8237%** for the previous
affine endpoint, but that does not pass the CE quality gate. The sampled-training
and development gaps to GRU remain **0.03075619 / 0.23277472** nat.

The solve stops at **200 accepted iterations / 208 attempted and completed
full objective-gradient calls**, with no restart. Its original-coordinate
numerical gap estimate is **0.01503954**, above the declared **1e-5** target.
The target is not met and optimization remains inconclusive. The estimate is
not a measured remaining loss gap or a formal floating-point certificate.
Better fitting of this same objective does not improve the repeatedly used
development slice. These results do not prove a backbone capacity limit,
identify a unique cause of the development deficit or establish an FPGA-ready
head. Neither the numerical-completion nor the joint quality-progress gate
passes. No recipe is promoted and no automatic follow-up is launched.

The training-only augmented Gram has estimated condition number **6.963e6**.
Its fixed factor passes normalized backward-error checks; the inverse absolute
residual is **3.776e-12** and the measured whitening residual is **6.427e-10**.
This is a numerical coordinate transformation of the same objective, not a
changed feature penalty or an exact finite-precision whitening claim.
Construction takes **1.4562 s**; the solve takes **224.6474 s** by UTC elapsed
(**224.6456 s** monotonic). The 208 solver calls represent **27,262,976**
labeled-position evaluations on the same 131,072 distinct training pairs.
Four producer endpoint-gradient checks and four producer training-Gram passes
are verification/preparation costs, separate from fitting. An independent audit
adds one Gram pass and two endpoint-gradient checks. Actual calls and wall time
are not matched to the previous affine or GRU controls; concurrent local tests
also make these timings unsuitable for an efficiency comparison.

The input-only preflight verifies **86 source identities**, the prior **1,097
payloads / 941,551,289 bytes**, the saved audit capsule and actual corpus/
checkpoint ancestry, with zero attempts at instrumented numerical/subprocess
entry points. An initial preflight without the runner's one-thread runtime was
rejected for its thread-count mismatch; rerunning admission with the prescribed
thread limits passes without a fit or output attempt. Output paths were absent
and the required disk reserve was available before the single producer started.
The new producer seals **eight payloads / 8,194,602 bytes**. Source and declaration
bytes remain frozen through the producer and separate audit terminal exits.

The conditioning suite passes **82 synthetic tests**. The default regression
run has **5,153 passes**, 14 skips, 10 deselections and three failures caused by
the sandbox denying loopback sockets. All eight TCP tests pass in a separate
run with loopback access, yielding **5,156 distinct passing tests** across the
two runs. The isolated native fresh-window C-simulation test also passes.
Ruff, strict documentation, 40 shell syntax checks, artifact inventory,
provenance inventory and PDF freshness checks pass. Provenance inventory still
reports five unresolved release records; this is not remote CI or release
clearance. Local stack: macOS arm64, Python 3.11.5, PyTorch 2.10.0,
NumPy 1.26.4, SciPy 1.17.1.

| Invocation | Start → actual completion UTC (2026-09-15) | Elapsed UTC / monotonic seconds |
| --- | --- | ---: |
| Producer | 16:06:09 → 16:12:43 | 393.592237 / 393.599425 |
| Independent audit | 16:13:35 → 16:15:53 | 137.889990 / 137.888856 |

Sources: [producer record](../evidence/fpga_lm_readout_preconditioned_2026-09-15.json)
and [independent audit](../evidence/fpga_lm_readout_preconditioned_audit_2026-09-15.json).
Private output: `manuscript/build/fpga_lm_readout_preconditioned_2026-09-15/v1`;
private audit: `manuscript/build/fpga_lm_readout_preconditioned_audit_2026-09-15/v1`.

The independent audit checks one training Gram, both coordinate endpoints,
two full original-coordinate autograd gradients, and **128 initial plus 128
final score windows**. It does not reconstruct the inherited backbone features
or replay unretained optimizer trial arrays. Its exact three-file capsule is
**270,597 bytes**, with a **14,462-byte** public receipt. A separate stdlib-only
postcheck verifies all 86 source hashes, eight sealed payloads, unchanged saved
declarations through both terminal exits, predecessor receipt pins and exact
private/public audit bytes. No numerical execution is part of that postcheck.

| Frozen identity | SHA-256 |
| --- | --- |
| Saved declaration | `f72831a15a08d761fd5f2924f5061120e086f8635abaa0dd05591b1926fa9819` |
| Producer public | `0d7842612cfa2e2c099b38daa3f8d8b38b1a95788b79086488b96caeae929c3d` |
| Producer seal | `84ec461027562748fb0e2ffa87491d30041b4aee8d430b416d745ac861d3222d` |
| Audit public | `8f94606ef081ea09cac96721ea154cf4b4f2ac97e73343c1d33d5be98cb95702` |
| Audit manifest | `0872b2707f31eb1e9815ea8cb3239d27192a2d44071bfeebcb405d392c50bacd` |

**Handoff decision:** review the fitting/development tradeoff before choosing
another bounded experiment. The current result supports better optimization of
the fixed training objective; it does not support paid hardware scaling or a
new claim of useful language pretraining. See the [Xiang handoff](../README.md#handoff-for-xiang).
