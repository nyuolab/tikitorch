# Combined TikiTorch manuscript: evidence ledger

Review date: 2026-09-08. This ledger records the evidence available for the
combined manuscript, including a separately recorded current-code experiment;
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
| Current device-route payload is smaller than a specified checkpoint boundary set | `manuscript/evidence/retained_state_cleanup_2026-09-08.json`, `measure_retained_state.py`, `generated/current_retained_state.tex` | Current-code CPU regression remeasurement after library cleanup, source-hashed dirty tree; exact logits/canonical routes and selected-VJP checks. Both earlier `evidence/` records are preserved as prior source snapshots. Main table. Comparator is analytical boundary storage, not measured peak or an optimal checkpoint policy. |
| Historical portable representation is smaller than a specified checkpoint-retained set | Run `529c45f4-7274-4999-b35c-c59d1db917e6`, detailed below | Historical imported representation evidence. Three batch-one FP32 CPU configurations, not a current GPU/full-step/optimized-checkpoint result. Appendix only. |
| Five-arm full-step validation exists | `scripts/route_certificates/training_benchmark.py`; `docs/codebase_audit_2026-09-08.md`; local run `14f00d16-e3ea-4622-9c80-c03d30bb259f` | Local CPU protocol smoke reported by the code audit, not a checked-in comparative research dataset. Width 16, one layer, vocabulary 11, batch 2, sequence 12, seed 0, momentum SGD, five total updates. Do not publish latency or capacity superiority from this run. Implementation/validation scope only. |
| D512 fused training executes with a matching archived observable record | `manuscript/results/fpga_silicon/d512_verify/{silicon_fused,golden_fused}.jsonl`, `BITEXACT`; AGFI `agfi-003a8a2f2b1ba7df7` | Historical silicon feasibility. The 30-update records agree on resident accuracy, weight CRC and 16 predictions. Not full-array, every-signal, or 2,000-step trajectory equivalence. Appendix; at most a bounded main-text feasibility sentence. |
| D512 maintains a live fused update loop for 2,000 updates | `manuscript/results/fpga_silicon/d512_verify/train_2000.err` and `train_2000.jsonl`; `tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp`; launcher `scripts/run_morph_lm_d512_verify_f2.sh` | Four layers, D512, vocabulary 128, seven taps, one resident 128-position window. 256,000 processed positions in 418.3963 s; log reports 612 token/s. No held-out language result, distinct-token count, final-route certification or accelerator comparison. Appendix. |
| Fabric-produced weights generalize on downsampled MNIST | `manuscript/results/predictive_coding/onfabric_train/{train,crosscheck}.txt`, dumped `trainedW0.i32`, `trainedW1.i32`; recipe report `manuscript/results/predictive_coding/ONFABRIC_TRAINING.md` | Historical small-task feasibility: 256→256→10 dyadic MLP, 16×16 inputs, batch 32, five epochs, 60,000 training/10,000 test examples. Final driver test accuracy 74.75%; independent exported-weight evaluation 74.13%. Unreconciled pipelines, not parity. Appendix. |

The theorem's operator inventory is broader than the validated production
bridge. The present `route_logits` path explicitly rejects, among other cases,
parameter aliases, hooks/forward overrides, non-reference custom backends,
non-unit residual scaling, recurrent scans, dense heads, soft routing and
higher-order differentiation. A theorem about composition does not establish
that all such graph features have been implemented in that bridge.

## Current-code main table

The current record fixes batch one, vocabulary 38, top-k head k=8, kernel seven,
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

Run `python manuscript/measure_retained_state.py --check` to verify the selected
record and table. It fails after executable source changes; a new result must
be created at new paths and explicitly selected, not overwrite the old one.

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

`generate_tables.py` checks pinned SHA256 identities for every selected raw
input, rejects duplicate/missing/unexpected selected cells, checks the middle
cell against the manifest, derives boundaries, and ignores the legacy
`ratio_*` fields. Run:

```bash
python manuscript/generate_tables.py --check
python -m pytest -q tests/test_manuscript_tables.py
```

The exact source hashes, source revisions, environments, accounting equations
and selected numerical rows are in
[`generated/table_provenance.json`](generated/table_provenance.json). Generated
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
current-code retained-state table with provenance, and the protocol required to
test practical utility. Keep the distinctive claim about useful derivative
state and explicit execution contracts, not generic tropical arithmetic novelty.

**Appendix:** proof details, complete operator/auxiliary-field inventory,
historical portable-table accounting, replay failure cases, hardware feasibility records,
normalization/target caveat and failed timing/comparison gates.

**Omit as positive results:** unmatched accelerator speedups, energy projections,
full-state bit-exact language, broad activation-free training, extrapolated GPU
capacity, obsolete test counts as current validation, and legacy comparisons
that mix portable tape state with current device packing or optimizer/loss state.

The code audit reports native compiled AdamW update-parity failures even for
ordinary compiled autograd on the tested machine; momentum-SGD passes do not
establish AdamW equivalence. CUDA validation and real-task multi-seed matched
experiments are unrun. A clean source-bound submission artifact, frozen
supported software stack and named author/affiliation/venue decisions also
remain necessary. Physical hardware work is required only for claims actually
retained; a focused software paper need not make unmeasured energy or FPGA
superiority claims to justify its contribution.

Distribution clearance is an independent gate:
`docs/release_provenance.json` still records four unresolved AWS-derived RTL
records and retained-history review. Neither manuscript improvement, a package
build, nor passing tests resolves those rights/authority questions.
