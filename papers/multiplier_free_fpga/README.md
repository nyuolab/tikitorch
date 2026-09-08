# Multiplier-Free Learning on FPGA

> Historical draft, superseded on 2026-09-08 by the unified
> [TikiTorch manuscript](../../manuscript/README.md). The scientific sources below
> are preserved for provenance, not maintained as a separate submission. Former
> separate-paper and venue instructions are historical. Make new scientific
> edits in `manuscript/`; do not relabel old hardware measurements as current code.

Status: evidence-bounded discussion draft, suitable for technical review but
not submission-ready.

This directory contains the standalone hardware/systems paper for TikiTorch's
on-silicon multiplier-free learning results. The draft intentionally separates:

- width-512 MorphSSM fused execution with an exact observable-record match and
  persistent updates;
- preliminary width-128 throughput/replication measurements; and
- held-out learning from the smaller fabric-trained MNIST classifier.

The D512 exactness statement is limited to the archived observable record; it
does not imply exhaustive equality of every internal signal. The D128 replica
ratios are 1.86x at two replicas and 3.72x at four, relative to the single
replica. The two MNIST evaluators report 74.75% and 74.13%; that difference is
unresolved pipeline variation until both use frozen identical inputs and
preprocessing.

The draft does not claim an iso-model GPU speedup, a board-level energy win,
timing closure for the preliminary replica builds, or language generalization
from the current D512 resident-window run.

## Build

From this directory:

```bash
latexmk main.tex
```

The local `.latexmkrc` selects PDF mode, fails on LaTeX errors, and keeps all
generated files under `build/`. The review PDF is `build/main.pdf`.

Clean generated files with:

```bash
latexmk -C main.tex
```

The source uses a standard `article` class and BibTeX; no venue style is locked
yet. Generated PDFs and auxiliary files are build products and should not be
committed.

## Submission gates

- Run an iso-model, iso-update CUDA comparison.
- Train on disjoint streaming windows and evaluate held-out language quality.
- Archive final HLS, utilization, and post-route timing reports; require timing
  closure with margin for every headline build.
- Measure full-card FPGA power with uncertainty and a matched GPU protocol.
- Freeze preprocessing and repeat the MNIST result with matched controls/seeds.
- Complete the related-work matrix before making any priority claim.
- Package source, build flags, AFI provenance, host drivers, and machine-readable
  manifests into an immutable release.

These gates are claim-changing, not editorial cleanup. Until they close, the
paper's durable result is feasibility: a full multiplier-free learning state
transition executes on FPGA fabric with a scoped observable-record match, and a
separate small classifier is trained by fabric-only updates.

## Evidence roots

- `../../manuscript/results/fpga_silicon/d512_verify/`
- `../../manuscript/results/lm_shakespeare/fig_hallmark_lm.md`
- `../../manuscript/results/predictive_coding/ONFABRIC_TRAINING.md`
- `../../manuscript/results/predictive_coding/onfabric_train/`
- `../../tikitorch/audit.py`
- `../../tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp`
- `../../tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp`

The research branch was integrated into `main` at merge commit `0a12c8b`, but
this discussion draft does not yet name an immutable release candidate. The
artifact ledger marks that identifier as pending; replace it only when a clean
candidate is selected and the claim-bearing checks have been rerun from that
revision.
