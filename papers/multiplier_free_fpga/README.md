# Multiplier-Free Learning on FPGA

Status: evidence-bounded internal draft, not submission-ready.

This directory contains the standalone hardware/systems paper for TikiTorch's
on-silicon multiplier-free learning results. The draft intentionally separates:

- exact width-512 MorphSSM execution and persistent updates;
- preliminary width-128 throughput/replication measurements; and
- held-out learning from the smaller fabric-trained MNIST classifier.

It does not claim an iso-model GPU speedup, a board-level energy win, or
generalization from the current D512 resident-window run.

## Build

From this directory:

```bash
latexmk -pdf main.tex
```

Clean generated files with:

```bash
latexmk -C
```

The source uses a standard `article` class and BibTeX; no venue style is locked
yet. Generated PDFs and auxiliary files should not be committed until the
repository's paper-artifact policy is decided.

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

## Evidence roots

- `../../results/fpga_silicon/d512_verify/`
- `../../results/lm_shakespeare/fig_hallmark_lm.md`
- `../../results/predictive_coding/ONFABRIC_TRAINING.md`
- `../../results/predictive_coding/onfabric_train/`
- `../../tikitorch/audit.py`
- `../../tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp`
- `../../tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp`

The draft was initialized against repository head `9d13474` on branch
`pc-morphssm`. Update that provenance marker whenever claims or numbers change.
