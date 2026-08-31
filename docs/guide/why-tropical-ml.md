# Why tropical algebra for ML/DL?

Tropical algebra replaces ordinary addition and multiplication with selection and
addition. In max-plus form, for example:

```text
y_j = max_i (x_i + w_ji)
```

That primitive is interesting for machine learning for three separate reasons. They
should not be collapsed into one claim.

## 1. It changes the representation

A hard tropical layer selects a winning input rather than forming a dense weighted
sum. This creates a routed, piecewise-linear representation with an inductive bias that
can suit morphological signal processing, content routing, and min-plus/max-plus
recurrences.

Whether that bias helps is task- and architecture-specific. TikiTorch provides models
and baselines for testing the question; the package name does not guarantee an accuracy
advantage.

## 2. The primitive maps naturally to add/compare hardware

The hard max-plus or min-plus operator itself contains additions and comparisons rather
than data-dependent multiplications. That can be valuable on FPGA or ASIC substrates.
The mapping claim applies to the operator, not automatically to an entire program.

A complete learning recipe may additionally include:

- a soft relaxation with exponential or logarithmic operations;
- dense input or output projections;
- normalization and dropout;
- loss construction and averaging;
- parameter-gradient outer products;
- optimizer state updates; and
- host preprocessing, transfers, or fallback execution.

TikiTorch's [closure tools](closure.md) make these boundaries observable in software.
Final hardware claims still require HLS and routed implementation reports.

## 3. It connects networks to tropical mathematics

Shortest-path relaxation, dynamic programming, mathematical morphology, and tropical
geometry use closely related min-plus or max-plus structure. A shared tensor substrate
makes it possible to study neural layers and those classical recurrences together.

See [Tropical geometry](tropical-geometry.md) and
[Beyond deep learning](beyond-deep-learning.md) for the mathematical view.

## What the current evidence supports

The canonical [research ledger](../research/index.md) distinguishes software
capabilities from measured hardware configurations. In summary:

- The PyTorch-shaped library surface and recipe-scoped software auditing are
  established by implementation and tests.
- Specific F2 builds support narrow multiplier-free inference and learning-execution
  results under their recorded configurations.
- A small fabric-trained MNIST model has a measured held-out result, with explicit
  preprocessing and physical-resource qualifications.
- Replica throughput, end-to-end energy, matched GPU comparisons, and broader learning
  quality still have unresolved gates.

Do not generalize one compiled kernel into “all TikiTorch models are multiplier-free,”
“the auditor proves zero DSP,” or “TikiTorch is faster or more efficient than PyTorch.”

## The practical trade-off

Hard selection creates sparse routing and removes dense multiply-accumulate structure.
That changes both optimization and capacity. A useful experiment therefore reports:

1. an ordinary learnability baseline;
2. a matched hard/soft or dense/tropical ablation;
3. the exact loss and optimizer boundary;
4. software closure diagnostics; and
5. compiled resource, timing, correctness, and power data when hardware is claimed.

Negative results belong in that comparison. The current ledger, for example, records a
strict MorphSSM language-quality gap in a matched host study and no present
end-to-end energy-win claim.

## When to use TikiTorch

TikiTorch is a good fit when the research question concerns:

- tropical or morphological neural representations;
- hard routing and selected-path differentiation;
- min-plus/max-plus algorithmic structure;
- recipe-level multiplier elimination; or
- FPGA/ASIC mappings backed by build-specific evidence.

If the only objective is mature operator coverage or optimized eager GPU throughput,
ordinary PyTorch is the better starting point. TikiTorch deliberately builds on it
rather than replacing it.
