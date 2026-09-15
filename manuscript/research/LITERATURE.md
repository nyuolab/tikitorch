# TikiTorch: literature positioning

Primary-source check: 2026-09-08. This is an editorial research note, not a
claim that the literature search is exhaustive. The bibliography is
[`refs.bib`](../source/refs.bib); preprints remain labeled as such.

## Recommended scientific position

The ICLR-oriented narrative starts with a mathematical payoff: **additive hard
routing turns backward computation into routing and accumulation**. Once the
winner of `max_i(x_i + w_ji)` is known, both input and weight gradients can be
formed from the incoming gradient and that index. Unlike a dense weight gradient,
neither needs the original activation values. This simplifies the supported
backward dependencies, not the forward search for winners or every operation
in a full learning step.

TikiTorch is the PyTorch companion that makes this structure usable: tropical
operators and modules, model composition, device-resident route tapes and
ordinary autograd/optimizer interfaces, backed by explicit derivative rules and
tests. Arithmetic audits additionally distinguish practical hybrid learning
from fully constrained recipes. Historical hardware artifacts test a separate,
configuration-specific implementation boundary; they are not the central
justification for the software contribution.

This is an integrated representation-and-systems contribution. Neither max-plus
networks, routing gradients, compact backward state, nor multiplication-free
training is new by itself. A defensible manuscript explains exactly which
operator/parameter contracts its representation supports and measures what the
integration buys. A smaller serialized record alone establishes neither lower
training peak memory nor faster or more energy-efficient learning.

## Closest-prior-art and comparator matrix

| Prior work | What is already established | TikiTorch distinction and necessary comparison |
|---|---|---|
| [Ritter & Sussner, 1996](https://doi.org/10.1109/ICPR.1996.547657); [Franchi et al., 2020](https://www.sciencedirect.com/science/article/pii/S0031320320300522) | Morphological neurons replace sum-of-products with extrema of sums; exact morphological layers can be learned in deep networks. | Do not claim invention of tropical layers or their backpropagation. Credit these directly before introducing the tape and audit contract. |
| [Groenendijk et al., 2023](https://pure.uva.nl/ws/files/161085600/Geometric_Back-Propagation_in_Morphological_Neural_Networks.pdf) | Geometric correspondences support morphological backpropagation; probe-geometry learning has a dedicated update rule. | The closest morphological differentiation comparator. Separate selected framework VJPs from their geometric learning rule; novelty cannot be merely sending error through winning correspondences. |
| [Zhang et al., 2018](https://proceedings.mlr.press/v80/zhang18i.html) | ReLU networks have tropical-rational geometric representations. | A geometric representation theorem does not establish multiplier-free execution of the original dense parameterization. Do not equate these two meanings of tropical. |
| [Fotopoulos & Maragos, 2025, revised 2026](https://arxiv.org/abs/2505.09710v4) | Deep morphological expressivity and sparse/uninformative-gradient limitations; modified architectures with approximation results and empirical training. | Important current architecture comparator. Test gradient coverage, depth, residuals, max/min mixing, and soft-to-hard schedules. Do not infer general-purpose approximation or trainability from a local VJP theorem. |
| [Maxout, 2013](https://proceedings.mlr.press/v28/goodfellow13.html); [pooling switches, 2014](https://arxiv.org/abs/1311.2901) | Hard branch selection and recorded pooling locations are established. | A maxout branch containing a learned dense map still needs its input for the weight outer product. Show this counterexample next to additive weights, not a generic claim that hard routing eliminates activations. |
| [Gist, 2018](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf) | Operator-specific exact and lossy encodings reduce retained training state; exact one-bit ReLU/pool state is explicit. | Exact backward-state encoding predates TikiTorch. The proposed scope extends to a declared trainable additive operator set, with portable/device tape contracts. Include equivalent liveness and retained-state accounting. |
| [Few-bit Backward, 2023](https://proceedings.mlr.press/v202/novikov23a.html); [GACT, 2022](https://proceedings.mlr.press/v162/liu22v.html) | Few-bit representations of pointwise derivatives and architecture-generic approximate activation compression. | Distinguish lossless selected-VJP encoding from gradient approximation. A quality-constrained comparison is needed if approximate compression is used as a systems baseline. |
| [Revolve, 2000](https://doi.org/10.1145/347837.347846); [Chen et al., 2016](https://arxiv.org/abs/1604.06174); [Checkmate, 2020](https://proceedings.mlsys.org/paper_files/paper/2020/file/0b816ae8f06f8dd3543dc3d9ef196cab-Paper.pdf); [RevNets, 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f9be311e65d81a9ad8150a60844bb94c-Abstract.html) | Recompute schedules and reversible architectures reduce activation storage with distinct cost/architecture tradeoffs. | Routes avoid reconstructing covered local primals; checkpointing remains complementary at uncovered boundaries. Compare several checkpoint policies and the best feasible policy, not just one weak segmentation. |
| [Bolte & Pauwels, 2020](https://proceedings.neurips.cc/paper/2020/hash/7a674153c63cff1ad7f0e261c369ab2c-Abstract.html); [Bolte et al., 2023](https://arxiv.org/abs/2206.01730) | Program-dependent nonsmooth AD and its complexity have established mathematical treatments. | Define the selected program pullback. A composition of local selections must not automatically be labeled a classical derivative or Clarke subgradient of the global function. |
| [AdderNet, 2020](https://arxiv.org/abs/1912.13200); [DeepShift, 2021](https://arxiv.org/abs/1905.13298v5); [ShiftAddNet, 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1cf44d7975e6c86cffa70cae95b5fbb2-Abstract.html); [Liu et al., 2023](https://arxiv.org/abs/2302.14458) | Addition/shift-oriented layers and low-precision multiplication-free forward/backward mechanisms already have substantial empirical evaluation. | Compare operator semantics, parameter gradients, losses, optimizers, and exact audited boundaries. Do not characterize all prior work as inference-only. An L1-adder sign pattern is not the same storage object as one extremum winner. |
| [Kosson & Jaggi, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/19df21cd4931bd0caaa4d8480e9a59cd-Abstract-Conference.html) | Piecewise-affine bit-level arithmetic eliminates products across forward, backward, and optimizer updates, with transformer experiments. | **Mandatory closest prior art for the closure claim.** TikiTorch cannot claim the first complete multiplication-free training loop. Contrast native additive tropical parameterization and explicit route/audit contracts with their arithmetic substitutions. |
| [Scalable MatMul-free LM, 2024, revised 2025](https://arxiv.org/abs/2406.02528v7); [BitNet, 2024](https://arxiv.org/abs/2402.17764); [BitNet 2B4T, 2025](https://arxiv.org/abs/2504.12285v2) | Strong low-bit and matmul-free language-model results at billion-parameter scale; mature efficiency-oriented competitors. | Eliminating matrix multiplication or using ternary weights is not an audit of every operation in a training step. These works also set a much higher modeling-utility bar than a short repeated token window. Avoid unrelated cross-paper speed/energy ratios. |
| [FlashAttention, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) | Exact algorithms can gain efficiency by explicitly reducing movement across memory levels. | This is a methodological standard, not an iso-model comparator. Measure bytes moved, peak memory, packing/decoding, and full-step latency rather than extrapolating from arithmetic counts. |
| [FINN, 2017](https://arxiv.org/abs/1612.07119) | Binarized FPGA inference has established framework support. | Distinguish full on-fabric updates from inference acceleration, but do not turn that distinction into a first-ever claim about FPGA training. Compare precision, board power, timing closure, and full workloads when claiming efficiency. |

The matrix's TikiTorch distinctions are interpretations of the cited work and
the implementation, not experimentally established superiority.

## Current-work screening notes

The latest checked version of Fotopoulos & Maragos is August 2026, not its
initial May 2025 abstract. Its updated argument makes expressivity and sparse
gradient coverage especially important experiments. The bibliography pins that
version. The latest checked MatMul-free LM version is July 2025 and has an
expanded author list; results must not be silently mixed across versions.

[Teska et al., 2026](https://arxiv.org/abs/2605.13894v2) study tropical axial
attention for phylogenetic distance inference. It is useful evidence of a
domain-specific tropical inductive bias, not direct evidence for TikiTorch's
training-memory claim. It is screened here without inflating the main
bibliography. This search does not establish that no closer 2026 paper exists;
refresh it before submission.

## Experiments needed to defend impact

1. **Semantic comparison:** same model, initialization, batches, loss, optimizer,
   and repeated updates across eager, compiled, checkpoint, unpacked-route, and
   packed-route paths. Report failures, numerical differences, exact-tie tests,
   and unsupported boundaries. Distinguish real-arithmetic identity from
   floating-point tolerance and bit identity.
2. **Systems comparison:** tuned checkpoint policies; packing and decoding
   inside full-step timing; device peak allocation and reservation; compile
   startup separately; memory traffic where measurable. Report a Pareto curve,
   not only the fastest or smallest favorable cell.
3. **Modeling comparison:** held-out tasks with independently sampled or
   disjoint examples, multiple seeds, matched optimization budgets, conventional
   dense baselines, and a relevant morphological/low-arithmetic alternative.
   Include gradient density, active-weight coverage, route churn, and
   soft-to-hard ablations. A deterministic onboarding regression is not a
   benchmark-quality generalization result.
4. **Hardware comparison:** freeze numerical recipe, source/build identifiers,
   preprocessing, host boundary, and exported-weight evaluation. Resolve the
   two MNIST evaluators before treating their accuracy as one reproducible
   result. Retain observed silicon behavior as feasibility evidence until
   iso-workload throughput and full-card energy measurements exist.

Venue requirements and human submission decisions are maintained only in the
[manuscript README](../README.md#submission-and-release-gates).
