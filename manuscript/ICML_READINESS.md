# ICML readiness: what is finished and what still blocks submission

Assessment: 2026-09-08. **Substantially revised scientific draft; not yet ready
to submit.** Formatting a paper and strengthening its claims cannot substitute
for the comparative experiments its central practical contribution requires.

## Completed in this pass

- One active TikiTorch manuscript; the two older drafts are preserved as history.
- An explicit joint state/parameter reverse contract, proofs and counterexamples.
  The tie counterexample distinguishes a selected reverse program from a global
  Clarke subgradient; the coding bound is conditioned on available information
  and is not mislabeled a universal memory lower bound.
- A fresh, source-fingerprinted current-code retained-state audit, with exact
  logits/routes and predeclared gradient tolerances; separate historical tables
  generated from raw records rather than transcribed ratios.
- Corrected and deduplicated primary-source citations, including morphological
  backpropagation, Gist/Few-bit and complete multiplication-free transformer
  training. No first-ever or new-gradient claim remains.
- An anonymous official ICML-style PDF, integrated appendices and impact
  statement; unmodified style hashes and main-body page checks in the build.
- Independent mathematical, literature and evidence reviews; explicit negative
  outcomes and hardware/source-rights qualifications.
- A [codebase/release roadmap](../docs/release_impact_roadmap.md) based on what the
  unified scientific story actually needs.

## Submission-blocking scientific work

1. **Demonstrate useful learning at a measured resource frontier.** The three
   current rows measure representation, not peak allocation, traffic or speed.
   Freeze and execute the [proposed experiment protocol](EXPERIMENT_PROTOCOL.md): same-model
   five-arm measurements, strong checkpoint policies (including compatible
   compiled variants and verified selective-save coverage) and a multi-seed held-out
   task with tuned conventional and relevant low-arithmetic/morphological
   baselines. Report quality/resource trade-offs and negative cells.
2. **Resolve or explicitly limit numerical support.** Native compiled AdamW
   fails the current strict update check, including ordinary compiled autograd.
   Identify the first divergence and validate the intended release stack.
   Changing optimizer or tolerances after seeing rankings is not resolution.
3. **Test the claimed generality.** The theorem covers more graphs than the
   current exact-class MorphSSM adapter. Either keep that implementation scope
   explicit or add a public operator-contract example on a second architecture
   with equivalent tests. A direct chain-rule proof alone is unlikely to carry
   the impact claim.
4. **Freeze and independently reproduce.** The new record binds exact current
   source hashes but reports a dirty tree. Build the final candidate, archive
   its source/environment, rerun selected evidence and regenerate the tables.
   Obtain an independent clean-environment reproduction before calling the
   artifact complete.

The FPGA appendix can remain carefully qualified feasibility context without
claiming energy or accelerator superiority. Expanding those claims additionally
requires frozen MNIST preprocessing and one-step KAT, disjoint-window D512
learning, matched reference checkpoints, timing-closed reports, matched CUDA
and full-card energy. Do not make all of those hardware projects prerequisites
for a narrower software paper unless its headline depends on them.

## Human and release authority gates

- Confirm authors, contributions/order, affiliations, acknowledgments, funding,
  conflicts and final scientific approval. Anonymous template names are not
  author decisions. Authors must independently check all proofs and citations.
- Resolve the AWS-derived RTL and retained-history distribution questions in
  `docs/release_provenance.json` before distributing a source artifact. Keep
  legally required notices; an anonymization policy is not permission to strip
  upstream attribution or licenses. A separate reviewed anonymous bundle may
  be needed.
- The latest verified official rules are the [ICML 2026 author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
  and [call for papers](https://icml.cc/Conferences/2026/CallForPapers).
  The 2026 conference has already occurred. The template is provisional;
  verify the actual next-cycle requirements, dates and submission policy when
  available. No next-cycle deadline or compliance is assumed.
- The cited cycle requires an anonymous main paper within eight pages,
  references/appendices in the same PDF and an impact statement. Check the
  actual target-cycle rules again before packaging or uploading anything.
- This work used AI assistance for drafting, coding and research checks. Human
  authors remain responsible; review the applicable disclosure requirements and
  describe assistance accurately rather than assigning authorship to tools.

## Adversarial review questions

1. What does the integrated contract enable that a morphological custom VJP,
   pooling-switch encoding or a tuned checkpoint policy does not already provide?
2. Does the advantage survive the vocabulary head, loss boundary, optimizer
   state, packing/decoding and live workspaces at useful validation quality?
3. Are purported exactness claims about real arithmetic, a declared tie program,
   tolerance agreement or bit identity—and are tests aligned with each claim?
4. Is every physical result tied to its actual numerical recipe/build, instead
   of being attributed to a newer Python implementation?
5. Could an independent researcher reproduce the main table and useful workload
   without privileged hardware, missing data or unreviewed source rights?

The decision is evidence-driven: an unfavorable speed result may still support
a valuable memory/capacity trade-off. If no useful workload advantage survives
strong baselines, narrow the paper rather than making the prose more assertive.
