# TikiTorch roadmap

TikiTorch is a PyTorch-shaped library for tropical, morphological, dyadic, and
hard-routed learning. The roadmap is library-first: make the supported surface
easy to install, learn, test, and extend while keeping research and physical
hardware claims tied to their exact evidence.

“PyTorch-shaped” means familiar namespaces, `torch.nn.Module` interoperability,
ordinary training-loop conventions, state dictionaries, and task-oriented
documentation. It does not mean complete PyTorch compatibility or that every
TikiTorch configuration is multiplier-free.

## 0.2 release gate

- [x] Public namespace map for `functional`, `nn`, `models`, `optim`, auditing,
      serialization, backend dispatch, and selected specialized APIs.
- [x] Five-minute installation and quickstart path.
- [x] Concept guide, migration guide, task-oriented recipes, and generated API
      reference.
- [x] Recipe-scoped closure guide distinguishing structural, executed-path,
      full-training-step, and physical-build evidence.
- [x] Green fast tests, lint, docs build, and isolated native-kernel tests.
- [x] Reconcile the development branch with the repository's default branch
      (`pc-morphssm` integrated into `main` at merge commit `0a12c8b`).
- [ ] Freeze the first documented public/experimental API inventory.
- [ ] Publish versioned documentation and a release candidate.
- [ ] Run install/import/example smoke tests from the built wheel and source
      distribution on every supported Python version.
- [ ] Cut `v0.2.0`, archive it with a DOI, and update citation metadata to the
      immutable release.

## Library priorities after 0.2

1. **API consistency.** Prefer namespace-qualified, PyTorch-familiar entry
   points; deprecate aliases deliberately rather than silently changing them.
2. **Recipe manifests.** Give closure reports and exported schedules a
   versioned, serializable schema binding model, loss, reverse rule, optimizer,
   inputs, and environment.
3. **Testing depth.** Add representative executed-path and full-step tests for
   every component advertised as part of a closed recipe, while keeping
   numerical correctness tests independent of closure checks.
4. **Backend contracts.** Make fallback, error, and native-binding behavior
   explicit for every logical device and FPGA entry point.
5. **Examples and recipes.** Keep one maintained example per supported task;
   move exploratory studies to `scripts/` and link their evidence rather than
   presenting them as stable tutorials.
6. **Release engineering.** Add compatibility notes, deprecation policy,
   wheel/sdist CI, documentation deployment, and a small downstream smoke
   project before considering a software-paper submission.

## Publication tracks

Scientific manuscripts have their own gates and do not define the whole
library's maturity.

- **Route certificates:** the root [manuscript handoff](manuscript/README.md)
  owns the selected-VJP theorem, schema, exact replay, and five-arm compiled
  comparison. It must not claim a new gradient or generic activation-free
  training.
- **Multiplier-free FPGA learning:** the
  [hardware paper](papers/multiplier_free_fpga/README.md) owns the physical
  kernels and silicon measurements. It still requires disjoint-stream held-out
  learning, matched CUDA, final routed reports, and defensible full-card power.
- **TikiTorch software:** pursue a software paper only after a tagged stable
  release, hosted versioned docs, external use, and reproducible examples.

The canonical [claim ledger](docs/research/CLAIM_LEDGER.md) controls wording for
all three tracks.

## Explicit non-goals

- Reimplementing the complete PyTorch tensor, compiler, distributed, or data
  ecosystem.
- Treating a tropical operator name or a structural audit as proof that a whole
  training loop or hardware build contains no multipliers.
- Claiming arbitrary-model FPGA compilation, GPU parity, an energy advantage,
  or production maturity without matched evidence.
- Maintaining claim-bearing experimental numbers in tutorials or hand-edited
  tables without append-only raw rows and provenance.
- Housing developmental transfer, lifelong memory, embodiment, or true-agent
  experiments. Those belong to the separate `ontogenic-agent` programme.

## Change policy

Every user-visible addition needs documentation, ordinary correctness tests,
and a changelog entry. Anything advertised as closed additionally needs the
audit level appropriate to its claim. Anything advertised as a physical result
needs exact synthesis/routed reports and workload provenance. Negative results
remain visible when they narrow a public claim or close a research direction.
