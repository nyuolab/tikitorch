# Release process

This checklist separates a release-ready tree from the external acts of changing
repository visibility, creating a tag, archiving a DOI, and publishing a package.
Those external acts require maintainer approval.

## 1. Legal and provenance gate

- Confirm every distributed file has compatible provenance and update
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Scan the release tree and every ref that will become public for credentials,
  private data, vendor-restricted sources, and encrypted IP with a dedicated
  history scanner such as gitleaks or trufflehog; the manual candidate audit is
  not a substitute for an every-ref scan.
- Review retained evidence metadata—including workstation hostnames and absolute
  paths, AWS account/bucket identifiers, and commit-author email addresses—and
  either approve it for publication or redact it without weakening provenance.
- Review the redistribution terms for generated AMD/Xilinx reports and logs that
  retain vendor copyright headers; generated evidence is not automatically free
  of upstream licensing conditions.
- Review licenses of runtime, build, documentation, and action dependencies.
- Resolve the current-tree provenance of the DDR-controller instantiation and
  wiring in `cl_forge_muzero_ddr.sv`, `cl_forge_pc_ddr.sv`,
  `cl_forge_pc_train_ddr.sv`, and `cl_forge_train_step_s1.sv`. The source says
  the pattern was adapted from the AWS HDK `unused_ddr_template`, but the files
  do not yet carry a reviewed file-level provenance/license record. Before
  distribution, either replace that portion with independently authored code or
  preserve the required upstream notice under a legally approved licensing plan.
- Obtain a PyTorch Foundation trademark/brand review before describing TikiTorch
  as a partner, using a partnership lockup, or otherwise implying an official
  relationship. Until then, use the documented “independent companion” wording.
  The project name itself should be included in that review.

### Current history-review requirement

The `pc-morphssm` history previously retained snapshots copied from the AWS FPGA HDK,
including an encrypted Xilinx module. Those snapshots are removed from the `main`
tree and replaced by provenance notes, but the blobs remain reachable through the
merge parent.

That merge also makes 927 historical object/path records under
`ontogenic_learner/` reachable from `main` (667 distinct paths and about 94 MiB
of unique blobs). The sibling research is absent from the current tree, but a
visibility change would still publish its historical manuscripts, results, and
code unless the history is filtered or its inclusion is explicitly approved.

In addition, the local `tropical-attention-routing` ref reaches three large
post-route Xilinx DCP files and three matching `Developer_CL.tar` archives,
about 1.33 GB total. They are absent from `main` but require an explicit
provenance and publication decision before that ref is pushed or exposed.

Before making a private repository public, enumerate the remote refs that will be
published, then either complete a coordinated history purge of restricted or
unapproved artifacts across every such ref or obtain explicit legal approval for
their distribution. Deleting a branch or deleting files in a later commit does not
by itself remove reachable history.

As of the 2026-09-01 audit, the live remote has six branches:
`archive/pre-scope-split-2026-08-31`, `backup-attn-routing`, `main`,
`pc-morphssm`, `tropical-attention-routing`, and `tropical-predictive-coding`.
The archive and `pc-morphssm` refs retain both the ontogenic material and vendor
snapshots; `tropical-attention-routing` also retains subsets of those histories.
History remediation must cover all surviving public refs atomically, not only
the new `main` tip.

### Repository-control gate

- Configure and verify a ruleset or branch-protection rule that requires pull
  requests, green required checks, and code-owner approval. CODEOWNERS alone
  only requests review.
- Enable and test GitHub private vulnerability reporting and publish a monitored
  private conduct-reporting channel.
- Enable secret scanning, push protection, and the chosen dependency/code
  security controls. These settings are currently disabled on the private
  repository and do not replace a manual all-ref history scan.

## 2. Version and claims

- Choose the version once in `pyproject.toml`, mirror it in `tikitorch.__version__`
  and `CITATION.cff`, and verify them in CI.
- Move completed entries from `[Unreleased]` to a dated changelog section.
- Freeze the research claim ledger against the candidate commit. Do not promote open
  manuscript experiments into release claims.
- Replace draft author, DOI, and release-date metadata only with agreed values.

## 3. Verification

From a clean checkout, run:

```bash
python -m ruff check .
python -m pytest -q
python -m pytest -m kernel -q
python -m pytest -m slow -q
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Run the CI C-simulation job, build both manuscripts, install the wheel in a clean
environment, import every public namespace, run the quickstart, and exercise
`tt-closure`. Preserve failures, skips, compiler fallbacks, and unavailable hardware
as part of the release record.

## 4. Publish

1. Open a release pull request containing the final changelog and candidate version.
2. Require green protected-branch CI and maintainer approval.
3. Create a signed `vX.Y.Z` tag from the reviewed merge commit.
4. Build distributions from that tag and publish through an approved, protected
   environment using trusted publishing.
5. Create the GitHub release, archive it, add the DOI/date to citation metadata, and
   verify the installed artifact from a clean environment.
