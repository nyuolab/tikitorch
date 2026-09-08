# Research artifact layout and preservation

The research archive moved from root `results/` to `manuscript/results/` on
2026-09-08. Reusable Python code stays in `tikitorch/`; active research runners
stay in `scripts/`. There is no root compatibility symlink. Use a Git source
checkout for research: neither the paper nor its archive ships in the library's
wheel or source distribution.

## What was preserved

The move was checked with SHA-256 before and after: all 686 locally present
files, including ignored files, were byte-identical immediately after relocation.
The durable [relocation inventory](artifact_relocation.json) covers the 544
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

### Paper-directory consolidation amendment

Later on 2026-09-08, root `papers/` was consolidated into this workspace;
see the [archive map and source preservation record](archive/README.md).
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

## Reading and reproducing historical records

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
reports do not certify current Python code. Follow the [evidence ledger](EVIDENCE_LEDGER.md)
and [release gates](../docs/release_validation.md); third-party source and
retained-history review remain open.
