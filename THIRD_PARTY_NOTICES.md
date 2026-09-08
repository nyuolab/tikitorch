# Third-party components

TikiTorch is distributed under GPL-3.0-or-later as stated in [LICENSE](LICENSE).
Its dependencies and optional hardware toolchains retain their
own licenses.

## PyTorch

TikiTorch depends on PyTorch but does not redistribute PyTorch in its source tree.
PyTorch is an independently governed project with its own license and trademarks.
Installing TikiTorch may cause a package manager to install PyTorch separately.
TikiTorch is not affiliated with or sponsored by the PyTorch Foundation. See the
[PyTorch brand guidelines](https://pytorch.org/brand-guidelines/) before using its
name or brand materials in release promotion.

## AWS FPGA HDK and Xilinx tools

Optional AWS F2 workflows require an AWS FPGA HDK checkout and, for physical builds,
the applicable AMD/Xilinx tools and IP. Captured vendor source snapshots are not
licensed as part of TikiTorch and are intentionally excluded from the current tree.
Users must obtain vendor toolchains and templates from their vendors and comply with
the corresponding terms.

The current release-candidate audit found four TikiTorch RTL files whose own
comments or structure indicate that their DDR-controller instantiation/wiring
was adapted from the AWS HDK `unused_ddr_template`:

- `tikitorch/hardware/hdk/cl_forge_muzero_ddr/design/cl_forge_muzero_ddr.sv`
- `tikitorch/hardware/hdk/cl_forge_pc_ddr/design/cl_forge_pc_ddr.sv`
- `tikitorch/hardware/hdk/cl_forge_pc_train_ddr/design/cl_forge_pc_train_ddr.sv`
- `tikitorch/hardware/hdk/cl_forge_train_step_s1/design/cl_forge_train_step_s1.sv`

Their file-level provenance and licensing treatment have not yet been resolved.
They must not be included in a public release until the adapted portion is
independently replaced or the required upstream notice and licensing plan receive
legal approval. Resolving that provenance is a prerequisite for public release.

The tracked [provenance inventory](docs/release_provenance.json) fingerprints these
four sources and records the separate retained-history review. Run
`python scripts/check_release_provenance.py` before publication or a visibility
change: it fails while any record is unresolved. Ordinary CI uses
`--inventory-only`, which checks record integrity and explicitly does **not** grant
release clearance. See [release validation](docs/release_validation.md) for the
evidence and maintainer authority still required; no legal clearance or independent
replacement has been inferred from the local comments alone.

The historical snapshots identified in
`results/fpga_silicon/ddr_arch/VENDOR_SNAPSHOT_MANIFEST.tsv` carry Amazon Software
License notices, whose redistribution conditions include retaining that license and
whose use is limited to Amazon services. The snapshots were not relicensed under
TikiTorch's GPL. See the [Amazon Software License](https://aws.amazon.com/asl/).

The retained files under `results/fpga_silicon/ddr_arch/` are TikiTorch-produced
indexes and diagnostic logs describing an earlier HDK inspection. They do not
include the inspected vendor HDL, constraints, templates, or encrypted IP.

Some non-`main` Git history also contains large routed Xilinx design checkpoints
and `Developer_CL.tar` archives. They are not part of the current release tree.
Reviewing and cleaning that history is a prerequisite for public release.

## Research references

Bibliographic entries and links identify related work; they do not incorporate the
referenced implementations into TikiTorch. If a future contribution vendors
third-party code or data, it must include provenance, its complete applicable license,
and compatibility review before release.
