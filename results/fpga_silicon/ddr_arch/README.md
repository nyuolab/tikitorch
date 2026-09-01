# AWS HDK DDR architecture inspection

This directory retains TikiTorch-produced logs and indexes from a 2026-06-15
inspection of the AWS FPGA HDK DDR/HBM examples. The inspection established which
shell interfaces and DMA staging paths a TikiTorch F2 design would need; it is not a
redistributable copy of the HDK.

Vendor HDL, XDC files, templates, and encrypted IP that were captured during the
experiment are intentionally excluded from the release tree. Obtain the AWS FPGA HDK
from its official repository and accept its license separately. The source locations
observed during the run remain in `ref_index.txt`, `interfaces_ls.txt`,
`ddr_module_index.txt`, `cl_ports_mem.txt`, and `host_ddr_stage.txt`.

`VENDOR_SNAPSHOT_MANIFEST.tsv` records the Git blob identities and captured filenames
for provenance only. It is not a retrieval mechanism and does not grant permission to
redistribute those blobs. See the current history-review gate in
[`RELEASING.md`](../../../RELEASING.md) before changing repository visibility.
