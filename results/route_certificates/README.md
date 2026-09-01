# Route-certificate evidence

This directory preserves the evidence used by the route-certificate manuscript.

- `reference/` contains the small registered-operator and residual-DAG records.
- `actual_morphssm/exp1_benchmark/` contains actual-model benchmark runs, including
  incidents and per-cell records.
- `actual_morphssm/exp1_saved_state/` contains the retained-state audit that supports
  the configuration-specific 2.45--3.07x payload/checkpoint ratios.

The migrated rows retain their historical provenance. They support a working draft but
are not yet a clean-checkout TikiTorch reproduction. Submission tables must be regenerated
from append-only native TikiTorch runs and must disclose both packed payload and portable
serialized-plus-boundary accounting.

Do not infer latency, allocator/device peak, memory traffic, energy, or arbitrary
whole-model activation savings from these records. Those endpoints belong to the pending
five-arm comparison described in the [paper README](../../papers/route_certificates/README.md).
