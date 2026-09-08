# Board-scaling artifacts

This directory preserves the recorded GridWorld, Connect4, and chess experiment outputs together
with their narrative summaries. Treat the JSON/JSONL files as historical run records; do not infer
claims beyond the measured fields and sample sizes in the corresponding Markdown summaries.

## Terminology note

Some historical JSON `note` fields use shorthand such as `fully multiplier-free`, `zero-DSP`, or
`closure []`. In those records, an empty closure list describes the audited PyTorch **model**
operators for that configuration. It does not cover Python MCTS (PUCT multiplication and square
root, min-max division, and floating-point tree statistics), exact softmax/loss and entropy paths,
replay/orchestration, or a physical FPGA implementation. A DSP claim is valid only for an explicitly
named kernel/configuration with a matching synthesis or route report.

The current interpretation and caveats live in:

- `TASK3_BENCHMARK.md` for the cross-domain result;
- `CONNECT4_PRIOR_SWEEP.md` for the paired Connect4 sweep;
- `CHESS_MUZERO_COMPARISON.md` for chess quality, throughput, energy, and silicon scope.
