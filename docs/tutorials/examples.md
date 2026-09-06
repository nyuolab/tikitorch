# Runnable examples

Examples use ordinary Python entry points and can be run from the repository root.
Most use synthetic data; read each script's arguments before starting a longer run.

| Example | Purpose | Suggested first command |
| --- | --- | --- |
| `closure_workflow.py` | Audit a hard path, pack its route, and replay a selected VJP | `python -m examples.closure_workflow` |
| `quickstart.py` | One `MorphSSM` training step | `python examples/quickstart.py` |
| `train_sequential_mnist.py` | PyTorch-style sequential classification | `python examples/train_sequential_mnist.py --smoke` |
| `closure_ablation.py` | Compare ingredients in a closure-oriented recipe | `python examples/closure_ablation.py` |
| `closure_ce.py` | Compare loss-conditioning boundaries | `python examples/closure_ce.py --smoke` |
| `backend_dispatch.py` | Inspect logical devices and CPU dispatch | `python examples/backend_dispatch.py` |
| `fpga_quickstart.py` | Inspect the FPGA catalog, fail-closed match, and PyTorch fallback | `python examples/fpga_quickstart.py` |
| `lion_vs_signsgd.py` | Compare sign, integer-momentum, and AdamW boundaries | `python examples/lion_vs_signsgd.py --smoke` |
| `topk_softening.py` | Inspect hard and softened top-k routing | `python examples/topk_softening.py --smoke` |
| `tropical_muzero.py` | Run the experimental planning demonstration | `python examples/tropical_muzero.py` |

The source and any additional flags are in the
[`examples/` directory](https://github.com/nyuolab/tikitorch/tree/main/examples).
The three character-LM ablations use deterministic built-in text by default and
accept `--text path/to/corpus.txt` for an external corpus.

!!! caution "Closure remains scoped"

    `closure_workflow.py` produces an operator-scoped route record and software-path
    audit. Wider examples can combine tropical layers with ordinary loss, optimizer,
    preprocessing, host, or fallback operations. Audit the concrete executed recipe
    before describing it as multiplier-free. Physical FPGA claims additionally require
    reports from the compiled build.
