# Runnable examples

Examples use ordinary Python entry points and can be run from the repository root.
Most use synthetic data; read each script's arguments before starting a longer run.

| Example | Purpose | Suggested first command |
| --- | --- | --- |
| `quickstart.py` | One `MorphSSM` training step | `python examples/quickstart.py` |
| `train_sequential_mnist.py` | PyTorch-style sequential classification | `python examples/train_sequential_mnist.py --smoke` |
| `closure_ablation.py` | Compare ingredients in a closure-oriented recipe | `python examples/closure_ablation.py` |
| `closure_ce.py` | Exercise the closure-aware loss path | `python examples/closure_ce.py` |
| `backend_dispatch.py` | Inspect logical devices and CPU dispatch | `python examples/backend_dispatch.py` |
| `fpga_quickstart.py` | Inspect the FPGA catalog, fail-closed match, and PyTorch fallback | `python examples/fpga_quickstart.py` |
| `lion_vs_signsgd.py` | Compare two sign-oriented optimizers | `python examples/lion_vs_signsgd.py` |
| `topk_softening.py` | Inspect soft and hard top-k routing | `python examples/topk_softening.py` |
| `tropical_muzero.py` | Run the experimental planning demonstration | `python examples/tropical_muzero.py` |

The source and any additional flags are in the
[`examples/` directory](https://github.com/nyuolab/tikitorch/tree/main/examples).

!!! caution "Examples are not closure certificates"

    A script can combine tropical layers with ordinary loss, optimizer, preprocessing,
    host, or fallback operations. Use the [closure audit](../guide/closure.md) on the
    concrete executed recipe before describing it as multiplier-free. Physical FPGA
    claims additionally require reports from the compiled build.
