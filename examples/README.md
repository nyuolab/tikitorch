# TikiTorch examples

Runnable, PyTorch-idiomatic usage demos. Each script is self-contained and runs
without downloads (synthetic data; `train_sequential_mnist.py` adds a `--smoke`
flag for its no-download path). Examples demonstrate APIs; only an explicit
audit of the complete executed recipe supports a closure claim.

| script | what it shows |
| --- | --- |
| [`quickstart.py`](quickstart.py) | The whole loop in ~10 lines: `MorphSSM` + `PureTropicalSGD`, `compute_loss` / `backward` / `step`. |
| [`train_sequential_mnist.py`](train_sequential_mnist.py) | A vanilla PyTorch training loop over sequential MNIST (28×28 read as a length-784 sequence of scalar pixels) using a tropical model and optimizer. |
| [`closure_ablation.py`](closure_ablation.py) | A recipe-boundary ablation: change soft/hard routing, projection, and optimizer choices one at a time and inspect the accuracy/closure trade-off. |
| [`backend_dispatch.py`](backend_dispatch.py) | The backend seam: logical `tt.device`, the op registry, byte-identical CPU dispatch, and transparent one-time-warned fallback. Runs CPU-only. |
| [`fpga_quickstart.py`](fpga_quickstart.py) | Safe FPGA catalog inspection, conservative structural matching, PyTorch fallback, and the checks required before registering a custom LOCAL or F2 runner. |
| [`tropical_muzero.py`](tropical_muzero.py) | A Tropical MuZero research demo that trains on GridWorld, compares with a value-iteration oracle, and exports supported dyadic weights to a shift schedule. |

## Run

```bash
# minimal
python examples/quickstart.py

# sequential MNIST — synthetic smoke (no torchvision download)
python examples/train_sequential_mnist.py --smoke

# sequential MNIST — real data (first install: pip install -e ".[vision]")
python examples/train_sequential_mnist.py --epochs 5 --batch-size 128

# closure-penalty ablation
python examples/closure_ablation.py

# backend seam (logical devices, dispatch, transparent fallback) — CPU-only
python examples/backend_dispatch.py

# FPGA quickstart — catalog/match inspection and PyTorch fallback
python examples/fpga_quickstart.py

# Tropical MuZero — planner demo + dyadic shift-schedule export
python examples/tropical_muzero.py
```

The common point is ergonomic: tropical layers and TikiTorch optimizers fit an
ordinary PyTorch-style loop. Whether a particular forward, loss, backward, and
update is closed must be checked separately with `tikitorch.audit`.
