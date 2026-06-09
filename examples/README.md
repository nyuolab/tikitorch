# TikiTorch examples

Runnable, PyTorch-idiomatic usage demos. Each script is self-contained and runs
without downloads (synthetic data; `train_sequential_mnist.py` adds a `--smoke`
flag for its no-download path).

| script | what it shows |
| --- | --- |
| [`quickstart.py`](quickstart.py) | The whole loop in ~10 lines: `MorphSSM` + `PureTropicalSGD`, `compute_loss` / `backward` / `step`. |
| [`train_sequential_mnist.py`](train_sequential_mnist.py) | A vanilla PyTorch training loop over sequential MNIST (28×28 read as a length-784 sequence of scalar pixels) with a multiplier-free tropical model + optimizer. |
| [`closure_ablation.py`](closure_ablation.py) | The "closure penalty": flip soft→hard, dense→tropical projection, AdamW→`PureTropicalSGD` one at a time and watch accuracy trade off for an FPGA-native (multiplier-free) learning loop. |
| [`backend_dispatch.py`](backend_dispatch.py) | The backend seam: logical `tt.device`, the op registry, byte-identical CPU dispatch, and transparent one-time-warned fallback. Runs CPU-only. |
| [`fpga_quickstart.py`](fpga_quickstart.py) | One line — `model = tt.fpga.accelerate(model)` — to target the FPGA, training loop unchanged. Binds to a silicon kernel on an F2 board; CPU fallback everywhere else. |
| [`tropical_muzero.py`](tropical_muzero.py) | The signature planner: a multiplier-free Tropical MuZero (dyadic zero-DSP cores) trains on GridWorld via self-play MCTS, scores vs a value-iteration oracle, then exports its weights to a bit-exact FPGA shift schedule. |

## Run

```bash
# minimal
python examples/quickstart.py

# sequential MNIST — synthetic smoke (no torchvision download)
python examples/train_sequential_mnist.py --smoke

# sequential MNIST — real data
python examples/train_sequential_mnist.py --epochs 5 --batch-size 128

# closure-penalty ablation
python examples/closure_ablation.py

# backend seam (logical devices, dispatch, transparent fallback) — CPU-only
python examples/backend_dispatch.py

# FPGA quickstart — one-line accelerate(); CPU fallback with no board
python examples/fpga_quickstart.py

# Tropical MuZero — multiplier-free planner on GridWorld + zero-DSP shift-schedule export
python examples/tropical_muzero.py
```

The point of every example: **tropical (multiplier-free) layers and optimizers
drop into ordinary PyTorch code with no change to how you write the loop.**
`model.compute_loss(x, y).backward(); opt.step()` is all of it.
