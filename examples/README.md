# TikiTorch examples

Runnable, PyTorch-idiomatic usage demos. Each script is self-contained and runs
without downloads (synthetic data; `train_sequential_mnist.py` adds a `--smoke`
flag for its no-download path). Examples demonstrate APIs; only an explicit
audit of the complete executed recipe supports a closure claim.

| script | what it shows |
| --- | --- |
| [`closure_workflow.py`](closure_workflow.py) | Hard route, executed-path audit, bit-packed certificate, and selected-VJP replay. |
| [`quickstart.py`](quickstart.py) | The whole loop in ~10 lines: `MorphSSM` + `PureTropicalSGD`, `compute_loss` / `backward` / `step`. |
| [`train_sequential_mnist.py`](train_sequential_mnist.py) | A vanilla PyTorch training loop over sequential MNIST (28×28 read as a length-784 sequence of scalar pixels) using a tropical model and optimizer. |
| [`closure_ablation.py`](closure_ablation.py) | A recipe-boundary ablation: change soft/hard routing, projection, and optimizer choices one at a time and inspect the accuracy/closure trade-off. |
| [`backend_dispatch.py`](backend_dispatch.py) | The backend seam: logical `tt.device`, the op registry, byte-identical CPU dispatch, and transparent one-time-warned fallback. Runs CPU-only. |
| [`fpga_quickstart.py`](fpga_quickstart.py) | Safe FPGA catalog inspection, conservative structural matching, PyTorch fallback, and the checks required before registering a custom LOCAL or F2 runner. |
| [`tropical_muzero.py`](tropical_muzero.py) | A Tropical MuZero research demo that trains on GridWorld, compares with a value-iteration oracle, and exports supported dyadic weights to a shift schedule. |
| [`closure_ce.py`](closure_ce.py) | Compare four loss-conditioning boundaries on built-in text or a supplied corpus. |
| [`topk_softening.py`](topk_softening.py) | Measure the mechanism and outcome of top-k value-pull softening. |
| [`lion_vs_signsgd.py`](lion_vs_signsgd.py) | Compare sign, integer-momentum, and ordinary AdamW optimizer boundaries. |

## Run

```bash
# hard route -> audit -> packed certificate -> replay
python -m examples.closure_workflow

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

# self-contained character-LM ablations (tiny deterministic smoke settings)
python examples/closure_ce.py --smoke
python examples/topk_softening.py --smoke
python examples/lion_vs_signsgd.py --smoke
```

The character-LM ablations use a small deterministic built-in corpus when `--text`
is omitted. Pass `--text path/to/corpus.txt` for a real experiment; a missing or
too-short supplied corpus fails clearly instead of silently switching data.

The common point is ergonomic: tropical layers and TikiTorch optimizers fit an
ordinary PyTorch-style loop. `closure_workflow.py` demonstrates one explicitly scoped
operator audit and route certificate. Whether a wider forward, loss, backward, and
update is closed must be checked separately with `tikitorch.audit`.
