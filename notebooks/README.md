# TikiTorch notebooks

Runnable, heavily documented notebooks introducing TikiTorch's tropical and
hard-routed APIs, recipe-scoped closure tools, and research hardware paths.

| notebook | what it covers |
|---|---|
| [`01_intro.ipynb`](01_intro.ipynb) | 10-minute tour: the PyTorch-shaped API, a training step, structural closure auditing, and scoped silicon evidence. |
| [`02_tropical_math.ipynb`](02_tropical_math.ipynb) | The min-plus/max-plus semiring by hand: morphological erosion/dilation, sparse route gradients, and the soft→hard curriculum. |
| [`03_vision_cnn.ipynb`](03_vision_cnn.ipynb) | A morphological CNN on MNIST with a declared-head comparison and executed-path audit. |
| [`04_sequence.ipynb`](04_sequence.ipynb) | Tropical sequence modeling, alternative loss boundaries, sign-oriented optimizers, and streaming `MorphSSM.step()`. |
| [`05_tropical_geometry.ipynb`](05_tropical_geometry.ipynb) | Sampled linear and decision regions plus Newton-polytope diagnostics. |
| [`06_beyond_deep_learning.ipynb`](06_beyond_deep_learning.ipynb) | Min-plus dynamic programming and experimental classical tropical utilities. |

## Running

Open any notebook in Google Colab. The first **Setup** cell installs TikiTorch —
edit `REPO_URL` to point at your TikiTorch checkout (the repository containing
`tikitorch/`). Locally, with TikiTorch installed (`pip install -e .`),
the Setup cell is a no-op and the rest runs as-is. Notebook 03 has an additional
setup cell that installs `torchvision` if the optional vision dependency is absent.

Notebook output is pedagogical, not a claim-bearing benchmark. Use the claim
ledger and append-only result artifacts for research numbers.

## Regenerating

The notebooks are generated from [`_build_notebooks.py`](_build_notebooks.py)
(keeps them diffable). After editing it:

```bash
python notebooks/_build_notebooks.py
```
