# TikiTorch notebooks

Runnable, heavily-documented Colab notebooks introducing TikiTorch — the
tropical-semiring, multiplier-free, FPGA-native deep-learning library.

| notebook | what it covers |
|---|---|
| [`01_intro.ipynb`](01_intro.ipynb) | 10-minute tour: the PyTorch-shaped API, a 10-line training step, `assert_closed` (the zero-multiply proof), the silicon results. |
| [`02_tropical_math.ipynb`](02_tropical_math.ipynb) | The min-plus/max-plus semiring by hand: tropical linear = morphological erosion/dilation, sparse argmax gradients, the soft→hard (Maslov) curriculum, the closure penalty. |
| [`03_vision_cnn.ipynb`](03_vision_cnn.ipynb) | A fully multiplier-free CNN on MNIST: `TropicalConv2d` as morphological dilation/erosion, pooling, `TropicalLayerNorm`, trained with a sign-only optimizer; closed-vs-boundary head comparison. |
| [`04_sequence.ipynb`](04_sequence.ipynb) | Multiplier-free sequence modeling: a tropical-transformer char-LM, the closure-faithful cross-entropy loss recovering accuracy without a multiply, `TropicalLion`, and streaming `MorphSSM.step()`. |

## Running

Open any notebook in Google Colab. The first **Setup** cell installs TikiTorch —
edit `REPO_URL` to point at your TikiTorch checkout (the repo containing
`tikitorch/` and `data/`). Locally, with TikiTorch installed (`pip install -e .`),
the Setup cell is a no-op and the rest runs as-is.

## Regenerating

The notebooks are generated from [`_build_notebooks.py`](_build_notebooks.py)
(keeps them diffable). After editing it:

```bash
python notebooks/_build_notebooks.py
```
