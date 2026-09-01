# Tutorials (notebooks)

Six self-contained Jupyter notebooks, runnable in Google Colab (the first cell installs
TikiTorch). They are generated from
[`notebooks/_build_notebooks.py`](https://github.com/nyuolab/tikitorch/blob/main/notebooks/_build_notebooks.py)
— edit the generator, not the `.ipynb`, then regenerate with
`python notebooks/_build_notebooks.py`.

| # | Notebook | What you learn |
|---|---|---|
| 01 | [Intro](https://github.com/nyuolab/tikitorch/blob/main/notebooks/01_intro.ipynb) | A 10-minute tour of the training-loop surface and closure diagnostics. |
| 02 | [The tropical semiring, by hand](https://github.com/nyuolab/tikitorch/blob/main/notebooks/02_tropical_math.ipynb) | Min/max-plus algebra, erosion/dilation, and a soft→hard demonstration. |
| 03 | [A morphological CNN](https://github.com/nyuolab/tikitorch/blob/main/notebooks/03_vision_cnn.ipynb) | `TropicalConv2d` as morphology; a small MNIST training example and head ablation. |
| 04 | [Sequence modeling](https://github.com/nyuolab/tikitorch/blob/main/notebooks/04_sequence.ipynb) | Character modeling, loss/optimizer variants, and streaming `MorphSSM.step()`. |
| 05 | [Tropical geometry](https://github.com/nyuolab/tikitorch/blob/main/notebooks/05_tropical_geometry.ipynb) | Nets as tropical rational maps: sampled linear regions, decision regions, and Newton polytopes. |
| 06 | [Beyond deep learning](https://github.com/nyuolab/tikitorch/blob/main/notebooks/06_beyond_deep_learning.ipynb) | Min-plus dynamic programming and experimental classical tropical utilities. |

## Reading order

- **New to the algebra?** 02 → 01 → pick a domain (03 vision, 04 sequence).
- **Here for the math/theory?** 02 → 05 (geometry) → 06 (beyond DL).
- **Here to study hardware paths?** 01 → 03/04 → the [closure guide](closure.md) →
  the [FPGA build guide](https://github.com/nyuolab/tikitorch/blob/main/tikitorch/hardware/BUILD_GUIDE.md).

Each notebook pairs with a concept page in this guide:
[Foundations](foundations.md) (02), [Why tropical for ML/DL](why-tropical-ml.md),
[Tropical geometry](tropical-geometry.md) (05), [Beyond deep learning](beyond-deep-learning.md) (06).

Notebook output is educational evidence for the exact cells that ran, not a release or
hardware claim. Use the [research evidence boundary](../research/index.md) for archived
results and remaining gates.
