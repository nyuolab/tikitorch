# Tutorials (notebooks)

Six self-contained Jupyter notebooks, runnable in Google Colab (the first cell installs
TikiTorch). They are generated from
[`notebooks/_build_notebooks.py`](https://github.com/nyuolab/tikitorch/blob/main/notebooks/_build_notebooks.py)
— edit the generator, not the `.ipynb`, then regenerate with
`python notebooks/_build_notebooks.py`.

| # | Notebook | What you learn |
|---|---|---|
| 01 | [Intro](https://github.com/nyuolab/tikitorch/blob/main/notebooks/01_intro.ipynb) | A 10-minute tour: a complete multiplier-free training step, and `assert_closed`. |
| 02 | [The tropical semiring, by hand](https://github.com/nyuolab/tikitorch/blob/main/notebooks/02_tropical_math.ipynb) | The min/max-plus algebra, erosion/dilation, the soft→hard curriculum, the closure penalty — measured. |
| 03 | [A multiplier-free CNN](https://github.com/nyuolab/tikitorch/blob/main/notebooks/03_vision_cnn.ipynb) | `TropicalConv2d` as morphology; train MNIST with a sign-only optimizer; closed vs declared-head. |
| 04 | [Sequence modeling](https://github.com/nyuolab/tikitorch/blob/main/notebooks/04_sequence.ipynb) | A char-LM; recovering accuracy with the closure-faithful loss; `TropicalLion`; streaming `MorphSSM.step()`. |
| 05 | [Tropical geometry](https://github.com/nyuolab/tikitorch/blob/main/notebooks/05_tropical_geometry.ipynb) | Nets as tropical rational maps: linear regions, decision regions, Newton polytopes. |
| 06 | [Beyond deep learning](https://github.com/nyuolab/tikitorch/blob/main/notebooks/06_beyond_deep_learning.ipynb) | The same min-plus op as shortest-path DP, plus tropical k-means / SVM / PCA. |

## Reading order

- **New to the algebra?** 02 → 01 → pick a domain (03 vision, 04 sequence).
- **Here for the math/theory?** 02 → 05 (geometry) → 06 (beyond DL).
- **Here to ship on hardware?** 01 → 03/04 → the [FPGA build guide](https://github.com/nyuolab/tikitorch/blob/main/tikitorch/hardware/BUILD_GUIDE.md).

Each notebook pairs with a concept page in this guide:
[Foundations](foundations.md) (02), [Why tropical for ML/DL](why-tropical-ml.md),
[Tropical geometry](tropical-geometry.md) (05), [Beyond deep learning](beyond-deep-learning.md) (06).
