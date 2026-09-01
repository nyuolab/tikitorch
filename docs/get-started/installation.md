# Installation

TikiTorch requires Python 3.9 or newer. PyTorch and NumPy are its only runtime Python
dependencies. A GPU or FPGA is not required for the normal CPU path.

!!! note "Source install"

    TikiTorch is currently research software and does not yet document a tagged PyPI
    release. Install the active repository branch so the package, documentation, and
    examples stay on the same revision.

## Editable development install

```bash
git clone https://github.com/nyuolab/tikitorch.git
cd tikitorch
python -m pip install -e ".[dev]"
```

Verify the checkout:

```bash
python -c "import tikitorch as tt; print(tt.__version__)"
pytest -q
```

The default test command runs the fast library checks. Stochastic end-to-end training
tests are marked `slow` and can be run separately with `pytest -m slow`.

## Documentation install

Install the documentation dependencies and build the site locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

`mkdocs serve` prints the local URL and rebuilds pages as they change.

## Optional vision examples

The sequential-MNIST example has a dependency-free synthetic `--smoke` path.
Install the vision extra before downloading and training on real MNIST:

```bash
python -m pip install -e ".[vision]"
python examples/train_sequential_mnist.py --epochs 5 --batch-size 128
```

## FPGA support

The Python FPGA namespace is included with the base package. Native drivers, vendor
tools, board images, and archived build products are intentionally managed outside
`pip`. Start with the repository's
[FPGA build guide](https://github.com/nyuolab/tikitorch/blob/main/tikitorch/hardware/BUILD_GUIDE.md)
and the
[external-artifact manifest](https://github.com/nyuolab/tikitorch/blob/main/results/fpga_silicon/s1_afibuild/ARTIFACTS.md).

Without compatible hardware, the library remains usable on CPU/GPU. Whether an FPGA
entry point falls back, raises, or binds a native implementation is API-specific; see
the [`tikitorch.fpga` reference](../api/fpga.md) rather than assuming that a CPU run
validates a board build.

## Next step

Continue to the [quickstart](quickstart.md), or read
[Migrating from PyTorch](../guide/migration.md) if you already have a training loop.
