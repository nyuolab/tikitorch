# Contributing to TikiTorch

Thanks for your interest. TikiTorch is a PyTorch-shaped research library with
two deliberately different surfaces: a general host-side API for tropical and
hard-routed experiments, and a narrower **audited closed subset** for concrete
multiplier-free recipes. Contributions must keep that boundary explicit.

## Dev setup

```bash
git clone --branch pc-morphssm --single-branch https://github.com/nyuolab/tikitorch.git
cd tikitorch
pip install -e ".[dev]"        # editable install + pytest + ruff
```

`torch` and `numpy` are runtime dependencies. The FPGA backend is pure
Python + ctypes; the native driver is built out-of-band against the Xilinx
toolchain (see [tikitorch/fpga/](tikitorch/fpga/)) and is not needed to develop
or test the library — the board-dependent tests self-skip without hardware.

## Before you open a PR

Run what CI runs — both must be green:

```bash
ruff check .          # lint
pytest -q             # fast building-block suite: closure, exactness, gradchecks
```

CI (`.github/workflows/ci.yml`) runs this on Python 3.9, 3.11, and 3.12.

The end-to-end **training** tests (a tropical model learns an algorithmic task
and generalizes) are marked `slow` and **deselected by default** — they validate
research/learning claims, not library correctness, and they're stochastic and
minutes-long. Run them on demand:

```bash
pytest -m slow        # the training tests only
```

## The closure contract (read this before adding an op or layer)

TikiTorch does not claim that every package component or configuration is
multiplier-free. Closure belongs to an executed recipe: hard model, loss,
reverse rule, optimizer, representative input path, and compiled build. A
component advertised as part of the closed subset must state which of those
boundaries it covers.

If you add a new op, layer, model, or optimizer:

- **No undeclared multiplies in an advertised closed path.** No `*` between tensors, no `@`/`matmul`,
  no `/`, no `exp`/`log`/`softmax` on the hot path. Tropical "multiply" is `+`;
  tropical "add" is `min`/`max`. Scaling, when unavoidable, must be a power-of-2
  bit-shift (`x * 2**k`), which the audit treats as closure-clean.
- **Test the declared boundary.** The repo ships structural and executed-path
  auditors; use both in tests:

  ```python
  from tikitorch.audit import assert_closed, assert_closed_runtime

  assert_closed(model)                         # declared module structure
  assert_closed_runtime(model, example_input)  # this executed forward path
  ```

  `assert_closed` cannot see raw arithmetic inside a custom `forward`, while
  runtime tracing covers only the path and inputs that actually execute. A
  full-training claim additionally needs `assert_training_closed` with the
  declared loss and optimizer. These are software checks, not substitutes for
  HLS operator/resource reports and routed hardware evidence.
- If a component genuinely needs a real multiply (e.g. a standard `nn.LayerNorm`
  or `nn.Linear` head), it must be a **declared, documented boundary** — not a
  silent one — so the audit can account for it.
- General host-side and experimental components may use ordinary PyTorch
  arithmetic. Do not label them closed, multiplier-free, or FPGA-ready unless
  the corresponding recipe-level tests and physical gates exist.

## Docs & notebooks

- The docs site is mkdocs + mkdocstrings: API pages are auto-generated from in-source
  docstrings, and the prose **Guide** lives under `docs/guide/`. Build it with
  `pip install -e ".[docs]" && mkdocs build` (what the CI docs job runs).
- The tutorial notebooks are **generated** from
  [`notebooks/_build_notebooks.py`](notebooks/_build_notebooks.py) (so they stay
  diffable). Edit the generator, not the `.ipynb`, then regenerate:

  ```bash
  python notebooks/_build_notebooks.py
  ```

## Code style

- Library code under `tikitorch/` is held to the full ruff ruleset (see
  `[tool.ruff]` in `pyproject.toml`). Keep it clean.
- Tests, examples, notebooks, and scripts relax the pure-style rules
  (compact one-liners are fine there) but keep the bug-catching `F` rules on.
- Match the surrounding code's idiom, naming, and comment density.

## PRs

- Until the 0.2 branch is reconciled with the repository default, branch from the
  maintained revision named in the installation guide. Keep the change focused
  and describe the *why*.
- Green CI is required. A component advertised as closed must include the
  appropriate structural, executed-path, and (when relevant) full-step tests.

## License

By contributing you agree your contributions are licensed under the project's
[GPL-3.0-or-later](LICENSE).
