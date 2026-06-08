# Contributing to TikiTorch

Thanks for your interest. TikiTorch is research code with one non-negotiable
invariant — **the learning loop stays multiplier-free** — so the contribution
flow is built around protecting that.

## Dev setup

```bash
git clone https://github.com/nyuolab/tikitorch.git
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

## The closure invariant (read this before adding an op or layer)

TikiTorch's whole premise is that **every operation in the forward and backward
pass reduces to compare / add / bit-shift — no multiplies anywhere in the
learning loop.** That is what makes the loop algebraically closed and mappable
to FPGA fabric with zero DSP blocks.

If you add a new op, layer, model, or optimizer:

- **No multiplies in the closed path.** No `*` between tensors, no `@`/`matmul`,
  no `/`, no `exp`/`log`/`softmax` on the hot path. Tropical "multiply" is `+`;
  tropical "add" is `min`/`max`. Scaling, when unavoidable, must be a power-of-2
  bit-shift (`x * 2**k`), which the audit treats as closure-clean.
- **Prove it.** The repo ships an auditor — use it in your tests:

  ```python
  from tikitorch.audit import assert_closed, find_runtime_multiplies
  assert_closed(model)                       # static: inspects registered modules
  assert not find_runtime_multiplies(model, example_input)   # dynamic: traces real ATen ops
  ```

  `assert_closed` inspects modules statically; `find_runtime_multiplies` traces
  an actual forward/backward and catches a raw `*`/`@`/`exp` hidden inside a
  custom `forward` (the gap static checks miss). New closed components should
  ship a test asserting both.
- If a component genuinely needs a real multiply (e.g. a standard `nn.LayerNorm`
  or `nn.Linear` head), it must be a **declared, documented boundary** — not a
  silent one — so the audit can account for it.

## Code style

- Library code under `tikitorch/` is held to the full ruff ruleset (see
  `[tool.ruff]` in `pyproject.toml`). Keep it clean.
- Tests, examples, notebooks, and scripts relax the pure-style rules
  (compact one-liners are fine there) but keep the bug-catching `F` rules on.
- Match the surrounding code's idiom, naming, and comment density.

## PRs

- Branch from `main`, keep the change focused, and describe the *why*.
- Green CI is required. A new op/layer without a closure test will be asked for
  one.

## License

By contributing you agree your contributions are licensed under the project's
[GPL-3.0-or-later](LICENSE).
