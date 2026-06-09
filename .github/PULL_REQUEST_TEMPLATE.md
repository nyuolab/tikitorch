<!-- Thanks for contributing to TikiTorch! -->

## Summary

<!-- What changes, and why. -->

## Checklist
- [ ] `ruff check .` is clean
- [ ] `pytest -q` is green (and `pytest -m slow` if you touched the algorithmic / training paths)
- [ ] New ops / layers / optimizers preserve the multiplier-free closure and ship a
      closure test (`assert_closed` / `find_runtime_multiplies`) — see [CONTRIBUTING](../CONTRIBUTING.md)
- [ ] Docstrings updated; user-facing changes noted in [CHANGELOG.md](../CHANGELOG.md)
