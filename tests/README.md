# TikiTorch test suite

```bash
pytest                 # from the package root
pytest -v
pytest tests/test_functional.py
```

## Layout

| file | covers |
| --- | --- |
| `test_functional.py` | `tropical_matmul` correctness, hard autograd ops, soft→hard limits, numeric `gradcheck` on the soft relaxations, attention primitives. **Always active.** |
| `test_nn.py` | core layers (active) + optional building blocks (skip-guarded). |
| `test_optim.py` | `SignSGD` / `TropicalSGD` / `PureTropicalSGD`: one-step + toy-loss descent (skip-guarded). |
| `test_models.py` | forward/backward/`compute_loss`/`param_count` for every model and every `task_type` (skip-guarded). |
| `test_smoke.py` | `import tikitorch`, submodule resolution, the "no forge imports" decoupling invariant, the idiomatic end-to-end path. |

## Skip-guarding

The core (`functional` + core `nn`) is tested unconditionally. Tests
that depend on the optional symbols (`optim`, `models`, higher-level `nn` blocks) are
guarded by the `requires_optim` / `requires_model` / `requires_nn` helpers in
`conftest.py`: they **skip** cleanly until the symbol exists, then **activate
automatically** the moment those modules are present. Running `pytest` today is green with
the optional-module tests reported as skipped; once those modules are present it
exercises the full surface.
