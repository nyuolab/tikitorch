# TikiTorch test suite

```bash
pytest                 # from the package root
pytest -v
pytest tests/test_functional.py
```

## Layout

| file | covers |
| --- | --- |
| `test_functional.py` | `tropical_matmul` correctness, hard autograd ops, soft→hard limits, numeric `gradcheck` on the soft relaxations, attention primitives. **Phase-1, always active.** |
| `test_nn.py` | Phase-1 core layers (active) + Phase-2 building blocks (skip-guarded). |
| `test_optim.py` | `SignSGD` / `TropicalSGD` / `PureTropicalSGD`: one-step + toy-loss descent (skip-guarded). |
| `test_models.py` | forward/backward/`compute_loss`/`param_count` for every model and every `task_type` (skip-guarded). |
| `test_smoke.py` | `import tikitorch`, submodule resolution, the "no forge imports" decoupling invariant, the idiomatic end-to-end path. |

## Phasing

Phase 1 (`functional` + core `nn`) is frozen and tested unconditionally. Tests
that depend on Phase-2 symbols (`optim`, `models`, higher-level `nn` blocks) are
guarded by the `requires_optim` / `requires_model` / `requires_nn` helpers in
`conftest.py`: they **skip** cleanly until the symbol exists, then **activate
automatically** the moment Phase 2 lands. Running `pytest` today is green with
the Phase-2 tests reported as skipped; running it after Phase 2 exercises the
full surface (this is the Phase-3 verification described in `API_SPEC.md` §6).
