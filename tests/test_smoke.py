"""Top-level smoke tests: package imports cleanly and the idiomatic path works.

These guard the public surface: ``import
tikitorch`` always succeeds, the submodules resolve, and the documented
``model -> compute_loss -> backward -> step`` loop runs end-to-end with the
required public ``optim`` and ``models`` modules.
"""
import importlib

import torch

from conftest import requires_model, requires_optim


def test_import_tikitorch():
    import tikitorch as tt
    assert tt.__version__
    assert tt.functional is not None
    assert tt.nn is not None
    # HardwareCostProfile is a re-exported dataclass.
    prof = tt.HardwareCostProfile()
    assert prof.multiplications == 0


def test_no_forge_imports_in_package():
    # Decoupling invariant: nothing under tikitorch may import forge.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "tikitorch"
    offenders = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import forge") or stripped.startswith("from forge"):
                offenders.append(f"{py}: {stripped}")
    assert not offenders, "forge imports leaked into tikitorch:\n" + "\n".join(offenders)


def test_submodules_importable():
    for name in ("tikitorch.functional", "tikitorch.nn",
                 "tikitorch.optim", "tikitorch.models", "tikitorch.hardware",
                 "tikitorch.backend", "tikitorch.fpga"):
        assert importlib.import_module(name) is not None


def test_backend_seam_present_and_additive():
    # The pluggable seam is on the public surface...
    import tikitorch as tt
    assert tt.backend is not None
    assert callable(tt.device) and callable(tt.to)
    assert tt.fpga is not None
    # ...and additive: CPU dispatch returns the SAME functional callables, so the
    # frozen CPU path is byte-identical (no wrapping, no overhead).
    import tikitorch.functional as F
    assert tt.backend.dispatch("tropical_linear", "cpu") is F.tropical_linear
    # 'fpga' registers but is unavailable without a board (clean CPU fallback).
    assert "fpga" in tt.backend.list_backends()
    assert tt.fpga.is_available() is False


@requires_model("MorphSSM")
@requires_optim("PureTropicalSGD")
def test_idiomatic_end_to_end_path():
    # The exact usage pattern from the README.
    import tikitorch as tt
    import tikitorch.optim as optim

    model = tt.models.MorphSSM(d_model=32, n_layers=2, task_type="lm", vocab_size=64)
    opt = optim.PureTropicalSGD(model.parameters())
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    loss = model.compute_loss(x, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


@requires_model("MorphSSM")
def test_models_dunder_all_resolves():
    import tikitorch.models as models
    if getattr(models, "__all__", None):
        for name in models.__all__:
            assert hasattr(models, name), f"models.__all__ lists missing symbol {name}"
