"""Export-hygiene guards for the public namespaces.

These tests pin the ``__all__`` contract so the namespace stays clean and
self-consistent as new layers/ops land:
  * every name in ``__all__`` actually resolves on the module (no typos / removals);
  * ``__all__`` has no duplicates;
  * every ``nn.Module`` subclass *defined in* ``tikitorch.nn`` is exported (so a new
    layer can't silently miss ``__all__`` and vanish from ``from tikitorch.nn import *``
    and the docs).
"""
import inspect

import torch.nn as torch_nn

import tikitorch.nn as nn
import tikitorch.functional as functional


def test_nn_all_names_resolve():
    missing = [name for name in nn.__all__ if not hasattr(nn, name)]
    assert not missing, f"nn.__all__ lists names not on the module: {missing}"


def test_nn_all_has_no_duplicates():
    dups = sorted({n for n in nn.__all__ if nn.__all__.count(n) > 1})
    assert not dups, f"duplicate entries in nn.__all__: {dups}"


def test_nn_module_subclasses_are_exported():
    """Any nn.Module subclass defined in tikitorch.nn (not imported) must be public."""
    exported = set(nn.__all__)
    unexported = []
    for name, obj in vars(nn).items():
        if name.startswith("_"):
            continue
        if not (inspect.isclass(obj) and issubclass(obj, torch_nn.Module)):
            continue
        # only classes defined in this package (skip re-imported torch.nn classes)
        if not getattr(obj, "__module__", "").startswith("tikitorch."):
            continue
        if name not in exported:
            unexported.append(name)
    assert not unexported, f"nn.Module subclasses missing from nn.__all__: {unexported}"


def test_loss_modules_are_exported():
    for name in ("TropicalCrossEntropyLoss", "TropicalPerceptronLoss"):
        assert name in nn.__all__ and hasattr(nn, name)


def test_functional_all_names_resolve():
    # functional declares __all__ too; keep it honest if present.
    names = getattr(functional, "__all__", None)
    if names is None:
        return
    missing = [n for n in names if not hasattr(functional, n)]
    assert not missing, f"functional.__all__ lists names not on the module: {missing}"
