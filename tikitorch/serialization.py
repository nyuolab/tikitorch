"""Checkpoint helpers with constructor metadata and a structural audit manifest.

A plain ``torch.save(model.state_dict(), path)`` throws away two things that
matter for TikiTorch: (1) *what model the weights belong to* (the class and the
constructor kwargs needed to rebuild it), and (2) the result of TikiTorch's
registered-module closure scan. :func:`save` / :func:`load` keep both.

The manifest is not a proof of graph identity or physical zero-DSP execution. It
records the output of :func:`tikitorch.audit.find_closure_violations`, a structural
deny-list scan, and detects changes in that classification after reconstruction.

A checkpoint is a single ``torch.save`` dict with four keys:

    ``"class"``     — ``{"module": ..., "qualname": ...}``, the import path of the
                      model class so :func:`load` can reconstruct it.
    ``"state_dict"``— the usual ``model.state_dict()``.
    ``"config"``    — the constructor kwargs (pass them explicitly to :func:`save`;
                      since the models take keyword-only args, the saved config is
                      exactly what ``Model(**config)`` needs). If omitted, a generic
                      snapshot of the model's simple scalar attributes is stored
                      instead (best-effort; explicit config is preferred).
    ``"closure_manifest"`` — :func:`tikitorch.audit.find_closure_violations`
                      output: the list of registered multiply-bearing modules.
                      On :func:`load` this is recomputed and compared; a mismatch
                      emits :class:`ClosureManifestMismatch`.

Idiomatic usage::

    import tikitorch as tt

    model = tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=2,
                                          task_type="classification", num_classes=10)
    tt.save(model, "ckpt.pt", config=dict(d_model=64, n_heads=4, n_layers=2,
                                          task_type="classification", num_classes=10))

    # Reconstruct from the checkpoint alone (class + config are stored):
    reloaded = tt.load("ckpt.pt")

    # Or the common torch pattern — load weights into an existing instance:
    m = tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=2,
                                      task_type="classification", num_classes=10)
    tt.load("ckpt.pt", model=m)
"""
import importlib
import warnings

import torch
import torch.nn as nn

from tikitorch.audit import find_closure_violations

__all__ = ["save", "load", "ClosureManifestMismatch"]

# Bump if the on-disk layout ever changes (kept in the dict for forward-compat).
_FORMAT_VERSION = 1


class ClosureManifestMismatch(UserWarning):
    """Warned when a reloaded model's closure manifest differs from the saved one.

    A registered multiply-bearing module appeared or disappeared, usually because
    of class/config drift or a manually edited checkpoint. The state dict is still
    loaded. Matching manifests establish only matching structural-audit output;
    they do not establish graph identity or physical closure.
    """


def _snapshot_simple_attrs(model: nn.Module) -> dict:
    """Best-effort generic config: the model's plain scalar/string/bool attributes.

    Fallback used only when an explicit ``config`` is not passed to :func:`save`.
    Captures the simple, JSON-ish public attributes the TikiTorch models store on
    ``self`` (e.g. ``d_model``, ``num_classes``, ``task_type``) while skipping
    private names, submodules/parameters/tensors, and anything non-scalar. This is
    a *snapshot* for inspection, not a guaranteed-complete constructor kwargs set —
    prefer passing ``config`` explicitly.
    """
    snap = {}
    for name, value in vars(model).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (int, float, bool, str, type(None))):
            snap[name] = value
        elif isinstance(value, (tuple, list)) and all(
            isinstance(v, (int, float, bool, str)) for v in value
        ):
            snap[name] = list(value)
    return snap


def save(model: nn.Module, path, config: dict = None) -> dict:
    """Save ``model`` to ``path`` with its config and a closure manifest.

    Args:
        model: the (tropical) model to serialize.
        path (Any): destination passed to ``torch.save`` (filename or file-like
            object).
        config: the constructor kwargs, such that ``Model(**config)`` rebuilds an
            architecturally-identical model. The TikiTorch models take keyword-only
            args, so this is exactly that kwargs dict. If ``None``, a generic
            snapshot of the model's simple scalar attributes is stored instead (see
            :func:`_snapshot_simple_attrs`) — pass ``config`` explicitly for a
            checkpoint that :func:`load` can reconstruct from scratch.

    Returns the checkpoint dict that was written (handy for inspection/testing).
    """
    cls = type(model)
    checkpoint = {
        "format_version": _FORMAT_VERSION,
        "class": {"module": cls.__module__, "qualname": cls.__qualname__},
        "state_dict": model.state_dict(),
        "config": dict(config) if config is not None else _snapshot_simple_attrs(model),
        # Provenance flag: was the config the user's explicit kwargs, or a snapshot?
        "config_is_explicit": config is not None,
        # Structural audit snapshot; an empty list is necessary, not sufficient,
        # for a recipe-level or physical closure claim.
        "closure_manifest": find_closure_violations(model),
    }
    torch.save(checkpoint, path)
    return checkpoint


def _resolve_class(module_name: str, qualname: str):
    """Import ``module_name`` and walk ``qualname`` (dotted) to the class object."""
    mod = importlib.import_module(module_name)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def load(path, model: nn.Module = None, config: dict = None, map_location="cpu"):
    """Load a checkpoint written by :func:`save`.

    Two modes, mirroring the common torch patterns:

      * **Into an existing instance** — pass ``model``. Its ``load_state_dict`` is
        called (you built the architecture yourself; this just restores weights).
      * **Reconstruct from scratch** — omit ``model``. The class is re-imported from
        the saved ``(module, qualname)`` and instantiated with the saved ``config``
        (overridable via the ``config`` argument), then the state_dict is loaded.

    In both modes the structural closure manifest is recomputed on the loaded
    model and compared to the one stored at save time; a mismatch emits a
    :class:`ClosureManifestMismatch` warning.

    Args:
        path (Any): source passed to ``torch.load``.
        model: an existing model instance to load weights into (optional).
        config: constructor-kwargs override used only in the reconstruct path; merged
            over the saved config (saved values are the base, ``config`` wins).
        map_location (Any): forwarded to ``torch.load`` (default ``"cpu"``).

    Returns the loaded model.
    """
    # weights_only=False: the checkpoint is a plain dict of metadata + a state_dict
    # (no arbitrary code), but it contains non-tensor python objects (the class
    # path / config / manifest) that the new default tensor-only unpickler rejects.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{path!r} is not a TikiTorch checkpoint (no 'state_dict' key). "
            "Was it written by tikitorch.serialization.save?"
        )

    state_dict = checkpoint["state_dict"]
    saved_config = checkpoint.get("config", {}) or {}

    if model is None:
        # Reconstruct path: re-import the class and instantiate it from the config.
        cls_info = checkpoint.get("class")
        if not cls_info:
            raise ValueError(
                f"{path!r} has no 'class' info; cannot reconstruct the model. "
                "Pass an existing `model=` instance to load weights into instead."
            )
        cls = _resolve_class(cls_info["module"], cls_info["qualname"])
        merged_config = dict(saved_config)
        if config is not None:
            merged_config.update(config)
        model = cls(**merged_config)

    model.load_state_dict(state_dict)

    # Compare the registered-module audit classification across serialization.
    saved_manifest = checkpoint.get("closure_manifest")
    if saved_manifest is not None:
        # Normalize to lists of tuples — torch.save round-trips tuples as tuples,
        # but be tolerant of list-of-list too.
        def _norm(man):
            return sorted((str(p), str(t)) for p, t in man)

        current_manifest = find_closure_violations(model)
        if _norm(current_manifest) != _norm(saved_manifest):
            warnings.warn(
                "Reloaded model's closure manifest does not match the saved one — "
                "the reconstructed model has a different registered-module "
                f"closure classification.\n  saved:   {sorted(saved_manifest)}\n"
                f"  current: {sorted(current_manifest)}\n"
                "Check the class/config used to reconstruct it. A matching "
                "manifest is a structural consistency check, not physical proof.",
                ClosureManifestMismatch,
                stacklevel=2,
            )

    return model
