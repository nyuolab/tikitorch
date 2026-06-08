"""Tests for tikitorch.serialization — save/load with a closure manifest.

Covers, per the API contract:
  * round-trip a TropicalTransformer and a TropicalCNN (params byte-identical),
  * the reconstruct-from-checkpoint path (class + config rebuild the model),
  * the model-instance path (load_state_dict into an existing instance),
  * the closure manifest is stored, recomputed, and a mismatch warns,
  * the generic-config fallback when no explicit config is given.

Skip-guarded behind ``requires_model`` so the suite stays green until the
Phase-2 models module lands, then activates automatically.
"""
import pytest
import torch

import tikitorch.models as models
from tikitorch.serialization import save, load, ClosureManifestMismatch
from conftest import requires_model


def _params_identical(a, b):
    """Every named parameter present in both, byte-for-byte equal."""
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), "state_dict keys differ"
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"param {k} differs after load"


# ---------------------------------------------------------------------------
# round-trip: reconstruct-from-checkpoint
# ---------------------------------------------------------------------------


@requires_model("TropicalTransformer")
def test_round_trip_tropical_transformer(tmp_path):
    cfg = dict(d_model=32, n_heads=2, n_layers=2,
               task_type="classification", input_dim=1, num_classes=5)
    model = models.TropicalTransformer(**cfg)
    # perturb weights so we are not comparing against a fresh-init coincidence.
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    path = tmp_path / "transformer.pt"
    save(model, path, config=cfg)

    reloaded = load(path)
    assert isinstance(reloaded, models.TropicalTransformer)
    _params_identical(model, reloaded)

    # forward agrees in eval mode (deterministic graph).
    x = torch.randn(4, 8, 1)
    model.eval(); reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)


@requires_model("TropicalCNN")
def test_round_trip_tropical_cnn(tmp_path):
    cfg = dict(in_channels=1, num_classes=10, channels=(8, 16),
               kernel_size=3, task_type="classification")
    model = models.TropicalCNN(**cfg)
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    path = tmp_path / "cnn.pt"
    save(model, path, config=cfg)

    reloaded = load(path)
    assert isinstance(reloaded, models.TropicalCNN)
    _params_identical(model, reloaded)

    x = torch.rand(2, 1, 28, 28)
    model.eval(); reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)


# ---------------------------------------------------------------------------
# the model-instance path (common torch load_state_dict pattern)
# ---------------------------------------------------------------------------


@requires_model("TropicalCNN")
def test_load_into_existing_instance(tmp_path):
    cfg = dict(in_channels=1, num_classes=10, channels=(8, 16), kernel_size=3)
    model = models.TropicalCNN(**cfg)
    for p in model.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)

    path = tmp_path / "cnn_inst.pt"
    save(model, path, config=cfg)

    # A freshly-initialized instance with a *different* random state...
    target = models.TropicalCNN(**cfg)
    for p in target.parameters():
        p.data.add_(torch.randn_like(p) * 0.5)
    # ...is not equal before load,
    sa, sb = model.state_dict(), target.state_dict()
    assert any(not torch.equal(sa[k], sb[k]) for k in sa)

    out = load(path, model=target)
    assert out is target                # loads in place, returns the same object
    _params_identical(model, target)


# ---------------------------------------------------------------------------
# closure manifest is stored + verified
# ---------------------------------------------------------------------------


@requires_model("TropicalCNN")
def test_closure_manifest_is_stored(tmp_path):
    # Fully-closed model -> empty manifest.
    model = models.TropicalCNN(in_channels=1, num_classes=10)
    ckpt = save(model, tmp_path / "closed.pt",
                config=dict(in_channels=1, num_classes=10))
    assert "closure_manifest" in ckpt
    assert ckpt["closure_manifest"] == []   # closed: no multiply-bearing modules

    # A declared DSP boundary (linear head) -> non-empty manifest that names it.
    boundary = models.TropicalCNN(in_channels=1, num_classes=10, head_type="linear")
    ckpt2 = save(boundary, tmp_path / "boundary.pt",
                 config=dict(in_channels=1, num_classes=10, head_type="linear"))
    types = [t for _, t in ckpt2["closure_manifest"]]
    assert "Linear" in types


@requires_model("TropicalCNN")
def test_closure_manifest_round_trips_without_warning(tmp_path, recwarn):
    model = models.TropicalCNN(in_channels=1, num_classes=10)
    save(model, tmp_path / "closed.pt", config=dict(in_channels=1, num_classes=10))
    load(tmp_path / "closed.pt")
    assert not [w for w in recwarn.list
                if issubclass(w.category, ClosureManifestMismatch)]


@requires_model("TropicalCNN")
def test_manifest_mismatch_warns(tmp_path):
    # Save a closed model, then reconstruct with a config that changes closure
    # (linear head) -> the recomputed manifest differs -> warn.
    model = models.TropicalCNN(in_channels=1, num_classes=10)
    path = tmp_path / "m.pt"
    save(model, path, config=dict(in_channels=1, num_classes=10))

    with pytest.warns(ClosureManifestMismatch):
        # override the config so the rebuilt model has a Linear head; state_dict
        # for the closed tropical head won't match the linear head, so load weights
        # into a hand-built instance instead to isolate the manifest check.
        boundary = models.TropicalCNN(in_channels=1, num_classes=10, head_type="linear")
        # hand the saved (closed) manifest a model with a different (open) graph:
        ckpt = torch.load(path, weights_only=False)
        ckpt["state_dict"] = boundary.state_dict()  # make load_state_dict succeed
        torch.save(ckpt, path)
        load(path, model=boundary)


# ---------------------------------------------------------------------------
# generic-config fallback (no explicit config given)
# ---------------------------------------------------------------------------


@requires_model("TropicalTransformer")
def test_generic_config_snapshot_when_none(tmp_path):
    model = models.TropicalTransformer(d_model=32, n_heads=2, n_layers=2,
                                       task_type="classification",
                                       input_dim=1, num_classes=7)
    ckpt = save(model, tmp_path / "snap.pt")          # no config -> snapshot
    assert ckpt["config_is_explicit"] is False
    snap = ckpt["config"]
    # simple scalar attrs are captured; submodules/tensors are not.
    assert snap.get("task_type") == "classification"
    assert snap.get("num_classes") == 7
    assert "blocks" not in snap and "embed" not in snap


@requires_model("TropicalTransformer")
def test_load_into_instance_ignores_missing_class(tmp_path):
    # A checkpoint saved with a snapshot config still loads into an instance.
    model = models.TropicalTransformer(d_model=16, n_heads=2, n_layers=1,
                                       task_type="classification",
                                       input_dim=1, num_classes=3)
    path = tmp_path / "snap2.pt"
    save(model, path)
    target = models.TropicalTransformer(d_model=16, n_heads=2, n_layers=1,
                                        task_type="classification",
                                        input_dim=1, num_classes=3)
    load(path, model=target)
    _params_identical(model, target)
