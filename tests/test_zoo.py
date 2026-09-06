"""Usability checks for every checkpoint tracked in the public model zoo."""
import argparse
import json
from pathlib import Path
import runpy

import pytest
import torch

import tikitorch as tt
from tikitorch.algorithmic import TropicalReasoner, load_reasoner


_ROOT = Path(__file__).resolve().parents[1]
_ZOO = _ROOT / "zoo"


def _state_matches_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = model.state_dict()
    assert state.keys() == checkpoint["state_dict"].keys()
    assert all(torch.equal(state[name], value)
               for name, value in checkpoint["state_dict"].items())


@pytest.mark.parametrize(
    "name",
    ["running_max", "sort", "subset_sum", "knapsack", "min_coin_change"],
)
def test_shipped_algorithmic_checkpoint_uses_documented_loader(name):
    path = _ZOO / "algorithmic" / f"{name}.tt"
    model = load_reasoner(path)
    assert isinstance(model, TropicalReasoner)
    assert model.task.name == name
    _state_matches_checkpoint(model, path)


@pytest.mark.parametrize("task", ["smnist", "scifar"])
def test_shipped_image_checkpoint_uses_documented_instance_loader(task):
    root = _ZOO / "image_seq"
    metrics = json.loads((root / f"{task}_metrics.json").read_text())
    model = tt.models.MorphSSM(
        d_model=metrics["d_model"],
        n_layers=metrics["n_layers"],
        dropout=0.0,
        tau=0.0,
        use_soft=False,
        out_proj_type="tropical",
        norm_type="tropical",
        head_type="tropical",
        input_type="tropical",
        task_type="classification",
        input_dim=metrics["input_dim"],
        num_classes=metrics["num_classes"],
    )
    path = root / f"{task}_morphssm.tt"
    assert tt.load(path, model=model) is model
    _state_matches_checkpoint(model, path)

    model.eval()
    x = torch.zeros(1, 8, metrics["input_dim"])
    with torch.no_grad():
        assert model(x).shape == (1, metrics["num_classes"])


def test_future_image_checkpoint_config_reconstructs_cleanly(tmp_path):
    runner = runpy.run_path(str(_ROOT / "scripts" / "run_closed_benchmarks.py"))
    args = argparse.Namespace(d_model=16, n_layers=1)
    config = runner["_checkpoint_config"](
        args, {"input_dim": 1, "num_classes": 3}
    )
    assert not ({"task", "model", "optimizer", "recipe", "best_acc"} & config.keys())

    model = tt.models.MorphSSM(**config)
    path = tmp_path / "future-image.tt"
    tt.save(model, path, config=config)
    restored = tt.load(path)
    assert isinstance(restored, tt.models.MorphSSM)
    _state_matches_checkpoint(restored, path)
