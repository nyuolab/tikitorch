"""Research CLI defaults belong to the workspace, never site-packages."""

from pathlib import Path

import tikitorch.research as research
from tikitorch.research import nanogpt, pc_convrnn


def test_checkout_defaults_resolve_repository_root():
    root = Path(__file__).resolve().parents[1]
    assert research._default_workspace() == root
    assert nanogpt.REPO == root
    assert pc_convrnn.REPO == root


def test_installed_defaults_use_callers_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "__file__",
                        str(tmp_path / "site-packages/tikitorch/research/__init__.py"))
    workspace = tmp_path / "working"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    assert research._default_workspace() == workspace
