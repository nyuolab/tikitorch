"""tikitorch.research — the shared PC research engine, promoted from
scripts/ (repo-split 2026-07-29) so both the text-paper drivers and the
ontogenic-agent repo consume ONE engine via the package.

scripts/pc_convrnn_charlm.py and scripts/nanogpt_shakespeare.py remain as
identity-preserving shims (same module object via sys.modules aliasing) so
every existing flat import keeps working unchanged."""

from pathlib import Path


def _default_workspace() -> Path:
    """Use the source checkout, or the caller's directory for an installed wheel.

    Research runner defaults must never put datasets or experiment output inside
    the installed package. Explicit ``--text`` and ``--out`` override these defaults.
    """
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "tikitorch").is_dir():
        return candidate
    return Path.cwd()
