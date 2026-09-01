"""Shim (repo-split 2026-07-29): promoted to tikitorch.research.nanogpt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root,
# for environments without an installed tikitorch (boxes use PYTHONPATH)
import tikitorch.research.nanogpt as _m
sys.modules[__name__] = _m
if __name__ == "__main__":
    _m.main()
