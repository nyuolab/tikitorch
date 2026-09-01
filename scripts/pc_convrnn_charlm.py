"""Shim (repo-split 2026-07-29): the engine now lives at
tikitorch.research.pc_convrnn; this file preserves every existing flat
import AND `python scripts/pc_convrnn_charlm.py ...` CLI invocation by
aliasing this module name to the promoted module object."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root,
# for environments without an installed tikitorch (boxes use PYTHONPATH)
import tikitorch.research.pc_convrnn as _m
sys.modules[__name__] = _m
if __name__ == "__main__":
    _m.main()
