"""Shared, download-free character data for the language-model examples."""

from pathlib import Path

import torch


_SYNTHETIC_CORPUS = """
tropical routes select a value and preserve the choice for replay.
max plus replaces products with additions and sums with comparisons.
the model the loss the reverse rule and the update form one audited recipe.
hard routes are discrete while soft routes are useful training relaxations.
measure the boundary report the configuration and keep the evidence reproducible.
""".strip()


def load_ascii_ids(path=None, *, vocab_size=128, min_length=0):
    """Return ASCII token IDs and a human-readable source label.

    With no ``path``, repeat a small deterministic, repository-owned corpus so
    examples work in a fresh clone without downloads. A user-supplied file is
    never silently replaced: missing or too-short inputs raise a clear error.
    """
    if path is None:
        repeats = max(1, (min_length + len(_SYNTHETIC_CORPUS)) // len(_SYNTHETIC_CORPUS))
        text = ((_SYNTHETIC_CORPUS + "\n") * repeats).strip()
        source = "built-in deterministic corpus"
    else:
        source_path = Path(path).expanduser()
        text = source_path.read_text(encoding="utf-8", errors="replace")
        source = str(source_path)

    if len(text) <= min_length:
        raise ValueError(
            f"character corpus must contain more than {min_length} characters; "
            f"{source!r} contains {len(text)}"
        )
    ids = torch.tensor([min(ord(char), vocab_size - 1) for char in text], dtype=torch.long)
    return ids, source
