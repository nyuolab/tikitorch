"""tikitorch.prune — tropical structured pruning.

Morphological / tropical layers prune unusually well: an output is the max
(dilation) or min (erosion) over taps ``x_i + W[o,i]``, so a tap whose weight is
much *smaller* than its row-mates (for a max) — or much *larger* (for a min) —
can never win and contributes nothing. Setting such taps to the semiring's
absorbing element (``-inf`` for max-plus, ``+inf`` for min-plus) removes them
from the computation with **no change to closure** (a pruned tap is simply not
compared), shrinking the per-output compare fan-in. This is the multiplier-free
analogue of magnitude pruning (cf. Dimitriadis & Maragos; *Max-plus operators
for filter selection and model pruning*, 2019).

    from tikitorch.prune import tropical_prune
    pruned, total = tropical_prune(model, sparsity=0.5)   # prune 50% of morph taps
    # model still runs and assert_closed-passes; fewer live taps per output.
"""
import torch
import torch.nn as nn

__all__ = ["tropical_prune", "tropical_sparsity"]

_NEG_INF = float("-inf")
_POS_INF = float("inf")


def _mode_of(module):
    """Return 'dilation' (max-plus), 'erosion' (min-plus), or None for a layer."""
    m = getattr(module, "mode", None)
    if m in ("dilation", "erosion"):
        return m
    cls = type(module).__name__
    if "Dilation" in cls:                 # TropicalDilationModule, ...
        return "dilation"
    if "Linear" in cls and "Tropical" in cls:   # TropicalLinearModule (min-plus)
        return "erosion"
    return None


@torch.no_grad()
def tropical_prune(model, sparsity: float = 0.5):
    """Prune ``sparsity`` of never-winning taps in every morphological layer.

    For each layer with a ``weight`` and a discernible max/min mode, the per-output
    taps least able to win are set to the absorbing element (-inf for dilation,
    +inf for erosion). Returns ``(pruned, total)`` tap counts. In-place.
    """
    assert 0.0 <= sparsity < 1.0, "sparsity in [0, 1)"
    pruned = total = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        mode = _mode_of(m)
        if w is None or mode is None or w.dim() < 2:
            continue
        flat = w.reshape(w.shape[0], -1)           # (out, taps)
        taps = flat.shape[1]
        k = int(sparsity * taps)
        total += flat.numel()
        if k <= 0:
            continue
        # dilation(max): smallest weights never win -> prune smallest (-> -inf).
        # erosion(min):  largest  weights never win -> prune largest  (-> +inf).
        descending = (mode == "erosion")
        idx = flat.argsort(dim=1, descending=descending)[:, :k]
        fill = _NEG_INF if mode == "dilation" else _POS_INF
        flat.scatter_(1, idx, fill)
        w.copy_(flat.reshape(w.shape))
        pruned += idx.numel()
    return pruned, total


@torch.no_grad()
def tropical_sparsity(model):
    """Fraction of morphological-layer taps currently pruned (set to +-inf)."""
    pruned = total = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if w is None or _mode_of(m) is None or w.dim() < 2:
            continue
        total += w.numel()
        pruned += int(torch.isinf(w).sum())
    return (pruned / total) if total else 0.0
