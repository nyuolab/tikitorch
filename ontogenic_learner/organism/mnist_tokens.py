"""MNIST as pixel-token sequences (the third modality). Spec (plan):
28->24 center crop, 2x2 avg pool -> 12x12, 16 gray levels -> PIX symbols; one image =
optional text label prefix ("digit 7:", deterministic 50% by index parity) + SOI +
12x(12 pixel symbols + '\\n') + EOI + '\\n'  (~167 tokens).

Parses the raw idx files directly (no torchvision dependency on the box)."""
import gzip
import os
import struct

import numpy as np
import torch

from .vocab import Vocab, PIX, SOI, EOI


def _read_idx(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist(repo: str):
    raw = os.path.join(repo, "data/MNIST/raw")
    def pick(stem):
        for suf in ("", ".gz"):
            p = os.path.join(raw, stem + suf)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(stem)
    Xtr = _read_idx(pick("train-images-idx3-ubyte"))
    Ytr = _read_idx(pick("train-labels-idx1-ubyte"))
    Xte = _read_idx(pick("t10k-images-idx3-ubyte"))
    Yte = _read_idx(pick("t10k-labels-idx1-ubyte"))
    return Xtr, Ytr, Xte, Yte


def image_to_grid(img28: np.ndarray) -> np.ndarray:
    """28x28 uint8 -> 12x12 int levels in [0,15]."""
    x = img28[2:26, 2:26].astype(np.float32) / 255.0          # crop 24x24
    x = x.reshape(12, 2, 12, 2).mean(axis=(1, 3))             # 2x2 avg pool -> 12x12
    return np.minimum((x * 16).astype(np.int64), 15)


def image_to_tokens(vocab: Vocab, img28, label: int, with_label: bool) -> list:
    grid = image_to_grid(img28)
    s = f"digit {label}:" if with_label else ""
    ids = vocab.encode_text(s)
    ids.append(vocab.soi)
    nl = vocab.stoi["\n"]
    for row in grid:
        ids.extend(int(vocab.pix0 + v) for v in row)
        ids.append(nl)
    ids.append(vocab.eoi)
    ids.append(nl)
    return ids


def epoch_stream(vocab: Vocab, X, Y, gen: torch.Generator) -> torch.Tensor:
    """One shuffled epoch of the train set as a single token-id tensor (~10M ids).
    Deterministic given the generator state (which is checkpointed)."""
    order = torch.randperm(len(X), generator=gen).tolist()
    out = []
    for i in order:
        out.extend(image_to_tokens(vocab, X[i], int(Y[i]), with_label=(i % 2 == 0)))
    return torch.tensor(out, dtype=torch.long)


def grid_to_ascii(grid: np.ndarray) -> str:
    from .vocab import ASCII_RAMP
    return "\n".join("".join(ASCII_RAMP[v] for v in row) for row in grid)


def tokens_to_grid(vocab: Vocab, ids) -> np.ndarray:
    """Extract the first complete 12x12 image following the first SOI in ids (probe use).
    Missing/foreign tokens render as level 0."""
    ids = [int(i) for i in ids]
    if vocab.soi in ids:
        ids = ids[ids.index(vocab.soi) + 1:]
    grid = np.zeros((12, 12), dtype=np.int64)
    r = c = 0
    nl = vocab.stoi["\n"]
    for i in ids:
        if i == vocab.eoi or r >= 12:
            break
        if i == nl:
            r, c = r + 1, 0
            continue
        if vocab.pix0 <= i < vocab.pix0 + 16 and c < 12:
            grid[r, c] = i - vocab.pix0
            c += 1
    return grid
