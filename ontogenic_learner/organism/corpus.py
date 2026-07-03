"""Domain corpora + lifetime cursors.

Text domains reuse the validated normalization (via Vocab). Each domain gets:
  - a stream region (first 90%) walked by a Cursor with deterministic offset-shift re-epoch,
  - a frozen held-out tail (last 10%) for probes (positions fixed at birth).
MNIST regenerates a shuffled epoch stream from its checkpointed generator on exhaustion.
`docs` is probe-only (held back for life); `code` is the day-8 novelty domain."""
import glob
import os
import sys

import torch

from .mnist_tokens import load_mnist, epoch_stream
from .vocab import Vocab

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _read(p):
    return open(p, encoding="utf-8", errors="ignore").read()


def raw_text(name: str) -> str:
    if name == "wiki":
        return _read(os.path.join(REPO, "data/text8_20m.txt"))
    if name == "shakespeare":
        return _read(os.path.join(REPO, "data/tinyshakespeare.txt"))
    if name == "docs":
        ps = sorted(p for p in glob.glob(os.path.join(REPO, "**/*.md"), recursive=True)
                    if ".git" not in p and "ontogenic_learner" not in p)
        return "\n\n".join(_read(p) for p in ps)
    if name == "code":
        ps = sorted(glob.glob(os.path.join(REPO, "tikitorch/**/*.py"), recursive=True) +
                    glob.glob(os.path.join(REPO, "scripts/*.py")))
        return "\n\n".join(_read(p) for p in ps)
    raise ValueError(name)


class Cursor:
    """Walks a token array in stride-1 windows, B tokens per step; deterministic
    offset-shift re-epoch on exhaustion."""

    def __init__(self, name: str, ids: torch.Tensor, ctx: int, batch: int):
        self.name, self.ids, self.ctx, self.B = name, ids, ctx, batch
        self.hi = int(0.9 * len(ids))            # stream region; tail is held out
        self.pos = 0
        self.epoch = 0

    def next_windows(self):
        if self.pos + self.B + self.ctx + 1 > self.hi:
            self.epoch += 1
            self.pos = (self.epoch * 4093) % self.B
        xs = torch.stack([self.ids[self.pos + j: self.pos + j + self.ctx]
                          for j in range(self.B)])
        ys = self.ids[self.pos + self.ctx: self.pos + self.ctx + self.B]
        self.pos += self.B
        return xs, ys

    def state(self):
        return {"pos": self.pos, "epoch": self.epoch}

    def restore(self, st):
        self.pos, self.epoch = st["pos"], st["epoch"]


class MnistCursor(Cursor):
    """Regenerates a shuffled epoch stream (checkpointed generator) on exhaustion."""

    def __init__(self, vocab: Vocab, ctx: int, batch: int, gen: torch.Generator):
        self.vocab, self.gen = vocab, gen
        Xtr, Ytr, Xte, Yte = load_mnist(REPO)
        self.Xtr, self.Ytr = Xtr, Ytr
        self.test = (Xte, Yte)                   # probe-only, never streamed
        ids = epoch_stream(vocab, Xtr, Ytr, gen)
        super().__init__("mnist", ids, ctx, batch)
        self.hi = len(ids) - ctx - batch - 1     # whole epoch streams; tail not needed

    def next_windows(self):
        if self.pos + self.B + self.ctx + 1 > self.hi:
            self.epoch += 1
            self.pos = 0
            self.ids = epoch_stream(self.vocab, self.Xtr, self.Ytr, self.gen)
            self.hi = len(self.ids) - self.ctx - self.B - 1
        return super(MnistCursor, self).next_windows()


class World:
    """All domains + frozen probe tails. Build once at birth; cursors are checkpointed."""

    TEXT_DOMAINS = ["wiki", "shakespeare", "code", "docs"]

    def __init__(self, vocab: Vocab, ctx: int, batch: int, mnist_gen: torch.Generator,
                 val_windows: int = 4096):
        self.vocab, self.ctx, self.B = vocab, ctx, batch
        self.ids = {}
        for d in self.TEXT_DOMAINS:
            self.ids[d] = torch.tensor(vocab.encode_text(raw_text(d)), dtype=torch.long)
        self.cursors = {d: Cursor(d, self.ids[d], ctx, batch)
                        for d in ["wiki", "shakespeare", "code"]}
        self.cursors["mnist"] = MnistCursor(vocab, ctx, batch, mnist_gen)
        # frozen held-out probe windows (positions fixed at construction = birth)
        self.val = {}
        for d in ["wiki", "shakespeare", "code", "docs"]:
            ids = self.ids[d]
            start = int(0.9 * len(ids))
            self.val[d] = self._windows(ids, start, val_windows, 3)
        Xte, Yte = self.cursors["mnist"].test
        self.mnist_test = (Xte, Yte)

    def _windows(self, ids, start, count, stride):
        xs, ys, i = [], [], start
        for _ in range(count):
            if i + self.ctx + 1 > len(ids):
                break
            xs.append(ids[i:i + self.ctx]); ys.append(ids[i + self.ctx]); i += stride
        return torch.stack(xs), torch.stack(ys)

    def add_guest(self, name: str, text: str):
        """`organism feed` — register a bounded guest domain (real experience)."""
        ids = torch.tensor(self.vocab.encode_text(text), dtype=torch.long)
        self.ids[name] = ids
        c = Cursor(name, ids, self.ctx, self.B)
        c.hi = len(ids) - self.ctx - self.B - 1
        self.cursors[name] = c
        return c

    def cursor_states(self):
        return {d: c.state() for d, c in self.cursors.items()}

    def restore_cursors(self, states):
        for d, st in states.items():
            if d in self.cursors:
                self.cursors[d].restore(st)
