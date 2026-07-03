"""Unified lifetime vocabulary: the validated 61-char text alphabet + 18 reserved pixel
symbols (16 gray levels + image delimiters). Frozen at birth to vocab.json — the input
dimension ctx*V is fixed for life.

Pixel symbols live in the Unicode private-use area so they can never collide with text."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
from two_speed_learner import ALPHABET  # the validated 61-char text alphabet

PIX = [chr(0xE000 + i) for i in range(16)]   # ⟨P0⟩..⟨P15⟩ gray levels
SOI = chr(0xE010)                             # start-of-image
EOI = chr(0xE011)                             # end-of-image
RESERVED = PIX + [SOI, EOI]

ASCII_RAMP = " .:-=+*#%@&$M@@@"               # 16-level human render for reports


class Vocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.V = len(self.chars)
        self.pix0 = self.stoi[PIX[0]]
        self.soi = self.stoi[SOI]
        self.eoi = self.stoi[EOI]

    @classmethod
    def build(cls) -> "Vocab":
        # text chars in ALPHABET order (deterministic), then reserved
        return cls(sorted(set(ALPHABET)) + RESERVED)

    def normalize_text(self, raw: str) -> str:
        """The validated text normalization (lowercase, alphabet filter, space-collapse)."""
        allowed = set(ALPHABET)
        norm = "".join(c if c in allowed else " " for c in raw.lower())
        return " ".join(norm.split(" "))

    def encode_text(self, raw: str) -> list:
        return [self.stoi[c] for c in self.normalize_text(raw)]

    def encode_mixed(self, s: str) -> list:
        """Encode a string that may already contain reserved pixel chars (e.g. prompts)."""
        out = []
        for c in s:
            if c in self.stoi and c in set(RESERVED):
                out.append(self.stoi[c])
            else:
                out.extend(self.stoi[x] for x in self.normalize_text(c))
        return out

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def render_human(self, ids) -> str:
        """Reserved chars -> ASCII ramp / [ ] markers, for reports and samples."""
        out = []
        for i in ids:
            c = self.itos[int(i)]
            if c in PIX:
                out.append(ASCII_RAMP[PIX.index(c)])
            elif c == SOI:
                out.append("[")
            elif c == EOI:
                out.append("]")
            else:
                out.append(c)
        return "".join(out)

    def vocab_hash(self) -> str:
        import hashlib
        return hashlib.sha256("".join(self.chars).encode()).hexdigest()[:16]

    def save(self, path: str):
        json.dump({"chars": self.chars, "hash": self.vocab_hash()}, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "Vocab":
        return cls(json.load(open(path))["chars"])
