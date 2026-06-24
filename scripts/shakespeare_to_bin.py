"""Tokenize TinyShakespeare -> char-level token-id bins for the distributed PC transformer driver
(forge_pc_xformer_shard.c). Char-level, vocab = X_VOCAB=64 (the kernel's vocab). TinyShakespeare has ~65
unique chars; we keep the 64 most frequent and map the rest to the most-frequent char (rare, ~0.0%), so the
64-way next-char prediction (and its bpb) is honest. Writes:
    <out>/train.u16  <out>/val.u16   (uint16 token streams; driver slices SEQ-length windows)
    <out>/vocab.txt   (the 64 chars, index order)
  PYTHONPATH=. python scripts/shakespeare_to_bin.py <out_dir>
"""
import os, sys, urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
VOCAB = 64


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cache = os.environ.get("SHAKESPEARE", os.path.expanduser("~/.cache/tiur/tinyshakespeare.txt"))
    if os.path.exists(cache):
        text = open(cache, encoding="utf-8").read()
    else:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        text = urllib.request.urlopen(URL).read().decode("utf-8")
        open(cache, "w", encoding="utf-8").write(text)
    # vocab = 64 most-frequent chars; rest -> most frequent (idx 0)
    from collections import Counter
    freq = Counter(text)
    chars = [c for c, _ in freq.most_common(VOCAB)]
    stoi = {c: i for i, c in enumerate(chars)}
    fallback = 0
    ids = np.fromiter((stoi.get(c, fallback) for c in text), dtype=np.uint16, count=len(text))
    n = len(ids); ntr = int(n * 0.9)
    ids[:ntr].tofile(os.path.join(out_dir, "train.u16"))
    ids[ntr:].tofile(os.path.join(out_dir, "val.u16"))
    with open(os.path.join(out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("".join(chars))
    cov = sum(freq[c] for c in chars) / n
    print(f"shakespeare: {n} chars, vocab={len(chars)} (coverage {cov:.4%}), "
          f"train {ntr} val {n-ntr} -> {out_dir}/{{train,val}}.u16", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pc_shakespeare")
