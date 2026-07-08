#!/usr/bin/env python3
"""enwik8_bpe8192.py -- BPE-8192 tokenizer + train/val bins for the 30M s2 PC-transformer gate run.

Pipeline (deterministic):
  1. Fetch the byte-level enwik8 stream (s3://tiki-sweep-739966866637/s2_bins/train.u16 = 20M uint16
     ids 0..255 == the raw bytes) unless --text is given.
  2. Split: last VAL_BYTES (default 1M) held out as val (never seen by tokenizer training OR training).
  3. Train a GPT-2-style byte-level BPE (HF tokenizers), vocab 8192, NO special tokens (pure stream LM;
     every id 0..8191 is a real piece, matching the kernel's dense S2_VOCAB head).
  4. Encode both splits -> train.u16 / val.u16 (uint16 ids < 8192).
  5. Emit tokenizer.json + STATS.json (bytes, tokens, bytes/token) -- the bytes/token ratio is REQUIRED
     to convert the driver's per-token bits into bits-per-BYTE for the honest cross-tokenizer metric:
         bpb_bytes = bits_per_token / (bytes/token)
  6. --upload: push bins+tokenizer+stats to s3://tiki-sweep-739966866637/s2_bpe8192_bins/

Usage:
  AWS_PROFILE=... python3 scripts/pc_science/enwik8_bpe8192.py --workdir /tmp/bpe8192 --upload
"""
import argparse, json, os, subprocess, sys

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as PreByteLevel
from tokenizers.decoders import ByteLevel as DecByteLevel

S3_SRC = "s3://tiki-sweep-739966866637/s2_bins/train.u16"
S3_OUT = "s3://tiki-sweep-739966866637/s2_bpe8192_bins"
VOCAB = 8192
VAL_BYTES = 1_000_000


def sh(cmd):
    print("+", " ".join(cmd)); subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/bpe8192")
    ap.add_argument("--text", help="optional local text file (default: fetch the byte-level u16 from S3)")
    ap.add_argument("--val-bytes", type=int, default=VAL_BYTES)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    # ---- 1. source text (latin-1 <-> bytes is a 1:1 mapping; preserves the exact byte stream) ----
    if a.text:
        raw = open(a.text, "rb").read()
    else:
        u16 = os.path.join(a.workdir, "src_bytes.u16")
        if not os.path.exists(u16):
            sh(["aws", "s3", "cp", S3_SRC, u16, "--only-show-errors"])
        ids = np.fromfile(u16, dtype=np.uint16)
        assert ids.max() < 256, "source is not byte-level"
        raw = ids.astype(np.uint8).tobytes()
    print(f"source: {len(raw):,} bytes")

    # ---- 2. split ----
    train_b, val_b = raw[: -a.val_bytes], raw[-a.val_bytes:]
    train_txt = os.path.join(a.workdir, "train.txt")
    with open(train_txt, "w", encoding="latin-1", newline="") as f:
        f.write(train_b.decode("latin-1"))

    # ---- 3. train byte-level BPE-8192 (no specials; byte alphabet guarantees full coverage) ----
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = PreByteLevel(add_prefix_space=False)
    tok.decoder = DecByteLevel()
    trainer = BpeTrainer(vocab_size=VOCAB, show_progress=True, special_tokens=[],
                         initial_alphabet=PreByteLevel.alphabet())
    tok.train([train_txt], trainer)
    got = tok.get_vocab_size()
    assert got <= VOCAB, f"vocab {got} > {VOCAB}"
    print(f"vocab: {got}")
    tok_path = os.path.join(a.workdir, "tokenizer.json"); tok.save(tok_path)

    # ---- 4. encode splits -> u16 bins ----
    def encode(bs, name):
        ids = tok.encode(bs.decode("latin-1")).ids
        arr = np.asarray(ids, dtype=np.uint16)
        assert arr.max() < VOCAB
        p = os.path.join(a.workdir, name); arr.tofile(p)
        print(f"{name}: {len(bs):,} bytes -> {len(arr):,} tokens ({len(bs)/len(arr):.3f} bytes/token)")
        return len(arr), len(bs)

    ntr, btr = encode(train_b, "train.u16")
    nva, bva = encode(val_b, "val.u16")

    # ---- 5. stats (bytes/token ratio drives the bits-per-BYTE conversion in analysis) ----
    stats = dict(vocab=got, train_tokens=ntr, train_bytes=btr, val_tokens=nva, val_bytes=bva,
                 train_bytes_per_token=btr / ntr, val_bytes_per_token=bva / nva,
                 note="bpb_bytes = bits_per_token / bytes_per_token; val split = LAST val_bytes of the stream, "
                      "held out from BPE training and LM training")
    with open(os.path.join(a.workdir, "STATS.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    # ---- 6. upload ----
    if a.upload:
        for n in ("train.u16", "val.u16", "tokenizer.json", "STATS.json"):
            sh(["aws", "s3", "cp", os.path.join(a.workdir, n), f"{S3_OUT}/{n}", "--only-show-errors"])
        print(f"uploaded -> {S3_OUT}/")


if __name__ == "__main__":
    sys.exit(main())
