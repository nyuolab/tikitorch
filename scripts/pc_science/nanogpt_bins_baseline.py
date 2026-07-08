#!/usr/bin/env python3
"""nanogpt_bins_baseline.py -- the matched float-BP baseline for the 30M s2 silicon gate run.

A standard pre-norm GPT (nanoGPT-class) trained on the EXACT pre-tokenized .u16 bins the silicon
run uses (s2_bpe8192_bins), at the silicon topology: D=512, L=8, H=8, FFN=2048, SEQ=256,
VOCAB=8192, untied head (the s2 kernel's WHEAD is separate from EMB). 2048 tok/step matches the
silicon's ACCUM=8 x SEQ=256. This is the ceiling for the design-doc gate: silicon converged
val-bpb within ~0.3-0.5 nat of this baseline.

Env config (defaults = the matched run):
  TOKBIN=/tmp/bins/train.u16  VALBIN=/tmp/bins/val.u16  OUT=/tmp/out/baseline.jsonl
  D=512 H=8 L=8 FFN=2048 SEQ=256 VOCAB=8192 BATCH=8 STEPS=25000
  LR=6e-4 WARMUP=500 MINFRAC=0.1 WD=0.1 DROPOUT=0.1 EVAL_EVERY=500 EVAL_BATCHES=64 SEED=7
  BYTES_PER_TOKEN=3.114   # from s2_bpe8192_bins/STATS.json -> converts bits/token to bits/BYTE

Output: jsonl lines {step, train_bpt, val_bpt, val_bpb_bytes, lr}, final summary line
{final: true, ...}, and a 512-token greedy+temperature sample decoded id-stream (ids only —
detokenize with tokenizer.json offline).
"""
import json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

E = os.environ.get
D      = int(E("D", 512)); H = int(E("H", 8)); L = int(E("L", 8)); FFN = int(E("FFN", 2048))
SEQ    = int(E("SEQ", 256)); VOCAB = int(E("VOCAB", 8192)); BATCH = int(E("BATCH", 8))
STEPS  = int(E("STEPS", 25000)); LR = float(E("LR", 6e-4)); WARMUP = int(E("WARMUP", 500))
MINFRAC= float(E("MINFRAC", 0.1)); WD = float(E("WD", 0.1)); DROPOUT = float(E("DROPOUT", 0.1))
EVERY  = int(E("EVAL_EVERY", 500)); EB = int(E("EVAL_BATCHES", 64)); SEED = int(E("SEED", 7))
BPT_RATIO = float(E("BYTES_PER_TOKEN", 3.114))
TOKBIN = E("TOKBIN", "/tmp/bins/train.u16"); VALBIN = E("VALBIN", "/tmp/bins/val.u16")
OUT    = E("OUT", "/tmp/out/baseline.jsonl")

torch.manual_seed(SEED); np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, dropout=DROPOUT, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, FFN), nn.GELU(), nn.Linear(FFN, D), nn.Dropout(DROPOUT))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D); self.pos = nn.Embedding(SEQ, D)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB, bias=False)          # UNTIED (matches the s2 WHEAD)
        mask = torch.full((SEQ, SEQ), float("-inf")).triu(1)
        self.register_buffer("mask", mask)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(t, device=idx.device)))
        for b in self.blocks: x = b(x, self.mask[:t, :t])
        return self.head(self.lnf(x))


def batches(data, rng):
    off = rng.integers(0, len(data) - SEQ - 1, size=BATCH)
    x = np.stack([data[o:o + SEQ] for o in off]).astype(np.int64)
    y = np.stack([data[o + 1:o + SEQ + 1] for o in off]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    train = np.fromfile(TOKBIN, dtype=np.uint16); val = np.fromfile(VALBIN, dtype=np.uint16)
    assert train.max() < VOCAB and val.max() < VOCAB
    model = GPT().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[baseline] {nparam/1e6:.2f}M params on {dev} | train {len(train):,} tok | val {len(val):,} tok", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                             {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))
    rng = np.random.default_rng(SEED)
    vr = np.random.default_rng(1234)  # fixed val batch stream

    def lr_at(s):
        if s < WARMUP: return LR * (s + 1) / WARMUP
        t = (s - WARMUP) / max(1, STEPS - WARMUP)
        return LR * (MINFRAC + (1 - MINFRAC) * 0.5 * (1 + math.cos(math.pi * t)))

    @torch.no_grad()
    def val_bpt():
        model.eval(); tot = 0.0; n = 0
        vrng = np.random.default_rng(1234)
        for _ in range(EB):
            off = vrng.integers(0, len(val) - SEQ - 1, size=BATCH)
            x = torch.from_numpy(np.stack([val[o:o + SEQ] for o in off]).astype(np.int64)).to(dev)
            y = torch.from_numpy(np.stack([val[o + 1:o + SEQ + 1] for o in off]).astype(np.int64)).to(dev)
            with amp:
                loss = F.cross_entropy(model(x).float().reshape(-1, VOCAB), y.reshape(-1))
            tot += loss.item() * x.numel(); n += x.numel()
        model.train()
        return tot / n / math.log(2)     # bits per TOKEN

    f = open(OUT, "a")
    t0 = time.time(); best = 1e9
    for step in range(STEPS):
        lr = lr_at(step)
        for g in opt.param_groups: g["lr"] = lr
        x, y = batches(train, rng)
        with amp:
            loss = F.cross_entropy(model(x).float().reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVERY == 0 or step == STEPS - 1:
            vb = val_bpt(); best = min(best, vb)
            rec = dict(step=step, train_bpt=loss.item() / math.log(2), val_bpt=vb,
                       val_bpb_bytes=vb / BPT_RATIO, lr=lr, sps=(step + 1) / (time.time() - t0))
            print("[baseline]", json.dumps(rec), flush=True); f.write(json.dumps(rec) + "\n"); f.flush()
    rec = dict(final=True, params=nparam, best_val_bpt=best, best_val_bpb_bytes=best / BPT_RATIO,
               steps=STEPS, tok_per_step=BATCH * SEQ, wall_s=time.time() - t0)
    print("[baseline]", json.dumps(rec), flush=True); f.write(json.dumps(rec) + "\n"); f.close()

    # 512-token sample (ids; detokenize offline with tokenizer.json)
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=dev)
    out = []
    with torch.no_grad():
        for _ in range(512):
            logits = model(ctx[:, -SEQ:])[0, -1] / 0.8
            p = F.softmax(logits.float(), dim=-1)
            nxt = torch.multinomial(p, 1)
            out.append(int(nxt)); ctx = torch.cat([ctx, nxt.view(1, 1)], 1)
    print("[sample-ids]", " ".join(map(str, out)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
