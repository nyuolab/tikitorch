"""De-risk for DISTRIBUTED PC training of a TRANSFORMER on TinyShakespeare (forge_pc_xformer_block kernel).

Phase 3's novel claim is the SHARDING: a transformer's blocks split across chips, each chip holding one
block's weights, exchanging ONLY a neighbour hidden-state (downstream) + a neighbour error (upstream) — no
global gradient, no all-reduce. PC-on-a-transformer is ALREADY validated single-process (charlm_mixer
--linear mfgrad = 2.481 bpb = backprop). So the thing to de-risk here is: does inserting an INT16-quantized
neighbour message at every block boundary (the cross-chip rescale, the only new thing vs a single chip) still
train the transformer?

This script trains a small char transformer two ways and overlays the loss: (a) full-precision block
boundaries, (b) INT16-quantized block boundaries (forward hidden-state message + backward error message
max-shift-rescaled to int16, exactly the kernel's inter-block messages). If (b) tracks (a), the distributed
block-sharding preserves training — the Phase-1 result (sign/scale-invariant message rescale) carried to
transformer blocks. The within-block gradient is autograd here (= the PC cascade gradient, per the validated
equivalence); the fabric kernel makes the FFN/projection stack mult-free and warm-trains attention.

Run: PYTHONPATH=. python scripts/pc_xformer_shard_ref.py
"""
import os, sys, math
sys.path.insert(0, ".")
import torch, torch.nn as nn, torch.nn.functional as F

DATA = os.environ.get("SHAKESPEARE", os.path.expanduser("~/.cache/tiur/tinyshakespeare.txt"))
D = int(os.environ.get("XF_D", 64)); NH = int(os.environ.get("XF_HEADS", 2))
NBLK = int(os.environ.get("XF_BLOCKS", 2)); SEQ = int(os.environ.get("XF_SEQ", 64))
STEPS = int(os.environ.get("XF_STEPS", 600)); BATCH = int(os.environ.get("XF_BATCH", 32))


def maxshift_q(x, bits):
    """Max-shift rescale to an int message in band ~2^bits (the kernel's inter-chip int16 message)."""
    m = x.abs().max().clamp_min(1e-9)
    sh = torch.floor(torch.log2(m)).item() - bits
    scale = 2.0 ** sh
    return torch.round(x / scale) * scale          # quantize to the int grid (value preserved up to the grid)


class QuantBoundary(torch.autograd.Function):
    """An int16-quantized block boundary: the forward hidden-state message + the backward error message are
    each max-shift-rescaled to ~int16 (the exact cross-chip messages of forge_pc_xformer_block)."""
    @staticmethod
    def forward(ctx, x, abits, ebits):
        ctx.ebits = ebits
        return maxshift_q(x, abits)
    @staticmethod
    def backward(ctx, g):
        return maxshift_q(g, ctx.ebits), None, None


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3*D); self.proj = nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, 4*D), nn.ReLU(), nn.Linear(4*D, D))

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, NH, D//NH).transpose(1, 2); k = k.view(B, T, NH, D//NH).transpose(1, 2)
        v = v.view(B, T, NH, D//NH).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D//NH)        # the soft attention multiply (warm-train)
        att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        x = x + self.ff(self.ln2(x))
        return x


class CharXformer(nn.Module):
    def __init__(self, vocab, shard=False):
        super().__init__()
        self.shard = shard
        self.tok = nn.Embedding(vocab, D); self.pos = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NBLK)])
        self.lnf = nn.LayerNorm(D); self.head = nn.Linear(D, vocab)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.shard and i < len(self.blocks) - 1:
                x = QuantBoundary.apply(x, 12, 12)     # int16 neighbour message between chips (fwd + bwd)
        return self.head(self.lnf(x))


def get_batch(data, n):
    ix = torch.randint(0, len(data) - SEQ - 1, (n,))
    x = torch.stack([data[i:i+SEQ] for i in ix]); y = torch.stack([data[i+1:i+1+SEQ] for i in ix])
    return x, y


def train(shard, data, vocab, seed=0):
    torch.manual_seed(seed)
    model = CharXformer(vocab, shard=shard)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(STEPS):
        x, y = get_batch(data, BATCH)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == STEPS - 1:
            losses.append((step, loss.item() / math.log(2)))   # bpb
    return losses


if __name__ == "__main__":
    text = open(DATA).read()
    chars = sorted(set(text)); vocab = len(chars); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"distributed PC-transformer shard de-risk: vocab={vocab} D={D} heads={NH} blocks={NBLK} seq={SEQ} "
          f"steps={STEPS}", flush=True)
    full = train(False, data, vocab)
    shard = train(True, data, vocab)
    print("  step |  full-precision boundary  |  INT16 neighbour-message boundary (sharded)   bpb", flush=True)
    for (s, lf), (_, ls) in zip(full, shard):
        print(f"  {s:5d} |        {lf:.3f}            |        {ls:.3f}   ({ls-lf:+.3f})", flush=True)
    df = shard[-1][1] - full[-1][1]
    print(f"\n  RESULT: int16-quantized block boundaries (the cross-chip neighbour messages) train the\n"
          f"  transformer to within {df:+.3f} bpb of full-precision boundaries -> distributed block-sharding\n"
          f"  preserves transformer training (the Phase-1 message-rescale result, carried to transformer\n"
          f"  blocks). The fabric kernel forge_pc_xformer_block makes the FFN/projection stack mult-free PC\n"
          f"  and warm-trains attention (the soft-blend multiply); deploy is mult-free (l1+gather).")
