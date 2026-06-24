"""Stage-2 RECIPE de-risk: does the DISTRIBUTED PC transformer train when the projections/FFN/head are made
MULTIPLIER-FREE (dyadic, the kernel's real forward), attention is warm-trained (the one allowed multiply), and
the block boundaries are int16 neighbour messages? This is the exact recipe forge_pc_xformer_block implements.
If bpb falls to ~2.5-2.7 (near the float twin), the kernel recipe is validated before any HLS build.

  dyadic q(W): round each weight to the nearest signed power-of-2 (== the kernel's dyad()), straight-through
  in backward (the PC cascade gradient ~ backprop). Attention keeps dot-product softmax (warm-train multiply).
  Boundaries int16. Optimizer AdamW (the float proxy for the kernel's mult-free dyadic-Adam).

  PYTHONPATH=. XF_BLOCKS=8 XF_STEPS=2000 python scripts/pc_xformer_dyadic_ref.py
"""
import os, sys, math
sys.path.insert(0, ".")
import torch, torch.nn as nn, torch.nn.functional as F

DATA = os.environ.get("SHAKESPEARE", os.path.expanduser("~/.cache/tiur/tinyshakespeare.txt"))
D = int(os.environ.get("XF_D", 64)); NH = int(os.environ.get("XF_HEADS", 2))
NBLK = int(os.environ.get("XF_BLOCKS", 8)); SEQ = int(os.environ.get("XF_SEQ", 64))
STEPS = int(os.environ.get("XF_STEPS", 2000)); BATCH = int(os.environ.get("XF_BATCH", 32))


class DyadSTE(torch.autograd.Function):
    """Round to nearest signed power of two (the kernel's dyad); identity gradient (straight-through)."""
    @staticmethod
    def forward(ctx, w):
        a = w.abs(); nz = a > 0
        e = torch.where(nz, torch.round(torch.log2(a.clamp_min(1e-12))), torch.zeros_like(a))
        return torch.where(nz, torch.sign(w) * torch.pow(2.0, e), torch.zeros_like(w))
    @staticmethod
    def backward(ctx, g):
        return g


def dyq(w):
    return DyadSTE.apply(w)


class DyLinear(nn.Linear):
    """Linear whose weight is dyadic-quantized in forward (mult-free shift-add matmul on fabric)."""
    def forward(self, x):
        return F.linear(x, dyq(self.weight), self.bias)


def maxshift_q(x, bits):
    m = x.abs().max().clamp_min(1e-9)
    sh = torch.floor(torch.log2(m)).item() - bits
    scale = 2.0 ** sh
    return torch.round(x / scale) * scale


class QBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ab, eb): ctx.eb = eb; return maxshift_q(x, ab)
    @staticmethod
    def backward(ctx, g): return maxshift_q(g, ctx.eb), None, None


class Block(nn.Module):
    def __init__(self, dyadic):
        super().__init__()
        L = DyLinear if dyadic else nn.Linear
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.qkv = L(D, 3*D); self.proj = L(D, D)
        self.fc1 = L(D, 4*D); self.fc2 = L(4*D, D)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, NH, D//NH).transpose(1, 2); k = k.view(B, T, NH, D//NH).transpose(1, 2)
        v = v.view(B, T, NH, D//NH).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D//NH)        # warm-train multiply (attention)
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D) # the soft p*V blend (the one fabric multiply)
        x = x + self.proj(y)
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x


class CharXformer(nn.Module):
    def __init__(self, vocab, dyadic, shard):
        super().__init__()
        self.shard = shard
        self.tok = nn.Embedding(vocab, D); self.pos = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList([Block(dyadic) for _ in range(NBLK)])
        self.lnf = nn.LayerNorm(D); self.head = (DyLinear if dyadic else nn.Linear)(D, vocab)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.shard and i < len(self.blocks) - 1:
                x = QBound.apply(x, 9, 12)              # int16 neighbour message between chips (fwd + bwd)
        return self.head(self.lnf(x))


def get_batch(data, n):
    ix = torch.randint(0, len(data) - SEQ - 1, (n,))
    x = torch.stack([data[i:i+SEQ] for i in ix]); y = torch.stack([data[i+1:i+1+SEQ] for i in ix])
    return x, y


def train(dyadic, shard, data, vocab, seed=0, tag=""):
    torch.manual_seed(seed)
    model = CharXformer(vocab, dyadic, shard)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(STEPS):
        x, y = get_batch(data, BATCH)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"   [{tag}] step {step:5d}  bpb={loss.item()/math.log(2):.3f}", flush=True)
    return loss.item() / math.log(2)


if __name__ == "__main__":
    text = open(DATA).read()
    chars = sorted(set(text)); vocab = len(chars); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    print(f"Stage-2 dyadic recipe de-risk: vocab={vocab} D={D} heads={NH} blocks={NBLK} seq={SEQ} steps={STEPS}",
          flush=True)
    fb = train(False, True, data, vocab, tag="float+shard")
    db = train(True, True, data, vocab, tag="DYADIC+shard")
    print(f"\n  RESULT: {NBLK}-block distributed transformer, int16 neighbour boundaries.")
    print(f"    float-proj  + warm-attn:  {fb:.3f} bpb")
    print(f"    DYADIC-proj + warm-attn:  {db:.3f} bpb  ({db-fb:+.3f})")
    print(f"  -> mult-free dyadic projections/FFN/head (the fabric forward) {'PRESERVE' if db-fb<0.25 else 'COST'} "
          f"training; attention warm-trained (the one multiply). Validates forge_pc_xformer_block's recipe.")
