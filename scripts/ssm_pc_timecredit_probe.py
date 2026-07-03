"""Architecture-gate probe: can a MULTIPLIER-FREE tropical recurrence train a char-LM,
and does its HARD (argmin-routed, mult-free) credit survive vs the SOFT (smooth) credit?

Context (the scaling thesis). The fabric/scale-out win needs PC-style LOCAL credit (no
global backward, no weight transport). The cleanest carrier is the TROPICAL MIN-PLUS
recurrence already in the repo (TropicalSequenceModel): h_t = min(W_h @+ h_{t-1}, W_x @+ x_t),
where @+ is min-plus matmul y_j = min_i(W[j,i] + x[i]). KEY FACT this probe rests on: a
min-plus layer is multiplier-free in BOTH forward (add+min) AND backward — the weight
gradient is a SCATTER of the upstream error to the argmin positions (no multiply, no
dyadic-quantize, no cascade trick). So the credit is pure routing via cached argmin
selectors = exactly the local / weight-transport-free / async-capable object the fabric
thesis needs. The cascade machinery (for real-matmul DyadicLinear layers) is not even
required here.

The gate therefore reduces to two cheap, decisive questions (the smooth S4/Mamba fork is
WALLED — its Jacobian diag(a) is a learned smooth scalar, the thing that broke mult-free
attention; min-plus dodges it because its Jacobian is (0/1 win-selector) ⊙ W):

  KILL 1 — CAPACITY/TRAINABILITY: does the (best-case SOFT) min-plus recurrence learn
    next-char above the bigram floor and near a dense real-matmul twin? If it stalls at
    the floor, the recurrence underfits LM → gate dies.
  KILL 2 — THE SMOOTHNESS WALL (decisive): does the HARD (argmin, fully multiplier-free
    credit) recurrence train comparably to SOFT? If HARD collapses while SOFT trains, the
    "mult-free underfits smooth" wall that broke attention hits the recurrence too → only
    mult-free DEPLOY survives, not mult-free TRAINING. If HARD ≈ SOFT, the recurrence
    DODGES the wall → we have a mult-free, routing-credit (fabric-friendly) sequence
    learner. Also reports cosine(soft-grad, hard-grad) as the wall indicator.

Honest scope: char tiny-shakespeare, tiny model, short budget — a FALSIFICATION probe, not
a competitive LM. A pass means "worth scaling + wiring to the async/fabric form (KILL 3,
not run here)"; a fail kills the gate cheaply and says exactly why.
Run: PYTHONPATH=. python scripts/ssm_pc_timecredit_probe.py
"""
import sys, math, time, urllib.request, os
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from tikitorch.nn import TropicalLinearModule

torch.manual_seed(0)
DEV = "cpu"

# ---- data: tiny-shakespeare char ----
PATH = "/tmp/_tiki_data/tinyshakespeare.txt"
os.makedirs("/tmp/_tiki_data", exist_ok=True)
if not os.path.exists(PATH):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, PATH)
text = open(PATH).read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_d, val_d = data[:n], data[n:]
print(f"chars={V}  train={len(train_d)}  val={len(val_d)}", flush=True)

SEQ = 48
BATCH = 32
DM = 96


def get_batch(d, gen):
    ix = torch.randint(len(d) - SEQ - 1, (BATCH,), generator=gen)
    x = torch.stack([d[i:i + SEQ] for i in ix])
    y = torch.stack([d[i + 1:i + SEQ + 1] for i in ix])
    return x, y


# ---- bigram / unigram floor (the must-beat) ----
def ngram_floor():
    uni = torch.bincount(train_d, minlength=V).float() + 1
    uni /= uni.sum()
    unigram_bpc = -(uni * uni.log2()).sum().item()
    big = torch.ones(V, V)
    for a, b in zip(train_d[:-1].tolist(), train_d[1:].tolist()):
        big[a, b] += 1
    big /= big.sum(1, keepdim=True)
    # bigram cross-entropy on val
    va, vb = val_d[:-1], val_d[1:]
    p = big[va, vb].clamp_min(1e-9)
    bigram_bpc = -(p.log2()).mean().item()
    return unigram_bpc, bigram_bpc


# ---- models ----
class MinPlusRNN(nn.Module):
    """One-layer tropical min-plus recurrence char-LM. soft=True uses softmin (smooth,
    has exp); soft=False uses hard argmin (fully multiplier-free credit, scatter grad)."""
    def __init__(self, soft):
        super().__init__()
        self.emb = nn.Embedding(V, DM)                                  # gather (boundary)
        self.Wh = TropicalLinearModule(DM, DM, use_softmin=soft, tau=1.0)
        self.Wx = TropicalLinearModule(DM, DM, use_softmin=soft, tau=1.0)
        self.head = TropicalLinearModule(DM, V, use_softmin=soft, tau=1.0)

    def forward(self, x):
        e = self.emb(x)                                                 # [B,S,DM]
        B, S, _ = e.shape
        h = torch.zeros(B, DM, device=e.device)
        outs = []
        for t in range(S):
            h = torch.min(self.Wh(h), self.Wx(e[:, t, :]))             # tropical recurrence
            outs.append(h)
        H = torch.stack(outs, 1)                                        # [B,S,DM]
        return self.head(H.reshape(B * S, DM)).reshape(B, S, V)


class DenseRNN(nn.Module):
    """Real-matmul linear recurrence twin (the competitive bar; HAS multiplies)."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, DM)
        self.Wh = nn.Linear(DM, DM)
        self.Wx = nn.Linear(DM, DM)
        self.head = nn.Linear(DM, V)

    def forward(self, x):
        e = self.emb(x)
        B, S, _ = e.shape
        h = torch.zeros(B, DM, device=e.device)
        outs = []
        for t in range(S):
            h = torch.tanh(self.Wh(h) + self.Wx(e[:, t, :]))
            outs.append(h)
        H = torch.stack(outs, 1)
        return self.head(H.reshape(B * S, DM)).reshape(B, S, V)


def evaluate(model, gen, iters=20):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(val_d, gen)
            logits = model(x)
            tot += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item()
    model.train()
    return tot / iters / math.log(2)        # bits per char


def train(model, steps, lr, tag):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(1)
    eg = torch.Generator().manual_seed(2)
    t0 = time.time()
    for s in range(steps):
        x, y = get_batch(train_d, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, steps // 6) == 0 or s == steps - 1:
            print(f"  [{tag}] step {s:>4} train_bpc {loss.item()/math.log(2):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return evaluate(model, eg)


def grad_cosine():
    """Wall indicator: cosine of the WEIGHT gradient under soft vs hard routing on one batch."""
    gen = torch.Generator().manual_seed(7)
    x, y = get_batch(train_d, gen)
    torch.manual_seed(123); soft = MinPlusRNN(soft=True)
    torch.manual_seed(123); hard = MinPlusRNN(soft=False)              # identical init
    def gvec(m):
        m.zero_grad()
        F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).backward()
        return torch.cat([m.Wh.weight.grad.reshape(-1), m.Wx.weight.grad.reshape(-1),
                          m.head.weight.grad.reshape(-1)])
    gs, gh = gvec(soft), gvec(hard)
    return F.cosine_similarity(gs, gh, dim=0).item()


# ---- run ----
uni, big = ngram_floor()
print(f"FLOOR: unigram {uni:.3f} bpc | bigram {big:.3f} bpc  (must beat bigram)\n", flush=True)

cos = grad_cosine()
print(f"KILL-2 indicator: cosine(soft-grad, hard-grad) = {cos:.3f}\n", flush=True)

STEPS_T = 400
STEPS_D = 600
print("training arms (short budget):", flush=True)
torch.manual_seed(10); dense = DenseRNN();           bpc_dense = train(dense, STEPS_D, 3e-3, "DENSE bar")
torch.manual_seed(10); soft = MinPlusRNN(soft=True);  bpc_soft = train(soft, STEPS_T, 5e-2, "SOFT minplus")
torch.manual_seed(10); hard = MinPlusRNN(soft=False); bpc_hard = train(hard, STEPS_T, 5e-2, "HARD minplus(mult-free)")

print("\n================ RESULTS (val bits-per-char) ================", flush=True)
print(f"  bigram floor (must beat)      : {big:.3f}")
print(f"  DENSE real-matmul twin (bar)  : {bpc_dense:.3f}")
print(f"  SOFT  min-plus (smooth ref)   : {bpc_soft:.3f}")
print(f"  HARD  min-plus (MULT-FREE)    : {bpc_hard:.3f}")
print(f"  cosine(soft,hard) grad        : {cos:.3f}")
print("\n================ VERDICT ================", flush=True)
k1 = bpc_soft < big - 0.05
k2 = bpc_hard < big - 0.05 and (bpc_hard - bpc_soft) < 0.30
print(f"  KILL-1 capacity  (SOFT beats bigram): {'PASS' if k1 else 'FAIL'} "
      f"(soft {bpc_soft:.3f} vs floor {big:.3f})")
print(f"  KILL-2 smoothness(HARD≈SOFT, mult-free trains): {'PASS' if k2 else 'FAIL'} "
      f"(hard {bpc_hard:.3f}, gap to soft {bpc_hard-bpc_soft:+.3f})")
print(f"  GATE (this probe): {'SURVIVES — scale + wire async (KILL-3)' if (k1 and k2) else 'DIES — see which kill'}")
