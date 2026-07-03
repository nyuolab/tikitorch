"""Diagnostic: is the min-plus recurrence's char-LM stall a real architecture limit, or a
probe artifact (min-plus HEAD as logit layer / lr too high)? 2x2 attribution + lr, matched
to the working dense arm (lr=3e-3). Cheap, decisive.

  dense-rec + dense-head   = 2.808 (works, from the probe)
  dense-rec + MINPLUS-head : isolates whether the HEAD kills it
  MINPLUS-rec + dense-head : isolates whether the RECURRENCE underfits LM
If minplus-rec+dense-head learns (beats bigram 3.58) -> probe was crippled, re-run properly.
If it stalls near unigram -> the cascade-friendly min-plus recurrence underfits LM (real).
"""
import sys, math, time, os
sys.path.insert(0, ".")
import torch, torch.nn as nn, torch.nn.functional as F
from tikitorch.nn import TropicalLinearModule
torch.manual_seed(0)

text = open("/tmp/_tiki_data/tinyshakespeare.txt").read()
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data))
train_d, val_d = data[:n], data[n:]
SEQ, BATCH, DM = 48, 32, 96


def get_batch(d, gen):
    ix = torch.randint(len(d) - SEQ - 1, (BATCH,), generator=gen)
    return (torch.stack([d[i:i+SEQ] for i in ix]),
            torch.stack([d[i+1:i+SEQ+1] for i in ix]))


class RNN(nn.Module):
    def __init__(self, rec, head):
        super().__init__()
        self.emb = nn.Embedding(V, DM)
        self.rec = rec                       # "minplus" | "dense"
        if rec == "minplus":
            self.Wh = TropicalLinearModule(DM, DM, use_softmin=True, tau=1.0)
            self.Wx = TropicalLinearModule(DM, DM, use_softmin=True, tau=1.0)
        else:
            self.Wh = nn.Linear(DM, DM); self.Wx = nn.Linear(DM, DM)
        self.headkind = head
        self.head = (TropicalLinearModule(DM, V, use_softmin=True, tau=1.0)
                     if head == "minplus" else nn.Linear(DM, V))
        self._hstat = None

    def forward(self, x):
        e = self.emb(x); B, S, _ = e.shape
        h = torch.zeros(B, DM); outs = []
        for t in range(S):
            if self.rec == "minplus":
                h = torch.min(self.Wh(h), self.Wx(e[:, t, :]))
            else:
                h = torch.tanh(self.Wh(h) + self.Wx(e[:, t, :]))
            outs.append(h)
        H = torch.stack(outs, 1)
        self._hstat = (H.mean().item(), H.std().item(), H.min().item(), H.max().item())
        return self.head(H.reshape(B*S, DM)).reshape(B, S, V)


def run(rec, head, lr, steps=600, tag=""):
    torch.manual_seed(10); m = RNN(rec, head)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1); eg = torch.Generator().manual_seed(2)
    t0 = time.time()
    for s in range(steps):
        x, y = get_batch(train_d, g)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); tot = 0.0
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(val_d, eg)
            tot += F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
    bpc = tot/20/math.log(2)
    print(f"  {tag:<34} val_bpc {bpc:.3f}  | h(mean {m._hstat[0]:+.2f} std {m._hstat[1]:.2f} "
          f"min {m._hstat[2]:+.2f} max {m._hstat[3]:+.2f})  {time.time()-t0:.0f}s", flush=True)
    return bpc


print("bigram floor 3.581 | dense-rec+dense-head 2.808 (from probe)\n", flush=True)
print("ATTRIBUTION (all lr=3e-3, soft min-plus):", flush=True)
run("dense", "minplus", 3e-3, tag="dense-rec + MINPLUS-head")     # head isolation
run("minplus", "dense", 3e-3, tag="MINPLUS-rec + dense-head")     # recurrence isolation
print("\nlr sweep for MINPLUS-rec + dense-head:", flush=True)
for lr in (1e-2, 1e-3):
    run("minplus", "dense", lr, tag=f"  minplus-rec dense-head lr={lr}")
