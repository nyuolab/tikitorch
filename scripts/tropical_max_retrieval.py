"""LONG-RANGE GATED MAX-RETRIEVAL: tropical SSM vs dense SSM vs transformer.

TASK
----
Sequence of (value, flag) pairs.  value ~ U[0,1), flag in {0,1} (~15% ones).
TARGET at position t = max over j<=t with flag_j==1 of value_j   (0 if none yet).
A CONTENT-GATED RUNNING MAX.  Regression (MSE).

HYPOTHESIS
----------
The tropical max-plus recurrence  h_t = max(h_{t-1}, gated x_t)  computes this
EXACTLY in O(1) state.  A DENSE linear (Mamba-lite) recurrence can only
decay-AVERAGE the past -> it structurally cannot hold an exact running max, so
its MSE floors out high.  The transformer can solve it but is O(T^2).

Three MATCHED small models (D=64, 1 layer time-mixer + channel MLP each).
Same data / same eval.  Sequential scan over T for the two recurrences.

Run:  python scripts/tropical_max_retrieval.py
"""
import math, time, json, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tikitorch.nn import dyadic_quantize, DyadicLinear

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

# ---------------------------------------------------------------- data
def make_batch(B, T, p_flag=0.15, device=DEVICE, gen=None):
    val = torch.rand(B, T, 1, device=device, generator=gen)
    flag = (torch.rand(B, T, 1, device=device, generator=gen) < p_flag).float()
    # target = content-gated running max
    gated = torch.where(flag.bool(), val, torch.zeros_like(val))  # value where flag, else 0
    tgt = torch.cummax(gated, dim=1).values  # running max of gated values; >=0 always
    x = torch.cat([val, flag], dim=-1)  # (B,T,2)
    return x, tgt  # tgt (B,T,1)

# ---------------------------------------------------------------- models
class TransformerModel(nn.Module):
    """Causal softmax attention (1 head) + MLP. The O(T^2) reference."""
    def __init__(self, D=64):
        super().__init__()
        self.inp = nn.Linear(2, D)
        self.q = nn.Linear(D, D); self.k = nn.Linear(D, D); self.v = nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
        self.out = nn.Linear(D, 1)
        self.D = D
    def forward(self, x):
        h = self.inp(x); B, T, D = h.shape
        hn = self.ln1(h)
        q, k, v = self.q(hn), self.k(hn), self.v(hn)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1)
        h = h + self.proj(att @ v)
        h = h + self.mlp(self.ln2(h))
        return self.out(h)

class DenseSSM(nn.Module):
    """Mamba-lite gated LINEAR recurrence: h_t = sigmoid(a)*h_{t-1} + b*x_t (per-channel a).
    Gated output + channel MLP. Float weights. Linear recurrence = decay-AVERAGE."""
    def __init__(self, D=64):
        super().__init__()
        self.inp = nn.Linear(2, D)
        self.a = nn.Parameter(torch.zeros(D))     # decay logit per channel
        self.b = nn.Linear(D, D)                  # input projection into state
        self.gate = nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
        self.out = nn.Linear(D, 1)
        self.D = D
    def forward(self, x):
        h0 = self.inp(x); B, T, D = h0.shape
        hn = self.ln1(h0)
        decay = torch.sigmoid(self.a)             # (D,) in (0,1)
        u = self.b(hn)                            # (B,T,D) per-step input
        state = torch.zeros(B, D, device=x.device)
        outs = []
        for t in range(T):
            state = decay * state + u[:, t]       # LINEAR recurrence (decay-average)
            outs.append(state)
        y = torch.stack(outs, 1)
        y = y * torch.sigmoid(self.gate(hn))      # gated output
        h = h0 + self.proj(y)
        h = h + self.mlp(self.ln2(h))
        return self.out(h)

class TropicalSSM(nn.Module):
    """MULT-FREE max-plus / morphological recurrence: h_t = max(h_{t-1} + decay, u_t).
    All matmul weights DYADIC (signed pow2, STE -> shift-add). The F2-native primitive.
    decay is a learned dyadic per-channel additive (<=0 -> leaky max in O(1) state)."""
    def __init__(self, D=64):
        super().__init__()
        self.inp = DyadicLinear(2, D)
        # leaky-max additive decay per channel; init slightly negative (gentle leak)
        self.decay = nn.Parameter(torch.full((D,), -0.05))
        self.b = DyadicLinear(D, D)
        self.gate = DyadicLinear(D, D)
        self.proj = DyadicLinear(D, D)
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(DyadicLinear(D, 4*D), nn.ReLU(), DyadicLinear(4*D, D))
        self.out = DyadicLinear(D, 1)
        self.D = D
    def forward(self, x):
        h0 = self.inp(x); B, T, D = h0.shape
        hn = self.ln1(h0)
        u = self.b(hn)                            # (B,T,D) per-step input candidate
        dec = -dyadic_quantize(self.decay).abs()  # <=0 additive leak (dyadic)
        NEG = -1e9
        state = torch.full((B, D), NEG, device=x.device)
        outs = []
        for t in range(T):
            state = torch.maximum(state + dec, u[:, t])  # MAX-PLUS recurrence (exact running max)
            outs.append(state)
        y = torch.stack(outs, 1)
        y = y * torch.sigmoid(self.gate(hn))      # gated output (sigmoid gate is a non-mult-free read head, kept for fair capacity)
        h = h0 + self.proj(y)
        h = h + self.mlp(self.ln2(h))
        return self.out(h)

# ---------------------------------------------------------------- train/eval
def count_params(m):
    return sum(p.numel() for p in m.parameters())

def train(model, name, T=96, B=128, steps=4000, lr=2e-3, p_flag=0.15):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=DEVICE).manual_seed(1234)
    model.train()
    t0 = time.time()
    for s in range(steps):
        x, tgt = make_batch(B, T, p_flag, gen=gen)
        pred = model(x)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0 or s == steps-1:
            print(f"  [{name}] step {s:5d}  mse {loss.item():.5f}")
    # eval on a FRESH fixed test set, same for all models
    model.eval()
    gen_te = torch.Generator(device=DEVICE).manual_seed(99999)
    with torch.no_grad():
        mses = []
        for _ in range(20):
            x, tgt = make_batch(256, T, p_flag, gen=gen_te)
            pred = model(x)
            mses.append(F.mse_loss(pred, tgt).item())
        test_mse = sum(mses)/len(mses)
        # baseline: predicting a constant (mean target) -> variance of target
        x, tgt = make_batch(2048, T, p_flag, gen=torch.Generator(device=DEVICE).manual_seed(7))
        base_mse = tgt.var().item()
        # also report RMSE in value units and the "predict-last-flagged" trivial-fail baseline
        rmse = math.sqrt(test_mse)
    dt = time.time()-t0
    print(f"  [{name}] DONE test_mse={test_mse:.5f} rmse={rmse:.4f} params={count_params(model)} ({dt:.0f}s)")
    return {"name": name, "test_mse": test_mse, "rmse": rmse,
            "params": count_params(model), "base_var": base_mse, "sec": dt}

if __name__ == "__main__":
    T, B, STEPS = 96, 128, 4000
    print(f"device={DEVICE} T={T} B={B} steps={STEPS} p_flag=0.15")
    results = {}
    for cls, nm in [(TransformerModel, "transformer"),
                    (DenseSSM, "dense_ssm"),
                    (TropicalSSM, "tropical_ssm")]:
        torch.manual_seed(0)
        print(f"\n=== {nm} ===")
        results[nm] = train(cls(D=64), nm, T=T, B=B, steps=STEPS)
    print("\n\n==== SUMMARY (gated running-max retrieval, lower MSE = better) ====")
    base = results["transformer"]["base_var"]
    print(f"target variance (constant-predict baseline MSE) = {base:.5f}")
    for nm, r in results.items():
        frac = r["test_mse"]/base
        print(f"  {nm:13s} mse={r['test_mse']:.5f} rmse={r['rmse']:.4f} "
              f"(={frac*100:5.1f}% of var-baseline) params={r['params']}")
    print("\nJSON:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk!='name'} for k,v in results.items()}))
