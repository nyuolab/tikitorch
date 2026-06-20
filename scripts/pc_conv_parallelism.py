"""Rung 3 (part A): measured parallelism economics on the COMPETITIVE conv model.

The fabric thesis: local PC trades [global backward sweep + weight transport + global
barrier/all-reduce] for [O(L/p) local async rounds]. On the MLP we measured the round-tax
(rounds for the async error-relaxation grad to reach the EXACT sync/backprop grad) = O(L/p)
(staleness_frontier). The OPEN question (design R-a): does the dilated conv's long receptive
field INFLATE the round-tax (error must propagate across the receptive field)? This measures
it directly on DyadicConvPC1d (the competitive arch), cheaply (gradient cosine vs rounds, no
training).

For a fixed batch: reference = SYNC error-relaxation grad (== backprop credit, bit-exact).
Then per condition (partial participation / stale reads / random order), sweep n_iter and
measure the mean per-layer weight-grad cosine to the reference. round-tax = smallest n_iter
to reach cosine >= 0.999. If round-tax stays ~O(depth) (not O(depth*RF)), the dilated conv
does NOT inflate it and the fabric economics carry over from the MLP. Run:
  PYTHONPATH=. python scripts/pc_conv_parallelism.py
"""
import sys, json, math
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from scripts.pc_charlm_conv import DyadicConvPC1d, load

torch.manual_seed(0)
DEV = "cpu"
DEPTH = 5
DILS = [2 ** i for i in range(DEPTH)]
RF = 1 + (5 - 1) * sum(DILS)
SEQ = 256
ROUNDS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
N_TRIALS = 5
WARM = 25

tr, va, V, d_full, _ = load(SEQ, subset=300000, stride=64)
xb, yb = tr[0][:24].to(DEV), tr[1][:24].to(DEV)
print(f"conv parallelism probe: depth={DEPTH} dils={DILS} RF={RF} seq={SEQ} V={V}", flush=True)

net = DyadicConvPC1d(V, width=96, depth=DEPTH, kernel=5, dilations=DILS).to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=7e-4)
for _ in range(WARM):                                       # realistic weights
    net.train_step_relax(xb, yb, opt)
tgt = F.one_hot(yb, V).float()


def gvec(net):
    return torch.cat([net.convs[l].weight.grad.reshape(-1) for l in range(DEPTH)] +
                     [net.head.weight.grad.reshape(-1)])


def measure(**kw):
    for p in net.parameters():
        p.grad = None
    net.error_relax_grads(xb, tgt, **kw)
    return gvec(net)


# reference: sync, fully converged
ref = measure(n_iter=DEPTH + 6, participation=1.0)

CONDS = {
    "sync":      dict(participation=1.0, stale=False, rand_order=False, stoch=False),
    "partial50": dict(participation=0.5, stale=False, rand_order=False, stoch=True),
    "partial25": dict(participation=0.25, stale=False, rand_order=False, stoch=True),
    "stale":     dict(participation=1.0, stale=True, rand_order=False, stoch=False),
    "worst50":   dict(participation=0.5, stale=True, rand_order=True, stoch=True),
}
out = {"meta": {"depth": DEPTH, "dils": DILS, "RF": RF, "rounds": ROUNDS}, "curves": {}, "round_tax": {}}
print(f"\n{'rounds':>7} | " + " | ".join(f"{c:>9}" for c in CONDS), flush=True)
curves = {c: [] for c in CONDS}
for r in ROUNDS:
    row = []
    for name, c in CONDS.items():
        c = dict(c); stoch = c.pop("stoch")
        trials = N_TRIALS if stoch else 1
        acc = 0.0
        for t in range(trials):
            gen = torch.Generator().manual_seed(100 * t + 3)
            g = measure(n_iter=r, gen=gen, **c)
            acc += F.cosine_similarity(g, ref, dim=0).item()
        v = acc / trials
        curves[name].append(round(v, 4)); row.append(v)
    print(f"{r:>7} | " + " | ".join(f"{v:>9.4f}" for v in row), flush=True)

print("\nround-tax (rounds to cosine>=0.999), vs depth", DEPTH, "and RF", RF, ":")
for name in CONDS:
    rt = next((r for r, v in zip(ROUNDS, curves[name]) if v >= 0.999), None)
    out["round_tax"][name] = rt
    mult = f"{rt/DEPTH:.1f}xL" if rt else "n/a"
    print(f"  {name:>10}: {rt}  (~{mult})")
out["curves"] = curves
print("\nVERDICT: round-tax ~O(depth) (NOT O(depth*RF)) => dilated conv does not inflate the "
      "fabric round-tax; economics carry from the MLP." if (out['round_tax'].get('stale') or 99) <= 2 * DEPTH
      else "\nVERDICT: round-tax exceeds ~2*depth => dilated RF inflates it; note for the economics.")
with open("results/predictive_coding/conv_parallelism.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote results/predictive_coding/conv_parallelism.json")
