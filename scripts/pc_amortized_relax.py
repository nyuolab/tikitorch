"""Rung-4 revisit: AMORTISED relaxation — does PC hold quality with FEW error-propagation
sweeps per step (warm-started across steps), instead of the full O(L) sweep?

Rung 4 showed PC's per-step cost (R relaxation sweeps + weight-grad pass) makes it
break-even-to-slower than async-backprop on a digital fabric. The lever: don't fully
propagate the error each step — do K << L *out-of-place (Jacobi)* sweeps, WARM-STARTED from
the persisted errors of the previous step, so the credit converges over TRAINING (a
delayed/pipelined backward) rather than within each step. If quality holds at small K, PC's
per-step error-route critical path collapses from O(L) to O(K), and its backward becomes
CHEAPER than backprop's full O(L) backward — flipping Rung 4.

K = L (or a fresh in-place sweep) = exact backprop credit. K small + warm-start = amortised.
K small + COLD (no warm-start) = the control (should degrade, isolating the warm-start).
Run: PYTHONPATH=. python scripts/pc_amortized_relax.py
"""
import os, sys, math, urllib.request
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

PATH = "/tmp/_tiki_data/tinyshakespeare.txt"
os.makedirs("/tmp/_tiki_data", exist_ok=True)
if not os.path.exists(PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", PATH)
text = open(PATH).read()[:400000]
chars = sorted(set(text)); V = len(chars); stoi = {c: i for i, c in enumerate(chars)}
d = torch.tensor([stoi[c] for c in text], dtype=torch.long)
CTX, WIDTH, DEPTH = 12, 256, 6
N = len(d) - CTX
X = torch.stack([d[i:i + CTX] for i in range(N)]); Y = d[CTX:CTX + N]
n = int(0.9 * N)
Xtr, Ytr, Xva, Yva = X[:n], Y[:n], X[n:], Y[n:]
torch.manual_seed(0)


def oh(xb):
    return F.one_hot(xb, V).float().reshape(xb.size(0), -1)


def init_net():
    dims = [CTX * V] + [WIDTH] * (DEPTH - 1) + [V]
    return [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(DEPTH)]


def fwd_cache(net, xb):
    acts, masks = [oh(xb)], []
    h = acts[0]
    for i, L in enumerate(net):
        pre = L(h)
        last = i == len(net) - 1
        m = torch.ones_like(pre) if last else (pre > 0).float()
        h = pre if last else torch.relu(pre)
        masks.append(m); acts.append(h)
    return acts, masks


@torch.no_grad()
def val_bpc(net):
    h = oh(Xva[:8192])
    for i, L in enumerate(net):
        h = L(h) if i == len(net) - 1 else torch.relu(L(h))
    return F.cross_entropy(h, Yva[:8192]).item() / math.log(2)


def train(mode, K, steps=2500, lr=2e-3, batch=128, warm=True):
    torch.manual_seed(0); net = init_net()
    params = [p for L in net for p in L.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(7)
    Dn = len(net)
    E = [None] * (Dn + 1)                                          # persisted per-layer errors (warm-start)
    best = 1e9
    for step in range(steps):
        idx = torch.randint(Xtr.size(0), (batch,), generator=gen)
        acts, masks = fwd_cache(net, Xtr[idx])
        with torch.no_grad():
            eL = torch.softmax(acts[Dn], -1) - F.one_hot(Ytr[idx], V).float()
            if mode == "backprop":
                # exact: in-place top-down sweep (one pass = exact for a triangular system)
                E[Dn] = eL
                for l in reversed(range(1, Dn)):
                    E[l] = (E[l + 1] * masks[l]) @ net[l].weight
            else:  # amortised: K OUT-OF-PLACE (Jacobi) sweeps, warm-started across steps
                if (E[1] is None) or (not warm) or (E[1].shape[0] != batch):
                    E = [torch.zeros(batch, a.shape[1]) for a in acts]   # cold start (or first step)
                E[Dn] = eL
                for _ in range(K):
                    snap = [t.clone() for t in E]
                    for l in range(1, Dn):
                        E[l] = (snap[l + 1] * masks[l]) @ net[l].weight   # uses last sweep's E (Jacobi)
                    E[Dn] = eL
            # local weight grads from the current (possibly partial) errors
            for l in range(Dn):
                delta = E[l + 1] * masks[l]
                net[l].weight.grad = delta.t() @ acts[l] / batch
                net[l].bias.grad = delta.mean(0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            best = min(best, val_bpc(net))
    return round(best, 4)


if __name__ == "__main__":
    print(f"char-LM context-MLP: depth {DEPTH}, width {WIDTH}, ctx {CTX}, V {V}\n", flush=True)
    bp = train("backprop", K=0)
    print(f"  backprop (exact, full O(L) sweep)          : {bp:.3f} bpc", flush=True)
    for K in (DEPTH, 3, 2, 1):
        warm = train("amortised", K=K, warm=True)
        cold = train("amortised", K=K, warm=False)
        tag = "= exact" if K >= DEPTH else f"amortised, {K} sweep(s)/step"
        print(f"  amortised K={K} ({tag:<22}) warm-start {warm:.3f} | cold {cold:.3f} "
              f"| gap-to-bp {warm-bp:+.3f}", flush=True)
    print("\n  Reading: if warm-start holds quality at K<<L (gap-to-bp small) while cold degrades,\n"
          "  the credit amortises over training -> PC's per-step error-route is O(K) not O(L)\n"
          "  -> its backward is cheaper than backprop's full sweep -> the Rung-4 economics flip.", flush=True)
