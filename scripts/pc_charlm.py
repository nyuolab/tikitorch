"""Does LOCAL tropical PC train a competitive char-LM to backprop quality (adequately
trained)? — the architecture gate done right.

The goal is NOT the cascade (which IS backprop, an ordered single-pass backward) — it is
LOCAL PC: iterative error-relaxation in its ASYNC / partial / stale form. Updates are
neighbour-local and schedule-tolerant, but retain the same directional top-down credit
relation as backprop. The fair question is final LM quality AT CONVERGENCE (not
gradient-exactness, not a short budget): does async-local-PC match backprop on a real
sequence task when adequately trained?

Model: a feedforward CONTEXT char-LM (Bengio-style) = DyadicPCNet residual stack mapping a
C-char one-hot context -> next char. Its DyadicLinear/max-residual internal path is
shift-add-oriented; CE and the optimizer remain open boundaries. It reuses the measured
error-relaxation machinery. No recurrence (the min-plus scan was unstable), no
transformer — this isolates the question "can local PC train an LM" on the architecture
family the cascade/relax paths already handle.

Arms (all trained to convergence): bp_float (quality bar) | bp_dyadic (dyadic fwd, bp)
| relax (sync local PC == cascade gradient) | relax_async (participation 0.5 + stale +
random order — the FABRIC form). Bars: bigram floor; the bp twin. Metric: val bits/char.
Run: PYTHONPATH=. python scripts/pc_charlm.py --epochs 12   (subset+few epochs to smoke)
"""
import sys, os, math, time, json, argparse, urllib.request
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F
from tikitorch.nn.predictive_coding import DyadicPCNet
from pc_scale_mnist import DyadicMLP, FloatMLP

PATH = "/tmp/_tiki_data/tinyshakespeare.txt"
os.makedirs("/tmp/_tiki_data", exist_ok=True)
if not os.path.exists(PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", PATH)


def load(context, subset=None):
    text = open(PATH).read()
    chars = sorted(set(text)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    d = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    if subset:
        d = d[:subset]
    # sliding windows: X[i] = d[i:i+C], y[i] = d[i+C]
    N = len(d) - context
    idx = torch.stack([d[i:i + context] for i in range(N)])
    y = d[context:context + N]
    n = int(0.9 * N)
    return (idx[:n], y[:n]), (idx[n:], y[n:]), V, stoi, d


def onehot(idx_b, V):
    return F.one_hot(idx_b, V).float().reshape(idx_b.size(0), -1)


def iterate_lm(Xidx, y, V, batch, gen, device):
    perm = torch.randperm(Xidx.size(0), generator=gen)
    for i in range(0, Xidx.size(0) - batch, batch):
        bi = perm[i:i + batch]
        yield onehot(Xidx[bi], V).to(device), y[bi].to(device)


@torch.no_grad()
def val_bpc(logits_fn, Xidx, y, V, device, batch=8192):
    tot, n = 0.0, 0
    for i in range(0, Xidx.size(0) - 1, batch):
        xb = onehot(Xidx[i:i + batch], V).to(device)
        yb = y[i:i + batch].to(device)
        tot += F.cross_entropy(logits_fn(xb), yb, reduction="sum").item()
        n += yb.size(0)
    return tot / n / math.log(2)


def bigram_floor(d, V, n_train):
    big = torch.ones(V, V)
    tr = d[:n_train]
    for a, b in zip(tr[:-1].tolist(), tr[1:].tolist()):
        big[a, b] += 1
    big /= big.sum(1, keepdim=True)
    va, vb = d[n_train:-1], d[n_train + 1:]
    return -(big[va, vb].clamp_min(1e-9).log2()).mean().item()


def run_one_lm(mode, dims, tr, va, V, *, seed, epochs, batch, lr, wd, drop, residual, device, log):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    Xtr, ytr = tr; Xva, yva = va
    depth = len(dims) - 1
    t0 = time.time()
    pc = mode in ("relax", "relax_async", "cascade")
    if pc:
        net = DyadicPCNet(dims, residual=residual).to(device)
    else:
        net = (DyadicMLP(dims, residual=residual) if mode == "bp_dyadic"
               else FloatMLP(dims, residual=residual)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)   # wd controls overfit
    rgen = torch.Generator().manual_seed(seed + 99)
    lf = net.predict_logits if pc else net.forward
    best, best_ep = 1e9, -1                                            # EARLY STOPPING: report best val
    for ep in range(epochs):
        for xb, yb in iterate_lm(Xtr, ytr, V, batch, gen, device):
            if drop > 0:
                xb = F.dropout(xb, p=drop, training=True)             # input (one-hot) dropout
            if mode == "cascade":
                net.train_step_cascade(xb, yb, opt, mode="dyadic")
            elif mode == "relax":
                net.train_step_relax(xb, yb, opt, mode="dyadic")
            elif mode == "relax_async":
                net.train_step_relax(xb, yb, opt, n_iter=3 * depth, mode="dyadic",
                                     participation=0.5, stale=True, rand_order=True, gen=rgen)
            else:
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(net(xb), yb).backward()
                opt.step()
        v = val_bpc(lf, Xva, yva, V, device)
        if v < best:
            best, best_ep = v, ep
        if log:
            print(f"  [{mode}] ep{ep} val_bpc {v:.3f} (best {best:.3f}@{best_ep}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return round(best, 4), best_ep, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)        # backprop arms
    ap.add_argument("--pc-lr", type=float, default=None,      # local-PC arms (mult-free updates
                    dest="pc_lr")                            # have a larger effective step -> ~3x smaller lr)
    ap.add_argument("--wd", type=float, default=1e-3)        # weight decay (overfit control)
    ap.add_argument("--drop", type=float, default=0.15)      # input (one-hot) dropout
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=["bp_float", "bp_dyadic", "relax", "relax_async"])
    ap.add_argument("--out", default="results/predictive_coding/charlm_localpc.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tr, va, V, stoi, d = load(args.context, args.subset)
    n_train = tr[0].size(0)
    floor = bigram_floor(d, V, n_train)
    dims = [args.context * V] + [args.hidden] * (args.depth - 1) + [V]
    print(f"chars={V} ctx={args.context} dims={dims} train={n_train} val={va[0].size(0)} "
          f"device={args.device}", flush=True)
    print(f"bigram floor = {floor:.3f} bpc  (val); wd={args.wd} drop={args.drop}; "
          f"reporting BEST val (early-stopping)\n", flush=True)

    pc_lr = args.pc_lr if args.pc_lr is not None else args.lr
    rows = []
    for mode in args.modes:
        mode_lr = pc_lr if mode in ("relax", "relax_async", "cascade") else args.lr
        bpc, best_ep, wall = run_one_lm(mode, dims, tr, va, V, seed=0, epochs=args.epochs,
                                        batch=args.batch, lr=mode_lr, wd=args.wd, drop=args.drop,
                                        residual=True, device=args.device, log=True)
        rows.append({"mode": mode, "val_bpc": bpc, "best_epoch": best_ep, "wall_s": wall,
                     "epochs": args.epochs, "context": args.context, "hidden": args.hidden,
                     "depth": args.depth, "wd": args.wd, "drop": args.drop, "lr": mode_lr})
        print(f"==> {mode}: best val_bpc {bpc} @ep{best_ep} ({wall}s)\n", flush=True)

    print("================ SUMMARY (BEST val bits/char, lower=better) ================", flush=True)
    print(f"  bigram floor        : {floor:.3f}")
    for r in rows:
        print(f"  {r['mode']:<14}: {r['val_bpc']:.3f}  (@ep{r['best_epoch']})")
    bp = next((r["val_bpc"] for r in rows if r["mode"] == "bp_float"), None)
    asy = next((r["val_bpc"] for r in rows if r["mode"] == "relax_async"), None)
    if bp and asy:
        print(f"\n  local-PC-async vs backprop: {asy:.3f} vs {bp:.3f}  "
              f"(gap {asy-bp:+.3f} bpc) — {'MATCH' if asy-bp < 0.10 else 'GAP'}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
