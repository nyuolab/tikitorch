"""Rung 2: LOCAL async tropical PC on a competitive CONV char-LM, vs backprop.

The 1D causal-conv twin of pc_conv_relax.py (CIFAR), for char-LM. A stack of CAUSAL
DILATED dyadic convs (±2^k shift-add) + ReLU, all-position next-char head. Credit by
frozen-routing error-relaxation: the error route A_l^T is conv_transpose1d with the cached
ReLU mask and dyadic weights (an EXACT linear transpose, multiplier-free); weight grad is
conv1d_weight on dyadic operands. The async loop (participation / stale / rand_order) is
the local/fabric form — NOT the cascade.

Decision (from the design workflow): STANDARD dyadic conv, not morphological — the dyadic
conv's adjoint is an exact linear transpose the async machinery already handles (proven in
pc_conv_relax.py), whereas morph conv's argmax-scatter adjoint hits the smoothness wall.

Causal conv via left-pad: out = conv1d(pad_left(h, P), dq(W), dilation=d), P=(K-1)*d  ->
length L, causal. Adjoint: conv_transpose1d(e, dq(W), dilation=d)[..., P:]. All-position
prediction (every position predicts its next char) = far more sample-efficient than
last-position (anti-undertraining). Run the gradient-cosine GATE first:
  PYTHONPATH=. python scripts/pc_charlm_conv.py --gradcheck
"""
from __future__ import annotations
import sys, os, math, time, json, argparse, urllib.request
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from tikitorch.nn import dyadic_quantize as dq, DyadicLinear

PATH = "/tmp/_tiki_data/tinyshakespeare.txt"
os.makedirs("/tmp/_tiki_data", exist_ok=True)
if not os.path.exists(PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", PATH)


class DyadicConv1d(nn.Conv1d):
    """nn.Conv1d with ±2^k weights (shift-add forward, zero DSP). 1D twin of DyadicConv2d."""
    _zero_dsp_layer = True
    def forward(self, x):
        return self._conv_forward(x, dq(self.weight), self.bias)


def stoch_dyadic(w):
    """UNBIASED stochastic round to a signed power of two (±2^k): rounds |w| up/down to the
    nearest two powers with linear probability, so E[q]=w (deterministic dq is biased -> a
    weight-grad noise FLOOR; stochastic averages it out over training). Still mult-free
    (±2^k -> product is a shift)."""
    mag = w.abs().clamp_min(1e-20)
    e = torch.floor(torch.log2(mag))
    lo = torch.exp2(e); hi = torch.exp2(e + 1)
    p = (mag - lo) / (hi - lo)                                # P(round up), unbiased in linear space
    up = (torch.rand_like(mag) < p).float()
    return torch.sign(w) * (lo * (1 - up) + hi * up)


class DyadicConvPC1d(nn.Module):
    """Causal dilated dyadic-conv char-LM trained by local async PC (all-position)."""
    def __init__(self, vocab, width=128, depth=6, kernel=5, dilations=None, float_conv=False):
        super().__init__()
        self.vocab, self.width, self.depth, self.K = vocab, width, depth, kernel
        self.dil = dilations or [2 ** i for i in range(depth)]
        assert len(self.dil) == depth
        self.embed = nn.Embedding(vocab, width)
        Conv = nn.Conv1d if float_conv else DyadicConv1d
        self.convs = nn.ModuleList(
            [Conv(width, width, kernel, dilation=self.dil[l], bias=True) for l in range(depth)])
        self.head = (nn.Linear(width, vocab) if float_conv else DyadicLinear(width, vocab))

    def _pad(self, l):
        return (self.K - 1) * self.dil[l]                      # causal left-pad / transpose drop

    def _feat(self, ids):
        h = self.embed(ids).transpose(1, 2)                    # (B, W, L)
        for l, conv in enumerate(self.convs):
            h = torch.relu(conv(F.pad(h, (self._pad(l), 0))))  # causal
        return h                                               # (B, W, L)

    def forward(self, ids):                                    # autograd path (bp arms)
        f = self._feat(ids).transpose(1, 2)                    # (B, L, W)
        return self.head(f)                                    # (B, L, V)

    @torch.no_grad()
    def predict_logits(self, ids):
        return self.forward(ids)

    # ---- frozen-routing cache ----
    @torch.no_grad()
    def _forward_cache(self, ids):
        h = self.embed(ids).transpose(1, 2)                    # (B, W, L)
        acts, masks = [h], []
        for l, conv in enumerate(self.convs):
            pre = conv(F.pad(h, (self._pad(l), 0)))
            m = (pre > 0).float()
            h = pre * m
            masks.append(m); acts.append(h)
        return acts, masks                                     # acts[L] = top feature (B,W,L)

    @torch.no_grad()
    def error_relax_grads(self, ids, target, n_iter=None, participation=1.0,
                          stale=False, rand_order=False, gen=None, quant=True, stoch=False):
        """target: (B, L, V) one-hot of next char. Sets .grad on convs + head.
        quant=True -> multiplier-free (±2^k operands); quant=False -> exact adjoint (for the gate).
        stoch=True -> UNBIASED stochastic dyadic rounding of the weight-grad activation (averages
        out the deterministic-rounding noise floor); still mult-free."""
        q = dq if quant else (lambda t: t)
        qw = stoch_dyadic if (quant and stoch) else q            # weight-grad activation quantizer
        acts, masks = self._forward_cache(ids)
        D = self.depth
        if n_iter is None:
            n_iter = D + 4
        feat = acts[D].transpose(1, 2)                         # (B, L, W)
        B, L, W = feat.shape
        logits = self.head(feat)                               # (B, L, V)
        eL = (torch.softmax(logits, -1) - target)              # CE top-error, per position
        flatE = eL.reshape(B * L, -1)
        flatF = feat.reshape(B * L, W)
        # SINGLE-operand quant: error REAL x activation ±2^k = a shift (mult-free), exact in
        # the error -> much less lossy than quantizing BOTH operands (the +0.23-bpc cause).
        self.head.weight.grad = flatE.transpose(0, 1) @ qw(flatF)     # sum over (B,L)
        self.head.bias.grad = flatE.sum(0)
        e_feat = (eL @ q(self.head.weight)).transpose(1, 2)    # (B, W, L) per-position feat error
        E = [torch.zeros_like(a) for a in acts]
        E[D] = e_feat
        free = list(reversed(range(1, D)))
        for _ in range(n_iter):
            snap = [t.clone() for t in E] if stale else E
            order = (free if not rand_order else
                     [free[i] for i in torch.randperm(len(free), generator=gen).tolist()])
            for l in order:
                if participation < 1.0 and torch.rand(1, generator=gen).item() >= participation:
                    continue
                d = snap[l + 1] * masks[l]
                e = F.conv_transpose1d(d, q(self.convs[l].weight), dilation=self.dil[l])
                E[l] = e[..., self._pad(l):]                   # drop the left-pad adjoint
            E[D] = e_feat
        for l in range(D):
            delta = E[l + 1] * masks[l]                              # error REAL (kept exact)
            P = self._pad(l)
            self.convs[l].weight.grad = torch.nn.grad.conv1d_weight(  # act ±2^k x delta real = shift
                F.pad(qw(acts[l]), (P, 0)), self.convs[l].weight.shape, delta,
                dilation=self.dil[l])
            self.convs[l].bias.grad = delta.sum(dim=(0, 2))

    def train_step_relax(self, ids, y, opt, n_iter=None, **akw):
        target = F.one_hot(y, self.vocab).float()              # (B, L, V)
        opt.zero_grad(set_to_none=True)
        self.error_relax_grads(ids, target, n_iter=n_iter, **akw)
        opt.step()


# ---------------- the cheapest-first GATE: adjoint vs backprop, run BEFORE training ----------------
def _cos_table(net, ids, y, V, quant, thr):
    net.zero_grad(); F.cross_entropy(net(ids).reshape(-1, V), y.reshape(-1)).backward()
    bp = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
    for p in net.parameters():
        p.grad = None
    net.error_relax_grads(ids, F.one_hot(y, V).float(), n_iter=net.depth + 4,
                          participation=1.0, quant=quant)
    ok = True
    for n, p in net.named_parameters():
        if p.grad is None or n.startswith("embed"):
            continue
        c = F.cosine_similarity(bp[n].flatten(), p.grad.flatten(), dim=0).item()
        ok = ok and c > thr
        print(f"  {n:26s} cos={c:.4f}  {'OK' if c > thr else 'FAIL'}")
    return ok


def gradcheck(seq=256):
    V = 65
    # the high-dilation top conv needs seq >= receptive field (1+(K-1)*sum(dil)); use seq=256
    dil = [1, 2, 4, 8, 16, 32]
    rf = 1 + (5 - 1) * sum(dil)
    print(f"receptive field = {rf}; testing at seq={seq} (must be >= RF so top conv isn't degenerate)\n")
    torch.manual_seed(0)
    ids = torch.randint(0, V, (8, seq)); y = torch.randint(0, V, (8, seq))
    print("[1] EXACT adjoint check — FLOAT net, no quantization (expect cos ~1.0 = adjoint correct):")
    fnet = DyadicConvPC1d(vocab=V, width=64, depth=6, kernel=5, dilations=dil, float_conv=True)
    ok_float = _cos_table(fnet, ids, y, V, quant=False, thr=0.999)
    print(f"  -> adjoint {'CORRECT' if ok_float else 'WRONG (fix causal alignment)'}\n")
    print("[2] DYADIC training signal — ±2^k operands (expect ~0.95+ = quantization, trains fine):")
    torch.manual_seed(0)
    dnet = DyadicConvPC1d(vocab=V, width=64, depth=6, kernel=5, dilations=dil)
    ok_dy = _cos_table(dnet, ids, y, V, quant=True, thr=0.90)
    print(f"\nGATE: adjoint {'PASS' if ok_float else 'FAIL'} | dyadic-signal {'PASS' if ok_dy else 'check'}")
    return ok_float


# ---------------- training harness (all-position conv char-LM) ----------------
def load(seq, subset=None, stride=None):
    """Sliding (overlapping) windows: stride < seq gives many more training samples
    (non-overlapping seq-256 windows undersample badly -> ~34 steps/epoch)."""
    text = open(PATH).read()
    chars = sorted(set(text)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    d = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    if subset:
        d = d[:subset]
    stride = stride or seq
    starts = list(range(0, len(d) - seq - 1, stride))
    X = torch.stack([d[s:s + seq] for s in starts])              # (N, seq)
    Y = torch.stack([d[s + 1:s + seq + 1] for s in starts])      # (N, seq) next-char
    n = int(0.9 * len(starts))
    split_char = starts[n]                                       # char index splitting train/val
    return (X[:n], Y[:n]), (X[n:], Y[n:]), V, d, split_char


def iterate(X, Y, batch, gen, device):
    perm = torch.randperm(X.size(0), generator=gen)
    for i in range(0, X.size(0) - batch + 1, batch):
        bi = perm[i:i + batch]
        yield X[bi].to(device), Y[bi].to(device)


@torch.no_grad()
def val_bpc(net, X, Y, V, device, batch=64):
    tot, n = 0.0, 0
    for i in range(0, X.size(0), batch):
        xb, yb = X[i:i + batch].to(device), Y[i:i + batch].to(device)
        tot += F.cross_entropy(net.predict_logits(xb).reshape(-1, V), yb.reshape(-1),
                               reduction="sum").item()
        n += yb.numel()
    return tot / n / math.log(2)


def bigram_floor(d, V, n_train):
    big = torch.ones(V, V)
    tr = d[:n_train]
    for a, b in zip(tr[:-1].tolist(), tr[1:].tolist()):
        big[a, b] += 1
    big /= big.sum(1, keepdim=True)
    va, vb = d[n_train:-1], d[n_train + 1:]
    return -(big[va, vb].clamp_min(1e-9).log2()).mean().item()


def run_one(mode, V, *, seq, width, depth, kernel, dils, tr, va, d_full, n_train,
            seed, epochs, batch, lr, wd, drop, device, pc_niter=None, stoch=False):
    torch.manual_seed(seed)
    net = DyadicConvPC1d(V, width, depth, kernel, dils, float_conv=(mode == "bp_float")).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    warmup = max(1, epochs // 20)                                  # warmup + cosine decay:
    def _lr(ep):                                                  # high peak lr (fast), decays
        if ep < warmup:                                          # to ~0 (no late early-peak-degrade)
            return (ep + 1) / warmup
        p = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    gen = torch.Generator().manual_seed(seed); rgen = torch.Generator().manual_seed(seed + 99)
    best, bep = 1e9, -1; t0 = time.time()
    Xtr, Ytr = tr
    for ep in range(epochs):
        for xb, yb in iterate(Xtr, Ytr, batch, gen, device):
            if mode == "relax":
                net.train_step_relax(xb, yb, opt, stoch=stoch)
            elif mode == "relax_exact":                          # DIAGNOSTIC: local form, REAL grad
                net.train_step_relax(xb, yb, opt, quant=False)   # (not mult-free) — isolates form vs quant
            elif mode == "relax_async":
                net.train_step_relax(xb, yb, opt, n_iter=(pc_niter or 3 * depth),
                                     participation=0.5, stale=True, rand_order=True, gen=rgen, stoch=stoch)
            else:
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(net(xb).reshape(-1, V), yb.reshape(-1)).backward()
                opt.step()
        v = val_bpc(net, va[0], va[1], V, device)
        if v < best:
            best, bep = v, ep
        print(f"  [{mode}] ep{ep} val_bpc {v:.3f} (best {best:.3f}@{bep}) lr={sched.get_last_lr()[0]:.1e} "
              f"({time.time()-t0:.0f}s)", flush=True)
        sched.step()
    return round(best, 4), bep, round(time.time() - t0, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--stride", type=int, default=None)      # sliding-window stride (<seq = more samples)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--pc-lr", type=float, default=5e-4, dest="pc_lr")
    ap.add_argument("--pc-niter", type=int, default=None, dest="pc_niter")  # async settling rounds (dilated conv needs more)
    ap.add_argument("--pc-stoch", action="store_true", dest="pc_stoch")     # unbiased stochastic weight-grad rounding
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--modes", nargs="+", default=["bp_float", "bp_dyadic", "relax", "relax_async"])
    ap.add_argument("--out", default="manuscript/results/predictive_coding/charlm_conv_localpc.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = ap.parse_known_args()
    if args.gradcheck:
        gradcheck(); sys.exit(0)

    dils = [2 ** i for i in range(args.depth)]
    rf = 1 + (args.kernel - 1) * sum(dils)
    tr, va, V, d_full, n_train = load(args.seq, args.subset, args.stride)
    floor = bigram_floor(d_full, V, n_train)
    print(f"conv char-LM: V={V} seq={args.seq} width={args.width} depth={args.depth} "
          f"dils={dils} RF={rf} train_win={tr[0].size(0)} val_win={va[0].size(0)} dev={args.device}")
    print(f"bigram floor={floor:.3f} bpc | bp lr={args.lr} pc lr={args.pc_lr}\n", flush=True)
    rows = []
    for mode in args.modes:
        mlr = args.pc_lr if mode in ("relax", "relax_async", "relax_exact") else args.lr
        b, bep, wall = run_one(mode, V, seq=args.seq, width=args.width, depth=args.depth,
                               kernel=args.kernel, dils=dils, tr=tr, va=va, d_full=d_full,
                               n_train=n_train, seed=0, epochs=args.epochs, batch=args.batch,
                               lr=mlr, wd=args.wd, drop=args.drop, device=args.device,
                               pc_niter=args.pc_niter, stoch=args.pc_stoch)
        rows.append({"mode": mode, "val_bpc": b, "best_epoch": bep, "wall_s": wall,
                     "seq": args.seq, "width": args.width, "depth": args.depth, "lr": mlr})
        print(f"==> {mode}: best {b} @ep{bep} ({wall}s)\n", flush=True)
    print("=========== SUMMARY (val bits/char) ===========")
    print(f"  bigram floor : {floor:.3f}")
    for r in rows:
        print(f"  {r['mode']:<13}: {r['val_bpc']:.3f} @ep{r['best_epoch']}")
    bpd = next((r["val_bpc"] for r in rows if r["mode"] == "bp_dyadic"), None)
    asy = next((r["val_bpc"] for r in rows if r["mode"] == "relax_async"), None)
    if bpd and asy:
        print(f"\n  local-PC-async vs bp_dyadic: {asy:.3f} vs {bpd:.3f} (gap {asy-bpd:+.3f}) "
              f"-- {'MATCH' if asy-bpd < 0.05 else 'GAP'}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}", flush=True)
