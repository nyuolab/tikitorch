"""Associative-recall (induction) bake-off: Transformer vs dense gated-SSM vs tropical max-plus SSM.

TASK (the HARD recurrence case = attention's strength):
  Sequences of (key, value) pairs followed by a query key; the model must emit the
  value paired with the matching key. Random-access associative recall.

THREE MODELS, matched small size:
  * TRANSFORMER     : causal softmax attention block(s) + MLP (the reference; O(T) random access).
  * DENSE gated-SSM : Mamba-lite gated LINEAR recurrence h_t = sigmoid(a)*h_{t-1} + b*x_t
                      (per-channel a), gated output + channel MLP. Decay-AVERAGE; canNOT do exact max.
  * TROPICAL SSM    : SAME block but the time-mixer is a MAX-PLUS / morphological recurrence
                      h_t = max(h_{t-1} + decay_dyadic, x_t)  (max/min by parity, MorphSSM-style).
                      All projection weights DYADIC (+-2^k via STE -> shift-add, no multiply).
                      Plus a MAX-PLUS MATCHING read head (devised here) to give the tropical
                      recurrence a fair shot at content routing without softmax.

HONEST HYPOTHESIS: the transformer wins (O(T) random access); recurrences struggle. We test whether
the tropical max-plus matching gives the SSM ANY edge over the dense SSM, or whether recall genuinely
needs attention-augmentation. A clean negative is the valuable, scoping result.

Run:  python scripts/tropical_recall_benchmark.py
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- dyadic STE
def dyadic_quantize(w: torch.Tensor) -> torch.Tensor:
    """Round each weight to the nearest SIGNED power of two (straight-through).
    forward: w -> sign(w) * 2^round(log2|w|) (a bit-shift); backward: identity (STE).
    Canonical copy of tikitorch.nn.dyadic_quantize so this script is standalone.
    """
    mag = w.abs().clamp_min(1e-12)
    q = torch.sign(w) * torch.exp2(torch.log2(mag).round())
    return w + (q - w).detach()


class DyadicLinear(nn.Linear):
    """nn.Linear with weights constrained to signed powers of two (multiplier-free)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, dyadic_quantize(self.weight), self.bias)


# ----------------------------------------------------------------------------- data
class RecallData:
    """Associative recall / induction.

    Vocab layout (ids):
      0                       : PAD (unused in loss except as ignore target on non-query positions)
      1                       : QUERY-marker token Q
      [2 .. 2+K-1]            : KEY symbols   (K of them)
      [2+K .. 2+K+V-1]        : VALUE symbols (V of them)

    A sequence: k1 v1 k2 v2 ... kN vN  Q  kq      -> predict (at the final position) the value vj
    where kj == kq. Keys within a sequence are DISTINCT (so the answer is unique). The query key
    is one of the keys already shown. Loss/accuracy is computed ONLY at the final position.
    """

    def __init__(self, n_pairs=8, n_keys=20, n_vals=20, seed=0):
        self.n_pairs = n_pairs
        self.n_keys = n_keys
        self.n_vals = n_vals
        self.Q = 1
        self.key0 = 2
        self.val0 = 2 + n_keys
        self.vocab = 2 + n_keys + n_vals
        self.T = 2 * n_pairs + 2  # pairs (2 each) + Q + query-key
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, B, device):
        T = self.T
        x = torch.zeros(B, T, dtype=torch.long)
        for b in range(B):
            keys = torch.randperm(self.n_keys, generator=self.g)[: self.n_pairs] + self.key0
            vals = torch.randint(self.n_vals, (self.n_pairs,), generator=self.g) + self.val0
            seq = torch.empty(2 * self.n_pairs, dtype=torch.long)
            seq[0::2] = keys
            seq[1::2] = vals
            qi = torch.randint(self.n_pairs, (1,), generator=self.g).item()
            x[b, : 2 * self.n_pairs] = seq
            x[b, 2 * self.n_pairs] = self.Q
            x[b, 2 * self.n_pairs + 1] = keys[qi]
        # target = value at the final (query-key) position; predicted from that position's output
        y = torch.zeros(B, dtype=torch.long)
        # recompute targets cleanly
        for b in range(B):
            qkey = x[b, -1].item()
            kk = x[b, : 2 * self.n_pairs : 2]
            vv = x[b, 1 : 2 * self.n_pairs : 2]
            j = (kk == qkey).nonzero(as_tuple=True)[0].item()
            y[b] = vv[j].item()
        return x.to(device), y.to(device)


# ----------------------------------------------------------------------------- transformer
class TransformerLM(nn.Module):
    def __init__(self, vocab, T, d=128, n_layers=2, n_heads=4, mlp=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList(
            [TBlock(d, n_heads, mlp) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.T = T

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))  # logits over all positions


class TBlock(nn.Module):
    def __init__(self, d, n_heads, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, h):
        T = h.shape[1]
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), 1)
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=mask, need_weights=False)
        h = h + a
        h = h + self.mlp(self.ln2(h))
        return h


# ----------------------------------------------------------------------------- dense gated SSM (Mamba-lite)
class DenseSSMLayer(nn.Module):
    """Gated LINEAR recurrence:  h_t = sigmoid(a) * h_{t-1} + b_t * x_t,  per-channel a.
    Input-dependent b (selective). Output gated by sigmoid. This is a DECAY-AVERAGE — it
    can NOT compute an exact running max."""

    def __init__(self, d):
        super().__init__()
        self.in_proj = nn.Linear(d, d)
        self.b_proj = nn.Linear(d, d)          # input-dependent input gate (selective)
        self.gate_proj = nn.Linear(d, d)       # output gate
        self.a = nn.Parameter(torch.zeros(d))  # per-channel decay logit -> sigmoid(a) in (0,1)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        v = self.in_proj(x)
        b = torch.sigmoid(self.b_proj(x))
        decay = torch.sigmoid(self.a)[None, :]  # (1,d)
        h = torch.zeros(B, d, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = decay * h + b[:, t] * v[:, t]
            outs.append(h)
        y = torch.stack(outs, 1)
        y = y * torch.sigmoid(self.gate_proj(x))
        return self.out_proj(y)


class SSMBlock(nn.Module):
    def __init__(self, d, mlp, layer_cls):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mix = layer_cls(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, h):
        h = h + self.mix(self.ln1(h))
        h = h + self.mlp(self.ln2(h))
        return h


class DenseSSM(nn.Module):
    def __init__(self, vocab, T, d=128, n_layers=2, mlp=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([SSMBlock(d, mlp, DenseSSMLayer) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


# ----------------------------------------------------------------------------- tropical max-plus SSM (mult-free)
class TropicalSSMLayer(nn.Module):
    """MAX-PLUS / morphological time-mixer (MorphSSM-style), multiplier-free.

    Time-mix per channel (max/min by parity = morphological dilation/erosion with leaky decay):
        even ch:  h_t = max(h_{t-1} + decay_c, x_t)     (running leaky-MAX  -- O(1) state)
        odd  ch:  h_t = min(h_{t-1} - decay_c, x_t)     (running leaky-MIN)
    decay_c is DYADIC (snapped to +-2^k). All projections are DyadicLinear (shift-add, no DSP).
    A linear SSM can only decay-AVERAGE; this carries an EXACT running extreme with decay.

    Plus a MAX-PLUS MATCHING read (devised to give recall a fair shot WITHOUT softmax):
        for the readout we compute a per-position tropical score and gather. See TropicalSSM.
    """

    def __init__(self, d):
        super().__init__()
        self.in_proj = DyadicLinear(d, d)
        self.gate_proj = DyadicLinear(d, d)
        self.out_proj = DyadicLinear(d, d)
        # dyadic decay (additive shift in the max-plus semiring), one per channel
        self.decay = nn.Parameter(torch.full((d,), -1.0))

    def forward(self, x):
        B, T, d = x.shape
        v = self.in_proj(x)
        dec = -F.softplus(dyadic_quantize(self.decay))  # negative additive leak, dyadic
        dec = dec[None, :]                               # (1,d)
        h = torch.full((B, d), -1e9, device=x.device, dtype=x.dtype)
        outs = []
        even = torch.arange(d, device=x.device) % 2 == 0
        for t in range(T):
            cand_max = torch.maximum(h + dec, v[:, t])   # leaky running MAX
            cand_min = torch.minimum(h - dec, v[:, t])   # leaky running MIN
            h = torch.where(even[None, :], cand_max, cand_min)
            outs.append(h)
        y = torch.stack(outs, 1)
        # tropical gate: hard ReLU gate (max(.,0)) -- closure-clean, no sigmoid multiply needed
        g = torch.maximum(self.gate_proj(x), torch.zeros_like(x))
        y = y + g  # additive (max-plus) gating
        return self.out_proj(y)


class TropicalSSM(nn.Module):
    """Tropical SSM with an optional MAX-PLUS MATCHING read head for associative recall.

    match_head=True adds ONE multiplier-free content-routing read (the tropical analogue of
    attention): for the final query position q, score each past position t by the NEGATIVE
    L1 distance between their (dyadic-projected) key reps  s_{q,t} = -||K_q - K_t||_1, then
    HARD-gather (argmax) the value rep V_t. This is the L1/argmax routing that was shown to do
    induction at the softmax ceiling (see MEMORY tropical-attention-induction), here driven by
    the SSM's own hidden state (no softmax, no multiply: |.| + add + max + gather)."""

    def __init__(self, vocab, T, d=128, n_layers=2, mlp=4, match_head=True, soft_train=False,
                 tau=1.0, induction_shift=False, float_route=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([SSMBlock(d, mlp, TropicalSSMLayer) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.match_head = match_head
        self.soft_train = soft_train  # τ-annealed soft routing (lets K learn; deploys as hard argmax)
        self.tau = tau
        # induction_shift: route query->PREVIOUS key by L1, then read the value at position j+1
        # (the value that FOLLOWED the matched key). This is the canonical induction-head read,
        # giving the tropical max-plus matching its fairest shot at associative recall.
        self.induction_shift = induction_shift
        # float_route: DIAGNOSTIC control -- use float (non-dyadic) Q/K/V to isolate whether the
        # failure is the dyadic quantization or the L1/argmax routing structure on the SSM state.
        self.float_route = float_route
        if match_head:
            Lin = nn.Linear if float_route else DyadicLinear
            self.q_proj = Lin(d, d)
            self.k_proj = Lin(d, d)
            self.v_proj = Lin(d, d)
            self.merge = nn.Linear(d, d)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        if self.match_head:
            Q = self.q_proj(h)  # (B,T,d) query reps
            K = self.k_proj(h)  # (B,T,d) key reps
            Vv = self.v_proj(h)
            if self.induction_shift:
                # value that FOLLOWED key j: Vshift[:, j] = Vv[:, j+1]  (last position -> zeros)
                Vread = torch.zeros_like(Vv)
                Vread[:, :-1] = Vv[:, 1:]
            else:
                Vread = Vv
            # tropical L1 routing with a strict causal mask (content-based, no softmax in deploy).
            # score[b,i,j] = -||Q_i - K_j||_1  for j<i.
            diff = (Q[:, :, None, :] - K[:, None, :, :]).abs().sum(-1)  # (B,T,T)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 0)  # j>=i disallowed
            score = -diff.masked_fill(mask, 1e9)
            if self.soft_train and self.training:
                # τ-annealed RELAXATION so Q/K get gradient (argmax is non-diff). TRAIN-time crutch
                # (a softmax) -> deploy/eval uses the hard argmax-gather below (mult-free).
                w = torch.softmax(score / self.tau, dim=-1)  # (B,T,T)
                gathered = torch.einsum("bij,bjd->bid", w, Vread)
            else:
                idx = score.argmax(-1)  # (B,T) hard route -- argmax is the tropical select (deploy)
                gathered = torch.gather(Vread, 1, idx[:, :, None].expand(-1, -1, Vread.shape[-1]))
            h = h + self.merge(gathered)
        return self.head(h)


# ----------------------------------------------------------------------------- train / eval
def run_model(name, model, data, device, steps, B, lr, eval_every, log):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    model.train()
    anneal = getattr(model, "soft_train", False)
    t0 = time.time()
    for step in range(1, steps + 1):
        if anneal:
            # τ from 1.0 -> 0.02 over training: soft routing hardens toward the deploy argmax
            frac = step / steps
            model.tau = max(0.02, 1.0 * (0.02 / 1.0) ** frac)
        x, y = data.batch(B, device)
        logits = model(x)[:, -1, :]  # predict at final (query) position
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            acc = evaluate(model, data, device, B, n_batches=8)
            log(f"  [{name}] step {step:5d}  loss {loss.item():.4f}  recall_acc {acc:.4f}")
    final_acc = evaluate(model, data, device, B, n_batches=20)
    dt = time.time() - t0
    return {"name": name, "params": n_params, "acc": final_acc, "time_s": dt}


@torch.no_grad()
def evaluate(model, data, device, B, n_batches=20):
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y = data.batch(B, device)
        pred = model(x)[:, -1, :].argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n_pairs", type=int, default=8)
    ap.add_argument("--n_keys", type=int, default=20)
    ap.add_argument("--n_vals", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=str, default="", help="comma-separated subset of model names")
    args = ap.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    logs = []
    def log(s):
        print(s, flush=True)
        logs.append(s)

    data = RecallData(n_pairs=args.n_pairs, n_keys=args.n_keys, n_vals=args.n_vals, seed=args.seed)
    log(f"TASK associative-recall: T={data.T} vocab={data.vocab} n_pairs={args.n_pairs} "
        f"keys={args.n_keys} vals={args.n_vals} | chance={1.0/args.n_vals:.4f}")
    log(f"CONFIG d={args.d} layers={args.layers} steps={args.steps} batch={args.batch} "
        f"lr={args.lr} device={device}")

    results = []
    builders = [
        ("transformer", lambda: TransformerLM(data.vocab, data.T, d=args.d, n_layers=args.layers)),
        ("dense_ssm",   lambda: DenseSSM(data.vocab, data.T, d=args.d, n_layers=args.layers)),
        ("tropical_ssm",lambda: TropicalSSM(data.vocab, data.T, d=args.d, n_layers=args.layers, match_head=True)),
        ("tropical_ssm_nomatch", lambda: TropicalSSM(data.vocab, data.T, d=args.d, n_layers=args.layers, match_head=False)),
        ("tropical_ssm_soft", lambda: TropicalSSM(data.vocab, data.T, d=args.d, n_layers=args.layers, match_head=True, soft_train=True)),
        ("tropical_ssm_induction", lambda: TropicalSSM(data.vocab, data.T, d=args.d, n_layers=args.layers, match_head=True, soft_train=True, induction_shift=True)),
        ("tropical_ssm_induction_hard", lambda: TropicalSSM(data.vocab, data.T, d=args.d, n_layers=args.layers, match_head=True, soft_train=False, induction_shift=True)),
        ("tropical_ssm_induction_float", lambda: TropicalSSM(data.vocab, data.T, d=args.d, n_layers=args.layers, match_head=True, soft_train=True, induction_shift=True, float_route=True)),
    ]
    if args.only:
        keep = set(args.only.split(","))
        builders = [b for b in builders if b[0] in keep]
    for name, build in builders:
        torch.manual_seed(args.seed)  # identical init seed regime
        log(f"\n=== {name} ===")
        m = build()
        r = run_model(name, m, data, device, args.steps, args.batch, args.lr, args.eval_every, log)
        log(f"  RESULT {name}: params={r['params']:,} final_recall_acc={r['acc']:.4f} time={r['time_s']:.1f}s")
        results.append(r)

    log("\n===== SUMMARY (associative recall) =====")
    log(f"{'model':24s} {'params':>10s} {'recall_acc':>12s}  {'time_s':>8s}")
    for r in results:
        log(f"{r['name']:24s} {r['params']:>10,d} {r['acc']:>12.4f}  {r['time_s']:>8.1f}")
    log(f"chance = {1.0/args.n_vals:.4f}")


if __name__ == "__main__":
    main()
