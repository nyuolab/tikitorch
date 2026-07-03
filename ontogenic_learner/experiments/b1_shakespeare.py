"""B1-Shakespeare — absolute LM quality bars + generation for the ontogenic learner
(pre-registered in ../BENCHMARKS.md). RAW TinyShakespeare (full char vocab, comparable to
pc_charlm/golden bars), ctx 16, ~4.4M-param DyadicPCNet context-LM.

Bars: bigram floor · i.i.d. ceiling (3 epochs cascade). Stream arms (ONE sequential pass,
1500-update budget): stream_cascade · stream_rehearsal · stream_two_speed. Plus 300-char
generations.

Run (repo root): PYTHONPATH=. python ontogenic_learner/experiments/b1_shakespeare.py [--smoke]
"""
import argparse
import copy
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import two_speed_learner as tsl

import torch
import torch.nn.functional as F
from tikitorch.nn.predictive_coding import DyadicPCNet

OUT = os.path.join(tsl.REPO, "ontogenic_learner/results/benchmarks")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    text = open(os.path.join(tsl.REPO, "data/tinyshakespeare.txt")).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    d = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(d))
    return d[:n], d[n:], len(chars), itos


def windows(data, ctx, start, count, stride=1):
    xs, ys, i = [], [], start
    for _ in range(count):
        if i + ctx + 1 > len(data):
            break
        xs.append(data[i:i + ctx]); ys.append(data[i + ctx]); i += stride
    return torch.stack(xs), torch.stack(ys)


def onehot(xi, V):
    return F.one_hot(xi, V).float().reshape(xi.size(0), -1)


@torch.no_grad()
def val_bpc(net, va, ctx, V, n_win=6000):
    X, y = windows(va, ctx, 0, n_win, 3)
    tot = 0.0
    for i in range(0, X.size(0), 4096):
        tot += F.cross_entropy(net.predict_logits(onehot(X[i:i+4096], V).to(DEV)),
                               y[i:i+4096].to(DEV), reduction="sum").item()
    return tot / X.size(0) / math.log(2)


def bigram_floor(tr, va, V):
    big = torch.ones(V, V)
    for a, b in zip(tr[:-1].tolist(), tr[1:].tolist()):
        big[a, b] += 1
    big /= big.sum(1, keepdim=True)
    return -(big[va[:-1], va[1:]].clamp_min(1e-9).log2()).mean().item()


@torch.no_grad()
def sample(net, seed_text, itos, stoi_data, ctx, V, n=300, temp=0.8, rng=None):
    idx = stoi_data.tolist()
    out = list(idx)
    for _ in range(n):
        x = torch.tensor(out[-ctx:], dtype=torch.long).unsqueeze(0)
        logits = net.predict_logits(onehot(x, V).to(DEV))[0] / temp
        p = F.softmax(logits, -1)
        out.append(torch.multinomial(p.cpu(), 1, generator=rng).item())
    return "".join(itos[i] for i in out[len(idx):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--ceiling-epochs", type=int, default=3, dest="ceiling_epochs")
    ap.add_argument("--gate-z", type=float, default=0.25, dest="gate_z")
    ap.add_argument("--sleep-every", type=int, default=300, dest="sleep_every")
    ap.add_argument("--sleep-updates", type=int, default=15, dest="sleep_updates")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.budget, args.ceiling_epochs, args.seeds = 150, 1, [0]

    tr, va, V, itos = load()
    ctx, B = args.ctx, args.bs
    dims = [ctx * V] + [args.hidden] * (args.depth - 1) + [V]
    n_stream = (len(tr) - ctx) // B                      # one sequential pass
    if args.smoke:
        n_stream = min(n_stream, 600)
    floor = bigram_floor(tr, va, V)
    os.makedirs(OUT, exist_ok=True)
    log, t0 = [], time.time()
    print(f"vocab={V} ctx={ctx} params~{sum([dims[i]*dims[i+1] for i in range(len(dims)-1)])/1e6:.1f}M "
          f"stream={n_stream} batches | bigram floor={floor:.3f} bpc | dev={DEV}", flush=True)

    for seed in args.seeds:
        torch.manual_seed(seed)
        base = DyadicPCNet(dims, residual=True, gamma=0.0625).to(DEV)
        results = {}

        # ---- i.i.d. ceiling: cascade, random batches, ceiling_epochs "epochs"
        net = copy.deepcopy(base)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        g = torch.Generator().manual_seed(seed)
        n_ceil = args.ceiling_epochs * n_stream
        for step in range(n_ceil):
            i = torch.randint(len(tr) - ctx * B - B - 1, (1,), generator=g).item()
            xi, yi = windows(tr, ctx, i, B, 1)
            net.train_step_cascade(onehot(xi, V).to(DEV), yi.to(DEV), opt, mode="dyadic")
        results["iid_ceiling"] = (val_bpc(net, va, ctx, V), n_ceil, net)

        # ---- stream arms: ONE sequential pass, budget updates
        for name in ["stream_cascade", "stream_rehearsal", "stream_two_speed"]:
            net = copy.deepcopy(base)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            opt_s = torch.optim.Adam(net.parameters(), lr=args.lr * 0.3)
            res = tsl.Reservoir(4096, seed + hash(name) % 997)
            slots = set(torch.linspace(0, n_stream - 1, args.budget).long().tolist())
            mu = var = None
            upd = n_fast = 0
            for step in range(n_stream):
                xi, yi = windows(tr, ctx, step * B, B, 1)
                if xi.size(0) < B // 2:
                    break
                xb, yb = onehot(xi, V).to(DEV), yi.to(DEV)
                if name == "stream_cascade":
                    if step in slots:
                        net.train_step_cascade(xb, yb, opt, mode="dyadic"); upd += 1
                elif name == "stream_rehearsal":
                    res.insert(xi, yi)
                    if step in slots:
                        rx, ry = res.sample(B // 4, weighted=False)
                        if rx is not None:
                            xb = torch.cat([onehot(xi[:B - rx.size(0)], V).to(DEV),
                                            onehot(rx, V).to(DEV)])
                            yb = torch.cat([yi[:B - rx.size(0)].to(DEV), ry.to(DEV)])
                        net.train_step_cascade(xb, yb, opt, mode="dyadic"); upd += 1
                else:
                    with torch.no_grad():
                        loss = F.cross_entropy(net.predict_logits(xb), yb).item()
                    if mu is None:
                        mu, var, z = loss, 1e-2, 0.0
                    else:
                        z = (loss - mu) / math.sqrt(var + 1e-8)
                        mu = 0.99 * mu + 0.01 * loss
                        var = 0.99 * var + 0.01 * (loss - mu) ** 2
                    if (z > args.gate_z or step < 20) and n_fast < int(args.budget * .75):
                        net.train_step_cascade(xb, yb, opt, mode="dyadic")
                        res.insert(xi, yi, prio=max(z, 0.1)); upd += 1; n_fast += 1
                    if step > 0 and step % args.sleep_every == 0:
                        for _ in range(args.sleep_updates):
                            rx, ry = res.sample(B, weighted=True)
                            if rx is not None:
                                net.train_step_cascade(onehot(rx, V).to(DEV),
                                                       ry.to(DEV), opt_s, mode="dyadic")
                                upd += 1
            results[name] = (val_bpc(net, va, ctx, V), upd, net)

        print(f"----- seed {seed} -----", flush=True)
        for name, (bpc, upd, _) in results.items():
            log.append({"seed": seed, "arm": name, "val_bpc": round(bpc, 4),
                        "updates": upd})
            print(f"  {name:>18}: val_bpc={bpc:.3f} (updates={upd})", flush=True)
        if seed == args.seeds[0]:                        # generation showcase once
            rng = torch.Generator().manual_seed(1234)
            seed_idx = tr[:ctx]
            for name in ["iid_ceiling", "stream_two_speed"]:
                txt = sample(results[name][2], "", itos, seed_idx, ctx, V, rng=rng)
                log.append({"arm": name, "event": "sample", "text": txt})
                print(f"\n--- {name} sample ---\n{txt}\n", flush=True)

    tag = "smoke" if args.smoke else "full"
    log.append({"event": "bars", "bigram_floor_bpc": round(floor, 4),
                "repo_bars": {"bf16_golden_transformer_bpb": 3.07,
                              "nanogpt_class_bpc": "~2.0-2.1"}})
    with open(os.path.join(OUT, f"b1_shakespeare_{tag}.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote b1_shakespeare_{tag}.jsonl ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
