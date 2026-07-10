"""G2 gate: recurrence stability + recurrence-earns-its-keep synthetics.

Pre-registered in results/pc_convrnn/PREREG.md.

2a — drift probe (d=256, L=7 conv-RNN, 20k-char stream, CPU):
  * learned-lambda cell: state must obey u_t <= h_t <= running-max(u); drift
    slope of the state mean < 1e-4/step over the last 10k positions; carry-win
    fraction bounded away from 1.
  * lambda=0 control (ratchet): h = running max — no forgetting; carry-win -> 1,
    state pins to the running extremum (the monotone-accumulation pathology).
  * negative-leak control (growth, lam=-0.05): tropical spectral radius > 0 —
    linear drift, the theorem's positive control for the measured min-plus RNN
    kill (state -> extremum, CHARLM_LOCAL_PC.md).

2b — synthetic battery (d=64, L=2, T=256, conv RF = 1+6*(1+2) = 19, CPU):
  * running-max: target[t] = running max of value tokens (recurrence-native).
  * selective copy (THE KILL GATE): marker M, value v at i+1, query Q at
    i+1+delay, delay in {32, 64, 128} >> RF; predict v at the query position.
    Pass: >95% at delay 64 where the norecur control is at chance (~6%).
  * induction: A B ... A -> B at gaps {32, 64} > RF.
  Arms: convrnn vs convrnn-norecur, 3 seeds, BP AdamW + tau-anneal, masked CE.

Usage::

    python scripts/pc_convrnn_stability.py --part 2a
    python scripts/pc_convrnn_stability.py --part 2b --tasks copy runmax induction
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pc_convrnn_charlm import ConvRNNLM, TropicalLeakyState  # noqa: E402
from nanogpt_shakespeare import CharData, tau_at              # noqa: E402

VOCAB = 64
VAL_LO, VAL_HI = 1, 17          # value tokens 1..16
MARK, QUERY = 17, 18
FILL_LO, FILL_HI = 20, VOCAB


# ---------------------------------------------------------------------------
# 2a — drift probe
# ---------------------------------------------------------------------------

class _FixedLam(nn.Module):
    """Monkeypatch target: forces lam() to a constant (0 = ratchet, <0 = growth)."""

    def __init__(self, scan: TropicalLeakyState, value: float):
        super().__init__()
        self.scan, self.value = scan, value

    def apply(self):
        c = self.scan.channels
        v = torch.full((c,), self.value)
        self.scan.lam = lambda: v


@torch.no_grad()
def drift_probe(args, out_path):
    text = (REPO / "data" / "tinyshakespeare.txt").read_text()[: args.stream + 1]
    chars = sorted(set(text))
    ids = torch.tensor([chars.index(c) for c in text], dtype=torch.long)[None]

    rows = []
    for variant, lam_val in [("learned", None), ("ratchet_lam0", 0.0),
                             ("growth_negleak", -0.05)]:
        torch.manual_seed(0)
        model = ConvRNNLM(len(chars), d_model=256, n_layers=7, dropout=0.0,
                          tau=0.0, recurrent=True)
        model.eval()
        model.set_tau(0.0)
        if lam_val is not None:
            for layer in model.layers:
                _FixedLam(layer.scan, lam_val).apply()

        # capture per-layer scan inputs/outputs on the long stream
        stats = []
        hooks = []

        def mk_hook(store):
            def hook(mod, inp, out):
                u, h = inp[0], out
                run_max = torch.cummax(u, dim=-1).values
                lower_ok = bool((h >= u - 1e-4).all())
                upper_ok = bool((h <= run_max + 1e-4).all())
                mean_t = h.mean(dim=(0, 1))                    # (T,)
                half = mean_t.shape[0] // 2
                t_idx = torch.arange(half, mean_t.shape[0], dtype=torch.float)
                seg = mean_t[half:]
                slope = float(((t_idx - t_idx.mean()) * (seg - seg.mean())).sum()
                              / ((t_idx - t_idx.mean()) ** 2).sum())
                carry = float((h > u + 1e-6).float().mean())
                store.append(dict(lower_ok=lower_ok, upper_ok=upper_ok,
                                  drift_slope=slope, carry_frac=carry,
                                  h_mean=float(h.mean()), h_max=float(h.max()),
                                  h_min=float(h.min())))
            return hook

        for layer in model.layers:
            store = []
            stats.append(store)
            hooks.append(layer.scan.register_forward_hook(mk_hook(store)))
        model(ids)
        for hk in hooks:
            hk.remove()

        for li, store in enumerate(stats):
            s = store[0]
            rows.append(dict(gate="G2a", variant=variant, layer=li, **s))
        worst_slope = max(abs(s[0]["drift_slope"]) for s in stats)
        carry_max = max(s[0]["carry_frac"] for s in stats)
        hmax = max(s[0]["h_max"] for s in stats)
        print(f"[G2a] {variant:16s}: worst |drift slope|={worst_slope:.2e}/step  "
              f"max carry-frac={carry_max:.3f}  h_max={hmax:.1f}  "
              f"bounds_ok={all(s[0]['lower_ok'] and s[0]['upper_ok'] for s in stats) if variant == 'learned' else 'n/a'}")

    with out_path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    learned = [r for r in rows if r["variant"] == "learned"]
    ok = all(r["lower_ok"] and r["upper_ok"] for r in learned) and \
        all(abs(r["drift_slope"]) < 1e-4 for r in learned)
    growth = [r for r in rows if r["variant"] == "growth_negleak"]
    control_fired = any(r["h_max"] > 50 or abs(r["drift_slope"]) > 1e-3
                        for r in growth)
    print(f"[G2a] learned-lambda: {'PASS' if ok else 'FAIL'}; "
          f"growth control reproduces drift: {control_fired}")
    return ok and control_fired


# ---------------------------------------------------------------------------
# 2b — synthetic battery
# ---------------------------------------------------------------------------

def gen_batch(task, B, T, gen, delays):
    x = torch.randint(FILL_LO, FILL_HI, (B, T), generator=gen)
    tgt = torch.full((B, T), -100, dtype=torch.long)          # CE ignore_index
    if task == "runmax":
        x = torch.randint(VAL_LO, VAL_HI, (B, T), generator=gen)
        tgt = torch.cummax(x, dim=1).values
        meta = torch.zeros(B, dtype=torch.long)
    elif task == "copy":
        d = torch.tensor(delays)[torch.randint(0, len(delays), (B,), generator=gen)]
        v = torch.randint(VAL_LO, VAL_HI, (B,), generator=gen)
        i = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            i[b] = torch.randint(4, T - int(d[b]) - 6, (1,), generator=gen)
        q = i + 1 + d
        rows = torch.arange(B)
        x[rows, i] = MARK
        x[rows, i + 1] = v
        x[rows, q] = QUERY
        tgt[rows, q] = v
        meta = d
    elif task == "mqar":
        # multi-query associative recall: 4 (key, value) pairs early, one key
        # queried at gap >= delay past its pair — attention's home turf
        n_pairs = 4
        # keys 20..39, filler 40..63 — DISJOINT (same-range filler would
        # duplicate the queried key and make the recall target ambiguous)
        x = torch.randint(40, FILL_HI, (B, T), generator=gen)
        d = torch.tensor(delays)[torch.randint(0, len(delays), (B,), generator=gen)]
        for b in range(B):
            dd = int(d[b])
            starts = torch.randperm(max(1, T - dd - 12), generator=gen)[:n_pairs] + 2
            keys = torch.randperm(20, generator=gen)[:n_pairs] + 20
            vals = torch.randint(VAL_LO, VAL_HI, (n_pairs,), generator=gen)
            for k in range(n_pairs):
                x[b, starts[k]] = keys[k]
                x[b, min(int(starts[k]) + 1, T - 1)] = vals[k]
            qi = int(starts[0]) + 1 + dd
            if qi >= T:
                qi = T - 1
            x[b, qi] = keys[0]
            tgt[b, qi] = vals[0]
        meta = d
    elif task == "induction":
        d = torch.tensor(delays)[torch.randint(0, len(delays), (B,), generator=gen)]
        a = torch.randint(VAL_LO, VAL_HI, (B,), generator=gen)
        b_tok = torch.randint(VAL_LO, VAL_HI, (B,), generator=gen)
        i = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            i[b] = torch.randint(4, T - int(d[b]) - 6, (1,), generator=gen)
        j = i + 1 + d
        rows = torch.arange(B)
        x[rows, i] = a
        x[rows, i + 1] = b_tok
        x[rows, j] = a
        tgt[rows, j] = b_tok
        meta = d
    else:
        raise ValueError(task)
    return x, tgt, meta


@torch.no_grad()
def eval_task(model, task, delays, gen, T, n_batches=16, batch=32):
    model.eval()
    hits = {d: [0, 0] for d in delays}
    for _ in range(n_batches):
        x, tgt, meta = gen_batch(task, batch, T, gen, delays)
        logits = model(x)
        pred = logits.argmax(-1)
        mask = tgt >= 0
        for b in range(batch):
            m = mask[b]
            if not m.any():
                continue
            ok = (pred[b][m] == tgt[b][m]).float().mean().item()
            d = int(meta[b]) if task != "runmax" else delays[0]
            hits[d][0] += ok
            hits[d][1] += 1
    model.train()
    return {d: (h / max(n, 1)) for d, (h, n) in hits.items() if n > 0}


def run_synthetic(task, arch, seed, args, out_path):
    delays = {"runmax": [0], "copy": [32, 64, 128],
              "induction": [32, 64, 128], "mqar": [32, 64, 128]}[task]
    torch.manual_seed(seed)
    model = ConvRNNLM(VOCAB, d_model=64, n_layers=2, dropout=0.0, tau=0.5,
                      recurrent=(arch != "convrnn-norecur"), lam_min=args.lam_min,
                      hybrid_attn=(arch == "convrnn-hybrid"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(args.steps):
        model.set_tau(tau_at(step, args.steps, 0.5, True))
        x, tgt, _ = gen_batch(task, args.batch, args.seq_len, gen, delays)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                               ignore_index=-100)
        loss.backward()
        opt.step()
        if step % 400 == 0:
            print(f"  [{task}/{arch} s{seed}] step {step} loss {loss.item():.4f}",
                  flush=True)
    model.set_tau(0.0)
    eg = torch.Generator().manual_seed(seed + 1000)
    acc = eval_task(model, task, delays, eg, args.seq_len)
    lam = (torch.cat([l.scan.lam().detach() for l in model.layers
                      if getattr(l, "scan", None)])
           if arch == "convrnn" else None)
    row = dict(gate="G2b", task=task, arch=arch, seed=seed, steps=args.steps,
               lr=args.lr, acc={str(k): round(v, 4) for k, v in acc.items()},
               lambda_p50=(float(lam.median()) if lam is not None else None),
               wall_s=round(time.time() - t0, 1))
    with out_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[G2b] {task:10s} {arch:16s} seed={seed}: acc={row['acc']} "
          f"({row['wall_s']}s)")
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part", choices=["2a", "2b", "all"], default="all")
    p.add_argument("--tasks", nargs="+", default=["copy", "runmax", "induction"])
    p.add_argument("--archs", nargs="+",
                   default=["convrnn", "convrnn-norecur"],
                   choices=["convrnn", "convrnn-norecur", "convrnn-hybrid"],
                   help="which arms to run (single-arm for queue parallelism)")
    p.add_argument("--seed-list", type=int, nargs="+", default=None,
                   help="explicit seeds (overrides --seeds range)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam-min", type=float, default=2 ** -8)
    p.add_argument("--stream", type=int, default=20000)
    p.add_argument("--out", type=Path, default=REPO / "results" / "pc_convrnn")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ok = True
    if args.part in ("2a", "all"):
        ok &= drift_probe(args, args.out / "g2_stability.jsonl")
    if args.part in ("2b", "all"):
        out = args.out / "g2_synthetic.jsonl"
        seeds = args.seed_list if args.seed_list is not None else list(range(args.seeds))
        for task in args.tasks:
            for arch in args.archs:
                for seed in seeds:
                    run_synthetic(task, arch, seed, args, out)
    print(f"[G2] done (2a {'PASS' if ok else 'CHECK'})")


if __name__ == "__main__":
    main()
