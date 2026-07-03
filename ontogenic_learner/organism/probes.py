"""The probe suite — clone-isolated observability (plan §probes).

ISOLATION CONTRACT: probes touch the organism only via predict_logits under no_grad; they
NEVER call policy.surprise (the EMA is the subtle contamination channel); adaptation happens
only on deepcopies with fresh optimizers; all randomness uses dedicated generators;
life_step does not advance. test_probe_isolation asserts the state-hash is unchanged."""
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .mnist_tokens import image_to_grid, image_to_tokens, tokens_to_grid, grid_to_ascii
from .policy import onehot
from .sampler import sample


@torch.no_grad()
def val_bpc(net, X, y, V, device, bs=4096):
    tot = 0.0
    for i in range(0, X.size(0), bs):
        tot += F.cross_entropy(net.predict_logits(onehot(X[i:i+bs], V).to(device)),
                               y[i:i+bs].to(device), reduction="sum").item()
    return tot / X.size(0) / math.log(2)


@torch.no_grad()
def digit_completion_bpc(net, vocab, world, device, n_images=200):
    """Teacher-forced bpc on the pixel tokens of held-out test digits (rows 7-12,
    conditioned on the true prefix)."""
    Xte, Yte = world.mnist_test
    ctx, V = world.ctx, vocab.V
    tot, n = 0.0, 0
    for k in range(n_images):
        ids = torch.tensor(image_to_tokens(vocab, Xte[k], int(Yte[k]), with_label=True),
                           dtype=torch.long)
        half = len(ids) // 2
        xs = torch.stack([ids[i:i+ctx] for i in range(half - ctx, len(ids) - ctx - 1)])
        ys = ids[half:len(ids) - 1]
        pix_mask = (ys >= vocab.pix0) & (ys < vocab.pix0 + 16)
        if pix_mask.sum() == 0:
            continue
        ce = F.cross_entropy(net.predict_logits(onehot(xs, V).to(device)),
                             ys.to(device), reduction="none")
        tot += ce[pix_mask.to(device)].sum().item()
        n += int(pix_mask.sum())
    return tot / max(n, 1) / math.log(2)


def digit_grids(net, vocab, ctx, gen, device):
    """Class-conditional + unconditional generations, rendered as ASCII (report payload)."""
    outs = {}
    from .vocab import SOI
    for k in list(range(10)) + ["free"]:
        prompt = (f"digit {k}:" + SOI) if k != "free" else SOI
        txt = sample(net, vocab, ctx, prompt, n=170, temp=0.8, gen=gen, device=device)
        grid = tokens_to_grid(vocab, vocab.encode_mixed(prompt + txt))
        outs[str(k)] = grid_to_ascii(grid)
    return outs


def plasticity_probe(net, vocab, world, cfg, gen, device):
    """Clone-isolated: deepcopy + fresh Adam, fixed 60-step gated adaptation on the
    held-back docs shot; report adapt gain + base drift; clone discarded."""
    clone = copy.deepcopy(net)
    opt = torch.optim.Adam(clone.parameters(), lr=cfg.lr_fast)
    Xd, yd = world.val["docs"]
    before = val_bpc(clone, Xd, yd, vocab.V, device)
    ids = world.ids["docs"]
    for step in range(60):
        off = 1000 + step * cfg.batch            # fixed, deterministic windows
        xs = torch.stack([ids[off+j: off+j+world.ctx] for j in range(cfg.batch)])
        ys = ids[off+world.ctx: off+world.ctx+cfg.batch]
        clone.train_step_cascade(onehot(xs, vocab.V).to(device), ys.to(device),
                                 opt, mode="dyadic")
    after = val_bpc(clone, Xd, yd, vocab.V, device)
    Xw, yw = world.val["wiki"]
    wiki_after = val_bpc(clone, Xw, yw, vocab.V, device)
    del clone, opt
    return {"docs_before": round(before, 4), "docs_after": round(after, 4),
            "adapt_gain": round(before - after, 4),
            "clone_wiki_after": round(wiki_after, 4)}


def health(net):
    norms = {}
    for i, p in enumerate(net.parameters()):
        norms[f"p{i}"] = round(p.detach().norm().item(), 3)
    return norms


def run_suite(net, policy, vocab, world, cfg, gens, life_step, out_dir, device):
    t0 = time.time()
    row = {"ts": round(time.time(), 1), "life_step": life_step, "kind": "probe"}
    for d in ["wiki", "shakespeare", "code", "docs"]:
        X, y = world.val[d]
        row[f"bpc_{d}"] = round(val_bpc(net, X, y, vocab.V, device), 4)
    row["bpc_digits"] = round(digit_completion_bpc(net, vocab, world, device), 4)
    row["plasticity"] = plasticity_probe(net, vocab, world, cfg, gens["probe"], device)
    row["gate"] = {"n_fast": policy.n_fast, "n_sleep": policy.n_sleep,
                   "fwd_eq": policy.fwd_eq, "rolling_rate": round(policy.rolling_rate(), 4),
                   "reservoir": len(policy.res.x)}
    row["health_norm_mean"] = round(float(np.mean(list(health(net).values()))), 3)
    grids = digit_grids(net, vocab, world.ctx, gens["probe"], device)
    text = sample(net, vocab, world.ctx, "", 300, 0.8, gens["probe"], device)
    row["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probes.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, f"samples-{life_step:012d}.json"), "w") as f:
        json.dump({"life_step": life_step, "text": text, "digits": grids}, f, indent=1)
    return row
