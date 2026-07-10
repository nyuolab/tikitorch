"""M4: FIM-trained MorphSSM — the cross-family uniqueness baseline for G6b.

Pre-registered in results/pc_convrnn/PREREG.md (M4). Trains the SAME MorphSSM
(arch/params/substrate identical; cascade credit) with fill-in-the-middle data
augmentation (PSM format, 50% of batches), then evaluates span infilling on the
G6b windows — the "one data-augmentation flag" answer a transformer shop would
give, reproduced on our substrate, against which the settled-PC mode is judged.

Vocab: char-65 + three sentinels  <PRE>=65  <SUF>=66  <MID>=67  (vocab 68).
FIM batch: [<PRE>] prefix [<SUF>] suffix [<MID>] middle   (predict everything;
the model learns to emit the middle conditioned on BOTH sides).

Usage::

    python scripts/pc_convrnn_fim.py --steps 6000 --credit cascade
    python scripts/pc_convrnn_fim.py --eval-only --ckpt results/pc_convrnn/ckpts/m4_fim_d256_6k.pt
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import tikitorch as tt                                     # noqa: E402
from pc_convrnn_charlm import MorphPCEngine, build_model   # noqa: E402
from nanogpt_shakespeare import CharData, tau_at           # noqa: E402

PRE, SUF, MID = 65, 66, 67
FIM_VOCAB = 68


def fim_rearrange(x: torch.Tensor, span_lo: int, span_hi: int,
                  gen: torch.Generator):
    """(B, T) plain windows -> (B, T+3) PSM-format FIM sequences + span slices."""
    B, T = x.shape
    out = torch.zeros(B, T + 3, dtype=torch.long)
    spans = []
    for b in range(B):
        span = int(torch.randint(span_lo, span_hi + 1, (1,), generator=gen))
        a = int(torch.randint(4, T - span - 4, (1,), generator=gen))
        prefix, middle, suffix = x[b, :a], x[b, a:a + span], x[b, a + span:]
        row = torch.cat([torch.tensor([PRE]), prefix, torch.tensor([SUF]),
                         suffix, torch.tensor([MID]), middle])
        out[b, :row.numel()] = row
        spans.append((a, span, row.numel()))
    return out, spans


def fim_batch(data, batch, seq_len, gen, fim_prob=0.5):
    x, y = data.batch("train", batch, seq_len, gen)
    if torch.rand(1, generator=gen).item() < fim_prob:
        xf, _ = fim_rearrange(x, 8, 32, gen)
        return xf[:, :-1], xf[:, 1:]
    return x, y


@torch.no_grad()
def eval_fim_infill(model, data, spans, n_windows, seq_len, seed, device):
    """G6b-protocol windows; greedy-generate the middle after the PSM prompt."""
    model.eval()
    model.set_tau(0.0)
    rows = []
    for span in spans:
        gen = torch.Generator().manual_seed(41 + span)      # G6b window seeds
        x, _ = data.batch("val", n_windows, seq_len, gen)
        a = (seq_len - span) // 2
        truth = x[:, a:a + span].to(device)
        prompt = torch.cat([
            torch.full((n_windows, 1), PRE), x[:, :a],
            torch.full((n_windows, 1), SUF), x[:, a + span:],
            torch.full((n_windows, 1), MID)], dim=1).to(device)
        cur = prompt
        outs = []
        for _ in range(span):
            nxt = model(cur)[:, -1].argmax(-1)
            outs.append(nxt)
            cur = torch.cat([cur, nxt[:, None]], dim=1)
        pred = torch.stack(outs, dim=1)
        em = float((pred == truth).float().mean())
        rows.append(dict(gate="M4", span=span, em_fim=round(em, 4),
                         n_windows=n_windows))
        print(f"[M4] span={span}: FIM-infill EM={em:.4f}")
    model.train()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", type=Path, default=REPO / "data" / "tinyshakespeare.txt")
    p.add_argument("--out", type=Path, default=REPO / "results" / "pc_convrnn")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--credit", choices=["bp", "cascade"], default="cascade")
    p.add_argument("--family", choices=["morphssm", "dense"], default="morphssm",
                   help="dense = DenseGPT FIM baseline (the true cross-family M4 form)")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lam-min", type=float, default=2 ** -6)
    p.add_argument("--fim-prob", type=float, default=0.5)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-spans", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--n-windows", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--ckpt", type=Path, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = CharData(args.text)
    log = args.out / "m4_fim.jsonl"

    if args.eval_only:
        ck = torch.load(args.ckpt, map_location="cpu")
        ns = argparse.Namespace(**{**vars(args), **ck["config"],
                                   "kernel": ck["config"]["kernel"]})
        model = build_model("morphssm", FIM_VOCAB, ns).to(device)
        model.load_state_dict(ck["state_dict"])
        rows = eval_fim_infill(model, data, args.eval_spans, args.n_windows,
                               args.seq_len, args.seed, device)
        with log.open("a") as f:
            for r in rows:
                f.write(json.dumps({**r, "ckpt": str(args.ckpt)}) + "\n")
        return

    torch.manual_seed(args.seed)
    if args.family == "dense":
        from nanogpt_shakespeare import DenseGPT
        model = DenseGPT(FIM_VOCAB, d=args.d_model, n_heads=4,
                         n_layers=args.n_layers, d_ff=3 * args.d_model,
                         max_seq_len=args.seq_len + 3,
                         dropout=args.dropout).to(device)
        model.set_tau = lambda t: None                     # dense: anneal no-op
        args.credit = "bp"                                 # cascade is tropical-only
    else:
        model = build_model("morphssm", FIM_VOCAB, args).to(device)
    engine = MorphPCEngine(model) if args.credit == "cascade" else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        model.set_tau(tau_at(step, args.steps, args.tau, True))
        x, y = fim_batch(data, args.batch_size, args.seq_len, gen, args.fim_prob)
        x, y = x.to(device), y.to(device)
        if args.credit == "cascade":
            loss = engine.train_step(x, y, opt, credit="cascade")
        else:
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, FIM_VOCAB), y.reshape(-1))
            loss.backward()
            opt.step()
        if step % args.eval_interval == 0 or step == args.steps - 1:
            # plain-LM CE on the standard protocol windows (the FIM tax
            # readout); inline because the model's vocab (68) includes the
            # sentinels — CE over the full softmax, honestly
            model.set_tau(0.0)
            model.eval()
            with torch.no_grad():
                egen = torch.Generator().manual_seed(args.seed + 1)
                ce_sum = correct = total = 0
                for _ in range(8):
                    ex, ey = data.batch("val", args.batch_size, args.seq_len, egen)
                    ex, ey = ex.to(device), ey.to(device)
                    lg = model(ex).reshape(-1, FIM_VOCAB).float()
                    yf = ey.reshape(-1)
                    ce_sum += F.cross_entropy(lg, yf, reduction="sum").item()
                    correct += (lg.argmax(-1) == yf).sum().item()
                    total += yf.numel()
            ce, acc = ce_sum / total, correct / total
            model.train()
            model.set_tau(tau_at(step, args.steps, args.tau, True))
            row = dict(gate="M4", family=args.family, step=step, train_loss=float(loss.item()),
                       val_ce_hard_plainlm=ce, val_acc=acc,
                       wall_s=round(time.time() - t0, 1), credit=args.credit)
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[M4] step {step:5d} train {loss.item():.4f} "
                  f"plainLM hard CE {ce:.4f}", flush=True)
    model.set_tau(0.0)
    ckpt_dir = args.out / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"arch": "morphssm", "credit": args.credit, "seed": args.seed,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": dict(d_model=args.d_model, n_layers=args.n_layers,
                               kernel=args.kernel, dropout=args.dropout,
                               lam_min=args.lam_min, vocab=FIM_VOCAB)},
               ckpt_dir / f"m4_fim_{args.family}_d{args.d_model}_{args.steps // 1000}k.pt")
    rows = eval_fim_infill(model, data, args.eval_spans, args.n_windows,
                           args.seq_len, args.seed, device)
    with log.open("a") as f:
        for r in rows:
            f.write(json.dumps({**r, "final": True, "credit": args.credit, "family": args.family}) + "\n")


if __name__ == "__main__":
    main()
