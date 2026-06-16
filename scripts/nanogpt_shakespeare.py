"""nanoGPT-style char-LM bakeoff: dense GPT vs TropicalTransformer on tiny-shakespeare.

Three runs at matched dims, one question each:

    dense            minimal GPT (causal attn, GELU MLP, LayerNorm), AdamW + float CE.
                     The unconstrained silicon-free reference.
    tropical-float   TropicalTransformer, AdamW + float CE, soft tau (annealed).
                     Substrate capacity under the best training conditions.
    tropical-closed  TropicalTransformer, SignSGD + tropical_ce, tau annealed to hard.
                     Sign-based training, multiplier-free at deployment (tau=0).
                     NOT fully multiplier-free as run: SignSGD momentum is a float
                     EMA (closure hole H7), tropical_ce uses a host-side softmax,
                     and the soft-tau forward uses multiplicative relaxations.

Metrics: val cross-entropy (nats and bits/char), top-1 next-char accuracy,
tokens/sec, plus a sampled continuation per model. Tropical models are
evaluated both at their training tau (soft) and at tau=0 (hard / deployable).

Usage::

    python scripts/nanogpt_shakespeare.py --steps 1500
    python scripts/nanogpt_shakespeare.py --runs dense tropical-float --steps 200
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import tikitorch as tt
import tikitorch.optim as ttoptim
from tikitorch.nn import DyadicLinear, TropicalAttention, TropicalTopKModule

REPO = Path(__file__).resolve().parents[1]

# Architecture variants for the tropical arms (applied post-construction where
# the TropicalTransformer constructor has no knob). All are multiplier-free at
# deployment; "dyadic-*" arms use ±2^k shift-add weights (STE-trained), the
# repo's documented fix for the max-plus oblique-capacity deficit.
VARIANTS = ("baseline", "dyadic-ffn", "dyadic-ffn-head", "dyadic-full",
            "maxmin", "iter-topk", "topk-proj", "hilbert", "morphssm",
            "hilbert-dyadic-ffn", "morphssm-dyadic")


# ---------------------------------------------------------------------------
# Data: char-level tiny-shakespeare, 90/10 split
# ---------------------------------------------------------------------------

class CharData:
    def __init__(self, path: Path, split: float = 0.9):
        text = path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.vocab = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * split)
        self.train_ids, self.val_ids = ids[:n], ids[n:]
        # tokens-per-byte on the val stream: 1.0 for chars (ASCII corpus).
        self.val_tokens_per_byte = self.val_ids.numel() / len(text[n:].encode())

    def batch(self, split: str, batch: int, seq_len: int, gen: torch.Generator):
        ids = self.train_ids if split == "train" else self.val_ids
        starts = torch.randint(0, ids.numel() - seq_len - 1, (batch,), generator=gen)
        x = torch.stack([ids[s:s + seq_len] for s in starts])
        y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
        return x, y

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([[self.stoi[c] for c in s]], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


class BPEData(CharData):
    """Byte-level BPE tokenization (trained on the TRAIN split only, cached).

    Inherits batch(); exposes val_tokens_per_byte so CE/token converts to
    bits-per-byte for honest comparison against char-level runs.
    """

    def __init__(self, path: Path, vocab_size: int, split: float = 0.9):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        text = path.read_text(encoding="utf-8")
        n_chars = int(len(text) * split)
        train_text, val_text = text[:n_chars], text[n_chars:]

        cache = path.parent / f"bpe_{path.stem}_{vocab_size}.json"
        if cache.exists():
            tok = Tokenizer.from_file(str(cache))
        else:
            tok = Tokenizer(models.BPE())
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
            tok.train_from_iterator([train_text], trainer)
            tok.save(str(cache))
        self.tok = tok
        self.vocab = tok.get_vocab_size()
        self.train_ids = torch.tensor(tok.encode(train_text).ids, dtype=torch.long)
        self.val_ids = torch.tensor(tok.encode(val_text).ids, dtype=torch.long)
        self.val_tokens_per_byte = self.val_ids.numel() / len(val_text.encode())

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.tok.encode(s).ids], dtype=torch.long)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))


# ---------------------------------------------------------------------------
# Dense baseline: minimal nanoGPT
# ---------------------------------------------------------------------------

class DenseBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d), nn.Dropout(dropout))

    def forward(self, x):
        h = self.ln1(x)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class DenseGPT(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DenseBlock(d, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        T = x.shape[1]
        h = self.drop(self.tok(x) + self.pos(torch.arange(T, device=x.device))[None])
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))

    def compute_loss(self, x, y):
        logits = self(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


class _DyadicUp(nn.Module):
    """DyadicLinear in the block's ffn_up slot (set_tau inert: dyadic deploys as-is)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = DyadicLinear(d_in, d_out)

    def set_tau(self, tau):
        pass

    def forward(self, x):
        return self.lin(x)


class _DyadicDown(_DyadicUp):
    """ReLU -> DyadicLinear in the ffn_down slot. Together with _DyadicUp the FFN
    becomes lin -> relu -> lin — the DyadicTropicalProjection shape, at exactly
    the param count of the tropical dilation->erosion FFN it replaces."""

    def forward(self, x):
        return self.lin(torch.relu(x))


def apply_variant(model, vocab, args):
    """Post-construction surgery for tropical-arm architecture variants."""
    v = args.variant
    if v in ("baseline", "hilbert", "morphssm"):
        return  # handled at construction
    if v == "morphssm-dyadic":
        # Morph-conv sequence mixing + dyadic ±2^k channel mixing: swap each
        # layer's tropical dilation out_proj for a shift-add oblique linear.
        for layer in model.layers:
            d = layer.out_proj.weight.shape[1]
            layer.out_proj = _DyadicUp(d, d)
            layer.out_proj_type = "dyadic"   # takes the N-D forward branch
        return
    d, dff = args.d_model, args.d_ff
    for block in model.blocks:
        if v.startswith("dyadic") or v == "hilbert-dyadic-ffn":
            block.ffn_up = _DyadicUp(d, dff)
            block.ffn_down = _DyadicDown(dff, d)
        if v == "dyadic-full":
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                setattr(block.attn, name, DyadicLinear(d, d))
            # Non-"dilation" tag -> attention takes the N-D forward path;
            # the set_tau cascade skips modules without set_tau (hasattr check).
            block.attn.proj_variant = "dyadic"
        if v in ("maxmin", "iter-topk"):
            block.attn = TropicalAttention(
                d, args.n_heads, dropout=args.dropout, tau=args.tau,
                agg_variant="maxmin" if v == "maxmin" else "iter_topk",
                agg_topk_k=4)
        if v == "topk-proj":
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                setattr(block.attn, name, TropicalTopKModule(d, d, k=4, tau=args.tau))
            block.attn.proj_variant = "topk"
    if v in ("dyadic-ffn-head", "dyadic-full"):
        model.output_head = DyadicLinear(args.d_model, vocab)


class DenseConvLM(nn.Module):
    """Real-arithmetic twin of MorphSSM: same shape (d, L, kernel, exponential
    dilation), same sandwich-norm residual layout, but standard depthwise causal
    Conv1d + GELU + linear channel mixing + LayerNorm. The attribution control
    for 'is MorphSSM's gap-closing the conv inductive bias or the substrate?'"""

    class Layer(nn.Module):
        def __init__(self, d, kernel, dilation, dropout):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(d, d, kernel, dilation=dilation, groups=d)
            self.drop = nn.Dropout(dropout)
            self.out_proj = nn.Linear(d, d)
            self.post_norm = nn.LayerNorm(d)

        def forward(self, x):
            h = self.norm(x).transpose(1, 2)
            h = self.conv(F.pad(h, (self.pad, 0))).transpose(1, 2)
            h = self.out_proj(self.drop(F.gelu(h)))
            return self.post_norm(x + h)

    def __init__(self, vocab, d=256, n_layers=7, kernel=7, dropout=0.1,
                 max_dilation=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            self.Layer(d, kernel, min(2 ** i, max_dilation), dropout)
            for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.drop(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.ln_f(h))

    def compute_loss(self, x, y):
        logits = self(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


class PosEmbedTropicalTransformer(tt.models.TropicalTransformer):
    """TropicalTransformer + learned additive positional embedding.

    Plain TropicalAttention has no positional signal of any kind, while the
    dense baseline gets a learned positional embedding — an unfair handicap.
    A position gather + add is closure-clean, and it gives the tropical arm
    the same positional mechanism (and nearly identical param count) as dense.
    """

    def __init__(self, *, max_seq_len: int, **kw):
        super().__init__(max_seq_len=max_seq_len, **kw)
        self.pos = nn.Embedding(max_seq_len, kw["d_model"])

    def _forward_lm(self, x):
        h = self.embed(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for block in self.blocks:
            h = block(h, causal_mask=True)
        return self.output_head(self.ln_final(h))


# ---------------------------------------------------------------------------
# Eval / sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, batch, seq_len, batches, seed, device):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ce_sum = correct = total = 0
    for _ in range(batches):
        x, y = data.batch("val", batch, seq_len, gen)
        x, y = x.to(device), y.to(device)
        logits = model(x).reshape(-1, data.vocab).float()
        yf = y.reshape(-1)
        ce_sum += F.cross_entropy(logits, yf, reduction="sum").item()
        correct += (logits.argmax(-1) == yf).sum().item()
        total += yf.numel()
    model.train()
    return ce_sum / total, correct / total


@torch.no_grad()
def sample(model, data, prompt, n_chars, seq_len, device, temperature=0.8, seed=0):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    ids = data.encode(prompt).to(device)
    for _ in range(n_chars):
        logits = model(ids[:, -seq_len:])[0, -1].float()
        probs = F.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1, generator=gen).to(device)
        ids = torch.cat([ids, nxt[None]], dim=1)
    model.train()
    return data.decode(ids[0].tolist())


def tau_at(step, steps, tau0, anneal, floor=0.0):
    """Hold tau0 for the first half, linear decay to `floor` by 90% of training.
    floor>0 deploys the soft (log-semiring) attention at a FIXED nonzero tau
    instead of annealing into the hard max-plus corner (the dead band)."""
    if not anneal:
        return max(tau0, floor)
    a, b = 0.5 * steps, 0.9 * steps
    if step < a:
        return max(tau0, floor)
    if step >= b:
        return floor
    return max(tau0 * (1.0 - (step - a) / (b - a)), floor)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def build_model(kind, data, args):
    if kind == "dense":
        return DenseGPT(data.vocab, args.d_model, args.n_heads, args.n_layers,
                        args.d_ff, args.seq_len, args.dropout)
    if kind == "dense-conv":
        return DenseConvLM(data.vocab, d=args.morph_d, n_layers=args.morph_layers,
                           kernel=7, dropout=args.dropout)
    if args.variant in ("morphssm", "morphssm-dyadic"):
        # Morphological-conv sequence mixer; dims chosen for ~matched params
        # (d=256 L=7 with tropical channel mixing ≈ 0.51M). Causal convs carry
        # position, so no positional embedding is needed.
        model = tt.models.MorphSSM(
            d_model=args.morph_d, n_layers=args.morph_layers, kernel_size=7,
            tropical_proj=True,
            multi_scale=args.morph_multi_scale,
            group_size=args.morph_group_size,
            dropout=args.dropout, use_soft=True, tau=max(args.tau, 1e-6),
            task_type="lm", vocab_size=data.vocab)
        apply_variant(model, data.vocab, args)
        return model
    cls = PosEmbedTropicalTransformer if args.pos_embed else tt.models.TropicalTransformer
    model = cls(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, tau=args.tau,
        use_shift_attn=args.use_shift_attn,
        score_variant=("hilbert" if args.variant in ("hilbert", "hilbert-dyadic-ffn")
                       else "dot_product"),
        max_seq_len=args.seq_len, task_type="lm", vocab_size=data.vocab)
    apply_variant(model, data.vocab, args)
    if getattr(args, "n_hops", 1) > 1 or getattr(args, "agg_variant", None):
        from tikitorch.nn import TropicalAttention
        attns = [m for m in model.modules() if isinstance(m, TropicalAttention)]
        for m in attns:
            if args.n_hops > 1:
                m.n_hops = args.n_hops
            if args.agg_variant:
                m.agg_variant = args.agg_variant   # e.g. "sinkhorn" (assignment routing)
                m.sink_eps = args.sink_eps
                m.sink_iters = args.sink_iters
        print(f"[attn] n_hops={args.n_hops} agg={args.agg_variant or 'default'} "
              f"sink_eps={args.sink_eps} sink_iters={args.sink_iters} on "
              f"{len(attns)} TropicalAttention module(s)")
    return model


# Strictly closed (zero-DSP) optimizers; "signsgd" with momentum is hole H7.
CLOSED_OPTS = ("signsgd0", "puretropical", "tropicallion", "countagree")
CLOSED_LOSSES = ("tropical_ce_lut", "perceptron")


def make_optimizer(name, model, args):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.adam_lr,
                                 weight_decay=0.01)
    if name == "signsgd":
        return ttoptim.SignSGD(model.parameters(), lr=args.sign_lr, momentum=0.9)
    if name == "signsgd0":
        return ttoptim.SignSGD(model.parameters(), lr=args.sign_lr, momentum=0.0)
    if name == "puretropical":
        return ttoptim.PureTropicalSGD(model.parameters(), lr_shift=args.lr_shift,
                                       momentum_bits=4,
                                       noise_scale=args.noise_scale)
    if name == "tropicallion":
        return ttoptim.TropicalLion(model.parameters(), lr_shift=args.lr_shift)
    if name == "countagree":
        return ttoptim.CountAgreementSGD(model.parameters(),
                                         base_shift=args.lr_shift)
    raise ValueError(name)


def strict_gate(model, data, args, opt_name, loss_type):
    """Machine-checked strict closure: deploy-closure always; training-closure
    (Level B) when training is hard from scratch (tau=0, no anneal)."""
    from tikitorch.audit import assert_closed_runtime
    from tikitorch.models import classification_loss

    if opt_name not in CLOSED_OPTS:
        raise SystemExit(f"--strict requires a closed optimizer {CLOSED_OPTS}, "
                         f"got {opt_name!r}")
    if loss_type not in CLOSED_LOSSES:
        raise SystemExit(f"--strict requires a closed loss {CLOSED_LOSSES}, "
                         f"got {loss_type!r}")
    gen = torch.Generator().manual_seed(0)
    x, y = data.batch("train", 2, args.seq_len, gen)
    assert_closed_runtime(model, x)          # hardens tau=0, traces the graph
    hard_training = args.tau == 0.0 and not args.anneal
    if hard_training:
        from tikitorch.audit import find_training_multiplies

        def loss_fn(logits, tgt):
            return classification_loss(logits.reshape(-1, logits.shape[-1]),
                                       tgt.reshape(-1), loss_type)
        opt_probe = make_optimizer(opt_name, model, args)
        hits = find_training_multiplies(model, x, y, loss_fn, optimizer=opt_probe)
        # Sanctioned float-sim artifacts (repo doctrine, tests/test_audit.py):
        # - elementwise-divide: the loss-VALUE batch mean; sign-invariant, the
        #   silicon emits the +/-1 gradient directly and never computes it.
        # - transcendental (LUT loss only): exp2 ROM table build + log2-as-
        #   leading-zero-count in shift_softmax; BRAM/priority-encoder on chip.
        exempt = {"elementwise-divide"}
        if loss_type == "tropical_ce_lut":
            exempt.add("transcendental")
        bad = [(c, op) for c, op in hits if c not in exempt]
        if bad:
            lines = "\n".join(f"    {c}: {op}" for c, op in bad)
            raise SystemExit(f"[strict] training step NOT closed:\n{lines}")
        waived = sorted({c for c, _ in hits if c in exempt})
        print(f"[strict] training-closure OK"
              + (f" (waived float-sim artifacts: {', '.join(waived)})" if waived else ""))
    print(f"[strict] deploy-closure OK"
          f"{'' if hard_training else ' (training uses soft-tau anneal — Level B not claimed)'}")


def train_run(kind, data, args, log_path):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model(kind, data, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    is_tropical = kind not in ("dense", "dense-conv")
    opt_name = args.opt or ("signsgd" if kind == "tropical-closed" else "adamw")
    loss_type = args.loss or ("tropical_ce" if kind == "tropical-closed"
                              else "cross_entropy")
    if is_tropical:
        if loss_type == "tropical_ce_lut_exact":
            # Diagnostic: LUT-quantized softmax with EXACT normalization —
            # isolates LUT quantization from the shift-normalize gradient bias
            # (sum(p) in [0.5,1] -> systematic sign-update drift hypothesis).
            from tikitorch.functional import tropical_cross_entropy

            def _diag_loss(x, y, _m=model):
                logits = _m(x)
                return tropical_cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                    softmax="lut", normalize="exact")
            model.compute_loss = _diag_loss
        else:
            model.loss_type = loss_type
        if args.strict:
            strict_gate(model, data, args, opt_name, loss_type)
    model = model.to(device)
    opt = make_optimizer(opt_name, model, args)
    is_countagree = opt_name == "countagree"

    anneal = is_tropical and args.anneal
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    t0 = time.time()
    tokens_seen = 0

    for step in range(args.steps):
        if anneal:
            model.set_tau(tau_at(step, args.steps, args.tau, True, args.tau_floor))
        if args.lr_decay_every and step > 0 and step % args.lr_decay_every == 0:
            # Bit-shift lr decay: halve the power-of-2 step. Closure-clean (the
            # step stays an exact power of two); works for the shift-based
            # closed optimizers via their lr/lr_shift group params.
            for grp in opt.param_groups:
                if "lr_shift" in grp:
                    grp["lr_shift"] += 1
                grp["lr"] = grp.get("lr", 2.0 ** -10) / 2.0
        x, y = data.batch("train", args.batch_size, args.seq_len, gen)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = model.compute_loss(x, y)
        loss.backward()
        if is_countagree:
            # Windowed sign-agreement: accumulate per microbatch, step per window.
            opt.accumulate()
            if (step + 1) % args.accum_window == 0:
                opt.step()
                opt.reset_accum()
        else:
            opt.step()
        tokens_seen += y.numel()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            row = {"run": kind, "variant": args.variant, "opt": opt_name,
                   "loss_type": loss_type if is_tropical else "cross_entropy",
                   "tokenizer": args.tokenizer, "step": step,
                   "train_loss": float(loss.item()),
                   "wall_s": round(time.time() - t0, 1),
                   "tok_per_s": round(tokens_seen / max(time.time() - t0, 1e-9))}
            if is_tropical:
                cur_tau = tau_at(step, args.steps, args.tau, anneal, args.tau_floor)
                ce, acc = evaluate(model, data, args.batch_size, args.seq_len,
                                   args.eval_batches, args.seed + 1, device)
                row.update(val_ce_soft=ce, val_acc_soft=acc, tau=cur_tau)
                model.set_tau(0.0)
                ce_h, acc_h = evaluate(model, data, args.batch_size, args.seq_len,
                                       args.eval_batches, args.seed + 1, device)
                row.update(val_ce_hard=ce_h, val_acc_hard=acc_h)
                model.set_tau(cur_tau)
            else:
                ce, acc = evaluate(model, data, args.batch_size, args.seq_len,
                                   args.eval_batches, args.seed + 1, device)
                row.update(val_ce=ce, val_acc=acc)
            shown = row.get("val_ce", row.get("val_ce_hard"))
            # bits-per-byte: tokenizer-independent comparison metric.
            row["val_bpb"] = shown * data.val_tokens_per_byte / math.log(2)
            rows.append(row)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{kind}] step {step:5d}  train {loss.item():.4f}  "
                  f"val_ce {shown:.4f} ({row['val_bpb']:.3f} bpb)  "
                  f"{row['tok_per_s']} tok/s", flush=True)

    # Final state: tropical models deploy hard.
    if is_tropical:
        model.set_tau(0.0)
    text = sample(model, data, "\n", args.sample_chars, args.seq_len, device,
                  seed=args.seed)
    final = dict(rows[-1])
    final.update(run=kind, n_params=n_params, sample=text,
                 total_wall_s=round(time.time() - t0, 1))
    with log_path.open("a") as f:
        f.write(json.dumps({**final, "final": True}) + "\n")
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["dense", "tropical-float", "tropical-closed"],
                   choices=["dense", "dense-conv", "tropical-float", "tropical-closed"])
    p.add_argument("--morph-d", type=int, default=256,
                   help="d_model for morphssm / dense-conv arms")
    p.add_argument("--morph-layers", type=int, default=7,
                   help="n_layers for morphssm / dense-conv arms")
    p.add_argument("--morph-multi-scale", action="store_true",
                   help="MultiScaleMorphConv1d (kernels 3/7/15) in morphssm arms")
    p.add_argument("--morph-group-size", type=int, default=1,
                   help="cross-channel group size for morphssm convs")
    p.add_argument("--noise-scale", type=float, default=0.0,
                   help="PureTropicalSGD LFSR noise injection (anneal substitute "
                        "for hard-from-scratch training)")
    p.add_argument("--lr-decay-every", type=int, default=0,
                   help="bit-shift lr decay: increment lr_shift (halve the "
                        "power-of-2 step) every N steps; 0 disables. Counters "
                        "the long-horizon sign-drift collapse into the "
                        "constant-predictor attractor.")
    p.add_argument("--text", type=Path, default=REPO / "data" / "tinyshakespeare.txt")
    p.add_argument("--out", type=Path, default=REPO / "results" / "lm_shakespeare")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"])
    p.add_argument("--vocab-size", type=int, default=1024,
                   help="BPE vocab size (--tokenizer bpe)")
    p.add_argument("--opt", type=str, default=None,
                   choices=["adamw", "signsgd", "signsgd0", "puretropical",
                            "tropicallion", "countagree"],
                   help="override the per-kind default optimizer")
    p.add_argument("--loss", type=str, default=None,
                   choices=["cross_entropy", "perceptron", "tropical_ce",
                            "tropical_ce_lut", "tropical_ce_lut_exact"],
                   help="override the per-kind default loss (tropical arms)")
    p.add_argument("--lr-shift", type=int, default=10,
                   help="power-of-2 lr exponent for shift-based closed optimizers")
    p.add_argument("--accum-window", type=int, default=4,
                   help="microbatches per CountAgreementSGD window")
    p.add_argument("--strict", action="store_true",
                   help="machine-checked strict closure: assert_closed_runtime at "
                        "build; assert_training_closed when hard-from-scratch "
                        "(--tau 0 --no-anneal); closed optimizer+loss required")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--tau-floor", type=float, default=0.0,
                   help="clamp the annealed soft tau to this floor — deploy soft "
                        "attention at a FIXED nonzero tau instead of the hard "
                        "max-plus corner. Use powers of 2 (0.5/0.25/0.125) for "
                        "fabric-legal shifts. 0.0 = original anneal-to-hard.")
    p.add_argument("--n-hops", type=int, default=1,
                   help="tropical attention Bellman-Ford diffusion hops: 1 = "
                        "single-hop (current); 2-4 = multi-hop all-pairs best-path "
                        "routing over the content score graph (purely max-plus). "
                        "Tests routing (not averaging) as attention's mechanism.")
    p.add_argument("--agg-variant", type=str, default=None,
                   choices=["argmax", "iter_topk", "min", "maxmin", "sinkhorn"],
                   help="override tropical attention value-aggregation. 'sinkhorn' = "
                        "assignment/permutation routing (train soft doubly-stochastic "
                        "transport -> deploy hard permutation): densifies the gradient "
                        "(every key carries column mass) AND changes the routing class "
                        "from value-copy to assignment.")
    p.add_argument("--sink-eps", type=float, default=1.0,
                   help="Sinkhorn entropic temperature (smaller = sharper -> closer to "
                        "a hard permutation; anneal toward 0 for pure-tropical deploy).")
    p.add_argument("--sink-iters", type=int, default=5,
                   help="Sinkhorn row/col normalization iterations.")
    p.add_argument("--variant", type=str, default="baseline", choices=VARIANTS,
                   help="tropical-arm architecture variant (ignored for dense)")
    p.add_argument("--pos-embed", action="store_true",
                   help="learned additive positional embedding for tropical arms "
                        "(parity with the dense baseline; closure-clean gather+add)")
    p.add_argument("--use-shift-attn", action="store_true",
                   help="TropicalShiftAttention (projection-free, positional shifts)")
    p.add_argument("--no-anneal", dest="anneal", action="store_false")
    p.add_argument("--adam-lr", type=float, default=1e-3)
    p.add_argument("--sign-lr", type=float, default=2e-3)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--sample-chars", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / f"{args.tag}.jsonl"
    data = (BPEData(args.text, args.vocab_size) if args.tokenizer == "bpe"
            else CharData(args.text))
    print(f"tiny-shakespeare [{args.tokenizer}]: vocab={data.vocab}  "
          f"train={data.train_ids.numel()} val={data.val_ids.numel()} tokens  "
          f"({data.val_tokens_per_byte:.3f} tok/byte)")
    print(f"config: d={args.d_model} L={args.n_layers} heads={args.n_heads} "
          f"d_ff={args.d_ff} seq={args.seq_len} batch={args.batch_size} "
          f"steps={args.steps} device={args.device}\n")

    finals = []
    for kind in args.runs:
        finals.append(train_run(kind, data, args, log_path))
        print()

    print(f"{'run':18s} {'params':>8s} {'val_ce':>7s} {'bpc':>6s} {'acc':>6s} "
          f"{'hard_ce':>8s} {'hard_acc':>8s} {'tok/s':>7s}")
    print("-" * 76)
    for r in finals:
        ce = r.get("val_ce", r.get("val_ce_soft"))
        acc = r.get("val_acc", r.get("val_acc_soft"))
        hce = r.get("val_ce_hard")
        hacc = r.get("val_acc_hard")
        hce_s = f"{hce:8.4f}" if hce is not None else f"{'—':>8s}"
        hacc_s = f"{hacc:8.3f}" if hacc is not None else f"{'—':>8s}"
        print(f"{r['run']:18s} {r['n_params']:8d} {ce:7.4f} {ce / math.log(2):6.3f} "
              f"{acc:6.3f} {hce_s} {hacc_s} {r['tok_per_s']:7d}")


if __name__ == "__main__":
    main()
