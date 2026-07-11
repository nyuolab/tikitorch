"""V0 — the fused live loop: streaming serve + fused learn + cost/parity audit.

North-star validation gate V0 (ontogenic_learner/NORTHSTAR_VALIDATION.md,
Goal A sub-claims A1-A4). Everything here is local, code-only, $0:

  A1  StreamServer drives the token-by-token .step() APIs; streaming logits
      must equal the batched forward within fp noise.
  A2  MorphPCEngine.serve_and_learn: fused grads must equal cascade_grads
      BITWISE (same caches, same sweep).
  A3  measured cost ratios: wall and op-volume for serve-only vs FUSED
      serve+learn vs UNFUSED (the old harness shape: serve fwd + re-forward
      + sweep). Pass: fused ratio <= 2.5x.
  A4  multiply audit on the streaming/fused path at tau=0 (must be zero
      unwaived) and a memory note (tensor-bytes proxy; the GPU peak-mem
      contrast is already banked: cascade trained d832 where autograd OOM'd).

Telemetry schema (designed for the D2 strip-chart): one jsonl row per served
window: {t, domain, ce_learner, ce_frozen, gate_z, fired, fwd_eq_serve,
fwd_eq_learn, norm, probe_*}.

Usage::

    python ontogenic_learner/experiments/st_stream.py --v0     # run the gate
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pc_convrnn_charlm import MorphPCEngine, build_model   # noqa: E402


# ---------------------------------------------------------------------------
# A1 — StreamServer: token-by-token serving on the .step() APIs (B=1)
# ---------------------------------------------------------------------------

class StreamServer:
    """Persistent streaming state for a MorphSSM-shaped LM (morphssm arch).

    Per-layer conv ring buffers (1, D, (K-1)*dilation+1), initialized to zero
    (identical semantics to the batched forward's zero left-padding). Feed one
    token id at a time; get next-token logits back. This IS the serve half of
    the north-star loop.
    """

    def __init__(self, model):
        self.model = model
        model.eval()
        self.reset()

    def reset(self):
        self.bufs = []
        for layer in self.model.layers:
            conv = layer.morph_conv
            blen = (conv.kernel_size - 1) * conv.dilation + 1
            self.bufs.append(torch.zeros(1, conv.channels, blen))

    @torch.no_grad()
    def step(self, tok_id: int) -> torch.Tensor:
        m = self.model
        h = m.embed(torch.tensor([[tok_id]]))[:, 0]        # (1, D); dropout eval-off
        for i, layer in enumerate(m.layers):
            h, self.bufs[i] = layer.step(h, self.bufs[i])
        h = m.ln_final(h.unsqueeze(1)).squeeze(1)
        return m.output_head(h)                             # (1, V)


# ---------------------------------------------------------------------------
# Prequential scorer + telemetry (the D2 schema)
# ---------------------------------------------------------------------------

class Prequential:
    """Predict-then-learn accounting: score theta(t-1) on x_t BEFORE learning.

    Tracks cumulative prequential CE for the learner and a frozen twin, the
    regret, gate statistics, and fwd-eq cost counters (serve=1; fired kick
    adds `kick_cost` — measured, not assumed).
    """

    def __init__(self, kick_cost: float):
        self.kick_cost = kick_cost
        self.n = 0
        self.ce_learner = 0.0
        self.ce_frozen = 0.0
        self.fires = 0
        self.fwd_eq = 0.0

    def record(self, ce_l: float, ce_f: float, fired: bool):
        self.n += 1
        self.ce_learner += ce_l
        self.ce_frozen += ce_f
        self.fwd_eq += 1.0 + (self.kick_cost if fired else 0.0)
        self.fires += int(fired)

    def row(self):
        n = max(self.n, 1)
        return dict(n=self.n, preq_ce=self.ce_learner / n,
                    preq_ce_frozen=self.ce_frozen / n,
                    regret_vs_frozen=(self.ce_learner - self.ce_frozen) / n,
                    fire_rate=self.fires / n,
                    amortized_mult=self.fwd_eq / n)


# ---------------------------------------------------------------------------
# A3/A4 — cost + audit instrumentation
# ---------------------------------------------------------------------------

class OpVolume(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts dispatched ops and total output element volume — the op-volume
    proxy for compute (adds/compares/gathers all count; the multiply AUDIT is
    separate via tikitorch.audit._make_catcher)."""

    def __init__(self):
        self.calls = 0
        self.volume = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.calls += 1
        for t in (out if isinstance(out, (list, tuple)) else (out,)):
            if isinstance(t, torch.Tensor):
                self.volume += t.numel()
        return out


def v0_gate(args):
    torch.manual_seed(0)
    ns = argparse.Namespace(d_model=args.d_model, n_layers=args.n_layers,
                            kernel=7, dropout=0.0, tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", args.vocab, ns)
    model.eval()
    model.set_tau(0.0)                                     # the deployed regime
    engine = MorphPCEngine(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(1)
    T = args.window
    x = torch.randint(0, args.vocab, (1, T), generator=gen)
    y = torch.randint(0, args.vocab, (1, T), generator=gen)
    results = {}

    # --- A1: streaming == batched forward ---------------------------------
    server = StreamServer(model)
    stream_logits = torch.cat([server.step(int(t)) for t in x[0]], dim=0)
    batch_logits = model(x)[0]
    rel = float((stream_logits - batch_logits).norm() /
                batch_logits.norm().clamp_min(1e-30))
    results["A1_stream_vs_batch_relL2"] = rel
    print(f"[V0/A1] streaming .step() vs batched forward: relL2={rel:.2e} "
          f"({'PASS' if rel < 1e-5 else 'FAIL'})")

    # --- A2: fused grads == cascade_grads bitwise ---------------------------
    model.zero_grad(set_to_none=True)
    engine.cascade_grads(x, y)
    ref = {n: p.grad.clone() for n, p in model.named_parameters()
           if p.grad is not None}
    model.zero_grad(set_to_none=True)
    _, _, fired = engine.serve_and_learn(
        x, y, opt=type("O", (), {"zero_grad": lambda *a, **k: None,
                                 "step": lambda *a: None})(), fire=True)
    bit = all(torch.equal(ref[n], p.grad) for n, p in model.named_parameters()
              if n in ref)
    results["A2_fused_bitwise"] = bit
    print(f"[V0/A2] fused serve_and_learn grads == cascade_grads: "
          f"bitwise={bit} ({'PASS' if bit else 'FAIL'})")

    # --- A3: cost ratios (wall + op-volume), fused vs unfused ---------------
    def timed(fn, reps=args.reps):
        fn()                                               # warm
        t0 = time.time()
        for _ in range(reps):
            fn()
        return (time.time() - t0) / reps

    def vol(fn):
        m = OpVolume()
        with m:
            fn()
        return m.volume

    serve = lambda: model(x)
    fused = lambda: engine.serve_and_learn(x, y, opt=opt, fire=True)
    unfused = lambda: (model(x), engine.train_step(x, y, opt, credit="cascade"))
    w_s, w_f, w_u = timed(serve), timed(fused), timed(unfused)
    v_s, v_f, v_u = vol(serve), vol(fused), vol(unfused)
    results.update(wall_serve=w_s, wall_fused=w_f, wall_unfused=w_u,
                   wall_ratio_fused=w_f / w_s, wall_ratio_unfused=w_u / w_s,
                   vol_ratio_fused=v_f / v_s, vol_ratio_unfused=v_u / v_s)
    print(f"[V0/A3] wall: serve {w_s*1e3:.1f}ms | fused {w_f*1e3:.1f}ms "
          f"({w_f/w_s:.2f}x) | unfused {w_u*1e3:.1f}ms ({w_u/w_s:.2f}x)")
    print(f"[V0/A3] op-volume ratios: fused {v_f/v_s:.2f}x | "
          f"unfused {v_u/v_s:.2f}x  "
          f"({'PASS' if v_f/v_s <= 2.5 else 'FAIL'} at the 2.5x bar)")

    # --- A4: multiply audit on the fused path at tau=0 ----------------------
    # Two audits: (i) the LOOP alone (sweep, no optimizer) must be clean with
    # the exact-CE research loss; (ii) the CLOSED configuration (LUT loss +
    # PureTropicalSGD — the G5-certified training step) must be clean end to
    # end. AdamW's own update arithmetic is the research arm's declared
    # boundary, audited separately for the record.
    from tikitorch.audit import _make_catcher
    import tikitorch.optim as ttoptim

    def audit(opt_, loss_mode):
        engine.loss_mode = loss_mode
        hits = []
        noop = type("O", (), {"zero_grad": lambda *a, **k: None,
                              "step": lambda *a: None})()
        with _make_catcher(hits):
            engine.serve_and_learn(x, y, opt=opt_ or noop, fire=True)
        engine.loss_mode = "ce"
        cats = Counter(c for c, _ in hits)
        waived = {"elementwise-divide", "transcendental"}
        return ({c: n for c, n in cats.items() if c not in waived},
                {c: n for c, n in cats.items() if c in waived})

    bad_loop, waived_loop = audit(None, "ce")
    pts = ttoptim.PureTropicalSGD(model.parameters(), lr_shift=10,
                                  momentum_bits=4)
    bad_closed, waived_closed = audit(pts, "lut")
    bad_adamw, _ = audit(opt, "ce")
    results["A4_loop_violations"] = bad_loop
    results["A4_closed_violations"] = bad_closed
    results["A4_adamw_violations_for_record"] = bad_adamw
    print(f"[V0/A4] loop-only audit @tau=0: violations={bad_loop} "
          f"({'PASS' if not bad_loop else 'FAIL'})")
    print(f"[V0/A4] CLOSED config (LUT + PureTropicalSGD): "
          f"violations={bad_closed} waived={waived_closed} "
          f"({'PASS' if not bad_closed else 'FAIL'})")
    print(f"[V0/A4] (record) AdamW research arm adds: {bad_adamw}")

    ok = (rel < 1e-5 and bit and v_f / v_s <= 2.5
          and not bad_loop and not bad_closed)
    results["V0_pass"] = ok
    out = REPO / "ontogenic_learner" / "results" / "st_stream"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "v0_gate.jsonl").open("a") as f:
        f.write(json.dumps({k: (v if not isinstance(v, float) else round(v, 6))
                            for k, v in results.items()}) + "\n")
    print(f"[V0] overall: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v0", action="store_true")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=7)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    ok = v0_gate(args)
    raise SystemExit(0 if ok else 1)
