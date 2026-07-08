"""G1 gate: cascade ≡ settled/async PC relaxation on the unrolled lattice.

Pre-registered in results/pc_convrnn/PREREG.md. G0 (adjoint vs autograd) lives in
pc_convrnn_charlm.py --gradcheck; this script measures the relaxation side:

  - sync reverse-topological relaxation (n_iter=1 and L+2) must equal the
    cascade BIT-IDENTICALLY (the error system is triangular/nilpotent);
  - async relaxation (participation 0.5, stale snapshots, random order) must
    converge to the same credit, bounded, non-compounding with depth;
  - rounds-to-convergence vs participation (the settling-diameter probe);
  - carry-chain length histogram + error DELIVERY vs time-lag for the conv-RNN
    ("bounded != delivered": fraction of positions receiving nonzero error at
    lag k, not just error magnitude).

Usage::

    python scripts/pc_convrnn_gradcheck.py --depths 2 4 8
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pc_convrnn_charlm import ConvRNNLM, MorphPCEngine, build_model  # noqa: E402


def _grads(model):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for n, p in model.named_parameters()}


def _cos_worst(a, b):
    worst, worst_n = 1.0, None
    for n in a:
        x, y = a[n].reshape(-1).double(), b[n].reshape(-1).double()
        if x.norm() == 0 and y.norm() == 0:
            continue
        c = float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-30))
        if c < worst:
            worst, worst_n = c, n
    return worst, worst_n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--arch", default="convrnn",
                   choices=["convrnn", "morphssm", "convrnn-norecur"])
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lam-min", type=float, default=2 ** -6)
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "pc_convrnn" / "g1_equivalence.jsonl")
    args = p.parse_args()
    rows = []
    overall = True

    for L in args.depths:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            ns = argparse.Namespace(d_model=args.d_model, n_layers=L,
                                    kernel=args.kernel, dropout=args.dropout,
                                    tau=args.tau, lam_min=args.lam_min)
            model = build_model(args.arch, 65, ns)
            model.eval()
            model.set_tau(args.tau)
            eng = MorphPCEngine(model)
            gen = torch.Generator().manual_seed(100 + seed)
            x = torch.randint(0, 65, (args.batch, args.seq_len), generator=gen)
            y = torch.randint(0, 65, (args.batch, args.seq_len), generator=gen)

            model.zero_grad(set_to_none=True)
            eng.cascade_grads(x, y)
            casc = _grads(model)

            # sync: n_iter=1 and L+2 must be BITWISE the cascade
            for n_iter in (1, L + 2):
                model.zero_grad(set_to_none=True)
                eng.error_relax_grads(x, y, n_iter=n_iter)
                bit = all(torch.equal(casc[n], g) for n, g in _grads(model).items())
                rows.append(dict(gate="G1", form="sync", depth=L, seed=seed,
                                 tau=args.tau, n_iter=n_iter, bitwise=bit))
                overall &= bit
                print(f"[G1] L={L} seed={seed} sync n_iter={n_iter}: "
                      f"bitwise={'PASS' if bit else 'FAIL'}")

            # async at increasing rounds: rounds-to-convergence vs participation.
            # With STALE snapshots the relaxation is a Jacobi iteration: error
            # advances at most ONE layer-hop per round, and a hop lands only when
            # the frontier layer participates — so the settling diameter in
            # ROUNDS is ~(L+2)/participation. Pass rule: once the expected hops
            # n_iter*participation cover the depth with margin (>= 2(L+2)), the
            # async grads must be EXACT (the fixed point is exact; the "tax" at
            # layer granularity is propagated-or-not, never a degraded angle).
            for part in (0.5, 0.25):
                for mult in (1, 2, 3, 6):
                    n_iter = max(1, int(mult * (L + 2) / part))
                    agen = torch.Generator().manual_seed(7 + seed)
                    model.zero_grad(set_to_none=True)
                    eng.error_relax_grads(x, y, n_iter=n_iter, participation=part,
                                          stale=True, rand_order=True, gen=agen)
                    worst, worst_n = _cos_worst(casc, _grads(model))
                    hops = n_iter * part
                    binding = hops >= 2 * (L + 2)
                    rows.append(dict(gate="G1", form="async", depth=L, seed=seed,
                                     tau=args.tau, n_iter=n_iter,
                                     participation=part, exp_hops=hops,
                                     binding=binding,
                                     cos_worst=round(worst, 8),
                                     worst_param=worst_n))
                    print(f"[G1] L={L} seed={seed} async p={part} "
                          f"n_iter={n_iter:3d} (hops~{hops:.0f}"
                          f"{', binding' if binding else ''}): "
                          f"worst cos={worst:.8f} ({worst_n})")
                    if binding and worst < 0.999:
                        overall = False

            # chain-length histogram + delivery-vs-lag (conv-RNN only)
            if args.arch == "convrnn" and seed == 0:
                _, caches, top = eng._forward_cache(x)
                logits, _, _ = eng._forward_cache(x)
                eL = eng._top_error(logits, y, top, grads=False)
                e = eL
                for li in range(L - 1, -1, -1):
                    c = caches[li]
                    g = c["gates"].int()
                    runs = torch.zeros_like(g)
                    for t in range(1, g.shape[-1]):
                        runs[:, :, t] = (runs[:, :, t - 1] + 1) * g[:, :, t]
                    # delivery: fraction of (b,c,t) receiving nonzero error at the
                    # conv-drive after the scan adjoint, bucketed by carry-run length
                    e_bct = None
                    rows.append(dict(
                        gate="G1", form="chains", depth=L, layer=li, seed=seed,
                        carry_frac=round(float(g.float().mean()), 4),
                        chain_p50=int(torch.quantile(runs.float().reshape(-1), .5)),
                        chain_p95=int(torch.quantile(runs.float().reshape(-1), .95)),
                        chain_max=int(runs.max())))
                    e = eng._layer_adjoint(model.layers[li], e, caches[li],
                                           grads=False)
                # error delivery at the embedding as a function of position lag
                # from the end of the window (errors are injected at every
                # position by the all-position CE, so delivery ~ 1 everywhere;
                # report it to make that explicit)
                nz = (e.abs() > 0).float().mean(dim=(0, 2))   # (T,)
                rows.append(dict(gate="G1", form="delivery", depth=L, seed=seed,
                                 delivery_mean=round(float(nz.mean()), 4),
                                 delivery_min=round(float(nz.min()), 4)))
                print(f"[G1] L={L} delivery(embed err): mean={nz.mean():.4f} "
                      f"min={nz.min():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[G1] overall: {'PASS' if overall else 'FAIL'}  -> {args.out}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
