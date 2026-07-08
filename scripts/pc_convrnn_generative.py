"""G6: generative PC inference on cascade-trained tropical sequence models.

The inference half of the program goal (results/pc_convrnn/PREREG.md): the SAME
checkpoint runs feedforward for ordinary decoding, and runs ITERATIVE energy
relaxation for the PC-unique modes — infilling, denoising — that a causal LM
has no native way to do.

Reduced-form settling (the DAG corollary applied at inference): with the input's
observed positions clamped and internal states free, the settled internal states
equal the forward pass in closed form, so generative inference needs iteration
ONLY over the genuinely free variables — the unknown/untrusted token embeddings
— descending the remaining energy (the CE of observed next-tokens) through the
soft forward. Full-state settling is retained as the G6a control, which must
converge TO the forward pass (the honest no-op demonstration).

Modes:
  g6a  full-state relaxation with input clamped -> must reproduce the forward
  g6b  infilling: clamp prefix+suffix, relax the gap embeddings, decode by
       nearest embedding; vs left-to-right greedy and best-of-N (suffix-
       rescored) baselines on the SAME checkpoint
  g6c  denoising: corrupt r% of tokens, clamp the trusted rest, relax, recover

Relaxation runs at soft tau (piecewise-linear energy at tau=0 chatters); all
evaluation is at hard tau=0 — soft-relax / hard-eval, the disclosed closure
boundary for this mode only.

Usage::

    python scripts/pc_convrnn_generative.py --gates g6a           # control (any init)
    python scripts/pc_convrnn_generative.py --gates g6b g6c --ckpt <path>
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

import tikitorch as tt                                     # noqa: E402
from pc_convrnn_charlm import ConvRNNLM, build_model       # noqa: E402
from nanogpt_shakespeare import CharData                   # noqa: E402


def forward_from_emb(model, e: torch.Tensor) -> torch.Tensor:
    """(B, T, D) embeddings -> logits, bypassing the token lookup (eval mode)."""
    h = e
    for layer in model.layers:
        h = layer(h)
    return model.output_head(model.ln_final(h))


class GenerativePC:
    """State-relaxation inference for MorphSSM-shaped models (G6)."""

    def __init__(self, model: nn.Module, tau_relax: float = 0.5):
        self.model = model
        self.tau_relax = tau_relax

    # -- G6a: full-state settling must converge to the forward pass ---------
    def settle_states(self, ids: torch.Tensor, n_steps: int = 200,
                      gamma: float = 0.125, order: str = "jacobi"):
        """order="jacobi": naive parallel energy descent (the slow crawl — the
        C3-at-inference conditioning problem, reported honestly). order="topo":
        settle in topological order — the closed-form fixed point in ONE sweep
        per layer, the inference-side mirror of the cascade theorem."""
        m = self.model
        m.eval()
        m.set_tau(self.tau_relax)
        with torch.no_grad():
            s0 = m.embed(ids)
            ref = [s0]
            h = s0
            for layer in m.layers:
                h = layer(h)
                ref.append(h)
        if order == "topo":
            with torch.no_grad():
                states = [s0] + [torch.zeros_like(r) for r in ref[1:]]
                for l, layer in enumerate(m.layers):
                    states[l + 1] = layer(states[l])
                rel = max(float((s - r).norm() / r.norm().clamp_min(1e-30))
                          for s, r in zip(states[1:], ref[1:]))
            m.set_tau(0.0)
            return [(len(m.layers), 0.0, rel)]
        states = [s0] + [torch.zeros_like(r).requires_grad_(True) for r in ref[1:]]
        for p in m.parameters():
            p.requires_grad_(False)
        traj = []
        for step in range(n_steps):
            E = states[0].new_zeros(())
            for l, layer in enumerate(m.layers):
                E = E + 0.5 * ((states[l + 1] - layer(states[l])) ** 2).sum()
            grads = torch.autograd.grad(E, states[1:])
            with torch.no_grad():
                for s, g in zip(states[1:], grads):
                    s -= gamma * g
                if step % 20 == 0 or step == n_steps - 1:
                    rel = max(float((s - r).norm() / r.norm().clamp_min(1e-30))
                              for s, r in zip(states[1:], ref[1:]))
                    traj.append((step, float(E.item()), rel))
            for i in range(1, len(states)):
                states[i] = states[i].detach().requires_grad_(True)
        for p in m.parameters():
            p.requires_grad_(True)
        m.set_tau(0.0)
        return traj

    # -- reduced-form generative relaxation ---------------------------------
    def relax_tokens(self, ids: torch.Tensor, free: torch.Tensor,
                     n_steps: int = 200, lr: float = 0.05,
                     init: str = "greedy", project_every: int = 25,
                     light_cone: int = 32):
        """Relax embeddings at ``free`` positions to minimize observed-token CE.

        ids: (B, T) tokens (values at free positions are untrusted).
        free: (B, T) bool. Three machinery choices (each fixed a measured
        failure of the naive form, which decoded to 0% exact-match):
        * init="greedy": start from the causal left-to-right fill — PC then
          REFINES the causal guess with downstream (suffix) information;
          "mean"/"obs" kept as ablations.
        * projected relaxation: every ``project_every`` steps snap z to the
          nearest embedding row — keeps the search on the discrete token
          manifold (unconstrained z satisfies the CE objective while decoding
          to nothing).
        * light-cone objective: CE is counted only at observed-target
          positions within ``light_cone`` of a free position (remote positions
          dilute the signal with terms the free variables cannot affect).
        Returns decoded tokens (B, T): nearest-embedding at free positions.
        """
        m = self.model
        m.eval()
        m.set_tau(self.tau_relax)
        for p in m.parameters():
            p.requires_grad_(False)
        B, T = ids.shape
        with torch.no_grad():
            if init == "greedy":
                start = l2r_fill(m, ids, free, n_samples=1)
                m.set_tau(self.tau_relax)                    # l2r_fill hardens
            else:
                start = ids
            e_obs = m.embed(ids)
            z0 = m.embed(start).clone()
            if init == "mean":
                z0[free] = m.embed.weight.mean(0)
        z = z0.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        # objective mask: observed targets within the light cone of any free var
        tgt_obs = ~free[:, 1:]                               # (B, T-1)
        near = torch.zeros_like(tgt_obs)
        fpos = free.float()
        for b in range(B):
            idxs = free[b].nonzero(as_tuple=True)[0]
            if idxs.numel():
                lo = max(0, int(idxs.min()) - 2)
                hi = min(T - 1, int(idxs.max()) + light_cone)
                near[b, lo:hi] = True
        w = (tgt_obs & near).float()
        clamp = ~free
        for step in range(n_steps):
            opt.zero_grad()
            zc = torch.where(clamp.unsqueeze(-1), e_obs, z)
            logits = forward_from_emb(m, zc)[:, :-1]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1), reduction="none").reshape(B, T - 1)
            loss = (ce * w).sum() / w.sum().clamp_min(1)
            loss.backward()
            opt.step()
            if project_every and (step + 1) % project_every == 0:
                with torch.no_grad():
                    d = torch.cdist(z.reshape(-1, z.shape[-1]), m.embed.weight)
                    z.copy_(m.embed.weight[d.argmin(-1)].reshape(B, T, -1))
        with torch.no_grad():
            d = torch.cdist(z.detach().reshape(-1, z.shape[-1]),
                            m.embed.weight)                 # (B*T, V)
            nearest = d.argmin(-1).reshape(B, T)
            out = torch.where(free, nearest, ids)
        for p in m.parameters():
            p.requires_grad_(True)
        m.set_tau(0.0)
        return out


    # -- discrete coordinate relaxation (hard-tropical-native settling) ------
    @torch.no_grad()
    def relax_tokens_coord(self, ids: torch.Tensor, free: torch.Tensor,
                           sweeps: int = 3, light_cone: int = 32,
                           init: str = "greedy"):
        """Coordinate descent on the SAME clamped energy, over the DISCRETE
        token set: per free position, try every vocab token and keep the
        lightcone-CE minimizer; sweep until fixed point or ``sweeps``. This is
        state relaxation in the hard-tropical-native form (the state space is
        discrete), and an upper reference for what the continuous-embedding
        gradient form could find on the same objective."""
        m = self.model
        m.eval()
        m.set_tau(0.0)
        V = m.output_head.out_features
        cur = l2r_fill(m, ids, free, n_samples=1) if init == "greedy" else ids.clone()
        B, T = ids.shape
        for _ in range(sweeps):
            changed = False
            for t in range(T):
                col = free[:, t]
                if not col.any():
                    continue
                # batch all candidate tokens at position t for the active rows
                rows = col.nonzero(as_tuple=True)[0]
                cand = cur[rows].repeat_interleave(V, dim=0)      # (n*V, T)
                cand[:, t] = torch.arange(V, device=ids.device).repeat(rows.numel())
                logits = m(cand)[:, :-1]
                ce = F.cross_entropy(
                    logits.reshape(-1, V), cand[:, 1:].reshape(-1),
                    reduction="none").reshape(rows.numel(), V, T - 1)
                lo = max(0, t - 2)
                hi = min(T - 1, t + light_cone)
                mask = (~free[rows, 1:]).float()[:, None, :]
                mask[:, :, :lo] = 0
                mask[:, :, hi:] = 0
                score = (ce * mask).sum(-1)                        # (n, V)
                best = score.argmin(-1)
                new = torch.where(col[rows], best, cur[rows, t])
                changed |= bool((new != cur[rows, t]).any())
                cur[rows, t] = new
            if not changed:
                break
        return cur


# ---------------------------------------------------------------------------
# Baselines (same checkpoint, causal-native)
# ---------------------------------------------------------------------------

@torch.no_grad()
def l2r_fill(model, ids, free, n_samples=1, temperature=0.8, gen=None):
    """Left-to-right fill of free positions from the left context only; if
    n_samples>1, rescore each candidate by FULL-window hard CE (suffix-aware
    best-of-N — the causal model's honest inference-compute baseline)."""
    model.eval()
    model.set_tau(0.0)
    B, T = ids.shape
    best, best_ce = None, None
    for s in range(max(1, n_samples)):
        cand = ids.clone()
        for t in range(1, T):
            col = free[:, t]
            if not col.any():
                continue
            logits = model(cand[:, :t])[:, -1]
            if n_samples == 1:
                nxt = logits.argmax(-1)
            else:
                probs = F.softmax(logits / temperature, -1)
                nxt = torch.multinomial(probs.cpu(), 1, generator=gen)[:, 0].to(ids.device)
            cand[:, t] = torch.where(col, nxt, cand[:, t])
        logits = model(cand)[:, :-1]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             cand[:, 1:].reshape(-1), reduction="none"
                             ).reshape(B, T - 1).mean(-1)
        if best is None:
            best, best_ce = cand, ce
        else:
            better = ce < best_ce
            best = torch.where(better.unsqueeze(-1), cand, best)
            best_ce = torch.where(better, ce, best_ce)
    return best


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def load_ckpt(path, data, args):
    ck = torch.load(path, map_location="cpu")
    ns = argparse.Namespace(d_model=ck["config"]["d_model"],
                            n_layers=ck["config"]["n_layers"],
                            kernel=ck["config"]["kernel"], dropout=0.0,
                            tau=0.5, lam_min=ck["config"]["lam_min"])
    model = build_model(ck["arch"], ck["config"]["vocab"], ns)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.set_tau(0.0)
    return model, ck["arch"]


def windows(data, n, T, seed):
    gen = torch.Generator().manual_seed(seed)
    x, _ = data.batch("val", n, T, gen)
    return x


def run_g6a(args, out):
    torch.manual_seed(0)
    ns = argparse.Namespace(d_model=64, n_layers=3, kernel=7, dropout=0.0,
                            tau=0.5, lam_min=2 ** -6)
    for arch in ("morphssm", "convrnn"):
        model = build_model(arch, 65, ns)
        gpc = GenerativePC(model, tau_relax=args.tau_relax)
        ids = torch.randint(0, 65, (4, 48))
        topo = gpc.settle_states(ids, order="topo")
        traj = gpc.settle_states(ids, n_steps=args.settle_steps, gamma=0.125)
        row = dict(gate="G6a", arch=arch, topo_rel=topo[0][2],
                   traj=[(s, round(e, 4), round(r, 6)) for s, e, r in traj])
        # pass: topological settle exact; naive Jacobi converges >=50x toward
        # the forward, then floors in a LIMIT CYCLE — the pow-2 norm factors
        # (stop-grad, recomputed each settle step) flip at rounding boundaries,
        # a tropical-specific C3-at-inference pathology. This is exactly why
        # the generative gates (G6b/c) use the reduced form: internal states
        # settled by the forward (the topo sweep), iteration only on the free
        # token embeddings.
        rels = [r for _, _, r in traj]
        ok = topo[0][2] < 1e-6 and rels[-1] < 0.02 * rels[0]
        print(f"[G6a] {arch}: topo relL2={topo[0][2]:.2e} (exact in L sweeps); "
              f"jacobi {rels[0]:.3f} -> {rels[-1]:.4f} over {args.settle_steps} "
              f"steps ({'PASS' if ok else 'FAIL'})")
        row["pass"] = ok
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


@torch.no_grad()
def lightcone_ce(model, filled, ids, free, light_cone=32):
    """Hard-eval CE at observed targets near the gap, for a candidate fill —
    the objective the relaxation descends, evaluated at tau=0."""
    model.set_tau(0.0)
    B, T = ids.shape
    logits = model(filled)[:, :-1]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         ids[:, 1:].reshape(-1), reduction="none").reshape(B, T - 1)
    tgt_obs = ~free[:, 1:]
    near = torch.zeros_like(tgt_obs)
    for b in range(B):
        idxs = free[b].nonzero(as_tuple=True)[0]
        if idxs.numel():
            near[b, max(0, int(idxs.min()) - 2):
                 min(T - 1, int(idxs.max()) + light_cone)] = True
    w = (tgt_obs & near).float()
    return float((ce * w).sum() / w.sum().clamp_min(1))


def run_g6b(args, out, data):
    model, arch = load_ckpt(args.ckpt, data, args)
    gpc = GenerativePC(model, tau_relax=args.tau_relax)
    gen = torch.Generator().manual_seed(1234)
    for span in args.spans:
        x = windows(data, args.n_windows, args.seq_len, seed=41 + span)
        a = (args.seq_len - span) // 2
        free = torch.zeros_like(x, dtype=torch.bool)
        free[:, a:a + span] = True
        truth = x.clone()
        t0 = time.time()
        pc = gpc.relax_tokens(x, free, n_steps=args.relax_steps, lr=args.relax_lr,
                              init=args.relax_init)
        t_pc = time.time() - t0
        greedy = l2r_fill(model, x, free, n_samples=1)
        bo16 = l2r_fill(model, x, free, n_samples=16, gen=gen)
        t1 = time.time()
        coord = gpc.relax_tokens_coord(x, free, sweeps=args.coord_sweeps)
        t_coord = time.time() - t1

        def em(y):
            return float((y[free] == truth[free]).float().mean())
        # signal diagnostic: does the objective prefer the TRUTH over greedy?
        # If not, no optimizer can beat greedy on this metric — task/model
        # signal limitation, not relaxation machinery.
        ce_truth = lightcone_ce(model, truth, x, free)
        ce_greedy = lightcone_ce(model, greedy, x, free)
        ce_pc = lightcone_ce(model, pc, x, free)
        ce_coord = lightcone_ce(model, coord, x, free)
        row = dict(gate="G6b", arch=arch, ckpt=str(args.ckpt), span=span,
                   n_windows=args.n_windows, relax_steps=args.relax_steps,
                   relax_lr=args.relax_lr, tau_relax=args.tau_relax,
                   init=args.relax_init, coord_sweeps=args.coord_sweeps,
                   em_pc=round(em(pc), 4), em_coord=round(em(coord), 4),
                   em_greedy=round(em(greedy), 4),
                   em_bestof16=round(em(bo16), 4),
                   ce_truth=round(ce_truth, 4), ce_greedy=round(ce_greedy, 4),
                   ce_pc=round(ce_pc, 4), ce_coord=round(ce_coord, 4),
                   wall_pc_s=round(t_pc, 1), wall_coord_s=round(t_coord, 1))
        print(f"[G6b] {arch} span={span}: PC {row['em_pc']:.3f}  "
              f"coord {row['em_coord']:.3f}  greedy {row['em_greedy']:.3f}  "
              f"best-of-16 {row['em_bestof16']:.3f} | lightcone-CE truth "
              f"{ce_truth:.3f} greedy {ce_greedy:.3f} pc {ce_pc:.3f} "
              f"coord {ce_coord:.3f}")
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


def run_g6c(args, out, data):
    model, arch = load_ckpt(args.ckpt, data, args)
    gpc = GenerativePC(model, tau_relax=args.tau_relax)
    for rate in args.rates:
        x = windows(data, args.n_windows, args.seq_len, seed=97 + int(100 * rate))
        truth = x.clone()
        gen = torch.Generator().manual_seed(int(1000 * rate))
        free = torch.rand(x.shape, generator=gen) < rate
        free[:, 0] = False
        x_cor = x.clone()
        x_cor[free] = torch.randint(0, data.vocab, (int(free.sum()),),
                                    generator=gen)
        pc = gpc.relax_tokens(x_cor, free, n_steps=args.relax_steps,
                              lr=args.relax_lr)
        fwd = l2r_fill(model, x_cor, free, n_samples=1)     # causal baseline

        def rec(y):
            return float((y[free] == truth[free]).float().mean())
        row = dict(gate="G6c", arch=arch, ckpt=str(args.ckpt), rate=rate,
                   n_windows=args.n_windows,
                   rec_pc=round(rec(pc), 4), rec_greedy=round(rec(fwd), 4),
                   rec_leave_corrupt=round(rec(x_cor), 4))
        print(f"[G6c] {arch} rate={rate}: PC {row['rec_pc']:.3f}  "
              f"greedy {row['rec_greedy']:.3f}  corrupt-floor "
              f"{row['rec_leave_corrupt']:.3f}")
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gates", nargs="+", default=["g6a"],
                   choices=["g6a", "g6b", "g6c"])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="cascade-trained checkpoint (required for g6b/g6c)")
    p.add_argument("--text", type=Path, default=REPO / "data" / "tinyshakespeare.txt")
    p.add_argument("--out", type=Path, default=REPO / "results" / "pc_convrnn")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--spans", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--relax-steps", type=int, default=200)
    p.add_argument("--relax-lr", type=float, default=0.05)
    p.add_argument("--coord-sweeps", type=int, default=3)
    p.add_argument("--relax-init", type=str, default="greedy",
                   choices=["greedy", "mean", "obs"])
    p.add_argument("--settle-steps", type=int, default=200)
    p.add_argument("--tau-relax", type=float, default=0.5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"g6_generative{('_' + args.tag) if args.tag else ''}.jsonl"
    data = CharData(args.text) if set(args.gates) & {"g6b", "g6c"} else None
    if "g6a" in args.gates:
        run_g6a(args, out)
    if "g6b" in args.gates:
        assert args.ckpt, "--ckpt required for g6b"
        run_g6b(args, out, data)
    if "g6c" in args.gates:
        assert args.ckpt, "--ckpt required for g6c"
        run_g6c(args, out, data)


if __name__ == "__main__":
    main()
