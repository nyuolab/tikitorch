"""Tropical-native stability machinery for DEEP dyadic PC.

Baseline dyadic PC degrades 0.89 (d8) -> 0.72 (d16) -> ? (d24): the no-residual
wall Qi et al. fight with precision-weighting + residual-energy control. Here we
test MULTIPLIER-FREE tropical duals:

  residual : tropical skip  f(x) = max(relu(W x), x)   (max = free, bounded, idempotent)
  discount : discounted skip f(x) = max(relu(W x), x - delta)  (contraction)
  norm     : max-shift normalize the layer input  z = x - max(x)  (subtractor)
  precision: per-layer power-of-2 error gain in settling (additive/shift precision-weighting)

Adam weight update used to probe the ARCHITECTURE ceiling (does the trick let the
optimisation reach depth); the winner is re-checked with the multiplier-free
dyadic-backward update afterwards.
"""
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch.nn import DyadicLinear, dyadic_quantize


def get_mnist(down, n_train, n_test):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((down, down), antialias=True),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    tr = datasets.MNIST("data", train=True, download=True, transform=tf)
    te = datasets.MNIST("data", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(n_train)])
    Ytr = torch.tensor([tr[i][1] for i in range(n_train)])
    Xte = torch.stack([te[i][0] for i in range(n_test)])
    Yte = torch.tensor([te[i][1] for i in range(n_test)])
    return Xtr, Ytr, Xte, Yte


class DeepDyadicPC(nn.Module):
    def __init__(self, dims, gamma=0.1, residual=False, discount=0.0, norm=False,
                 precision=False):
        super().__init__()
        self.depth = len(dims) - 1
        self.gamma = gamma
        self.residual = residual
        self.discount = discount
        self.norm = norm
        self.dims = dims
        self.layers = nn.ModuleList([DyadicLinear(dims[l], dims[l + 1])
                                     for l in range(self.depth)])
        # additive (power-of-2) precision: per-layer integer shift on the error
        # weight, learned-free here as a fixed schedule boosting deeper layers.
        if precision:
            # GENTLE capped power-of-2 error gain: deeper layers boosted, but
            # capped at 4x (2 bits) and stepped every 6 layers, so settling does
            # not blow up (the un-capped 2^l schedule diverged).
            self.prec = [1 << min(l // 6, 2) for l in range(self.depth)]
        else:
            self.prec = [1] * self.depth

    def _norm(self, x):
        if not self.norm:
            return x
        return x - x.max(dim=-1, keepdim=True).values        # max-shift (subtractor)

    def f(self, l, x):
        pre = self.layers[l](self._norm(x))
        interior = l < self.depth - 1
        out = torch.relu(pre) if interior else pre
        if self.residual and interior and x.shape[-1] == out.shape[-1]:
            out = torch.maximum(out, x - self.discount)      # tropical residual skip
        return out

    @torch.no_grad()
    def feedforward(self, x0):
        s = [x0]
        for l in range(self.depth):
            s.append(self.f(l, s[-1]))
        return s

    def energy(self, states):
        E = states[0].new_zeros(())
        for l in range(self.depth):
            mu = self.f(l, states[l])
            # power-of-2 precision weight on this layer's error (a shift)
            E = E + 0.5 * self.prec[l] * ((states[l + 1] - mu) ** 2).sum()
        return E

    def infer(self, x0, target, n_steps):
        states = [s.detach() for s in self.feedforward(x0)]
        states[self.depth] = target
        free = list(range(1, self.depth))
        for p in self.parameters():
            p.requires_grad_(False)
        for i in free:
            states[i] = states[i].clone().requires_grad_(True)
        for _ in range(n_steps):
            grads = torch.autograd.grad(self.energy(states), [states[i] for i in free])
            with torch.no_grad():
                for i, g in zip(free, grads):
                    states[i] -= self.gamma * g
            for i in free:
                states[i] = states[i].detach().requires_grad_(True)
        for p in self.parameters():
            p.requires_grad_(True)
        return [s.detach() for s in states]

    # ---- Gauss-Seidel settling on the TRUE energy: iterative PC, depth-stable --
    @torch.no_grad()
    def _route_l(self, l, x):
        """Recompute layer l's hard routing from the CURRENT input x (not frozen):
        returns (relu_mask, gate-or-None, has_skip, relu_out, pre)."""
        pre = self.layers[l](x)
        interior = l < self.depth - 1
        relu_out = torch.relu(pre) if interior else pre
        relu_mask = (pre > 0).float() if interior else torch.ones_like(pre)
        if self.residual and interior and x.shape[-1] == relu_out.shape[-1]:
            return relu_mask, (relu_out >= x).float(), True, relu_out, pre
        return relu_mask, None, False, relu_out, pre

    @torch.no_grad()
    def _AT(self, l, u, x):
        """A_l^T u with the routing recomputed at the current x (true Jacobian)."""
        relu_mask, gate, has_skip, _, _ = self._route_l(l, x)
        if has_skip:
            return (gate * relu_mask * u) @ dyadic_quantize(self.layers[l].weight) \
                + (1.0 - gate) * u
        return (relu_mask * u) @ dyadic_quantize(self.layers[l].weight)

    @torch.no_grad()
    def _diag_precond(self, l, x):
        """Per-neuron diagonal Hessian of the state-l energy block (exact, because
        the binary gates kill the cross-terms): D[i] = 1 + Σ_j(active_j)·W[j,i]^2
        (+ 1 where the residual skip won). Dyadic W -> W^2 = 2^(2k) -> D is an
        add-tree of powers of two; 1/round_pow2(D) is a shift."""
        relu_mask, gate, has_skip, _, _ = self._route_l(l, x)
        active = relu_mask * gate if has_skip else relu_mask          # (B, out) in {0,1}
        W2 = dyadic_quantize(self.layers[l].weight) ** 2              # (out, in) = 2^(2k)
        D = 1.0 + active @ W2                                          # (B, in)
        if has_skip:
            D = D + (1.0 - gate)                                      # skip identity curvature
        return D

    @torch.no_grad()
    def _cascade_latents(self, x0, target, warm_alpha):
        """SKIP-init (idea 3): one global top-down cascade pass that propagates the
        clamped-top error across ALL depth in a single sweep (via the residual
        skip-highways), warm-starting each free latent toward equilibrium. This is
        the 'skip connection' shortcut; gentle GS then refines IN-basin."""
        acts, cache = [x0], []
        h = x0
        for l in range(self.depth):
            pre = self.layers[l](h)
            interior = l < self.depth - 1
            relu_out = torch.relu(pre) if interior else pre
            relu_mask = (pre > 0).float() if interior else torch.ones_like(pre)
            if self.residual and interior and h.shape[-1] == relu_out.shape[-1]:
                gate, out, has_skip = (relu_out >= h).float(), torch.maximum(relu_out, h), True
            else:
                gate, out, has_skip = None, relu_out, False
            cache.append((relu_mask, gate, has_skip))
            acts.append(out)
            h = out
        warm = [a.clone() for a in acts]
        warm[self.depth] = target
        e = acts[-1] - target                                       # error at top state
        for l in reversed(range(self.depth)):
            relu_mask, gate, has_skip = cache[l]
            delta = (e * gate if has_skip else e) * relu_mask       # error at layer-l pre-act
            if l > 0:
                e_down = delta @ dyadic_quantize(self.layers[l].weight)   # Wᵀ·delta
                if has_skip:
                    e_down = e_down + e * (1.0 - gate)              # unit-gain skip highway
                e = e_down                                          # error at state l
                warm[l] = acts[l] - warm_alpha * e                  # skip-propagated warm init
        return warm

    @torch.no_grad()
    def settle_gs(self, x0, target, n_sweeps, alpha=1.0, anneal=False, warm=False,
                  warm_alpha=0.5):
        """Diagonal-preconditioned Gauss-Seidel settling (true energy). Knobs for
        the three explorations:
          alpha<1            -> DAMPED step (idea 1: stay nearer the basin)
          anneal=True        -> step ramps 0.25->1.0 over sweeps (idea 2: trust region)
          warm=True          -> cascade skip-init the latents (idea 3)
        Reaches the PC equilibrium (keeps PC's advantages), multiplier-free."""
        if warm:
            states = self._cascade_latents(x0, target, warm_alpha)
        else:
            states = [s.clone() for s in self.feedforward(x0)]
            states[self.depth] = target
        free = list(range(1, self.depth))
        for t in range(n_sweeps):
            a = (0.25 + 0.75 * (t / max(n_sweeps - 1, 1))) if anneal else alpha
            for l in free + free[::-1]:                    # symmetric GS sweep
                eps_l = states[l] - self.f(l - 1, states[l - 1])
                eps_up = states[l + 1] - self.f(l, states[l])
                grad = eps_l - self._AT(l, eps_up, states[l])
                states[l] = states[l] - a * grad / self._diag_precond(l, states[l])
        return states

    @torch.no_grad()
    def settle_nudge(self, x0, target, n_sweeps, beta=0.3, async_frac=1.0, gen=None,
                     stale=False, rand_order=False):
        """NUDGED iterative settling (the convergent regime). The output layer is
        FREE (init = feed-forward), softly pulled toward the target by `beta` —
        the small-nudge / EqProp limit where PC converges near the feed-forward
        basin. Diagonal-preconditioned GS.

        Async knobs (the distributed model the cascade CANNOT match — its backward
        sweep is strictly ordered, e_l needs e_{l+1}):
          async_frac<1 : each node updates with prob async_frac per pass (partial participation)
          stale=True   : neighbour reads come from a frozen round-start snapshot
                         (stale-by-1 / Jacobi-style — models communication lag)
          rand_order   : random per-pass update order (no global schedule)
        """
        states = [s.clone() for s in self.feedforward(x0)]
        L = self.depth
        free = list(range(1, L + 1))                        # interior + OUTPUT (free, nudged)
        for _ in range(n_sweeps):
            snap = [s.clone() for s in states] if stale else states   # stale vs live reads
            order = free + free[::-1]
            if rand_order:
                order = [order[i] for i in torch.randperm(len(order), generator=gen).tolist()]
            for l in order:
                if async_frac < 1.0 and torch.rand(1, generator=gen).item() >= async_frac:
                    continue                                # node "doesn't report" this pass
                eps_l = states[l] - self.f(l - 1, snap[l - 1])
                if l < L:
                    eps_up = snap[l + 1] - self.f(l, states[l])
                    grad = eps_l - self._AT(l, eps_up, states[l])
                    D = self._diag_precond(l, states[l])
                else:                                       # output: pred error + soft target nudge
                    grad = eps_l + beta * (states[l] - target)
                    D = 1.0 + beta
                states[l] = states[l] - grad / D
        return states

    @torch.no_grad()
    def settle_mom(self, x0, target, n_sweeps, gamma=0.3, beta=0.7):
        """Momentum (heavy-ball) on gentle preconditioned GS. Small step (stays in
        basin) + momentum (accelerates the stiff chain, O(k)->O(sqrt k))."""
        states = [s.clone() for s in self.feedforward(x0)]
        states[self.depth] = target
        free = list(range(1, self.depth))
        vel = {l: torch.zeros_like(states[l]) for l in free}
        for _ in range(n_sweeps):
            for l in free + free[::-1]:
                eps_l = states[l] - self.f(l - 1, states[l - 1])
                eps_up = states[l + 1] - self.f(l, states[l])
                d = (eps_l - self._AT(l, eps_up, states[l])) / self._diag_precond(l, states[l])
                vel[l] = beta * vel[l] + d
                states[l] = states[l] - gamma * vel[l]
        return states

    @torch.no_grad()
    def settle_mg(self, x0, target, n_cycles, smooth=3, coarse_alpha=0.5, smooth_alpha=0.4):
        """Two-grid (multigrid-ish) settling: each cycle = a few GENTLE local GS
        smoothing sweeps (kill high-frequency error, stay in basin) + ONE global
        top-down correction sweep (the cascade/coarse mode that propagates the
        clamped-label boundary across all depth, fixing the slow long-range mode)."""
        states = [s.clone() for s in self.feedforward(x0)]
        states[self.depth] = target
        free = list(range(1, self.depth))
        for _ in range(n_cycles):
            for _ in range(smooth):                        # pre-smooth: local high-freq
                for l in free + free[::-1]:
                    eps_l = states[l] - self.f(l - 1, states[l - 1])
                    eps_up = states[l + 1] - self.f(l, states[l])
                    grad = eps_l - self._AT(l, eps_up, states[l])
                    states[l] = states[l] - smooth_alpha * grad / self._diag_precond(l, states[l])
            for l in reversed(free):                       # coarse: global top-down boundary propagation
                r_up = states[l + 1] - self.f(l, states[l])
                states[l] = states[l] - coarse_alpha * self._AT(l, r_up, states[l])
        return states

    @torch.no_grad()
    def gs_weight_grads(self, states):
        """Local PC weight grads at the settled equilibrium (dyadic, multiplier-free)."""
        for l in range(self.depth):
            relu_mask, gate, has_skip, _, _ = self._route_l(l, states[l])
            eps = states[l + 1] - self.f(l, states[l])     # error at this layer
            a = relu_mask * eps
            if has_skip:
                a = gate * a
            self.layers[l].weight.grad = -(dyadic_quantize(a).transpose(0, 1)
                                           @ dyadic_quantize(states[l]))
            self.layers[l].bias.grad = -dyadic_quantize(a).sum(0)

    @torch.no_grad()
    def predict_logits(self, x0):
        return self.feedforward(x0)[-1]

    @torch.no_grad()
    def dyadic_mf_step(self, states, lr_shift):
        """Gate-aware, fully multiplier-free dyadic-backward weight update.

        Accounts for the residual max-gate: the weight grad flows only through
        the transformed branch when it won the max. Both operands quantized to
        ±2^k so the contraction is a shift-add tree.
        """
        lr = 1.0 / (1 << lr_shift)
        for l, layer in enumerate(self.layers):
            z = self._norm(states[l])
            pre = layer(z)
            interior = l < self.depth - 1
            relu_out = torch.relu(pre) if interior else pre
            relu_mask = (pre > 0) if interior else torch.ones_like(pre, dtype=torch.bool)
            if self.residual and interior and states[l].shape[-1] == relu_out.shape[-1]:
                skip = states[l] - self.discount
                mu = torch.maximum(relu_out, skip)
                gate = relu_out >= skip                      # transformed branch won the max
            else:
                mu = relu_out
                gate = torch.ones_like(relu_out, dtype=torch.bool)
            a = (mu - states[l + 1]) * relu_mask * gate      # dE/dpre (through W-branch only)
            da = dyadic_quantize(a)
            db = dyadic_quantize(z)
            G = da.transpose(0, 1) @ db                      # shift-add tree
            layer.weight.data -= lr * torch.sign(G)
            layer.bias.data -= lr * torch.sign(da.sum(0))


@torch.no_grad()
def cascade_grads(model, x0, target):
    """UNIQUELY-TROPICAL single-pass backward via cached HARD ROUTING.

    Forward once, caching the discrete routing (relu masks, residual max-gates);
    then route error top-down in ONE pass (no iterative settling). Multiplier-free:
    error propagation `delta @ W` with dyadic W is shift-add even for real delta
    (real * 2^k = exponent shift); residual skip-wins are unit-gain highways
    (error passes straight down); only the weight-grad outer product is dyadic-
    quantized. Sets .grad (dyadic-backward) for a sign/pow2 step. O(depth) total.
    """
    # forward with routing cache
    acts, cache = [x0], []
    h = x0
    for l in range(model.depth):
        pre = model.layers[l](h)
        interior = l < model.depth - 1
        relu_out = torch.relu(pre) if interior else pre
        relu_mask = (pre > 0).float() if interior else torch.ones_like(pre)
        if model.residual and interior and h.shape[-1] == relu_out.shape[-1]:
            skip_gate = (relu_out >= h).float()          # 1 where transformed branch won
            out = torch.maximum(relu_out, h)
            has_skip = True
        else:
            skip_gate, out, has_skip = None, relu_out, False
        cache.append((relu_mask, skip_gate, has_skip))
        acts.append(out)
        h = out
    # top-down error cascade (L2 energy: dE/d logits = logits - target)
    e = acts[-1] - target
    for l in reversed(range(model.depth)):
        relu_mask, skip_gate, has_skip = cache[l]
        # split error across the residual max: transformed branch (skip_gate=1) vs skip
        e_relu = e * skip_gate if has_skip else e
        delta = e_relu * relu_mask                        # hard-routed local error at pre-act
        qa = dyadic_quantize(delta)
        qb = dyadic_quantize(acts[l])
        model.layers[l].weight.grad = qa.transpose(0, 1) @ qb   # dyadic-backward (shift-add)
        model.layers[l].bias.grad = qa.sum(0)
        if l > 0:
            Wq = dyadic_quantize(model.layers[l].weight)
            e_down = delta @ Wq                           # Wᵀ·delta (shift-add, dyadic W)
            if has_skip:
                e_down = e_down + e * (1.0 - skip_gate)   # unit-gain skip highway
            e = e_down
    return acts, cache


def train_gs(model, Xtr, Ytr, Xte, Yte, epochs, n_sweeps, opt, bs=128, n_classes=10,
             **settle_kw):
    """Train with ITERATIVE preconditioned-GS settling (reaches equilibrium) + local
    dyadic weight update. settle_kw -> alpha/anneal/warm (the three explorations)."""
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = model.settle_gs(x0, target, n_sweeps, **settle_kw)
            opt.zero_grad(set_to_none=True)
            model.gs_weight_grads(states)
            opt.step()
    with torch.no_grad():
        return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()


def train_settle(model, Xtr, Ytr, Xte, Yte, epochs, settle_call, opt, bs=128, n_classes=10):
    """Generic iterative-PC trainer: settle_call(model,x0,target)->states, then the
    local dyadic weight update at the settled equilibrium."""
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = settle_call(model, x0, target)
            opt.zero_grad(set_to_none=True)
            model.gs_weight_grads(states)
            opt.step()
    with torch.no_grad():
        return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()


def train_nudge(model, Xtr, Ytr, Xte, Yte, epochs, n_sweeps, opt, beta=0.3,
                async_frac=1.0, bs=128, n_classes=10, seed=0, **async_kw):
    """Train with nudged iterative settling (the convergent regime).
    async_kw -> stale / rand_order (the harder distributed conditions)."""
    n = Xtr.shape[0]
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = model.settle_nudge(x0, target, n_sweeps, beta, async_frac, gen,
                                        **async_kw)
            opt.zero_grad(set_to_none=True)
            model.gs_weight_grads(states)
            opt.step()
    with torch.no_grad():
        return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()


def train_cascade(model, Xtr, Ytr, Xte, Yte, epochs, opt, bs=128, n_classes=10):
    """Train with the single-pass cached-routing cascade backward."""
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            opt.zero_grad(set_to_none=True)
            cascade_grads(model, x0, target)
            opt.step()
    with torch.no_grad():
        return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()


def train_mf(model, Xtr, Ytr, Xte, Yte, epochs, n_steps, lr_shift=10, bs=128,
             n_classes=10):
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = model.infer(x0, target, n_steps=n_steps)
            model.dyadic_mf_step(states, lr_shift)
    with torch.no_grad():
        return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()


def train(model, Xtr, Ytr, Xte, Yte, epochs, n_steps, lr=1e-3, bs=128, n_classes=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - bs, bs):
            idx = perm[i:i + bs]
            x0, y = Xtr[idx], Ytr[idx]
            target = torch.zeros(y.shape[0], n_classes)
            target.scatter_(1, y.unsqueeze(1), 1.0)
            states = model.infer(x0, target, n_steps=n_steps)
            opt.zero_grad()
            model.energy(states).backward()
            opt.step()
    with torch.no_grad():
        return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()


VARIANTS = {
    "baseline":      dict(),
    "residual":      dict(residual=True),
    "residual_d1":   dict(residual=True, discount=1.0),
    "norm":          dict(norm=True),
    "resid+norm":    dict(residual=True, norm=True),
    "resid+prec":    dict(residual=True, precision=True),
    "precision":     dict(precision=True),
}

if __name__ == "__main__":
    down, n_train, n_test, epochs = 12, 6000, 2000, 5
    d = down * down
    Xtr, Ytr, Xte, Yte = get_mnist(down, n_train, n_test)

    if sys.argv[1:] and sys.argv[1] == "eqceiling":
        # DECISIVE: is the iterative PC EQUILIBRIUM as good as backprop at depth,
        # or is the solver the only problem? Settle to true convergence with a
        # strong inner optimizer (Adam on the latents), then the EXACT (autograd)
        # PC equilibrium weight update. If this plateaus below backprop, no solver
        # (multigrid incl.) can reach backprop iteratively — it's the equilibrium.
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [48, 64]

        def settle_adam(model, x0, target, inner=80, lr=0.05):
            states = [s.detach() for s in model.feedforward(x0)]
            states[model.depth] = target
            free = list(range(1, model.depth))
            for p in model.parameters():
                p.requires_grad_(False)
            fs = [states[i].clone().requires_grad_(True) for i in free]
            io = torch.optim.Adam(fs, lr=lr)
            for _ in range(inner):
                st = list(states)
                for i, s in zip(free, fs):
                    st[i] = s
                io.zero_grad(); model.energy(st).backward(); io.step()
            for p in model.parameters():
                p.requires_grad_(True)
            out = list(states)
            for i, s in zip(free, fs):
                out[i] = s.detach()
            return out

        def train_eq(model, opt):
            n = n_tr
            for _ in range(ep):
                perm = torch.randperm(n)
                for i in range(0, n - 128, 128):
                    idx = perm[i:i+128]; x0b, yb = Xtr2[idx], Ytr2[idx]
                    tgt = torch.zeros(yb.shape[0], 10); tgt.scatter_(1, yb.unsqueeze(1), 1.0)
                    states = settle_adam(model, x0b, tgt)
                    opt.zero_grad()
                    model.energy([s.detach() for s in states]).backward()  # EXACT PC eq update
                    opt.step()
            with torch.no_grad():
                return (model.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()

        def bp(model, opt):
            n = n_tr
            for _ in range(ep + 5):
                perm = torch.randperm(n)
                for i in range(0, n - 128, 128):
                    idx = perm[i:i+128]; h = Xtr2[idx]
                    for l in range(model.depth):
                        h = model.f(l, h)
                    opt.zero_grad(); F.cross_entropy(h, Ytr2[idx]).backward(); opt.step()
            with torch.no_grad():
                h = Xte
                for l in range(model.depth):
                    h = model.f(l, h)
                return (h.argmax(-1) == Yte).float().mean().item()

        print("== EQUILIBRIUM-CEILING: fully-settled PC eq update vs backprop ==")
        print(f"{'depth':>6} {'PC-eq (Adam-settled)':>20} {'backprop':>9} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            t0 = time.time()
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, residual=True)
            a_eq = train_eq(net, torch.optim.Adam(net.parameters(), lr=1e-3))
            torch.manual_seed(0)
            net2 = DeepDyadicPC(dims, residual=True)
            a_bp = bp(net2, torch.optim.Adam(net2.parameters(), lr=1e-3))
            print(f"{depth:>6} {a_eq:>20.3f} {a_bp:>9.3f} {time.time()-t0:>6.0f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "hybrid":
        # ACCURACY<->ASYNC frontier: a fraction p of steps use the cascade (sync,
        # exact gradient — needs a controller); the rest are FULLY async nudge
        # (partial+stale+rand). p=0 = pure async; p=1 = cascade(=backprop). How
        # little sync buys back the accuracy?
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        dims = [d] + [width] * (depth - 1) + [10]
        n_sw, bs = max(12, depth), 128
        print(f"== accuracy<->async frontier (depth {depth}): fraction p of SYNC cascade steps ==")
        print(f"{'p (sync frac)':>13} {'acc':>7} {'sec':>6}")
        for p in (0.0, 0.1, 0.25, 0.5, 1.0):
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, residual=True)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            gen = torch.Generator().manual_seed(0)
            t0 = time.time()
            for _ in range(ep):
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr - bs, bs):
                    idx = perm[i:i+bs]; x0b, yb = Xtr2[idx], Ytr2[idx]
                    tgt = torch.zeros(yb.shape[0], 10); tgt.scatter_(1, yb.unsqueeze(1), 1.0)
                    opt.zero_grad(set_to_none=True)
                    if torch.rand(1, generator=gen).item() < p:           # SYNC cascade step
                        cascade_grads(net, x0b, tgt)
                    else:                                                  # FULLY ASYNC nudge step
                        st = net.settle_nudge(x0b, tgt, n_sw, beta=0.5, async_frac=0.5,
                                              gen=gen, stale=True, rand_order=True)
                        net.gs_weight_grads(st)
                    opt.step()
            acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
            print(f"{p:>13.2f} {acc:>7.3f} {time.time()-t0:>6.0f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "harden":
        # Rigorous distributed result: multi-seed error bars across async conditions
        # + depth scan. (Cascade has NO async form, so it is N/A for every async row.)
        import statistics
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        seeds = [0, 1, 2]

        def run(depth, n_sw, kw, seed):
            dims = [d] + [width] * (depth - 1) + [10]
            torch.manual_seed(seed)
            net = DeepDyadicPC(dims, residual=True)
            return train_nudge(net, Xtr2, Ytr2, Xte, Yte, ep, n_sw,
                               torch.optim.Adam(net.parameters(), lr=1e-3),
                               beta=0.5, seed=seed, **kw)

        conditions = [
            ("sync",                  dict(async_frac=1.0)),
            ("partial 50%",           dict(async_frac=0.5)),
            ("partial 25%",           dict(async_frac=0.25)),
            ("stale (Jacobi)",        dict(async_frac=1.0, stale=True)),
            ("rand-order",            dict(async_frac=1.0, rand_order=True)),
            ("HARD 50%+stale+rand",   dict(async_frac=0.5, stale=True, rand_order=True)),
        ]
        depth, n_sw = 16, 20
        print(f"== async robustness, multi-seed (depth {depth}, {n_sw} rounds, seeds {seeds}) ==")
        print(f"{'condition':>22} {'mean±std':>14}  per-seed")
        for name, kw in conditions:
            accs = [run(depth, n_sw, kw, s) for s in seeds]
            m, sd = statistics.mean(accs), statistics.pstdev(accs)
            print(f"{name:>22}  {m:.3f}±{sd:.3f}  {[f'{a:.3f}' for a in accs]}")

        print("\n== depth scan (seed 0): sync vs partial-50% ==")
        print(f"{'depth':>6} {'sync':>7} {'partial50':>10}")
        for dp in (8, 16, 32):
            nsw = max(12, dp)
            a_s = run(dp, nsw, dict(async_frac=1.0), 0)
            a_p = run(dp, nsw, dict(async_frac=0.5), 0)
            print(f"{dp:>6} {a_s:>7.3f} {a_p:>10.3f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "nudgesweep":
        # Does soft-nudge CONVERGE to backprop with enough sweeps (vs hard-clamp's
        # 0.67 plateau)? Scan sweeps at depth 32; small beta = closer to backprop.
        width, n_tr, ep = 64, 2000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        dims = [d] + [width] * (depth - 1) + [10]

        def _bp(model, e=8):
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(e):
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr - 128, 128):
                    idx = perm[i:i+128]; h = Xtr2[idx]
                    for l in range(model.depth):
                        h = model.f(l, h)
                    opt.zero_grad(); F.cross_entropy(h, Ytr2[idx]).backward(); opt.step()
            with torch.no_grad():
                h = Xte
                for l in range(model.depth):
                    h = model.f(l, h)
                return (h.argmax(-1) == Yte).float().mean().item()

        torch.manual_seed(0)
        a_bp = _bp(DeepDyadicPC(dims, residual=True))
        print(f"== nudge convergence vs sweeps (depth {depth}); backprop = {a_bp:.3f} ==")
        print(f"{'beta':>6} {'sw=depth':>9} {'sw=2depth':>10} {'sw=4depth':>10}")
        for beta in (0.1, 0.2):
            row = []
            for mult in (1, 2, 4):
                torch.manual_seed(0)
                net = DeepDyadicPC(dims, residual=True)
                acc = train_nudge(net, Xtr2, Ytr2, Xte, Yte, ep, mult * depth,
                                  torch.optim.Adam(net.parameters(), lr=1e-3), beta=beta)
                row.append(f"{acc:0.3f}")
            print(f"{beta:>6.1f} {row[0]:>9} {row[1]:>10} {row[2]:>10}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "nudge":
        # CONVERGENCE: does the soft-NUDGE iterative regime converge AND match
        # backprop at depth (where hard-clamp settling failed)?
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [16, 32, 48, 64]

        def _bp(model, e=8, lr=1e-3, bs=128):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(e):
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr - bs, bs):
                    idx = perm[i:i+bs]; h = Xtr2[idx]
                    for l in range(model.depth):
                        h = model.f(l, h)
                    opt.zero_grad(); F.cross_entropy(h, Ytr2[idx]).backward(); opt.step()
            with torch.no_grad():
                h = Xte
                for l in range(model.depth):
                    h = model.f(l, h)
                return (h.argmax(-1) == Yte).float().mean().item()

        print(f"== nudged iterative PC (sweeps=10) vs backprop, width {width} ==")
        print(f"{'depth':>6} {'nudge β=0.2':>11} {'nudge β=0.5':>11} {'backprop':>9} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            t0 = time.time()
            torch.manual_seed(0); net = DeepDyadicPC(dims, residual=True)
            a2 = train_nudge(net, Xtr2, Ytr2, Xte, Yte, ep, 10,
                             torch.optim.Adam(net.parameters(), lr=1e-3), beta=0.2)
            torch.manual_seed(0); net = DeepDyadicPC(dims, residual=True)
            a5 = train_nudge(net, Xtr2, Ytr2, Xte, Yte, ep, 10,
                             torch.optim.Adam(net.parameters(), lr=1e-3), beta=0.5)
            torch.manual_seed(0)
            a_bp = _bp(DeepDyadicPC(dims, residual=True))
            print(f"{depth:>6} {a2:>11.3f} {a5:>11.3f} {a_bp:>9.3f} {time.time()-t0:>6.0f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "async":
        # DISTRIBUTED demo: iterative PC under PARTIAL/async updates (each layer
        # reports with prob async_frac per pass) — the cascade has no async form.
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        dims = [d] + [width] * (depth - 1) + [10]
        print(f"== async/partial-update robustness (nudged PC, depth {depth}, β=0.5) ==")
        print(f"{'async_frac':>11} {'sweeps':>7} {'acc':>7} {'sec':>6}")
        for frac, sw in [(1.0, 10), (0.5, 14), (0.25, 20), (0.5, 10), (0.25, 10)]:
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, residual=True)
            t0 = time.time()
            acc = train_nudge(net, Xtr2, Ytr2, Xte, Yte, ep, sw,
                              torch.optim.Adam(net.parameters(), lr=1e-3),
                              beta=0.5, async_frac=frac)
            print(f"{frac:>11.2f} {sw:>7} {acc:>7.3f} {time.time()-t0:>6.0f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "deepfix":
        # The two principled fixes for the stiff-chain conditioning that keep PC's
        # equilibrium: multigrid two-grid + momentum. vs the best prior (skip).
        # refs: Jacobi-960 @d48 = 0.67, backprop @d48 = 0.80, @d64 = 0.82.
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [48, 64]
        configs = [
            ("skip",  lambda m, x, t: m.settle_gs(x, t, 6, warm=True, alpha=0.5)),
            ("mom",   lambda m, x, t: m.settle_mom(x, t, 12, gamma=0.3, beta=0.7)),
            ("mg",    lambda m, x, t: m.settle_mg(x, t, 4, smooth=3)),
            ("mg-warm", lambda m, x, t: m.settle_mg(x, t, 5, smooth=2, coarse_alpha=0.6)),
        ]
        print(f"== deepfix: multigrid + momentum vs skip (width {width}) ==")
        print("depth  " + "  ".join(f"{n:>8}" for n, _ in configs))
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            cells = []
            for _, fn in configs:
                torch.manual_seed(0)
                net = DeepDyadicPC(dims, residual=True)
                acc = train_settle(net, Xtr2, Ytr2, Xte, Yte, ep, fn,
                                   torch.optim.Adam(net.parameters(), lr=1e-3))
                cells.append(f"{acc:0.3f}   ")
            print(f"{depth:>4}   " + "  ".join(cells))
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "skiptune":
        # The winner (skip = cascade-warm-start) — does more refinement climb?
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [48, 64]
        print(f"== skip (cascade-warm) ceiling: more sweeps, width {width} ==")
        print(f"{'depth':>6} {'sw=6':>7} {'sw=12':>7} {'sw=20':>7} {'warm-only':>10}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            row = []
            for n_sw in (6, 12, 20):
                torch.manual_seed(0)
                net = DeepDyadicPC(dims, residual=True)
                acc = train_gs(net, Xtr2, Ytr2, Xte, Yte, ep, n_sw,
                               torch.optim.Adam(net.parameters(), lr=1e-3),
                               warm=True, alpha=0.5)
                row.append(f"{acc:0.3f}")
            # warm-only (0 refine sweeps) — how good is the cascade init alone as the eq?
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, residual=True)
            acc0 = train_gs(net, Xtr2, Ytr2, Xte, Yte, ep, 0,
                            torch.optim.Adam(net.parameters(), lr=1e-3),
                            warm=True, alpha=0.5)
            print(f"{depth:>6} {row[0]:>7} {row[1]:>7} {row[2]:>7} {acc0:>10.3f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "explore":
        # Explore 3 fixes for iterative-PC-at-depth (all keep the equilibrium):
        #   full   = preconditioned GS, full step (the plateauing baseline)
        #   damped = idea 1: alpha=0.5 step (stay nearer the basin)
        #   anneal = idea 2: step ramps 0.25->1.0 over sweeps (trust region)
        #   skip   = idea 3: cascade skip-warm-start latents, then gentle (a=0.5) GS
        width, n_tr, ep, n_sw = 64, 3000, 3, 6
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [32, 48, 64]
        variants = [
            ("full",   dict(alpha=1.0)),
            ("damped", dict(alpha=0.5)),
            ("anneal", dict(anneal=True)),
            ("skip",   dict(warm=True, alpha=0.5)),
        ]
        print(f"== iterative-PC-at-depth exploration (GS, {n_sw} sweeps, width {width}) ==")
        hdr = "depth  " + "  ".join(f"{n:>7}" for n, _ in variants)
        print(hdr)
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            cells = []
            for _, kw in variants:
                torch.manual_seed(0)
                net = DeepDyadicPC(dims, residual=True)
                acc = train_gs(net, Xtr2, Ytr2, Xte, Yte, ep, n_sw,
                               torch.optim.Adam(net.parameters(), lr=1e-3), **kw)
                cells.append(f"{acc:0.3f}  ")
            print(f"{depth:>4}   " + "  ".join(cells))
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "gsp":
        # Diagonal-preconditioned GS: does curvature-normalized exact-coordinate
        # settling converge in FEW sweeps at depth (the real tropical trick)?
        width, n_tr, ep = 64, 3000, 3
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [32, 48, 64]
        print("== diagonal-preconditioned GS settling (Adam weight update) ==")
        print(f"{'depth':>6} {'sweeps':>7} {'acc':>7} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            for n_sw in (3, 5, 10):
                torch.manual_seed(0)
                net = DeepDyadicPC(dims, residual=True)
                t0 = time.time()
                # train with preconditioned GS settling (default precond=True)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3)
                n = n_tr
                for _ in range(ep):
                    perm = torch.randperm(n)
                    for i in range(0, n - 128, 128):
                        idx = perm[i:i+128]
                        x0b, yb = Xtr2[idx], Ytr2[idx]
                        tgt = torch.zeros(yb.shape[0], 10); tgt.scatter_(1, yb.unsqueeze(1), 1.0)
                        st = net.settle_gs(x0b, tgt, n_sw)
                        opt.zero_grad(set_to_none=True); net.gs_weight_grads(st); opt.step()
                acc = (net.predict_logits(Xte).argmax(-1) == Yte).float().mean().item()
                print(f"{depth:>6} {n_sw:>7} {acc:>7.3f} {time.time()-t0:>6.0f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "deepq":
        # Foundational: can ITERATIVE settling reach depth 48 with enough sweeps?
        # (Jacobi, more sweeps) — and does GS converge or diverge with more sweeps?
        width, n_tr = 64, 3000
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        ep = 3
        print("== Q1: Jacobi settling at depth 48, increasing sweeps (Adam) ==")
        dims48 = [d] + [width] * 47 + [10]
        for n_steps in (240, 480, 960):
            torch.manual_seed(0)
            net = DeepDyadicPC(dims48, gamma=0.125, residual=True)
            t0 = time.time()
            acc = train(net, Xtr2, Ytr2, Xte, Yte, ep, n_steps=n_steps)
            print(f"  Jacobi sweeps={n_steps:>4}  acc={acc:.3f}  ({time.time()-t0:.0f}s)")
        print("== Q2: GS settling at depth 32, increasing sweeps (converge or diverge?) ==")
        dims32 = [d] + [width] * 31 + [10]
        for n_sw in (5, 20, 60):
            torch.manual_seed(0)
            net = DeepDyadicPC(dims32, gamma=0.125, residual=True)
            t0 = time.time()
            acc = train_gs(net, Xtr2, Ytr2, Xte, Yte, ep, n_sw,
                           torch.optim.Adam(net.parameters(), lr=1e-3))
            print(f"  GS sweeps={n_sw:>3}  acc={acc:.3f}  ({time.time()-t0:.0f}s)")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "gs":
        # ITERATIVE tropical PC at depth: Gauss-Seidel (chain-solver) settling.
        # Reaches the energy equilibrium (keeps PC) but converges in O(1) sweeps,
        # depth-stable, where Jacobi settling walls at ~48.
        from tikitorch.optim import PureTropicalSGD
        width, n_tr = 64, 4000
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [16, 32, 48, 64, 100]
        n_sweeps = 5
        print(f"== iterative GS-settling PC (width {width}, {n_sweeps} sweeps, {epochs} ep) ==")
        print(f"{'depth':>6} {'GS+Adam':>8} {'GS+mf':>7} {'Jacobi':>7} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            t0 = time.time()
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=0.125, residual=True)
            a_gs = train_gs(net, Xtr2, Ytr2, Xte, Yte, epochs, n_sweeps,
                            torch.optim.Adam(net.parameters(), lr=1e-3))
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=0.125, residual=True)
            a_mf = train_gs(net, Xtr2, Ytr2, Xte, Yte, epochs, n_sweeps,
                            PureTropicalSGD(net.parameters(), lr_shift=10, momentum_bits=0))
            # Jacobi reference (the old settling) at matched sweep budget
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=0.125, residual=True)
            a_ja = train(net, Xtr2, Ytr2, Xte, Yte, epochs, n_steps=5 * depth)
            print(f"{depth:>6} {a_gs:>8.3f} {a_mf:>7.3f} {a_ja:>7.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "cascadefinal":
        # Headline: fully multiplier-free cached-routing cascade at adequate
        # epochs (15) — flat to depth 100, matching backprop. vs settling (walls
        # at 48) and backprop ceiling.
        from tikitorch.optim import PureTropicalSGD
        width, n_tr, eps = 64, 4000, 15
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [16, 32, 48, 64, 100]

        def _bp(model, e=12, lr=1e-3, bs=128):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(e):
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr - bs, bs):
                    idx = perm[i:i+bs]
                    h = Xtr2[idx]
                    for l in range(model.depth):
                        h = model.f(l, h)
                    opt.zero_grad(); F.cross_entropy(h, Ytr2[idx]).backward(); opt.step()
            with torch.no_grad():
                h = Xte
                for l in range(model.depth):
                    h = model.f(l, h)
                return (h.argmax(-1) == Yte).float().mean().item()

        print(f"== FINAL: multiplier-free cascade (15 ep) vs backprop, width {width} ==")
        print(f"{'depth':>6} {'casc mf':>8} {'BP ref':>7} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            t0 = time.time()
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, residual=True)
            opt = PureTropicalSGD(net.parameters(), lr_shift=10, momentum_bits=0)
            a_mf = train_cascade(net, Xtr2, Ytr2, Xte, Yte, eps, opt)
            torch.manual_seed(0)
            a_bp = _bp(DeepDyadicPC(dims, residual=True))
            print(f"{depth:>6} {a_mf:>8.3f} {a_bp:>7.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "casc100":
        # Close the depth-100 gap: tune the depth-robust multiplier-free cascade
        # (epochs x lr), vs the backprop ceiling (0.81).
        from tikitorch.optim import PureTropicalSGD
        width, n_tr, depth = 64, 4000, 100
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        dims = [d] + [width] * (depth - 1) + [10]
        print("== depth-100 cascade+multiplier-free tuning (BP ref ~0.81) ==")
        print(f"{'epochs':>7} {'lr_shift':>8} {'acc':>7} {'sec':>6}")
        for ep, ls in [(5, 10), (10, 10), (15, 10), (10, 9), (15, 11)]:
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, residual=True)
            opt = PureTropicalSGD(net.parameters(), lr_shift=ls, momentum_bits=0)
            t0 = time.time()
            acc = train_cascade(net, Xtr2, Ytr2, Xte, Yte, ep, opt)
            print(f"{ep:>7} {ls:>8} {acc:>7.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "cascade":
        # The uniquely-tropical fix: single-pass cached-routing backward (no
        # iterative settling). Does it reach depth 64/100 where PC-settling walls
        # at 48? Compare cascade+Adam, cascade+multiplier-free, and backprop ref.
        from tikitorch.optim import PureTropicalSGD
        width, n_tr = 64, 4000
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        depths = [int(a) for a in sys.argv[2:]] or [32, 48, 64, 100]

        def _bp(model, eps=8, lr=1e-3, bs=128):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(eps):
                perm = torch.randperm(n_tr)
                for i in range(0, n_tr - bs, bs):
                    idx = perm[i:i+bs]
                    h = Xtr2[idx]
                    for l in range(model.depth):
                        h = model.f(l, h)
                    opt.zero_grad(); F.cross_entropy(h, Ytr2[idx]).backward(); opt.step()
            with torch.no_grad():
                h = Xte
                for l in range(model.depth):
                    h = model.f(l, h)
                return (h.argmax(-1) == Yte).float().mean().item()

        print(f"== cascade (single-pass cached-routing) vs backprop, width {width} ==")
        print(f"{'depth':>6} {'casc+Adam':>10} {'casc+mf':>8} {'BP ref':>7} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            t0 = time.time()
            torch.manual_seed(0)
            net_ca = DeepDyadicPC(dims, residual=True)
            a_ca = train_cascade(net_ca, Xtr2, Ytr2, Xte, Yte, epochs,
                                 torch.optim.Adam(net_ca.parameters(), lr=1e-3))
            torch.manual_seed(0)
            net_mf = DeepDyadicPC(dims, residual=True)
            a_mf = train_cascade(net_mf, Xtr2, Ytr2, Xte, Yte, epochs,
                                 PureTropicalSGD(net_mf.parameters(), lr_shift=10, momentum_bits=0))
            torch.manual_seed(0)
            a_bp = _bp(DeepDyadicPC(dims, residual=True))
            print(f"{depth:>6} {a_ca:>10.3f} {a_mf:>8.3f} {a_bp:>7.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "deepdiag":
        # PC walls at depth ~48 even though backprop on the same residual arch is
        # fine. Is it undersettling (more sweeps / smaller step recover) or a
        # genuine pathology? Grid sweeps x gamma at depth 48 (width 64).
        width, depth, n_tr = 64, 48, 4000
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        dims = [d] + [width] * (depth - 1) + [10]
        print(f"== depth-48 settling diagnosis (width {width}, {n_tr} train, {epochs} ep) ==")
        print(f"{'sweeps':>7} {'gamma':>7} {'acc':>7} {'sec':>6}")
        for n_steps, gamma in [(240, 0.125), (480, 0.125), (720, 0.125),
                               (480, 0.0625), (720, 0.03125)]:
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=gamma, residual=True)
            t0 = time.time()
            acc = train(net, Xtr2, Ytr2, Xte, Yte, epochs, n_steps=n_steps)
            print(f"{n_steps:>7} {gamma:>7} {acc:>7.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "resnet":
        # Push toward deep: how far does the tropical residual + O(depth) settling
        # hold? Faster setup (width 64) so deep runs are tractable. Adam first
        # (establishes the architecture ceiling), then we go multiplier-free.
        width = int(sys.argv[2]) if len(sys.argv) > 2 else 64
        depths = [int(a) for a in sys.argv[3:]] or [16, 32, 48, 64]
        n_tr = 4000
        Xtr2, Ytr2 = Xtr[:n_tr], Ytr[:n_tr]
        print(f"== deep residual dyadic PC (width {width}, {n_tr} train, {epochs} ep) ==")
        print(f"{'depth':>6} {'sweeps':>7} {'PC resid (Adam)':>16} {'BP resid':>9} {'sec':>6}")
        for depth in depths:
            dims = [d] + [width] * (depth - 1) + [10]
            n_steps = 5 * depth
            t0 = time.time()
            torch.manual_seed(0)
            a_pc = train(DeepDyadicPC(dims, gamma=0.125, residual=True),
                         Xtr2, Ytr2, Xte, Yte, epochs, n_steps=n_steps)
            # backprop ceiling on the same residual arch (no settling, fast)
            def _bp(model, eps=8, lr=1e-3, bs=128):
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                for _ in range(eps):
                    perm = torch.randperm(n_tr)
                    for i in range(0, n_tr - bs, bs):
                        idx = perm[i:i+bs]
                        h = Xtr2[idx]
                        for l in range(model.depth):
                            h = model.f(l, h)
                        opt.zero_grad(); F.cross_entropy(h, Ytr2[idx]).backward(); opt.step()
                with torch.no_grad():
                    h = Xte
                    for l in range(model.depth):
                        h = model.f(l, h)
                    return (h.argmax(-1) == Yte).float().mean().item()
            torch.manual_seed(0)
            a_bp = _bp(DeepDyadicPC(dims, residual=True))
            print(f"{depth:>6} {n_steps:>7} {a_pc:>16.3f} {a_bp:>9.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "bpresid":
        # FAIR backprop baseline: same dyadic-RESIDUAL architecture, trained with
        # plain global backprop (feedforward + CE, no PC settling), so the table
        # is PC-vs-backprop on the same net.
        def logits_grad(model, x):
            h = x
            for l in range(model.depth):
                h = model.f(l, h)
            return h

        def train_bp(model, epochs, lr=1e-3, bs=128):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            n = Xtr.shape[0]
            for ep in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n - bs, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad()
                    F.cross_entropy(logits_grad(model, Xtr[idx]), Ytr[idx]).backward()
                    opt.step()
            with torch.no_grad():
                return (logits_grad(model, Xte).argmax(-1) == Yte).float().mean().item()

        print("== backprop ceiling on the dyadic-RESIDUAL architecture ==")
        print(f"{'depth':>6} {'BP resid':>9} {'BP plain':>9} {'sec':>6}")
        for depth in (16, 24, 32):
            dims = [d] * depth + [10]
            t0 = time.time()
            torch.manual_seed(0)
            a_res = train_bp(DeepDyadicPC(dims, residual=True), 8)
            torch.manual_seed(0)
            a_plain = train_bp(DeepDyadicPC(dims, residual=False), 8)
            print(f"{depth:>6} {a_res:>9.3f} {a_plain:>9.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "powgamma":
        # Direction A: does a POWER-OF-TWO settling step (a shift) train as well
        # as the non-pow2 gamma=0.1? If so the settling step is multiplier-free.
        print("== power-of-two settling step (residual, depth 16) ==")
        depth = 16
        dims = [d] * depth + [10]
        n_steps = 96
        print(f"{'gamma':>8} {'2^-k?':>6} {'Adam':>7} {'mf-dyadic':>10} {'sec':>6}")
        for gamma, label in [(0.1, "no"), (0.125, "2^-3"), (0.0625, "2^-4")]:
            t0 = time.time()
            torch.manual_seed(0)
            a = train(DeepDyadicPC(dims, gamma=gamma, residual=True),
                      Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps)
            torch.manual_seed(0)
            m = train_mf(DeepDyadicPC(dims, gamma=gamma, residual=True),
                         Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps, lr_shift=10)
            print(f"{gamma:>8} {label:>6} {a:>7.3f} {m:>10.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "deepen":
        # Direction B: is the depth-32 drop UNDERSETTLING (more sweeps recover)?
        # and does GENTLE precision-weighting on top of the residual help?
        print("== break the depth-32 wall: more settling + gentle precision ==")
        depth = 32
        dims = [d] * depth + [10]
        print(f"{'config':>22} {'acc':>7} {'sec':>6}")
        configs = [
            ("resid n_steps=120",  dict(residual=True), 120),
            ("resid n_steps=200",  dict(residual=True), 200),
            ("resid n_steps=300",  dict(residual=True), 300),
            ("resid+prec n=200",   dict(residual=True, precision=True), 200),
        ]
        for label, kw, n_steps in configs:
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=0.125, **kw)   # pow2 gamma
            t0 = time.time()
            acc = train(net, Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps)
            print(f"{label:>22} {acc:>7.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "deepmf":
        # confirm the FULLY MULTIPLIER-FREE update also holds at depth 32 with
        # adequate settling (~6*depth sweeps).
        print("== residual + multiplier-free dyadic-backward at depth 32 ==")
        depth = 32
        dims = [d] * depth + [10]
        print(f"{'n_steps':>8} {'mf_acc':>8} {'sec':>6}")
        for n_steps in (200, 300):
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=0.125, residual=True)
            t0 = time.time()
            acc = train_mf(net, Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps, lr_shift=10)
            print(f"{n_steps:>8} {acc:>8.3f} {time.time()-t0:>6.1f}")
        sys.exit(0)

    if sys.argv[1:] and sys.argv[1] == "scale":
        # does the tropical residual stay flat deep, and with the fully
        # multiplier-free dyadic-backward update (not just Adam)?
        print("== tropical residual depth scaling (baseline d16=0.717) ==")
        print(f"{'depth':>6} {'resid(Adam)':>12} {'resid(mf-dyadic)':>16} "
              f"{'plain(Adam)':>12} {'sec':>6}")
        for depth in (16, 24, 32):
            dims = [d] * depth + [10]
            n_steps = min(6 * depth, 120)
            t0 = time.time()
            torch.manual_seed(0)
            a_res = train(DeepDyadicPC(dims, gamma=0.1, residual=True),
                          Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps)
            torch.manual_seed(0)
            a_mf = train_mf(DeepDyadicPC(dims, gamma=0.1, residual=True),
                            Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps, lr_shift=10)
            torch.manual_seed(0)
            a_plain = train(DeepDyadicPC(dims, gamma=0.1, residual=False),
                            Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps)
            print(f"{depth:>6} {a_res:>12.3f} {a_mf:>16.3f} {a_plain:>12.3f} "
                  f"{time.time()-t0:>6.1f}")
        sys.exit(0)

    depths = [int(a) for a in sys.argv[1:]] or [16]
    for depth in depths:
        print(f"\n== depth {depth} (MNIST 12x12, Adam probe, {epochs} ep) ==")
        dims = [d] * depth + [10]
        n_steps = min(8 * depth, 128)
        print(f"{'variant':>12} {'test_acc':>9} {'sec':>6}")
        for name, kw in VARIANTS.items():
            torch.manual_seed(0)
            net = DeepDyadicPC(dims, gamma=0.1, **kw)
            t0 = time.time()
            acc = train(net, Xtr, Ytr, Xte, Yte, epochs, n_steps=n_steps)
            print(f"{name:>12} {acc:>9.3f} {time.time()-t0:>6.1f}")
