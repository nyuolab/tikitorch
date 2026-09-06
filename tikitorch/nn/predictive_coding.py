"""tikitorch.nn.predictive_coding — tropical predictive-coding (PC) layers.

Predictive coding (Rao-Ballard / Whittington-Bogacz) replaces the backward pass
with iterative *settling* of per-layer latent states, followed by local weight
updates. :class:`TropicalPCNet` provides a closure-audited max/min-plus recipe;
:class:`DyadicPCNet` explores shift-add-oriented ReLU variants whose exact
closure depends on the selected settling, loss, gradient, and optimizer
boundaries. On the tropical substrate, the core routing is unusually simple:

  * **Prediction** of layer ``l`` from the state below is the max-plus (or
    min-plus) primitive ``mu_l[j] = max_i(W_l[j,i] + x_{l-1}[i])`` — the same op
    as the silicon-ready forward kernel.

  * **Error** ``eps_l = x_l - mu_l`` is *tropical division* (the ⊗-inverse of
    the semiring's ``+``): a subtractor, not a multiply.  Closure is preserved —
    the only operation the tropical semiring lacks is the ⊕ (max) inverse, which
    PC never needs.

  * **Credit assignment** rides the cached argmax.  The Jacobian of a tropical
    layer is a 0/1 selection matrix ``S`` (one winner per output); its transpose
    ``S^T`` — the operator that routes error downward during settling — is just a
    ``scatter_add`` over the *same indices computed on the forward pass*.  No
    transposed weight matrix, no weight transport, no multiply.  Because ``S`` is
    0/1, error is neither amplified nor attenuated across layers (unit gain) —
    the structural cure for the vanishing/exploding-error pathology that limits
    the depth of conventional PC.

Two depth levers are built in:

  * **Forward-init** (Z-IL-style): initialise the latents to the feed-forward
    pass so interior errors start at zero and only the clamped top carries
    signal — deep layers receive correctly-routed error on settling step 1.

  * **Incremental PC (iPC)**: optionally update weights *during* settling rather
    than after convergence (``learn_every``).

Depth is made representationally real by *alternating* dilation (max-plus) and
erosion (min-plus) layers: a pure max-plus stack collapses to a single
max-plus-linear map, whereas min-plus∘max-plus is a genuine tropical-rational
map. In :class:`TropicalPCNet`, state and weight updates are adds, comparisons,
scatters, and power-of-two shifts — wrap that concrete recipe in
:func:`tikitorch.assert_no_multiply` to verify its executed path. Dyadic recipe
boundaries are documented on :class:`DyadicPCNet`.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import List, Optional

import torch
import torch.nn as nn

__all__ = ["TropicalPCLayer", "TropicalPCNet", "DyadicPCNet", "SparseCodingPC"]


def _predict(weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, kind: str):
    """Tropical prediction mu = (max|min)_i(W[j,i] + x[i]) + b, with winner index.

    Returns (mu, indices). ``indices[b, j]`` is the argmax/argmin input feeding
    output j for sample b — the support of this layer's Jacobian.
    """
    expanded = weight.unsqueeze(0) + x.unsqueeze(1)  # (batch, out, in)
    if kind == "dilation":
        mu, idx = expanded.max(dim=-1)
    else:  # erosion (min-plus)
        mu, idx = expanded.min(dim=-1)
    return mu + bias, idx


def _route_down(err: torch.Tensor, idx: torch.Tensor, in_features: int) -> torch.Tensor:
    """Apply S^T to an error vector: scatter err into its winning input slots.

    err: (batch, out_features), idx: (batch, out_features) -> (batch, in_features).
    This is the tropical "free adjoint" — the transpose of the forward selection,
    reconstructed from cached indices with a multiply-free scatter_add.
    """
    batch = err.shape[0]
    down = torch.zeros(batch, in_features, device=err.device, dtype=err.dtype)
    down.scatter_add_(1, idx, err)
    return down


class TropicalPCLayer(nn.Module):
    """One predictive-coding edge: predicts its output state from the state below.

    Holds the generative weight ``W: (out, in)`` and ``bias: (out,)``.  ``kind``
    selects max-plus (``"dilation"``) or min-plus (``"erosion"``) prediction.
    The layer is *stateless* w.r.t. the latents — states live in
    :class:`TropicalPCNet`; this class only owns parameters and the local
    prediction/error/update primitives.  Updates are applied in-place under
    ``no_grad`` (PC needs no autograd graph).
    """

    def __init__(self, in_features: int, out_features: int, kind: str = "dilation",
                 init_identity: bool = False, identity_gap: float = 30.0,
                 identity_noise: float = 0.0):
        super().__init__()
        if kind not in ("dilation", "erosion"):
            raise ValueError(f"kind must be 'dilation' or 'erosion', got {kind!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        # Identity init only makes sense for square layers; silently fall back to
        # uniform otherwise (e.g. the input/readout edges of a net).
        self.init_identity = init_identity and (in_features == out_features)
        self.identity_gap = identity_gap
        self.identity_noise = identity_noise
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_identity:
            # Seed the layer near the identity map so a deep stack starts as
            # identity (no representation collapse) and error routes down the
            # diagonal. For dilation the diagonal wins the max (off-diagonal =
            # diagonal - gap); for erosion it wins the min (off-diagonal =
            # diagonal + gap). The GAP controls plasticity: a large gap (~30)
            # freezes the layer as a permutation — the diagonal always wins the
            # argmax, so off-diagonal weights never receive a gradient and the
            # interior stays inert. A small gap (~1-2) lets an off-diagonal input
            # win when it is gap-larger than the diagonal, so the routing — and
            # thus the features — can actually learn. Noise breaks ties so units
            # specialise.
            with torch.no_grad():
                g = self.identity_gap
                fill = -g if self.kind == "dilation" else g
                self.weight.fill_(fill)
                self.weight.fill_diagonal_(0.0)
                if self.identity_noise > 0:
                    self.weight.add_(torch.empty_like(self.weight).uniform_(
                        -self.identity_noise, self.identity_noise))
        else:
            nn.init.uniform_(self.weight, -0.5, 0.5)
        nn.init.zeros_(self.bias)

    def predict(self, x_below: torch.Tensor):
        """mu, indices for this layer given the state below it."""
        return _predict(self.weight, self.bias, x_below, self.kind)

    @torch.no_grad()
    def local_update(self, err: torch.Tensor, idx: torch.Tensor, lr_shift: int,
                     mode: str = "value"):
        """Local weight/bias step from this layer's error and cached winners.

        Gradient-descent on (1/2)||eps||^2 gives  W[j, win] += lr * eps[j],  on
        the single winning edge per output (the rest get nothing) and
        b[j] += lr * eps[j].  ``lr = 2^-lr_shift`` so the scaling is a bit-shift.
        ``mode="sign"`` uses lr * sign(eps) instead (SignSGD-style); ``"value"``
        keeps PC's linear update.  All multiply-free.
        """
        signal = torch.sign(err) if mode == "sign" else err
        step = signal / float(1 << lr_shift)          # divide by power of two = shift
        # bias: every output unit
        self.bias.add_(step.sum(dim=0))
        # weight: scatter the step onto the winning input of each output, sum over batch
        batch = err.shape[0]
        gw = torch.zeros(batch, self.out_features, self.in_features,
                         device=err.device, dtype=err.dtype)
        gw.scatter_(2, idx.unsqueeze(-1), step.unsqueeze(-1))
        self.weight.add_(gw.sum(dim=0))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, kind={self.kind}"


class TropicalPCNet(nn.Module):
    """A deep tropical predictive-coding network (discriminative / classifier).

    ``dims = [d_in, d_1, ..., d_out]`` defines ``L = len(dims)-1`` PC edges.  By
    default layers alternate dilation/erosion so depth is representationally
    meaningful.  The state stack is ``x_0..x_L`` with ``x_0`` clamped to the
    input and (during training) ``x_L`` clamped to the target; interior states
    settle to minimise the local errors, then weights take a local step.

    This is *not* trained with ``loss.backward()`` / a torch optimizer — call
    :meth:`train_step`.  Everything is add/compare/scatter/shift; wrap a step in
    ``tt.assert_no_multiply()`` to confirm closure.
    """

    def __init__(self, dims: List[int], alternate: bool = True,
                 first_kind: str = "dilation",
                 gamma_shift: int = 2, lr_shift: int = 6,
                 weight_update: str = "value",
                 target_mode: str = "fixed", margin: float = 1.0,
                 target_high: float = 1.0, target_low: float = 0.0,
                 init_identity: bool = False, identity_gap: float = 30.0,
                 identity_noise: float = 0.0):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims needs at least [in, out]")
        self.dims = list(dims)
        self.depth = len(dims) - 1
        self.gamma_shift = gamma_shift       # state step size = 2^-gamma_shift
        self.lr_shift = lr_shift             # weight step size = 2^-lr_shift
        self.weight_update = weight_update
        self.target_mode = target_mode
        self.margin = margin
        self.target_high = target_high
        self.target_low = target_low

        layers = []
        for l in range(self.depth):
            if alternate:
                kind = first_kind if (l % 2 == 0) else (
                    "erosion" if first_kind == "dilation" else "dilation")
            else:
                kind = first_kind
            layers.append(TropicalPCLayer(dims[l], dims[l + 1], kind=kind,
                                          init_identity=init_identity,
                                          identity_gap=identity_gap,
                                          identity_noise=identity_noise))
        self.layers = nn.ModuleList(layers)

    # -- state initialisation ------------------------------------------------
    @torch.no_grad()
    def feedforward(self, x0: torch.Tensor):
        """Pure feed-forward sweep. Returns (states, indices) with all errors 0."""
        states = [x0]
        idxs = []
        for layer in self.layers:
            mu, idx = layer.predict(states[-1])
            states.append(mu)
            idxs.append(idx)
        return states, idxs

    @torch.no_grad()
    def _init_states(self, x0: torch.Tensor, init: str):
        if init == "forward":
            states, _ = self.feedforward(x0)
            return states
        # zero-init interior, feed-forward only x0
        states = [x0]
        for l in range(self.depth):
            states.append(torch.zeros(x0.shape[0], self.dims[l + 1],
                                       device=x0.device, dtype=x0.dtype))
        return states

    def _make_target(self, y: torch.Tensor, n_classes: int,
                     ref: torch.Tensor) -> torch.Tensor:
        """Clamp value for the top state x_L.

        ``"fixed"`` (the constructor default): a constant one-hot in
        ``{target_low, target_high}``. Trains well on the tasks measured here
        (e.g. 1.0 on separable blobs).

        ``"margin"`` (scale-free alternative): the network's own feed-forward
        logits with the correct class boosted by ``+margin`` — so the injected
        top error is ``+margin`` on the true class and ``0`` elsewhere,
        independent of the absolute scale of the max-plus activations. Motivated
        by the failure mode where, on some tasks, max-plus logits sit above a
        fixed constant so every error goes negative; ``"fixed"`` is fine on the
        separable tasks tested, so prefer it unless that pathology appears.
        """
        if self.target_mode == "margin":
            ff_logits = self.predict_logits(ref)
            tgt = ff_logits.clone()
            tgt.scatter_add_(1, y.unsqueeze(1),
                             torch.full((y.shape[0], 1), self.margin,
                                        device=ref.device, dtype=ref.dtype))
            return tgt
        tgt = torch.full((y.shape[0], n_classes), self.target_low,
                         device=ref.device, dtype=ref.dtype)
        tgt.scatter_(1, y.unsqueeze(1), self.target_high)
        return tgt

    # -- inference (settling) -----------------------------------------------
    @torch.no_grad()
    def infer(self, x0: torch.Tensor, target: Optional[torch.Tensor] = None,
              n_steps: int = 8, init: str = "forward",
              learn_every: int = 0):
        """Settle the latent states (and optionally do iPC weight updates).

        ``target`` (if given) clamps the top state x_L; x_0 is always clamped to
        the input.  Returns a dict with the settled ``states``, per-layer
        ``errors`` and ``indices`` (post-settle), and a diagnostics ``trace``
        (L1 energy per sweep, plus winner-flip fraction vs the feed-forward
        pass).  ``learn_every > 0`` enables incremental PC: a local weight step
        every ``learn_every`` sweeps.
        """
        states = self._init_states(x0, init)        # states[0] == x0 in both modes
        ff_idx = None
        if init == "forward":
            # winners from the clean feed-forward pass, for the flip diagnostic
            _, ff_idx = self.feedforward(x0)
        if target is not None:
            states[-1] = target

        gamma = float(1 << self.gamma_shift)
        energy_trace = []
        last_errors, last_idx = None, None

        for t in range(n_steps):
            # recompute predictions, errors, winners from current states
            mus, idxs, errs = [], [], []
            for l, layer in enumerate(self.layers):
                mu, idx = layer.predict(states[l])
                mus.append(mu)
                idxs.append(idx)
                errs.append(states[l + 1] - mu)        # tropical division
            last_errors, last_idx = errs, idxs

            # L1 norm of the errors — a MULTIPLY-FREE diagnostic only. NOTE: the
            # settling below descends the L2 energy (1/2)||eps||^2, so this L1
            # trace is NOT the minimized objective and can rise while L2 falls;
            # use it as a rough convergence signal, not a strict monotone check.
            energy_trace.append(float(sum(e.abs().sum().item() for e in errs)))

            # update each free interior state: grad = eps_l - S_{l+1}^T eps_{l+1}
            new_states = list(states)
            for l in range(1, self.depth):            # x_1 .. x_{L-1} are free
                down = _route_down(errs[l], idxs[l], self.dims[l])
                grad = errs[l - 1] - down
                new_states[l] = states[l] - grad / gamma
            # re-apply clamps
            new_states[0] = x0
            if target is not None:
                new_states[-1] = target
            states = new_states

            if learn_every and ((t + 1) % learn_every == 0):
                self._apply_local_updates(states)

        # final winners/errors for the flip diagnostic and post-settle learning
        flip_frac = None
        if ff_idx is not None and last_idx is not None:
            tot = sum(i.numel() for i in last_idx)
            chg = sum((a != b).sum().item() for a, b in zip(ff_idx, last_idx))
            flip_frac = chg / max(tot, 1)

        return {
            "states": states,
            "errors": last_errors,
            "indices": last_idx,
            "trace": {"energy": energy_trace, "winner_flip_frac": flip_frac},
        }

    @torch.no_grad()
    def _apply_local_updates(self, states):
        for l, layer in enumerate(self.layers):
            mu, idx = layer.predict(states[l])
            err = states[l + 1] - mu
            layer.local_update(err, idx, self.lr_shift, mode=self.weight_update)

    # -- training / prediction ----------------------------------------------
    @torch.no_grad()
    def train_step(self, x0: torch.Tensor, y: torch.Tensor, n_steps: int = 8,
                   init: str = "forward", learn_every: int = 0):
        """One supervised PC update on a batch. Returns a small metrics dict."""
        n_classes = self.dims[-1]
        target = self._make_target(y, n_classes, ref=x0)
        out = self.infer(x0, target=target, n_steps=n_steps, init=init,
                         learn_every=learn_every)
        if not learn_every:                            # post-settle (standard) update
            self._apply_local_updates(out["states"])
        # train-time accuracy from the feed-forward readout (what test uses)
        logits = self.predict_logits(x0)
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        return {
            "energy": out["trace"]["energy"],
            "winner_flip_frac": out["trace"]["winner_flip_frac"],
            "train_acc": acc,
        }

    @torch.no_grad()
    def predict_logits(self, x0: torch.Tensor) -> torch.Tensor:
        """Feed-forward readout of the top state (the test-time classifier)."""
        states, _ = self.feedforward(x0)
        return states[-1]

    @torch.no_grad()
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.predict_logits(x0)

    def extra_repr(self) -> str:
        return (f"dims={self.dims}, depth={self.depth}, "
                f"weight_update={self.weight_update!r}, "
                f"target_mode={self.target_mode!r}")


# ---------------------------------------------------------------------------
# Dyadic predictive coding — the variant that trains DEEP and matches backprop
# ---------------------------------------------------------------------------


class DyadicPCNet(nn.Module):
    """Predictive coding over multiplier-free **dyadic** (±2^k) ReLU layers.

    The pure max-plus :class:`TropicalPCNet` is fully multiplier-free but
    information-lossy: a ``max_i`` output keeps one input, so stacking compounds
    the loss and depth never helps. Replacing the lossy max mixing with a
    **dyadic shift-add** mixing (:class:`DyadicLinear`, weights snapped to ±2^k
    so ``W·x`` is a shift+add tree) plus a ReLU non-linearity gives oblique
    normals and a non-lossy map — i.e. a shift-add-oriented ReLU MLP trained by PC.
    Unlike :class:`TropicalPCNet`, settling is autograd-driven, but the energy's
    grads w.r.t. states/weights are still mathematically local (each ``dE/dW_l``
    uses only the adjacent error and activity).

    Three weight-update modes, selected on :meth:`train_step`:

      * **Adam** (``train_step`` default; a real outer-product multiply): the
        fast digital learner — trains deep and tracks the dyadic backprop
        ceiling. NOT a multiplier-free step.
      * **Dyadic magnitude-aware** (``multiplier_free=True``,
        ``mf_mode="dyadic"``): the recommended multiplier-free *gradient* path.
        Both the error and the activity are quantized to ±2^k, so the weight-grad
        contraction is a shift-add tree and matches Adam accuracy in the reported
        experiments. Other full-step boundaries remain separate.
      * **Pure sign-product** (``mf_mode="sign"``): exact per-sample sign
        ``sign(dE/dW[j,i]) = -sign(eps_j) ⊕ sign(x_{l,i})`` gated by ``relu'``
        (``eps = x - mu``). Lags the other two — the batched ±1 majority vote
        disagrees with the true batch-gradient sign on a sizeable fraction of
        weights — but kept for the exact per-sample identity.

    Closure is recipe- and boundary-scoped. Forward / ``Wᵀ·error`` feedback
    with ±2^k weights maps to shift-add and ``relu'`` is a 0/1 mask;
    :meth:`set_mf_grads` supplies a dyadic or ternary weight-gradient surrogate.
    In :meth:`infer`, ``states -= gamma·g`` maps to a shift only when ``gamma`` is
    dyadic (the default is 2^-3). :meth:`settle_nudge` additionally divides by
    the exact diagonal preconditioner ``D``, which is a general digital division
    in the current implementation. Adam is another explicit open boundary. A
    closed optimizer can close the optimizer boundary, but it does not by itself
    make the nudge recipe a fully closed step.

    Settling needs a small power-of-two ``gamma`` (~2^-3..2^-4, a shift;
    ``0.5=2^-1`` diverges on the deeper energy's curvature) and sweeps scaling
    with depth (``n_steps`` ~ 6·depth; undersettling looks like a depth wall but
    is not). See ``results/predictive_coding/TROPICAL_PC_DEPTH.md`` for measured
    depth/accuracy curves.
    """

    def __init__(self, dims: List[int], gamma: float = 0.125,
                 relu_readout: bool = False, residual: bool = False):
        super().__init__()
        from tikitorch.nn import DyadicLinear        # lazy: avoid import cycle
        if len(dims) < 2:
            raise ValueError("dims needs at least [in, out]")
        if (isinstance(gamma, bool) or not isinstance(gamma, Real)
                or not math.isfinite(float(gamma)) or not 0.0 < float(gamma) <= 1.0):
            raise ValueError("gamma must be a finite real in (0, 1]")
        self.dims = list(dims)
        self.depth = len(dims) - 1
        self.gamma = float(gamma)
        self.relu_readout = relu_readout
        # ``residual`` adds a TROPICAL skip on square interior layers:
        # ``f(x) = max(relu(W x), x)`` — tropical addition of the identity path, a
        # single ``max`` (multiplier-free, bounded, idempotent). It keeps dyadic
        # PC flat to depth 32 (MNIST: 0.909 Adam / 0.856 dyadic-gradient at
        # n_steps≈6·depth) where plain dyadic PC collapses to chance by ~depth 24
        # (the earlier apparent drop at depth 32 was undersettling, not a wall).
        # Because ``max`` cannot accumulate the way additive residual energy does,
        # it addresses the "residual energy propagates too fast" failure natively.
        self.residual = residual
        self.layers = nn.ModuleList([DyadicLinear(dims[l], dims[l + 1])
                                     for l in range(self.depth)])

    def _has_skip(self, l: int, x: torch.Tensor, out: torch.Tensor) -> bool:
        return (self.residual and l < self.depth - 1
                and x.shape[-1] == out.shape[-1])

    def f(self, l: int, x: torch.Tensor) -> torch.Tensor:
        pre = self.layers[l](x)
        interior = l < self.depth - 1 or self.relu_readout
        out = torch.relu(pre) if interior else pre    # linear readout (logits)
        if self._has_skip(l, x, out):
            out = torch.maximum(out, x)               # tropical residual skip
        return out

    @torch.no_grad()
    def feedforward(self, x0: torch.Tensor) -> List[torch.Tensor]:
        s = [x0]
        for l in range(self.depth):
            s.append(self.f(l, s[-1]))
        return s

    def energy(self, states: List[torch.Tensor]) -> torch.Tensor:
        E = states[0].new_zeros(())
        for l in range(self.depth):
            mu = self.f(l, states[l])
            E = E + 0.5 * ((states[l + 1] - mu) ** 2).sum()
        return E

    def _make_target(self, y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        tgt = torch.zeros(y.shape[0], self.dims[-1], device=ref.device, dtype=ref.dtype)
        tgt.scatter_(1, y.unsqueeze(1), 1.0)
        return tgt

    def infer(self, x0: torch.Tensor, target: Optional[torch.Tensor],
              n_steps: int):
        """Forward-init, then settle free states (weights frozen)."""
        states = [s.detach() for s in self.feedforward(x0)]
        free = list(range(1, self.depth)) if target is not None \
            else list(range(1, self.depth + 1))
        if target is not None:
            states[self.depth] = target
        for p in self.parameters():
            p.requires_grad_(False)
        for i in free:
            states[i] = states[i].clone().requires_grad_(True)
        for _ in range(n_steps):
            grads = torch.autograd.grad(self.energy(states),
                                        [states[i] for i in free])
            with torch.no_grad():
                for i, g in zip(free, grads):
                    states[i] -= self.gamma * g
            for i in free:
                states[i] = states[i].detach().requires_grad_(True)
        for p in self.parameters():
            p.requires_grad_(True)
        return [s.detach() for s in states]

    def train_step(self, x0: torch.Tensor, y: torch.Tensor, opt,
                   n_steps: int = 40, multiplier_free: bool = False,
                   mf_mode: str = "dyadic"):
        """One supervised PC step: settle, then a LOCAL weight update via ``opt``.

        ``opt`` is any torch optimizer over ``self.parameters()`` (e.g. Adam).
        The gradients it consumes come from the energy at fixed settled states,
        so ``dE/dW_l`` is local to layer ``l``'s error — predictive coding, not
        backprop through the stack.

        ``multiplier_free=True`` replaces the real ``dE/dW`` (an outer-product
        multiply) with a multiplier-free surrogate (:meth:`set_mf_grads`); pair
        it with a sign-only, power-of-two optimizer (``PureTropicalSGD``). Two
        modes:

          * ``mf_mode="dyadic"`` (default, **recommended**) — quantize BOTH the
            local error and the activity to ±2^k, so each product is a single
            shift; ``sign(Σ dyadic·dyadic)`` recovers the magnitude the pure sign
            rule discards and **matches Adam / the backprop ceiling** (~0.88–0.92
            on MNIST). This closes the local gradient contraction; a full-step
            mapping additionally requires a dyadic ``gamma`` and a closed
            optimizer, and must be audited for the concrete recipe.
          * ``mf_mode="sign"`` — the pure ±1 sign-product. The per-sample sign
            identity is exact but the batch majority vote is lossy, so it lags
            (~0.65–0.68). Kept for the exact per-sample identity.
        """
        target = self._make_target(y, ref=x0)
        states = self.infer(x0, target, n_steps=n_steps)
        opt.zero_grad(set_to_none=True)
        if multiplier_free:
            self.set_mf_grads(states, mode=mf_mode)
        else:
            self.energy(states).backward()            # local: dE/dW_l uses adjacent eps
        opt.step()

    @torch.no_grad()
    def set_mf_grads(self, states: List[torch.Tensor],
                     mode: str = "dyadic") -> None:
        """Set ``.grad`` to a multiplier-free surrogate for ``dE/dW``.

        Per layer the local error at the pre-activation is ``a = (mu - x_{l+1})``
        gated by ``relu'`` (= ``-eps`` = ``dE/dpre``; dead units → 0) and the
        activity is ``b = x_l``. The true ``dE/dW = a^T @ b`` is a real multiply.
        Both surrogates quantize the operands first, so every product is a SHIFT
        (XOR signs, add exponents) and the contraction is an add-tree — no
        multiply on fabric:

          * ``mode="dyadic"`` — ``q(v) = sign(v)·2^round(log2|v|)`` (nearest power
            of two; a leading-bit extraction in HW). ``q(a)^T @ q(b)`` keeps
            magnitudes within 2×, so its sign tracks the true-gradient sign
            closely — matches the Adam update in accuracy. This closes the local
            gradient contraction, not every settling/loss/optimizer boundary.
          * ``mode="sign"`` — ``q(v) = sign(v)`` (ternary {-1,0,+1}). The
            per-sample sign-product equals ``sign(dE/dW)`` exactly, but the batch
            sum-of-signs is a lossy majority vote that disagrees with the true
            batch-gradient sign on a sizeable fraction of weights, so it lags.

        Closure caveat: the float-sim runs the contraction as ``aten.mm``, so
        ``assert_no_multiply`` WILL flag it — closure is the quantized-operand
        (±2^k / ternary) property, not the op-trace, exactly as for
        :class:`DyadicLinear`.
        """
        from tikitorch.nn import dyadic_quantize
        if mode not in ("dyadic", "sign"):
            raise ValueError(f"mf_mode must be 'dyadic' or 'sign', got {mode!r}")
        q = (lambda v: torch.sign(v)) if mode == "sign" else dyadic_quantize
        for l, layer in enumerate(self.layers):
            pre = layer(states[l])
            interior = (l < self.depth - 1) or self.relu_readout
            relu_out = torch.relu(pre) if interior else pre
            mask = (pre > 0) if interior else torch.ones_like(pre, dtype=torch.bool)
            if self._has_skip(l, states[l], relu_out):
                mu = torch.maximum(relu_out, states[l])
                mask = mask & (relu_out >= states[l])     # W-branch won the max-gate
            else:
                mu = relu_out
            qa = q((mu - states[l + 1]) * mask)           # quantized local error (-eps)
            qb = q(states[l])                             # quantized activity
            layer.weight.grad = qa.transpose(0, 1) @ qb   # shift-add tree (no mult on HW)
            layer.bias.grad = qa.sum(0)

    # back-compat alias: the pure ±1 sign-product update
    @torch.no_grad()
    def set_sign_grads(self, states):
        """Pure ±1 sign-product surrogate (``set_mf_grads(mode='sign')``)."""
        self.set_mf_grads(states, mode="sign")

    # -- cached-routing cascade (the deep-PC depth solution) ----------------
    @torch.no_grad()
    def _forward_cache(self, x0: torch.Tensor):
        """Feed-forward pass -> ``(acts, cache)`` where ``cache[l] = (relu_mask,
        skip_gate, has_skip)`` is layer l's FROZEN hard routing. Shared by the
        cascade and the error-relaxation, which both freeze routing at this pass."""
        acts, cache = [x0], []
        h = x0
        for l in range(self.depth):
            pre = self.layers[l](h)
            interior = (l < self.depth - 1) or self.relu_readout
            relu_out = torch.relu(pre) if interior else pre
            relu_mask = (pre > 0).float() if interior else torch.ones_like(pre)
            if self._has_skip(l, h, relu_out):
                skip_gate = (relu_out >= h).float()          # 1 where transformed branch won
                out = torch.maximum(relu_out, h)
                has_skip = True
            else:
                skip_gate, out, has_skip = None, relu_out, False
            cache.append((relu_mask, skip_gate, has_skip))
            acts.append(out)
            h = out
        return acts, cache

    @torch.no_grad()
    def _route_T(self, l: int, u: torch.Tensor, cache) -> torch.Tensor:
        """A_l^T u via the FROZEN cached routing (shift-add, multiplier-free): route
        the error at layer l's output down to its input; unit-gain on skip-wins."""
        from tikitorch.nn import dyadic_quantize
        relu_mask, skip_gate, has_skip = cache[l]
        delta = (u * skip_gate * relu_mask) if has_skip else (u * relu_mask)
        e_down = delta @ dyadic_quantize(self.layers[l].weight)
        if has_skip:
            e_down = e_down + u * (1.0 - skip_gate)          # unit-gain skip highway
        return e_down

    @torch.no_grad()
    def cascade_grads(self, x0: torch.Tensor, target: torch.Tensor,
                      mode: str = "dyadic", loss: str = "ce") -> None:
        """Single-pass **cached-routing** backward — the tropical depth solution.

        ``loss`` sets the top-layer error (the only place a loss derivative enters):
        ``"ce"`` (default) uses the softmax-cross-entropy gradient ``softmax(logits) -
        target`` — this is what makes the cascade match backprop at depth on
        classification. It is deliberately an **open loss boundary**: softmax uses
        exponentials, normalization, and dense arithmetic. ``"mse"`` supplies only
        the subtractive L2 top error ``logits - target`` and avoids that softmax
        boundary (but trails CE at depth in the reported experiments). In either
        case, the network-internal cached routing remains shift/add oriented.

        Iterative settling (:meth:`infer`) walls at ~depth 48; this exploits the
        fact that tropical routing is *discrete and cacheable*. Forward once,
        caching the hard routing (ReLU masks, residual ``max``-gates); then route
        error top-down in ONE O(depth) pass (no settling iteration). It trains to
        **depth 100 matching backprop** with a multiplier-free internal routing path:

          * error propagation ``delta @ W`` with ±2^k weights is shift-add even
            for real ``delta`` (real × 2^k = exponent shift) — so error flows
            *exactly*, no settling instability;
          * residual skip-wins are **unit-gain highways** (error passes straight
            down where the skip branch won the ``max``);
          * only the weight-grad outer product is quantized (``mode`` "dyadic" =
            ±2^k, or "sign" = ternary), giving a shift-add / sign-XOR weight grad.

        This is the Z-IL single-ordered-pass equivalence, made exact and
        multiplier-free by hard routing. Use with a sign / power-of-two optimizer
        (``PureTropicalSGD``) via :meth:`train_step_cascade`.
        """
        from tikitorch.nn import dyadic_quantize
        if mode not in ("dyadic", "sign"):
            raise ValueError(f"mode must be 'dyadic' or 'sign', got {mode!r}")
        if loss not in ("ce", "mse"):
            raise ValueError(f"loss must be 'ce' or 'mse', got {loss!r}")
        q = (lambda v: torch.sign(v)) if mode == "sign" else dyadic_quantize
        acts, cache = self._forward_cache(x0)
        e = (torch.softmax(acts[-1], dim=-1) - target) if loss == "ce" \
            else (acts[-1] - target)                         # top error: CE | L2(MSE)
        for l in reversed(range(self.depth)):
            relu_mask, skip_gate, has_skip = cache[l]
            e_relu = e * skip_gate if has_skip else e
            delta = e_relu * relu_mask                        # hard-routed local error
            self.layers[l].weight.grad = q(delta).transpose(0, 1) @ q(acts[l])
            self.layers[l].bias.grad = q(delta).sum(0)
            if l > 0:
                e_down = delta @ dyadic_quantize(self.layers[l].weight)   # Wᵀ·delta, shift-add
                if has_skip:
                    e_down = e_down + e * (1.0 - skip_gate)  # unit-gain skip highway
                e = e_down

    def train_step_cascade(self, x0: torch.Tensor, y: torch.Tensor, opt,
                           mode: str = "dyadic", loss: str = "ce"):
        """One deep training step via the single-pass cascade (no settling).
        ``loss="ce"`` (default) matches backprop at depth but retains the explicit
        softmax loss boundary; see :meth:`cascade_grads`."""
        target = self._make_target(y, ref=x0)
        opt.zero_grad(set_to_none=True)
        self.cascade_grads(x0, target, mode=mode, loss=loss)
        opt.step()

    # -- error-relaxation: the ASYNC / distributed form that works at scale -----
    @torch.no_grad()
    def error_relax_grads(self, x0: torch.Tensor, target: torch.Tensor,
                          n_iter: Optional[int] = None, mode: str = "dyadic",
                          loss: str = "ce", participation: float = 1.0,
                          stale: bool = False, rand_order: bool = False,
                          gen: Optional[torch.Generator] = None) -> None:
        """ASYNC / iterative form of the cascade — the distributed learner that
        works at scale.

        Relaxes the PC equilibrium-error condition ``E_l = A_l^T E_{l+1}`` (routing
        FROZEN at the feed-forward pass) by repeated LOCAL updates in any order,
        then applies the same multiplier-free weight update as the cascade. This is
        the iterative regime that has been validated at full scale. Archived
        unit-step state-settling runs diverged as weights grew, but those runs
        predated the fix that made :meth:`settle_nudge` apply ``gamma``; its
        corrected full-scale behavior still needs remeasurement. In contrast,
        error-relaxation is a triangular (nilpotent) system — unconditionally
        stable regardless of weight magnitude.

        - **Sync** (``participation=1``, top-down, in-place): converges in ONE pass
          to *exactly* :meth:`cascade_grads` (= backprop credit assignment).
        - **Async** (``participation<1`` partial node updates / ``stale`` lagged
          reads / ``rand_order`` no global schedule): the neuromorphic / in-memory /
          distributed form the cascade's strict ordered pass cannot do. Robust —
          converges to the same credit signal given enough local rounds (measured
          within ~0.6pp of sync at 50% participation + stale + random order).

        ``n_iter`` defaults to ``depth + 4`` (enough for sync; raise for heavy
        async). ``gen`` (if given) must be a **CPU** ``torch.Generator`` — the async
        draws are control-flow (participation coin, layer order), not on the data
        device. See ``results/predictive_coding/SCALE_TESTS.md``.
        """
        from tikitorch.nn import dyadic_quantize
        if mode not in ("dyadic", "sign"):
            raise ValueError(f"mode must be 'dyadic' or 'sign', got {mode!r}")
        if loss not in ("ce", "mse"):
            raise ValueError(f"loss must be 'ce' or 'mse', got {loss!r}")
        acts, cache = self._forward_cache(x0)
        L = self.depth
        if n_iter is None:
            n_iter = L + 4
        E = [torch.zeros_like(a) for a in acts]
        eL = (torch.softmax(acts[L], dim=-1) - target) if loss == "ce" \
            else (acts[L] - target)                          # clamped top error: CE | MSE
        E[L] = eL
        free = list(reversed(range(1, L)))                   # E_1..E_{L-1} relaxed
        for _ in range(n_iter):
            snap = [t.clone() for t in E] if stale else E
            order = free
            if rand_order:
                order = [free[i] for i in
                         torch.randperm(len(free), generator=gen).tolist()]
            for l in order:
                if participation < 1.0 and \
                        torch.rand(1, generator=gen).item() >= participation:
                    continue
                E[l] = self._route_T(l, snap[l + 1], cache)
            E[L] = eL                                        # re-clamp output error
        q = (lambda v: torch.sign(v)) if mode == "sign" else dyadic_quantize
        for l in range(L):
            relu_mask, skip_gate, has_skip = cache[l]
            delta = (E[l + 1] * skip_gate * relu_mask) if has_skip else (E[l + 1] * relu_mask)
            self.layers[l].weight.grad = q(delta).transpose(0, 1) @ q(acts[l])
            self.layers[l].bias.grad = q(delta).sum(0)

    def train_step_relax(self, x0: torch.Tensor, y: torch.Tensor, opt,
                         n_iter: Optional[int] = None, mode: str = "dyadic",
                         loss: str = "ce", **async_kw) -> None:
        """One iterative / async error-relaxation step (see :meth:`error_relax_grads`).
        ``async_kw`` -> participation / stale / rand_order / gen. Sync == the cascade;
        async is the distributed-capable regime. ``loss="ce"`` matches backprop."""
        target = self._make_target(y, ref=x0)
        opt.zero_grad(set_to_none=True)
        self.error_relax_grads(x0, target, n_iter=n_iter, mode=mode, loss=loss,
                               **async_kw)
        opt.step()

    # -- nudged ITERATIVE settling: the fully-local / async / distributed regime --
    @torch.no_grad()
    def _route_l(self, l, x):
        """Layer l's hard routing at the current input x: (relu_mask, gate, has_skip)."""
        pre = self.layers[l](x)
        interior = (l < self.depth - 1) or self.relu_readout
        relu_out = torch.relu(pre) if interior else pre
        relu_mask = (pre > 0).float() if interior else torch.ones_like(pre)
        if self._has_skip(l, x, relu_out):
            return relu_mask, (relu_out >= x).float(), True
        return relu_mask, None, False

    @torch.no_grad()
    def _AT(self, l, u, x):
        """A_l^T u (transpose of layer l's local linear map) for the settling grad."""
        from tikitorch.nn import dyadic_quantize
        relu_mask, gate, has_skip = self._route_l(l, x)
        Wq = dyadic_quantize(self.layers[l].weight)
        if has_skip:
            return (gate * relu_mask * u) @ Wq + (1.0 - gate) * u
        return (relu_mask * u) @ Wq

    @torch.no_grad()
    def _diag_precond(self, l, x):
        """Exact per-neuron diagonal Hessian (binary gates kill cross-terms):
        D = 1 + Σ_j(active_j) W[j,i]^2 (+ skip). Dyadic W makes construction
        of D an add-tree of 2^(2k), but the current solver divides by exact D;
        that division is an explicit open arithmetic boundary."""
        from tikitorch.nn import dyadic_quantize
        relu_mask, gate, has_skip = self._route_l(l, x)
        active = relu_mask * gate if has_skip else relu_mask
        D = 1.0 + active @ (dyadic_quantize(self.layers[l].weight) ** 2)
        if has_skip:
            D = D + (1.0 - gate)
        return D

    @torch.no_grad()
    def settle_nudge(self, x0: torch.Tensor, target: torch.Tensor,
                     n_sweeps: int = 20, beta: float = 0.5,
                     async_frac: float = 1.0,
                     gen: Optional[torch.Generator] = None,
                     stale: bool = False,
                     rand_order: bool = False) -> List[torch.Tensor]:
        """ITERATIVE settling in the damped SOFT-NUDGE regime (the fully-local,
        async execution mode the cascade is *not*).

        The output state is FREE (init = feed-forward), softly pulled toward the
        target by ``beta`` — the small-nudge / EqProp limit that stays in the
        feed-forward basin. Diagonal-preconditioned Gauss-Seidel. There is NO
        global backward phase: every state updates from local prediction errors,
        in any order. Deterministic tests cover the gamma-damped recurrence and
        objective descent for representative settings; convergence remains
        configuration-dependent. Archived accuracy and async-robustness numbers
        used an unintended unit step and need to be rerun before they are
        attributed to this corrected implementation. The eager solver divides by
        the exact diagonal preconditioner, so this state-relaxation path has an
        explicit open digital-arithmetic boundary. It supports the distributed
        conditions a strict backward sweep can't take:

          async_frac<1 : each node updates with prob async_frac per pass (partial participation)
          stale=True   : neighbour reads from a round-start snapshot (stale-by-1 / lag)
          rand_order   : random per-pass order (no global schedule)

        See ``results/predictive_coding/ITERATIVE_PC_DISTRIBUTED.md`` for the
        multi-seed async-robustness table. Drive it via ``train_step_nudge``.
        """
        if (isinstance(n_sweeps, bool) or not isinstance(n_sweeps, Integral)
                or n_sweeps < 0):
            raise ValueError("n_sweeps must be a nonnegative integer")
        if (isinstance(beta, bool) or not isinstance(beta, Real)
                or not math.isfinite(float(beta)) or float(beta) < 0.0):
            raise ValueError("beta must be a finite nonnegative real")
        if (isinstance(async_frac, bool) or not isinstance(async_frac, Real)
                or not math.isfinite(float(async_frac))
                or not 0.0 <= float(async_frac) <= 1.0):
            raise ValueError("async_frac must be a finite real in [0, 1]")
        beta = float(beta)
        async_frac = float(async_frac)
        states = [s.clone() for s in self.feedforward(x0)]
        L = self.depth
        free = list(range(1, L + 1))                        # interior + OUTPUT (free, nudged)
        for _ in range(n_sweeps):
            snap = [s.clone() for s in states] if stale else states
            order = free + free[::-1]
            if rand_order:
                order = [order[i] for i in torch.randperm(len(order), generator=gen).tolist()]
            for l in order:
                if async_frac < 1.0 and torch.rand(1, generator=gen).item() >= async_frac:
                    continue
                eps_l = states[l] - self.f(l - 1, snap[l - 1])
                if l < L:
                    eps_up = snap[l + 1] - self.f(l, states[l])
                    grad = eps_l - self._AT(l, eps_up, states[l])
                    D = self._diag_precond(l, states[l])
                else:                                       # output: pred error + soft target nudge
                    grad = eps_l + beta * (states[l] - target)
                    D = 1.0 + beta
                states[l] = states[l] - self.gamma * grad / D
        return states

    def train_step_nudge(self, x0: torch.Tensor, y: torch.Tensor, opt,
                         n_sweeps: int = 20, beta: float = 0.5,
                         mf_mode: str = "dyadic", **async_kw) -> None:
        """One fully-local iterative-PC step: settle, then form a local dyadic or
        ternary gradient surrogate and pass it to ``opt``. A closed optimizer such
        as :class:`tikitorch.optim.PureTropicalSGD` closes the optimizer boundary;
        Adam and other ordinary optimizers leave it open. Independently, the eager
        state solver divides by its exact diagonal preconditioner, so the complete
        nudge step is not currently a zero-DSP recipe. ``async_kw`` selects
        ``async_frac`` / ``stale`` / ``rand_order``. No global backward pass."""
        target = self._make_target(y, ref=x0)
        states = self.settle_nudge(x0, target, n_sweeps=n_sweeps, beta=beta, **async_kw)
        opt.zero_grad(set_to_none=True)
        self.set_mf_grads(states, mode=mf_mode)
        opt.step()

    @torch.no_grad()
    def predict_logits(self, x0: torch.Tensor) -> torch.Tensor:
        return self.feedforward(x0)[-1]

    @torch.no_grad()
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.predict_logits(x0)

    def extra_repr(self) -> str:
        return (f"dims={self.dims}, depth={self.depth}, gamma={self.gamma}, "
                f"relu_readout={self.relu_readout}, residual={self.residual}")


# ---------------------------------------------------------------------------
# Generative PC — dyadic sparse coding (the inference capability backprop lacks)
# ---------------------------------------------------------------------------


class SparseCodingPC(nn.Module):
    """Single-layer **generative** predictive coding = dyadic sparse coding.

    Models each input as a SPARSE combination of dictionary atoms, ``x ≈ D·s`` with
    ``s`` k-sparse. The code ``s`` is **inferred by settling** (descend
    ``½‖mask⊙(x − D·s)‖²`` then keep the top-k atoms — winner-take-all, the
    tropical/max-plus primitive); the dyadic dictionary ``D`` (signed powers of two,
    column-scaled by powers of two) is learned by a **local** update
    (``ΔD ∝ err ⊗ s``) — *no backprop*. This
    is the Olshausen-Field / Rao-Ballard generative PC on the tropical/dyadic
    substrate, the one PC formulation where the max/WTA structure is the point.

    The contrast with a feed-forward net is the whole value: ONE trained model
    answers *arbitrary* inference queries by settling — reconstruction, in-painting
    under ANY mask, denoising — which a discriminative backprop net structurally
    cannot. (Measured: MNIST, recovers digits from 50% random pixel masking ~60%
    better than mean-fill; see ``results/predictive_coding/GENERATIVE_PC.md``.)

    Closure: the deployed dictionary is ±2^k (shift-add) and the code is a top-k
    gate — multiplier-free as a weight-VALUE property; the float-sim inference
    matmuls are the usual artifact (cf. :class:`DyadicLinear`).
    """

    def __init__(self, n_pixels: int, n_atoms: int, n_active: int = 15):
        super().__init__()
        if n_pixels < 1 or n_atoms < 1:
            raise ValueError("n_pixels and n_atoms must be positive")
        if not 1 <= n_active <= n_atoms:
            raise ValueError("n_active must be between 1 and n_atoms")
        self.n_pixels = n_pixels
        self.n_atoms = n_atoms
        self.n_active = n_active
        self.D = nn.Parameter(torch.randn(n_pixels, n_atoms) * 0.1)
        self._renormalize_shadow()

    @torch.no_grad()
    def _renormalize_shadow(self) -> None:
        """Bound continuous shadow columns using only a power-of-two rescale."""
        magnitude = torch.linalg.vector_norm(
            self.D, dim=0, keepdim=True
        ).clamp_min(1e-12)
        self.D.mul_(torch.exp2(-torch.log2(magnitude).round()))

    def _dict(self) -> torch.Tensor:
        """Deployed dictionary with exactly signed-dyadic entries.

        The continuous shadow is kept near unit column norm using a power-of-two
        rescale after each update; deployment then only quantizes each entry.
        There is no runtime division that can turn ±2^k entries into arbitrary
        real coefficients.
        """
        from tikitorch.nn import dyadic_quantize
        return dyadic_quantize(self.D)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
              n_steps: int = 40, lr: float = 0.25):
        """Settle the k-sparse code for ``x``; ``mask`` (1=observed) restricts the fit
        to observed pixels. Returns ``(s, reconstruction)``."""
        Dn = self._dict()
        s = torch.zeros(x.shape[0], self.n_atoms, device=x.device, dtype=x.dtype)
        for _ in range(n_steps):
            err = x - s @ Dn.t()
            if mask is not None:
                err = err * mask
            s = s + lr * (err @ Dn)
            thr = s.abs().topk(self.n_active, dim=1).values[:, -1:]   # top-k / WTA
            s = s * (s.abs() >= thr).float()
        return s, s @ Dn.t()

    @torch.no_grad()
    def complete(self, x: torch.Tensor, mask: torch.Tensor,
                 n_steps: int = 120, lr: float = 0.25) -> torch.Tensor:
        """In-paint / denoise: infer the code from the OBSERVED pixels and fill the
        rest. ``mask`` is 1 on observed pixels, 0 on missing. Any mask pattern."""
        _, recon = self.infer(x * mask, mask=mask, n_steps=n_steps, lr=lr)
        return mask * x + (1.0 - mask) * recon

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, n_steps: int = 60,
                    lr: float = 0.25) -> torch.Tensor:
        return self.infer(x, n_steps=n_steps, lr=lr)[1]

    @torch.no_grad()
    def train_step(self, x: torch.Tensor, lr_dict: float = 0.05,
                   n_steps: int = 25, lr_code: float = 0.25) -> float:
        """One local learning step: settle the code, apply the LOCAL dyadic dictionary
        update (operand-quantized ``ΔD ∝ err ⊗ s``), re-normalize atoms. No backprop,
        no optimizer. Returns the reconstruction MSE."""
        from tikitorch.nn import dyadic_quantize
        s, recon = self.infer(x, n_steps=n_steps, lr=lr_code)
        err = x - recon
        self.D += lr_dict * (dyadic_quantize(err).t() @ dyadic_quantize(s)) / x.shape[0]
        # Bound continuous shadow weights with a power-of-two column rescale.
        # Quantizing the shadow after every mini-batch destroys the small local
        # updates; `_dict()` applies the signed-dyadic deployment constraint.
        self._renormalize_shadow()
        return float((err ** 2).mean())

    def extra_repr(self) -> str:
        return (f"n_pixels={self.n_pixels}, n_atoms={self.n_atoms}, "
                f"n_active={self.n_active}")
