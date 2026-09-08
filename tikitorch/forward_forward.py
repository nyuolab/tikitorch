"""tikitorch.forward_forward — Tropical Forward-Forward (TFF).

A layer-local rule without end-to-end backpropagation: Hinton's Forward-Forward
(2022, *The Forward-Forward Algorithm*) adapted to the tropical (max-plus)
substrate. Each layer is its own learner — it adjusts its weights from a local
goodness objective on its own activations, with no global backward pass threading
gradients from the output back through the stack. There are two forward passes
(one on *positive* data, one on *negative* data) instead of a forward + backward.

Why this is interesting *here*: the reference rule uses compare/add/shift-oriented
choices in places where Hinton's float recipe uses products. These choices make a
constrained implementation possible, but the complete configured step still needs
an executed software audit and a physical build audit before it supports a closure
or DSP-utilization claim:

  * **Layer** = max-plus dilation ``h_j = max_i(W[j,i] + x_i)`` (the silicon
    forward kernel), optionally followed by :func:`tikitorch.functional.tropical_relu`.
  * **Goodness** is NOT Hinton's sum-of-squares ``Σ h_j²`` (a multiply). The
    default is ``G = sum of the top-k activations`` (additions only); ``G = max``
    is also provided. The goodness function is pluggable (:data:`GOODNESS_FNS`).
  * **Local objective** is a **hinge / margin** loss (push ``G_pos ≥ θ+m`` and
    ``G_neg ≤ θ−m``). Its gradient w.r.t. goodness is ``±1`` — sign-valued — so the
    weight update reduces to a **sign-gated update on the winning max-plus edges**,
    exactly the SignSGD / ``PureTropicalSGD`` path. The per-activation gradient is
    built by *scattering* ``{-1,0,+1}`` into the top-k positions (no tensor×tensor
    multiply), mirroring :class:`tikitorch.functional.TropicalPerceptronLossFn`.
    The reference loss returns a batch mean. A whole-step shift-only certificate
    therefore also requires a power-of-two reduction extent (the closure examples
    use a power-of-two batch); the auditor correctly reports a non-dyadic mean as
    a divide.
  * **Inter-layer normalization is TROPICAL**: subtract the max,
    ``h ← h − max(h)`` (:func:`tropical_max_normalize`) — the tropical analogue of
    FF's length-normalization (it removes the magnitude the previous layer's
    goodness used so the next layer cannot trivially read it off). NOT an L2 norm
    (that is a square + a divide + an ``rsqrt``).

  An optional **logistic** FF loss (:func:`tropical_ff_logistic_loss`, Hinton's
  original) is also provided; its per-sample coefficient is a host-side sigmoid
  (a transcendental, like CE's host softmax) so the *per-weight* update still has
  no multiply — but with a sign-only optimizer it degenerates to an always-on
  push (see that function's docstring). The hinge loss is the recommended default.

Usage::

    import tikitorch as tt
    from tikitorch.forward_forward import (
        TropicalFFNet, TropicalFFTrainer, overlay_label, make_negative_labels,
        goodness_classify,
    )

    net = TropicalFFNet([784, 500, 500], num_classes=10, goodness="topk", k=8)
    trainer = TropicalFFTrainer(net, lr_shift=4)          # PureTropicalSGD, sign-only

    for x, y in loader:                                   # x: (B, 784) in [0,1]
        x_pos = overlay_label(x, y, num_classes=10)       # correct label overlaid
        x_neg = overlay_label(x, make_negative_labels(y, 10), num_classes=10)
        trainer.train_step(x_pos, x_neg)                  # layer-local update

    preds, _ = goodness_classify(net, x_test, num_classes=10)

The max-plus projection can target the existing dilation kernel
(``tikitorch.fpga`` / the HLS ``forge`` kernels): a TFF layer's inference is a
single ``y = max_i(W + x)`` followed by a subtract-max, so a trained TFF stack
uses the same core primitive. Export, numeric equivalence, and resource use remain
properties of the concrete hardware build.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from tikitorch.functional import tropical_relu, tropical_leaky_relu
from tikitorch.nn import TropicalDilationModule

try:  # optimizer is import-guarded in the package __init__; degrade gracefully.
    from tikitorch.optim import PureTropicalSGD
except Exception:  # pragma: no cover - import guard
    PureTropicalSGD = None  # type: ignore[assignment]

__all__ = [
    "goodness_topk", "goodness_max", "GOODNESS_FNS",
    "tropical_max_normalize",
    "TropicalFFHingeLossFn", "tropical_ff_hinge_loss", "tropical_ff_logistic_loss",
    "TropicalFFLayer", "TropicalFFNet", "TropicalFFTrainer",
    "overlay_label", "make_negative_labels", "scramble_features",
    "goodness_classify",
]


# ---------------------------------------------------------------------------
# Goodness functions — multiplier-free, pluggable
# ---------------------------------------------------------------------------
#
# Hinton's goodness is ``Σ_j h_j²`` (the sum of squared activities). That square
# is a hardware multiply, so it is off the table for a closed loop. Both options
# below are addition / comparison only.


def goodness_topk(h: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Goodness = sum of the top-``k`` activations along the last dim.

    Pure additions (a top-k select is comparisons; the sum is adds), so it is
    closure-clean. Reduces to :func:`goodness_max` at ``k=1``. ``k`` is clamped to
    the feature width. Returns shape ``(...,)`` (the last dim reduced)."""
    kk = max(1, min(int(k), h.size(-1)))
    return h.topk(kk, dim=-1).values.sum(dim=-1)


def goodness_max(h: torch.Tensor) -> torch.Tensor:
    """Goodness = the single max activation along the last dim (a comparison)."""
    return h.max(dim=-1).values


GOODNESS_FNS: Dict[str, Callable[..., torch.Tensor]] = {
    "topk": goodness_topk,
    "max": goodness_max,
}


def _goodness_and_indices(h: torch.Tensor, goodness: str, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(g, idx)``: the per-row goodness and the indices it summed over.

    ``idx`` is ``(B, kk)`` (top-k) or ``(B, 1)`` (max) — the positions that own the
    goodness, i.e. exactly where the gradient w.r.t. ``h`` is non-zero.
    """
    if goodness == "topk":
        kk = max(1, min(int(k), h.size(-1)))
        vals, idx = h.topk(kk, dim=-1)
        return vals.sum(dim=-1), idx
    if goodness == "max":
        vals, idx = h.max(dim=-1, keepdim=True)
        return vals.squeeze(-1), idx
    raise ValueError(f"goodness must be one of {list(GOODNESS_FNS)}, got {goodness!r}")


# ---------------------------------------------------------------------------
# Tropical inter-layer normalization — subtract the max (NOT L2)
# ---------------------------------------------------------------------------


def tropical_max_normalize(h: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Tropical length-normalization: ``h − max(h)`` along ``dim``.

    FF normalizes each hidden vector to unit L2 length so the *next* layer cannot
    cheat by reading off the previous layer's goodness (its length); it must use
    the *relative* activities. The tropical analogue subtracts the max, which pins
    the peak to 0 and leaves only relative offsets — a comparison and a subtract,
    no square / divide / ``rsqrt``. Closure-clean."""
    return h - h.max(dim=dim, keepdim=True).values


# ---------------------------------------------------------------------------
# The local FF loss — hinge / margin, multiplier-free sign-gated backward
# ---------------------------------------------------------------------------


class TropicalFFHingeLossFn(torch.autograd.Function):
    """Forward-Forward hinge loss with a sign-gated custom backward.

    Goodness should be HIGH on positive data and LOW on negative data. With a
    threshold ``θ`` and margin ``m`` the per-sample objective is::

        positive:  push  G ≥ θ + m   ->   loss = relu(θ + m − G)
        negative:  push  G ≤ θ − m   ->   loss = relu(G − (θ − m))

    ``forward`` returns the mean hinge (a readable, continuous loss curve).
    ``backward`` returns the gradient w.r.t. the activations ``h``: the goodness
    gradient ``dL/dG`` is ``−1`` (active positive), ``+1`` (active negative), or
    ``0`` (margin satisfied), and ``dG/dh`` is a 0/1 mask at the top-k positions —
    so ``dL/dh`` is ``{-1,0,+1}`` placed by **scatter** into those positions. No
    tensor×tensor multiply (same straight-through construction as
    :class:`tikitorch.functional.TropicalPerceptronLossFn`). Fed through the
    max-plus :class:`~tikitorch.functional.TropicalDilationFn` backward (a scatter)
    and a sign optimizer, the intended update algebra uses comparisons and
    additions. Audit the complete executed recipe and any synthesized build.

    The continuous forward value and the sign-valued backward are intentionally
    decoupled (a straight-through pattern), so ``gradcheck`` does not apply; the
    backward is the object the silicon realizes.
    """

    @staticmethod
    def forward(ctx, h, is_pos, theta, margin, goodness, k):
        # h: (B, D) float; is_pos: (B,) bool; theta: scalar tensor/number.
        g, idx = _goodness_and_indices(h, goodness, int(k))     # g:(B,), idx:(B,kk)

        pos_target = theta + margin
        neg_target = theta - margin
        active_pos = is_pos & (g < pos_target)                  # below target -> push up
        active_neg = (~is_pos) & (g > neg_target)               # above target -> push down

        # Per-sample goodness gradient sign in {-1, 0, +1} (comparisons only).
        coeff = torch.zeros_like(g)
        coeff = torch.where(active_pos, coeff - 1.0, coeff)     # dL/dG = -1
        coeff = torch.where(active_neg, coeff + 1.0, coeff)     # dL/dG = +1

        # Scatter the (shared) per-sample coeff into the goodness positions: this
        # is dL/dh = (dL/dG) * 1{position owns the goodness}, built WITHOUT a
        # multiply (the {-1,0,+1} value is written, not multiplied in).
        grad_h = torch.zeros_like(h)
        src = coeff.unsqueeze(1).expand(-1, idx.size(1)).contiguous()
        grad_h.scatter_(1, idx, src)
        ctx.save_for_backward(grad_h)
        ctx.in_shape = h.shape

        # Readable scalar loss (continuous); does not condition the update.
        loss_pos = (pos_target - g).clamp_min(0.0)
        loss_neg = (g - neg_target).clamp_min(0.0)
        hinge = torch.where(is_pos, loss_pos, loss_neg)
        return hinge.mean()

    @staticmethod
    def backward(ctx, grad_output):
        (grad_h,) = ctx.saved_tensors
        # grad_output is the scalar 1.0 (a power of two -> a shift, closure-clean)
        # when this is the loss the step backprops from.
        return (grad_output * grad_h).reshape(ctx.in_shape), None, None, None, None, None


def tropical_ff_hinge_loss(h: torch.Tensor, is_pos: torch.Tensor,
                           theta: Union[float, torch.Tensor] = 0.0, margin: float = 1.0,
                           goodness: str = "topk", k: int = 8) -> torch.Tensor:
    """Scalar FF hinge loss with a sign-gated custom gradient.

    ``h`` is ``(B, D)`` activations, ``is_pos`` a ``(B,)`` bool mask (True =
    positive sample). A tensor threshold remains on-device and is treated as a
    detached control value, not a differentiable parameter. See
    :class:`TropicalFFHingeLossFn`.
    """
    if is_pos.dtype != torch.bool:
        is_pos = is_pos.bool()
    if isinstance(theta, torch.Tensor):
        if theta.ndim != 0:
            raise ValueError("theta must be a scalar")
        theta = theta.detach().to(device=h.device, dtype=h.dtype)
    return TropicalFFHingeLossFn.apply(h, is_pos, theta, float(margin), goodness, int(k))


def tropical_ff_logistic_loss(h: torch.Tensor, is_pos: torch.Tensor,
                              theta: Union[float, torch.Tensor] = 0.0, goodness: str = "topk",
                              k: int = 8) -> torch.Tensor:
    """Hinton's original logistic FF loss — host-conditioned coefficient.

    Loss = ``softplus(−(G−θ))`` on positives and ``softplus(G−θ)`` on negatives;
    the gradient w.r.t. ``G`` is ``−σ(θ−G)`` (positive) / ``σ(G−θ)`` (negative), a
    per-sample scalar in ``(−1, 1)``. That coefficient is computed with a **host
    sigmoid** (a transcendental, exactly like ``tropical_cross_entropy``'s host
    softmax) and then *scattered* into the goodness positions — so the **per-weight
    update has no multiply**, but computing the coefficient does use a
    transcendental. Two consequences to be honest about:

    * Under the closure guard this needs ``allow_transcendental=True`` (the
      coefficient sigmoid), unlike the fully-closed :func:`tropical_ff_hinge_loss`.
    * With a **sign-only** optimizer the magnitude of the coefficient is discarded,
      so the soft confidence-weighting collapses to an always-on ``±1`` push
      (``sign(−σ) = −1`` for every positive, ``+1`` for every negative) — i.e. it
      reduces to a hinge with an infinite margin. The soft weighting only does
      anything with a magnitude-aware update (a non-closed optimizer, or a
      frexp/agreement-tiered one). Prefer the hinge loss for the closed loop.
    """
    if isinstance(theta, torch.Tensor):
        if theta.ndim != 0:
            raise ValueError("theta must be a scalar")
        theta = theta.detach().to(device=h.device, dtype=h.dtype)
    g, idx = _goodness_and_indices(h, goodness, int(k))
    with torch.no_grad():
        # coeff = dL/dG; host sigmoid (transcendental) -> scattered, never multiplied per-weight.
        coeff_pos = -torch.sigmoid(theta - g)                   # in (-1, 0)
        coeff_neg = torch.sigmoid(g - theta)                    # in (0, 1)
        coeff = torch.where(is_pos.bool(), coeff_pos, coeff_neg)
    return _ScatterCoeffLossFn.apply(h, idx, coeff, g, is_pos.bool(), theta)


class _ScatterCoeffLossFn(torch.autograd.Function):
    """Straight-through loss for :func:`tropical_ff_logistic_loss`: a precomputed
    per-sample coefficient scattered into the goodness positions (no per-weight
    multiply); forward returns the readable softplus loss."""

    @staticmethod
    def forward(ctx, h, idx, coeff, g, is_pos, theta):
        grad_h = torch.zeros_like(h)
        src = coeff.unsqueeze(1).expand(-1, idx.size(1)).contiguous()
        grad_h.scatter_(1, idx, src)
        ctx.save_for_backward(grad_h)
        ctx.in_shape = h.shape
        loss = torch.where(is_pos, torch.nn.functional.softplus(theta - g),
                           torch.nn.functional.softplus(g - theta))
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        (grad_h,) = ctx.saved_tensors
        return (grad_output * grad_h).reshape(ctx.in_shape), None, None, None, None, None


# ---------------------------------------------------------------------------
# The TFF layer — one local learner
# ---------------------------------------------------------------------------


SUBSTRATES = ("maxplus", "dyadic")


class TropicalFFLayer(nn.Module):
    """One Forward-Forward layer: a configurable projection plus optional ReLU.

    ``forward(x) = relu(proj(x))``. The projection is the pluggable
    *substrate*:

    * ``substrate="maxplus"`` (default) — :class:`tikitorch.nn.TropicalDilationModule`,
      ``h_j = max_i(W[j,i] + x_i) + b_j``. The tropical winner-take-all kernel;
      passes the *runtime* closure guard (``tt.assert_no_multiply``) and supports a
      soft (``logsumexp``) ``set_tau`` anneal (harden ``tau→0`` for deploy).
    * ``substrate="dyadic"`` — :class:`tikitorch.nn.DyadicLinear`, a dense linear
      whose weights are signed powers of two (``±2^k``), so ``W·x`` admits a
      shift-add export schedule. Dense/additive, it is a distinct capacity choice
      from max-plus. Its export property is checked statically
      (``fpga.dyadic_export.verify_schedule``), NOT the eager op trace — the
      float-sim runs a real ``addmm``, so it does **not** pass the runtime
      ``assert_no_multiply`` (this is the documented dyadic caveat; the *update*
      is sign-oriented). Physical DSP use is determined from synthesis reports.

    The goodness, subtract-max normalization, and hinge loss are substrate-agnostic
    (they act on ``h``) — a TFF layer *is* its own local learner either way.

    Args:
        in_features: input projection dimension.
        out_features: output projection dimension.
        substrate: ``"maxplus"`` (default) or ``"dyadic"``.
        use_relu: whether to apply ReLU after the projection.
        leaky_shift: optional power-of-two leaky-ReLU shift.
        goodness: ``"topk"`` (default) or ``"max"``.
        k: top-k width for ``goodness="topk"``.
        init_identity: seed a square max-plus dilation as identity (max-plus only).
    """

    def __init__(self, in_features: int, out_features: int, *, substrate: str = "maxplus",
                 use_relu: bool = True, leaky_shift: Optional[int] = None,
                 goodness: str = "topk", k: int = 8, init_identity: bool = False):
        super().__init__()
        if goodness not in GOODNESS_FNS:
            raise ValueError(f"goodness must be one of {list(GOODNESS_FNS)}, got {goodness!r}")
        if substrate not in SUBSTRATES:
            raise ValueError(f"substrate must be one of {SUBSTRATES}, got {substrate!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.substrate = substrate
        self.use_relu = use_relu
        self.leaky_shift = leaky_shift
        self.goodness_name = goodness
        self.k = int(k)
        if substrate == "maxplus":
            self.proj = TropicalDilationModule(in_features, out_features,
                                               init_identity=init_identity)
        else:
            from tikitorch.nn import DyadicLinear
            self.proj = DyadicLinear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        if self.use_relu:
            h = (tropical_leaky_relu(h, shift=self.leaky_shift)
                 if self.leaky_shift is not None else tropical_relu(h))
        return h

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """The layer's goodness on activations ``h`` (per sample)."""
        if self.goodness_name == "topk":
            return goodness_topk(h, self.k)
        return goodness_max(h)

    def normalize(self, h: torch.Tensor) -> torch.Tensor:
        """Tropical normalization (subtract-max) of this layer's output, the
        (detached) input handed to the next layer."""
        return tropical_max_normalize(h)

    def set_tau(self, tau: float) -> None:
        """Flip the projection into the soft (``tau>0``) / hard (``tau==0``) path.
        No-op for the dyadic substrate (it deploys as-is)."""
        if hasattr(self.proj, "set_tau"):
            self.proj.set_tau(tau)

    def extra_repr(self) -> str:
        relu = (f"leaky2**-{self.leaky_shift}" if self.leaky_shift is not None
                else ("relu" if self.use_relu else "linear"))
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"substrate={self.substrate}, act={relu}, "
                f"goodness={self.goodness_name}, k={self.k}")


# ---------------------------------------------------------------------------
# The TFF network — a stack of local learners + the goodness classifier
# ---------------------------------------------------------------------------


class TropicalFFNet(nn.Module):
    """A stack of :class:`TropicalFFLayer` learners.

    ``dims = [in, h1, h2, ...]`` builds ``len(dims)-1`` layers. The forward pass
    runs each layer and tropically-normalizes between them; :meth:`layer_goodness`
    and :meth:`accumulated_goodness` expose the per-layer and summed goodness used
    for training and the label-overlay classifier.

    This is a plain ``nn.Module`` feature extractor — it is trained by
    :class:`TropicalFFTrainer` (forward-only, local), NOT by ``loss.backward()``
    on a global head.
    """

    def __init__(self, dims: Sequence[int], *, num_classes: int = 10,
                 substrate: str = "maxplus", use_relu: bool = True,
                 leaky_shift: Optional[int] = None, goodness: str = "topk", k: int = 8):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must list at least [in, out]")
        self.num_classes = num_classes
        self.substrate = substrate
        self.layers = nn.ModuleList([
            TropicalFFLayer(dims[i], dims[i + 1], substrate=substrate, use_relu=use_relu,
                            leaky_shift=leaky_shift, goodness=goodness, k=k)
            for i in range(len(dims) - 1)
        ])

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten to ``(B, features)`` and floatify (the dilation needs floats)."""
        if not x.is_floating_point():
            x = x.float()
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final tropically-normalized activations (the features)."""
        h = self._prep(x)
        for layer in self.layers:
            h = layer.normalize(layer(h))
        return h

    @torch.no_grad()
    def layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Per-layer post-activation (pre-normalization) activations."""
        h = self._prep(x)
        acts = []
        for layer in self.layers:
            a = layer(h)
            acts.append(a)
            h = layer.normalize(a)
        return acts

    def accumulated_goodness(self, x: torch.Tensor, skip_first: bool = True) -> torch.Tensor:
        """Sum of per-layer goodness for each sample (the classifier's score).

        ``skip_first`` drops the first layer's goodness from the sum (Hinton's
        recipe: the first hidden layer is too correlated with the raw label
        overlay to be a clean class signal). Auto-disabled for a 1-layer net.

        Grad-capable (the goodness logits are differentiable, so a model can put a
        loss on them); the eval classifier :func:`goodness_classify` wraps its own
        ``no_grad``.
        """
        h = self._prep(x)
        total = None
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            a = layer(h)
            if not (skip_first and n > 1 and i == 0):
                g = layer.goodness(a)
                total = g if total is None else total + g
            h = layer.normalize(a)
        return total

    def set_tau(self, tau: float) -> None:
        for layer in self.layers:
            layer.set_tau(tau)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# The greedy per-layer trainer
# ---------------------------------------------------------------------------


class TropicalFFTrainer:
    """Greedy, layer-local trainer for a :class:`TropicalFFNet`.

    Each layer owns a sign-only optimizer and trains on its **own** local hinge
    loss; the (tropically-normalized) activations are detached before being handed
    to the next layer, so no gradient ever crosses a layer boundary — there is no
    global backward pass. One :meth:`train_step` does one positive + one negative
    forward pass per layer.

    The threshold ``θ`` defaults to the **adaptive batch midpoint**
    ``0.5·(mean G_pos + mean G_neg)``, which tracks the unknown goodness scale of
    a max-plus layer;
    pass a float for a fixed threshold (Hinton's original).

    Args:
        net: the :class:`TropicalFFNet` to train.
        optimizer_factory: ``params -> Optimizer``; default builds a
            :class:`tikitorch.optim.PureTropicalSGD` (sign-oriented reference).
        theta: ``"adaptive"`` (default) or a float threshold.
        margin: hinge margin ``m`` (goodness units).
        lr_shift: forwarded to the default ``PureTropicalSGD``.
        momentum_bits: forwarded to the default ``PureTropicalSGD``.
        host_metrics: return Python floats by default. Set False for detached
            scalar-tensor diagnostics during an executed closure audit; convert
            them to host values only after leaving the audit scope.
    """

    def __init__(self, net: TropicalFFNet, *,
                 optimizer_factory: Optional[Callable[[object], object]] = None,
                 theta: Union[str, float] = "adaptive", margin: float = 1.0,
                 lr_shift: int = 4, momentum_bits: int = 0,
                 host_metrics: bool = True):
        self.net = net
        if optimizer_factory is None:
            if PureTropicalSGD is None:  # pragma: no cover - import guard
                raise RuntimeError("tikitorch.optim.PureTropicalSGD unavailable; "
                                   "pass optimizer_factory explicitly")

            def optimizer_factory(params):
                return PureTropicalSGD(params, lr_shift=lr_shift, momentum_bits=momentum_bits)
        self.optimizers = [optimizer_factory(layer.parameters()) for layer in net.layers]
        self.theta_policy = theta
        self.margin = float(margin)
        self.host_metrics = host_metrics

    def _theta(self, gp: torch.Tensor, gn: torch.Tensor) -> Union[float, torch.Tensor]:
        if isinstance(self.theta_policy, str):
            if self.theta_policy != "adaptive":
                raise ValueError(f"theta must be 'adaptive' or a float, got {self.theta_policy!r}")
            # batch midpoint: (mean + mean) * 0.5  -> add + power-of-2 shift (closed)
            return ((gp.mean() + gn.mean()) * 0.5).detach()
        return float(self.theta_policy)

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor, *,
                   host_metrics: Optional[bool] = None) -> Dict[str, list]:
        """Train every layer one step on a positive/negative batch pair.

        Returns ``{"loss": [...per layer], "g_pos": [...], "g_neg": [...]}`` (the
        per-layer mean goodness on positive and negative data — the separation
        diagnostic). Forward-only: each layer's gradient touches only its own
        weights.

        ``host_metrics=False`` retains detached scalar tensors and avoids
        tensor-to-Python conversions. The default follows the constructor's
        setting (Python floats for backward compatibility). Host conversion is
        an explicit audit boundary; it never feeds back into the weight update.
        """
        net = self.net
        hp = net._prep(x_pos)
        hn = net._prep(x_neg)
        out = {"loss": [], "g_pos": [], "g_neg": []}
        host_metrics = self.host_metrics if host_metrics is None else host_metrics
        for layer, opt in zip(net.layers, self.optimizers):
            inp_p, inp_n = hp.detach(), hn.detach()
            ap = layer(inp_p)                       # grad flows to layer.weight only
            an = layer(inp_n)
            with torch.no_grad():
                gp = layer.goodness(ap)
                gn = layer.goodness(an)
                theta = self._theta(gp, gn)
            h = torch.cat([ap, an], dim=0)
            is_pos = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
            is_pos[:ap.size(0)] = True
            opt.zero_grad()
            loss = tropical_ff_hinge_loss(h, is_pos, theta=theta, margin=self.margin,
                                          goodness=layer.goodness_name, k=layer.k)
            loss.backward()
            opt.step()
            for name, value in (("loss", loss.detach()), ("g_pos", gp.mean()),
                                ("g_neg", gn.mean())):
                out[name].append(float(value) if host_metrics else value.detach())
            # tropically-normalized, DETACHED activations -> next layer's input.
            hp = layer.normalize(ap).detach()
            hn = layer.normalize(an).detach()
        return out


# ---------------------------------------------------------------------------
# Negative-data generators
# ---------------------------------------------------------------------------


def overlay_label(x: torch.Tensor, labels: torch.Tensor, num_classes: int,
                  on: float = 1.0, off: float = 0.0) -> torch.Tensor:
    """Hinton's label-overlay: write a one-hot label into the first ``num_classes``
    features of a flattened input (a *copy* is returned).

    Positive data overlays the **correct** label; negative data overlays a
    **wrong** one (see :func:`make_negative_labels`). At eval, the class whose
    overlay maximizes the accumulated goodness is the prediction
    (:func:`goodness_classify`). ``on``/``off`` are the high/low overlay values
    (defaults suit ``[0,1]``-scaled images). ``x`` is flattened to ``(B, -1)``.
    """
    if x.dim() > 2:
        x = x.reshape(x.size(0), -1)
    x = x.clone()
    rows = torch.arange(x.size(0), device=x.device)
    x[:, :num_classes] = off
    x[rows, labels.to(x.device)] = on
    return x


def make_negative_labels(labels: torch.Tensor, num_classes: int,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """A wrong label for each sample: uniform over the ``num_classes-1`` classes
    that are not the true one (``(label + U[1, C-1]) mod C`` — never equal).

    Input labels must lie in ``[0, num_classes)``. The bounded sum needs only
    one conditional subtraction to wrap, without integer division/remainder.
    """
    offset = torch.randint(1, num_classes, labels.shape, generator=generator,
                           device=labels.device)
    candidate = labels + offset
    return torch.where(candidate >= num_classes, candidate - num_classes, candidate)


def scramble_features(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Tropical-native negative data: independently permute each feature across
    the batch (a per-column shuffle).

    This is the routing-scramble negative: it leaves every feature's marginal
    distribution **unchanged** but destroys the cross-feature *joint* structure
    that a max-plus layer's argmax routing depends on (which input wins each
    output's max). The result looks locally plausible per-feature yet has no
    coherent object — Hinton's "combine features from different images", made
    tropical. Use it for an unlabeled / label-free negative source (no overlay).
    ``x`` is flattened to ``(B, -1)``. Requires ``B ≥ 2``.
    """
    if x.dim() > 2:
        x = x.reshape(x.size(0), -1)
    b, d = x.shape
    if b < 2:
        return x.clone()
    # An independent permutation of the batch index for each of the d columns.
    perm = torch.argsort(torch.rand(b, d, generator=generator, device=x.device), dim=0)
    return torch.gather(x, 0, perm)


# ---------------------------------------------------------------------------
# Evaluation — the label-overlay goodness classifier
# ---------------------------------------------------------------------------


@torch.no_grad()
def goodness_classify(net: TropicalFFNet, x: torch.Tensor, num_classes: int,
                      on: float = 1.0, off: float = 0.0,
                      skip_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify by the label whose overlay maximizes accumulated goodness.

    For each candidate class ``c``, overlay ``c`` on the input, run the forward
    pass, and sum the per-layer goodness (:meth:`TropicalFFNet.accumulated_goodness`,
    ``skip_first`` per Hinton). Predict ``argmax_c``. Returns ``(preds, scores)``
    with ``scores`` shape ``(B, num_classes)``. Fully multiplier-free (forward
    kernel + goodness adds)."""
    x = net._prep(x)
    b = x.size(0)
    scores = x.new_zeros(b, num_classes)
    for c in range(num_classes):
        labels = torch.full((b,), c, device=x.device, dtype=torch.long)
        xc = overlay_label(x, labels, num_classes, on=on, off=off)
        scores[:, c] = net.accumulated_goodness(xc, skip_first=skip_first)
    return scores.argmax(dim=1), scores
