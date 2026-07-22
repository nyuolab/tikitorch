"""E6 optimizer family — IDBD-style per-weight adaptive step sizes + the
anchor seam (E6_SPEC arms 1-4).

Three meta-step-size optimizers, all sharing the same skeleton: each weight
carries a log2 step-size offset beta (alpha_i = lr * 2^beta_i), updated by
Sutton's delta-bar-delta correlation rule (IDBD, Sutton 1992) with
Autostep-style running-max normalization of the meta-gradient (Mahmood,
Sutton, Degris & Pilarski 2012) so every meta-step is bounded:
|delta beta| <= theta_meta, always. The normalization is the stability story
on the tropical substrate, whose top-k routing produces long exact-zero
gradient runs punctuated by spikes.

- IDBDAdam   : alpha scales the bias-corrected Adam DIRECTION (m/sqrt(v)).
- IDBDDyadic : same beta dynamics; the APPLIED exponent is an int8 q moved
               with +-hysteresis at half-integer boundaries (shift-class
               steps, no flapping).
- IDBDSGD    : classic gradient-form IDBD (delta*x -> -g), plus a dyadic
               h-leak so zero-grad droughts can't freeze stale correlations.

Weight decay is DECOUPLED (AdamW semantics) and scaled by the per-weight
alpha, not the group lr: with group-scaled decay, a coordinate whose alpha
shrank 64x would have decay dominate its own updates 64:1 — a systematic
pull-to-zero on exactly the consolidated weights.

theta_meta=0 reduces IDBDAdam to AdamW bitwise-modulo-arithmetic (unit-
tested with allclose); optimizer="adamw" in build_life_opt returns the E5
torch.optim.AdamW verbatim.
"""
import torch


class _IDBDBase(torch.optim.Optimizer):
    """Shared state plumbing; subclasses implement _update(group, p, st)."""

    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self._step_count = 0

    def state_dict(self):
        sd = super().state_dict()
        sd["tt_step_count"] = self._step_count
        return sd

    def load_state_dict(self, state_dict):
        sd = dict(state_dict)
        self._step_count = sd.pop("tt_step_count", 0)
        super().load_state_dict(sd)
        self._coerce_state()

    def _coerce_state(self):
        pass  # IDBDDyadic re-coerces its int8 q (base Optimizer float-casts)

    def beta_mean(self) -> float:
        tot, n = 0.0, 0
        for st in self.state.values():
            b = st.get("beta")
            if b is not None:
                tot += float(b.sum())
                n += b.numel()
        return tot / max(n, 1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._step_count += 1
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    continue    # inf/NaN grads would NaN beta via raw/N
                                # (review finding: Autostep bound is finite-
                                # gradient-conditional); skip, let nan_guard
                                # handle the upstream cause
                st = self.state[p]
                if not st:
                    self._init_state(p, st)
                self._update(group, p, st)
        return loss


class IDBDAdam(_IDBDBase):
    """Per-weight alpha = lr * 2^beta on the Adam direction; delta-bar-delta
    meta rule on the applied update, Autostep-normalized."""

    WARMUP = 10  # meta rule sleeps while Adam moments are still biased junk

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, theta_meta=0.01, lambda_h=1.0,
                 beta_min=-6.0, beta_max=2.0, norm_decay=1.0 - 2.0 ** -10):
        super().__init__(params, dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            theta_meta=theta_meta, lambda_h=lambda_h,
            beta_min=beta_min, beta_max=beta_max, norm_decay=norm_decay))

    def _init_state(self, p, st):
        st["m"] = torch.zeros_like(p)
        st["v"] = torch.zeros_like(p)
        st["h"] = torch.zeros_like(p)
        st["beta"] = torch.zeros_like(p)
        st["N"] = torch.full_like(p, 1e-12)
        st["t"] = torch.zeros((), dtype=torch.int64)

    def _alpha(self, group, st):
        return group["lr"] * torch.exp2(st["beta"])

    def _update(self, group, p, st):
        b1, b2 = group["betas"]
        g = p.grad
        st["t"] += 1
        t = int(st["t"])
        st["m"].mul_(b1).add_(g, alpha=1 - b1)
        st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
        d = (st["m"] / (1 - b1 ** t)) / ((st["v"] / (1 - b2 ** t)).sqrt()
                                         + group["eps"])
        if group["theta_meta"] > 0 and t > self.WARMUP:
            u_pre = -self._alpha(group, st) * d
            raw = u_pre * st["h"]
            st["N"] = torch.maximum(raw.abs(), group["norm_decay"] * st["N"])
            st["beta"].add_(group["theta_meta"] * raw
                            / st["N"].clamp_min(1e-12)) \
                      .clamp_(group["beta_min"], group["beta_max"])
        alpha = self._alpha(group, st)
        applied = -alpha * d
        if group["weight_decay"]:
            p.mul_(1 - alpha * group["weight_decay"])
        p.add_(applied)
        st["h"].mul_((1 - alpha * group["lambda_h"]).clamp_min(0)) \
               .add_(applied)
        bound = 1e3 * group["lr"]
        st["h"].clamp_(-bound, bound)


class IDBDDyadic(IDBDAdam):
    """IDBDAdam with the APPLIED exponent quantized to int8 q (shift-class
    alpha = lr * 2^q); q follows beta with +-hysteresis so it never flaps."""

    def __init__(self, params, hysteresis=0.25, **kw):
        super().__init__(params, **kw)
        for group in self.param_groups:
            group["hysteresis"] = hysteresis

    def _init_state(self, p, st):
        super()._init_state(p, st)
        st["q"] = torch.zeros(p.shape, dtype=torch.int8, device=p.device)

    def _coerce_state(self):
        for st in self.state.values():
            q = st.get("q")
            if q is not None and q.dtype != torch.int8:
                st["q"] = q.to(torch.int8)

    def _alpha(self, group, st):
        return group["lr"] * torch.exp2(st["q"].float())

    def _update(self, group, p, st):
        super()._update(group, p, st)
        hyst = group["hysteresis"]
        q = st["q"].float()
        up = st["beta"] > q + 0.5 + hyst
        dn = st["beta"] < q - 0.5 - hyst
        if bool(up.any()) or bool(dn.any()):
            q = q + up.float() - dn.float()
            st["q"] = q.clamp(group["beta_min"], group["beta_max"]) \
                       .to(torch.int8)


class IDBDSGD(_IDBDBase):
    """Classic Sutton-1992 IDBD in gradient form, Autostep-normalized, with
    a dyadic h-leak (half-life ~700 steps) bounding the trace across the
    substrate's zero-grad droughts."""

    def __init__(self, params, lr=1e-3, weight_decay=0.01, theta_meta=0.01,
                 beta_min=-6.0, beta_max=2.0, norm_decay=1.0 - 2.0 ** -10,
                 h_leak_shift=10):
        super().__init__(params, dict(
            lr=lr, weight_decay=weight_decay, theta_meta=theta_meta,
            beta_min=beta_min, beta_max=beta_max, norm_decay=norm_decay,
            h_leak_shift=h_leak_shift))

    def _init_state(self, p, st):
        st["h"] = torch.zeros_like(p)
        st["beta"] = torch.zeros_like(p)
        st["N"] = torch.full_like(p, 1e-12)

    def _update(self, group, p, st):
        g = p.grad
        if group["theta_meta"] > 0:
            raw = (-g) * st["h"]
            st["N"] = torch.maximum(raw.abs(), group["norm_decay"] * st["N"])
            st["beta"].add_(group["theta_meta"] * raw
                            / st["N"].clamp_min(1e-12)) \
                      .clamp_(group["beta_min"], group["beta_max"])
        alpha = group["lr"] * torch.exp2(st["beta"])
        applied = -alpha * g
        if group["weight_decay"]:
            p.mul_(1 - alpha * group["weight_decay"])
        p.add_(applied)
        st["h"].mul_((1 - alpha * g * g).clamp_min(0)).add_(applied)
        st["h"].sub_(st["h"] * 2.0 ** -group["h_leak_shift"])


def build_life_opt(params, cfg):
    """One construction seam for every E6 arm; 'adamw' is E5 verbatim."""
    kind = cfg.get("optimizer", "adamw")
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    common = dict(lr=cfg["lr"], weight_decay=0.01,
                  theta_meta=cfg.get("theta_meta", 0.01),
                  beta_min=cfg.get("idbd_beta_min", -6.0),
                  beta_max=cfg.get("idbd_beta_max", 2.0))
    if kind == "idbd-adam":
        return IDBDAdam(params, lambda_h=cfg.get("idbd_lambda_h", 1.0),
                        **common)
    if kind == "idbd-dyadic":
        return IDBDDyadic(params, hysteresis=cfg.get("idbd_hyst", 0.25),
                          lambda_h=cfg.get("idbd_lambda_h", 1.0), **common)
    if kind == "idbd-sgd":
        return IDBDSGD(params, **common)
    raise ValueError(f"unknown optimizer {kind!r}")


class _AnchorOpt:
    """st4's pull-then-step ordering delivered through the engine's opt seam
    (serve_and_learn calls opt.step() internally — pc_convrnn_charlm ~L666),
    so no frozen-st6 edits: .step() pulls, then really steps."""

    def __init__(self, opt, anchor, model):
        self.opt, self.anchor, self.model = opt, anchor, model

    def zero_grad(self, **kw):
        self.opt.zero_grad(**kw)

    def step(self):
        self.anchor.pull(self.model, self.opt)
        self.opt.step()

    @property
    def param_groups(self):
        return self.opt.param_groups

    @property
    def state(self):
        return self.opt.state

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)
