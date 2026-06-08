"""tikitorch.optim — multiplier-free optimizers (a la ``torch.optim``).

These optimizers close the learning loop without floating-point multiplies, so
the *update* rule maps to the same comparison/addition/bit-shift hardware as the
tropical forward and backward passes — the property that makes FORGE-style
training FPGA-native (zero DSP blocks).

Optimizers (all ``torch.optim.Optimizer`` subclasses):
    SignSGD          — ``W -= lr * sign(grad)``        (sign-only, momentum-on-sign)
    TropicalSGD      — the semiring view of SignSGD: sign extraction is a
                       comparison, the update is an addition
    PureTropicalSGD  — strictly multiplier-free: integer-counter momentum,
                       power-of-2 (bit-shift) learning rate, LFSR noise
                       injection, and sign-L1 adaptive rates

Idiomatic usage::

    import tikitorch.optim as optim

    opt = optim.PureTropicalSGD(model.parameters(), lr_shift=10, momentum_bits=4)
    loss.backward()
    opt.step()
    opt.zero_grad()

There is intentionally no ``build_optimizer`` factory: TikiTorch is config-free,
so users instantiate optimizers directly the way they would with ``torch.optim``.

The math (sign updates, integer-counter momentum, bit-shift learning rate, LFSR
noise, sign-L1 adaptive scaling) is ported verbatim from the validated research
code; only imports were changed (no ``forge`` dependencies).
"""

import torch
import torch.optim as toptim

__all__ = ["SignSGD", "TropicalSGD", "PureTropicalSGD", "TropicalLion", "TropicalAdam",
           "FrexpMagnitudeSGD", "CountAgreementSGD", "ExponentEMAAdam", "PerLayerStepSGD"]


def _arshift(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Arithmetic right shift by ``bits`` on a signed integer tensor.

    Implemented as floor-division by ``2**bits`` so the truncation floors toward
    -inf for negative values -- exactly the two's-complement arithmetic-shift
    semantics of the FPGA fabric. This truncation is load-bearing: it is the
    reason an integer-momentum EMA collapses to zero on small gradients where a
    float EMA would not, and modeling it is what makes the CPU optimizer a
    faithful predictor of the silicon update (not an optimistic one).
    """
    return torch.div(x, 1 << bits, rounding_mode="floor")


class SignSGD(toptim.Optimizer):
    """SignSGD: weight update uses only the sign of gradients.

    W -= lr * sign(grad)

    The base update (``momentum=0``) is sign-only — no float multiply — matching
    the {-1, 0, +1} arithmetic of the closed forward pass.

    NOTE (closure caveat, hole H7): the optional ``momentum`` path uses a FLOAT
    EMA ``m = beta*m + (1-beta)*grad`` (``buf.mul_(momentum)`` below), which is a
    data-dependent multiply and is therefore **not** zero-DSP. For a strictly
    multiplier-free momentum use :class:`TropicalLion` (sign of an integer
    fixed-point momentum) or :class:`PureTropicalSGD` (integer-counter momentum).
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    p.add_(torch.sign(buf), alpha=-lr)
                else:
                    p.add_(torch.sign(grad), alpha=-lr)

        return loss


class TropicalSGD(toptim.Optimizer):
    """Tropical SGD: weight update using only comparisons and additions.

    W_new = W + (-lr if grad > 0 else +lr if grad < 0 else 0)

    The base update (``momentum=0``) is the tropical-semiring view of SignSGD:
    sign extraction is a comparison and the update is an addition — zero DSP.

    NOTE (closure caveat, hole H7): the optional ``momentum`` path here uses a
    FLOAT EMA ``buf = momentum*buf + (1-momentum)*grad`` (a data-dependent
    multiply), identical to :class:`SignSGD`, so it is **not** zero-DSP. For a
    strictly multiplier-free momentum use :class:`PureTropicalSGD` (integer-counter
    momentum) or :class:`TropicalLion`.
    """

    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    p.add_(torch.sign(buf), alpha=-lr)
                else:
                    p.add_(torch.sign(grad), alpha=-lr)

        return loss


class PureTropicalSGD(toptim.Optimizer):
    """Strictly multiplier-free optimizer for FPGA-native training.

    With the defaults (``adaptive=False``) all operations map to comparisons,
    additions, and bit-shifts — zero DSP blocks. ``momentum_bits`` must be in
    ``[0, 6]`` (int8 counter safety).

    Features (individually toggleable):
      1. Integer counter momentum: bounded leaky integrator of gradient signs.
         Replaces EMA (which needs mul) with increment/decrement/decay-toward-zero.
      2. LFSR noise injection: adds pseudo-random noise before sign extraction
         to smooth tropical hypersurface boundaries and escape dead cells.
      3. Power-of-2 learning rate: lr = 2^{-lr_shift}, applied as bit-shift on FPGA.
      4. Sign-L1 adaptive rates (``adaptive=True``): **EXPERIMENTAL, NOT closed.**
         This path uses ``v.median()`` (a sort), a data-dependent divide
         (``v / v_median``), and ``log2`` — genuine multiplies/transcendentals, so
         it is NOT zero-DSP. Leave ``adaptive=False`` for the multiplier-free loop.

    Args:
        params: model parameters
        lr_shift: int, learning rate = 2^{-lr_shift} (default 10 -> lr~=0.001)
        momentum_bits: int, counter bound = 2^momentum_bits - 1. 0 disables momentum.
        noise_scale: float, LFSR noise magnitude. 0 disables noise injection.
        adaptive: bool, enable sign-L1 adaptive rate scaling.
        adaptive_bits: int, number of rate tiers (buckets) for adaptive scaling.
        weight_decay_shift: int, weight decay = 2^{-shift}. 0 disables decay.
    """

    def __init__(self, params, lr_shift=10, momentum_bits=4, decay_every=8,
                 noise_scale=0.0, adaptive=False, adaptive_bits=3,
                 weight_decay_shift=0):
        # The momentum counter is int8; momentum_bits>6 lets counter.add_ wrap
        # 127->-128 before the clamp (and abs(-128) overflow in decay), flipping
        # the update sign. Cap to keep the counter in safe int8 range.
        if not 0 <= momentum_bits <= 6:
            raise ValueError("momentum_bits must be in [0, 6] (int8 counter safety)")
        lr = 2.0 ** (-lr_shift)
        defaults = dict(
            lr=lr, lr_shift=lr_shift,
            momentum_bits=momentum_bits,
            decay_every=decay_every,
            noise_scale=noise_scale,
            adaptive=adaptive, adaptive_bits=adaptive_bits,
            weight_decay_shift=weight_decay_shift,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            m_bits = group["momentum_bits"]
            decay_every = group["decay_every"]
            noise_scale = group["noise_scale"]
            adaptive = group["adaptive"]
            adaptive_bits = group["adaptive_bits"]
            wd_shift = group["weight_decay_shift"]

            max_counter = (1 << m_bits) - 1 if m_bits > 0 else 0
            # Periodic decay: halve counters every decay_every steps (per-group;
            # recomputed here so groups with different decay_every don't
            # cross-contaminate). On FPGA: arithmetic right-shift, 0 LUTs.
            do_decay = decay_every > 0 and self._step_count % decay_every == 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # --- Weight decay via bit-shift (addition only) ---
                # On FPGA: arithmetic right-shift of fixed-point weight
                # In PyTorch simulation: multiply by 2^{-shift} (exact power-of-2)
                if wd_shift > 0:
                    decay_rate = 2.0 ** (-wd_shift)
                    grad = grad.add(p, alpha=decay_rate)

                # --- LFSR noise injection ---
                # Smooths tropical hypersurface boundaries
                # On FPGA: ~20 LUTs for an LFSR, practically free
                if noise_scale > 0:
                    noise = torch.empty_like(grad).uniform_(-noise_scale, noise_scale)
                    grad = grad + noise

                # --- Adaptive learning rate (sign-L1) ---
                # Track running max absolute gradient, bucket into 2^k tiers
                effective_lr = lr
                if adaptive:
                    state = self.state[p]
                    if "v_max" not in state:
                        state["v_max"] = torch.zeros_like(p)
                    v = state["v_max"]
                    # Tropical max update: v = max(v/2, |grad|)
                    # The v/2 is a bit-shift decay (no multiplier on FPGA)
                    v.copy_(torch.max(v * 0.5, grad.abs()))
                    # Bucket into power-of-2 tiers: tier = floor(log2(v / v_median))
                    # Clamp to [-adaptive_bits, adaptive_bits] range
                    v_median = v.median().clamp(min=1e-8)
                    ratio = v / v_median
                    # log2 via frexp is cheap; on FPGA just read the exponent bits
                    tier = torch.log2(ratio.clamp(min=2.0**(-adaptive_bits)))
                    tier = tier.round().clamp(-adaptive_bits, adaptive_bits)
                    # Scale lr by 2^{-tier}: large grads get smaller lr
                    lr_scale = (2.0 ** (-tier))
                    # Apply per-parameter scaled lr below
                    effective_lr = lr * lr_scale

                # --- Integer counter momentum ---
                if m_bits > 0:
                    state = self.state[p]
                    if "momentum_counter" not in state:
                        state["momentum_counter"] = torch.zeros_like(p, dtype=torch.int8)

                    counter = state["momentum_counter"]

                    # Accumulate gradient sign (comparison + integer add)
                    grad_sign = torch.sign(grad).to(torch.int8)
                    counter.add_(grad_sign)

                    # Periodic decay: arithmetic right-shift (halve toward zero)
                    # On FPGA: just rewire bits, costs 0 LUTs
                    # Decay every N steps so counter can accumulate direction
                    if do_decay:
                        # Arithmetic right shift by 1: halve magnitude, preserve sign
                        # Python // rounds toward -inf, so use explicit formula
                        halved = (counter.abs() >> 1) * counter.sign().to(torch.int8)
                        counter.copy_(halved)

                    # Clamp to bounded range (comparator)
                    counter.clamp_(min=-max_counter, max=max_counter)

                    # Weight update from counter sign
                    update = torch.sign(counter.float())
                else:
                    update = torch.sign(grad)

                # --- Apply update ---
                if adaptive and isinstance(effective_lr, torch.Tensor):
                    # Per-parameter adaptive lr (still power-of-2 per element)
                    p.add_(update * (-effective_lr))
                else:
                    p.add_(update, alpha=-effective_lr)

        return loss


class TropicalLion(toptim.Optimizer):
    """Lion (sign-of-momentum) optimizer — multiplier-free, integer fixed-point.

    .. warning::
       **Measured shortcoming (Phase 3, see FUTURE_TODO.md).** On char-LM at
       D=128/256 with tropical_ce and a tau→0 anneal, TropicalLion did NOT beat
       SignSGD and in fact *hurt* by 0.8–2.0pp, while the SignSGD→AdamW gap it
       targets failed to open (AdamW ≈ SignSGD). Sign-of-momentum appears to lag
       the annealed-to-hard tropical landscape. **Prefer :class:`SignSGD` or
       :class:`PureTropicalSGD`.** TropicalLion is kept as a reference/option; a
       definitive larger-scale (D=512, 4k+ steps, multi-seed) study is a TODO.

    Lion (Chen et al. 2023) takes the SIGN of an interpolated momentum::

        c_t     = beta1 * m_{t-1} + (1 - beta1) * g_t        # interpolate
        theta_t = theta_{t-1} - lr * (sign(c_t) + wd * theta_{t-1})
        m_t     = beta2 * m_{t-1} + (1 - beta2) * g_t        # momentum EMA

    With power-of-2 betas (``beta = 1 - 2^-b``) each EMA reduces to a shift+add,
    since ``beta*m + (1-beta)*g = m + 2^-b * (g - m)``::

        delta  = g - m
        c      = m + (delta >> beta1_bits)
        update = sign(c)                                     # in {-1, 0, +1}
        m      = m + (delta >> beta2_bits)

    so the whole update is comparisons, additions, and bit-shifts — zero DSP,
    matching the tropical forward/backward and SignSGD. It is strictly stronger
    than SignSGD (a real momentum, not momentum-on-sign) and recovers part of the
    SignSGD -> AdamW closure gap *without* AdamW's adaptive second moment (the
    genuinely multiply/divide part, which is NOT recovered here).

    **Silicon fidelity — the load-bearing detail.** The momentum buffer ``m`` is
    simulated in **integer fixed-point with a truncating arithmetic right shift**
    (:func:`_arshift`), NOT a float EMA. On chip ``(g - m) >> b`` floors to zero
    whenever ``|g - m| < 2^b``; a float EMA never truncates, so it would make
    momentum appear to accumulate where on chip it collapses — over-stating the
    lift. Gradients are pre-scaled into a wide integer domain by ``grad_bits`` (a
    left-shift) so the EMA stays alive on the small (+-1..+-128) integer
    gradients the perceptron/silicon path produces. Inspect
    :meth:`momentum_density` after a few steps to confirm momentum has not
    collapsed (it should be well above zero); if it has, lower ``beta2_bits`` or
    raise ``grad_bits``.

    Args:
        params: parameters.
        lr_shift: learning rate = ``2^-lr_shift`` (bit-shift on FPGA). Default 9.
        beta1_bits: interpolation EMA shift, ``beta1 = 1 - 2^-beta1_bits``.
            Default 1 (beta1 = 0.5).
        beta2_bits: momentum EMA shift, ``beta2 = 1 - 2^-beta2_bits``.
            Default 3 (beta2 = 0.875).
        grad_bits: gradient pre-scale, ``g_int = round(grad * 2^grad_bits)``.
            Default 16. Keep it large relative to ``beta2_bits``.
        weight_decay_shift: decoupled weight decay ``2^-shift`` scaled by ``lr``
            (true Lion ordering). 0 disables.
    """

    def __init__(self, params, lr_shift=9, beta1_bits=1, beta2_bits=3,
                 grad_bits=16, weight_decay_shift=0):
        if beta1_bits < 0 or beta2_bits < 0 or grad_bits < 0:
            raise ValueError("bit-shift counts must be non-negative")
        lr = 2.0 ** (-lr_shift)
        defaults = dict(lr=lr, lr_shift=lr_shift, beta1_bits=beta1_bits,
                        beta2_bits=beta2_bits, grad_bits=grad_bits,
                        weight_decay_shift=weight_decay_shift)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1_bits"]
            b2 = group["beta2_bits"]
            gbits = group["grad_bits"]
            wd_shift = group["weight_decay_shift"]
            scale = float(1 << gbits)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Gradient -> integer fixed-point domain (left-shift pre-scale).
                g_int = torch.round(p.grad.detach() * scale).to(torch.int64)

                state = self.state[p]
                if "m_int" not in state:
                    state["m_int"] = torch.zeros_like(g_int)
                m = state["m_int"]

                delta = g_int - m
                # Interpolated momentum -> sign update (truncating arithmetic shift).
                c = m + _arshift(delta, b1)
                update = torch.sign(c.to(p.dtype))          # in {-1, 0, +1}
                # Momentum EMA, in place (truncating arithmetic shift).
                m.add_(_arshift(delta, b2))

                # Decoupled weight decay, scaled by lr (true Lion ordering).
                if wd_shift > 0:
                    p.mul_(1.0 - lr * (2.0 ** (-wd_shift)))
                p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def momentum_density(self) -> float:
        """Fraction of momentum-buffer entries that are non-zero.

        A collapse diagnostic: if this is ~0 the integer EMA has truncated to
        zero (Lion degenerates to SignSGD) and you should lower ``beta2_bits`` or
        raise ``grad_bits``. Returns 0.0 before the first step.
        """
        total = nz = 0
        for group in self.param_groups:
            for p in group["params"]:
                m = self.state.get(p, {}).get("m_int")
                if m is not None:
                    nz += int((m != 0).sum())
                    total += m.numel()
        return (nz / total) if total else 0.0


class TropicalAdam(toptim.Optimizer):
    """Closed adaptive optimizer — the multiplier-free analogue of Adam's per-parameter
    adaptive learning rates (mechanism: sign-streak Rprop, not moment estimates).

    Adam's edge is a *per-parameter* learning rate from running gradient-moment
    estimates -- which need EMAs, a divide, and a sqrt (all multiplies). This
    recovers per-parameter adaptivity using only **gradient sign, comparisons, and
    power-of-two bit-shifts**, so the whole update is zero-DSP (passes
    ``tikitorch.audit.assert_training_closed``):

    Each parameter keeps a power-of-two step ``s_p = 2^-shift``. Per step, let
    ``g = sign(grad)``:
      * if ``g`` agrees with the previous sign for ``streak`` consecutive steps
        (the parameter is moving consistently) -> **double** the step
        (``shift -= 1``) and reset the streak -- accelerate;
      * if ``g`` flips (oscillating near a minimum) -> **halve** the step
        (``shift += 1``) and reset the streak -- decelerate;
      * otherwise hold.
    The update is the sign-gated shifted step ``p -= g * s_p`` -- a comparison plus
    a barrel-shift, no multiplier. Steps are clamped to ``[step_min, step_max]``
    (both powers of two), so ``s_p`` is always an exact power of two and every
    ``*2`` / ``*0.5`` is a bit-shift.

    Args:
        params: iterable of parameters / param groups.
        step_init: initial per-parameter step (a power of two; default 2^-4).
        step_min: smallest step (power of two; default 2^-12).
        step_max: largest step (power of two; default 2^-2).
        streak: consecutive same-sign steps before accelerating (default 2). 1 is
            classic Rprop (accelerate on every agreement); larger is more cautious.

    Lineage: Riedmiller & Braun, *Rprop* (1993), made power-of-two (hence closed);
    the streak gate is the "sign-streak" refinement. Momentum-free by design (the
    step adaptation provides the persistence).
    """

    def __init__(self, params, step_init=2.0 ** -4, step_min=2.0 ** -12,
                 step_max=2.0 ** -2, streak=2):
        if not (0 < step_min <= step_init <= step_max):
            raise ValueError("require 0 < step_min <= step_init <= step_max")
        if streak < 1:
            raise ValueError("streak must be >= 1")
        defaults = dict(step_init=step_init, step_min=step_min, step_max=step_max,
                        streak=streak, lr=step_init)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            step_min = group["step_min"]
            step_max = group["step_max"]
            thr = group["streak"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.sign()                       # {-1,0,+1}  (comparison)
                st = self.state[p]
                if "step" not in st:
                    st["step"] = torch.full_like(p, group["step_init"])
                    st["prev"] = torch.zeros_like(p)
                    st["streak"] = torch.zeros_like(p)
                step, prev, streak = st["step"], st["prev"], st["streak"]
                zeros = torch.zeros_like(streak)

                nz = g != 0
                flip = nz & (prev != 0) & (g != prev)   # sign flipped -> oscillating
                agree = nz & (g == prev)                # same direction
                streak = torch.where(agree, streak + 1.0,
                                     torch.where(flip, zeros, streak))
                accel = streak >= thr                   # consistent for `thr` steps

                step = torch.where(flip, step * 0.5, step)    # halve  (right-shift)
                step = torch.where(accel, step * 2.0, step)   # double (left-shift)
                step = step.clamp(step_min, step_max)         # stays a power of two
                streak = torch.where(accel, zeros, streak)    # reset after accelerating

                p.add_(g * step, alpha=-1.0)            # p -= sign(grad) * 2^-shift
                st["prev"] = g
                st["step"] = step
                st["streak"] = streak

        return loss

    @torch.no_grad()
    def step_density(self) -> float:
        """Mean per-parameter step size across all params (a learning-rate gauge).

        Rises as parameters find consistent directions (accelerate) and falls as
        they oscillate (decelerate); 0.0 before the first step.
        """
        total = acc = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                s = self.state.get(p, {}).get("step")
                if s is not None:
                    acc += float(s.sum()); total += s.numel()
        return (acc / total) if total else 0.0


# =============================================================================
# Closed ADAPTIVE optimizers (P4): recover Adam's per-parameter magnitude scaling
# using only sign / compare / power-of-two shift -- verified zero-DSP by
# tikitorch.audit.assert_training_closed. The magnitude tier is read with
# torch.frexp (exponent = floor(log2|x|)+1 = a leading-bit / priority-encoder on
# FPGA); the integer shift is turned into a power-of-two step by GATHERING from a
# constant table (never `2**tier` / ldexp, which lower to a transcendental `pow`).
# =============================================================================

_POW2_CACHE = {}


def _pow2_table(K: int, device, dtype):
    """Constant ``[2^0, 2^-1, ..., 2^-K]`` table (a ROM on FPGA), cached per device.

    Built from Python floats via a tensor constructor -- no traced ``aten.pow`` /
    multiply -- so indexing ``table[shift]`` is a pure gather (closed).
    """
    key = (K, device, dtype)
    t = _POW2_CACHE.get(key)
    if t is None:
        t = torch.tensor([2.0 ** -k for k in range(K + 1)], device=device, dtype=dtype)
        _POW2_CACHE[key] = t
    return t


class FrexpMagnitudeSGD(toptim.Optimizer):
    """Closed magnitude-scaled SGD: per-parameter power-of-two step from the gradient's
    frexp exponent tier -- the multiplier-free realization of Adam's ``1/sqrt(v)``
    scaling (and the closed replacement for PureTropicalSGD's non-closed ``adaptive``
    path, which used ``median``/divide/``log2``).

    ``shift = base_shift - (frexp_exp(grad) - e_ref)`` clamped; ``step = 2^-shift``
    (gathered from the pow2 table); ``p -= sign(grad) * step``. ``e_ref`` is a
    per-parameter integer EMA of the exponent (self-centering: a gradient larger
    than this param's recent typical magnitude takes a bigger step). Bigger gradient
    -> higher exponent -> smaller shift -> larger step. ``g==0`` -> no update.

    Closed: frexp (exponent read), integer subtract/clamp, ``_arshift`` EMA, table
    gather, sign, ``sign*step`` (sign x pow2 = shift). Zero DSP.
    """

    def __init__(self, params, base_shift=8, tier_span=4, shift_min=2, shift_max=16,
                 ema_bits=3):
        if not (0 <= shift_min <= base_shift <= shift_max):
            raise ValueError("require 0 <= shift_min <= base_shift <= shift_max")
        defaults = dict(base_shift=base_shift, tier_span=tier_span, shift_min=shift_min,
                        shift_max=shift_max, ema_bits=ema_bits, lr=2.0 ** -base_shift)
        super().__init__(params, defaults)
        self._K = shift_max

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            base, span = group["base_shift"], group["tier_span"]
            smin, smax, eb = group["shift_min"], group["shift_max"], group["ema_bits"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                _, e = torch.frexp(g)                       # int exponent; 0 where g==0
                nz = e != 0
                st = self.state[p]
                if "eref" not in st:
                    st["eref"] = e.clone()
                er = st["eref"]
                # integer EMA of the exponent (skip zero-grad params): er += (e-er)>>eb
                er += torch.where(nz, _arshift(e - er, max(eb, 0)), torch.zeros_like(er))
                tier = (e - er).clamp(-span, span)
                shift = (base - tier).clamp(smin, smax).to(torch.long)
                table = _pow2_table(self._K, p.device, p.dtype)
                step = table[shift]
                upd = torch.sign(g) * step                  # sign x pow2 = shift
                p.add_(torch.where(nz, upd, torch.zeros_like(upd)), alpha=-1.0)
                st["eref"] = er
        return loss


class CountAgreementSGD(toptim.Optimizer):
    """Closed, magnitude-aware SGD scaled by gradient-sign AGREEMENT over an
    accumulation window -- the primary candidate to close the closure tax.

    Per window of ``N`` microbatches, count (int32) how many had ``grad>0`` vs
    ``grad<0`` for each parameter. High agreement = consistent gradient = confident
    -> bigger step (and the count, pooled over the window, is a DENSE signal even
    when any single tropical gradient is sparse/argmax-gated). ``step()``:
    ``agree=max(pos,neg)``; ``net=sign(pos-neg)``; ``tier=frexp_exp(agree)`` clamped;
    ``shift=base_shift-tier``; ``p -= net * 2^-shift``.

    Usage (the harness wires this): ``reset_accum()``; per microbatch ``backward()``
    then ``accumulate()`` (reads ``sign(grad)``, zeroes ``grad``); then ``step()``.
    Do NOT divide the loss by ``N`` -- signs are scale-invariant. If ``accumulate``
    was never called it degrades gracefully to a 1-sample SignSGD off ``p.grad``.

    Closed: sign, int compares/add, ``maximum``, frexp, clamp, table gather,
    ``net*step`` (sign x pow2). Zero DSP.
    """

    def __init__(self, params, base_shift=8, max_accel=6, shift_min=2, shift_max=16):
        defaults = dict(base_shift=base_shift, max_accel=max_accel,
                        shift_min=shift_min, shift_max=shift_max, lr=2.0 ** -base_shift)
        super().__init__(params, defaults)
        self._K = shift_max

    @torch.no_grad()
    def reset_accum(self):
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                if "pos" in st:
                    st["pos"].zero_(); st["neg"].zero_()

    @torch.no_grad()
    def accumulate(self):
        """Read this microbatch's gradient signs into the counters, then zero grad."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "pos" not in st:
                    st["pos"] = torch.zeros_like(p, dtype=torch.int32)
                    st["neg"] = torch.zeros_like(p, dtype=torch.int32)
                s = torch.sign(p.grad)
                st["pos"] += (s > 0).to(torch.int32)
                st["neg"] += (s < 0).to(torch.int32)
                p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            base, ma = group["base_shift"], group["max_accel"]
            smin, smax = group["shift_min"], group["shift_max"]
            for p in group["params"]:
                st = self.state[p]
                if "pos" in st and int(st["pos"].sum() + st["neg"].sum()) > 0:
                    pos, neg = st["pos"], st["neg"]
                elif p.grad is not None:               # graceful 1-sample fallback
                    s = torch.sign(p.grad)
                    pos = (s > 0).to(torch.int32); neg = (s < 0).to(torch.int32)
                else:
                    continue
                agree = torch.maximum(pos, neg)
                net = torch.sign((pos - neg).to(p.dtype))
                _, ec = torch.frexp(agree.to(p.dtype))      # tier grows with agreement
                tier = ec.clamp(0, ma)
                shift = (base - tier).clamp(smin, smax).to(torch.long)
                table = _pow2_table(self._K, p.device, p.dtype)
                step = table[shift]
                p.add_(net * step, alpha=-1.0)              # sign x pow2 = shift
                if "pos" in st:
                    st["pos"].zero_(); st["neg"].zero_()
        return loss


class ExponentEMAAdam(toptim.Optimizer):
    """Closed "tropical Adam v2": integer-counter momentum (first moment) + an
    integer EMA of the gradient's frexp exponent (second moment) -- the closest
    multiplier-free analogue of Adam's ``m / sqrt(v)``.

    Second moment ``ve`` is a per-param integer EMA of ``frexp_exp(grad)``; a higher
    ``ve`` (consistently larger gradients) -> larger ``shift`` -> SMALLER step (the
    ``1/sqrt(v)`` analogue). First moment is the int8 sign counter from
    ``PureTropicalSGD`` (sign-accumulate + periodic ``>>1`` decay + clamp).
    ``update = sign(counter) * 2^-shift``.

    The reference exponent self-centers on each param's INITIAL gradient exponent
    (``shift == base_shift`` at step 1); ``shift = base + (ve - e0).clamp(+-span)``,
    so the step only adapts as gradients grow/shrink RELATIVE to their own start --
    a fixed ``base_exp`` cannot know each tensor's gradient scale and starved the
    step (e.g. exponent-3 grads under ``base_exp=-7`` -> a 2^-12 micro-step). Pass an
    explicit ``base_exp`` to override the per-param auto-centering.

    Closed: frexp, ``_arshift`` EMA, int compares/clamp, table gather, the closed
    counter-momentum block, ``sign*step``. Zero DSP. Most hyperparameter-sensitive.
    """

    def __init__(self, params, base_shift=8, base_exp=None, v_bits=3, span=4,
                 momentum_bits=4, decay_every=8, shift_min=2, shift_max=16):
        if not 0 <= momentum_bits <= 6:
            raise ValueError("momentum_bits must be in [0, 6] (int8 counter safety)")
        defaults = dict(base_shift=base_shift, base_exp=base_exp, v_bits=v_bits,
                        span=span, momentum_bits=momentum_bits, decay_every=decay_every,
                        shift_min=shift_min, shift_max=shift_max, lr=2.0 ** -base_shift)
        super().__init__(params, defaults)
        self._K = shift_max
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1
        for group in self.param_groups:
            base, be, vb, span = group["base_shift"], group["base_exp"], group["v_bits"], group["span"]
            mbits, decay_every = group["momentum_bits"], group["decay_every"]
            smin, smax = group["shift_min"], group["shift_max"]
            maxc = (1 << mbits) - 1 if mbits > 0 else 0
            do_decay = decay_every > 0 and self._step_count % decay_every == 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                _, e = torch.frexp(g)
                nz = e != 0
                st = self.state[p]
                if "ve" not in st:
                    st["ve"] = e.clone()
                    # self-center on this tensor's own initial exponent unless an
                    # explicit base_exp was supplied (frozen int reference).
                    st["e0"] = (torch.full_like(e, int(be)) if be is not None
                                else e.clone())
                    st["counter"] = torch.zeros_like(p, dtype=torch.int8)
                ve, e0, counter = st["ve"], st["e0"], st["counter"]
                ve += torch.where(nz, _arshift(e - ve, max(vb, 0)), torch.zeros_like(ve))
                tier = (ve - e0).clamp(-span, span)
                shift = (base + tier).clamp(smin, smax).to(torch.long)
                step = _pow2_table(self._K, p.device, p.dtype)[shift]
                # first moment: int8 counter momentum (closed)
                counter += torch.sign(g).to(torch.int8)
                if do_decay and mbits > 0:
                    counter.copy_((counter.abs() >> 1) * counter.sign().to(torch.int8))
                if mbits > 0:
                    counter.clamp_(-maxc, maxc)
                upd = torch.sign(counter.to(p.dtype)) * step
                p.add_(upd, alpha=-1.0)
                st["ve"] = ve
        return loss


class PerLayerStepSGD(toptim.Optimizer):
    """Closed per-LAYER adaptive SGD: one shared power-of-two step per parameter
    TENSOR, adapted by the layer-aggregated gradient direction (Rprop hysteresis on
    a scalar shift). Robust to the sparse tropical gradient -- the per-tensor
    aggregate is dense even when individual params rarely win an argmax.

    Per tensor: ``net=sign(grad).sum()``, ``dir=sign(net)``; a sign flip vs the last
    step -> ``shift+=1`` (slower), ``streak`` consecutive agreements -> ``shift-=1``
    (faster), clamped. Update keeps per-element direction with the shared scale:
    ``p -= sign(grad) * 2^-shift``.

    Closed: sign/sum/compare, int shift +-1, table gather, ``sign*step``. Zero DSP.
    """

    def __init__(self, params, base_shift=8, streak=2, shift_min=2, shift_max=16):
        if streak < 1:
            raise ValueError("streak must be >= 1")
        defaults = dict(base_shift=base_shift, streak=streak, shift_min=shift_min,
                        shift_max=shift_max, lr=2.0 ** -base_shift)
        super().__init__(params, defaults)
        self._K = shift_max

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            base, thr = group["base_shift"], group["streak"]
            smin, smax = group["shift_min"], group["shift_max"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "shift" not in st:
                    st["shift"] = torch.tensor(float(base), device=p.device)
                    st["prev"] = torch.zeros((), device=p.device)
                    st["streak"] = torch.zeros((), device=p.device)
                shift, prev, streak = st["shift"], st["prev"], st["streak"]
                layerdir = torch.sign(torch.sign(g).sum())          # scalar {-1,0,1}
                nz = layerdir != 0
                flip = nz & (prev != 0) & (layerdir != prev)
                agree = nz & (layerdir == prev)
                streak = torch.where(agree, streak + 1.0,
                                     torch.where(flip, torch.zeros_like(streak), streak))
                accel = streak >= thr
                shift = torch.where(flip, shift + 1.0,
                                    torch.where(accel, shift - 1.0, shift)).clamp(smin, smax)
                streak = torch.where(accel, torch.zeros_like(streak), streak)
                table = _pow2_table(self._K, p.device, p.dtype)
                step = table[shift.to(torch.long)]                  # scalar pow2
                p.add_(torch.sign(g) * step, alpha=-1.0)
                st["shift"], st["prev"], st["streak"] = shift, layerdir, streak
        return loss
