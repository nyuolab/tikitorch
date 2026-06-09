"""tikitorch.audit — mechanically verify a model is multiplier-free (closed).

The TikiTorch claim is that a whole model — forward *and* backward — maps to
comparisons, additions, and bit-shifts, with **zero hardware multipliers** (zero
DSP blocks on FPGA). :func:`assert_closed` makes that claim *checkable*: it walks
the module tree and raises if it finds any standard ``torch.nn`` layer that
contains a data-dependent multiply or divide (``nn.Linear``, ``nn.Conv2d``,
``nn.LayerNorm``, ``nn.MultiheadAttention``, the cuDNN RNNs, ...).

This is a reproducible artifact behind the paper's headline: paste a model,
call ``assert_closed(model)``, and either it passes or it names every
multiply-bearing standard module to fix.

**Scope (necessary, not sufficient).** This is a *structural* check: it scans for
registered standard ``torch.nn`` modules that contain a data-dependent multiply
or divide. It does NOT trace the forward graph, so it cannot catch a raw ``*`` /
``/`` / ``torch.exp`` / ``@`` / ``einsum`` written directly in a custom module's
``forward`` (a TikiTorch tropical layer must be closed by construction and by its
own unit tests). The denylist covers the common offenders — Linear/Conv/Norm,
the cuDNN RNNs, ``MultiheadAttention``, gated/parametric activations
(``GLU``/``PReLU``), ``EmbeddingBag`` (mean/per-sample-weights), and LRN — but a
clean ``assert_closed`` is a *necessary* condition for zero-DSP, not a proof of
it. For the sufficient, graph-level check use :func:`assert_closed_runtime`, which
traces a real forward pass (via ``TorchDispatchMode``) and flags every actual
multiply/divide ATen op — catching a raw ``*`` / ``@`` / ``torch.exp`` in a custom
``forward`` that the structural scan cannot see. It correctly treats power-of-two
multiplies/divides (``x * 2**k`` / ``x / 8``) as bit-shifts, not multipliers, so
it passes the tropical models and catches genuine DSP ops.

Usage::

    import tikitorch as tt
    from tikitorch.audit import assert_closed, find_closure_violations

    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    assert_closed(cnn)                       # passes — fully multiplier-free

    # A model with a standard nn.Linear head is a *declared* DSP boundary:
    cnn2 = tt.models.TropicalCNN(head_type="linear")
    assert_closed(cnn2, allow={"head"})      # explicitly allow that one module

Note: ``nn.Embedding`` is a pure gather (no multiply) and is NOT flagged.
Dropout / Identity / containers are inert and not flagged. The soft (``tau>0``)
tropical relaxations use ``logsumexp`` (exp) and are training-time aids, not the
deploy path — they are layer *modules* in TikiTorch (not ``torch.nn`` types) so
they pass here; harden with ``tau -> 0`` for the closed deploy graph.
"""
import torch
import torch.nn as nn

__all__ = ["assert_closed", "find_closure_violations", "MULTIPLY_MODULES",
           "find_runtime_multiplies", "assert_closed_runtime",
           "find_training_multiplies", "assert_training_closed", "closure_report",
           "assert_no_multiply", "closed", "ClosureViolation"]

# Standard torch modules that contain a data-dependent hardware multiply/divide.
MULTIPLY_MODULES = tuple(
    m for m in (
        getattr(nn, "Linear", None), getattr(nn, "LazyLinear", None),
        getattr(nn, "Bilinear", None),
        getattr(nn, "Conv1d", None), getattr(nn, "Conv2d", None), getattr(nn, "Conv3d", None),
        getattr(nn, "ConvTranspose1d", None), getattr(nn, "ConvTranspose2d", None),
        getattr(nn, "ConvTranspose3d", None),
        getattr(nn, "LazyConv1d", None), getattr(nn, "LazyConv2d", None), getattr(nn, "LazyConv3d", None),
        getattr(nn, "LayerNorm", None), getattr(nn, "RMSNorm", None),
        getattr(nn, "BatchNorm1d", None), getattr(nn, "BatchNorm2d", None),
        getattr(nn, "BatchNorm3d", None), getattr(nn, "GroupNorm", None),
        getattr(nn, "InstanceNorm1d", None), getattr(nn, "InstanceNorm2d", None),
        getattr(nn, "InstanceNorm3d", None),
        getattr(nn, "LazyBatchNorm1d", None), getattr(nn, "LazyBatchNorm2d", None),
        getattr(nn, "LazyBatchNorm3d", None),
        getattr(nn, "LazyInstanceNorm1d", None), getattr(nn, "LazyInstanceNorm2d", None),
        getattr(nn, "LazyInstanceNorm3d", None),
        getattr(nn, "LocalResponseNorm", None), getattr(nn, "CrossMapLRN2d", None),
        # gated/parametric activations: a * sigmoid(b) / weight * x (data-dep multiply)
        getattr(nn, "GLU", None), getattr(nn, "PReLU", None),
        # embedding-bag mean / per-sample-weights multiply (mode='sum' is a pure gather,
        # but we cannot see the mode from the type, so flag it conservatively)
        getattr(nn, "EmbeddingBag", None),
        getattr(nn, "MultiheadAttention", None),
        getattr(nn, "RNN", None), getattr(nn, "LSTM", None), getattr(nn, "GRU", None),
        getattr(nn, "RNNCell", None), getattr(nn, "LSTMCell", None), getattr(nn, "GRUCell", None),
    )
    if m is not None
)


def find_closure_violations(model: nn.Module):
    """Return ``[(module_path, module_type), ...]`` for every multiply-bearing module.

    Empty list means the model is multiplier-free (closed).
    """
    out = []
    for name, m in model.named_modules():
        if isinstance(m, MULTIPLY_MODULES) and not getattr(m, "_zero_dsp_layer", False):
            out.append((name or "<root>", type(m).__name__))
    return out


# aten ops that ARE a hardware multiply (dense or elementwise) or transcendental.
# Scalar multiplies (``aten::mul.Scalar``) are reported separately: a power-of-2
# scalar is a bit-shift (closed); a general scalar is a multiply.
_DENSE_MUL = ("mm", "bmm", "addmm", "matmul", "linear", "einsum",
              "convolution", "conv1d", "conv2d", "conv3d", "addbmm", "baddbmm")
_TRANSCENDENTAL = ("exp", "log", "pow", "sin", "cos", "tanh", "sigmoid",
                   "erf", "rsqrt", "reciprocal")


def _is_pow2(x) -> bool:
    try:
        ax = abs(float(x))
    except (TypeError, ValueError):
        return False
    if ax == 0.0:
        return True
    import math
    lg = math.log2(ax)
    return abs(lg - round(lg)) < 1e-6


def _all_pow2_tensor(t, max_elems: int = 1 << 16) -> bool:
    """True if every (nonzero) element of ``t`` is a power of two -- i.e. multiplying
    by ``t`` is a per-element bit-SHIFT, not a DSP multiply (closure-clean).

    This is exactly TropicalLayerNorm's pow2 rescale (``x * 2**k``). Only checked
    for the small broadcast operand of an elementwise multiply (large×large is a
    real multiply and is flagged regardless)."""
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        return False
    if t.numel() == 0 or t.numel() > max_elems:
        return False
    a = t.detach().abs()
    nz = a[a > 0]
    if nz.numel() == 0:
        return True
    lg = torch.log2(nz)
    return bool((lg - lg.round()).abs().max() < 1e-5)


def find_runtime_multiplies(model, *example_inputs, run_eval=True, harden_tau=True):
    """Trace a forward pass and return the multiply/transcendental aten ops it runs.

    The static :func:`assert_closed` only inspects *registered modules*, so it
    cannot see a raw ``*`` / ``@`` / ``torch.exp`` written in a custom ``forward``
    (the gap that let a ``rope_relative`` shift hide a real multiply). This runs
    the model on ``example_inputs`` under a ``TorchDispatchMode`` and records the
    actual ATen ops, classifying each as ``dense-multiply`` (mm/conv/...),
    ``elementwise-multiply`` (tensor*tensor), ``elementwise-divide``,
    ``scalar-multiply`` (flagged only if the scalar is NOT a power of two -> a true
    multiply, vs a shift), or ``transcendental`` (exp/log/... -- the soft tau>0
    relaxations; absent once annealed to tau=0).

    By default the model is put in ``eval()`` and annealed to the hard tau=0 deploy
    path first (``harden_tau``), since that is the graph the closure claim is about.
    Returns ``[(category, op_name), ...]`` (empty == multiplier-free at runtime).
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    hits = []

    class _Catcher(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            base = name[len("aten."):].split(".")[0] if name.startswith("aten.") else ""
            def _float_operands():
                # Only FLOAT tensor*tensor is a hardware multiplier. bool*bool
                # (logical AND, e.g. inside torch.isfinite) and int*int are LUT/
                # ALU logic ops -- no DSP -- so they are NOT closure violations.
                ts = [a for a in args if isinstance(a, torch.Tensor)]
                return any(t.is_floating_point() for t in ts)

            def _is_shift_multiply():
                # A multiply is a bit-SHIFT (closed) if any operand is a power of
                # two: a pow2 python scalar, OR a tensor whose elements are all
                # pow2 (TropicalLayerNorm's x*2**k rescale; leaky-relu's x*0.125).
                for a in args:
                    if isinstance(a, torch.Tensor):
                        if _all_pow2_tensor(a):
                            return True
                    elif isinstance(a, (int, float)) and _is_pow2(a):
                        return True
                return False

            if any(d == base for d in _DENSE_MUL):
                hits.append(("dense-multiply", name))
            elif name.startswith("aten.mul") and _float_operands() and not _is_shift_multiply():
                cat = "scalar-multiply" if name.endswith("Scalar") else "elementwise-multiply"
                hits.append((cat, name))
            elif name.startswith("aten.div") and _float_operands():
                # division BY a power of two is a right-shift (closed); the divisor
                # is the second operand.
                divisor = args[1] if len(args) > 1 else None
                shift = (_all_pow2_tensor(divisor) if isinstance(divisor, torch.Tensor)
                         else isinstance(divisor, (int, float)) and _is_pow2(divisor))
                if not shift:
                    hits.append(("elementwise-divide", name))
            elif any(t == base for t in _TRANSCENDENTAL):
                hits.append(("transcendental", name))
            return func(*args, **(kwargs or {}))

    if run_eval and hasattr(model, "eval"):
        model.eval()
    if harden_tau and hasattr(model, "set_tau"):
        model.set_tau(0.0)
    with torch.no_grad():
        with _Catcher():
            model(*example_inputs)
    # de-dup preserving order
    seen, out = set(), []
    for cat, op in hits:
        key = (cat, op)
        if key not in seen:
            seen.add(key)
            out.append((cat, op))
    return out


def assert_closed_runtime(model, *example_inputs, allow_transcendental=False,
                          harden_tau=True):
    """Assert a forward pass runs NO hardware multiply (the runtime closure check).

    Complements :func:`assert_closed` (structural) with a graph-level trace via
    :func:`find_runtime_multiplies`. Raises ``AssertionError`` naming every
    multiply/divide op actually executed. ``allow_transcendental=True`` permits
    the soft ``tau>0`` exp/log relaxations (set it when checking a soft model);
    by default the model is hardened to ``tau=0`` first, where none should appear.
    Returns the model on success.
    """
    hits = find_runtime_multiplies(model, *example_inputs, harden_tau=harden_tau)
    if allow_transcendental:
        hits = [(c, op) for c, op in hits if c != "transcendental"]
    if hits:
        lines = "\n".join(f"    {cat}: {op}" for cat, op in hits)
        raise AssertionError(
            "Model executes hardware multiplies at runtime (closure violated in a "
            "custom forward, which the structural assert_closed cannot see):\n" +
            lines + "\n\nRewrite these as add/compare/shift (or anneal tau->0 for "
            "the transcendental soft-relaxation ops)."
        )
    return model


def _make_catcher(hits):
    """A TorchDispatchMode that appends hardware-multiply ATen ops to ``hits``.

    Same classification as :func:`find_runtime_multiplies` (dense-multiply,
    elementwise-multiply, scalar-multiply [non-pow2], elementwise-divide [non-pow2],
    transcendental); bool/int and power-of-two mul/div are logic/shifts, not flagged.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Catcher(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            base = name[len("aten."):].split(".")[0] if name.startswith("aten.") else ""

            def _float_operands():
                ts = [a for a in args if isinstance(a, torch.Tensor)]
                return any(t.is_floating_point() for t in ts)

            def _is_shift_multiply():
                for a in args:
                    if isinstance(a, torch.Tensor):
                        if _all_pow2_tensor(a):
                            return True
                    elif isinstance(a, (int, float)) and _is_pow2(a):
                        return True
                return False

            if any(d == base for d in _DENSE_MUL):
                hits.append(("dense-multiply", name))
            elif ("mul" in base) and _float_operands() and not _is_shift_multiply():
                # covers mul, mul_, multiply, addcmul, _foreach_mul (Adam's MAC ops).
                cat = "scalar-multiply" if name.endswith("Scalar") else "elementwise-multiply"
                hits.append((cat, name))
            elif ("div" in base) and _float_operands():
                # covers div, divide, addcdiv, true/floor_divide, _foreach_div.
                divisor = args[1] if len(args) > 1 else None
                shift = (_all_pow2_tensor(divisor) if isinstance(divisor, torch.Tensor)
                         else isinstance(divisor, (int, float)) and _is_pow2(divisor))
                if not shift:
                    hits.append(("elementwise-divide", name))
            elif any(t == base for t in _TRANSCENDENTAL) or base in ("sqrt", "rsqrt"):
                hits.append(("transcendental", name))
            return func(*args, **(kwargs or {}))

    return _Catcher()


def find_training_multiplies(model, example_inputs, targets, loss_fn, optimizer=None,
                             harden_tau=True):
    """Trace a WHOLE training step (forward + loss + backward + optimizer) for multiplies.

    This is the **Level-B** check: it verifies that *training itself* is
    multiplier-free, not just inference. A soft ``tau>0`` relaxation (``exp``), a
    non-closure-faithful loss (cross-entropy = ``softmax``+``log``), or an adaptive
    optimizer (Adam's moments / divide / sqrt) all show up here even though the
    deployed forward is closed. The closure-faithful recipe -- hard tropical
    forward, perceptron / L1 loss (±1 gradients), and a sign-only optimizer
    (``SignSGD``/``PureTropicalSGD`` with a power-of-two lr) -- traces clean.

    ``example_inputs`` is a tuple/list of model args; ``targets`` is passed to
    ``loss_fn(predictions, targets)``. If ``optimizer`` is given, ``.step()`` is
    traced too. Returns ``[(category, op_name), ...]`` (empty == multiplier-free).
    """
    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    if hasattr(model, "train"):
        model.train()
    if harden_tau and hasattr(model, "set_tau"):
        model.set_tau(0.0)
    if optimizer is not None:
        optimizer.zero_grad()
    hits = []
    with _make_catcher(hits):
        pred = model(*example_inputs)
        loss = loss_fn(pred, targets)
        loss.backward()
        if optimizer is not None:
            optimizer.step()
    seen, out = set(), []
    for cat, op in hits:
        if (cat, op) not in seen:
            seen.add((cat, op))
            out.append((cat, op))
    return out


def assert_training_closed(model, example_inputs, targets, loss_fn, optimizer=None,
                           harden_tau=True):
    """Assert a full training step is multiplier-free (the Level-B training-closure check).

    See :func:`find_training_multiplies`. Raises ``AssertionError`` naming every
    multiply across forward+loss+backward(+optimizer); returns the model on success.
    """
    hits = find_training_multiplies(model, example_inputs, targets, loss_fn,
                                    optimizer=optimizer, harden_tau=harden_tau)
    if hits:
        lines = "\n".join(f"    {cat}: {op}" for cat, op in hits)
        raise AssertionError(
            "TRAINING step is NOT multiplier-free (Level-B closure violated):\n" + lines +
            "\n\nUse a hard tropical forward (no logsumexp/tau>0), a closure-faithful "
            "loss (tropical_perceptron_loss or L1 -- not cross-entropy), and a sign-only "
            "optimizer (SignSGD/PureTropicalSGD with a power-of-two lr)."
        )
    return model


def closure_report(model, example_inputs=None, targets=None, loss_fn=None,
                   optimizer=None, harden_tau=True):
    """One-call closure audit of a model -- structural + inference (+ training).

    Returns a dict with ``structural`` (module-level :func:`find_closure_violations`),
    ``inference`` (:func:`find_runtime_multiplies`, when ``example_inputs`` given),
    ``training`` (:func:`find_training_multiplies`, when ``targets`` + ``loss_fn``
    given), and booleans ``inference_closed`` / ``training_closed`` /
    ``training_zero_dsp`` (training without the sign-invariant loss-value ``divide``).
    This is the "double-check TikiTorch" one-liner::

        tt.closure_report(model, x)                       # structural + inference
        tt.closure_report(model, x, y, loss_fn, opt)      # + Level-B training

    Reminder: training closure requires the *closed recipe* -- ``dropout=0`` (dropout
    is a train-time mask multiply), hard tropical (``tau=0``; the soft logsumexp path
    is the non-closed trainable relaxation), a closure-faithful loss
    (``tropical_perceptron_loss``/L1, not cross-entropy), a sign-only optimizer
    (``PureTropicalSGD``), and a power-of-two ``d_model`` (so the norm's mean is a
    shift). Inference is closed by construction for the tropical models.
    """
    rep = {"structural": find_closure_violations(model)}
    if example_inputs is not None:
        if not isinstance(example_inputs, (tuple, list)):
            example_inputs = (example_inputs,)
        rep["inference"] = find_runtime_multiplies(model, *example_inputs, harden_tau=harden_tau)
        rep["inference_closed"] = not rep["inference"]
    if targets is not None and loss_fn is not None and example_inputs is not None:
        tr = find_training_multiplies(model, example_inputs, targets, loss_fn,
                                      optimizer=optimizer, harden_tau=harden_tau)
        rep["training"] = tr
        rep["training_closed"] = not tr
        # zero-DSP = no multiply/transcendental; the only tolerated op is the
        # sign-invariant loss-value divide (host-side, not on-chip).
        rep["training_zero_dsp"] = not [h for h in tr if h[0] != "elementwise-divide"]
    return rep


class ClosureViolation(AssertionError):
    """Raised by :class:`assert_no_multiply` / :func:`closed` when a hardware
    multiply / divide / transcendental executes inside the guarded block."""


class assert_no_multiply:
    """Guard that asserts NO hardware multiply runs inside a block of YOUR code.

    The fail-fast, wrap-your-own-loop closure check — ``torch.no_grad()``-shaped::

        import tikitorch as tt
        with tt.assert_no_multiply():
            loss = model.compute_loss(x, y)
            loss.backward()
            opt.step()
        # raises ClosureViolation listing every multiply/divide/transcendental ATen op.

    Unlike :func:`assert_closed` (which only inspects *registered modules*), this traces
    the actual ATen ops via the same :func:`_make_catcher` ``TorchDispatchMode`` the audit
    helpers use, so it catches a raw ``*`` / ``@`` / ``torch.exp`` written in any custom
    ``forward`` or training step. Powers of two (``x * 2**k``, ``x / 8``) are bit-shifts
    and pass; bool/int logic passes. ``allow_transcendental=True`` permits the soft
    ``tau>0`` ``exp``/``log`` relaxations. Tracing has per-op overhead — use it as an
    opt-in guard / test / CI gate, not in the hot training loop.

    Also usable as a decorator via :func:`closed` (preferred) or ``@assert_no_multiply()``.
    """

    def __init__(self, allow_transcendental: bool = False, label: str = None):
        self.allow_transcendental = allow_transcendental
        self.label = label
        self._hits = []
        self._mode = None

    def __enter__(self):
        self._hits = []
        self._mode = _make_catcher(self._hits)
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mode.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return False                       # let the user's own exception propagate
        hits = self._hits
        if self.allow_transcendental:
            hits = [(c, op) for c, op in hits if c != "transcendental"]
        seen, uniq = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        if uniq:
            where = f" in {self.label}" if self.label else ""
            lines = "\n".join(f"    {c}: {op}" for c, op in uniq)
            raise ClosureViolation(
                f"Hardware multiply/divide executed{where} — closure violated:\n" + lines +
                "\n\nRewrite as add/compare/shift (powers of two are bit-shifts and pass); "
                "anneal tau->0 for the soft exp/log relaxations, or pass allow_transcendental=True.")
        return False

    def __call__(self, fn):                    # @assert_no_multiply() decorator form
        import functools

        @functools.wraps(fn)
        def wrapped(*a, **k):
            with assert_no_multiply(self.allow_transcendental, self.label or fn.__name__):
                return fn(*a, **k)
        return wrapped


def closed(fn=None, *, allow_transcendental=False):
    """Decorator asserting the wrapped function/forward runs no hardware multiply::

        @tt.closed
        def train_step(x, y): ...

        @tt.closed(allow_transcendental=True)     # permit soft tau>0 exp/log
        def soft_forward(x): ...

    Thin wrapper over :class:`assert_no_multiply`; raises :class:`ClosureViolation`
    (naming every offending op) if the call executes a hardware multiply/divide.
    """
    if fn is None:
        return lambda f: closed(f, allow_transcendental=allow_transcendental)
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        with assert_no_multiply(allow_transcendental=allow_transcendental, label=fn.__name__):
            return fn(*a, **k)
    return wrapped


def assert_closed(model: nn.Module, allow=()):
    """Assert ``model`` has no multiply-bearing standard module; raise listing any.

    Args:
        model: the module to audit.
        allow: names (module paths, e.g. ``"head"``) or type names
            (e.g. ``"Linear"``) to treat as *declared* DSP boundaries and skip.

    Returns the model on success (so it can be chained). Raises ``AssertionError``
    naming every violation otherwise.
    """
    allow = set(allow)

    def _allowed(path, module):
        if path in allow:
            return True
        # Match allow against the whole MRO so e.g. allow={"Linear"} also skips
        # MultiheadAttention's internal NonDynamicallyQuantizableLinear (a Linear
        # subclass), not just the exact leaf type name.
        return any(base.__name__ in allow for base in type(module).__mro__)

    violations = [
        (path, typ) for path, mod, typ in (
            (p, m, type(m).__name__)
            for p, m in model.named_modules()
            if isinstance(m, MULTIPLY_MODULES) and not getattr(m, "_zero_dsp_layer", False))
        if not _allowed(path, mod)
    ]
    if violations:
        lines = "\n".join(f"    {path}: {typ}" for path, typ in violations)
        raise AssertionError(
            "Model is NOT multiplier-free — closure violated by these standard "
            "torch modules (each contains a hardware multiply/divide):\n" + lines +
            "\n\nReplace with TikiTorch tropical equivalents (TropicalConv2d, "
            "TropicalLayerNorm/TropicalRMSNorm, TropicalDilationModule head, "
            "TropicalAttention, ...), or pass allow={...} to declare a DSP boundary."
        )
    return model
