"""A thin Trainer for TikiTorch's soft-to-hard reference workflow.

The helper anneals a model's tropical relaxation temperature, executes an ordinary
PyTorch loss/backward/optimizer loop, and hardens the model to ``tau=0`` afterward.
At the hard endpoint it can run :func:`tikitorch.audit.assert_closed` and
:func:`~tikitorch.audit.assert_closed_runtime` on the observed input path.

Those checks cover registered modules and one executed forward path. They do not
audit the loss, backward, or optimizer used during :meth:`Trainer.fit`, and they
do not establish properties of a synthesized hardware build. Use
:func:`tikitorch.audit.assert_training_closed` for a concrete full-step software
audit and bind any physical claim to build reports and a known-answer test.

TikiTorch is config-free in the same limited sense as ``torch``: instantiate the
optimizer yourself and pass it in. The Trainer owns the temperature schedule,
the final hardening step, optional forward-path audits, and plain progress output.

    import tikitorch as tt
    import tikitorch.optim as optim

    model = tt.models.TropicalMLP(d_model=64, n_layers=2,
                                  input_dim=8, num_classes=3, dropout=0.0)
    opt = optim.SignSGD(model.parameters(), lr=2e-2)
    trainer = tt.Trainer(model, opt, steps=400)
    trainer.fit((X, y))                  # trains, hardens, audits an observed path
    print(trainer.evaluate((Xte, yte)))  # hard (tau=0) accuracy

The loss follows the same precedence as the rest of the package: call
``model.compute_loss(x, y)`` when the model defines it (the closure-faithful
tropical CE etc.), else a provided ``loss_fn(model(x), y)``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

import torch

from tikitorch.audit import assert_closed, assert_closed_runtime

__all__ = ["Trainer"]

Batch = Tuple[torch.Tensor, torch.Tensor]
DataSource = Union[Batch, Iterable[Batch]]


# ---------------------------------------------------------------------------
# tau scheduling
# ---------------------------------------------------------------------------

def _tau_at(step: int, steps: int, *, schedule: str, tau_max: float,
            tau_min: float, hard_frac: float) -> float:
    """Training-time tau for ``step`` in ``[0, steps)`` under the soft->hard anneal.

    ``tau`` ramps from ``tau_max`` (soft) down to the small floor ``tau_min`` over
    the first ``hard_frac`` of training, then holds at ``tau_min`` for the tail --
    matching the benchmark harness's ``max(1 - step/(0.8*steps), 0.0)`` schedule
    (``hard_frac=0.8``), whose TropicalMLP clamps the floor to ``tau_min=0.01``.

    The floor is deliberately *not* exactly 0: training the tropical landscape at
    fully-hard ``tau=0`` gives sign-only optimizers near-zero / degenerate
    gradients and they collapse to a constant prediction. The truly-hard ``tau=0``
    deploy graph is pinned once, *after* the loop (see :meth:`Trainer.fit`), which
    is the graph the closure gate then checks. ``"linear"`` ramps linearly;
    ``"cosine"`` follows a half-cosine; ``"constant"`` holds ``tau_max`` until the
    floor.
    """
    if steps <= 1:
        return tau_min
    frac = step / max(1.0, hard_frac * (steps - 1))   # 0 -> 1 across the anneal
    if frac >= 1.0:
        return tau_min                               # hold at the floor in the tail
    if schedule == "constant":
        return tau_max
    if schedule == "cosine":
        import math
        warm = 0.5 * (1.0 + math.cos(math.pi * frac))   # 1 -> 0
        return tau_min + (tau_max - tau_min) * warm
    if schedule == "linear":
        return tau_min + (tau_max - tau_min) * (1.0 - frac)
    raise ValueError(f"unknown tau_schedule {schedule!r}; "
                     "choose 'linear', 'cosine', or 'constant'")


def _set_tau(model, tau: float) -> bool:
    """Drive the tropical relaxation temperature across ``model`` to ``tau``.

    Replicates the benchmark harness's ``set_tau`` helper: prefer a model-level
    ``set_tau`` (TropicalReasoner / MorphSSM harden their layers through it), else
    walk the submodules (``set_tau`` method, then ``.tau`` / ``.use_soft`` /
    ``use_softmin`` flags). We do **not** use a model-level ``set_temperature``
    (e.g. TropicalMLP's), because it *clamps* ``tau`` to a small ``tau_min`` and so
    never reaches the truly hard ``tau=0`` deploy point -- its per-submodule
    ``TropicalLinearModule.set_tau`` does (it flips ``use_softmin = tau > 0``), so
    the submodule walk is what hardens it. Returns ``True`` if anything tau-aware
    was found, so a non-tropical (pure ``loss_fn``) model doesn't get spurious
    "anneal was a no-op" warnings.
    """
    if hasattr(model, "set_tau") and callable(model.set_tau):
        model.set_tau(tau)
        return True
    found = False
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "set_tau") and callable(m.set_tau):
            m.set_tau(tau)
            found = True
        elif hasattr(m, "set_temperature") and callable(m.set_temperature):
            m.set_temperature(tau)
            found = True
        elif hasattr(m, "tau"):
            m.tau = tau
            found = True
        if hasattr(m, "use_soft"):
            m.use_soft = tau > 0
            found = True
        if hasattr(m, "use_softmin"):
            m.use_softmin = tau > 0
            found = True
    return found


# ---------------------------------------------------------------------------
# data plumbing
# ---------------------------------------------------------------------------

def _is_tensor_pair(data) -> bool:
    return (isinstance(data, (tuple, list)) and len(data) == 2
            and isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor))


def _batch_iter(data: DataSource, *, batch_size: int, steps: int,
                generator: Optional[torch.Generator]) -> Iterator[Batch]:
    """Yield exactly ``steps`` ``(x, y)`` batches from ``data``.

    ``data`` may be a single ``(X, y)`` tensor pair -- in which case we draw random
    minibatches of ``batch_size`` with replacement, exactly as the algorithmic /
    benchmark harnesses do -- or any iterable of ``(x, y)`` batches, which we cycle
    until ``steps`` batches have been produced.
    """
    if _is_tensor_pair(data):
        X, y = data
        n = X.shape[0]
        bs = min(batch_size, n)
        for _ in range(steps):
            idx = torch.randint(0, n, (bs,), generator=generator)
            yield X[idx], y[idx]
        return
    # iterable of batches: cycle it until `steps` batches are produced.
    produced = 0
    while produced < steps:
        any_batch = False
        for batch in data:
            any_batch = True
            yield batch
            produced += 1
            if produced >= steps:
                return
        if not any_batch:
            raise ValueError("data iterable yielded no batches")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """A reference training loop with a soft-to-hard ``tau`` anneal.

    Args:
        model (nn.Module): a TikiTorch model. If it defines
            ``compute_loss(x, y)`` that is used as the loss; otherwise pass
            ``loss_fn``.
        optimizer (torch.optim.Optimizer): an instantiated optimizer (e.g.
            ``optim.SignSGD(...)``). The
            Trainer never builds one for you -- TikiTorch is config-free.
        tau_schedule: ``"linear"`` (default), ``"cosine"``, or ``"constant"`` --
            how the relaxation temperature anneals from ``tau_max`` toward 0.
        steps: number of optimizer steps. Training uses at least ``tau_min``;
            the model is hardened to ``tau=0`` after the final update.
        loss_fn: ``loss_fn(pred, y) -> scalar`` used only when the model has no
            ``compute_loss``.
        batch_size: minibatch size when ``fit`` is given a single ``(X, y)`` pair.
        tau_max: starting anneal temperature.
        tau_min: ending anneal temperature. During training ``tau`` ramps from
            ``tau_max`` to the small floor ``tau_min`` (default 0.01) and holds
            there -- NOT exactly 0, because sign-only optimizers collapse on the
            fully-hard landscape. The exact ``tau=0`` deploy graph is pinned once
            after the loop, which is what the closure gate checks.
        hard_frac: fraction of ``steps`` over which ``tau`` reaches ``tau_min``;
            the remaining tail trains at the floor. Matches the harness's ``0.8``.
        assert_closed_on_deploy: when ``True`` (default), at the hard endpoint run
            :func:`assert_closed` (structural) and, if an example batch is
            available, :func:`assert_closed_runtime` (graph-level). A model with a
            multiply-bearing standard module (e.g. an ``nn.Linear`` head) raises.
            This does not audit the training step or a physical implementation.
        closed_allow: names/types forwarded to :func:`assert_closed` ``allow=`` to
            declare an *intentional* DSP boundary (skips the gate for those).
        log_every: print a progress line every ``log_every`` steps (0 = silent).
        generator: optional ``torch.Generator`` for reproducible minibatching.
    """

    def __init__(self, model, optimizer, *, tau_schedule: str = "linear",
                 steps: int, loss_fn: Optional[Callable] = None,
                 batch_size: int = 128, tau_max: float = 1.0, tau_min: float = 0.01,
                 hard_frac: float = 0.8, assert_closed_on_deploy: bool = True,
                 closed_allow: Iterable[str] = (), log_every: int = 0,
                 generator: Optional[torch.Generator] = None):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if not 0.0 < hard_frac <= 1.0:
            raise ValueError("hard_frac must be in (0, 1]")
        self.model = model
        self.optimizer = optimizer
        self.tau_schedule = tau_schedule
        self.steps = int(steps)
        self.loss_fn = loss_fn
        self.batch_size = int(batch_size)
        self.tau_max = float(tau_max)
        self.tau_min = float(tau_min)
        self.hard_frac = float(hard_frac)
        self.assert_closed_on_deploy = assert_closed_on_deploy
        self.closed_allow = tuple(closed_allow)
        self.log_every = int(log_every)
        self.generator = generator
        # populated by fit(): the per-step loss trace and the last seen batch.
        self.history: list = []
        self._last_batch: Optional[Batch] = None

    # -- loss ---------------------------------------------------------------
    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """compute_loss when the model defines it, else loss_fn(model(x), y)."""
        cl = getattr(self.model, "compute_loss", None)
        if callable(cl):
            return cl(x, y)
        if self.loss_fn is None:
            raise ValueError(
                "model has no compute_loss(x, y); pass loss_fn=... to Trainer")
        return self.loss_fn(self.model(x), y)

    # -- training -----------------------------------------------------------
    def fit(self, data: DataSource) -> "Trainer":
        """Train for ``steps``, then harden and audit the observed forward path.

        ``data`` is either a single ``(X, y)`` tensor pair (random minibatches) or
        an iterable of ``(x, y)`` batches. Returns ``self`` (the trained model is
        left in ``eval()`` at the hard ``tau=0`` endpoint).
        """
        self.history = []
        self.model.train()
        tau_aware = None
        batches = _batch_iter(data, batch_size=self.batch_size, steps=self.steps,
                              generator=self.generator)
        for step, (xb, yb) in enumerate(batches):
            self._last_batch = (xb, yb)
            tau = _tau_at(step, self.steps, schedule=self.tau_schedule,
                          tau_max=self.tau_max, tau_min=self.tau_min,
                          hard_frac=self.hard_frac)
            tau_aware = _set_tau(self.model, tau)
            self.optimizer.zero_grad()
            loss = self._loss(xb, yb)
            loss.backward()
            self.optimizer.step()
            lv = float(loss.detach())
            self.history.append(lv)
            if self.log_every and (step % self.log_every == 0 or step == self.steps - 1):
                print(f"[trainer] step {step + 1:5d}/{self.steps} "
                      f"tau {tau:.3f} loss {lv:.4f}", flush=True)

        # --- hard (tau=0) evaluation/export candidate endpoint ---
        _set_tau(self.model, 0.0)
        self.model.eval()
        if self.assert_closed_on_deploy:
            self._assert_deploy_closed()
        if self.log_every and tau_aware is False:
            print("[trainer] note: model exposed no tau hook (loss_fn path); "
                  "anneal was a no-op", flush=True)
        return self

    def _assert_deploy_closed(self) -> None:
        """Run structural and, when possible, observed-path audits at ``tau=0``.

        :func:`assert_closed` is the structural check (named standard modules);
        when a batch was seen during ``fit`` we additionally trace the real forward
        with :func:`assert_closed_runtime`, which catches a raw multiply hidden in a
        custom ``forward`` that the structural scan cannot see.
        """
        assert_closed(self.model, allow=self.closed_allow)
        if self._last_batch is not None and not self.closed_allow:
            xb, _ = self._last_batch
            # We already drove tau->0 above via _set_tau (which hardens models that
            # only expose set_tau on their submodules, not at the top level), so
            # disable the runtime tracer's own harden_tau -- it would call a missing
            # model.set_tau and silently leave the graph soft.
            example = xb[:1] if xb.shape[0] > 1 else xb
            assert_closed_runtime(self.model, example, harden_tau=False)

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, data: DataSource, *, metric: str = "accuracy") -> dict:
        """Evaluate at the hard ``tau=0`` endpoint.

        Returns ``{"loss": ..., metric: ...}`` averaged over ``data`` (a single
        ``(X, y)`` pair or an iterable of batches). ``metric`` is ``"accuracy"``
        (argmax over the last dim vs integer ``y``) or ``"none"`` (loss only).
        """
        _set_tau(self.model, 0.0)
        self.model.eval()
        batches: Iterable[Batch]
        batches = [data] if _is_tensor_pair(data) else data
        tot_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        for xb, yb in batches:
            n_batches += 1
            try:
                tot_loss += float(self._loss(xb, yb).detach())
            except ValueError:
                pass                                 # loss_fn-less + no compute_loss
            if metric == "accuracy":
                pred = self.model(xb)
                correct += int((pred.argmax(-1) == yb).sum())
                total += int(yb.numel())
        out = {"loss": tot_loss / max(1, n_batches)}
        if metric == "accuracy":
            out["accuracy"] = correct / total if total else 0.0
        return out
