"""tikitorch.fabric.runner — the distributed-PC pipeline runtime.

``PipelineRunner`` drives a model sharded one block per stage across a ``NeighborLink``
chain. It owns exactly the three things PC's local credit assignment makes possible —
and that backprop's global sync would forbid:

  1. **No global barrier.** Each stage updates its OWN weights from its local error as
     soon as that error arrives from its downstream neighbour. There is no all-reduce
     and no wait for a full backward sweep, so stages overlap (the bubble PC removes).

  2. **A grad-accumulation window W to bound staleness.** Keeping the pipeline full means
     a microbatch's backward lands several steps after its forward; if weights were
     applied in between, the gradient is *stale*. Silicon-measured: staleness tolerates
     only ~1 step (stale-8 ≈ +2.0 bpb, unrecoverable by LR). The fix is to freeze
     weights over an aligned window W and apply once — the runner measures the staleness
     its schedule actually produces, so W can be tuned against a real number, not a hope.

  3. **Instrumentation.** wall-clock, the serial-vs-pipelined overlap speedup, bytes moved
     (from the link), and measured staleness — the Gate-C numbers that turn a *projected*
     pipeline speedup into a measured one.

A "stage" is any object exposing (duck-typed):
    params: list[np.ndarray]                       # weights, mutated in place on apply
    forward(tag:int, a_in) -> a_out                # caches per-tag what backward needs
    backward(tag:int, e_out) -> (grads, e_in)      # local grad list + error to send up
    loss_error(tag:int, a_out, target) -> (loss, e_out)   # LAST stage only (originates error)

The schedule here is all-forward-then-all-backward (GPipe-like) under real threads — it
overlaps across stages (the point) and is simple enough to test bit-exact. True 1F1B on
hardware has depth-proportional staleness with the same W >= 2*(L-1) fix; the staleness
the runner *reports* is the genuine version gap of whatever schedule ran.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunResult:
    losses: list[float]                       # final loss per microbatch (from last stage)
    wall_s: float                             # measured wall-clock of the pipelined run
    serial_wall_s: float | None = None        # measured wall-clock of the serial baseline
    max_staleness: int = 0                    # worst (version_at_bwd - version_at_fwd) over all stages/tags
    mean_staleness: float = 0.0
    per_stage_staleness: list[int] = field(default_factory=list)
    link_stats: list[dict] = field(default_factory=list)
    num_stages: int = 0
    num_microbatches: int = 0
    window: int = 1
    schedule: str = "gpipe"

    @property
    def overlap_speedup(self) -> float | None:
        if self.serial_wall_s is None or self.wall_s <= 0:
            return None
        return self.serial_wall_s / self.wall_s

    @property
    def steps_per_s(self) -> float:
        return self.num_microbatches / self.wall_s if self.wall_s > 0 else float("nan")

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.losses)) if self.losses else float("nan")

    def summary(self) -> str:
        sp = self.overlap_speedup
        return (
            f"stages={self.num_stages} microbatches={self.num_microbatches} W={self.window} | "
            f"wall={self.wall_s*1e3:.1f}ms steps/s={self.steps_per_s:.1f} "
            + (f"overlap={sp:.2f}x " if sp else "")
            + f"| staleness max={self.max_staleness} mean={self.mean_staleness:.2f} "
            f"| mean_loss={self.mean_loss:.4f}"
        )


class PipelineRunner:
    """Run a sharded model over a NeighborLink chain with windowed local updates."""

    def __init__(self, stages, links, lr: float = 0.1, window: int = 1,
                 schedule: str = "gpipe"):
        if len(stages) != len(links):
            raise ValueError("stages and links must be the same length")
        if schedule not in ("gpipe", "1f1b"):
            raise ValueError("schedule must be 'gpipe' or '1f1b'")
        self.stages = stages
        self.links = links
        self.lr = float(lr)
        self.window = max(1, int(window))
        self.schedule = schedule
        self.S = len(stages)
        # per-stage weight version + per-(stage,tag) version snapshots for staleness
        self._version = [0] * self.S
        self._ver_at_fwd = [dict() for _ in range(self.S)]
        self._ver_at_bwd = [dict() for _ in range(self.S)]
        self._losses: dict[int, float] = {}
        self._loss_lock = threading.Lock()

    # -- per-stage primitives (shared by both schedules) ---------------------------
    def _do_forward(self, l, k, inputs):
        stage, link = self.stages[l], self.links[l]
        a_in = inputs[k] if link.is_first else link.recv_forward(k)
        self._ver_at_fwd[l][k] = self._version[l]
        a_out = stage.forward(k, a_in)
        link.send_forward(k, a_out)
        if link.is_last:                          # stash output for the loss/backward step
            stage._last_out = getattr(stage, "_last_out", {})
            stage._last_out[k] = a_out

    def _do_backward(self, l, k, targets, acc):
        stage, link = self.stages[l], self.links[l]
        if link.is_last:
            loss, e_out = stage.loss_error(k, stage._last_out[k], targets[k])
            with self._loss_lock:
                self._losses[k] = float(loss)
        else:
            e_out = link.recv_backward(k)
        self._ver_at_bwd[l][k] = self._version[l]
        grads, e_in = stage.backward(k, e_out)
        link.send_backward(k, e_in)
        for i in range(len(acc)):
            acc[i] += grads[i]

    def _apply(self, l, acc, n):                  # apply the windowed mean grad + bump version
        for i, p in enumerate(self.stages[l].params):
            p -= self.lr * acc[i] / n
            acc[i][...] = 0.0
        self._version[l] += 1

    # -- GPipe schedule: all forwards, then all backwards (a fwd->bwd drain remains) ----
    def _worker_gpipe(self, l, inputs, targets, M):
        for k in range(M):
            self._do_forward(l, k, inputs)
        acc = [np.zeros_like(p) for p in self.stages[l].params]
        n = 0
        for k in range(M):
            self._do_backward(l, k, targets, acc)
            n += 1
            if n == self.window or k == M - 1:
                self._apply(l, acc, n); n = 0

    # -- 1F1B schedule: warmup forwards, then interleave 1F+1B, then drain backwards ----
    # Stage l keeps (S-l) microbatches in flight, so its backward lands ~(S-1-l) applies
    # after its forward -> DEPTH-PROPORTIONAL staleness (deepest stage worst, loss stage 0),
    # the real hardware mechanism. W >= the in-flight depth bounds it back to ~1.
    def _worker_1f1b(self, l, inputs, targets, M):
        warmup = min(self.S - 1 - l, M)
        acc = [np.zeros_like(p) for p in self.stages[l].params]
        n = 0
        f = b = 0
        for _ in range(warmup):                   # warmup: fill the pipe ahead of us
            self._do_forward(l, f, inputs); f += 1
        while b < M:                              # steady (1F+1B) then cooldown (B only)
            if f < M:
                self._do_forward(l, f, inputs); f += 1
            self._do_backward(l, b, targets, acc); b += 1
            n += 1
            if n == self.window or b == M:
                self._apply(l, acc, n); n = 0

    # -- run the whole pipeline ----------------------------------------------------
    def run(self, inputs, targets) -> RunResult:
        """inputs[k], targets[k] are the per-microbatch input / target. Returns a
        RunResult with the measured losses, overlap, staleness, and link stats."""
        M = len(inputs)
        # feed the input stage up-front (queues buffer the forward wavefront)
        for k in range(M):
            self.links[0].feed_forward(k, inputs[k])
        worker = self._worker_1f1b if self.schedule == "1f1b" else self._worker_gpipe
        threads = [
            threading.Thread(target=worker, args=(l, inputs, targets, M))
            for l in range(self.S)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0

        # staleness = how many weight-applies landed between a microbatch's fwd and bwd
        per_stage = []
        all_gaps = []
        for l in range(self.S):
            gaps = [self._ver_at_bwd[l][k] - self._ver_at_fwd[l][k] for k in range(M)]
            per_stage.append(max(gaps) if gaps else 0)
            all_gaps.extend(gaps)
        return RunResult(
            losses=[self._losses[k] for k in range(M)],
            wall_s=wall,
            max_staleness=max(all_gaps) if all_gaps else 0,
            mean_staleness=float(np.mean(all_gaps)) if all_gaps else 0.0,
            per_stage_staleness=per_stage,
            link_stats=[lk.stats.as_dict() for lk in self.links],
            num_stages=self.S,
            num_microbatches=M,
            window=self.window,
            schedule=self.schedule,
        )


def run_serial(stages, inputs, targets, lr: float = 0.1) -> tuple[list[float], float]:
    """Synchronous full-sweep baseline (= PC cascade with no staleness): all forwards,
    all backwards, ONE accumulated apply. This is what W = M converges to, and the
    reference the pipelined run is checked against. Returns (losses, wall_s)."""
    S, M = len(stages), len(inputs)
    t0 = time.perf_counter()
    losses = [0.0] * M
    acc = [[np.zeros_like(p) for p in st.params] for st in stages]
    for k in range(M):
        a = inputs[k]
        for l in range(S):                               # forward
            a = stages[l].forward(k, a)
        loss, e = stages[-1].loss_error(k, a, targets[k])
        losses[k] = float(loss)
        for l in range(S - 1, -1, -1):                   # backward
            grads, e = stages[l].backward(k, e)
            for i in range(len(grads)):
                acc[l][i] += grads[i]
    for l in range(S):                                   # one apply
        for i, p in enumerate(stages[l].params):
            p -= lr * acc[l][i] / M
    return losses, time.perf_counter() - t0
