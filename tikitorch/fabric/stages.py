"""tikitorch.fabric.stages — a reference Stage for the fabric runtime + demos/tests.

``LinearStage`` is a single per-chip block: ``a_out = act(a_in @ W)``. Its ``backward``
is the local cascade (= backprop through the block): it returns the weight gradient
(to accumulate) and the error w.r.t. its input (to send up to the previous chip). This
is the same local-credit op the F2 PC kernels implement, in a few lines of numpy, so
the schedule/staleness machinery can be validated without silicon.

A stage caches per-tag (because the pipeline keeps several microbatches in flight at
once), and exposes the duck-typed interface ``PipelineRunner`` expects:
    params, forward(tag, a_in), backward(tag, e_out), [last] loss_error(tag, a_out, y).
"""
from __future__ import annotations

import time

import numpy as np


class LinearStage:
    def __init__(self, d_in, d_out, act="relu", seed=0, compute_ms=0.0, last=False):
        rng = np.random.default_rng(seed)
        # small init; the demo task is a tiny synthetic regression
        self.params = [rng.standard_normal((d_in, d_out)).astype(np.float64) * (1.0 / np.sqrt(d_in))]
        self.act = act
        self.last = last
        self.compute_ms = float(compute_ms)   # simulated per-block wall-time (to show overlap)
        self._cache: dict[int, tuple] = {}

    @property
    def W(self):
        return self.params[0]

    def _spin(self):
        if self.compute_ms > 0:
            time.sleep(self.compute_ms / 1e3)   # releases the GIL -> real cross-thread overlap

    def forward(self, tag, a_in):
        self._spin()
        pre = a_in @ self.W
        a_out = np.maximum(pre, 0.0) if self.act == "relu" else pre
        self._cache[tag] = (a_in, pre)
        return a_out

    def backward(self, tag, e_out):
        self._spin()
        a_in, pre = self._cache.pop(tag)
        d = e_out * (pre > 0.0) if self.act == "relu" else e_out   # relu' gate
        grad_W = np.outer(a_in, d)
        e_in = self.W @ d                                          # error w.r.t. a_in -> sent up
        return [grad_W], e_in

    def loss_error(self, tag, a_out, target):
        # MSE per sample (sum over output dims); e_out = dL/da_out for the local backward
        diff = a_out - target
        loss = float(np.dot(diff, diff))
        e_out = 2.0 * diff
        return loss, e_out


def make_linear_pipeline(dims, act_hidden="relu", seed=0, compute_ms=0.0):
    """Build a chain of LinearStages for dims = [d0, d1, ..., dS]; the last is linear
    (logits) and originates the loss. Returns the list of stages."""
    stages = []
    S = len(dims) - 1
    for l in range(S):
        last = l == S - 1
        stages.append(
            LinearStage(
                dims[l], dims[l + 1],
                act=("identity" if last else act_hidden),
                seed=seed + l, compute_ms=compute_ms, last=last,
            )
        )
    return stages
