"""tikitorch.fabric.thread_link — in-process ThreadLink backend (CPU simulation / CI).

Each pipeline stage runs in its own thread; adjacent stages exchange messages over
blocking queues. This is the reference backend: it exercises the ``NeighborLink``
interface under *real concurrency* (so overlap and staleness are genuine, not modelled)
while staying laptop-runnable and deterministic enough for bit-exact tests.

Topology — for a chain of S stages there is, into each stage l, one forward queue
(activations from l-1, or the host feeder for l=0) and one backward queue (errors from
l+1, or the host feeder for l=S-1):

    host --fq[0]--> [0] --fq[1]--> [1] --fq[2]--> ... --fq[S-1]--> [S-1]
                    [0] <--bq[0]-- [1] <--bq[1]-- ... <--bq[S-2]-- [S-1] <--bq[S-1]-- host
"""
from __future__ import annotations

import queue
import time

import numpy as np

from .link import NeighborLink


class ThreadLink(NeighborLink):
    """A single stage's endpoint, backed by Python queues shared with its neighbours."""

    def __init__(self, stage, num_stages, fq_in, fq_out, bq_in, bq_out):
        super().__init__(stage, num_stages)
        self._fq_in = fq_in     # activations into THIS stage (from l-1 / host)
        self._fq_out = fq_out   # activations out to l+1            (None on last)
        self._bq_in = bq_in     # errors into THIS stage (from l+1 / host)
        self._bq_out = bq_out   # errors out to l-1                 (None on first)

    # forward dataflow ----------------------------------------------------------
    def send_forward(self, tag, act):
        if self._fq_out is None:                       # last stage: nothing downstream
            return
        self._account_down(act)
        self._fq_out.put((tag, act))

    def recv_forward(self, tag):
        t0 = time.perf_counter()
        got_tag, act = self._fq_in.get()
        self.stats.wait_s_fwd += time.perf_counter() - t0
        assert got_tag == tag, f"stage {self.stage} fwd order: want {tag} got {got_tag}"
        return act

    # backward dataflow ---------------------------------------------------------
    def send_backward(self, tag, err):
        if self._bq_out is None:                       # first stage: nothing upstream
            return
        self._account_up(err)
        self._bq_out.put((tag, err))

    def recv_backward(self, tag):
        t0 = time.perf_counter()
        got_tag, err = self._bq_in.get()
        self.stats.wait_s_bwd += time.perf_counter() - t0
        assert got_tag == tag, f"stage {self.stage} bwd order: want {tag} got {got_tag}"
        return err

    # host injection at the open ends ------------------------------------------
    def feed_forward(self, tag, act):
        if not self.is_first:
            raise NotImplementedError("feed_forward is only valid on the first stage")
        self._fq_in.put((tag, act))

    def feed_backward(self, tag, err):
        if not self.is_last:
            raise NotImplementedError("feed_backward is only valid on the last stage")
        self._bq_in.put((tag, err))


def make_thread_links(num_stages: int, maxsize: int = 0) -> list[ThreadLink]:
    """Build a wired chain of ``num_stages`` ThreadLinks sharing the right queues.

    ``maxsize`` bounds each queue (0 = unbounded). A small bound (e.g. the
    accumulation window) models finite in-flight buffering and back-pressures a stage
    that races too far ahead of its neighbour — closer to real on-chip FIFO depth.
    """
    # fq[l] feeds INTO stage l (host feeds fq[0]); bq[l] feeds INTO stage l (host feeds bq[S-1]).
    fq = [queue.Queue(maxsize) for _ in range(num_stages)]
    bq = [queue.Queue(maxsize) for _ in range(num_stages)]
    links = []
    for l in range(num_stages):
        fq_in = fq[l]
        fq_out = fq[l + 1] if l + 1 < num_stages else None
        bq_in = bq[l]
        bq_out = bq[l - 1] if l - 1 >= 0 else None
        links.append(ThreadLink(l, num_stages, fq_in, fq_out, bq_in, bq_out))
    return links
