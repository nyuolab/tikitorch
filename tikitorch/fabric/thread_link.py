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

import math
import queue
import threading
import time


from .link import NeighborLink


class ThreadLink(NeighborLink):
    """A single stage's endpoint, backed by Python queues shared with its neighbours."""

    def __init__(self, stage, num_stages, fq_in, fq_out, bq_in, bq_out,
                 transport=None, msg_bytes=None, double_buffer=False, quantize=None,
                 cancel_event=None):
        super().__init__(stage, num_stages)
        self._fq_in = fq_in     # activations into THIS stage (from l-1 / host)
        self._fq_out = fq_out   # activations out to l+1            (None on last)
        self._bq_in = bq_in     # errors into THIS stage (from l+1 / host)
        self._bq_out = bq_out   # errors out to l-1                 (None on first)
        self._transport = transport   # optional cost.TransportProfile -> models comms time
        self._msg_bytes = msg_bytes   # model a realistic message size (else the array's nbytes)
        # quantize: a callable applied to each neighbour message before it is sent — models the
        # silicon's lossy inter-chip encoding (e.g. the int16 error rescale). None = exact (float).
        self._quantize = quantize
        self._cancel_event = cancel_event or threading.Event()
        self._transport_error = None
        # double-buffering: the stage posts to an outbox and keeps computing while a per-link
        # FIFO transport thread carries the message -> comms overlaps the NEXT compute (rate
        # -> 1/max(compute,comms) instead of 1/(compute+comms)). FIFO = one thread per
        # direction, so message order is preserved (real links don't reorder).
        self._double_buffer = double_buffer
        self._fwd_outbox = queue.Queue() if double_buffer else None
        self._bwd_outbox = queue.Queue() if double_buffer else None
        self._xport = []        # (thread, outbox) pairs to stop at the end

    def _transit(self, msg):
        """Model the transport time of one message. None -> instantaneous (pure overlap study)."""
        if self._transport is not None:
            nb = self._msg_bytes if self._msg_bytes is not None else int(msg.nbytes)
            delay = float(self._transport.time_for(nb))
            if not math.isfinite(delay) or delay < 0:
                raise ValueError("transport delay must be finite and non-negative")
            # Event.wait is an interruptible sleep: if any stage fails, abort()
            # wakes a simulated long transfer immediately instead of making the
            # runner wait for the entire modeled delay in stop_transport().
            if self._cancel_event.wait(delay):
                raise RuntimeError("pipeline transport cancelled")

    def _put(self, dest_q, item):
        while not self._cancel_event.is_set():
            try:
                dest_q.put(item, timeout=0.05)
                return
            except queue.Full:
                continue
        raise RuntimeError("pipeline transport cancelled")

    def _get(self, source_q):
        while not self._cancel_event.is_set():
            try:
                return source_q.get(timeout=0.05)
            except queue.Empty:
                continue
        raise RuntimeError("pipeline transport cancelled")

    def _xport_loop(self, outbox, dest_q):
        try:
            while not self._cancel_event.is_set():
                try:
                    item = outbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:                       # sentinel -> stop
                    return
                tag, msg = item
                self._transit(msg)                     # transit happens OFF the stage thread
                self._put(dest_q, (tag, msg))
        except BaseException as exc:                    # noqa: BLE001 — wake blocked stages
            self._transport_error = exc
            self._cancel_event.set()

    def start_transport(self):
        if not self._double_buffer:
            return
        for outbox, dest in ((self._fwd_outbox, self._fq_out), (self._bwd_outbox, self._bq_out)):
            if dest is not None:
                t = threading.Thread(target=self._xport_loop, args=(outbox, dest), daemon=True)
                t.start()
                self._xport.append((t, outbox))

    def stop_transport(self):
        for _, outbox in self._xport:
            outbox.put(None)                           # drain in FIFO order, then exit
        for t, _ in self._xport:
            t.join()
        self._xport = []

    def abort(self):
        """Cancel this chain and wake queue operations blocked by back-pressure."""
        self._cancel_event.set()

    # forward dataflow ----------------------------------------------------------
    def send_forward(self, tag, act):
        if self._fq_out is None:                       # last stage: nothing downstream
            return
        if self._quantize is not None:
            act = self._quantize(act)
        self._account_down(act)
        if self._double_buffer:
            self._fwd_outbox.put((tag, act))           # returns immediately -> overlap
        else:
            self._transit(act)
            self._put(self._fq_out, (tag, act))

    def recv_forward(self, tag):
        t0 = time.perf_counter()
        got_tag, act = self._get(self._fq_in)
        self.stats.wait_s_fwd += time.perf_counter() - t0
        if got_tag != tag:
            raise RuntimeError(
                f"stage {self.stage} fwd order: want {tag} got {got_tag}"
            )
        return act

    # backward dataflow ---------------------------------------------------------
    def send_backward(self, tag, err):
        if self._bq_out is None:                       # first stage: nothing upstream
            return
        if self._quantize is not None:
            err = self._quantize(err)
        self._account_up(err)
        if self._double_buffer:
            self._bwd_outbox.put((tag, err))           # returns immediately -> overlap
        else:
            self._transit(err)
            self._put(self._bq_out, (tag, err))

    def recv_backward(self, tag):
        t0 = time.perf_counter()
        got_tag, err = self._get(self._bq_in)
        self.stats.wait_s_bwd += time.perf_counter() - t0
        if got_tag != tag:
            raise RuntimeError(
                f"stage {self.stage} bwd order: want {tag} got {got_tag}"
            )
        return err

    # host injection at the open ends ------------------------------------------
    def feed_forward(self, tag, act):
        if not self.is_first:
            raise NotImplementedError("feed_forward is only valid on the first stage")
        self._put(self._fq_in, (tag, act))

    def feed_backward(self, tag, err):
        if not self.is_last:
            raise NotImplementedError("feed_backward is only valid on the last stage")
        self._put(self._bq_in, (tag, err))


def make_thread_links(num_stages: int, maxsize: int = 0, transport=None,
                      msg_bytes=None, double_buffer=False, quantize=None) -> list[ThreadLink]:
    """Build a wired chain of ``num_stages`` ThreadLinks sharing the right queues.

    ``maxsize`` bounds each queue (0 = unbounded). A small bound (e.g. the
    accumulation window) models finite in-flight buffering and back-pressures a stage
    that races too far ahead of its neighbour — closer to real on-chip FIFO depth.
    ``transport`` (a ``cost.TransportProfile``) makes each message take its modelled
    transit time, so the runner's wall-clock includes comms (None = instantaneous).
    """
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1")
    if maxsize < 0:
        raise ValueError("maxsize must be non-negative")
    # fq[l] feeds INTO stage l (host feeds fq[0]); bq[l] feeds INTO stage l (host feeds bq[S-1]).
    fq = [queue.Queue(maxsize) for _ in range(num_stages)]
    bq = [queue.Queue(maxsize) for _ in range(num_stages)]
    cancel_event = threading.Event()
    links = []
    for l in range(num_stages):
        fq_in = fq[l]
        fq_out = fq[l + 1] if l + 1 < num_stages else None
        bq_in = bq[l]
        bq_out = bq[l - 1] if l - 1 >= 0 else None
        links.append(ThreadLink(l, num_stages, fq_in, fq_out, bq_in, bq_out,
                                transport=transport, msg_bytes=msg_bytes,
                                double_buffer=double_buffer, quantize=quantize,
                                cancel_event=cancel_event))
    return links
