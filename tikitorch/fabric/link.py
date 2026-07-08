"""tikitorch.fabric.link — the transport-agnostic neighbour-message interface.

A distributed predictive-coding (PC) net is sharded one transformer block per chip.
Unlike data-parallel backprop, PC needs **no global gradient all-reduce**: each chip
updates its own weights from a *local* error. The only communication is between
ADJACENT chips:

    forward  (down the pipeline):  stage l  --activation-->  stage l+1
    backward (up   the pipeline):  stage l  <----error-----  stage l+1

`NeighborLink` is exactly that — one stage's view of its two neighbours. The PC
training loop (see ``runner.PipelineRunner``) is written against THIS interface only;
*how* the bytes move is a swappable backend:

    ThreadLink     in-process threads + queues          (CPU simulation / CI)        [here]
    HostRelayLink  bursted host DMA via the existing AGFI (no rebuild)               [F2]
    PcieP2PLink    direct neighbour-DDR within an instance                           [F2]
    TcpLink        cross-instance neighbour link                                     [farm]
    FabricLink     on-die SLR/GT link                                               [future]

The win is "write once, run anywhere": validate the schedule + staleness logic
bit-exact on ``ThreadLink`` on a laptop, then run the *identical* logic on silicon and
just measure the difference — turning the projected pipeline speedups into measured ones.

Sign conventions / tags:
  * ``tag`` identifies the microbatch; it is what the staleness/accumulation
    bookkeeping keys on (which weight version a message was produced under).
  * messages are opaque ndarrays; the link only moves bytes + counts them.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class LinkStats:
    """Per-stage communication counters (the Gate-C instrumentation)."""

    bytes_down: int = 0          # activation bytes sent to stage+1
    bytes_up: int = 0            # error bytes sent to stage-1
    msgs_down: int = 0
    msgs_up: int = 0
    wait_s_fwd: float = 0.0      # seconds blocked waiting for a forward input
    wait_s_bwd: float = 0.0      # seconds blocked waiting for a backward input

    @property
    def bytes_total(self) -> int:
        return self.bytes_down + self.bytes_up

    def as_dict(self) -> dict:
        return {
            "bytes_down": self.bytes_down,
            "bytes_up": self.bytes_up,
            "bytes_total": self.bytes_total,
            "msgs_down": self.msgs_down,
            "msgs_up": self.msgs_up,
            "wait_s_fwd": round(self.wait_s_fwd, 6),
            "wait_s_bwd": round(self.wait_s_bwd, 6),
        }


class NeighborLink(abc.ABC):
    """One pipeline stage's two-neighbour communication endpoint.

    A stage drives four operations against its neighbours. ``recv_*`` block until the
    message for ``tag`` is available (that blocking *is* the pipeline back-pressure —
    a fast stage naturally waits on a slow neighbour). Backends provide the transport;
    this base provides the byte accounting + the boundary semantics.
    """

    def __init__(self, stage: int, num_stages: int):
        if not (0 <= stage < num_stages):
            raise ValueError(f"stage {stage} out of range [0,{num_stages})")
        self.stage = stage
        self.num_stages = num_stages
        self.stats = LinkStats()

    # -- pipeline-boundary helpers -------------------------------------------------
    @property
    def is_first(self) -> bool:
        """True for the INPUT stage (embed+pos): its forward input comes from the
        host feeder, not a neighbour."""
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        """True for the OUTPUT stage (final-LN + head + loss): it originates the
        backward error itself, rather than receiving one from a neighbour."""
        return self.stage == self.num_stages - 1

    # -- forward dataflow (down the pipeline: l -> l+1) ----------------------------
    @abc.abstractmethod
    def send_forward(self, tag: int, act: np.ndarray) -> None:
        """Send activation ``act`` (microbatch ``tag``) to stage l+1. No-op on the
        last stage (nothing downstream)."""

    @abc.abstractmethod
    def recv_forward(self, tag: int) -> np.ndarray:
        """Receive this stage's forward input (microbatch ``tag``) from stage l-1.
        On the first stage, returns the host-fed input (see ``feed_forward``)."""

    # -- backward dataflow (up the pipeline: l -> l-1) -----------------------------
    @abc.abstractmethod
    def send_backward(self, tag: int, err: np.ndarray) -> None:
        """Send error ``err`` (microbatch ``tag``) to stage l-1. No-op on the first
        stage (nothing upstream needs it)."""

    @abc.abstractmethod
    def recv_backward(self, tag: int) -> np.ndarray:
        """Receive this stage's backward error (microbatch ``tag``) from stage l+1.
        On the last stage, returns the host-fed top error (see ``feed_backward``)."""

    # -- host injection at the open ends of the chain ------------------------------
    def feed_forward(self, tag: int, act: np.ndarray) -> None:
        """Host feeds the INPUT stage its forward input. Default: error unless first."""
        raise NotImplementedError("feed_forward is only valid on the first stage")

    def feed_backward(self, tag: int, err: np.ndarray) -> None:
        """Host feeds the OUTPUT stage its top error (or it computes its own).
        Default: error unless last."""
        raise NotImplementedError("feed_backward is only valid on the last stage")

    # -- accounting ----------------------------------------------------------------
    def _account_down(self, act: np.ndarray) -> None:
        self.stats.bytes_down += int(act.nbytes)
        self.stats.msgs_down += 1

    def _account_up(self, err: np.ndarray) -> None:
        self.stats.bytes_up += int(err.nbytes)
        self.stats.msgs_up += 1
