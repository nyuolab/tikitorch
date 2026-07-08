"""tikitorch.fabric.tcp_link — TCP neighbour-link backend (cross-instance / farm).

The chain topology is identical to ``thread_link`` — into each stage l there is one
forward channel (activations from l-1, or the host feeder for l=0) and one backward
channel (errors from l+1, or the host feeder for l=S-1) — but each hop is now EITHER a
local in-process queue OR a TCP socket. That mixed wiring is exactly the 2-instance
case: stages 0..k-1 on box A, k..S-1 on box B, local queues inside each box and ONE
TCP hop per direction across the k-1 | k boundary:

    box A:  host --q--> [0] --q--> ... --q--> [k-1] ==TCP==> [k] --q--> ... [S-1]  :box B
            [0] <--q-- ... <--q-- [k-1] <==TCP== [k] <--q-- ...  <--q-- [S-1] <--q-- host

Each TCP endpoint is one direction of one hop on its own socket; the RIGHT (downstream)
process listens on both boundary ports and the LEFT process connects (so only one box
needs a reachable address). Connects retry for up to ``connect_timeout_s`` so the two
processes can start in any order — the real cross-instance requirement.

Wire protocol — one message = one framed numpy array, all fields LITTLE-ENDIAN:

    offset    size      field
    0         4    u32  MAGIC = 0x54504C31 ("TPL1")  -- re-verified on every message
    4         8    i64  tag (microbatch id)
    12        1    u8   direction: 0 = forward (activation), 1 = backward (error)
    13        1    u8   dtype code (see _DTYPES)
    14        1    u8   ndim
    15        8*ndim i64 shape
    15+8*ndim 8    u64  payload_len (must equal prod(shape) * itemsize)
    ...       payload_len   raw C-order array bytes
    ...       4    u32  crc32(payload)               -- catches corruption / desync

Any magic, direction, length, or checksum mismatch raises ``WireError`` (a desynced TCP
stream fails loudly instead of training on garbage). TCP preserves order per socket, so
the same tag-order assertion as ``ThreadLink`` guards the schedule.

Unlike ``ThreadLink`` there is no transport-cost model / double-buffering here — a TCP
hop has a *real* cost, measured by the existing ``LinkStats`` wait/byte counters.
"""
from __future__ import annotations

import math
import queue
import socket
import struct
import threading
import time
import zlib

import numpy as np

from .link import NeighborLink
from .runner import PipelineRunner

MAGIC = 0x54504C31          # "TPL1"
DIR_FWD, DIR_BWD = 0, 1

_HDR = struct.Struct("<IqBBB")   # magic, tag, direction, dtype code, ndim
_LEN = struct.Struct("<Q")       # payload_len
_CRC = struct.Struct("<I")       # crc32(payload) trailer

# dtype code <-> numpy dtype (fixed-width, endian-free-on-wire: we send native little-endian)
_DTYPES = {0: "<f8", 1: "<f4", 2: "<f2", 3: "<i8", 4: "<i4", 5: "<i2",
           6: "i1", 7: "<u8", 8: "<u4", 9: "<u2", 10: "u1", 11: "?"}
_DTYPE_CODE = {np.dtype(v): k for k, v in _DTYPES.items()}


class WireError(RuntimeError):
    """A framing/transport violation on the neighbour link (desync, truncation, ...)."""


# -- message framing ----------------------------------------------------------------
def encode_msg(tag: int, direction: int, arr: np.ndarray) -> bytes:
    """Frame one (tag, direction, array) message per the wire protocol above."""
    arr = np.asarray(arr)
    if not arr.flags.c_contiguous:      # (not ascontiguousarray: that would promote 0-d to 1-d)
        arr = arr.copy(order="C")
    code = _DTYPE_CODE.get(arr.dtype)
    if code is None:
        raise TypeError(f"dtype {arr.dtype} not supported on the wire")
    payload = arr.tobytes()
    return b"".join((
        _HDR.pack(MAGIC, tag, direction, code, arr.ndim),
        struct.pack(f"<{arr.ndim}q", *arr.shape),
        _LEN.pack(len(payload)),
        payload,
        _CRC.pack(zlib.crc32(payload) & 0xFFFFFFFF),
    ))


def _decode_stream(read) -> tuple[int, int, np.ndarray]:
    """Decode one message from ``read(n) -> exactly-n-bytes buffer`` (socket or bytes)."""
    magic, tag, direction, code, ndim = _HDR.unpack(read(_HDR.size))
    if magic != MAGIC:
        raise WireError(f"bad magic 0x{magic:08x} (stream desync?)")
    if code not in _DTYPES:
        raise WireError(f"unknown dtype code {code}")
    if direction not in (DIR_FWD, DIR_BWD):
        raise WireError(f"bad direction byte {direction}")
    shape = struct.unpack(f"<{ndim}q", read(8 * ndim)) if ndim else ()
    (plen,) = _LEN.unpack(read(_LEN.size))
    dt = np.dtype(_DTYPES[code])
    if any(s < 0 for s in shape) or plen != dt.itemsize * math.prod(shape):
        raise WireError(f"payload_len {plen} != prod{shape} * {dt.itemsize} (desync?)")
    payload = read(plen)
    (crc,) = _CRC.unpack(read(_CRC.size))
    if crc != zlib.crc32(payload) & 0xFFFFFFFF:
        raise WireError("payload checksum mismatch (corruption / desync)")
    a = np.frombuffer(payload, dtype=dt)
    if not a.flags.writeable:                      # bytes-backed -> copy so stages may mutate
        a = a.copy()
    return tag, direction, a.reshape(shape)


def decode_msg(buf, offset: int = 0) -> tuple[int, int, np.ndarray, int]:
    """Decode one message from a bytes-like at ``offset``; returns
    (tag, direction, array, next_offset) so consecutive messages can be walked."""
    mv = memoryview(buf)
    pos = offset

    def read(n):
        nonlocal pos
        end = pos + n
        if end > len(mv):
            raise WireError(f"truncated message ({len(mv) - pos}/{n} bytes left)")
        chunk = mv[pos:end]
        pos = end
        return chunk

    tag, direction, arr = _decode_stream(read)
    return tag, direction, arr, pos


def recv_exact(sock: socket.socket, n: int) -> bytearray:
    """Read exactly ``n`` bytes from ``sock`` (socket recv may return short reads)."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise WireError(f"connection closed mid-message ({got}/{n} bytes)")
        got += r
    return buf


# -- per-direction endpoints ----------------------------------------------------------
class _QueueEndpoint:
    """A LOCAL hop (or the host-feed open end): a plain queue shared with the neighbour."""

    def __init__(self, q):
        self.q = q

    def send(self, tag, arr):
        self.q.put((tag, arr))

    def recv(self):
        return self.q.get()                        # (tag, arr)

    def close(self):
        pass


class _TcpEndpoint:
    """One direction of one REMOTE hop, on its own socket.

    ``spec`` is ``{"listen": (host, port)}`` or ``{"connect": (host, port)}``. A
    listener binds IMMEDIATELY (reserving the port) but accepts lazily on first use;
    a connector dials lazily on first use, retrying for up to ``connect_timeout_s``
    — so both processes can be started in either order.
    """

    def __init__(self, spec, direction, connect_timeout_s=10.0, io_timeout_s=None):
        self.direction = direction
        self._timeout = float(connect_timeout_s)
        self._io_timeout = io_timeout_s
        self._sock = None
        self._listener = None
        self._lock = threading.Lock()
        if "listen" in spec:
            self._addr = tuple(spec["listen"])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(self._addr)
            s.listen(1)
            self._listener = s
        elif "connect" in spec:
            self._addr = tuple(spec["connect"])
        else:
            raise ValueError(f"endpoint spec needs 'listen' or 'connect': {spec!r}")

    def _ensure(self) -> socket.socket:
        with self._lock:
            if self._sock is not None:
                return self._sock
            if self._listener is not None:                     # accept side
                self._listener.settimeout(self._timeout)
                try:
                    conn, _peer = self._listener.accept()
                except socket.timeout:
                    raise WireError(f"accept on {self._addr} timed out "
                                    f"after {self._timeout}s (peer never connected)")
                sock = conn
            else:                                              # connect side, w/ retry
                deadline = time.monotonic() + self._timeout
                while True:
                    try:
                        sock = socket.create_connection(self._addr, timeout=1.0)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise WireError(f"connect to {self._addr} timed out "
                                            f"after {self._timeout}s")
                        time.sleep(0.05)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self._io_timeout)
            self._sock = sock
            return sock

    def send(self, tag, arr):
        self._ensure().sendall(encode_msg(tag, self.direction, arr))

    def recv(self):
        sock = self._ensure()
        tag, direction, arr = _decode_stream(lambda n: recv_exact(sock, n))
        if direction != self.direction:
            raise WireError(f"direction byte {direction} on a "
                            f"{'fwd' if self.direction == DIR_FWD else 'bwd'} socket (cross-wired?)")
        return tag, arr

    def close(self):
        for s in (self._sock, self._listener):
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass
        self._sock = self._listener = None


def _make_endpoint(spec, direction, connect_timeout_s, io_timeout_s):
    """None -> open end; {'listen'|'connect': (host,port)} -> TCP; queue-like -> local."""
    if spec is None:
        return None
    if isinstance(spec, (_QueueEndpoint, _TcpEndpoint)):
        return spec
    if isinstance(spec, dict):
        return _TcpEndpoint(spec, direction, connect_timeout_s, io_timeout_s)
    if hasattr(spec, "put") and hasattr(spec, "get"):
        return _QueueEndpoint(spec)
    raise TypeError(f"endpoint must be None, a queue, or a TCP spec dict: {spec!r}")


class TcpLink(NeighborLink):
    """A single stage's endpoint where either neighbour may be LOCAL (queue) or REMOTE (TCP).

    ``fwd_in/fwd_out/bwd_in/bwd_out`` are each None (open end), a queue-like object
    (shared with an in-process neighbour, as in ``ThreadLink``), or a TCP endpoint spec
    ``{"listen": (host, port)}`` / ``{"connect": (host, port)}``. Runner-facing API is
    identical to ``ThreadLink`` (``start_transport``/``stop_transport`` are no-ops —
    sends are synchronous ``sendall``).
    """

    def __init__(self, stage, num_stages, fwd_in=None, fwd_out=None, bwd_in=None,
                 bwd_out=None, quantize=None, connect_timeout_s=10.0, io_timeout_s=None):
        super().__init__(stage, num_stages)
        self._fwd_in = _make_endpoint(fwd_in, DIR_FWD, connect_timeout_s, io_timeout_s)
        self._fwd_out = _make_endpoint(fwd_out, DIR_FWD, connect_timeout_s, io_timeout_s)
        self._bwd_in = _make_endpoint(bwd_in, DIR_BWD, connect_timeout_s, io_timeout_s)
        self._bwd_out = _make_endpoint(bwd_out, DIR_BWD, connect_timeout_s, io_timeout_s)
        # quantize: callable applied to each neighbour message before send — models/implements
        # the silicon's lossy inter-chip encoding (e.g. the int16 error rescale). None = exact.
        self._quantize = quantize

    # transport lifecycle (API parity with ThreadLink's double-buffer hooks) ------
    def start_transport(self):
        pass

    def stop_transport(self):
        pass

    def close(self):
        """Shut down every socket this link owns (local queues are untouched)."""
        for ep in (self._fwd_in, self._fwd_out, self._bwd_in, self._bwd_out):
            if ep is not None:
                ep.close()

    # forward dataflow ----------------------------------------------------------
    def send_forward(self, tag, act):
        if self._fwd_out is None:                  # last stage: nothing downstream
            return
        if self._quantize is not None:
            act = self._quantize(act)
        act = np.asarray(act)
        self._account_down(act)
        self._fwd_out.send(tag, act)

    def recv_forward(self, tag):
        t0 = time.perf_counter()
        got_tag, act = self._fwd_in.recv()
        self.stats.wait_s_fwd += time.perf_counter() - t0
        assert got_tag == tag, f"stage {self.stage} fwd order: want {tag} got {got_tag}"
        return act

    # backward dataflow ---------------------------------------------------------
    def send_backward(self, tag, err):
        if self._bwd_out is None:                  # first stage: nothing upstream
            return
        if self._quantize is not None:
            err = self._quantize(err)
        err = np.asarray(err)
        self._account_up(err)
        self._bwd_out.send(tag, err)

    def recv_backward(self, tag):
        t0 = time.perf_counter()
        got_tag, err = self._bwd_in.recv()
        self.stats.wait_s_bwd += time.perf_counter() - t0
        assert got_tag == tag, f"stage {self.stage} bwd order: want {tag} got {got_tag}"
        return err

    # host injection at the open ends ------------------------------------------
    def feed_forward(self, tag, act):
        if not self.is_first:
            raise NotImplementedError("feed_forward is only valid on the first stage")
        self._fwd_in.send(tag, np.asarray(act))

    def feed_backward(self, tag, err):
        if not self.is_last:
            raise NotImplementedError("feed_backward is only valid on the last stage")
        self._bwd_in.send(tag, np.asarray(err))


# -- chain builders -------------------------------------------------------------------
def make_tcp_chain(num_stages: int, boundary: int, role: str, base_port: int = 41000,
                   host: str = "127.0.0.1", quantize=None, connect_timeout_s: float = 10.0,
                   io_timeout_s=None) -> list[TcpLink]:
    """Build ONE process's links for a chain split at ``boundary`` with a single TCP hop.

    Call it twice, in two processes (possibly on two machines):
      * ``role="left"``  hosts stages ``[0, boundary)``   — links returned for those stages
      * ``role="right"`` hosts stages ``[boundary, num_stages)``
    Intra-process hops are local queues; the boundary hop is TCP. The RIGHT process
    LISTENS on ``base_port`` (forward) and ``base_port+1`` (backward, bound on all
    interfaces); the LEFT process CONNECTS to ``host`` (the right box's address) — so
    only one box needs a reachable address, and either side may start first
    (connects retry for ``connect_timeout_s``).
    """
    if not 0 < boundary < num_stages:
        raise ValueError(f"boundary must split the chain: 0 < {boundary} < {num_stages}")
    if role not in ("left", "right"):
        raise ValueError(f"role must be 'left' or 'right', got {role!r}")
    kw = dict(quantize=quantize, connect_timeout_s=connect_timeout_s, io_timeout_s=io_timeout_s)
    links = []
    if role == "left":
        # fq[l]/bq[l] feed INTO local stage l (host feeds fq[0]); boundary-1 talks TCP.
        fq = [queue.Queue() for _ in range(boundary)]
        bq = [queue.Queue() for _ in range(boundary)]
        for l in range(boundary):
            links.append(TcpLink(
                l, num_stages,
                fwd_in=fq[l],
                fwd_out=(fq[l + 1] if l + 1 < boundary
                         else {"connect": (host, base_port)}),
                bwd_in=(bq[l] if l < boundary - 1
                        else {"connect": (host, base_port + 1)}),
                bwd_out=(bq[l - 1] if l > 0 else None),
                **kw))
    else:
        idx = range(boundary, num_stages)
        fq = {l: queue.Queue() for l in idx}
        bq = {l: queue.Queue() for l in idx}       # bq[S-1] is the host feed (feed_backward)
        for l in idx:
            links.append(TcpLink(
                l, num_stages,
                fwd_in=({"listen": ("", base_port)} if l == boundary else fq[l]),
                fwd_out=(fq[l + 1] if l + 1 < num_stages else None),
                bwd_in=bq[l],
                bwd_out=({"listen": ("", base_port + 1)} if l == boundary
                         else bq[l - 1]),
                **kw))
    return links


def make_tcp_links_local(num_stages: int, base_port: int = 41000, host: str = "127.0.0.1",
                         quantize=None, connect_timeout_s: float = 10.0,
                         io_timeout_s=None) -> list[TcpLink]:
    """Wire EVERY hop over localhost TCP in one process (the all-TCP test topology).

    Hop l <-> l+1 uses two ports: ``base_port + 2*l`` (forward, stage l+1 listens) and
    ``base_port + 2*l + 1`` (backward, stage l listens) — ``2*(num_stages-1)``
    consecutive ports total. Listeners bind at construction; connects are lazy, so
    building the whole chain up-front never deadlocks. The open ends (host feed into
    stage 0 / error feed into stage S-1) stay local queues, as in ``ThreadLink``.
    """
    kw = dict(quantize=quantize, connect_timeout_s=connect_timeout_s, io_timeout_s=io_timeout_s)
    S = num_stages
    links = []
    for l in range(S):
        links.append(TcpLink(
            l, S,
            fwd_in=(queue.Queue() if l == 0
                    else {"listen": (host, base_port + 2 * (l - 1))}),
            fwd_out=({"connect": (host, base_port + 2 * l)} if l + 1 < S else None),
            bwd_in=(queue.Queue() if l == S - 1
                    else {"listen": (host, base_port + 2 * l + 1)}),
            bwd_out=({"connect": (host, base_port + 2 * (l - 1) + 1)} if l > 0 else None),
            **kw))
    return links


def run_chain_segment(stages, links, inputs, targets, lr: float = 0.1, window: int = 1,
                      schedule: str = "gpipe") -> dict[int, float]:
    """Drive the LOCAL segment of a cross-process pipeline (the stages this process hosts).

    ``PipelineRunner.run()`` assumes it owns the WHOLE chain (feeds stage 0, collects
    the loss from stage S-1); a segment process owns neither end in general, so this
    runs the same per-stage workers over just the local ``stages``/``links`` (built by
    ``make_tcp_chain``). ``inputs`` is only read if the segment hosts stage 0 and
    ``targets`` only if it hosts the last stage — the other side may pass placeholders
    of the right length. Returns ``{tag: loss}`` ({} unless this segment hosts the
    last stage). Worker exceptions are re-raised here (a desynced link fails loudly).
    """
    runner = PipelineRunner(stages, links, lr=lr, window=window, schedule=schedule)
    M = len(inputs)
    worker = runner._worker_1f1b if schedule == "1f1b" else runner._worker_gpipe
    errors: list[tuple[int, BaseException]] = []

    def _run(l):
        try:
            worker(l, inputs, targets, M)
        except BaseException as e:                 # noqa: BLE001 — surfaced below
            errors.append((l, e))

    threads = [threading.Thread(target=_run, args=(l,)) for l in range(len(stages))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        l, first = errors[0]
        raise RuntimeError(f"segment stage {links[l].stage} failed: {first!r}") from first
    return dict(runner._losses)
