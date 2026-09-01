"""Tests for tikitorch.fabric.tcp_link — the TCP neighbour-link backend.

Validates the cross-instance transport against the ThreadLink reference:
  * the wire protocol round-trips arrays exactly (and fails loudly on desync);
  * an all-TCP chain reproduces the ThreadLink run (same seed/schedule -> same
    losses + updated weights, ideally bit-equal);
  * a REAL 2-process split (multiprocessing spawn, one TCP boundary hop) trains and
    matches the single-process reference — the 2-instance pattern;
  * connect-side-first startup works (the retry logic the cross-instance case needs).
"""
import multiprocessing as mp
import os
import struct
import threading
import time

import numpy as np
import pytest

import tikitorch.fabric.tcp_link as tcp_wire
from tikitorch.fabric import PipelineRunner, make_thread_links
from tikitorch.fabric.stages import make_linear_pipeline
from tikitorch.fabric.tcp_link import (
    DIR_BWD,
    DIR_FWD,
    TcpLink,
    WireError,
    decode_msg,
    encode_msg,
    make_tcp_chain,
    make_tcp_links_local,
    run_chain_segment,
)

# collision-resistant port block: 10 ports per pid, distinct slots per test
_BASE = 21000 + (os.getpid() % 4000) * 10


def _synthetic_task(n=16, d_in=5, d_out=3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))
    Wt = rng.standard_normal((d_in, d_out))
    Y = np.tanh(X @ Wt)            # mildly nonlinear target
    inputs = [X[k] for k in range(n)]
    targets = [Y[k] for k in range(n)]
    return inputs, targets


# -- 1. wire protocol -----------------------------------------------------------------
def test_wire_roundtrip():
    rng = np.random.default_rng(0)
    arrays = [
        rng.standard_normal((3, 4)),                             # float64 2-D
        rng.standard_normal(7).astype(np.float32),               # float32 1-D
        rng.integers(-1000, 1000, (2, 3, 4)).astype(np.int32),   # int32 3-D
        rng.integers(-30000, 30000, 5).astype(np.int16),         # int16 (the silicon msg dtype)
        rng.integers(0, 255, (4,)).astype(np.uint8),
        np.float64(3.25) * np.ones(()),                          # 0-d scalar
    ]
    for i, a in enumerate(arrays):
        direction = DIR_FWD if i % 2 == 0 else DIR_BWD
        tag, got_dir, out, end = decode_msg(encode_msg(i, direction, a))
        assert tag == i and got_dir == direction
        assert out.dtype == a.dtype and out.shape == a.shape
        assert np.array_equal(out, a)

    # two messages back-to-back decode sequentially (framed stream)
    buf = encode_msg(1, DIR_FWD, arrays[0]) + encode_msg(2, DIR_BWD, arrays[1])
    tag, _, out, off = decode_msg(buf)
    assert tag == 1 and np.array_equal(out, arrays[0])
    tag, _, out, off = decode_msg(buf, off)
    assert tag == 2 and np.array_equal(out, arrays[1]) and off == len(buf)

    # desync detection: bad magic and corrupted payload both fail loudly
    msg = bytearray(encode_msg(0, DIR_FWD, arrays[0]))
    bad_magic = bytearray(msg)
    bad_magic[0] ^= 0xFF
    with pytest.raises(WireError, match="magic"):
        decode_msg(bad_magic)
    bad_payload = bytearray(msg)
    bad_payload[-8] ^= 0xFF                                      # flip a payload byte
    with pytest.raises(WireError, match="checksum"):
        decode_msg(bad_payload)


def _frame_prefix(shape, payload_len, *, ndim=None, dtype_code=10):
    """Build metadata only; limit failures must occur before absent payload bytes are read."""
    if ndim is None:
        ndim = len(shape)
    return b"".join((
        struct.pack("<IqBBB", tcp_wire.MAGIC, 0, DIR_FWD, dtype_code, ndim),
        struct.pack(f"<{len(shape)}q", *shape),
        struct.pack("<Q", payload_len),
    ))


def test_wire_rejects_untrusted_resource_sizes_before_payload_read():
    # An excessive rank is rejected from the fixed header, before reading its shape.
    header_only = _frame_prefix((), 0, ndim=tcp_wire.MAX_WIRE_NDIM + 1)
    with pytest.raises(WireError, match=r"ndim .* exceeds wire limit"):
        decode_msg(header_only)

    # These buffers intentionally end after their metadata.  Seeing the stated limit
    # errors (rather than "truncated message") proves no payload read was requested.
    too_large = tcp_wire.MAX_WIRE_PAYLOAD_BYTES + 1
    with pytest.raises(WireError, match=r"payload_len .* exceeds wire limit"):
        decode_msg(_frame_prefix((too_large,), too_large))

    # The product check rejects before multiplying beyond the configured bound.
    with pytest.raises(WireError, match=r"tensor shape .* exceeds wire payload limit"):
        decode_msg(_frame_prefix((tcp_wire.MAX_WIRE_AXIS_LENGTH, 2), 0))

    # A zero axis cannot be used to smuggle an otherwise unbounded dimension through.
    with pytest.raises(WireError, match=r"shape axis 1 length .* exceeds wire limit"):
        decode_msg(_frame_prefix((0, tcp_wire.MAX_WIRE_AXIS_LENGTH + 1), 0))


def test_wire_rejects_invalid_shape_and_length_before_payload_read():
    with pytest.raises(WireError, match="negative shape axis"):
        decode_msg(_frame_prefix((-1,), 0))
    with pytest.raises(WireError, match=r"payload_len 3 != prod\(4,\) \* 1"):
        decode_msg(_frame_prefix((4,), 3))


def test_wire_encoder_applies_limits_before_contiguous_copy():
    # broadcast_to creates a constant-size view; a misplaced contiguous conversion
    # would allocate the peer-sized buffer before enforcing the wire bound.
    oversized_view = np.broadcast_to(
        np.array(1, dtype=np.uint8),
        (tcp_wire.MAX_WIRE_PAYLOAD_BYTES + 1,),
    )
    assert not oversized_view.flags.c_contiguous
    with pytest.raises(ValueError, match="array exceeds wire limits"):
        encode_msg(0, DIR_FWD, oversized_view)


def test_split_chain_listener_defaults_to_loopback_and_wildcard_is_explicit(monkeypatch):
    captured = []

    class FakeLink:
        def __init__(self, stage, num_stages, **kwargs):
            captured.append((stage, num_stages, kwargs))

    monkeypatch.setattr(tcp_wire, "TcpLink", FakeLink)
    make_tcp_chain(4, 2, "right")
    assert captured[0][2]["fwd_in"] == {"listen": ("127.0.0.1", 41000)}
    assert captured[0][2]["bwd_out"] == {"listen": ("127.0.0.1", 41001)}

    captured.clear()
    make_tcp_chain(4, 2, "right", listen_host="0.0.0.0")
    assert captured[0][2]["fwd_in"] == {"listen": ("0.0.0.0", 41000)}
    assert captured[0][2]["bwd_out"] == {"listen": ("0.0.0.0", 41001)}


# -- 2. all-TCP chain == ThreadLink ---------------------------------------------------
def test_all_tcp_chain_matches_threadlink():
    inputs, targets = _synthetic_task()
    lr, M = 0.05, len(inputs)
    s_thread = make_linear_pipeline([5, 8, 6, 3], seed=2)
    s_tcp = make_linear_pipeline([5, 8, 6, 3], seed=2)           # identical init
    res_thread = PipelineRunner(s_thread, make_thread_links(len(s_thread)), lr=lr,
                                window=M).run(inputs, targets)
    links = make_tcp_links_local(len(s_tcp), base_port=_BASE, io_timeout_s=20)
    try:
        res_tcp = PipelineRunner(s_tcp, links, lr=lr, window=M).run(inputs, targets)
    finally:
        for lk in links:
            lk.close()
    dmax_loss = float(np.max(np.abs(np.array(res_tcp.losses) - np.array(res_thread.losses))))
    dmax_w = max(float(np.max(np.abs(a.W - b.W))) for a, b in zip(s_thread, s_tcp))
    print(f"max |loss diff| = {dmax_loss:.3e}  max |W diff| = {dmax_w:.3e}")
    assert np.allclose(res_tcp.losses, res_thread.losses, atol=1e-12)
    for a, b in zip(s_thread, s_tcp):
        assert np.allclose(a.W, b.W, atol=1e-12)
    # bytes moved over TCP are accounted through the same LinkStats hooks
    assert res_tcp.link_stats[0]["msgs_down"] == M
    assert res_tcp.link_stats[0]["bytes_down"] == res_thread.link_stats[0]["bytes_down"]
    assert res_tcp.link_stats[-1]["msgs_up"] == M


# -- 3. real 2-process split with one TCP boundary hop --------------------------------
_DIMS = [5, 8, 6, 4, 3]        # 4 stages; boundary=2 -> stages 0..1 proc A, 2..3 proc B
_S = len(_DIMS) - 1


def _segment_proc(role, boundary, base_port, lr, out_q):
    """Child process: host one side of the split chain and train one sweep."""
    inputs, targets = _synthetic_task()
    stages_all = make_linear_pipeline(_DIMS, seed=2)             # deterministic shared init
    lo, hi = (0, boundary) if role == "left" else (boundary, _S)
    local = stages_all[lo:hi]
    links = make_tcp_chain(_S, boundary, role, base_port=base_port,
                           connect_timeout_s=15.0, io_timeout_s=20)
    try:
        losses = run_chain_segment(local, links, inputs, targets, lr=lr,
                                   window=len(inputs))
    finally:
        for lk in links:
            lk.close()
    out_q.put((role, losses, [np.asarray(st.W) for st in local]))


def test_two_process_boundary():
    boundary, lr = 2, 0.05
    inputs, targets = _synthetic_task()
    base_port = _BASE + 6

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_segment_proc, args=(role, boundary, base_port, lr, out_q))
             for role in ("left", "right")]
    for p in procs:
        p.start()
    got = {}
    try:
        for _ in range(2):
            role, losses, weights = out_q.get(timeout=25)
            got[role] = (losses, weights)
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    assert all(p.exitcode == 0 for p in procs), f"child exitcodes {[p.exitcode for p in procs]}"
    assert set(got) == {"left", "right"}

    # single-process ThreadLink reference, same init/seed/schedule
    s_ref = make_linear_pipeline(_DIMS, seed=2)
    res = PipelineRunner(s_ref, make_thread_links(_S), lr=lr,
                         window=len(inputs)).run(inputs, targets)

    # the loss stage lives on the right; the left reports none (it owns no chain end)
    right_losses = got["right"][0]
    assert got["left"][0] == {}
    losses = [right_losses[k] for k in range(len(inputs))]
    dmax = float(np.max(np.abs(np.array(losses) - np.array(res.losses))))
    print(f"2-proc vs ThreadLink: max |loss diff| = {dmax:.3e}")
    assert np.allclose(losses, res.losses, atol=1e-12)
    # BOTH sides' updated weights match the reference
    trained = got["left"][1] + got["right"][1]
    for W_split, st_ref in zip(trained, s_ref):
        assert np.allclose(W_split, st_ref.W, atol=1e-12)


# -- 4. out-of-order start (connect retry) --------------------------------------------
def test_out_of_order_start():
    """The CONNECT side starts (and sends) BEFORE the listener exists — the retry loop
    must carry it across the gap (the real any-order cross-instance startup)."""
    port = _BASE + 8
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    errs = []

    def connect_side():
        try:
            lk = TcpLink(0, 2, fwd_out={"connect": ("127.0.0.1", port)},
                         connect_timeout_s=10.0, io_timeout_s=20)
            lk.send_forward(0, arr)                # retries until the listener appears
            assert lk.stats.bytes_down == arr.nbytes and lk.stats.msgs_down == 1
            lk.close()
        except BaseException as e:                 # noqa: BLE001 — surfaced below
            errs.append(e)

    t = threading.Thread(target=connect_side)
    t0 = time.perf_counter()
    t.start()
    time.sleep(1.0)                                # listener is deliberately late
    listener = TcpLink(1, 2, fwd_in={"listen": ("127.0.0.1", port)},
                       connect_timeout_s=10.0, io_timeout_s=20)
    try:
        got = listener.recv_forward(0)
    finally:
        t.join(timeout=15)
        listener.close()
    assert not errs, f"connect side failed: {errs[0]!r}"
    assert np.array_equal(got, arr) and got.dtype == arr.dtype
    assert time.perf_counter() - t0 >= 1.0         # the send really did span the gap


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
