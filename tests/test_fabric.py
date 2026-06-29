"""Tests for tikitorch.fabric — the distributed-PC pipeline runtime.

Validates the load-bearing invariants on the CPU ThreadLink backend (the same logic
that will drive silicon):
  * ThreadLink moves activations down / errors up correctly and counts bytes.
  * PipelineRunner(W=M) == the synchronous serial baseline (no staleness) within fp tol
    -- i.e. the windowed local update reproduces backprop/PC-cascade exactly when the
    window spans the whole sweep. This is the bit-exact gate.
  * Staleness is what the schedule actually produces: W=M -> 0, W=1 -> >0.
  * Repeated windowed training reduces loss (the runtime actually learns).
"""
import numpy as np
import pytest

from tikitorch.fabric import PipelineRunner, make_thread_links, run_serial
from tikitorch.fabric.stages import make_linear_pipeline


def _synthetic_task(n=24, d_in=5, d_out=3, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_in))
    Wt = rng.standard_normal((d_in, d_out))
    Y = np.tanh(X @ Wt)            # mildly nonlinear target
    inputs = [X[k] for k in range(n)]
    targets = [Y[k] for k in range(n)]
    return inputs, targets


def test_thread_link_moves_data_and_counts_bytes():
    inputs, targets = _synthetic_task()
    stages = make_linear_pipeline([5, 8, 6, 3], seed=1)
    links = make_thread_links(len(stages))
    runner = PipelineRunner(stages, links, lr=0.05, window=len(inputs))
    res = runner.run(inputs, targets)
    # every microbatch produced a (finite) loss -> data flowed end-to-end
    assert len(res.losses) == len(inputs)
    assert all(np.isfinite(L) for L in res.losses)
    # interior stages moved activations down AND errors up; bytes counted
    assert res.link_stats[0]["msgs_down"] == len(inputs)          # stage 0 sends down
    assert res.link_stats[0]["bytes_down"] > 0
    assert res.link_stats[-1]["msgs_up"] == len(inputs)           # last stage sends errors up
    assert res.link_stats[-1]["msgs_down"] == 0                   # last stage has no downstream


def test_windowed_full_equals_serial():
    """W = M (one window over the whole sweep) must reproduce the synchronous baseline."""
    inputs, targets = _synthetic_task()
    lr = 0.05
    s_serial = make_linear_pipeline([5, 8, 6, 3], seed=2)
    s_pipe = make_linear_pipeline([5, 8, 6, 3], seed=2)        # identical init
    serial_losses, _ = run_serial(s_serial, inputs, targets, lr=lr)
    res = PipelineRunner(s_pipe, make_thread_links(len(s_pipe)), lr=lr,
                         window=len(inputs)).run(inputs, targets)
    # losses match (same forward pass, same v0 weights), and the UPDATED weights match
    assert np.allclose(res.losses, serial_losses, atol=1e-9)
    for a, b in zip(s_serial, s_pipe):
        assert np.allclose(a.W, b.W, atol=1e-9), "W=M update must equal serial update"
    assert res.max_staleness == 0, "no apply happens within one window -> zero staleness"


def test_small_window_is_stale():
    inputs, targets = _synthetic_task()
    res = PipelineRunner(make_linear_pipeline([5, 8, 6, 3], seed=3),
                         make_thread_links(3), lr=0.05, window=1).run(inputs, targets)
    # per-microbatch apply -> later microbatches' backwards see weights that moved
    assert res.max_staleness > 0


def test_runtime_actually_learns():
    """Repeated synchronous (W=M) epochs must drive the loss down — the runtime trains."""
    inputs, targets = _synthetic_task(n=32)
    stages = make_linear_pipeline([5, 12, 8, 3], seed=4)
    links = make_thread_links(len(stages))
    first = PipelineRunner(stages, links, lr=0.1, window=len(inputs)).run(inputs, targets).mean_loss
    last = first
    for _ in range(40):
        last = PipelineRunner(stages, make_thread_links(len(stages)),
                              lr=0.1, window=len(inputs)).run(inputs, targets).mean_loss
    assert last < first * 0.6, f"loss did not fall enough: {first:.4f} -> {last:.4f}"


def test_1f1b_full_window_equals_serial():
    """1F1B with W=M must ALSO reproduce the synchronous baseline bit-exact (no apply lands
    within the window, so every fwd/bwd uses v0 -> = full-batch backprop)."""
    inputs, targets = _synthetic_task()
    lr = 0.05
    s_serial = make_linear_pipeline([5, 8, 6, 3], seed=2)
    s_pipe = make_linear_pipeline([5, 8, 6, 3], seed=2)
    serial_losses, _ = run_serial(s_serial, inputs, targets, lr=lr)
    res = PipelineRunner(s_pipe, make_thread_links(len(s_pipe)), lr=lr,
                         window=len(inputs), schedule="1f1b").run(inputs, targets)
    assert np.allclose(res.losses, serial_losses, atol=1e-9)
    for a, b in zip(s_serial, s_pipe):
        assert np.allclose(a.W, b.W, atol=1e-9)
    assert res.max_staleness == 0 and res.schedule == "1f1b"


def test_1f1b_staleness_is_depth_proportional():
    """The hallmark of 1F1B: the deepest stage (l=0, farthest from the loss) carries the
    most weight-staleness; the loss stage (l=S-1) carries ~none. (W=1, the worst case.)"""
    S, M = 5, 16
    res = PipelineRunner(make_linear_pipeline([4] + [8] * (S - 1) + [2], seed=3),
                         make_thread_links(S), lr=0.05, window=1,
                         schedule="1f1b").run(*_synthetic_task(M, 4, 2))
    ps = res.per_stage_staleness
    assert ps[-1] == 0, "the loss stage does F then immediately B -> zero staleness"
    assert ps[0] >= ps[-1] + (S - 2), f"deepest stage should be ~S-1 stale, got {ps}"
    assert ps == sorted(ps, reverse=True), f"staleness must decrease with depth: {ps}"


def test_1f1b_window_bounds_staleness():
    """A big enough accumulation window collapses the depth-proportional staleness to <=1."""
    S, M = 5, 20
    res = PipelineRunner(make_linear_pipeline([4] + [8] * (S - 1) + [2], seed=3),
                         make_thread_links(S), lr=0.05, window=M,
                         schedule="1f1b").run(*_synthetic_task(M, 4, 2))
    assert res.max_staleness == 0   # W=M -> a single apply at the very end


def test_cost_word_profile_reproduces_measured_relay_fraction():
    """The WORD profile is calibrated to the MEASURED 61% host-relay fraction — guard it."""
    from tikitorch.fabric import comms_fraction
    from tikitorch.fabric.cost import CAL_COMPUTE_S, CAL_MSG_BYTES, HOST_RELAY_WORD
    f = comms_fraction(HOST_RELAY_WORD, CAL_MSG_BYTES, CAL_COMPUTE_S)
    assert abs(f - 0.61) < 0.01, f"word relay fraction {f:.3f} != measured 0.61"
    assert HOST_RELAY_WORD.measured is True


def test_cost_transport_ordering():
    """Fixing the transport must monotonically cut per-message time; the real links are
    all >>faster than word-at-a-time on an 8 KB message."""
    from tikitorch.fabric.cost import (CAL_MSG_BYTES, HOST_RELAY_BURST,
                                       HOST_RELAY_WORD, ONFABRIC, PCIE_P2P)
    n = CAL_MSG_BYTES
    tw, tb, tp, tf = (p.time_for(n) for p in
                      (HOST_RELAY_WORD, HOST_RELAY_BURST, PCIE_P2P, ONFABRIC))
    assert tw > 50 * tb            # word-at-a-time is catastrophically slower
    assert tb >= tp >= tf or (max(tb, tp, tf) < 1e-4)   # the fast links are ~equivalent (all compute-bound)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
