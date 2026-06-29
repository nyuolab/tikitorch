# tikitorch.fabric — the distributed predictive-coding runtime

The reusable runtime for training a model **sharded one block per chip** with
predictive coding's *local* credit assignment. It exists because PC's distributed
story is fundamentally lighter than backprop's, and that lightness should be a
first-class, transport-agnostic module — not re-implemented inside every driver.

## Why this is a module (the PC advantage, made concrete)

Distributed **backprop** pays two coordination costs that limit scaling:
- a **global gradient all-reduce** (every chip's weight gradient summed across *all*
  chips, every step), and
- a **forward→backward barrier** (a stage can't start its backward until the gradient
  has traversed the whole network and returned) → the pipeline **bubble**.

**PC pays neither.** Each chip updates its own weights from a *local* error the moment
that error arrives from its downstream neighbour. So the only communication is
**nearest-neighbour, point-to-point**:

```
forward  (down):  stage l  --activation-->  stage l+1
backward (up):    stage l  <----error-----  stage l+1
```

No all-reduce, no global barrier, no switch fabric — a linear neighbour chain suffices.
That is *exactly* a clean interface boundary: the algorithm needs four neighbour
operations; the transport behind them is an implementation detail.

## The interface — `NeighborLink`

One stage's view of its two neighbours (`link.py`):

```python
send_forward(tag, act)   recv_forward(tag) -> act     # activation to l+1 / from l-1
send_backward(tag, err)  recv_backward(tag) -> err     # error to l-1 / from l+1
feed_forward / feed_backward                            # host injection at the open ends
```

`tag` is the microbatch id — what the staleness/accumulation bookkeeping keys on. Byte
counts and wait-times are recorded per stage (the Gate-C instrumentation).

## Backends (write once, run anywhere)

| Backend          | Where        | Status | Notes |
|------------------|--------------|--------|-------|
| `ThreadLink`     | CPU sim / CI | **done** | threads + queues; real concurrency, laptop-runnable |
| `HostRelayLink`  | F2           | next   | bursted host DMA on the **existing AGFI** (no rebuild) — first measured relay-bursted throughput |
| `PcieP2PLink`    | F2           | next   | direct neighbour-DDR within an instance (bypasses the host) |
| `TcpLink`        | farm         | later  | cross-instance neighbour link (16-FPGA / 2-instance) |
| `FabricLink`     | future       | later  | on-die SLR / GT link |

The PC training loop is written against `NeighborLink` only, so the *same*
`PipelineRunner` logic runs on each backend — validate on `ThreadLink`, then measure on
silicon.

## The runtime — `PipelineRunner`

`runner.py` owns the three things PC's local credit makes possible (and backprop forbids):

1. **No global barrier** → stages overlap (the bubble removed).
2. **A grad-accumulation window `W`** to bound staleness. Keeping the pipeline full means
   a microbatch's backward lands after its forward; if weights were applied in between,
   the gradient is *stale*. `W` freezes weights over an aligned window and applies once.
   The runner **measures** the version gap its schedule produces, so `W` is tuned against
   a real number. (Silicon-measured: weight-staleness tolerates only ~1 step; `stale-8` =
   `+2.0 bpb`, unrecoverable by LR — hence `W >= 2*(L-1)` for an `L`-stage pipeline.)
3. **Instrumentation** — wall-clock, overlap speedup, bytes, measured staleness, steps/s.

A *stage* is any duck-typed object with `params`, `forward(tag, a_in)`,
`backward(tag, e_out) -> (grads, e_in)`, and (last stage) `loss_error(tag, a_out, y)`.
`stages.LinearStage` is the reference: its local backward *is* the PC cascade (= backprop
through one block), the same op the F2 kernels implement.

## Measured today (CPU `ThreadLink`, `scripts/pc_fabric_sim_demo.py`)

```
OVERLAP   8-stage pipeline, simulated 3ms/block, M=16 microbatches:
          serial 976ms -> pipelined 174ms  =  5.60x  (theoretical max 5.57x for this depth)

STALENESS vs window W (depth 6, M=24):  W=1 -> max gap 23 ;  W=12 -> 1 ;  W=24(=M) -> 0
          at W=M: zero staleness AND the updated weights are bit-exact to synchronous backprop
```

The bit-exact `W = M == serial` equivalence (`tests/test_fabric.py`) is the correctness
gate: the windowed local update reproduces backprop exactly when the window spans the
sweep, so anything `W < M` is a *deliberate* staleness/throughput trade, measured.

## Status & next

- **Done:** interface, `ThreadLink`, `PipelineRunner` (windowed local update + staleness +
  overlap instrumentation), reference `LinearStage`, demo, tests (4/4).
- **Next:** `HostRelayLink` (bursted) on the existing MNIST/Fashion AGFI → first
  *measured* relay-free/bursted distributed throughput; a true 1F1B schedule (depth-
  proportional staleness) + a real PC-block stage; then `PcieP2PLink`.
- **Refactor target:** factor the transport out of `tikitorch/fpga/silicon/
  forge_pc_xformer_*_shard.cpp` and the cross-instance TCP scaffold to *back* these
  interfaces, so the silicon drivers and the CPU sim share one runtime.

## Comms cost: the case for a chip-to-chip link (`cost.py`)

The overlap above measures *how well stages pipeline*; `cost.py` adds *how expensive a
neighbour message is*, so the runner produces **comms-aware** throughput and the value of
each transport is quantified before it's built. `cost.TransportProfile.time_for(nbytes)`
is one transparent formula (setup + bytes/bandwidth + per-word overhead); pass a profile to
`make_thread_links(transport=..., msg_bytes=...)` and the runner's wall-clock includes comms.

The `HOST_RELAY_WORD` profile is **calibrated to the MEASURED 61% host-relay fraction** (at
the measured 10.25 ms/block, 8 KB neighbour message); `HOST_RELAY_BURST` / `PCIE_P2P` /
`ONFABRIC` are realistic transport params (PROJECTED). `scripts/pc_fabric_comms_sweep.py`
runs the *same* pipeline under each (measured-in-sim):

```
transport                      comms/msg  comms-frac   tok/s   vs word
host-relay word (today)          16.05ms     61.0%      1125    1.0x     <- MEASURED-calibrated, relay-bound
host-relay bursted DMA            0.02ms      0.2%       4319    3.9x     <- compute-bound
PCIe P2P / on-fabric              ~0  ms      ~0 %      ~4300    3.9x     <- equivalent once compute-bound
```

So **getting off word-at-a-time host relay ≈ 3.9× throughput** (it removes the 61% relay and
makes the pipeline compute-bound); the choice *among* burst / P2P / on-fabric is about scaling
and host-CPU load, not this throughput. That transport lever multiplies with two independently
measured ones — pipeline overlap (5.6×) and silicon replication (4.0×). A chip-to-chip link is
the **load-bearing first lever**: it is what turns the measured `25.8 -> 4.0 steps/s`
relay-bound *regression* into compute-bound scaling. (Unlike a throughput model that asserts
"comms = 0", every number here is explicit, calibrated, and adjustable.)

This module is the concrete answer to "PC should make chip-to-chip comms simple" — it
makes the simple pattern explicit, measurable, and reusable across every backend.
