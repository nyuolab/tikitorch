# Multi-F2 scale-out: a PC backward sharded across two physical FPGAs, neighbour-only

**Date:** 2026-06-21. The model-parallel predictive-coding correctness proof — previously
shown only in `torch.distributed` simulation (`pc_model_parallel.py`, the gloo Test 1) — now
**runs across two real F2 FPGAs**. A 2-layer `cascade_grads` backward sweep is sharded across
slots 0 and 1 of one `f2.12xlarge` (2 FPGAs), each chip holding one layer, exchanging **only a
single neighbour error message**, and the result is **bit-for-bit identical to single-process
`cascade_grads`**.

## What ran

`forge_pc_2layer_shard.c` on an f2.12xlarge, both FPGAs loaded with the verified PC backward
AFI `agfi-063e2dd0d3255dcaa` (`forge_pc_wgrad`; see `PC_BACKWARD_SILICON.md`):

```
slot 1 (layer 1, top): forge_pc_wgrad(e_top, W1, acts1, mask1, skip1) -> gW1, gB1, e_down1
        |  NEIGHBOUR MESSAGE: e_down1 (the ONLY cross-FPGA data), rescaled to int16 fixed-point
        v
slot 0 (layer 0):      forge_pc_wgrad(e_down1, W0, acts0, mask0, skip0) -> gW0, gB0, e_down0
```

- Each FPGA computes **only its own layer's gradient, locally** — neither ever holds the other
  layer's weights or any global gradient.
- The error route on each chip **reuses that chip's own forward `q(W)`** — no weight transport.
- The two chips touch exactly **one message** (`e_down1`); no all-reduce, no global barrier, no
  globally-ordered backward.

## Result (`results/predictive_coding/multif2/shard.txt`)

```
slot 1: AP_DONE 0.693 ms ; layer 1: max|gW|=0  max|e_down|=0  max|gB|=0
neighbour message e_down1 (max|.|=4858480) rescaled >>10 to int16 fixed-point
slot 0: AP_DONE 0.706 ms ; layer 0: max|gW|=0  max|e_down|=0  max|gB|=0
-> BIT-EXACT vs single-process cascade_grads
```

Both layers' gradients, computed on **two separate physical FPGAs** with neighbour-message-only
communication, match the single-process reference exactly. This is the structural mechanism the
training-at-scale thesis rests on (`TRAINING_AT_SCALE.md`): **PC removes backprop's global ops,
so a model shards across multiplier-free chips with only neighbour messages** — demonstrated on
real silicon, not simulation.

## Honest scope — what this is and is NOT

- It **is** the PC *backward* credit assignment (where the no-global-ops claim lives) run across
  two physical FPGAs, neighbour-message-only, **bit-exact** — i.e. the carrier survives real
  device sharding, not just a single-process simulation.
- It is **correctness**, not a multi-step training run (no weight-update loop yet) and not an
  accuracy result.
- The **forward pass and the orchestration run on the host**; only the backward is sharded on
  the FPGAs. (The forward is a standard, equally-shardable sweep; the thesis is about the
  backward. A forward-on-fabric kernel per shard is the next extension.)
- It is **2 FPGAs on one host** (the neighbour message is passed via host memory), so it shows
  the *structural* property — each chip = one layer, neighbour-only, no global gradient — not yet
  network-distributed message-passing across separate hosts.
- The **inter-layer message is rescaled** to the int16 fixed-point range (a single host-side
  arithmetic shift): propagated PC errors grow each layer, standard fixed-point cascade dynamics
  (the muzero net needed activation scaling too). This is a per-message shift, not a global op.
- **Not a throughput win.** Per Rung 4 (`RUNG4_FABRIC_SIM.md`) the digital economics do not make
  PC faster than backprop; this demonstrates the *structural* "no global coordination" property
  on real separate devices, which is the substrate bet — not a speed claim.

## Reproduce

```
AGFI=agfi-063e2dd0d3255dcaa bash scripts/run_pc_multif2.sh launch   # f2.12xlarge, AGFI on both slots
bash scripts/run_pc_multif2.sh status ; bash scripts/run_pc_multif2.sh fetch
```

## Next

- Deeper shard (≥3 layers across ≥3 FPGAs; an f2.48xlarge has 8) to show the chain scales.
- A weight-update loop → a real *training* run sharded across FPGAs (not just gradient correctness).
- Network-distributed (separate hosts, TCP neighbour messages) for true no-shared-host messaging.
- A forward-on-fabric kernel per shard (fully on-device forward + backward + update).
