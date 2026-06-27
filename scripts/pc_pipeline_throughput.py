"""North-star projection (BUILD_PLAN.md): can a ~124M distributed PC transformer train on the multi-F2 fabric in
HOURS, via the no-global-sync bubble-free pipeline? Grounds the projection in MEASURED fabric numbers and labels
every step MEASURED vs PROJECTED. Run: python scripts/pc_pipeline_throughput.py

The honest claim is NOT "beats an A100" (a single F2 < a GPU). It is: the no-global-backward property removes the
pipeline bubble that bleeds GPU model-parallel utilization, so the SCALABLE architecture trains 124M in hours and
PROJECTS to favorable strong-scaling. Numbers below make that concrete.
"""
import math

# ---------- MEASURED anchors (from the distributed PC fabric + the GPU reference; see memory) ----------
M_SERIAL_STEPS_S   = 4.0      # MEASURED: serial distributed driver, 8 chips, D=64, host-relay-bound
M_RELAY_FRAC       = 0.54     # MEASURED: 54% of wall-clock is the word-at-a-time host PCIe relay
M_TCHIP_MS         = 8.92     # MEASURED: per-chip transformer-block time, D=64 ("hidden 8.92ms")
M_MORPH_N4_TOKS    = 53_000   # MEASURED: best F2 LM throughput (morph-LM N=4 replicated), single card
M_A10G_TOKS        = 107_000  # MEASURED: A10G nanoGPT reference (107k tok/s, 1.46 mJ/tok)
SEQ                = 64       # tokens per microbatch kick (current kernel)

# ---------- target ----------
D124, L124, NH124  = 768, 12, 12     # GPT-2-small / "124M" config (124.4M params)
TINYSHAKES_TOKENS  = 1_000_000       # ~1M chars; coherent overfit ~ a few epochs (~5-10M tokens)
TRAIN_TOKENS       = 6_000_000

def banner(s): print("\n"+s+"\n"+"-"*len(s))

banner("1. The bubble-free pipeline speedup (the core no-global-sync claim)")
# serial: a training step sweeps fwd+bwd through all L chips serially -> latency ~ L*(t_chip + t_relay)
# pipeline (PC, bubble-free): each chip is ALWAYS busy (fwd of microbatch k+1 while bwd of k); throughput
#   is bounded by ONE chip's work, not the sum over L. PC has no global backward -> no pipeline-flush bubble.
L = 8
t_relay_ms = M_TCHIP_MS * M_RELAY_FRAC/(1-M_RELAY_FRAC)
serial_step_ms   = L*(M_TCHIP_MS + t_relay_ms)
pipe_step_ms_now = (M_TCHIP_MS + t_relay_ms)            # still host-relay, but pipelined (overlapped)
pipe_step_ms_fab = M_TCHIP_MS                            # + on-fabric interconnect (relay removed) -> compute-bound
print(f"  serial step (MEASURED-grounded):        {serial_step_ms:7.1f} ms   ({1000/serial_step_ms*SEQ:6.0f} tok/s)")
print(f"  pipelined, still host-relay (PROJ):     {pipe_step_ms_now:7.1f} ms   x{serial_step_ms/pipe_step_ms_now:.1f} over serial")
print(f"  pipelined + on-fabric interconnect:     {pipe_step_ms_fab:7.1f} ms   x{serial_step_ms/pipe_step_ms_fab:.1f} over serial  (compute-bound)")
print(f"  vs GPU pipeline-parallel: GPipe bubble ~ (L-1)/(M+L-1); PC bubble = 0 -> ~2x better utilization at M~L")

banner("2. Scale the layer D=64 -> D=768 (124M), with the FPGA compute-density levers")
# per-layer compute scales ~D^2 (the projection/FFN matmuls dominate). Levers that recover it on FPGA:
comp_scale = (D124/64)**2                                  # 144x more MACs per layer
# DSP-as-adder array: ~9000 DSP48 used as shift-add ALUs (mult-free bulk) vs the handful used at D=64.
# At D=64 the kernel uses a small slice of the fabric; D=768 unrolls token- and channel-parallel across it.
ADDER_PARALLELISM = 64        # PROJECTED: effective extra parallel shift-add lanes from lighting the adder array
t_chip_768_ms = M_TCHIP_MS * comp_scale / ADDER_PARALLELISM
print(f"  per-layer compute D=768 vs D=64:        x{comp_scale:.0f} MACs")
print(f"  DSP-adder array + token/channel unroll: /{ADDER_PARALLELISM} parallel lanes (PROJECTED)")
print(f"  => per-chip layer time D=768 (PROJ):    {t_chip_768_ms:7.1f} ms  ({1000/t_chip_768_ms*SEQ:6.0f} tok/s/pipeline)")

banner("3. 124M training time on multi-F2 (bubble-free pipeline + on-fabric + replication)")
# 12 layers -> 12 pipeline stages (>=12 SLRs; an f2.48xlarge has 8 cards x SLRs, or 2 instances). Throughput
# of a bubble-free pipeline = ONE stage's rate. Replication R adds data-parallel copies (measured 4x linear).
for R in (1, 4, 8):
    toks_s = (1000.0/t_chip_768_ms)*SEQ*R
    hrs = TRAIN_TOKENS/toks_s/3600
    print(f"  replication R={R}:  {toks_s:8.0f} tok/s   -> TinyShakespeare {TRAIN_TOKENS/1e6:.0f}M tokens in {hrs:5.2f} h")

banner("HONEST verdict")
print("""  MEASURED: serial driver 4 steps/s @ D=64 (host-relay-bound); per-chip block 8.92ms; morph-LM 53k tok/s.
  PROJECTED (contingent on the build): bubble-free pipeline ~Lx over serial; +on-fabric interconnect removes the
  54% relay; +DSP-adder array recovers the D=768 cost; replication multiplies. Net: 124M -> coherent-overfit
  TinyShakespeare in SINGLE-DIGIT HOURS on one f2.48xlarge-class fabric. NOT faster than an A100 on this toy
  (an A100 is minutes) -- the point is the SCALABLE no-sync pipeline (where GPU model-parallel bleeds the bubble)
  + energy. Every "PROJECTED" number above becomes MEASURED only after the pipeline driver + on-fabric link +
  D=768 kernel are built and instrumented.""")
