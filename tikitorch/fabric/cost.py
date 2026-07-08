"""tikitorch.fabric.cost — transport cost model for the neighbour link.

The fabric's measured *overlap* (from the threaded runner) answers "how well do stages
pipeline." This adds the other half: "how expensive is moving a neighbour message,"
so the runner produces **comms-aware** throughput and we can quantify the win of each
chip-to-chip transport BEFORE building it on silicon.

Grounding (results/predictive_coding/multi_fpga/, MEASURED on the L=8 dyadic-MLP rig):
  * per-chip block compute  ~ 10.25 ms  (fwd-kick 8.924 ms + bwd-kick 1.329 ms)
  * host-relay comms is     ~ 61 %  of wall-clock at L=8 (word-at-a-time fpga_pci_peek)
  * throughput regresses 25.8 -> 9.1 -> 4.0 steps/s as chips go 2 -> 4 -> 8
  * the doc's own conclusion: "a chip-to-chip link or DMA-burst readback would collapse
    the 61 % comms."

So the WORD profile is CALIBRATED to reproduce that 61 % at 10.25 ms/block for an 8 KB
neighbour message; BURST / P2P / FABRIC are realistic transport params (PROJECTED) that
show how far the relay collapses. Everything is one transparent formula — unlike a model
that asserts "comms = 0," you can see and change every number.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransportProfile:
    """A neighbour-message transport. ``time_for(nbytes)`` = seconds to move one message."""

    name: str
    setup_s: float            # fixed per-message latency (DMA descriptor / handshake / round-trip)
    bandwidth_Bps: float      # sustained bytes/sec once streaming
    per_word_s: float = 0.0   # extra per-32-bit-word cost — the word-at-a-time host-peek killer
    measured: bool = False    # True only for the calibrated WORD profile

    def time_for(self, nbytes: int) -> float:
        words = (nbytes + 3) // 4
        return self.setup_s + nbytes / self.bandwidth_Bps + self.per_word_s * words


# Reference neighbour-message size used for calibration: a D=64 / seq=64 int16 activation
# (or error) tensor = 64*64*2 = 8192 B. The transport *characteristics* (per-word overhead,
# bandwidth) are message-size-independent, so calibrating here transfers to any D.
CAL_MSG_BYTES = 64 * 64 * 2          # 8 KB
CAL_COMPUTE_S = 10.25e-3             # MEASURED per-block compute (fwd+bwd kick)
CAL_RELAY_FRAC = 0.61                # MEASURED host-relay fraction at L=8


def _calibrate_per_word() -> float:
    """per_word_s that makes the WORD profile reproduce the measured 61% relay fraction:
    relay_time / (relay_time + compute) = 0.61  ->  relay_time = compute * f/(1-f)."""
    relay_s = CAL_COMPUTE_S * CAL_RELAY_FRAC / (1.0 - CAL_RELAY_FRAC)   # ~16.0 ms
    words = CAL_MSG_BYTES // 4                                          # 2048 words
    return relay_s / words                                             # ~7.8 us / word


# today's path: host reads each neighbour message word-at-a-time over PCIe (fpga_pci_peek).
# CALIBRATED (measured) — reproduces the 61% relay-bound rig.
HOST_RELAY_WORD = TransportProfile(
    "host-relay word-at-a-time (today)", setup_s=20e-6, bandwidth_Bps=1e12,
    per_word_s=_calibrate_per_word(), measured=True,
)
# Stage-0 fix: a single bursted BAR4 DMA instead of N peeks. PCIe DMA ~8 GB/s. PROJECTED.
HOST_RELAY_BURST = TransportProfile(
    "host-relay bursted DMA (Stage-0 fix)", setup_s=20e-6, bandwidth_Bps=8e9)
# direct neighbour-DDR within an instance — no host bounce (half the hops). PROJECTED.
PCIE_P2P = TransportProfile(
    "PCIe peer-to-peer (direct neighbour DDR)", setup_s=5e-6, bandwidth_Bps=12e9)
# on-die SLR / GT link — the eventual interconnect. PROJECTED.
ONFABRIC = TransportProfile(
    "on-fabric SLR/GT link", setup_s=0.5e-6, bandwidth_Bps=50e9)

PROFILES = [HOST_RELAY_WORD, HOST_RELAY_BURST, PCIE_P2P, ONFABRIC]


def comms_fraction(profile: TransportProfile, msg_bytes: int, compute_s: float) -> float:
    """Fraction of a serialized (compute + comms) microbatch spent in comms."""
    c = profile.time_for(msg_bytes)
    return c / (c + compute_s)
