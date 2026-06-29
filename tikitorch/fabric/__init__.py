"""tikitorch.fabric — the distributed predictive-coding fabric runtime.

PC shards a model one block per chip and trains with *local* credit assignment: no
global gradient all-reduce, no forward-backward barrier — only nearest-neighbour
messages (activation down, error up). This package is the transport-agnostic runtime
for exactly that pattern, so the PC training loop is written once and run anywhere:

    from tikitorch.fabric import make_thread_links, PipelineRunner

    links  = make_thread_links(num_stages)          # CPU sim today; HostRelay/P2P on F2
    runner = PipelineRunner(stages, links, lr=..., window=W)
    result = runner.run(inputs, targets)            # overlap + staleness + bytes, measured

See ``link.NeighborLink`` for the interface every transport backend implements, and
``README.md`` for the design + the backend roadmap.
"""
from .cost import (
    HOST_RELAY_BURST,
    HOST_RELAY_WORD,
    ONFABRIC,
    PCIE_P2P,
    PROFILES,
    TransportProfile,
    comms_fraction,
)
from .link import LinkStats, NeighborLink
from .runner import PipelineRunner, RunResult, run_serial
from .thread_link import ThreadLink, make_thread_links

__all__ = [
    "NeighborLink",
    "LinkStats",
    "ThreadLink",
    "make_thread_links",
    "PipelineRunner",
    "RunResult",
    "run_serial",
    "TransportProfile",
    "comms_fraction",
    "PROFILES",
    "HOST_RELAY_WORD",
    "HOST_RELAY_BURST",
    "PCIE_P2P",
    "ONFABRIC",
]
