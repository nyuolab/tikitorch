"""Lightweight hardware-cost accounting for tikitorch models.

This is the decoupled, optional analogue of forge's ``HardwareCostProfile``.
It is a plain dataclass with no abstract base class and no required methods —
models may *optionally* expose a ``hardware_cost(...)`` method that returns one
of these, but nothing in tikitorch requires it. Models subclass
``torch.nn.Module`` directly (PyTorch-idiomatic), not an abstract base.
"""

from dataclasses import dataclass

__all__ = ["HardwareCostProfile"]


@dataclass
class HardwareCostProfile:
    """Hardware cost accounting for a single forward or forward+backward pass.

    All fields default to zero. The headline number for tropical models is
    ``multiplications`` / ``dsp_blocks_required``: a fully-closed tropical core
    reports zero multiplications in its sequence/attention/FFN path, with DSPs
    needed only for the (optional, standard) embedding and output head.
    """

    # Per-forward-pass costs
    multiplications: int = 0
    additions: int = 0
    comparisons: int = 0
    table_lookups: int = 0
    xor_operations: int = 0
    memory_reads_bytes: int = 0
    memory_writes_bytes: int = 0

    # Per-backward-pass costs
    backward_multiplications: int = 0
    backward_additions: int = 0
    backward_comparisons: int = 0
    backward_table_lookups: int = 0

    # Derived FPGA estimates
    dsp_blocks_required: int = 0
    lut_estimate: int = 0
    bram_kb: int = 0
    estimated_clock_mhz: int = 250
    estimated_throughput_samples_sec: float = 0.0
    estimated_power_watts: float = 0.0

    # Derived GPU comparison
    equivalent_gpu_flops: int = 0
    gpu_throughput_samples_sec: float = 0.0

    @property
    def efficiency_ratio(self) -> float:
        """Samples/sec/watt on FPGA vs GPU. >1 means FPGA wins."""
        if self.estimated_power_watts == 0 or self.gpu_throughput_samples_sec == 0:
            return 0.0
        fpga_eff = self.estimated_throughput_samples_sec / self.estimated_power_watts
        gpu_eff = self.gpu_throughput_samples_sec / 300  # A100 TDP
        return fpga_eff / gpu_eff

    def summary(self) -> dict:
        """Return a dict summary for logging."""
        return {
            "fwd_multiplications": self.multiplications,
            "fwd_additions": self.additions,
            "fwd_comparisons": self.comparisons,
            "fwd_table_lookups": self.table_lookups,
            "fwd_xor_operations": self.xor_operations,
            "dsp_blocks": self.dsp_blocks_required,
            "lut_estimate": self.lut_estimate,
            "bram_kb": self.bram_kb,
            "efficiency_ratio": self.efficiency_ratio,
        }
