"""tikitorch.models — ready-to-use tropical architectures (a la torchvision/timm).

Every model subclasses
``torch.nn.Module`` directly (no abstract base class), takes plain
keyword arguments (no ``config`` dict), and composes the tropical building
blocks from :mod:`tikitorch.nn`. The tropical / closure math (forward,
backward via the autograd Functions in :mod:`tikitorch.functional`, and the
architecture wiring) is ported verbatim from the validated research code.

Public models:

    TropicalMLP           — alternating-min/max tropical MLP (classification)
    TropicalSequenceModel — tropical recurrence for sequence classification
    MorphSSM              — morphological SSM (dilation/erosion conv) LM/classifier
    MorphMLP              — alternating dilation/erosion MLP
    TropicalTransformer   — transformer with tropical attention + tropical FFN
    MoTEModel (alias MoTE)— Mixture of Tropical Experts

The ``task_type`` / loss convention:
    Every model takes a ``task_type`` keyword (``"classification"``, ``"lm"``,
    or ``"regression"`` — the subset each supports is documented per class) and
    exposes a concrete ``compute_loss(x, y)`` method. ``compute_loss`` is the
    standard task loss (cross-entropy for
    classification / lm, MSE for regression). It is *not* enforced by any base
    class — it is provided because the training loops rely on it. MoTEModel
    additionally adds its load-balancing ``aux_loss`` term.

Each model also exposes a cheap ``param_count`` property and, where applicable,
an optional ``hardware_cost(...)`` method returning a
:class:`tikitorch.HardwareCostProfile` (with ``get_hardware_cost_profile`` kept
as an alias). These are optional conveniences, never required.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tikitorch import HardwareCostProfile
from tikitorch.backend import FpgaAccelerable
from tikitorch.functional import (
    tropical_cross_entropy,
    tropical_leaky_relu,
    tropical_perceptron_loss,
)
from tikitorch.nn import (
    _make_norm,
    CompositionalTropicalProjection,
    MorphConvBlock,
    MorphSSMLayer,
    MoTEBlock,
    TropicalAdaptiveMaxPool2d,
    TropicalBatchShift,
    TropicalDilationModule,
    TropicalDropout,
    TropicalGlobalMaxPool,
    TropicalLinearModule,
    TropicalQuantizedEmbed,
    TropicalTopKModule,
    TropicalTransformerBlock,
)


def _make_head(head_type: str, in_dim: int, out_dim: int, topk_k: int = 8):
    """Build a classification/LM output head.

    ``"tropical"`` (default for a fully-closed model) is a multiplier-free
    :class:`TropicalTopKModule` -- a top-k max-plus aggregation that beat every
    other closed head in the sweep (MNIST 0.80 at the same param count as a dense
    head, vs 0.62 for a single dilation∘erosion). ``"linear"`` is a standard
    ``nn.Linear`` head, the declared DSP boundary.
    """
    if head_type == "tropical":
        return TropicalTopKModule(in_dim, out_dim, k=min(topk_k, in_dim))
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    raise ValueError(f"head_type must be 'tropical' or 'linear', got {head_type!r}")


def _make_input_proj(input_type: str, in_dim: int, out_dim: int):
    """Build an input projection (continuous-feature -> d_model).

    ``"tropical"`` (default closed) is :class:`TropicalQuantizedEmbed` --
    quantize each continuous feature and gather a learned per-bin vector. This is
    the *expressive* closed input: a tropical add-only projection of a scalar is
    rank-degenerate (can shift but not scale), which stalls training, whereas the
    quantize+gather embedding is closed AND expressive (it unlocked fully-closed
    sequence tasks). Assumes inputs in [0, 1] (normalize first). ``"linear"`` is
    the ``nn.Linear`` boundary. (LM models embed tokens with ``nn.Embedding``.)
    """
    if input_type == "tropical":
        return TropicalQuantizedEmbed(in_dim, out_dim, n_bins=32, lo=0.0, hi=1.0)
    if input_type == "linear":
        return nn.Linear(in_dim, out_dim)
    raise ValueError(f"input_type must be 'tropical' or 'linear', got {input_type!r}")


_VALID_TASK_TYPES = ("classification", "lm", "regression")


def _validate_task_type(task_type):
    """Validate the ``task_type`` keyword shared by every model.

    Fails fast with a clear message at construction instead of a cryptic error
    deep in a later ``compute_loss``/forward. (Not every model supports all three
    — e.g. ``TropicalCNN`` is classification-only — but every supported value is
    one of these; an unsupported-but-valid value still errors where it's used.)
    """
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {_VALID_TASK_TYPES}, got {task_type!r}"
        )
    return task_type


__all__ = [
    "TropicalMLP",
    "TropicalSequenceModel",
    "MorphSSM",
    "MorphMLP",
    "TropicalCNN",
    "TropicalTransformer",
    "MoTEModel",
    "MoTE",
    "classification_loss",
]


# ---------------------------------------------------------------------------
# Tropical MLP
# ---------------------------------------------------------------------------


class TropicalMLP(FpgaAccelerable, nn.Module):
    """MLP using tropical linear layers + max nonlinearity.

    Args:
        d_model: hidden dimension (default 512)
        n_layers: number of hidden layers (default 4)
        dropout: dropout rate (default 0.1)
        temperature: softmin temperature tau (default 1.0)
        use_softmin: use differentiable softmin (default True for training)
        tau_min: minimum tau for annealing (default 0.01)
        input_dim: input dimension (default 784)
        num_classes: output classes (default 10)
        task_type: "classification" (default)
    """

    def __init__(self, *, d_model: int = 512, n_layers: int = 4,
                 dropout: float = 0.1, temperature: float = 1.0,
                 use_softmin: bool = True, tau_min: float = 0.01,
                 input_dim: int = 784, num_classes: int = 10,
                 task_type: str = "classification"):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_softmin = use_softmin
        self.tau_min = tau_min

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task_type = _validate_task_type(task_type)

        # Build tropical layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_dim = self.input_dim
        for _ in range(n_layers):
            self.layers.append(
                TropicalLinearModule(
                    in_dim, self.d_model,
                    use_softmin=self.use_softmin,
                    tau=self.temperature,
                )
            )
            self.dropouts.append(TropicalDropout(dropout) if dropout > 0 else nn.Identity())
            in_dim = self.d_model

        # Standard linear output head (for fair perplexity comparison)
        self.head = _make_head("tropical", self.d_model, self.num_classes)

        # Track which layers use min vs max for alternating nonlinearity
        self._use_max = [i % 2 == 1 for i in range(n_layers)]

    def _tropical_activation(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Tropical activation: alternating min/max between layers.

        Even layers -> already computed min (from tropical linear).
        Odd layers -> apply element-wise max(x, 0) as nonlinearity.
        This alternation avoids collapsing everything to a single min path.
        """
        if self._use_max[layer_idx]:
            return torch.max(x, torch.zeros_like(x))
        else:
            # Even layers: leaky tropical activation with a POWER-OF-2 slope
            # (max(x, x>>3) = slope 0.125), so it is multiplier-free. Previously
            # this was `x * 0.1`, a non-power-of-2 float multiply (closure hole H1).
            return tropical_leaky_relu(x, shift=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.float()
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        for i, (layer, drop) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            x = self._tropical_activation(x, i)
            x = drop(x)

        return self.head(x)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _default_compute_loss(self, x, y)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_temperature(self, tau: float):
        """Update temperature across all tropical layers (for annealing schedule)."""
        self.temperature = max(tau, self.tau_min)
        for layer in self.layers:
            layer.set_tau(self.temperature)

    def switch_to_hard_min(self):
        """At eval time, switch all layers to hard min (no multiplications)."""
        for layer in self.layers:
            layer.use_softmin = False

    def switch_to_soft_min(self):
        """Switch back to soft min for training."""
        for layer in self.layers:
            layer.use_softmin = True

    def get_gradient_sparsity(self) -> dict:
        """Report per-layer gradient sparsity (fraction of zero gradients).

        This is THE key diagnostic for tropical networks: gradients should be
        highly sparse because only argmin indices receive non-zero gradient.
        """
        sparsity = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                total = param.grad.numel()
                zeros = (param.grad == 0).sum().item()
                sparsity[name] = zeros / total if total > 0 else 0.0
        return sparsity

    def get_gradient_sparsity_summary(self) -> dict:
        """Aggregate gradient sparsity stats for logging."""
        per_layer = self.get_gradient_sparsity()
        if not per_layer:
            return {"grad_sparsity/tropical_mean": 0.0}

        values = list(per_layer.values())
        return {
            "grad_sparsity/tropical_mean": sum(values) / len(values),
            "grad_sparsity/tropical_max": max(values),
            "grad_sparsity/tropical_min": min(values),
        }

    def hardware_cost(self, batch_size: int = 1, seq_len: int = 1) -> HardwareCostProfile:
        """Hardware cost: zero multiplications in tropical layers.

        Forward: each tropical linear does additions + comparisons (argmin).
        Backward: sparse gradient routing (additions only, no multiplies).
        Only the output head uses standard multiplications.
        """
        total_additions = 0
        total_comparisons = 0
        total_multiplications = 0

        n_samples = batch_size * seq_len

        # Tropical layers: additions + comparisons, zero multiplications
        in_dim = self.input_dim
        for layer in self.layers:
            out_dim = layer.out_features
            # Forward: out_dim * in_dim additions (W + x), out_dim * (in_dim-1) comparisons (min)
            total_additions += n_samples * out_dim * in_dim
            total_comparisons += n_samples * out_dim * (in_dim - 1)
            in_dim = out_dim

        # Output head: standard linear -> multiplications
        head_in = self.d_model
        head_out = self.num_classes
        total_multiplications += n_samples * head_in * head_out

        # Backward pass: tropical layers produce sparse gradients (additions only)
        bwd_additions = total_additions  # same order as forward
        bwd_comparisons = 0  # argmin indices cached from forward
        bwd_multiplications = n_samples * head_in * head_out  # only the head

        # Memory: weights + activations
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        activation_bytes = n_samples * self.d_model * len(self.layers) * 4  # float32

        return HardwareCostProfile(
            multiplications=total_multiplications,
            additions=total_additions,
            comparisons=total_comparisons,
            backward_multiplications=bwd_multiplications,
            backward_additions=bwd_additions,
            backward_comparisons=bwd_comparisons,
            memory_reads_bytes=param_bytes + activation_bytes,
            memory_writes_bytes=activation_bytes,
            # FPGA estimates: tropical layers need only comparators + adders, not DSP blocks
            dsp_blocks_required=head_in * head_out // 16,  # only the head needs DSPs
            lut_estimate=total_comparisons // 4,  # comparators are cheap in LUTs
            bram_kb=param_bytes // 1024,
        )

    get_hardware_cost_profile = hardware_cost


# ---------------------------------------------------------------------------
# Tropical Sequence Model
# ---------------------------------------------------------------------------


class TropicalSequenceModel(FpgaAccelerable, nn.Module):
    """Tropical model for sequence tasks (sMNIST, sCIFAR).

    Processes sequences via tropical recurrence:
        h_t = min(W_h + h_{t-1}, W_x + x_t)
    where + is standard addition (tropical multiplication).

    Args:
        d_model: hidden/recurrence dimension (default 256)
        n_layers: number of stacked tropical recurrence layers (default 2)
        dropout: dropout rate (default 0.1)
        temperature: softmin tau (default 1.0)
        use_softmin: differentiable relaxation (default True)
        tau_min: minimum tau for annealing (default 0.01)
        input_dim: per-timestep input dim (default 1)
        seq_len: sequence length (default 784)
        num_classes: output classes (default 10)
        task_type: "classification" (default)
    """

    def __init__(self, *, d_model: int = 256, n_layers: int = 2,
                 dropout: float = 0.1, temperature: float = 1.0,
                 use_softmin: bool = True, tau_min: float = 0.01,
                 input_dim: int = 1, seq_len: int = 784,
                 num_classes: int = 10, task_type: str = "classification"):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.temperature = temperature
        self.use_softmin = use_softmin
        self.tau_min = tau_min

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.task_type = _validate_task_type(task_type)

        # Closed tropical input projection (N-D capable).
        self.input_proj = _make_input_proj("tropical", self.input_dim, self.d_model)

        # Tropical recurrence layers
        # Each layer has W_h (hidden-to-hidden) and W_x (input-to-hidden)
        self.W_h = nn.ModuleList()
        self.W_x = nn.ModuleList()
        self.layer_drops = nn.ModuleList()
        for _ in range(self.n_layers):
            self.W_h.append(
                TropicalLinearModule(
                    self.d_model, self.d_model,
                    use_softmin=self.use_softmin,
                    tau=self.temperature,
                )
            )
            self.W_x.append(
                TropicalLinearModule(
                    self.d_model, self.d_model,
                    use_softmin=self.use_softmin,
                    tau=self.temperature,
                )
            )
            self.layer_drops.append(TropicalDropout(dropout) if dropout > 0 else nn.Identity())

        # Output head
        self.head = _make_head("tropical", self.d_model, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.float()

        # x: (batch, seq_len) or (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)

        batch, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Process through tropical recurrence layers
        for layer_idx in range(self.n_layers):
            h = torch.zeros(batch, self.d_model, device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(seq_len):
                x_t = x[:, t, :]  # (batch, d_model)
                h_from_h = self.W_h[layer_idx](h)   # tropical: min(W_h + h)
                h_from_x = self.W_x[layer_idx](x_t)  # tropical: min(W_x + x_t)
                # Tropical addition (min) of the two paths
                h = torch.min(h_from_h, h_from_x)
                outputs.append(h)

            x = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
            x = self.layer_drops[layer_idx](x)

        # Use final hidden state for classification
        final_h = x[:, -1, :]  # (batch, d_model)
        return self.head(final_h)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _default_compute_loss(self, x, y)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_temperature(self, tau: float):
        """Update temperature across all tropical layers."""
        self.temperature = max(tau, self.tau_min)
        for wh, wx in zip(self.W_h, self.W_x):
            wh.set_tau(self.temperature)
            wx.set_tau(self.temperature)

    def get_gradient_sparsity(self) -> dict:
        """Report per-layer gradient sparsity."""
        sparsity = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                total = param.grad.numel()
                zeros = (param.grad == 0).sum().item()
                sparsity[name] = zeros / total if total > 0 else 0.0
        return sparsity

    def get_gradient_sparsity_summary(self) -> dict:
        """Aggregate gradient sparsity stats for logging."""
        per_layer = self.get_gradient_sparsity()
        if not per_layer:
            return {"grad_sparsity/tropical_mean": 0.0}
        values = list(per_layer.values())
        return {
            "grad_sparsity/tropical_mean": sum(values) / len(values),
            "grad_sparsity/tropical_max": max(values),
            "grad_sparsity/tropical_min": min(values),
        }

    def hardware_cost(self, batch_size: int = 1, seq_len=None) -> HardwareCostProfile:
        """Hardware cost: zero multiplies in tropical recurrence layers."""
        if seq_len is None:
            seq_len = self.seq_len

        n_samples = batch_size

        # Input projection: standard linear multiplications
        proj_muls = n_samples * seq_len * self.input_dim * self.d_model

        # Tropical recurrence: additions + comparisons per timestep per layer
        trop_additions = 0
        trop_comparisons = 0
        for _ in range(self.n_layers):
            # W_h and W_x each: d_model * d_model additions, d_model*(d_model-1) comparisons
            per_step = 2 * self.d_model * self.d_model  # additions
            per_step_cmp = 2 * self.d_model * (self.d_model - 1) + self.d_model  # comparisons + element min
            trop_additions += n_samples * seq_len * per_step
            trop_comparisons += n_samples * seq_len * per_step_cmp

        # Output head
        head_muls = n_samples * self.d_model * self.num_classes

        total_muls = proj_muls + head_muls  # only non-tropical parts

        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())

        return HardwareCostProfile(
            multiplications=total_muls,
            additions=trop_additions + proj_muls + head_muls,
            comparisons=trop_comparisons,
            backward_multiplications=total_muls,
            backward_additions=trop_additions,
            backward_comparisons=0,
            memory_reads_bytes=param_bytes,
            memory_writes_bytes=n_samples * seq_len * self.d_model * 4 * self.n_layers,
            dsp_blocks_required=(proj_muls + head_muls) // (16 * seq_len),
            lut_estimate=trop_comparisons // 4,
            bram_kb=param_bytes // 1024,
        )

    get_hardware_cost_profile = hardware_cost


# ---------------------------------------------------------------------------
# Morphological SSM
# ---------------------------------------------------------------------------


class MorphSSM(FpgaAccelerable, nn.Module):
    """Sequence model using morphological convolutions.

    Alternating dilation (max-plus) and erosion (min-plus) depthwise
    convolutions for sequence mixing.  Dilated kernels provide exponentially
    growing receptive fields (WaveNet-style).

    Args:
        d_model: model dimension (default 256)
        n_layers: number of layers (default 8)
        kernel_size: morphological kernel size (default 7)
        dropout: dropout rate (default 0.1)
        use_soft: soft relaxation for training (default True)
        tau: temperature for soft max/min (default 1.0)
        tropical_proj: use tropical (vs nn.Linear) channel-mixing projection
        norm_type: "layernorm" (default) or "tropical_shift"
        group_size: morph-conv channel grouping (default 1)
        max_dilation: cap on the exponential per-layer dilation (default 2048)
        multi_scale: use MultiScaleMorphConv1d per layer (default False)
        kernel_sizes: kernel sizes when multi_scale (default (3, 7, 15))
        residual_scaling: scale residual by 1/sqrt(n_layers) (default False).
            NOTE: 1/sqrt(n_layers) is a non-power-of-2 data-path multiply, so
            residual_scaling=True is NOT closure-preserving; leave False (or use a
            power-of-2 scale) for the multiplier-free configuration.
        gradient_checkpointing: checkpoint layers during training (default False)
        task_type: "classification" | "lm" | "regression" (default "classification")
        vocab_size: vocabulary size when task_type == "lm" (default 32000)
        input_dim: input feature dim otherwise (default 1)
        num_classes: output classes / regression output dim otherwise (default 10)
    """

    def __init__(self, *, d_model: int = 256, n_layers: int = 8,
                 kernel_size: int = 7, dropout: float = 0.1,
                 use_soft: bool = True, tau: float = 1.0,
                 tropical_proj: bool = False, norm_type: str = "tropical",
                 group_size: int = 1, max_dilation: int = 2048,
                 multi_scale: bool = False, kernel_sizes=(3, 7, 15),
                 residual_scaling: bool = False,
                 gradient_checkpointing: bool = False,
                 head_type: str = "tropical", input_type: str = "tropical",
                 task_type: str = "classification",
                 vocab_size: int = 32000, input_dim: int = 1,
                 num_classes: int = 10):
        super().__init__()
        kernel_sizes = tuple(kernel_sizes)
        residual_scale = 1.0 / (n_layers ** 0.5) if residual_scaling else 1.0

        self.task_type = _validate_task_type(task_type)
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_checkpoint = gradient_checkpointing

        # Input embedding + output head (closed by default: nn.Embedding is a pure
        # gather, the input projection / head are tropical; set *_type="linear"
        # for the standard DSP-boundary versions).
        if self.task_type == "lm":
            self.embed = nn.Embedding(vocab_size, d_model)
            self.output_head = _make_head(head_type, d_model, vocab_size)
        else:
            self.embed = _make_input_proj(input_type, input_dim, d_model)
            self.output_head = _make_head(head_type, d_model, num_classes)

        # Build layers with alternating dilation/erosion and exponential dilation
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mode = "dilation" if i % 2 == 0 else "erosion"
            # Exponentially growing dilation, capped at max_dilation
            layer_dilation = min(2 ** i, max_dilation)
            self.layers.append(
                MorphSSMLayer(
                    d_model, kernel_size, dilation=layer_dilation,
                    mode=mode, dropout=dropout, use_soft=use_soft, tau=tau,
                    tropical_proj=tropical_proj, norm_type=norm_type,
                    group_size=group_size, residual_scale=residual_scale,
                    multi_scale=multi_scale, kernel_sizes=kernel_sizes,
                )
            )

        self.ln_final = _make_norm(norm_type, d_model)
        self.dropout = TropicalDropout(dropout) if dropout > 0 else nn.Identity()

        # GPT-2 style residual scaling: scale out_proj weights by
        # 1/sqrt(2*n_layers) at init to prevent the residual branch from
        # overwhelming the skip connection in deep models.  This eliminates
        # the "ignition delay" where the model spends 500-1000 steps at
        # random chance before learning begins.
        for layer in self.layers:
            nn.init.normal_(layer.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    def set_tau(self, tau: float) -> None:
        """Anneal the soft->hard tropical temperature across the whole model.

        Walks every sub-module: calls ``set_tau`` where defined and otherwise sets
        the ``tau`` attribute and ``use_soft = tau > 0`` (tau==0 is the hard,
        multiplier-free deploy form). Without this the morph convs / projections
        stay at their construction temperature through training (soft-anneal freeze).
        """
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "set_tau") and callable(m.set_tau):
                m.set_tau(tau)
            elif hasattr(m, "tau"):
                m.tau = tau
            if hasattr(m, "use_soft"):
                m.use_soft = tau > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_type == "lm":
            return self._forward_lm(x)
        elif self.task_type == "regression":
            return self._forward_regression(x)
        else:
            return self._forward_classification(x)

    def _run_layers(self, h: torch.Tensor) -> torch.Tensor:
        """Run through all MorphSSM layers, with optional gradient checkpointing."""
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            for layer in self.layers:
                h = checkpoint(layer, h, use_reentrant=False)
        else:
            for layer in self.layers:
                h = layer(h)
        return h

    def _forward_lm(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) -> (batch, seq_len, vocab_size)"""
        h = self.embed(x)
        h = self.dropout(h)
        h = self._run_layers(h)
        h = self.ln_final(h)
        return self.output_head(h)

    def _forward_regression(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep regression: (batch, seq_len, input_dim) -> (batch, seq_len, output_dim).

        Used for next-step prediction (generation). Unlike classification,
        does NOT mean-pool over the sequence — produces per-timestep outputs.
        """
        if x.dim() == 2:
            if not x.is_floating_point():
                x = x.float()
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.embed.in_features:
            B = x.shape[0]
            x = x.reshape(B, -1, self.embed.in_features)
        h = self.embed(x)
        h = self.dropout(h)
        h = self._run_layers(h)
        h = self.ln_final(h)
        return self.output_head(h)  # (B, T, output_dim) — no mean pooling

    def _forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, input_dim) or (batch, features) -> (batch, num_classes)"""
        if x.dim() == 2:
            if not x.is_floating_point():
                x = x.float()
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.embed.in_features:
            B = x.shape[0]
            x = x.reshape(B, -1, self.embed.in_features)
        h = self.embed(x)
        h = self.dropout(h)
        h = self._run_layers(h)
        h = self.ln_final(h)
        h = h.mean(dim=1)
        return self.output_head(h)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CE for lm/classification, MSE for regression (no GenerationLoss)."""
        return _default_compute_loss(self, x, y)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def hardware_cost(self, batch_size: int = 1, seq_len: int = 1) -> HardwareCostProfile:
        """Morphological conv: comparisons + additions only. Linear proj uses DSP.

        For fully-closed config (use_soft=False + tropical_proj=True): the
        sequence processing core uses ZERO DSP blocks — only LUTs + carry chains.
        """
        profile = HardwareCostProfile()

        # Detect fully-closed config: no float multiplications in sequence core
        fully_closed = (not self.layers[0].morph_conv.use_soft
                        if self.layers else False)
        # Check if out_proj is a TropicalDilationModule (tropical_proj=True)
        proj_is_tropical = (isinstance(self.layers[0].out_proj, TropicalDilationModule)
                            if self.layers else False)
        sequence_core_closed = fully_closed and proj_is_tropical

        for layer in self.layers:
            conv = layer.morph_conv
            K = conv.kernel_size
            C = conv.channels
            n_out = batch_size * seq_len * C

            # Per output element: K additions (weight[k] + input[t-k*d]) + K-1 comparisons
            profile.additions += n_out * K
            profile.comparisons += n_out * (K - 1)

            # Channel mixing: nn.Linear uses multiplications; TropicalDilationModule does not
            proj = layer.out_proj
            if proj_is_tropical:
                # Tropical out_proj: max-plus over (weight + input) — comparisons + additions
                if hasattr(proj, 'weight'):
                    in_f = proj.weight.shape[1]
                    profile.additions += batch_size * seq_len * in_f
                    profile.comparisons += batch_size * seq_len * (in_f - 1)
            else:
                profile.multiplications += batch_size * seq_len * proj.in_features * proj.out_features
                profile.additions += batch_size * seq_len * proj.in_features * proj.out_features

        # Input embedding (nn.Linear — float MACs, small overhead)
        if hasattr(self.embed, 'in_features'):
            profile.multiplications += batch_size * seq_len * self.embed.in_features * self.embed.out_features
        # Output head (single mean-pooled vector — negligible)
        profile.multiplications += batch_size * self.output_head.in_features * self.output_head.out_features

        # FPGA resource estimates (Xilinx UltraScale+); 1 FP32 multiply = 3 DSP blocks
        if sequence_core_closed:
            # Only embedding + output head use DSPs (tiny fraction)
            embed_muls = (batch_size * seq_len * self.embed.in_features * self.embed.out_features
                          if hasattr(self.embed, 'in_features') else 0)
            head_muls = batch_size * self.output_head.in_features * self.output_head.out_features
            profile.dsp_blocks_required = (embed_muls + head_muls) * 3
        else:
            profile.dsp_blocks_required = profile.multiplications * 3

        # LUT estimate: 6 LUTs per 32-bit comparison, additions use carry chain (~free)
        profile.lut_estimate = profile.comparisons * 6 + profile.additions // 8

        # BRAM: weight storage (4 bytes/param, BRAM18 = 2304 bytes)
        total_params = sum(p.numel() for p in self.parameters())
        profile.bram_kb = total_params * 4 // 1024

        # Throughput at 250 MHz: pipeline depth = kernel_size clock cycles per output
        pipeline_depth = self.layers[0].morph_conv.kernel_size if self.layers else 7
        profile.estimated_clock_mhz = 250
        profile.estimated_throughput_samples_sec = 250e6 / pipeline_depth

        return profile

    get_hardware_cost_profile = hardware_cost


# ---------------------------------------------------------------------------
# Morphological MLP
# ---------------------------------------------------------------------------


class MorphMLP(FpgaAccelerable, nn.Module):
    """MLP with alternating dilation (max-plus) and erosion (min-plus) layers.

    The alternation of max and min IS the nonlinearity — no separate
    activation function needed.  This produces tropical rational maps,
    which are universal approximators.

    Architecture:
        input -> [dilation -> batch_shift -> erosion -> batch_shift] x N -> linear head

    Args:
        d_model: hidden dimension (default 128)
        n_layers: number of tropical layers (default 4, should be even)
        dropout: dropout rate (default 0.0)
        use_soft: soft relaxation for training (default True)
        tau: temperature for soft max/min (default 1.0)
        input_dim: input dimension (default 784)
        num_classes: output dimension (default 10)
        task_type: "classification" (default)
    """

    def __init__(self, *, d_model: int = 128, n_layers: int = 4,
                 dropout: float = 0.0, use_soft: bool = True, tau: float = 1.0,
                 input_dim: int = 784, num_classes: int = 10,
                 task_type: str = "classification"):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task_type = _validate_task_type(task_type)

        # Build alternating dilation/erosion layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = self.input_dim
        for i in range(n_layers):
            if i % 2 == 0:
                # Dilation (max-plus)
                self.layers.append(
                    TropicalDilationModule(
                        in_dim, self.d_model,
                        use_softmax=use_soft, tau=tau,
                    )
                )
            else:
                # Erosion (min-plus)
                self.layers.append(
                    TropicalLinearModule(
                        in_dim, self.d_model,
                        use_softmin=use_soft, tau=tau,
                    )
                )
            self.norms.append(TropicalBatchShift())
            self.dropouts.append(TropicalDropout(dropout) if dropout > 0 else nn.Identity())
            in_dim = self.d_model

        # Standard linear output head
        self.head = _make_head("tropical", self.d_model, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.float()
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
            x = layer(x)
            x = norm(x)
            x = drop(x)

        if self.task_type == "regression":
            return self.head(x)
        return self.head(x)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _default_compute_loss(self, x, y)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def hardware_cost(self, batch_size: int = 1, seq_len: int = 1) -> HardwareCostProfile:
        """Morphological MLP: zero multiplications in tropical layers."""
        n_samples = batch_size * seq_len
        total_additions = 0
        total_comparisons = 0

        in_dim = self.input_dim
        for layer in self.layers:
            out_dim = layer.out_features
            # Each output: in_dim additions (W + x) + (in_dim - 1) comparisons (max/min)
            total_additions += n_samples * out_dim * in_dim
            total_comparisons += n_samples * out_dim * (in_dim - 1)
            in_dim = out_dim

        # Linear head uses multiplications
        head_muls = n_samples * self.d_model * self.num_classes

        return HardwareCostProfile(
            multiplications=head_muls,
            additions=total_additions + head_muls,
            comparisons=total_comparisons,
            backward_additions=total_additions,
            backward_comparisons=total_comparisons,
            backward_multiplications=head_muls,
            dsp_blocks_required=self.d_model * self.num_classes // 16,
            lut_estimate=total_comparisons // 4,
        )

    get_hardware_cost_profile = hardware_cost


# ---------------------------------------------------------------------------
# Tropical Transformer
# ---------------------------------------------------------------------------


class TropicalTransformer(FpgaAccelerable, nn.Module):
    """Transformer built entirely from tropical semiring operations.

    Attention uses max-plus scoring + argmax selection.
    FFN uses TropicalDilation -> TropicalLinear (tropical rational map).
    All operations in the transformer blocks are comparisons + additions —
    zero multiplications.

    Args:
        d_model: model dimension (default 256)
        n_heads: number of attention heads (default 4)
        n_layers: number of transformer blocks (default 6)
        d_ff: FFN inner dim; if None -> d_model * 4 (default None)
        dropout: dropout rate (default 0.1)
        tau: temperature for soft attention; 0.0 is hard tropical (default 0.0)
        use_shift_attn: use TropicalShiftAttention (no Q/K/V projections)
        max_seq_len: max sequence length for shift-attention tables (default 1024)
        score_variant: "dot_product" (default) or "l_inf"
        shift_mode: "free" (default) or "rope_relative"
        task_type: "classification" (default) or "lm"
        vocab_size: vocabulary size when task_type == "lm" (default 32000)
        input_dim: input feature dim otherwise (default 1)
        num_classes: output classes otherwise (default 10)
    """

    def __init__(self, *, d_model: int = 256, n_heads: int = 4, n_layers: int = 6,
                 d_ff=None, dropout: float = 0.1, tau: float = 0.0,
                 use_shift_attn: bool = False, max_seq_len: int = 1024,
                 score_variant: str = "dot_product", shift_mode: str = "free",
                 norm_type: str = "tropical",
                 head_type: str = "tropical", input_type: str = "tropical",
                 task_type: str = "classification",
                 vocab_size: int = 32000, input_dim: int = 1,
                 num_classes: int = 10):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.task_type = _validate_task_type(task_type)
        self.d_model = d_model
        self.num_classes = num_classes

        # Input embedding + head (closed by default; nn.Embedding is a gather).
        if self.task_type == "lm":
            self.embed = nn.Embedding(vocab_size, d_model)
            self.output_head = _make_head(head_type, d_model, vocab_size)
        else:
            self.embed = _make_input_proj(input_type, input_dim, d_model)
            self.output_head = _make_head(head_type, d_model, num_classes)

        # Tropical transformer blocks
        self.blocks = nn.ModuleList([
            TropicalTransformerBlock(
                d_model, n_heads, d_ff, dropout, tau=tau,
                use_shift_attn=use_shift_attn,
                max_seq_len=max_seq_len,
                score_variant=score_variant,
                shift_mode=shift_mode,
                norm_type=norm_type,
            )
            for _ in range(n_layers)
        ])
        self.ln_final = _make_norm(norm_type, d_model)

    def set_tau(self, tau: float) -> None:
        """Anneal soft <-> hard for all attention modules."""
        for block in self.blocks:
            block.set_tau(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return self._forward_classification(x)
        else:
            return self._forward_lm(x)

    def _forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, causal_mask=False)
        h = self.ln_final(h)
        h = h.mean(dim=1)  # global average pooling
        return self.output_head(h)

    def _forward_lm(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h, causal_mask=True)
        h = self.ln_final(h)
        return self.output_head(h)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _default_compute_loss(self, x, y)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Mixture of Tropical Experts
# ---------------------------------------------------------------------------


class MoTEModel(FpgaAccelerable, nn.Module):
    """Mixture of Tropical Experts model.

    Architecture:
        embed -> [shared_pre MorphSSM layers] -> MoTEBlock -> [shared_post layers] -> head

    Shared layers handle sequence mixing (temporal context via morphological
    convolutions). The MoTE block handles channel mixing (capacity /
    specialization via expert routing). Experts are pointwise (token-level)
    while temporal context is shared.

    Args:
        d_model: model dimension (default 128)
        n_experts: number of experts (default 8)
        expert_type: "transformer" (default) or "morph"
        expert_layers: layers per expert (default 1)
        d_ff: expert FFN inner dim; if None -> d_model * 4 (default None)
        n_heads: attention heads in transformer experts (default 4)
        router_dim: routing projection dim (default 32)
        n_shared_pre: shared pre-layers (default 4)
        n_shared_post: shared post-layers (default 2)
        kernel_size: morph-conv kernel size (default 7)
        use_soft: soft relaxation for training (default True)
        tau: temperature for soft max/min (default 1.0)
        dropout: dropout rate (default 0.1)
        max_dilation: cap on the exponential per-layer dilation (default 2048)
        capacity_factor: per-expert token capacity factor (default 1.5)
        use_shift_attn: use TropicalShiftAttention in transformer experts
        score_variant: "dot_product" (default) or "l_inf"
        out_proj_type: shared-layer projection: "linear" (default), "tropical", "identity"
        max_seq_len: max sequence length for shift-attention tables (default 1024)
        aux_loss_weight: weight on the load-balancing aux loss (default 0.01)
        task_type: "classification" | "lm" | "regression" (default "classification")
        vocab_size: vocabulary size when task_type == "lm" (default 32000)
        input_dim: input feature dim otherwise (default 1)
        num_classes: output classes / regression output dim otherwise (default 10)
    """

    def __init__(self, *, d_model: int = 128, n_experts: int = 8,
                 expert_type: str = "transformer", expert_layers: int = 1,
                 d_ff=None, n_heads: int = 4, router_dim: int = 32,
                 n_shared_pre: int = 4, n_shared_post: int = 2,
                 kernel_size: int = 7, use_soft: bool = True, tau: float = 1.0,
                 dropout: float = 0.1, max_dilation: int = 2048,
                 capacity_factor: float = 1.5, use_shift_attn: bool = False,
                 score_variant: str = "dot_product", out_proj_type: str = "tropical",
                 max_seq_len: int = 1024, aux_loss_weight: float = 0.01,
                 head_type: str = "tropical", input_type: str = "tropical",
                 task_type: str = "classification",
                 vocab_size: int = 32000, input_dim: int = 1,
                 num_classes: int = 10):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.aux_loss_weight = aux_loss_weight
        self.task_type = _validate_task_type(task_type)
        self.d_model = d_model
        self.num_classes = num_classes

        # Input embedding + head (closed by default; nn.Embedding is a gather).
        if self.task_type == "lm":
            self.embed = nn.Embedding(vocab_size, d_model)
            self.output_head = _make_head(head_type, d_model, vocab_size)
        else:
            self.embed = _make_input_proj(input_type, input_dim, d_model)
            self.output_head = _make_head(head_type, d_model, num_classes)

        # Shared pre-layers (sequence mixing)
        self.pre_layers = nn.ModuleList()
        for i in range(n_shared_pre):
            mode = "dilation" if i % 2 == 0 else "erosion"
            layer_dilation = min(2 ** i, max_dilation)
            self.pre_layers.append(
                MorphSSMLayer(
                    d_model, kernel_size, dilation=layer_dilation,
                    mode=mode, dropout=dropout, use_soft=use_soft, tau=tau,
                    out_proj_type=out_proj_type,
                )
            )

        # MoTE block (channel mixing via experts)
        self.mote_block = MoTEBlock(
            d_model=d_model,
            n_experts=n_experts,
            expert_type=expert_type,
            expert_layers=expert_layers,
            d_ff=d_ff,
            n_heads=n_heads,
            kernel_size=kernel_size,
            dropout=dropout,
            d_router=router_dim,
            use_soft=use_soft,
            tau=tau,
            capacity_factor=capacity_factor,
            use_shift_attn=use_shift_attn,
            max_seq_len=max_seq_len,
            score_variant=score_variant,
        )

        # Shared post-layers (sequence mixing)
        self.post_layers = nn.ModuleList()
        for i in range(n_shared_post):
            mode = "dilation" if (n_shared_pre + i) % 2 == 0 else "erosion"
            layer_dilation = min(2 ** (n_shared_pre + i), max_dilation)
            self.post_layers.append(
                MorphSSMLayer(
                    d_model, kernel_size, dilation=layer_dilation,
                    mode=mode, dropout=dropout, use_soft=use_soft, tau=tau,
                    out_proj_type=out_proj_type,
                )
            )

        # Closed final norm; the pre/post MorphSSMLayers and the MoTEBlock's
        # transformer/morph experts already default to TropicalLayerNorm (N1).
        self.ln_final = _make_norm("tropical", d_model)
        self.dropout = TropicalDropout(dropout) if dropout > 0 else nn.Identity()

        # GPT-2 style init scaling for deep residual paths
        n_total_layers = n_shared_pre + n_shared_post
        if n_total_layers > 0:
            for layer_list in [self.pre_layers, self.post_layers]:
                for layer in layer_list:
                    if hasattr(layer.out_proj, 'weight'):
                        nn.init.normal_(layer.out_proj.weight, std=0.02 / math.sqrt(2 * n_total_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return self._forward_classification(x)
        elif self.task_type == "lm":
            return self._forward_lm(x)
        else:
            return self._forward_regression(x)

    def _forward_core(self, h: torch.Tensor):
        """Shared forward through pre-layers, MoTE, post-layers."""
        h = self.dropout(h)

        for layer in self.pre_layers:
            h = layer(h)

        h, aux_loss = self.mote_block(h)

        for layer in self.post_layers:
            h = layer(h)

        h = self.ln_final(h)
        return h, aux_loss

    def _forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if not x.is_floating_point():
                x = x.float()
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.embed.in_features:
            B = x.shape[0]
            x = x.reshape(B, -1, self.embed.in_features)
        h = self.embed(x)
        h, self._last_aux_loss = self._forward_core(h)
        h = h.mean(dim=1)
        return self.output_head(h)

    def _forward_lm(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h, self._last_aux_loss = self._forward_core(h)
        return self.output_head(h)

    def _forward_regression(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if not x.is_floating_point():
                x = x.float()
            x = x.unsqueeze(-1)
        h = self.embed(x)
        h, self._last_aux_loss = self._forward_core(h)
        return self.output_head(h)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Task loss + auxiliary load-balancing loss (no GenerationLoss)."""
        logits = self.forward(x)
        aux_loss = getattr(self, "_last_aux_loss",
                           torch.zeros((), device=next(self.parameters()).device))

        if self.task_type in ("classification", "lm"):
            if logits.dim() == 3:
                logits = logits.reshape(-1, logits.size(-1))
                y = y.reshape(-1)
            task_loss = classification_loss(logits, y, getattr(self, "loss_type", "cross_entropy"))
        elif self.task_type == "regression":
            if logits.shape == y.shape:
                task_loss = F.mse_loss(logits, y.float())
            else:
                task_loss = F.mse_loss(logits.squeeze(-1), y.float())
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        return task_loss + self.aux_loss_weight * aux_loss

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_expert_utilization(self) -> dict:
        """Expert-utilization stats for logging — UNIMPLEMENTED, returns ``{}``.

        Tracking per-expert routing load needs forward hooks on the router/experts,
        which are not wired up; this returns an empty dict so callers must treat it as
        "no stats available" rather than expecting populated counts.
        """
        return {}

    def hardware_cost(self, batch_size: int = 1, seq_len: int = 1) -> HardwareCostProfile:
        """MoTE hardware cost: router + (1/n_experts) * per-expert cost."""
        profile = HardwareCostProfile()
        n_tokens = batch_size * seq_len
        n_experts = self.mote_block.n_experts

        # Router cost: d_model * d_router additions + comparisons
        router = self.mote_block.router
        profile.additions += n_tokens * router.d_model * router.d_router
        profile.comparisons += n_tokens * router.d_router * (router.d_model - 1)
        # Hash: d_router additions + 1 modulo per token
        profile.additions += n_tokens * router.d_router

        # Shared pre/post layers (same as MorphSSM)
        for layer_list in [self.pre_layers, self.post_layers]:
            for layer in layer_list:
                conv = layer.morph_conv
                K = conv.kernel_size
                C = conv.channels
                n_out = n_tokens * C
                profile.additions += n_out * K
                profile.comparisons += n_out * (K - 1)
                if isinstance(layer.out_proj, nn.Linear):
                    profile.multiplications += n_tokens * layer.out_proj.in_features * layer.out_proj.out_features

        # Expert cost: only 1/n_experts tokens per expert (sparse)
        tokens_per_expert = max(n_tokens // n_experts, 1)
        # Each expert processes tokens_per_expert tokens
        for expert_layers in self.mote_block.experts:
            for layer in expert_layers:
                if isinstance(layer, TropicalTransformerBlock):
                    # Attention + FFN cost for tokens_per_expert tokens
                    d = self.d_model
                    profile.additions += tokens_per_expert * d * 4  # Q,K,V,O projections (tropical)
                    profile.comparisons += tokens_per_expert * d * 4
                break  # All experts have same architecture

        # Embedding + output head
        if hasattr(self.embed, 'in_features'):
            profile.multiplications += n_tokens * self.embed.in_features * self.embed.out_features
        profile.multiplications += batch_size * self.output_head.in_features * self.output_head.out_features

        return profile

    get_hardware_cost_profile = hardware_cost


# Required alias.
MoTE = MoTEModel


# ---------------------------------------------------------------------------
# TropicalCNN — a fully multiplier-free convolutional network (vision)
# ---------------------------------------------------------------------------


class TropicalCNN(FpgaAccelerable, nn.Module):
    """A fully multiplier-free CNN: morphological conv + max-pool + tropical head.

    Every operation is add/compare/shift only — morphological ``TropicalConv2d``
    (max/min-plus), ``TropicalLayerNorm2d`` (subtract-mean + power-of-2 shift),
    ``tropical_leaky_relu`` (power-of-2 leak), ``TropicalMaxPool2d`` (compare),
    global max-pool (compare), and a ``TropicalDilationModule`` head (max-plus).
    There is no ``nn.Conv2d``, no ``nn.LayerNorm``, no dense matmul anywhere; the
    whole forward and backward pass synthesizes to zero DSP blocks. Verify with
    :func:`tikitorch.audit.assert_closed`.

    Args:
        in_channels: input image channels (1 for MNIST, 3 for CIFAR).
        num_classes: number of output classes.
        channels: width of each conv stage (one ``MorphConvBlock`` per entry).
        kernel_size: square kernel size for every conv.
        mode: base semiring of the first conv block; with ``alternate=True`` the
            blocks alternate ``dilation``/``erosion`` (opening/closing structure,
            which is markedly more expressive than an all-dilation stack).
        alternate: alternate dilation/erosion across blocks (default True).
        pool_grid: spatial size the final feature map is adaptive-max-pooled to
            before the head. ``g>1`` keeps coarse spatial layout (important for
            digits/objects) by flattening a ``g x g`` grid; ``g=1`` is a pure
            global max-pool (location-invariant). Default 4.
        head_type: ``"tropical"`` (fully closed, default — a compositional
            dilation∘erosion projection, a universal piecewise-linear map) or
            ``"linear"`` (standard ``nn.Linear`` head, the declared DSP boundary).
        task_type: ``"classification"`` only.
    """

    def __init__(self, *, in_channels: int = 1, num_classes: int = 10,
                 channels=(16, 32, 64), kernel_size: int = 3, mode: str = "dilation",
                 alternate: bool = True, pool_grid: int = 4,
                 head_type: str = "tropical", task_type: str = "classification",
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        assert task_type == "classification", "TropicalCNN is a classifier"
        assert head_type in ("tropical", "linear"), head_type
        self.task_type = _validate_task_type(task_type)
        self.num_classes = num_classes
        self.head_type = head_type
        self.pool_grid = pool_grid

        blocks = []
        c_in = in_channels
        other = "erosion" if mode == "dilation" else "dilation"
        for i, c_out in enumerate(channels):
            block_mode = mode if (not alternate or i % 2 == 0) else other
            blocks.append(MorphConvBlock(c_in, c_out, kernel_size, mode=block_mode,
                                         pool=True, use_soft=use_soft, tau=tau))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        if pool_grid > 1:
            self.spatial_pool = TropicalAdaptiveMaxPool2d((pool_grid, pool_grid))
            head_in = c_in * pool_grid * pool_grid
        else:
            self.spatial_pool = TropicalGlobalMaxPool()
            head_in = c_in
        # Closed top-k tropical head by default (best closed head in the sweep);
        # "linear" is the declared DSP boundary.
        self.head = _make_head(head_type, head_in, num_classes)

    def set_tau(self, tau: float) -> None:
        """Anneal every conv block's (and the head's) soft temperature -> hard at 0."""
        for block in self.blocks:
            block.set_tau(tau)
        if hasattr(self.head, "set_tau"):
            self.head.set_tau(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            x = x.float()
        if x.dim() == 3:                       # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        x = self.spatial_pool(x)               # (B, C, g, g) or (B, C) if g==1
        if x.dim() == 4:
            x = x.flatten(1)                   # (B, C*g*g) — keeps coarse layout
        return self.head(x)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _default_compute_loss(self, x, y)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Shared loss helper
# ---------------------------------------------------------------------------


def classification_loss(logits: torch.Tensor, y: torch.Tensor,
                        loss_type: str = "cross_entropy") -> torch.Tensor:
    """Dispatch a classification/LM loss by ``loss_type``.

    Selects how the gradient that conditions training is produced, trading
    accuracy against hardware closure:

      * ``"cross_entropy"`` — ``F.cross_entropy`` (real softmax; uses multiplies).
        The float reference ceiling.
      * ``"perceptron"`` — :func:`~tikitorch.functional.tropical_perceptron_loss`,
        the multiplier-free 2-hot gradient the F2 silicon kernel actually emits.
        The honest silicon baseline.
      * ``"tropical_ce"`` — :func:`~tikitorch.functional.tropical_cross_entropy`
        with a host-side exact softmax (zero-DSP kernel + tiny host softmax; the
        silicon-validated +3.2pp path).
      * ``"tropical_ce_lut"`` — the same, with the fully multiplier-free
        ``shift_softmax`` (on-chip conditioning, whole loop multiply-free).
    """
    if loss_type == "cross_entropy":
        return F.cross_entropy(logits, y)
    if loss_type == "perceptron":
        return tropical_perceptron_loss(logits, y)
    if loss_type == "tropical_ce":
        return tropical_cross_entropy(logits, y, softmax="exact")
    if loss_type == "tropical_ce_lut":
        # normalize="shift" is the on-chip power-of-2 normalization; the default
        # "exact" uses a host reciprocal, contradicting this label's whole-loop
        # multiplier-free promise. Safe ONLY because TropicalCrossEntropyFn
        # scatters sum(p) (not 1.0) at the target — an unbalanced gradient with
        # sum(p) < 1 collapses sign-trained models (see functional.py and
        # tests/test_functional.py::test_lut_shift_ce_grad_rows_sum_to_zero).
        return tropical_cross_entropy(logits, y, softmax="lut", normalize="shift")
    raise ValueError(
        f"Unknown loss_type {loss_type!r}; expected one of "
        "'cross_entropy', 'perceptron', 'tropical_ce', 'tropical_ce_lut'."
    )


def _default_compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Classification/LM loss (dispatched by ``model.loss_type``), MSE for regression.

    The shared default loss body. Used by every model that does
    not override ``compute_loss`` with extra terms. The classification/LM loss is
    selected by ``model.loss_type`` (default ``"cross_entropy"``); set it to
    ``"perceptron"`` for the silicon baseline or ``"tropical_ce"`` /
    ``"tropical_ce_lut"`` for the closure-faithful recovery (see
    :func:`classification_loss`).
    """
    logits = model.forward(x)
    if model.task_type in ("classification", "lm"):
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            y = y.reshape(-1)
        return classification_loss(logits, y, getattr(model, "loss_type", "cross_entropy"))
    else:  # regression
        if logits.shape == y.shape:
            return F.mse_loss(logits, y.float())
        return F.mse_loss(logits.squeeze(-1), y.float())
