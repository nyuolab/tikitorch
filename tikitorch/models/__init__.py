"""tikitorch.models — ready-to-use tropical architectures (a la torchvision/timm).

Every model subclasses
``torch.nn.Module`` directly (no abstract base class), takes plain
keyword arguments (no ``config`` dict), and composes the tropical building
blocks from :mod:`tikitorch.nn`. The implementations originated in the research
codebase, but closure and hardware properties remain configuration- and
recipe-specific; a model class name is not itself a certificate.

Public models:

    TropicalMLP           — alternating-min/max tropical MLP (classification)
    TropicalSequenceModel — tropical recurrence for sequence classification
    MorphSSM              — morphological SSM (dilation/erosion conv) LM/classifier
    MorphMLP              — alternating dilation/erosion MLP
    TropicalTransformer   — transformer with tropical attention + tropical FFN
    MoTEModel (alias MoTE)— Mixture of Tropical Experts
    TropicalForwardForward — layer-local learner without end-to-end backpropagation
    DyadicForwardForward  — the same Forward-Forward rule on the dyadic ±2^k
                            substrate (dense/additive -> much higher accuracy)

The ``task_type`` / loss convention:
    Every model takes a ``task_type`` keyword (``"classification"``, ``"lm"``,
    or ``"regression"`` — the subset each supports is documented per class) and
    exposes a concrete ``compute_loss(x, y)`` method. ``compute_loss`` is the
    standard task loss (cross-entropy for
    classification / lm, MSE for regression). It is *not* enforced by any base
    class — it is provided because the training loops rely on it. MoTEModel
    additionally adds its load-balancing ``aux_loss`` term.

    The Forward-Forward models (``TropicalForwardForward`` / ``DyadicForwardForward``)
    are the **exception**: they train *forward-only* (a local per-layer goodness
    objective, no global backward) via ``forward_forward_fit`` / ``make_trainer``,
    not ``loss.backward()``. Their ``forward(x)`` returns goodness-based class logits
    so they are still drop-in classifiers, and ``compute_loss`` is provided only as a
    *non-FF* global-backprop baseline. See :mod:`tikitorch.forward_forward` and
    ``manuscript/results/forward_forward/TROPICAL_FORWARD_FORWARD.md``.

Each model also exposes a cheap ``param_count`` property and, where applicable,
an optional ``hardware_cost(...)`` method returning a
:class:`tikitorch.HardwareCostProfile` (with ``get_hardware_cost_profile`` kept
as an alias). These are optional conveniences, never required.
"""
from __future__ import annotations

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

    ``"tropical"`` is a :class:`TropicalTopKModule`, a top-k max-plus
    aggregation. ``"linear"`` is a standard ``nn.Linear`` head and therefore an
    ordinary dense boundary. This choice alone does not close a full recipe.
    """
    if head_type == "tropical":
        return TropicalTopKModule(in_dim, out_dim, k=min(topk_k, in_dim))
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    raise ValueError(f"head_type must be 'tropical' or 'linear', got {head_type!r}")


def _make_input_proj(input_type: str, in_dim: int, out_dim: int):
    """Build an input projection (continuous-feature -> d_model).

    ``"tropical"`` is :class:`TropicalQuantizedEmbed` --
    quantize each continuous feature and gather a learned per-bin vector. This is
    the *expressive* closed input: a tropical add-only projection of a scalar is
    rank-degenerate (can shift but not scale), which stalls training, whereas the
    quantize+gather embedding avoids that scalar-rank limitation. Assumes inputs
    in [0, 1] (normalize first). ``"linear"`` is the ``nn.Linear`` boundary. (LM
    models embed tokens with ``nn.Embedding``.)
    """
    if input_type == "tropical":
        return TropicalQuantizedEmbed(in_dim, out_dim, n_bins=32, lo=0.0, hi=1.0)
    if input_type == "linear":
        return nn.Linear(in_dim, out_dim)
    raise ValueError(f"input_type must be 'tropical' or 'linear', got {input_type!r}")


def _accumulate_projection_cost(
    profile: HardwareCostProfile, module: nn.Module, n_items: int
) -> None:
    """Accumulate a projection's cost from the module that will actually run.

    These estimates describe the hard/export implementation.  In particular,
    tropical projections are compare/add networks, quantized embeddings are
    table lookups, and only an ordinary ``nn.Linear`` contributes DSP-bearing
    multiplications.  Keeping this dispatch in one place prevents a tropical
    default head from being accidentally costed as a dense head.
    """
    n_items = int(n_items)
    if isinstance(module, nn.Identity):
        return
    if isinstance(module, nn.Embedding):
        profile.table_lookups += n_items
        profile.backward_table_lookups += n_items
        return
    if isinstance(module, TropicalQuantizedEmbed):
        lookups = n_items * module.in_features
        additions = n_items * module.out_features * max(module.in_features - 1, 0)
        profile.table_lookups += lookups
        profile.backward_table_lookups += lookups
        profile.additions += additions
        profile.backward_additions += additions
        return

    if isinstance(module, TropicalTopKModule):
        combinations = n_items * module.out_features * module.in_features
        reductions = n_items * module.out_features * max(module.k - 1, 0)
        bias_adds = n_items * module.out_features
        comparisons = n_items * module.out_features * max(module.in_features - 1, 0)
        profile.additions += combinations + reductions + bias_adds
        profile.comparisons += comparisons
        profile.backward_additions += combinations + reductions
        return

    if isinstance(module, (TropicalDilationModule, TropicalLinearModule)):
        combinations = n_items * module.out_features * module.in_features
        bias_adds = n_items * module.out_features
        comparisons = n_items * module.out_features * max(module.in_features - 1, 0)
        profile.additions += combinations + bias_adds
        profile.comparisons += comparisons
        profile.backward_additions += combinations
        return

    if isinstance(module, CompositionalTropicalProjection):
        first = n_items * module.d_intermediate * module.in_features
        second = n_items * module.out_features * module.d_intermediate
        bias_adds = n_items * (module.d_intermediate + module.out_features)
        comparisons = n_items * (
            module.d_intermediate * max(module.in_features - 1, 0)
            + module.out_features * max(module.d_intermediate - 1, 0)
        )
        profile.additions += first + second + bias_adds
        profile.comparisons += comparisons
        profile.backward_additions += first + second
        return

    if isinstance(module, nn.Linear):
        macs = n_items * module.in_features * module.out_features
        # DyadicLinear is an nn.Linear subclass whose exported weights are shifts.
        if getattr(module, "_zero_dsp_layer", False):
            profile.additions += macs
            profile.backward_additions += macs
            # The input adjoint is shift/add, but an ordinary autograd weight
            # gradient is an outer product of two arbitrary tensors. Only a
            # specialized sign/local update can remove this training boundary.
            profile.backward_multiplications += macs
        else:
            profile.multiplications += macs
            profile.additions += macs
            profile.backward_multiplications += macs
            profile.backward_additions += macs
        return

    raise TypeError(
        f"no hardware-cost rule for projection module {type(module).__name__}"
    )


def _accumulate_morph_conv_cost(
    profile: HardwareCostProfile, conv: nn.Module, n_items: int
) -> None:
    """Accumulate hard morphological convolution compare/add counts."""
    if hasattr(conv, "convs"):
        for scale in conv.convs:
            _accumulate_morph_conv_cost(profile, scale, n_items)
        profile.comparisons += n_items * conv.channels * max(len(conv.convs) - 1, 0)
        return
    taps = conv.kernel_size * getattr(conv, "group_size", 1)
    outputs = n_items * conv.channels
    profile.additions += outputs * taps
    profile.comparisons += outputs * max(taps - 1, 0)


def _is_power_of_two_scalar(value: float) -> bool:
    """Return whether a finite nonzero scalar is exactly a signed power of two."""
    try:
        mantissa, _ = math.frexp(abs(float(value)))
    except (TypeError, ValueError):
        return False
    return math.isfinite(float(value)) and mantissa == 0.5


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
    "TropicalForwardForward",
    "DyadicForwardForward",
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

    def set_tau(self, tau: float) -> None:
        """Set an exact soft/hard temperature; ``0`` selects hard operators.

        ``set_temperature`` retains the training-time ``tau_min`` floor. This
        method is the deployment/auditing hook and therefore must not clamp zero.
        """
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.temperature = float(tau)
        self.use_softmin = tau > 0
        for layer in self.layers:
            layer.set_tau(float(tau))
        if hasattr(self.head, "set_tau"):
            self.head.set_tau(float(tau))

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
        """Estimate the hard/export cost of this model's actual module types.

        Forward: each tropical linear does additions + comparisons (argmin).
        Backward: sparse gradient routing (additions only, no multiplies).
        A dense opt-in head contributes multiplications; the default tropical
        head does not.
        """
        n_samples = batch_size * seq_len
        profile = HardwareCostProfile()
        for layer in self.layers:
            _accumulate_projection_cost(profile, layer, n_samples)
        _accumulate_projection_cost(profile, self.head, n_samples)

        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        activation_bytes = n_samples * self.d_model * len(self.layers) * 4  # float32
        profile.memory_reads_bytes = param_bytes + activation_bytes
        profile.memory_writes_bytes = activation_bytes
        profile.dsp_blocks_required = (
            math.ceil(profile.multiplications / max(n_samples * 16, 1))
            if profile.multiplications else 0
        )
        profile.lut_estimate = profile.comparisons // 4
        profile.bram_kb = param_bytes // 1024
        return profile

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

    def set_tau(self, tau: float) -> None:
        """Set an exact soft/hard temperature; ``0`` selects hard operators."""
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.temperature = float(tau)
        self.use_softmin = tau > 0
        for wh, wx in zip(self.W_h, self.W_x):
            wh.set_tau(float(tau))
            wx.set_tau(float(tau))
        if hasattr(self.head, "set_tau"):
            self.head.set_tau(float(tau))

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
        """Estimate hard/export cost, including the configured input and head."""
        if seq_len is None:
            seq_len = self.seq_len

        n_samples = batch_size
        n_tokens = n_samples * seq_len
        profile = HardwareCostProfile()
        _accumulate_projection_cost(profile, self.input_proj, n_tokens)

        # Tropical recurrence: additions + comparisons per timestep per layer
        trop_additions = 0
        trop_comparisons = 0
        for _ in range(self.n_layers):
            # W_h and W_x each: d_model * d_model additions, d_model*(d_model-1) comparisons
            per_step = 2 * self.d_model * self.d_model  # additions
            per_step_cmp = 2 * self.d_model * (self.d_model - 1) + self.d_model  # comparisons + element min
            trop_additions += n_samples * seq_len * per_step
            trop_comparisons += n_samples * seq_len * per_step_cmp

        profile.additions += trop_additions
        profile.comparisons += trop_comparisons
        profile.backward_additions += trop_additions
        _accumulate_projection_cost(profile, self.head, n_samples)

        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        profile.memory_reads_bytes = param_bytes
        profile.memory_writes_bytes = n_tokens * self.d_model * 4 * self.n_layers
        profile.dsp_blocks_required = (
            math.ceil(profile.multiplications / max(n_tokens * 16, 1))
            if profile.multiplications else 0
        )
        profile.lut_estimate = profile.comparisons // 4
        profile.bram_kb = param_bytes // 1024
        return profile

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
        tropical_proj: legacy bool projection selector (True=tropical,
            False=linear). Omit for the closure-first tropical default.
        out_proj_type: explicit channel projection selector: "tropical"
            (default), "linear", or "identity". It must agree with
            ``tropical_proj`` when both are supplied.
        norm_type: "tropical" (default), "layernorm", or another documented norm
        group_size: morph-conv channel grouping (default 1)
        max_dilation: cap on the exponential per-layer dilation (default 2048)
        multi_scale: use MultiScaleMorphConv1d per layer (default False)
        kernel_sizes (tuple[int, ...]): kernel sizes when multi_scale
            (default (3, 7, 15))
        residual_scaling: scale each transformed residual branch by
            1/sqrt(n_layers) (default False).
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
                 tropical_proj: bool | None = None,
                 out_proj_type: str | None = None,
                 norm_type: str = "tropical",
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
                    tropical_proj=tropical_proj, out_proj_type=out_proj_type,
                    norm_type=norm_type,
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
            if hasattr(layer.out_proj, "weight"):
                nn.init.normal_(
                    layer.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers)
                )

    def set_tau(self, tau: float) -> None:
        """Anneal the soft-to-hard tropical temperature across the whole model.

        Walks every sub-module: calls ``set_tau`` where defined and otherwise sets
        the ``tau`` attribute and ``use_soft = tau > 0`` (``tau==0`` selects the
        hard operator path). Without this the morph convs / projections
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
        """Return an operation-count estimate for a MorphSSM configuration.

        The estimator classifies hard morphological convolution as comparisons
        and additions and dense linear projection as multiply-bearing. It is not a
        synthesis report and does not predict routed DSP utilization.
        """
        profile = HardwareCostProfile()

        n_tokens = batch_size * seq_len
        for layer in self.layers:
            _accumulate_morph_conv_cost(profile, layer.morph_conv, n_tokens)
            _accumulate_projection_cost(profile, layer.out_proj, n_tokens)
            residual_items = n_tokens * self.d_model
            profile.additions += residual_items
            profile.backward_additions += residual_items
            if (
                layer.residual_scale != 1.0
                and not _is_power_of_two_scalar(layer.residual_scale)
            ):
                # A non-dyadic residual coefficient is a real data-path multiply.
                # Exact powers of two are implemented as shifts and stay out of
                # the multiplication/DSP count, matching the executed-path audit.
                profile.multiplications += residual_items
                profile.backward_multiplications += residual_items

        _accumulate_projection_cost(profile, self.embed, n_tokens)
        head_items = n_tokens if self.task_type in ("lm", "regression") else batch_size
        _accumulate_projection_cost(profile, self.output_head, head_items)

        profile.dsp_blocks_required = (
            math.ceil(profile.multiplications / max(n_tokens * 16, 1))
            if profile.multiplications else 0
        )

        # LUT estimate: 6 LUTs per 32-bit comparison, additions use carry chain (~free)
        profile.lut_estimate = profile.comparisons * 6 + profile.additions // 8

        # BRAM: weight storage (4 bytes/param, BRAM18 = 2304 bytes)
        total_params = sum(p.numel() for p in self.parameters())
        profile.bram_kb = total_params * 4 // 1024

        # Throughput at 250 MHz: pipeline depth = kernel_size clock cycles per output
        first_conv = self.layers[0].morph_conv if self.layers else None
        pipeline_depth = (
            max(first_conv.kernel_sizes)
            if first_conv is not None and hasattr(first_conv, "kernel_sizes")
            else first_conv.kernel_size if first_conv is not None else 7
        )
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

    def set_tau(self, tau: float) -> None:
        """Anneal every morphological layer; ``0`` selects hard operators."""
        if tau < 0:
            raise ValueError("tau must be non-negative")
        for layer in self.layers:
            layer.set_tau(float(tau))
        if hasattr(self.head, "set_tau"):
            self.head.set_tau(float(tau))

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

        profile = HardwareCostProfile(
            additions=total_additions,
            comparisons=total_comparisons,
            backward_additions=total_additions,
        )
        _accumulate_projection_cost(profile, self.head, n_samples)
        profile.dsp_blocks_required = (
            math.ceil(profile.multiplications / max(n_samples * 16, 1))
            if profile.multiplications else 0
        )
        profile.lut_estimate = profile.comparisons // 4
        return profile

    get_hardware_cost_profile = hardware_cost


# ---------------------------------------------------------------------------
# Tropical Transformer
# ---------------------------------------------------------------------------


class TropicalTransformer(FpgaAccelerable, nn.Module):
    """Transformer whose attention and FFN cores use tropical operators.

    Attention supports max-plus or distance scoring with hard value routing, and
    the FFN uses TropicalDilation -> TropicalLinear (a tropical rational map).
    The hardened core is comparison/addition-oriented, but soft ``tau>0`` paths,
    dropout, normalization, pooling, the loss, and optimizer remain part of the
    executed recipe. Use :mod:`tikitorch.audit` before making a closure claim;
    hardware resource claims require a configuration-matched build report.

    Args:
        d_model: model dimension (default 256)
        n_heads: number of attention heads (default 4)
        n_layers: number of transformer blocks (default 6)
        d_ff (int | None): FFN inner dim; if None -> d_model * 4 (default None)
        dropout: dropout rate (default 0.1)
        tau: temperature for soft attention, projections, and FFNs; 0.0 is hard
            tropical (default 0.0)
        use_shift_attn: use TropicalShiftAttention (no Q/K/V projections)
        max_seq_len: max sequence length for shift-attention tables (default 1024)
        score_variant: "dot_product" (default), "l_inf", "l1", "hilbert", or
            "hilbert_asym"
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
        """Anneal soft <-> hard for attention, projections, and FFNs."""
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
        d_ff (int | None): expert FFN inner dim; if None -> d_model * 4
            (default None)
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
        out_proj_type: shared-layer projection: "tropical" (default), "linear", "identity"
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

    def set_tau(self, tau: float) -> None:
        """Anneal every soft tropical submodule; ``0`` selects hard operators."""
        if tau < 0:
            raise ValueError("tau must be non-negative")
        for module in self.modules():
            if module is self:
                continue
            setter = getattr(module, "set_tau", None)
            if callable(setter):
                setter(float(tau))
            elif hasattr(module, "tau"):
                module.tau = float(tau)
            if hasattr(module, "use_soft"):
                module.use_soft = tau > 0

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
                _accumulate_morph_conv_cost(profile, layer.morph_conv, n_tokens)
                _accumulate_projection_cost(profile, layer.out_proj, n_tokens)

        # Expert cost: only 1/n_experts tokens per expert (sparse)
        tokens_per_expert = max(
            1, int(self.mote_block.capacity_factor * n_tokens / n_experts)
        )
        # Each expert processes its routed tokens. Count projection modules by
        # their real type rather than assuming every head/projection is dense.
        for expert_layers in self.mote_block.experts:
            for layer in expert_layers:
                if isinstance(layer, TropicalTransformerBlock):
                    for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        projection = getattr(layer.attn, projection_name, None)
                        if projection is not None:
                            _accumulate_projection_cost(
                                profile, projection, tokens_per_expert
                            )
                    _accumulate_projection_cost(
                        profile, layer.ffn_up, tokens_per_expert
                    )
                    _accumulate_projection_cost(
                        profile, layer.ffn_down, tokens_per_expert
                    )
                elif isinstance(layer, MorphSSMLayer):
                    _accumulate_morph_conv_cost(
                        profile, layer.morph_conv, tokens_per_expert
                    )
                    _accumulate_projection_cost(
                        profile, layer.out_proj, tokens_per_expert
                    )

        # Embedding + output head
        _accumulate_projection_cost(profile, self.embed, n_tokens)
        head_items = n_tokens if self.task_type in ("lm", "regression") else batch_size
        _accumulate_projection_cost(profile, self.output_head, head_items)
        profile.dsp_blocks_required = (
            math.ceil(profile.multiplications / max(n_tokens * 16, 1))
            if profile.multiplications else 0
        )
        profile.lut_estimate = profile.comparisons // 4

        return profile

    get_hardware_cost_profile = hardware_cost


# Required alias.
MoTE = MoTEModel


# ---------------------------------------------------------------------------
# TropicalCNN — a fully multiplier-free convolutional network (vision)
# ---------------------------------------------------------------------------


class TropicalCNN(FpgaAccelerable, nn.Module):
    """A morphological CNN with max-pooling and a configurable output head.

    With ``head_type="tropical"`` its main hard operators have compare/add/shift
    formulations; ``head_type="linear"`` is an ordinary dense boundary. The
    structural auditor can reject registered dense modules, but it does not prove
    the whole forward, backward, optimizer, or synthesized resource use. Audit the
    exact recipe and bind physical wording to its build reports.

    Args:
        in_channels: input image channels (1 for MNIST, 3 for CIFAR).
        num_classes: number of output classes.
        channels (tuple[int, ...]): width of each conv stage (one
            ``MorphConvBlock`` per entry).
        kernel_size: square kernel size for every conv.
        mode: base semiring of the first conv block; with ``alternate=True`` the
            blocks alternate ``dilation``/``erosion`` (opening/closing structure,
            which is markedly more expressive than an all-dilation stack).
        alternate: alternate dilation/erosion across blocks (default True).
        pool_grid: spatial size the final feature map is adaptive-max-pooled to
            before the head. ``g>1`` keeps coarse spatial layout (important for
            digits/objects) by flattening a ``g x g`` grid; ``g=1`` is a pure
            global max-pool (location-invariant). Default 4.
        head_type: ``"tropical"`` (default, a compositional dilation-erosion
            projection) or
            ``"linear"`` (standard ``nn.Linear`` head, the declared DSP boundary).
        task_type: ``"classification"`` only.
    """

    def __init__(self, *, in_channels: int = 1, num_classes: int = 10,
                 channels=(16, 32, 64), kernel_size: int = 3, mode: str = "dilation",
                 alternate: bool = True, pool_grid: int = 4,
                 head_type: str = "tropical", task_type: str = "classification",
                 use_soft: bool = False, tau: float = 0.0):
        super().__init__()
        if task_type != "classification":
            raise ValueError("TropicalCNN supports task_type='classification' only")
        if head_type not in ("tropical", "linear"):
            raise ValueError(
                f"head_type must be 'tropical' or 'linear', got {head_type!r}"
            )
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
# Tropical Forward-Forward (forward-only, backprop-free, closure-clean)
# ---------------------------------------------------------------------------


class TropicalForwardForward(FpgaAccelerable, nn.Module):
    """Tropical Forward-Forward classifier with layer-local objectives.

    Hinton's Forward-Forward on the max-plus substrate, packaged as a standard
    tikitorch model. **Unlike every other model here it is NOT trained by
    ``loss.backward()`` on a global head** — each layer is a *local* learner
    trained on a goodness contrast between positive (correct-label-overlaid) and
    negative (wrong-label-overlaid) data. This avoids end-to-end gradient transport,
    but local updates still need to be audited as a complete, concrete recipe.
    Software audit results do not establish synthesized DSP utilization. The
    mechanics live in
    :mod:`tikitorch.forward_forward`; this class wires them to the model API.

    Training (forward-only)::

        m = tt.models.TropicalForwardForward(d_model=500, n_layers=3, num_classes=10)
        m.forward_forward_fit(X_train, Y_train, epochs=20, lr_shift=4, margin=2.0)
        acc = (m(X_test).argmax(1) == Y_test).float().mean()     # drop-in classifier

    - :meth:`forward` returns **goodness-based class logits** ``(B, num_classes)``
      (overlay each candidate label, sum the per-layer goodness) — so a trained
      model is a drop-in classifier (``argmax`` = prediction).
    - :meth:`forward_forward_fit` / :meth:`forward_forward_step` /
      :meth:`make_trainer` are the forward-forward training API.
    - :meth:`compute_loss` is the cross-entropy on the goodness logits, provided
      for monitoring and as a *non-FF* global-backprop baseline; it does **not**
      implement the forward-forward rule (use ``forward_forward_*`` for that).

    Method + honest measured numbers: ``manuscript/results/forward_forward/TROPICAL_FORWARD_FORWARD.md``.
    Classification only (``task_type='classification'``).
    """

    def __init__(self, *, d_model: int = 500, n_layers: int = 3, input_dim: int = 784,
                 num_classes: int = 10, substrate: str = "maxplus", goodness: str = "topk",
                 k: int = 10, use_relu: bool = True, leaky_shift=None,
                 task_type: str = "classification", on: float = 1.0, off: float = 0.0,
                 skip_first: bool = False, loss_type: str = "cross_entropy"):
        super().__init__()
        from tikitorch.forward_forward import TropicalFFNet
        self.task_type = _validate_task_type(task_type)
        if self.task_type != "classification":
            raise ValueError("TropicalForwardForward supports task_type='classification' only")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.substrate = substrate
        self.on, self.off, self.skip_first = on, off, skip_first
        self.loss_type = loss_type
        dims = [input_dim] + [d_model] * n_layers
        self.net = TropicalFFNet(dims, num_classes=num_classes, substrate=substrate,
                                 use_relu=use_relu, leaky_shift=leaky_shift,
                                 goodness=goodness, k=k)

    @property
    def layers(self):
        return self.net.layers

    def _goodness_logits(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, num_classes)`` accumulated goodness, one column per overlaid label
        (grad-capable; eval should call under ``torch.no_grad``)."""
        from tikitorch.forward_forward import overlay_label
        x = self.net._prep(x)
        b = x.size(0)
        cols = []
        for c in range(self.num_classes):
            labels = torch.full((b,), c, device=x.device, dtype=torch.long)
            xc = overlay_label(x, labels, self.num_classes, on=self.on, off=self.off)
            cols.append(self.net.accumulated_goodness(xc, skip_first=self.skip_first))
        return torch.stack(cols, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._goodness_logits(x)

    def make_trainer(self, **kwargs):
        """A :class:`tikitorch.forward_forward.TropicalFFTrainer` over this model's
        layers (default optimizer: sign-only ``PureTropicalSGD``)."""
        from tikitorch.forward_forward import TropicalFFTrainer
        return TropicalFFTrainer(self.net, **kwargs)

    def forward_forward_step(self, x: torch.Tensor, y: torch.Tensor, trainer) -> dict:
        """One forward-only local update: overlay the correct label (positive) and a
        wrong label (negative), then run the per-layer trainer. Returns the per-layer
        loss / goodness diagnostics."""
        from tikitorch.forward_forward import overlay_label, make_negative_labels
        xp = overlay_label(x, y, self.num_classes, on=self.on, off=self.off)
        yn = make_negative_labels(y, self.num_classes)
        xn = overlay_label(x, yn, self.num_classes, on=self.on, off=self.off)
        return trainer.train_step(xp, xn)

    def forward_forward_fit(self, X: torch.Tensor, Y: torch.Tensor, *, epochs: int = 10,
                            batch: int = 100, trainer=None, shuffle: bool = True,
                            seed: int = 0, **trainer_kwargs):
        """Convenience forward-only training loop over an in-memory ``(X, Y)``.

        Builds a trainer (unless one is passed; extra kwargs go to
        :meth:`make_trainer`) and runs ``epochs`` of :meth:`forward_forward_step`.
        Returns the trainer (reuse it to continue training). Audit the exact
        trainer configuration before making a closure claim.
        """
        if trainer is None:
            trainer = self.make_trainer(**trainer_kwargs)
        gen = torch.Generator().manual_seed(seed)
        n = len(X)
        for _ in range(epochs):
            order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
            for i in range(0, n - batch + 1, batch):
                idx = order[i:i + batch]
                self.forward_forward_step(X[idx], Y[idx], trainer)
        return trainer

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on the goodness logits — for monitoring or a *non-FF*
        global-backprop baseline. NOT the forward-forward training rule (which is
        local and backprop-free; use :meth:`forward_forward_fit`)."""
        return _default_compute_loss(self, x, y)

    def set_tau(self, tau: float) -> None:
        self.net.set_tau(tau)

    @property
    def param_count(self) -> int:
        return self.net.param_count


class DyadicForwardForward(TropicalForwardForward):
    """Dyadic Forward-Forward with signed-power-of-two projection weights.

    Same forward-only local FF rule as :class:`TropicalForwardForward`, but each
    layer is a :class:`tikitorch.nn.DyadicLinear` (dense weights constrained to
    signed powers of two, ``±2^k``, which admit a shift-add export schedule)
    instead of max-plus dilation. The dense/additive substrate is a distinct
    capacity choice; accuracy and physical-resource claims remain experiment- and
    build-specific.

    Closure caveat (inherited from :class:`tikitorch.nn.DyadicLinear`): a
    shift-add forward schedule is a *static weight-value* property checked by
    ``tikitorch.fpga.dyadic_export.verify_schedule``, not by the eager op trace —
    the float sim runs a real ``addmm``, so a dyadic FF does **not** pass the
    runtime ``tt.assert_no_multiply`` (unlike the max-plus substrate, which does).
    The reference weight update is sign-oriented, while any on-fabric reverse and
    update path must be validated as its own implementation (see ``DyadicLinear``).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("substrate", "dyadic")
        if kwargs["substrate"] != "dyadic":
            raise ValueError("DyadicForwardForward requires substrate='dyadic'")
        super().__init__(**kwargs)


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
        the 2-hot gradient used by a documented F2 kernel path.
      * ``"tropical_ce"`` — :func:`~tikitorch.functional.tropical_cross_entropy`
        with a host-side exact softmax. Historical F2 experiments paired this
        conditioning rule with a zero-DSP kernel; that build-specific evidence
        does not certify this Python loss or an arbitrary training recipe.
      * ``"tropical_ce_lut"`` — the same with ``shift_softmax``; this is a
        lookup/shift-oriented conditioning path, not by itself a whole-loop or
        physical closure claim.
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
