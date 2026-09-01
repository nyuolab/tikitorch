"""tikitorch.fpga.kernels — the compiled-kernel catalog (KernelCache).

A pure-Python catalog for *precompiled* fused training-step kernels. Its primary
config lookup is O(1); model matching adds a bounded structural compatibility
check. **No compilation ever happens in the hot path.**

Two facts from the bundled HLS sources shape this module (both verified against
``tikitorch/hardware/hls/forge_kernels/``):

1. **Two distinct shape schemas.** The tropical-LM kernels are keyed by
   ``{V, D, D_FF, L_max, n_layers}`` (``FT_*`` / ``FT2_*`` / ``FT3_*`` constants);
   the morphological-SSM kernels use a *different* schema
   ``{D, n_layers, K, L_seq, dil_cap}`` (``MD_*`` / ``MDE_*``). A single Config
   would silently mis-key the morph kernels, so there are two config types.

2. **Two C ABIs.** ``forge_train_step`` / ``forge_train_step_morph[_ecg]`` take
   ``(buffer, mode, arg1, arg2, arg3)`` (4 ints after the buffer); the 2-/3-layer
   kernels add a ``layer_id`` -> ``(buffer, mode, arg1, arg2, arg3, layer_id)``
   (5 ints). :class:`KernelABI` records the arity so the ctypes loader binds the
   right prototype per kernel.

The public config objects remain compact primary keys, but a key hit is only the
first stage of matching. :meth:`KernelCache.match` also checks every fixed model
property represented by the bundled buffer layout and recipe. Ambiguous models
fail closed. In particular, catalog membership is not itself permission to bind
a Python model to a native runner.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# Canonical fused-step mode codes, identical across every bundled kernel
# (verified: forge_train_step.cpp header, MD_MODE_*/MDE_MODE_* = 0/1/2).
MODE_FWD_LOSS = 0
MODE_BWD = 1
MODE_SIGNSGD = 2


# ---------------------------------------------------------------------------
# Preliminary shape keys (model-derivable; compatibility is checked separately)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LMKernelConfig:
    """Match key for the tropical shift-LM kernels: ``(V, D, n_layers)``."""

    V: int
    D: int
    n_layers: int

    @staticmethod
    def from_model(model) -> Optional["LMKernelConfig"]:
        """Derive an LM shape key from a model, or ``None`` if it isn't a clean match.

        Reads only attributes ``TropicalTransformer`` actually stores: ``task_type``,
        an ``nn.Embedding`` ``embed`` (LM), ``blocks`` (a ``ModuleList``), and
        ``d_model``. Returns ``None`` on any ambiguity -> CPU fallback (the safe
        direction).
        """
        import torch.nn as tnn

        if getattr(model, "task_type", None) != "lm":
            return None
        embed = getattr(model, "embed", None)
        if not isinstance(embed, tnn.Embedding):
            return None
        blocks = getattr(model, "blocks", None)
        d_model = getattr(model, "d_model", None)
        if blocks is None or d_model is None:
            return None
        try:
            n_layers = len(blocks)
            vocab = int(embed.num_embeddings)
            width = int(d_model)
        except (AttributeError, OverflowError, TypeError, ValueError):
            return None
        return LMKernelConfig(V=vocab, D=width, n_layers=int(n_layers))


@dataclass(frozen=True)
class MorphKernelConfig:
    """Match key for the morphological-SSM kernels: ``(D, n_layers)``."""

    D: int
    n_layers: int

    @staticmethod
    def from_model(model) -> Optional["MorphKernelConfig"]:
        """Derive a morph shape key from a model (reads ``d_model`` + ``layers``)."""
        layers = getattr(model, "layers", None)
        d_model = getattr(model, "d_model", None)
        if layers is None or d_model is None:
            return None
        try:
            n_layers = len(layers)
            width = int(d_model)
        except (OverflowError, TypeError, ValueError):
            return None
        return MorphKernelConfig(D=width, n_layers=int(n_layers))


@dataclass(frozen=True)
class MuZeroKernelConfig:
    """Match key for the Tropical MuZero net: ``(n_actions, latent_dim, core_kind)``.

    ``core_kind`` is ``"dyadic"`` (the multiplier-free OBLIQUE cores — signed power-of-two
    weights -> shift+add, the shippable flagship core) or ``"maxplus"`` (axis-aligned).
    Derived from attributes ``TropicalMuZeroNet`` reliably exposes; returns ``None`` on any
    ambiguity (-> CPU fallback, the safe direction). NOTE: no MuZero ``KernelSpec`` is
    registered in :meth:`KernelCache.default` yet (the HLS kernel is the §12 build task), so
    ``match()`` returns ``None`` and the loop runs on CPU until that kernel exists — keying a
    model is necessary but not sufficient to bind on-board.
    """

    n_actions: int
    latent_dim: int
    core_kind: str

    @staticmethod
    def from_model(model) -> Optional["MuZeroKernelConfig"]:
        repr_mod = getattr(model, "repr", None)
        pred = getattr(model, "pred", None)
        n_actions = getattr(model, "n_actions", None)
        if repr_mod is None or pred is None or n_actions is None:
            return None
        enc = getattr(repr_mod, "enc", None)
        vhead = getattr(getattr(pred, "value_head", None), "head", None)  # nn.Linear(latent, .)
        if enc is None or vhead is None or not hasattr(vhead, "in_features"):
            return None
        core = {"DyadicTropicalProjection": "dyadic",
                "CompositionalTropicalProjection": "maxplus"}.get(type(enc).__name__)
        if core is None:
            return None
        return MuZeroKernelConfig(n_actions=int(n_actions),
                                  latent_dim=int(vhead.in_features), core_kind=core)


KernelConfig = Union[LMKernelConfig, MorphKernelConfig, MuZeroKernelConfig]


# ---------------------------------------------------------------------------
# C ABI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelABI:
    """A kernel's extern-C call signature.

    Every kernel is ``int <entry>(int32_t* buffer, int mode, ...)``. ``n_int_args``
    counts the ints *after* the buffer pointer; ``has_layer_id`` flags the trailing
    ``layer_id`` of the multi-layer kernels. The total C arity is ``1 + n_int_args``.
    """

    name: str
    n_int_args: int
    has_layer_id: bool


# The two verified signatures.
KernelABI.FUSED_5 = KernelABI("fused5", 4, False)  # (buffer, mode, arg1, arg2, arg3)
KernelABI.FUSED_6 = KernelABI("fused6", 5, True)   # (..., layer_id)


# ---------------------------------------------------------------------------
# Kernel description
# ---------------------------------------------------------------------------


@dataclass
class KernelSpec:
    """Description of one precompiled fused training-step kernel.

    ``entry`` is the verified extern-C symbol; ``source_file`` is the bundled
    ``.cpp`` (resolved via :attr:`source_path`); ``buffer_words_const`` names the
    ``.hpp`` constant giving the flat int32 buffer size. The ``d_ff/l_max`` (LM) or
    ``k/l_seq/dil_cap`` (morph) fields are the kernel's *fixed* secondary dims (not
    part of the public config key). Provenance metrics (``throughput_tok_s`` etc.)
    are build-specific reported operating points from ``hardware/README.md``;
    they are not source-level invariants or promises about another build. The
    same qualification applies to ``dsp``.
    """

    model_type: str
    config: KernelConfig
    entry: str
    abi: KernelABI
    source_file: str
    buffer_words_const: str
    # Secondary (kernel-fixed) dims enforced by KernelCache.match.
    d_ff: Optional[int] = None
    l_max: Optional[int] = None
    k: Optional[int] = None
    l_seq: Optional[int] = None
    dil_cap: Optional[int] = None
    dtype: str = "int16"
    grad_dtype: str = "int32"
    dsp: int = 0
    throughput_tok_s: Optional[float] = None
    energy_mj_tok: Optional[float] = None
    wns_ns: Optional[float] = None
    modes: Dict[str, int] = field(
        default_factory=lambda: {"fwd_loss": MODE_FWD_LOSS, "bwd": MODE_BWD, "signsgd": MODE_SIGNSGD}
    )

    @property
    def key(self) -> Tuple[str, KernelConfig]:
        return (self.model_type, self.config)

    @property
    def source_path(self) -> str:
        """Absolute path to the bundled HLS source for this kernel."""
        from tikitorch.hardware import hls_dir

        return os.path.join(hls_dir(), "forge_kernels", self.source_file)


# ---------------------------------------------------------------------------
# Conservative model/layout validation
# ---------------------------------------------------------------------------


# The HLS kernels center each token over its final feature axis with an integer
# right shift and INT16 saturation. None of TikiTorch's stock normalization
# modules implements that exact contract: TropicalBatchShift reduces dim=0 and
# TropicalLayerNorm additionally rescales and has a beta. A dedicated module may
# declare this narrow contract; the shape/state checks below still apply. Keeping
# the marker private avoids presenting it as a general-purpose public norm API.
_FEATURE_SHIFT_CONTRACT = "int16-last-axis-mean-shift-v1"


def _shape_is(value, *shape: int) -> bool:
    try:
        return tuple(value.shape) == tuple(shape)
    except (AttributeError, TypeError):
        return False


def _int_attr(value, name: str) -> Optional[int]:
    try:
        raw = getattr(value, name)
        # Reject floats such as 128.5 rather than silently truncating them.
        converted = int(raw)
        return converted if converted == raw else None
    except (AttributeError, OverflowError, TypeError, ValueError):
        return None


def _is_inert_bias(module, size: int) -> bool:
    """True for no bias, or an explicitly frozen all-zero bias of ``size``."""
    import torch

    bias = getattr(module, "bias", None)
    if bias is None:
        return True
    if not _shape_is(bias, size) or bool(getattr(bias, "requires_grad", False)):
        return False
    try:
        return bool(torch.count_nonzero(bias.detach()).item() == 0)
    except (NotImplementedError, RuntimeError, TypeError):
        return False


def _is_feature_shift(module, width: int) -> bool:
    """Recognize the explicit, parameter-free norm contract used by the HLS."""
    if getattr(module, "_tikitorch_fpga_norm", None) != _FEATURE_SHIFT_CONTRACT:
        return False
    if _int_attr(module, "normalized_shape") != width:
        return False
    if _int_attr(module, "mean_shift_bits") != width.bit_length() - 1:
        return False
    if _int_attr(module, "saturating_bits") != 16:
        return False
    try:
        return not any(True for _ in module.parameters()) and not any(
            True for _ in module.buffers()
        )
    except (AttributeError, TypeError):
        return False


def _plain_embedding_matches(embed, vocab: int, width: int) -> bool:
    import torch.nn as tnn

    return (
        type(embed) is tnn.Embedding
        and _int_attr(embed, "num_embeddings") == vocab
        and _int_attr(embed, "embedding_dim") == width
        and _shape_is(getattr(embed, "weight", None), vocab, width)
        and getattr(embed, "padding_idx", None) is None
        and getattr(embed, "max_norm", None) is None
        and getattr(embed, "norm_type", None) == 2.0
        and getattr(embed, "scale_grad_by_freq", None) is False
        and getattr(embed, "sparse", None) is False
    )


def _lm_head_matches(head, vocab: int, width: int) -> bool:
    from tikitorch.nn import TropicalDilationModule, TropicalTopKModule

    common = (
        _int_attr(head, "in_features") == width
        and _int_attr(head, "out_features") == vocab
        and _shape_is(getattr(head, "weight", None), vocab, width)
        and bool(getattr(getattr(head, "weight", None), "requires_grad", False))
        and _is_inert_bias(head, vocab)
    )
    if not common:
        return False
    if type(head) is TropicalTopKModule:
        return _int_attr(head, "k") == 1 and getattr(head, "tau", None) == 0
    if type(head) is TropicalDilationModule:
        return (
            getattr(head, "use_softmax", None) is False
            and getattr(head, "tau", None) == 0
        )
    return False


def _matches_lm_layout(model, spec: KernelSpec) -> bool:
    """Check the full model-observable contract of an LM catalog entry."""
    import torch.nn as tnn

    from tikitorch.models import TropicalTransformer
    from tikitorch.nn import (
        TropicalDilationModule,
        TropicalLinearModule,
        TropicalShiftAttention,
        TropicalTransformerBlock,
    )

    cfg = spec.config
    if not isinstance(cfg, LMKernelConfig) or spec.d_ff is None or spec.l_max is None:
        return False
    if type(model) is not TropicalTransformer or getattr(model, "task_type", None) != "lm":
        return False
    if _int_attr(model, "d_model") != cfg.D:
        return False
    if not _plain_embedding_matches(getattr(model, "embed", None), cfg.V, cfg.D):
        return False
    if not _lm_head_matches(getattr(model, "output_head", None), cfg.V, cfg.D):
        return False
    if type(getattr(model, "ln_final", None)) is not tnn.Identity:
        return False

    blocks = getattr(model, "blocks", None)
    try:
        if len(blocks) != cfg.n_layers:
            return False
    except TypeError:
        return False

    expected_params = {
        "embed.weight": (cfg.V, cfg.D),
        "output_head.weight": (cfg.V, cfg.D),
    }
    for layer_id, block in enumerate(blocks):
        if type(block) is not TropicalTransformerBlock:
            return False
        if not _is_feature_shift(getattr(block, "norm1", None), cfg.D):
            return False
        if not _is_feature_shift(getattr(block, "norm2", None), cfg.D):
            return False
        if type(getattr(block, "dropout", None)) is not tnn.Identity:
            return False

        attn = getattr(block, "attn", None)
        if type(attn) is not TropicalShiftAttention:
            return False
        if (
            _int_attr(attn, "d_model") != cfg.D
            or _int_attr(attn, "n_heads") != 1
            or _int_attr(attn, "d_head") != cfg.D
            or _int_attr(attn, "max_seq_len") != spec.l_max
            or getattr(attn, "tau", None) != 0
            or getattr(attn, "score_variant", None) != "dot_product"
            or getattr(attn, "shift_mode", None) != "free"
            or getattr(attn, "agg_variant", None) != "argmax"
            or getattr(attn, "rank_bias", None) is not None
        ):
            return False
        if not _plain_embedding_matches(getattr(attn, "shift_q", None), spec.l_max, cfg.D):
            return False
        if not _plain_embedding_matches(getattr(attn, "shift_k", None), spec.l_max, cfg.D):
            return False

        ffn_up = getattr(block, "ffn_up", None)
        if (
            type(ffn_up) is not TropicalDilationModule
            or _int_attr(ffn_up, "in_features") != cfg.D
            or _int_attr(ffn_up, "out_features") != spec.d_ff
            or getattr(ffn_up, "use_softmax", None) is not False
            or getattr(ffn_up, "tau", None) != 0
            or not _shape_is(getattr(ffn_up, "weight", None), spec.d_ff, cfg.D)
            or not _shape_is(getattr(ffn_up, "bias", None), spec.d_ff)
        ):
            return False

        ffn_down = getattr(block, "ffn_down", None)
        if (
            type(ffn_down) is not TropicalLinearModule
            or _int_attr(ffn_down, "in_features") != spec.d_ff
            or _int_attr(ffn_down, "out_features") != cfg.D
            or getattr(ffn_down, "use_softmin", None) is not False
            or getattr(ffn_down, "tau", None) != 0
            or not _shape_is(getattr(ffn_down, "weight", None), cfg.D, spec.d_ff)
            or not _shape_is(getattr(ffn_down, "bias", None), cfg.D)
        ):
            return False

        prefix = f"blocks.{layer_id}"
        expected_params.update({
            f"{prefix}.attn.shift_q.weight": (spec.l_max, cfg.D),
            f"{prefix}.attn.shift_k.weight": (spec.l_max, cfg.D),
            f"{prefix}.ffn_up.weight": (spec.d_ff, cfg.D),
            f"{prefix}.ffn_up.bias": (spec.d_ff,),
            f"{prefix}.ffn_down.weight": (cfg.D, spec.d_ff),
            f"{prefix}.ffn_down.bias": (cfg.D,),
        })

    actual_params = {
        name: tuple(param.shape)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if actual_params != expected_params:
        return False

    # The sole bundled runner flattens parameters in registration order, skips
    # B1/B2 regions, and cannot sync the kernel-updated biases back by name. No
    # Python model can make that runner's training state contract safe merely by
    # changing its module shapes, so this entry must not auto-bind.
    if spec.entry == "forge_train_step":
        return False
    return True


def _boundary_width(module, attribute: str) -> Optional[int]:
    return _int_attr(module, attribute)


def _matches_morph_layout(model, spec: KernelSpec) -> bool:
    """Check a MorphSSM core recipe, including explicit capacity declarations."""
    import torch.nn as tnn

    from tikitorch.models import MorphSSM
    from tikitorch.nn import (
        DepthwiseMorphConv1d,
        MorphSSMLayer,
        TropicalDilationModule,
        TropicalQuantizedEmbed,
        TropicalTopKModule,
    )

    cfg = spec.config
    if (
        not isinstance(cfg, MorphKernelConfig)
        or spec.k is None
        or spec.l_seq is None
        or spec.dil_cap is None
    ):
        return False
    if type(model) is not MorphSSM or getattr(model, "task_type", None) != "classification":
        return False
    if _int_attr(model, "d_model") != cfg.D:
        return False
    # MorphSSM does not retain these constructor values today. A prospective
    # binding must therefore declare them explicitly on the model; absence or a
    # conflicting declaration is ambiguous and fails closed.
    if _int_attr(model, "fpga_sequence_length") != spec.l_seq:
        return False
    if _int_attr(model, "fpga_dilation_cap") != spec.dil_cap:
        return False
    if type(getattr(model, "dropout", None)) is not tnn.Identity:
        return False
    if type(getattr(model, "ln_final", None)) is not tnn.Identity:
        return False
    if getattr(model, "use_checkpoint", None) is not False:
        return False

    embed = getattr(model, "embed", None)
    head = getattr(model, "output_head", None)
    if type(embed) not in (TropicalQuantizedEmbed, tnn.Linear):
        return False
    if _boundary_width(embed, "out_features") != cfg.D:
        return False
    if type(head) not in (TropicalTopKModule, TropicalDilationModule, tnn.Linear):
        return False
    if _boundary_width(head, "in_features") != cfg.D:
        return False
    if _int_attr(head, "out_features") != _int_attr(model, "num_classes"):
        return False

    layers = getattr(model, "layers", None)
    try:
        if len(layers) != cfg.n_layers:
            return False
    except TypeError:
        return False

    expected_core_params = {}
    for layer_id, layer in enumerate(layers):
        if type(layer) is not MorphSSMLayer:
            return False
        if not _is_feature_shift(getattr(layer, "norm", None), cfg.D):
            return False
        if type(getattr(layer, "post_norm", None)) is not tnn.Identity:
            return False
        if type(getattr(layer, "dropout", None)) is not tnn.Identity:
            return False
        if getattr(layer, "residual_scale", None) != 1.0:
            return False

        conv = getattr(layer, "morph_conv", None)
        expected_dilation = min(2 ** layer_id, spec.dil_cap)
        expected_mode = "dilation" if layer_id % 2 == 0 else "erosion"
        if (
            type(conv) is not DepthwiseMorphConv1d
            or _int_attr(conv, "channels") != cfg.D
            or _int_attr(conv, "kernel_size") != spec.k
            or _int_attr(conv, "dilation") != expected_dilation
            or getattr(conv, "mode", None) != expected_mode
            or _int_attr(conv, "group_size") != 1
            or getattr(conv, "use_soft", None) is not False
            or getattr(conv, "tau", None) != 0
            or not _shape_is(getattr(conv, "weight", None), cfg.D, spec.k)
            or not _is_inert_bias(conv, cfg.D)
        ):
            return False

        proj = getattr(layer, "out_proj", None)
        if (
            getattr(layer, "out_proj_type", None) != "tropical"
            or type(proj) is not TropicalDilationModule
            or _int_attr(proj, "in_features") != cfg.D
            or _int_attr(proj, "out_features") != cfg.D
            or getattr(proj, "use_softmax", None) is not False
            or getattr(proj, "tau", None) != 0
            or not _shape_is(getattr(proj, "weight", None), cfg.D, cfg.D)
            or not _is_inert_bias(proj, cfg.D)
        ):
            return False

        prefix = f"layers.{layer_id}"
        expected_core_params.update({
            f"{prefix}.morph_conv.weight": (cfg.D, spec.k),
            f"{prefix}.out_proj.weight": (cfg.D, cfg.D),
        })

    actual_core_params = {
        name: tuple(param.shape)
        for name, param in model.named_parameters()
        if name.startswith("layers.") and param.requires_grad
    }
    if actual_core_params != expected_core_params:
        return False

    # The small kernel exposes no input gradient, so a host-side embedding could
    # not be kept in sync by a full training-step binding.
    if spec.entry == "forge_train_step_morph":
        try:
            if any(param.requires_grad for param in embed.parameters()):
                return False
        except (AttributeError, TypeError):
            return False

    # The small Morph HLS core skips invalid causal taps, whereas the public
    # DepthwiseMorphConv1d pads with zeros and includes those taps in the
    # max/min reduction. The two recipes can therefore disagree at a sequence
    # boundary even when every shape and declared capacity matches. Keep the
    # catalog record inspectable, but fail the whole-model association until a
    # dedicated HLS-contract module (and boundary-equivalence tests) exists.
    if spec.entry == "forge_train_step_morph":
        return False

    # The ECG artifact pools with sum>>12 (/4096) at L=5000, whereas MorphSSM's
    # public forward uses an exact arithmetic mean. Until a runner/factory makes
    # that boundary explicit, this whole-model association is not safe.
    if spec.entry == "forge_train_step_morph_ecg":
        return False
    return True


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class KernelCache:
    """A registry of precompiled kernels, looked up by ``(model_type, config)``."""

    def __init__(self) -> None:
        self._specs: Dict[Tuple[str, KernelConfig], KernelSpec] = {}

    def register(self, spec: KernelSpec) -> None:
        """Add (or overwrite) a kernel spec, keyed by its ``(model_type, config)``."""
        self._specs[spec.key] = spec

    def lookup(self, model_type: str, config: KernelConfig) -> Optional[KernelSpec]:
        """Exact lookup by model type + shape key; ``None`` on miss (-> CPU fallback)."""
        return self._specs.get((model_type, config))

    def match(self, model) -> Optional[KernelSpec]:
        """Return a kernel spec only for a fully established model/layout match.

        Routing first derives the backwards-compatible compact config key. A hit
        then passes an exact, spec-specific structural check covering secondary
        dimensions, operators, hard/soft variants, normalization, heads, and
        parameter regions. Missing attributes and unrepresentable contracts fail
        closed. (Runtime input validation still belongs to a native runner.)
        """
        model_type = type(model).__name__
        if model_type == "TropicalTransformer":
            cfg = LMKernelConfig.from_model(model)
        elif model_type == "MorphSSM":
            cfg = MorphKernelConfig.from_model(model)
        elif model_type == "TropicalMuZeroNet":
            cfg = MuZeroKernelConfig.from_model(model)   # keyable; no spec registered yet (CPU fallback)
        else:
            return None
        if cfg is None:
            return None
        spec = self.lookup(model_type, cfg)
        if spec is None:
            return None
        if model_type == "TropicalTransformer":
            return spec if _matches_lm_layout(model, spec) else None
        if model_type == "MorphSSM":
            return spec if _matches_morph_layout(model, spec) else None
        return spec

    def specs(self) -> List[KernelSpec]:
        """All registered kernel specs."""
        return list(self._specs.values())

    @classmethod
    def default(cls) -> "KernelCache":
        """A catalog pre-seeded with the bundled, build-specific kernel specs.

        Constants and reported operating points are transcribed from the bundled
        ``forge_kernels/*.hpp`` and ``hardware/README.md``. Entries describe known
        artifacts; they are not automatically safe model bindings. :meth:`match`
        applies the separate conservative compatibility gate.
        """
        cache = cls()

        # --- Tropical shift-LM kernels (V=128, D=128, D_FF=512, L_max=128) ---
        cache.register(KernelSpec(
            model_type="TropicalTransformer", config=LMKernelConfig(V=128, D=128, n_layers=1),
            entry="forge_train_step", abi=KernelABI.FUSED_5,
            source_file="forge_train_step.cpp", buffer_words_const="FT_BUFFER_WORDS",
            d_ff=512, l_max=128, throughput_tok_s=1200.0,
        ))
        cache.register(KernelSpec(
            model_type="TropicalTransformer", config=LMKernelConfig(V=128, D=128, n_layers=2),
            entry="forge_train_step_2layer", abi=KernelABI.FUSED_6,
            source_file="forge_train_step_2layer.cpp", buffer_words_const="FT2_BUFFER_WORDS",
            d_ff=512, l_max=128, throughput_tok_s=770.0, energy_mj_tok=9.0, wns_ns=-0.197,
        ))
        cache.register(KernelSpec(
            model_type="TropicalTransformer", config=LMKernelConfig(V=128, D=128, n_layers=3),
            entry="forge_train_step_3layer", abi=KernelABI.FUSED_6,
            source_file="forge_train_step_3layer.cpp", buffer_words_const="FT3_BUFFER_WORDS",
            d_ff=512, l_max=128,
        ))

        # --- Morphological-SSM kernels (different shape schema + 5-int ABI) ---
        cache.register(KernelSpec(
            model_type="MorphSSM", config=MorphKernelConfig(D=64, n_layers=4),
            entry="forge_train_step_morph", abi=KernelABI.FUSED_5,
            source_file="forge_train_step_morph.cpp", buffer_words_const="MD_BUFFER_WORDS",
            k=4, l_seq=64, dil_cap=16,
        ))
        cache.register(KernelSpec(
            model_type="MorphSSM", config=MorphKernelConfig(D=128, n_layers=8),
            entry="forge_train_step_morph_ecg", abi=KernelABI.FUSED_5,
            source_file="forge_train_step_morph_ecg.cpp", buffer_words_const="FT_BUFFER_WORDS_E",
            k=7, l_seq=5000, dil_cap=32,
            throughput_tok_s=1207.0, energy_mj_tok=5.8, wns_ns=0.009,
        ))
        return cache
