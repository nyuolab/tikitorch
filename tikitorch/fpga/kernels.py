"""tikitorch.fpga.kernels — the compiled-kernel catalog (KernelCache).

A pure-Python, O(1) lookup from a model's shape to a *precompiled* fused
training-step kernel. **No compilation ever happens in the hot path** — selection
is a dict lookup, exactly as the strategy doc requires.

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

The *match key* is intentionally restricted to the fields a TikiTorch model
reliably exposes (``V, D, n_layers`` for LM; ``D, n_layers`` for morph). The
remaining kernel dimensions (``d_ff``, ``l_max``, ``k``, ``l_seq``, ``dil_cap``)
ride along on the :class:`KernelSpec` as metadata; the native bind path verifies
them (e.g. weight-shape introspection) before going on-board. Until a native
handle exists, no model is ever *bound* (see :mod:`tikitorch.fpga.backend`), so a
permissive primary key cannot cause a false on-board run.
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
# Shape keys (model-derivable; conservative)
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
        except TypeError:
            return None
        return LMKernelConfig(V=int(embed.num_embeddings), D=int(d_model), n_layers=int(n_layers))


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
        except TypeError:
            return None
        return MorphKernelConfig(D=int(d_model), n_layers=int(n_layers))


KernelConfig = Union[LMKernelConfig, MorphKernelConfig]


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
    """Immutable description of one precompiled fused training-step kernel.

    ``entry`` is the verified extern-C symbol; ``source_file`` is the bundled
    ``.cpp`` (resolved via :attr:`source_path`); ``buffer_words_const`` names the
    ``.hpp`` constant giving the flat int32 buffer size. The ``d_ff/l_max`` (LM) or
    ``k/l_seq/dil_cap`` (morph) fields are the kernel's *fixed* secondary dims (not
    part of the match key). Provenance metrics (``throughput_tok_s`` etc.) are the
    silicon-measured operating points from ``hardware/README.md``. ``dsp`` is 0 by
    the closure invariant.
    """

    model_type: str
    config: KernelConfig
    entry: str
    abi: KernelABI
    source_file: str
    buffer_words_const: str
    # Secondary (kernel-fixed) dims, verified the native bind path must honor.
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
        """Derive a model's shape key and return its kernel spec, or ``None``.

        Routes by model class name to the correct config schema ("no multi-kernel
        mux": one fused kernel per exact ``(model_type, config)`` match).
        """
        model_type = type(model).__name__
        if model_type == "TropicalTransformer":
            cfg = LMKernelConfig.from_model(model)
        elif model_type == "MorphSSM":
            cfg = MorphKernelConfig.from_model(model)
        else:
            return None
        if cfg is None:
            return None
        return self.lookup(model_type, cfg)

    def specs(self) -> List[KernelSpec]:
        """All registered kernel specs."""
        return list(self._specs.values())

    @classmethod
    def default(cls) -> "KernelCache":
        """A cache pre-seeded with the silicon-validated kernels (the allow-list).

        Constants and operating points are transcribed from the bundled
        ``forge_kernels/*.hpp`` and ``hardware/README.md``. This is the conservative
        allow-list: only these exact shapes ever bind on-board.
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
