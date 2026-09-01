"""Tests for the fail-closed FPGA/native integration.

Native-endpoint assertions are skip-guarded with ``requires_fpga_native``.
Everything else exercises catalog-versus-match behavior, explicit runner
registration, and the always-present PyTorch fallback.
"""
import pathlib
import warnings

import pytest
import torch

import tikitorch as tt
import tikitorch.backend as B

from conftest import requires_fpga_native, requires_model, requires_optim

fpga = pytest.importorskip("tikitorch.fpga")


class _HLSFeatureShift(torch.nn.Module):
    """Explicit model-side declaration of the bundled INT16 feature shift."""

    _tikitorch_fpga_norm = "int16-last-axis-mean-shift-v1"

    def __init__(self, width):
        super().__init__()
        self.normalized_shape = width
        self.mean_shift_bits = width.bit_length() - 1
        self.saturating_bits = 16

    def forward(self, x):
        centered = x - x.sum(dim=-1, keepdim=True) / (1 << self.mean_shift_bits)
        return centered.clamp(-32768, 32767)


def _hls_lm(*, n_layers=2, d_ff=512, max_seq_len=128, n_heads=1,
            use_shift_attn=True, score_variant="dot_product", shift_mode="free"):
    """Build the model-observable recipe of the 2-/3-layer HLS LM kernels."""
    m = tt.models.TropicalTransformer(
        d_model=128,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0,
        tau=0,
        use_shift_attn=use_shift_attn,
        max_seq_len=max_seq_len,
        score_variant=score_variant,
        shift_mode=shift_mode,
        norm_type="tropical_shift",
        task_type="lm",
        vocab_size=128,
    )
    m.set_tau(0)
    for block in m.blocks:
        block.norm1 = _HLSFeatureShift(128)
        block.norm2 = _HLSFeatureShift(128)
    m.ln_final = torch.nn.Identity()
    m.output_head.k = 1
    with torch.no_grad():
        m.output_head.bias.zero_()
    m.output_head.bias.requires_grad_(False)
    return m


def _hls_small_morph(*, kernel_size=4, l_seq=64, dil_cap=16):
    """Build the representable, frozen-input contract of the small morph core."""
    m = tt.models.MorphSSM(
        d_model=64,
        n_layers=4,
        kernel_size=kernel_size,
        dropout=0,
        use_soft=False,
        tau=0,
        norm_type="tropical_shift",
        group_size=1,
        max_dilation=dil_cap,
        task_type="classification",
        input_dim=1,
        num_classes=10,
    )
    m.set_tau(0)
    m.fpga_sequence_length = l_seq
    m.fpga_dilation_cap = dil_cap
    m.ln_final = torch.nn.Identity()
    for parameter in m.embed.parameters():
        parameter.requires_grad_(False)
    for layer in m.layers:
        layer.norm = _HLSFeatureShift(64)
        layer.post_norm = torch.nn.Identity()
        with torch.no_grad():
            layer.morph_conv.bias.zero_()
            layer.out_proj.bias.zero_()
        layer.morph_conv.bias.requires_grad_(False)
        layer.out_proj.bias.requires_grad_(False)
    return m


# --- availability + import safety -----------------------------------------

def test_fpga_imports_and_registers():
    assert tt.fpga is not None
    assert "fpga" in B.list_backends()
    # A normal CI wheel has no endpoint, while a supported LOCAL installation
    # may expose one host-reference endpoint. The public probes must agree in
    # either environment.
    available = fpga.is_available()
    assert fpga.device_count() == int(available)
    assert ("fpga" in B.available_backends()) is available
    capabilities = fpga.get_backend().capabilities()
    assert capabilities.model_binding is True
    assert capabilities.op_dispatch is False
    assert capabilities.on_device_training is False
    assert capabilities.integer_only is False


def test_unsafe_reference_runner_compatibility_hook_fails_closed():
    import tikitorch.fpga._session as session

    before = dict(session.BOARD_RUNNERS)
    with pytest.raises(fpga.NativeUnavailable, match="reference runner is disabled"):
        fpga.register_reference_runners()
    assert session.BOARD_RUNNERS == before


def test_no_forge_import_in_fpga_package():
    root = pathlib.Path(fpga.__file__).resolve().parent
    offenders = []
    for py in root.rglob("*.py"):
        for line in py.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("import forge") or s.startswith("from forge"):
                offenders.append(f"{py}: {s}")
    assert not offenders, "forge import leaked into tikitorch.fpga:\n" + "\n".join(offenders)


# --- kernel catalog (two schemas, two ABIs) -------------------------------

@requires_model("TropicalTransformer")
def test_kernelcache_matches_lm_2layer():
    cache = fpga.KernelCache.default()
    m = _hls_lm(n_layers=2)
    spec = cache.match(m)
    assert spec is not None
    assert spec.entry == "forge_train_step_2layer"
    assert spec.abi is fpga.KernelABI.FUSED_6     # 6-int ABI (has layer_id)
    assert spec.config == fpga.LMKernelConfig(V=128, D=128, n_layers=2)
    assert spec.dsp == 0
    assert pathlib.Path(spec.source_path).exists()


@requires_model("TropicalTransformer")
def test_kernelcache_misses_offspec_shape():
    cache = fpga.KernelCache.default()
    # Same primary V/D/layer key, but the stock projected 4-head recipe is not
    # the one-head shift-attention kernel recipe.
    assert cache.match(tt.models.TropicalTransformer(
        d_model=128, n_heads=4, n_layers=2, task_type="lm", vocab_size=128)) is None
    assert cache.match(tt.models.TropicalTransformer(
        d_model=64, n_heads=4, n_layers=2, task_type="lm", vocab_size=128)) is None
    assert cache.match(tt.models.TropicalTransformer(
        d_model=128, n_heads=4, n_layers=5, task_type="lm", vocab_size=128)) is None
    # classification (no nn.Embedding) -> no LM match
    assert cache.match(tt.models.TropicalTransformer(
        d_model=128, n_heads=4, n_layers=2, task_type="classification")) is None


@requires_model("TropicalTransformer")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("d_ff", 256),
        ("max_seq_len", 64),
        ("n_heads", 2),
        ("use_shift_attn", False),
        ("score_variant", "l_inf"),
        ("shift_mode", "rope_relative"),
    ],
)
def test_kernelcache_lm_rejects_secondary_dimension_and_attention_variants(field, value):
    kwargs = {field: value}
    assert fpga.KernelCache.default().match(_hls_lm(**kwargs)) is None


@requires_model("TropicalTransformer")
def test_kernelcache_lm_rejects_head_norm_and_unsafe_legacy_layout():
    cache = fpga.KernelCache.default()

    bad_head = _hls_lm()
    bad_head.output_head.k = 2
    assert cache.match(bad_head) is None

    bad_norm = _hls_lm()
    bad_norm.blocks[0].norm1 = tt.nn.TropicalBatchShift()
    assert cache.match(bad_norm) is None

    bad_aggregation = _hls_lm()
    bad_aggregation.blocks[0].attn.agg_variant = "topk"
    assert cache.match(bad_aggregation) is None

    biased_head = _hls_lm()
    biased_head.output_head.bias.requires_grad_(True)
    assert cache.match(biased_head) is None

    # Even an otherwise exact one-layer structure fails closed: the retired
    # generic marshaller did not preserve named-region order or B1/B2 state.
    assert cache.match(_hls_lm(n_layers=1)) is None
    assert cache.lookup(
        "TropicalTransformer", fpga.LMKernelConfig(V=128, D=128, n_layers=1)
    ) is not None


@requires_model("MorphSSM")
def test_kernelcache_morph_catalog_is_inspectable_but_match_fails_closed():
    cache = fpga.KernelCache.default()
    m = _hls_small_morph()
    spec = cache.lookup(
        "MorphSSM", fpga.MorphKernelConfig(D=64, n_layers=4)
    )
    assert spec is not None
    assert spec.entry == "forge_train_step_morph"
    assert spec.abi is fpga.KernelABI.FUSED_5     # 5-int ABI (no layer_id)
    assert isinstance(spec.config, fpga.MorphKernelConfig)
    assert not isinstance(spec.config, fpga.LMKernelConfig)
    # Public zero-padding and HLS skip-invalid causal boundaries differ.
    assert cache.match(m) is None


@requires_model("MorphSSM")
def test_kernelcache_morph_rejects_secondary_dimensions_and_recipe_variants():
    cache = fpga.KernelCache.default()
    # The stock same-primary-shape model has K=7, soft ops, trainable biases,
    # post/final norms, and no declared sequence/cap contract.
    assert cache.match(tt.models.MorphSSM(
        d_model=64, n_layers=4, task_type="classification")) is None
    assert cache.match(_hls_small_morph(kernel_size=7)) is None
    assert cache.match(_hls_small_morph(l_seq=63)) is None
    assert cache.match(_hls_small_morph(dil_cap=8)) is None

    missing_capacity = _hls_small_morph()
    del missing_capacity.fpga_sequence_length
    assert cache.match(missing_capacity) is None

    soft = _hls_small_morph()
    soft.layers[0].morph_conv.use_soft = True
    soft.layers[0].morph_conv.tau = 1
    assert cache.match(soft) is None

    biased = _hls_small_morph()
    biased.layers[0].out_proj.bias.requires_grad_(True)
    assert cache.match(biased) is None


def test_two_abis_are_distinct():
    assert fpga.KernelABI.FUSED_5.has_layer_id is False
    assert fpga.KernelABI.FUSED_6.has_layer_id is True
    assert fpga.KernelABI.FUSED_5.n_int_args == 4
    assert fpga.KernelABI.FUSED_6.n_int_args == 5


# --- AcceleratedModel wrapper and PyTorch fallback ------------------------

@requires_model("TropicalTransformer")
def test_accelerate_returns_unbound_wrapper():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    acc = fpga.accelerate(m)
    assert isinstance(acc, fpga.AcceleratedModel)
    assert isinstance(acc, torch.nn.Module)
    assert acc.bound is False                 # no endpoint / explicitly registered runner
    assert acc.module is m


@requires_model("TropicalTransformer")
def test_accelerate_is_idempotent():
    # Wrapping twice must NOT nest wrappers (which would break kernel matching and
    # state_dict keys); the inner module is rebound instead.
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    once = fpga.accelerate(m)
    twice = fpga.accelerate(once)
    assert twice.module is m                  # unwrapped, not nested
    assert not isinstance(twice.module, fpga.AcceleratedModel)
    assert set(twice.state_dict()) == set(once.state_dict())  # keys not doubly-prefixed


@requires_model("TropicalTransformer")
@pytest.mark.parametrize("device", ["cpu", "cuda:0", "fpga:1"])
def test_accelerate_rejects_wrong_device_type_and_unsupported_endpoint(device):
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    with pytest.raises(ValueError, match="FPGA acceleration|unsupported FPGA endpoint"):
        fpga.accelerate(m, device=device)


@requires_model("TropicalTransformer")
def test_accelerated_delegation_and_proxy():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128).eval()
    acc = fpga.accelerate(m)
    x = torch.randint(0, 128, (2, 16))
    assert torch.equal(acc.forward(x), m.forward(x))     # forward delegates
    assert acc.param_count == m.param_count              # property delegates
    assert acc.task_type == "lm"                         # __getattr__ proxy
    # wrapper exposes the inner params (DDP-style submodule registration)
    assert sum(1 for _ in acc.parameters()) == sum(1 for _ in m.parameters())


@requires_model("TropicalTransformer")
def test_fallback_loss_is_byte_identical_to_unwrapped():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128).eval()
    acc = fpga.accelerate(m)
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    loss_wrapped = acc.compute_loss(x, y)
    loss_inner = m.compute_loss(x, y)
    assert torch.isfinite(loss_wrapped)
    assert torch.equal(loss_wrapped, loss_inner)         # frozen-core guarantee
    loss_wrapped.backward()                               # autograd flows through


@requires_model("TropicalTransformer")
@requires_optim("PureTropicalSGD")
def test_train_step_fallback_runs_and_warns_once():
    import tikitorch.optim as optim
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    acc = fpga.accelerate(m, optimizer=optim.PureTropicalSGD(m.parameters()))
    x = torch.randint(0, 128, (2, 16))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss = acc.train_step(x, x)
        acc.train_step(x, x)   # second time: no new warning
    assert isinstance(loss, float) and loss == loss   # finite
    fb = [w for w in caught if issubclass(w.category, B.BackendFallbackWarning)]
    assert len(fb) == 1


@requires_model("TropicalTransformer")
def test_train_step_requires_targets_on_fallback():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    acc = fpga.accelerate(m)
    with pytest.raises(ValueError):
        acc.train_step(torch.randint(0, 128, (2, 16)))


@requires_model("TropicalTransformer")
def test_strict_raises_without_native_endpoint():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    with pytest.raises(fpga.NativeUnavailable):
        fpga.accelerate(m, strict=True)


@requires_model("TropicalTransformer")
def test_to_cpu_unwraps():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    acc = fpga.accelerate(m)
    assert acc.to(tt.device("cpu")) is m       # Device('cpu') returns the plain model
    # but the STRING 'cpu' keeps torch semantics (moves params, wrapper preserved)
    acc2 = fpga.accelerate(tt.models.TropicalTransformer(
        d_model=128, n_heads=4, n_layers=2, task_type="lm", vocab_size=128))
    assert isinstance(acc2.to("cpu"), fpga.AcceleratedModel)
    with pytest.raises(ValueError, match="logical 'cpu' or 'fpga'"):
        acc2.to(tt.device("cuda:0"))
    with pytest.raises(ValueError, match="unsupported FPGA endpoint"):
        acc2.to(tt.device("fpga:1"))


# --- the .fpga() ergonomic mover (FpgaAccelerable mixin) ------------------

@requires_model("TropicalTransformer")
def test_model_dot_fpga_returns_wrapper_and_matches_kernel():
    m = _hls_lm(n_layers=2)
    am = m.fpga()                                   # the one-liner
    assert isinstance(am, fpga.AcceleratedModel)
    assert am.module is m
    assert am.kernel_spec is not None and am.kernel_spec.entry == "forge_train_step_2layer"
    assert am.bound is False                        # no endpoint/runner -> PyTorch fallback


@requires_model("MorphSSM")
def test_morphssm_dot_fpga_fails_closed_on_boundary_semantics():
    m = _hls_small_morph()
    am = m.fpga()
    assert am.kernel_spec is None


@requires_model("TropicalTransformer")
def test_dot_fpga_is_idempotent_and_cpu_unwraps():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    am = m.fpga()
    again = am.fpga()                               # wrapper.fpga() must not nest
    assert again.module is m and not isinstance(again.module, fpga.AcceleratedModel)
    assert am.cpu() is m                            # unwrap inverse


@requires_model("TropicalTransformer")
def test_dot_fpga_does_not_shadow_nn_module_cpu():
    # The mixin must NOT override nn.Module.cpu() on the bare model (that would stop
    # parameters moving to the torch CPU device).
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    assert m.cpu() is m                             # nn.Module.cpu returns self
    assert all(p.device.type == "cpu" for p in m.parameters())


@requires_model("TropicalTransformer")
def test_dot_fpga_strict_raises_without_native_endpoint():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    with pytest.raises(fpga.NativeUnavailable):
        m.fpga(strict=True)


# --- explicitly registered runner path (faked handle; no hardware) --------
# These cover the rewritten bound/train_step delegation that is otherwise
# reachable only with a usable native endpoint, using a sentinel handle plus a
# registered runner so no native library or FPGA is required.

@requires_model("TropicalTransformer")
def test_bound_train_step_delegates_to_registered_runner():
    import tikitorch.fpga._session as session
    m = _hls_lm(n_layers=2)
    spec = fpga.KernelCache.default().match(m)
    seen = {}

    def fake_runner(handle, spec, model, tokens, *, targets, lr, threshold):
        seen.update(entry=spec.entry, lr=lr, threshold=threshold, handle=handle)
        return 0.5

    session.register_board_runner("forge_train_step_2layer", fake_runner)
    try:
        am = fpga.AcceleratedModel(m, spec=spec, native=object(), device=tt.device("fpga"))
        assert am.bound is True                       # native + spec + runner
        x = torch.randint(0, 128, (2, 16))
        out = am.train_step(x, x, lr=4, threshold=2)
        assert out == 0.5
        assert seen["entry"] == "forge_train_step_2layer"
        assert seen["lr"] == 4 and seen["threshold"] == 2 and seen["handle"] is not None
    finally:
        session.BOARD_RUNNERS.pop("forge_train_step_2layer", None)


def test_run_board_step_raises_without_handle_or_runner():
    import tikitorch.fpga._session as session
    # No runner ships by default; select any catalog entry to exercise the
    # explicit-registration gate.
    spec = next(s for s in fpga.KernelCache.default().specs()
                if not session.has_board_runner(s))
    # no handle
    with pytest.raises(fpga.NativeUnavailable):
        session.run_board_step(None, spec, object(), [0, 1], targets=[1, 2])
    # handle present but no runner registered for this kernel
    with pytest.raises(fpga.NativeUnavailable):
        session.run_board_step(object(), spec, object(), [0, 1], targets=[1, 2])


# --- native-endpoint-only --------------------------------------------------

@requires_model("TropicalTransformer")
@requires_fpga_native()
def test_native_endpoint_cannot_bypass_unsafe_legacy_layout_gate():
    # Endpoint availability must not bypass the structural gate. The retired
    # one-layer marshaller is not registered and the layout remains ineligible.
    with pytest.raises(fpga.NativeUnavailable):
        fpga.accelerate(_hls_lm(n_layers=1), strict=True)


# ---------------------------------------------------------------------------
# Dyadic weight -> software shift-schedule export
# ---------------------------------------------------------------------------
def test_dyadic_export_roundtrip_within_software_tolerance():
    """The exported schedule reconstructs weights within the verifier tolerance.

    This software check does not establish physical DSP use, timing, or a
    configuration-matched FPGA implementation.
    """
    from tikitorch.planning.muzero.networks import TropicalMuZeroNet
    net = TropicalMuZeroNet(75, 4, latent_dim=16, core_kind="dyadic", value_support=None)
    sched = fpga.export_net(net)
    assert len(sched) == 4                      # repr.enc {lin1,lin2} + pred.mix {lin1,lin2}
    assert fpga.verify_schedule(net, sched) == []     # empty == within configured tolerance
    # exponents are integers (bit-shifts), and reconstruct == sign * 2**exponent
    for ls in sched.values():
        assert ls["exponent"].dtype == torch.int16
        w = fpga.reconstruct_weight(ls)
        lg = torch.log2(w.abs().clamp_min(1e-30))
        assert torch.allclose(lg, lg.round())   # every weight is a pure power of two


def test_muzero_kernel_config_keys_but_does_not_bind():
    """MuZeroKernelConfig keys a TropicalMuZeroNet, but no KernelSpec is registered
    yet (the HLS kernel is the build task) -> match() returns None -> PyTorch fallback."""
    from tikitorch.planning.muzero.networks import TropicalMuZeroNet
    net = TropicalMuZeroNet(75, 4, latent_dim=16, core_kind="dyadic", value_support=None)
    cfg = fpga.MuZeroKernelConfig.from_model(net)
    assert cfg is not None and cfg.core_kind == "dyadic" and cfg.n_actions == 4
    assert fpga.KernelCache.default().match(net) is None   # honest: keyable != bindable
