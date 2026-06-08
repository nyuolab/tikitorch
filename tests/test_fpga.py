"""Tests for the FPGA backend (tikitorch.fpga).

The board-dependent assertions are skip-guarded with ``requires_fpga_native`` and
run only on an F2 instance with a built native driver. Everything else exercises
the always-present CPU-fallback path and the pure-Python kernel catalog, so the
suite is green (and honest) in this repo / CI.
"""
import pathlib
import warnings

import pytest
import torch

import tikitorch as tt
import tikitorch.backend as B

from conftest import requires_fpga_native, requires_model, requires_optim

fpga = pytest.importorskip("tikitorch.fpga")


# --- availability + import safety -----------------------------------------

def test_fpga_imports_and_registers():
    assert tt.fpga is not None
    assert "fpga" in B.list_backends()
    # registered but not available without a board
    assert fpga.is_available() is False
    assert fpga.device_count() == 0
    assert "fpga" not in B.available_backends()


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
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
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
    assert cache.match(tt.models.TropicalTransformer(
        d_model=64, n_heads=4, n_layers=2, task_type="lm", vocab_size=128)) is None
    assert cache.match(tt.models.TropicalTransformer(
        d_model=128, n_heads=4, n_layers=5, task_type="lm", vocab_size=128)) is None
    # classification (no nn.Embedding) -> no LM match
    assert cache.match(tt.models.TropicalTransformer(
        d_model=128, n_heads=4, n_layers=2, task_type="classification")) is None


@requires_model("MorphSSM")
def test_kernelcache_morph_uses_separate_schema_and_abi():
    cache = fpga.KernelCache.default()
    m = tt.models.MorphSSM(d_model=64, n_layers=4, task_type="classification",
                           input_dim=1, num_classes=10)
    spec = cache.match(m)
    assert spec is not None
    assert spec.entry == "forge_train_step_morph"
    assert spec.abi is fpga.KernelABI.FUSED_5     # 5-int ABI (no layer_id)
    assert isinstance(spec.config, fpga.MorphKernelConfig)
    assert not isinstance(spec.config, fpga.LMKernelConfig)


def test_two_abis_are_distinct():
    assert fpga.KernelABI.FUSED_5.has_layer_id is False
    assert fpga.KernelABI.FUSED_6.has_layer_id is True
    assert fpga.KernelABI.FUSED_5.n_int_args == 4
    assert fpga.KernelABI.FUSED_6.n_int_args == 5


# --- AcceleratedModel: the DDP-style wrapper, CPU fallback ----------------

@requires_model("TropicalTransformer")
def test_accelerate_returns_unbound_wrapper():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    acc = fpga.accelerate(m)
    assert isinstance(acc, fpga.AcceleratedModel)
    assert isinstance(acc, torch.nn.Module)
    assert acc.bound is False                 # no board / no on-board runner
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
def test_strict_raises_without_board():
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


# --- the .fpga() ergonomic mover (FpgaAccelerable mixin) ------------------

@requires_model("TropicalTransformer")
def test_model_dot_fpga_returns_wrapper_and_matches_kernel():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    am = m.fpga()                                   # the one-liner
    assert isinstance(am, fpga.AcceleratedModel)
    assert am.module is m
    assert am.kernel_spec is not None and am.kernel_spec.entry == "forge_train_step_2layer"
    assert am.bound is False                        # no board/runner here -> CPU fallback


@requires_model("MorphSSM")
def test_morphssm_dot_fpga_uses_morph_kernel():
    m = tt.models.MorphSSM(d_model=64, n_layers=4, task_type="lm", vocab_size=128)
    am = m.fpga()
    assert am.kernel_spec is not None and am.kernel_spec.entry == "forge_train_step_morph"


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
def test_dot_fpga_strict_raises_without_board():
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    with pytest.raises(fpga.NativeUnavailable):
        m.fpga(strict=True)


# --- on-board runner path (faked native handle; no hardware) --------------
# These cover the rewritten bound/train_step delegation that is otherwise
# reachable only on a real board, using a sentinel native handle + a registered
# runner so no FPGA is required.

@requires_model("TropicalTransformer")
def test_bound_train_step_delegates_to_registered_runner():
    import tikitorch.fpga._session as session
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
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
    # Pick a kernel that has NO registered runner (the 1-layer forge_train_step now
    # ships a reference runner; the 3-layer does not), so the "no runner" branch is
    # exercised honestly.
    spec = next(s for s in fpga.KernelCache.default().specs()
                if not session.has_board_runner(s))
    # no handle
    with pytest.raises(fpga.NativeUnavailable):
        session.run_board_step(None, spec, object(), [0, 1], targets=[1, 2])
    # handle present but no runner registered for this kernel
    with pytest.raises(fpga.NativeUnavailable):
        session.run_board_step(object(), spec, object(), [0, 1], targets=[1, 2])


# --- board-only (skipped without an F2 + native driver) -------------------

@requires_model("TropicalTransformer")
@requires_fpga_native()
def test_on_board_train_step_is_zero_dsp_and_trains():
    import tikitorch.optim as optim
    m = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=2,
                                      task_type="lm", vocab_size=128)
    acc = fpga.accelerate(m, optimizer=optim.PureTropicalSGD(m.parameters()), strict=True)
    assert acc.bound is True
    assert acc.kernel_spec.dsp == 0
    x = torch.randint(0, 128, (4, 64))
    first = acc.train_step(x, x)
    for _ in range(20):
        last = acc.train_step(x, x)
    assert last <= first
