"""Tests for the general backend seam (tikitorch.backend) — CPU-only, no board.

Covers the load-bearing backend guarantees: the CPU path is
byte-identical and zero-overhead (the seam is additive), device parsing mirrors
torch, and an unsupported device degrades to a one-time-warned CPU fallback rather
than an error.
"""
import warnings

import pytest
import torch

import tikitorch as tt
import tikitorch.backend as B
import tikitorch.functional as F


# --- logical Device -------------------------------------------------------

def test_device_parsing_and_value_semantics():
    assert B.device("cpu").is_cpu
    assert B.device(None).is_cpu
    d = B.device("fpga:0")
    assert d.type == "fpga" and d.index == 0
    assert repr(d) == "device(type='fpga', index=0)"
    assert str(d) == "fpga:0"
    assert str(B.device("fpga")) == "fpga"
    # value-based equality + hashable (keys the registries)
    assert B.device("fpga:0") == B.device("fpga:0")
    assert B.device("fpga:0") != B.device("fpga:1")
    assert len({B.device("fpga:0"), B.device("fpga:0")}) == 1
    # idempotent pass-through
    assert B.device(d) is d


@pytest.mark.parametrize("bad", ["fpga:x", "fpga:", ":0", "  ", "fpga:-1"])
def test_device_malformed_raises(bad):
    with pytest.raises(ValueError):
        B.device(bad)


# --- backend registry -----------------------------------------------------

def test_cpu_backend_registered_and_available():
    assert "cpu" in B.list_backends()
    assert "cpu" in B.available_backends()
    cpu = B.get_backend("cpu")
    assert cpu is not None and cpu.is_available()
    assert cpu.capabilities().op_dispatch is True
    # get_backend accepts a Device too
    assert B.get_backend(B.device("cpu")) is cpu
    # unknown backend -> None, never raises
    assert B.get_backend("does-not-exist") is None


# --- op auto-registration + byte-identical CPU dispatch -------------------

def test_cpu_ops_autoregistered_cover_functional():
    # Every dispatchable functional callable is registered as a CPU op EXCEPT the
    # *Fn autograd.Function classes (the .apply aliases are the dispatchable ops).
    registered = set(B.list_ops("cpu"))
    dispatchable = {
        name for name in F.__all__
        if callable(getattr(F, name, None))
        and not (isinstance(getattr(F, name), type)
                 and issubclass(getattr(F, name), torch.autograd.Function))
    }
    missing = dispatchable - registered
    assert not missing, f"functional ops not auto-registered on CPU: {sorted(missing)}"
    # the autograd CLASSES must NOT be registered (calling them would instantiate)
    assert "TropicalLinearFn" not in registered
    assert "tropical_linear" in registered  # but the apply-alias is


def test_dispatch_cpu_is_byte_identical():
    # dispatch returns the SAME object functional exports (zero overhead, no wrap).
    assert B.dispatch("tropical_linear", "cpu") is F.tropical_linear
    assert B.dispatch("tropical_dilation", "cpu") is F.tropical_dilation
    # and produces identical numerics
    x, w = torch.randn(4, 8), torch.randn(6, 8)
    assert torch.equal(B.dispatch("tropical_linear", "cpu")(x, w), F.tropical_linear(x, w))


def test_dispatch_unknown_device_falls_back_once():
    B._reset_fallback_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        impl = B.dispatch("tropical_linear", "totally-unknown-dev")
        again = B.dispatch("tropical_linear", "totally-unknown-dev")
    assert impl is F.tropical_linear and again is F.tropical_linear
    fb = [w for w in caught if issubclass(w.category, B.BackendFallbackWarning)]
    assert len(fb) == 1, f"expected exactly one fallback warning, got {len(fb)}"


def test_dispatch_unknown_op_raises_keyerror():
    with pytest.raises(KeyError):
        B.dispatch("not_a_real_op", "cpu")


def test_dispatch_survives_backend_is_available_raising():
    # A misbehaving backend whose is_available() throws must NOT break dispatch /
    # is_op_supported — they degrade to the CPU fallback ("correctness never breaks").
    class Flaky(B.Backend):
        name = "flaky-test-backend"

        def is_available(self):
            raise RuntimeError("driver on fire")

        def capabilities(self):
            return B.BackendCapabilities(op_dispatch=True)

        def ops(self):
            return {"tropical_linear": (lambda *a, **k: "device-impl")}

    try:
        Flaky().register()
        B._reset_fallback_warnings()
        # dispatch returns the CPU impl instead of propagating the exception
        assert B.dispatch("tropical_linear", "flaky-test-backend") is F.tropical_linear
        # pure query returns False instead of raising
        assert B.is_op_supported("tropical_linear", "flaky-test-backend") is False
        # enumeration also skips it rather than crashing
        assert "flaky-test-backend" not in B.available_backends()
    finally:
        B.registry._BACKENDS.pop("flaky-test-backend", None)
        B.registry._OPS.pop(("tropical_linear", "flaky-test-backend"), None)


def test_is_op_supported_is_pure_query():
    B._reset_fallback_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert B.is_op_supported("tropical_linear", "cpu") is True
        assert B.is_op_supported("tropical_linear", "unknown-dev") is False
        assert B.is_op_supported("not_a_real_op", "cpu") is False
    assert not [w for w in caught if issubclass(w.category, B.BackendFallbackWarning)]


# --- to() + default device ------------------------------------------------

def test_to_cpu_is_identity():
    m = torch.nn.Linear(3, 3)
    assert tt.to(m, "cpu") is m
    assert tt.to(m, B.device("cpu")) is m


def test_to_tensor_logical_device_is_noop_with_warning():
    B._reset_fallback_warnings()
    x = torch.randn(2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tt.to(x, "some-accelerator")
    assert out is x  # tensors never move to a logical device
    assert any(issubclass(w.category, B.BackendFallbackWarning) for w in caught)


def test_default_device_scoping():
    assert B.get_default_device().is_cpu
    with B.default_device("fpga"):
        assert B.get_default_device().type == "fpga"
    assert B.get_default_device().is_cpu  # restored


# --- custom backend registration (the extension point) --------------------

def test_custom_backend_registers_and_dispatches():
    sentinel = object()

    class ToyBackend(B.Backend):
        name = "toy-test-backend"

        def is_available(self):
            return True

        def capabilities(self):
            return B.BackendCapabilities(op_dispatch=True)

        def ops(self):
            return {"tropical_linear": (lambda *a, **k: sentinel)}

    try:
        ToyBackend().register()
        assert "toy-test-backend" in B.available_backends()
        impl = B.dispatch("tropical_linear", "toy-test-backend")
        assert impl() is sentinel  # the toy impl, not the CPU one
    finally:
        # clean up global state so other tests are unaffected
        B.registry._BACKENDS.pop("toy-test-backend", None)
        B.registry._OPS.pop(("tropical_linear", "toy-test-backend"), None)
