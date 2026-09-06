"""Focused regressions for public-release correctness fixes."""

import importlib
import queue
import threading
import time

import numpy as np
import pytest
import torch

import tikitorch as tt
import tikitorch.backend as backend
import tikitorch.planning as planning
from tikitorch.fabric import PipelineRunner, make_thread_links
from tikitorch.fabric.stages import make_linear_pipeline
from tikitorch.fabric.tcp_link import TcpLink, WireError


def test_morphssm_projection_selectors_are_real_and_validated():
    common = dict(d_model=8, n_layers=1, input_dim=1, num_classes=2)
    default = tt.models.MorphSSM(**common)
    tropical = tt.models.MorphSSM(**common, tropical_proj=True)
    dense = tt.models.MorphSSM(**common, tropical_proj=False)
    identity = tt.models.MorphSSM(**common, out_proj_type="identity")

    assert isinstance(default.layers[0].out_proj, tt.nn.TropicalDilationModule)
    assert isinstance(tropical.layers[0].out_proj, tt.nn.TropicalDilationModule)
    assert isinstance(dense.layers[0].out_proj, torch.nn.Linear)
    assert isinstance(identity.layers[0].out_proj, torch.nn.Identity)
    with pytest.raises(ValueError, match="out_proj_type"):
        tt.models.MorphSSM(**common, out_proj_type="typo")
    with pytest.raises(ValueError, match="conflicting"):
        tt.models.MorphSSM(
            **common, tropical_proj=False, out_proj_type="tropical"
        )


def test_hardware_cost_follows_projection_types():
    common = dict(
        d_model=8,
        n_layers=1,
        input_dim=1,
        num_classes=2,
        use_soft=False,
    )
    tropical = tt.models.MorphSSM(**common)
    dense = tt.models.MorphSSM(**common, out_proj_type="linear")
    assert tropical.hardware_cost(batch_size=2, seq_len=4).multiplications == 0
    assert dense.hardware_cost(batch_size=2, seq_len=4).multiplications > 0

    for model in (
        tt.models.TropicalMLP(
            d_model=8, n_layers=1, input_dim=4, num_classes=2,
            use_softmin=False,
        ),
        tt.models.TropicalSequenceModel(
            d_model=8, n_layers=1, input_dim=1, seq_len=4,
            num_classes=2, use_softmin=False,
        ),
        tt.models.MorphMLP(
            d_model=8, n_layers=2, input_dim=4, num_classes=2,
            use_soft=False,
        ),
    ):
        assert model.hardware_cost(batch_size=2, seq_len=4).multiplications == 0


def test_morphssm_hardware_cost_counts_non_dyadic_residual_scaling():
    common = dict(
        d_model=8,
        input_dim=1,
        num_classes=2,
        use_soft=False,
        tau=0.0,
        dropout=0.0,
        residual_scaling=True,
    )
    non_dyadic = tt.models.MorphSSM(**common, n_layers=2)
    dyadic = tt.models.MorphSSM(**common, n_layers=4)

    # Each of two layers scales every B*T*D residual element by 1/sqrt(2).
    assert non_dyadic.hardware_cost(batch_size=2, seq_len=4).multiplications == 128
    # 1/sqrt(4) == 1/2, so this residual scaling is a shift.
    assert dyadic.hardware_cost(batch_size=2, seq_len=4).multiplications == 0


def test_tropical_mlp_hardening_hook_and_audit_restore_soft_training_path():
    model = tt.models.TropicalMLP(
        d_model=16, n_layers=2, input_dim=8, num_classes=3,
        use_softmin=True,
    )
    assert all(layer.use_softmin for layer in model.layers)
    hits = tt.find_runtime_multiplies(
        model, torch.rand(2, 8), harden_tau=True
    )
    assert hits == []
    assert all(layer.use_softmin for layer in model.layers)
    assert model.temperature == 1.0

    model.set_tau(0)
    assert all(not layer.use_softmin for layer in model.layers)


def test_transformer_soft_temperature_is_applied_at_construction():
    block = tt.nn.TropicalTransformerBlock(
        d_model=8, n_heads=2, d_ff=16, tau=0.7, dropout=0.0
    )
    assert block.attn.tau == 0.7
    assert block.attn.q_proj.use_softmax is True
    assert block.ffn_up.use_softmax is True
    assert block.ffn_down.use_softmin is True

    mote = tt.models.MoTEModel(
        d_model=8,
        n_experts=2,
        expert_type="transformer",
        expert_layers=1,
        d_ff=16,
        n_heads=2,
        n_shared_pre=0,
        n_shared_post=0,
        use_soft=True,
        tau=0.7,
        dropout=0.0,
        input_dim=1,
        num_classes=2,
    )
    expert = mote.mote_block.experts[0][0]
    assert expert.attn.tau == 0.7
    assert expert.ffn_up.use_softmax is True
    assert expert.ffn_down.use_softmin is True


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.75])
def test_dilation_erosion_mixture_has_shift_add_implementation(lam):
    layer = tt.nn.DilationErosionPerceptron(4, 3, lam=lam)
    assert tt.find_runtime_multiplies(layer, torch.randn(2, 4)) == []


@pytest.mark.parametrize("lam", [0.2, 1 / 3, 1.0])
def test_dilation_erosion_mixture_rejects_unimplemented_coefficients(lam):
    with pytest.raises(ValueError, match="shift-add"):
        tt.nn.DilationErosionPerceptron(4, 3, lam=lam)


def _fabric_batch(n=8):
    rng = np.random.default_rng(0)
    return (
        [rng.standard_normal(3) for _ in range(n)],
        [rng.standard_normal(2) for _ in range(n)],
    )


def test_pipeline_handles_bounded_queues():
    stages = make_linear_pipeline([3, 4, 2], seed=1)
    result = PipelineRunner(
        stages, make_thread_links(2, maxsize=1), window=8
    ).run(*_fabric_batch())
    assert len(result.losses) == 8


def test_pipeline_propagates_worker_failure_without_deadlock():
    stages = make_linear_pipeline([3, 4, 2], seed=1)
    original = stages[1].forward

    def fail_on_third(tag, activation):
        if tag == 2:
            raise RuntimeError("intentional stage failure")
        return original(tag, activation)

    stages[1].forward = fail_on_third
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="intentional stage failure"):
        PipelineRunner(
            stages, make_thread_links(2, maxsize=1), window=8
        ).run(*_fabric_batch())
    assert time.monotonic() - started < 2.0


def test_double_buffered_transport_delay_is_interruptible_on_failure():
    transport_started = threading.Event()

    class SlowTransport:
        @staticmethod
        def time_for(_nbytes):
            transport_started.set()
            return 2.5

    stages = make_linear_pipeline([3, 4, 2], seed=1)
    original = stages[0].forward

    def fail_after_transfer_starts(tag, activation):
        if tag == 0:
            return original(tag, activation)
        assert transport_started.wait(1.0)
        raise RuntimeError("intentional stage failure")

    stages[0].forward = fail_after_transfer_starts
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="intentional stage failure"):
        PipelineRunner(
            stages,
            make_thread_links(2, double_buffer=True, transport=SlowTransport()),
            window=8,
        ).run(*_fabric_batch())
    assert time.monotonic() - started < 2.0


def test_link_tag_mismatches_are_runtime_protocol_errors():
    links = make_thread_links(1)
    links[0].feed_forward(2, np.ones(1))
    with pytest.raises(RuntimeError, match="want 1 got 2"):
        links[0].recv_forward(1)

    q = queue.Queue()
    tcp = TcpLink(1, 2, fwd_in=q)
    q.put((2, np.ones(1)))
    with pytest.raises(WireError, match="want 1 got 2"):
        tcp.recv_forward(1)


def test_sparse_dictionary_remains_signed_dyadic_after_normalization():
    model = tt.nn.SparseCodingPC(n_pixels=8, n_atoms=5, n_active=2)
    dictionary = model._dict()
    nonzero = dictionary[dictionary != 0].abs()
    assert torch.allclose(torch.log2(nonzero), torch.log2(nonzero).round())
    shadow_norms = torch.linalg.vector_norm(model.D, dim=0)
    assert ((shadow_norms >= 2 ** -0.5) & (shadow_norms <= 2 ** 0.5)).all()

    model.train_step(torch.rand(4, 8), n_steps=2)
    deployed = model._dict()
    nonzero = deployed[deployed != 0].abs()
    assert torch.allclose(torch.log2(nonzero), torch.log2(nonzero).round())
    shadow_norms = torch.linalg.vector_norm(model.D, dim=0)
    assert ((shadow_norms >= 2 ** -0.5) & (shadow_norms <= 2 ** 0.5)).all()


def test_cascade_options_fail_closed():
    model = tt.nn.DyadicPCNet([4, 4, 2])
    x = torch.randn(2, 4)
    target = model._make_target(torch.tensor([0, 1]), ref=x)
    with pytest.raises(ValueError, match="mode"):
        model.cascade_grads(x, target, mode="typo")
    with pytest.raises(ValueError, match="loss"):
        model.cascade_grads(x, target, loss="typo")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_simulations": 0},
        {"backup": "typo"},
        {"topk_k": 0},
        {"players": 3},
        {"root_exploration_frac": 1.1},
    ],
)
def test_mcts_rejects_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        planning.MCTSConfig(**kwargs)


def test_mcts_terminal_root_returns_without_selecting_an_action():
    from tikitorch.planning.muzero.env import GridWorldNav, KnownModel

    env = GridWorldNav(["SG"])
    model = KnownModel(env, value_fn=lambda state: 7.0)
    policy, value = planning.TropicalMCTS(
        model, planning.MCTSConfig(num_simulations=2)
    ).run(env.goal_state)
    assert torch.equal(policy, torch.zeros(env.n_actions))
    assert value == 7.0


def test_indexed_backend_strings_resolve_by_device_type():
    assert backend.get_backend("cpu:0") is backend.get_backend("cpu")
    assert backend.dispatch("tropical_linear", "cpu:7") is tt.functional.tropical_linear
    assert backend.device(" CPU:2 ") == backend.Device("cpu", 2)


def test_native_probe_always_closes_an_open_handle():
    from tikitorch.fpga._native import NativeHandle

    class FakeLibrary:
        closed = []

        def tt_fpga_open(self, _):
            return 42

        def tt_fpga_board_present(self, _):
            raise RuntimeError("probe failed")

        def tt_fpga_close(self, handle):
            self.closed.append(handle)

    lib = FakeLibrary()
    assert NativeHandle(lib, "fake").board_present() is False
    assert lib.closed == [42]


def test_fallback_train_step_requires_optimizer_before_backward():
    from tikitorch.fpga.accelerate import AcceleratedModel

    class Tiny(torch.nn.Linear):
        def compute_loss(self, x, y):
            return torch.nn.functional.mse_loss(self(x), y)

    model = Tiny(2, 1)
    accelerated = AcceleratedModel(
        model, spec=None, native=None, device=tt.device("fpga")
    )
    with pytest.raises(ValueError, match="requires an optimizer"):
        accelerated.train_step(torch.ones(1, 2), torch.ones(1, 1))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_board_runner_registration_validates_and_prevents_accidental_replace():
    import tikitorch.fpga._session as session

    entry = "release_correctness_test_kernel"
    first = lambda *args, **kwargs: 0.0
    second = lambda *args, **kwargs: 1.0
    try:
        with pytest.raises(TypeError, match="callable"):
            session.register_board_runner(entry, None)
        session.register_board_runner(entry, first)
        with pytest.raises(ValueError, match="already registered"):
            session.register_board_runner(entry, second)
        session.register_board_runner(entry, second, replace=True)
        assert session.BOARD_RUNNERS[entry] is second
    finally:
        session.BOARD_RUNNERS.pop(entry, None)


def test_public_all_contracts_are_unique_and_resolve():
    modules = [
        "tikitorch",
        "tikitorch.algorithmic",
        "tikitorch.audit",
        "tikitorch.autodiff",
        "tikitorch.autodiff.packed_tape",
        "tikitorch.autodiff.primitives",
        "tikitorch.backend",
        "tikitorch.backend.registry",
        "tikitorch.classical",
        "tikitorch.fabric",
        "tikitorch.forward_forward",
        "tikitorch.fpga",
        "tikitorch.fpga.dyadic_export",
        "tikitorch.functional",
        "tikitorch.geometry",
        "tikitorch.hardware",
        "tikitorch.models",
        "tikitorch.nn",
        "tikitorch.nn.predictive_coding",
        "tikitorch.optim",
        "tikitorch.planning",
        "tikitorch.planning.bellman",
        "tikitorch.planning.dp",
        "tikitorch.planning.functional",
        "tikitorch.planning.mcts",
        "tikitorch.planning.muzero",
        "tikitorch.planning.muzero.agent",
        "tikitorch.planning.muzero.board_agent",
        "tikitorch.planning.muzero.games",
        "tikitorch.planning.muzero.networks",
        "tikitorch.planning.muzero.networks_dense",
        "tikitorch.planning.muzero.replay",
        "tikitorch.planning.muzero.selfplay",
        "tikitorch.profile",
        "tikitorch.prune",
        "tikitorch.serialization",
        "tikitorch.train",
    ]
    for name in modules:
        module = importlib.import_module(name)
        exports = getattr(module, "__all__", ())
        assert len(exports) == len(set(exports)), f"duplicate export in {name}"
        missing = [symbol for symbol in exports if not hasattr(module, symbol)]
        assert missing == [], f"unresolved exports in {name}: {missing}"
        assert all(getattr(module, symbol) is not None for symbol in exports)
