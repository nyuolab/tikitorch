"""Tests for tikitorch.audit — the multiplier-free closure checker."""
import copy

import pytest
import torch
import torch.nn as tnn

from tikitorch.audit import assert_closed, find_closure_violations


def test_flags_standard_torch_modules():
    model = tnn.Sequential(tnn.Linear(4, 4), tnn.ReLU(), tnn.LayerNorm(4))
    types = [t for _, t in find_closure_violations(model)]
    assert "Linear" in types and "LayerNorm" in types
    with pytest.raises(AssertionError):
        assert_closed(model)


def test_embedding_and_inert_modules_pass():
    # Embedding is a pure gather (no multiply); dropout/relu are inert.
    model = tnn.Sequential(tnn.Embedding(10, 4), tnn.Dropout(0.1), tnn.ReLU())
    assert find_closure_violations(model) == []
    assert_closed(model) is model


def test_allow_declares_boundary():
    model = tnn.Linear(4, 4)
    with pytest.raises(AssertionError):
        assert_closed(model)
    # allow by type name
    assert_closed(model, allow={"Linear"}) is model


def test_tropical_modules_are_closed():
    import tikitorch.nn as nn
    closed = tnn.Sequential(
        nn.TropicalConv2d(1, 4, 3, padding=1),
        nn.TropicalLayerNorm2d(4),
        nn.TropicalMaxPool2d(2),
    )
    assert find_closure_violations(closed) == []
    assert_closed(closed)


def test_gated_and_parametric_activations_flagged():
    # Soundness fix: GLU/PReLU carry data-dependent multiplies and must be caught.
    import torch.nn as tnn
    for layer in (tnn.GLU(), tnn.PReLU()):
        from tikitorch.audit import find_closure_violations
        assert find_closure_violations(layer), f"{type(layer).__name__} should be flagged"


def test_allow_is_mro_aware():
    # MultiheadAttention's internal out_proj is a Linear subclass; allow={'Linear'}
    # plus the MHA type should let an MHA-containing model pass.
    import torch.nn as tnn
    from tikitorch.audit import assert_closed
    m = tnn.MultiheadAttention(8, 2, batch_first=True)
    assert_closed(m, allow={"MultiheadAttention", "Linear"})


# --- runtime closure checker (traces actual aten ops; catches raw * / @ / exp) ---

def test_runtime_checker_passes_closed_models():
    import tikitorch as tt
    from tikitorch.audit import find_runtime_multiplies
    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    assert find_runtime_multiplies(cnn, torch.rand(1, 1, 28, 28)) == []   # pow2 rescales = shifts
    mssm = tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm", vocab_size=128)
    assert find_runtime_multiplies(mssm, torch.randint(0, 128, (2, 16))) == []


def test_runtime_checker_passes_rope_and_iter_topk_reasoner():
    import tikitorch.algorithmic as A
    from tikitorch.audit import assert_closed_runtime
    m = A.build_reasoner("running_max", d_model=32, n_layers=2, attention="tropical",
                         agg_variant="iter_topk", agg_topk_k=3, pos_mode="rope_shift")
    x, _ = A.gen_running_max(2, 20, n_symbols=20, seed=0)
    assert_closed_runtime(m, x)                                  # closure holds at runtime


def test_runtime_checker_catches_real_multiply():
    import pytest
    from tikitorch.audit import assert_closed_runtime, find_runtime_multiplies

    class Bad(torch.nn.Module):           # a non-pow2 scalar multiply (real DSP op)
        def forward(self, x):
            return x * 1.3
    with pytest.raises(AssertionError):
        assert_closed_runtime(Bad(), torch.randn(3))

    # an nn.Linear (dense matmul) is caught too
    lin = torch.nn.Sequential(torch.nn.Linear(4, 4))
    hits = find_runtime_multiplies(lin, torch.randn(2, 4))
    assert any(c == "dense-multiply" for c, _ in hits)


def test_runtime_checker_pow2_multiply_is_a_shift():
    from tikitorch.audit import find_runtime_multiplies

    class Pow2(torch.nn.Module):          # x * 0.25 and x / 8 are shifts, not multiplies
        def forward(self, x):
            return (x * 0.25) / 8.0
    assert find_runtime_multiplies(Pow2(), torch.randn(3)) == []


@pytest.mark.parametrize(
    "operation",
    [
        lambda: torch.nn.functional.cross_entropy(
            torch.randn(3, 5), torch.tensor([1, 2, 0])
        ),
        lambda: torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(torch.randn(3, 5), dim=-1),
            torch.tensor([1, 2, 0]),
        ),
        lambda: torch.nn.functional.mse_loss(torch.randn(3), torch.randn(3)),
        lambda: torch.softmax(torch.randn(3, 5), dim=-1),
        lambda: torch.nn.functional.layer_norm(torch.randn(3, 5), (5,)),
        lambda: torch.nn.functional.gelu(torch.randn(3)),
        lambda: torch.nn.functional.silu(torch.randn(3)),
        lambda: torch.randn(3).mean(),
        lambda: torch.randn(2, 3).mean(dim=()),
        lambda: torch.randn(3).var(),
        lambda: torch.randn(3).prod(),
        lambda: torch.randn(3).abs().sqrt(),
        lambda: torch.randn(3).abs().rsqrt(),
        lambda: torch.add(torch.randn(3), torch.randn(3), alpha=0.3),
        lambda: torch.log2(torch.rand(3)),
    ],
    ids=[
        "cross-entropy", "log-softmax-nll", "mse", "softmax", "layer-norm",
        "gelu", "silu", "non-dyadic-mean", "empty-dim-mean", "variance", "product", "sqrt",
        "rsqrt", "add-alpha", "standalone-log2",
    ],
)
def test_guard_catches_fused_and_high_level_arithmetic(operation):
    import tikitorch as tt

    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            operation()


def test_tensor_values_do_not_establish_shift_provenance():
    """Runtime tensors that happen to contain powers of two remain data-dependent."""
    import tikitorch as tt

    left = torch.ones(8)
    right = torch.full((8,), 0.5)
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            left * right

    # In contrast, a literal scalar and an explicitly rounded exponent have
    # operation-derived shift provenance and are closure-clean.
    with tt.assert_no_multiply():
        left * 0.5
        exponent = torch.round(torch.randn(8))
        left * torch.exp2(exponent)


def test_clamp_drops_shift_provenance():
    """Clamp can turn a dyadic factor into a non-dyadic boundary value."""
    import tikitorch as tt

    x = torch.ones(8)
    z = torch.full((8,), 2.0)
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            x * torch.exp2(torch.round(z)).clamp_max(3.0)

    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            shift = torch.exp2(torch.round(z))
            shift.clamp_max_(3.0)
            x * shift


def test_zero_is_a_gate_for_multiply_but_never_a_shift_for_divide():
    import tikitorch as tt

    x = torch.arange(1.0, 4.0)
    with tt.assert_no_multiply():
        x * 0.0
        x * torch.zeros_like(x)
        x * torch.tensor([0.0, 0.5, 2.0])

    divisors = [
        lambda: 0,
        lambda: 0.0,
        lambda: torch.tensor(0.0),
        lambda: torch.zeros_like(x),
        lambda: torch.tensor([0.0, 0.5, 2.0]),
    ]
    for make_divisor in divisors:
        with pytest.raises(tt.ClosureViolation):
            with tt.assert_no_multiply():
                x / make_divisor()

    # A freshly constructed, wholly nonzero dyadic literal is still a bank of
    # static shifts and remains accepted as a divisor.
    with tt.assert_no_multiply():
        x / torch.tensor([1.0, 0.5, 2.0])


def test_literal_dyadic_scaling_composes_only_trusted_factor_provenance():
    import tikitorch as tt

    x = torch.arange(1.0, 5.0)
    with tt.assert_no_multiply():
        gate = (x > 2).to(x.dtype)
        x * (gate / 0.5)
        x * (torch.sign(x - 2) * 0.25)
        x * (torch.tensor([0.0, 0.5, 1.0, 2.0]) / 2.0)
        x / (torch.ones_like(x) * 0.5)

    # Scaling an arbitrary runtime tensor by a dyadic literal makes that one
    # operation a shift; it does not turn the arbitrary result into a trusted
    # factor for a subsequent tensor-by-tensor multiply.
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            x * (torch.randn(4) / 0.5)


def test_dynamic_integer_multiply_and_divide_are_not_logic_ops():
    import tikitorch as tt

    left = torch.arange(1, 9)
    right = torch.arange(2, 10)
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            left * right
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            left // right

    # Boolean gating remains LUT/logic rather than numeric multiplication.
    with tt.assert_no_multiply():
        (left > 3) * (right < 7)


def test_fused_attention_ops_are_conservative_boundaries():
    import tikitorch as tt

    query = torch.randn(2, 4, 8, 16)
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            torch.nn.functional.scaled_dot_product_attention(query, query, query)

    attention = tnn.MultiheadAttention(16, 4, batch_first=True).eval()
    value = torch.randn(2, 8, 16)
    with pytest.raises(tt.ClosureViolation):
        with torch.no_grad(), tt.assert_no_multiply():
            attention(value, value, value, need_weights=False)


def test_fused_embedding_bag_is_a_conservative_boundary():
    import tikitorch as tt

    weight = torch.randn(5, 3)
    indices = torch.tensor([0, 1, 2, 3])
    offsets = torch.tensor([0, 2])
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            torch.nn.functional.embedding_bag(
                indices, weight, offsets, mode="mean"
            )


@pytest.mark.parametrize(
    ("shape", "mode"),
    [
        ((1, 1, 4), "linear"),
        ((1, 1, 4, 4), "bilinear"),
        ((1, 1, 4, 4), "bicubic"),
        ((1, 1, 2, 2, 2), "trilinear"),
    ],
)
def test_interpolation_arithmetic_is_flagged(shape, mode):
    import tikitorch as tt

    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            torch.nn.functional.interpolate(
                torch.randn(shape), scale_factor=2, mode=mode, align_corners=False
            )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: torch.linalg.solve(torch.eye(2), torch.ones(2)),
        lambda: torch.linalg.vector_norm(torch.ones(4)),
        lambda: torch.linalg.det(torch.eye(2)),
    ],
    ids=["solve", "vector-norm", "determinant"],
)
def test_linalg_arithmetic_is_flagged(operation):
    import tikitorch as tt

    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            operation()


def test_power_of_two_mean_and_add_alpha_are_static_shifts():
    import tikitorch as tt

    with tt.assert_no_multiply():
        torch.randn(8).mean()
        torch.add(torch.randn(8), torch.randn(8), alpha=-0.25)


# --- Level-B training-closure check (forward + loss + backward + optimizer) ---

def _running_max_setup(d_model=24):
    import tikitorch.algorithmic as A
    import tikitorch.functional as F
    # Both the batch mean (8) and flattened loss mean (8*16) are dyadic shifts.
    x, y = A.gen_running_max(8, 16, n_symbols=12, seed=0)
    m = A.build_reasoner("running_max", d_model=d_model, n_layers=1,
                         attention="tropical", agg_variant="argmax", pos_mode="none")
    C = A.get_task("running_max").n_symbols
    loss_fn = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, C), yy.reshape(-1))
    return m, x, y, loss_fn


def test_training_closure_perceptron_puretropicalsgd_has_no_multiply():
    # This concrete recipe uses power-of-two reduction extents, so its means are
    # fixed shifts and the whole traced step is clean without blanket exemptions.
    import tikitorch.optim as optim
    from tikitorch.audit import find_training_multiplies
    m, x, y, loss_fn = _running_max_setup(d_model=32)   # pow2 d_model: norm mean = shift
    opt = optim.PureTropicalSGD(m.parameters(), lr_shift=4)
    hits = find_training_multiplies(m, (x,), y, loss_fn, optimizer=opt)
    assert hits == []


def test_training_closure_catches_adam_and_cross_entropy():
    import pytest
    import tikitorch.algorithmic as A
    from tikitorch.audit import assert_training_closed, find_training_multiplies
    m, x, y, _ = _running_max_setup()
    C = A.get_task("running_max").n_symbols
    ce = lambda p, yy: torch.nn.functional.cross_entropy(p.reshape(-1, C), yy.reshape(-1))
    adam = torch.optim.Adam(m.parameters(), lr=1e-3)
    hits = find_training_multiplies(m, (x,), y, ce, optimizer=adam)
    assert hits, "Adam + cross-entropy must be flagged as non-closed training"
    with pytest.raises(AssertionError):
        assert_training_closed(m, (x,), y, ce, optimizer=adam)


def test_training_closure_catches_signsgd_float_momentum():
    # SignSGD's float momentum does buf.mul_(0.9) -- a real multiply, so it is NOT
    # Level-B closed (this is the bug the checker is designed to surface).
    import tikitorch.optim as optim
    from tikitorch.audit import find_training_multiplies
    m, x, y, loss_fn = _running_max_setup()
    opt = optim.SignSGD(m.parameters(), lr=2 ** -4, momentum=0.9)
    # Audits are observational and now roll optimizer state back. Build the
    # momentum buffer with one intentional, ordinary training step first.
    opt.zero_grad()
    loss_fn(m(x), y).backward()
    opt.step()
    hits = find_training_multiplies(m, (x,), y, loss_fn, optimizer=opt)
    assert any("mul" in op for _, op in hits), "float-momentum SignSGD should not be closed"


def test_training_zero_dsp_does_not_exempt_division():
    import tikitorch as tt

    x = torch.randn(3, requires_grad=True)
    report = tt.closure_report(
        tnn.Identity(), x, torch.zeros_like(x), lambda pred, _target: pred.sum() / 3
    )
    assert any(category == "elementwise-divide" for category, _ in report["training"])
    assert report["training_closed"] is False
    assert report["training_zero_dsp"] is False


def _assert_state_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_state_tree_equal(lhs, rhs)
    else:
        assert left == right


def test_executed_audits_and_profile_restore_all_observable_state():
    from tikitorch.audit import find_runtime_multiplies, find_training_multiplies
    from tikitorch.profile import profile_model

    class Stateful(tnn.Module):
        def __init__(self):
            super().__init__()
            self.linear = tnn.Linear(3, 2)
            self.register_buffer("calls", torch.zeros((), dtype=torch.long))
            self.tau = 0.75
            self.use_soft = True

        def set_tau(self, tau):
            self.tau = float(tau)
            self.use_soft = tau > 0

        def forward(self, value):
            self.calls.add_(1)
            return self.linear(value) + torch.rand((), device=value.device)

    model = Stateful()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x, target = torch.randn(4, 3), torch.randn(4, 2)

    # Populate optimizer state outside the audit, then install an existing grad
    # whose object identity and values must survive the audit.
    optimizer.zero_grad()
    tnn.functional.mse_loss(model(x), target).backward()
    optimizer.step()
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 7.0)
    original_grad = parameter.grad

    model.eval()
    model.linear.train()  # deliberately mixed child mode
    expected_modes = [module.training for module in model.modules()]
    expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_rng = torch.get_rng_state().clone()

    find_runtime_multiplies(model, x)
    profile_model(model, x)
    find_training_multiplies(
        model, x, target, tnn.functional.mse_loss, optimizer=optimizer
    )

    assert [module.training for module in model.modules()] == expected_modes
    assert model.tau == 0.75 and model.use_soft is True
    assert parameter.grad is original_grad
    assert torch.equal(parameter.grad, torch.full_like(parameter, 7.0))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected_model[key])
    _assert_state_tree_equal(optimizer.state_dict(), expected_optimizer)


def test_runtime_audit_restores_inputs_and_private_mutable_module_state():
    from tikitorch.audit import find_runtime_multiplies

    class Mutating(tnn.Module):
        def __init__(self):
            super().__init__()
            self._private_config = {"events": ["before"]}
            self.history = [1]
            self._cache = torch.tensor([2.0])

        def forward(self, value):
            self._last_aux_loss = value.sum()
            self.new_public_cache = "temporary"
            self._private_config["events"].append("audit")
            self.history.append(2)
            self._cache.add_(3)
            value.add_(10)
            return value

    model = Mutating()
    private_config = model._private_config
    history = model.history
    cache = model._cache
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    x.grad = torch.tensor([7.0, 8.0])
    original_grad = x.grad

    find_runtime_multiplies(model, x)

    assert not hasattr(model, "_last_aux_loss")
    assert not hasattr(model, "new_public_cache")
    assert model._private_config is private_config
    assert model._private_config == {"events": ["before"]}
    assert model.history is history and model.history == [1]
    assert model._cache is cache and torch.equal(model._cache, torch.tensor([2.0]))
    assert torch.equal(x, torch.tensor([1.0, 2.0]))
    assert x.grad is original_grad
    assert torch.equal(x.grad, torch.tensor([7.0, 8.0]))


def test_training_audit_restores_replaced_registration_objects_and_optimizer_refs():
    from tikitorch.audit import find_training_multiplies

    class Replacing(tnn.Module):
        def __init__(self):
            super().__init__()
            self.weight = tnn.Parameter(torch.tensor([3.0]))
            self.register_buffer("scale", torch.tensor([2.0]))
            self.child = tnn.ReLU()

        def forward(self, value):
            self.weight = tnn.Parameter(torch.tensor([9.0]))
            self.scale = torch.tensor([8.0])
            self.child = tnn.Identity()
            self.temporary_parameter = tnn.Parameter(torch.tensor([7.0]))
            self.register_buffer("temporary_buffer", torch.tensor([6.0]))
            self.temporary_child = tnn.Identity()
            return value

    model = Replacing()
    weight = model.weight
    scale = model.scale
    child = model.child
    parameters_registry = model._parameters
    buffers_registry = model._buffers
    modules_registry = model._modules
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    optimizer_parameter = optimizer.param_groups[0]["params"][0]
    x = torch.tensor([1.0], requires_grad=True)

    find_training_multiplies(
        model, x, torch.tensor([0.0]), lambda prediction, _target: prediction.sum(),
        optimizer=optimizer,
    )

    assert model._parameters is parameters_registry
    assert model._buffers is buffers_registry
    assert model._modules is modules_registry
    assert model.weight is weight and torch.equal(weight, torch.tensor([3.0]))
    assert model.scale is scale and torch.equal(scale, torch.tensor([2.0]))
    assert model.child is child
    assert optimizer.param_groups[0]["params"][0] is optimizer_parameter is weight
    assert not hasattr(model, "temporary_parameter")
    assert not hasattr(model, "temporary_buffer")
    assert not hasattr(model, "temporary_child")


def test_runtime_audit_restores_registered_tensor_shape_and_value_after_resize():
    from tikitorch.audit import find_runtime_multiplies

    class Resizing(tnn.Module):
        def __init__(self):
            super().__init__()
            self.weight = tnn.Parameter(torch.tensor([3.0]))
            self.register_buffer("state", torch.tensor([2.0]))

        def forward(self, value):
            self.weight.set_(torch.tensor([8.0, 9.0]))
            self.state.resize_(2).copy_(torch.tensor([6.0, 7.0]))
            return value

    model = Resizing()
    weight = model.weight
    state = model.state

    find_runtime_multiplies(model, torch.tensor([1.0]))

    assert model.weight is weight
    assert model.state is state
    assert weight.shape == (1,) and torch.equal(weight, torch.tensor([3.0]))
    assert state.shape == (1,) and torch.equal(state, torch.tensor([2.0]))


def test_training_audit_restores_target_tensor_values():
    from tikitorch.audit import find_training_multiplies

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([3.0, 4.0])

    def mutating_loss(prediction, mutable_target):
        mutable_target.add_(10)
        return prediction.sum()

    find_training_multiplies(tnn.Identity(), x, target, mutating_loss)

    assert torch.equal(x, torch.tensor([1.0, 2.0]))
    assert x.grad is None
    assert torch.equal(target, torch.tensor([3.0, 4.0]))


def test_executed_audit_rejects_uninitialized_lazy_modules_without_mutating_them():
    from tikitorch.audit import find_runtime_multiplies

    model = tnn.LazyLinear(3)
    assert model.has_uninitialized_params()

    with pytest.raises(RuntimeError, match="cannot safely materialize lazy modules"):
        find_runtime_multiplies(model, torch.randn(2, 4))

    assert isinstance(model, tnn.LazyLinear)
    assert model.has_uninitialized_params()


def test_training_audit_restores_state_when_user_loss_raises():
    from tikitorch.audit import find_training_multiplies

    model = tnn.Linear(2, 2).eval()
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 3.0)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

    def broken_loss(_pred, _target):
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError, match="intentional"):
        find_training_multiplies(model, torch.randn(2, 2), None, broken_loss)

    assert model.training is False
    assert torch.equal(parameter.grad, torch.full_like(parameter, 3.0))
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


# --- closed-by-construction CONTRACT: the closed recipe is verified Level-B ---

def test_closure_report_one_call_inference_and_training():
    import tikitorch as tt
    import tikitorch.functional as F
    import tikitorch.optim as optim
    m = tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=1,
                                      task_type="lm", vocab_size=32, dropout=0.0)
    x = torch.randint(0, 32, (2, 16)); y = torch.randint(0, 32, (2, 16))
    perc = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, p.shape[-1]), yy.reshape(-1))
    rep = tt.closure_report(m, x, y, perc, optimizer=optim.PureTropicalSGD(m.parameters(), lr_shift=4))
    assert rep["inference_closed"] is True
    assert rep["training_zero_dsp"] is True          # closed recipe -> zero-DSP training


def test_inference_closed_by_construction_for_tropical_models():
    # Every standard tropical model is closed at INFERENCE by construction.
    import tikitorch as tt
    cases = [
        (tt.models.TropicalCNN(in_channels=1, num_classes=8), torch.rand(2, 1, 16, 16)),
        (tt.models.TropicalMLP(d_model=32, n_layers=2, input_dim=16, num_classes=8), torch.rand(2, 16)),
        (tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm", vocab_size=32), torch.randint(0, 32, (2, 16))),
        (tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=1, task_type="lm", vocab_size=32), torch.randint(0, 32, (2, 16))),
    ]
    for m, x in cases:
        assert tt.closure_report(m, x)["inference_closed"], f"{type(m).__name__} not inference-closed"


def test_tropical_dropout_keeps_training_zero_dsp():
    # The models now use TropicalDropout (a {0,1} gate, no 1/(1-p) rescale), so dropout>0
    # NO LONGER breaks training closure — it is zero-DSP at any dropout rate. (Previously the
    # raw nn.Dropout survivor rescale was a non-pow2 multiply that broke closure off-pow2-p.)
    import tikitorch as tt
    import tikitorch.functional as F
    import tikitorch.nn as ttnn
    import tikitorch.optim as optim
    import torch.nn as nn
    x = torch.randint(0, 32, (2, 16)); y = torch.randint(0, 32, (2, 16))
    perc = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, p.shape[-1]), yy.reshape(-1))
    m = tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=1,
                                      task_type="lm", vocab_size=32, dropout=0.1)
    rep = tt.closure_report(m, x, y, perc, optimizer=optim.PureTropicalSGD(m.parameters(), lr_shift=4))
    assert rep["training_zero_dsp"] is True          # TropicalDropout>0 stays zero-DSP
    assert rep["training_closed"] is True

    # contrast: the raw nn.Dropout it replaced rescales by 1/(1-p); at non-pow2 p that is a
    # real multiply the closure guard flags (1/(1-0.3)=1.43); TropicalDropout(0.3) does not.
    with tt.assert_no_multiply():
        ttnn.TropicalDropout(0.3).train()(torch.randn(256))
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            nn.Dropout(0.3).train()(torch.randn(256))
